"""WiZ LAN adapter (UDP) for Gauss Wi‑Fi bulbs controlled via WiZ app.

Many Gauss Wi‑Fi bulbs are rebrands of the WiZ ecosystem. They can be controlled
locally over UDP/38899 with JSON messages.

This implementation is intentionally conservative:
- best-effort discovery via broadcast
- on/off, brightness, CT, RGB via setPilot/getPilot
- no account/cloud needed

Notes:
- WiZ LAN protocol is not officially stable across all firmwares.
- Some devices may require "local control" enabled in the WiZ app.
"""

from __future__ import annotations

import asyncio
import json
import os
import ipaddress
import socket
import time
from dataclasses import dataclass

from .lights_gauss_adapter import GaussAdapterError, GaussDeviceOffline
from .lights_types import Lamp, LampCapabilities, LampState


WIZ_PORT = 38899


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _udp_broadcast_discover_one(*, broadcast_addr: str, timeout_sec: float = 1.2) -> list[dict]:
    """Blocking UDP broadcast discovery.

    Returns raw response dicts.
    """

    payload = {"method": "getSystemConfig", "params": {}}
    msg = json.dumps(payload).encode("utf-8")

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.settimeout(float(timeout_sec))
        s.bind(("", 0))
        # NOTE: some networks/OS setups can reject the global broadcast route.
        # We let the caller handle OSError and provide fallback (manual IPs).
        s.sendto(msg, (str(broadcast_addr), WIZ_PORT))

        out: list[dict] = []
        t_end = time.time() + float(timeout_sec)
        while time.time() < t_end:
            try:
                data, addr = s.recvfrom(65535)
            except socket.timeout:
                break
            try:
                obj = json.loads(data.decode("utf-8", errors="ignore"))
                if isinstance(obj, dict):
                    obj["_ip"] = addr[0]
                    out.append(obj)
            except Exception:
                continue
        return out
    finally:
        try:
            s.close()
        except Exception:
            pass


def _local_private_ips_best_effort() -> list[str]:
    """Best-effort list of local IPs.

    We avoid external deps. This is not perfect but good enough for broadcast
    candidates derivation.
    """

    cands: list[str] = []

    # 1) Explicit override (recommended if you have VPNs).
    for env_name in ("WIZ_LOCAL_IP", "GAUSS_LOCAL_IP"):
        v = (os.environ.get(env_name) or "").strip()
        if v:
            cands.append(v)

    # 2) Hostname addresses.
    try:
        _hn, _aliases, addrs = socket.gethostbyname_ex(socket.gethostname())
        for a in addrs or []:
            cands.append(a)
    except Exception:
        pass

    # 3) Routing-based guess (may pick VPN; still useful).
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            cands.append(s.getsockname()[0])
        finally:
            s.close()
    except Exception:
        pass

    # Keep only private v4 addresses.
    out: list[str] = []
    for x in cands:
        try:
            ip = ipaddress.ip_address(x)
            if ip.version == 4 and ip.is_private:
                xs = str(ip)
                if xs not in out:
                    out.append(xs)
        except Exception:
            continue
    return out


def _broadcast_candidates() -> list[str]:
    # 0) User override.
    env = (os.environ.get("WIZ_BROADCAST_ADDR") or os.environ.get("GAUSS_BROADCAST_ADDR") or "").strip()
    if env:
        out = [x.strip() for x in env.replace(";", ",").split(",") if x.strip()]
        if out:
            return out

    # 1) Always try global broadcast.
    out: list[str] = ["255.255.255.255"]

    # 2) Derive /24 broadcast from private IPs (most home LANs).
    for ip in _local_private_ips_best_effort():
        try:
            a, b, c, _d = ip.split(".")
            bc = f"{a}.{b}.{c}.255"
            if bc not in out:
                out.append(bc)
        except Exception:
            continue

    # 3) Common home LAN ranges fallback.
    for bc in ("192.168.0.255", "192.168.1.255", "10.0.0.255", "10.0.1.255"):
        if bc not in out:
            out.append(bc)
    return out


def _udp_request(ip: str, payload: dict, *, timeout_sec: float = 1.0, retries: int = 2) -> dict:
    """Blocking UDP request/response to a single bulb."""

    msg = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None
    for _ in range(max(1, int(retries))):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.settimeout(float(timeout_sec))
            s.sendto(msg, (ip, WIZ_PORT))
            data, _addr = s.recvfrom(65535)
            obj = json.loads(data.decode("utf-8", errors="ignore"))
            if not isinstance(obj, dict):
                raise GaussAdapterError("Bad response")
            return obj
        except Exception as e:
            last_err = e
        finally:
            try:
                s.close()
            except Exception:
                pass
    raise GaussAdapterError(f"UDP request failed: {last_err}")


@dataclass(slots=True)
class WizLanAdapter:
    """WiZ LAN adapter implementing our GaussAdapter protocol."""

    timeout_sec: float = 1.0
    retries: int = 2

    # Optional: manual IP list fallback if broadcast discovery is blocked.
    manual_ips: list[str] | None = None

    # dynamic mapping discovered at runtime
    _id_to_ip: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._id_to_ip = {}
        if self.manual_ips is None:
            env = os.environ.get("WIZ_IPS") or os.environ.get("GAUSS_WIZ_IPS") or ""
            ips = [x.strip() for x in env.replace(";", ",").replace(" ", ",").split(",") if x.strip()]
            self.manual_ips = ips or None

    async def discover(self) -> list[Lamp]:
        raws: list[dict] = []
        # 1) Try broadcast discovery.
        try:
            for bc in _broadcast_candidates():
                try:
                    raws = await asyncio.to_thread(
                        _udp_broadcast_discover_one,
                        broadcast_addr=bc,
                        timeout_sec=max(0.5, self.timeout_sec),
                    )
                except OSError:
                    raws = []
                    continue
                if raws:
                    break
        except OSError as e:
            # Typical on misconfigured host/network: "No route to host".
            if not self.manual_ips:
                raise GaussAdapterError(
                    "Не могу отправить broadcast для поиска WiZ ламп (UDP 255.255.255.255:38899). "
                    "Проверь, что ассистент в той же Wi‑Fi сети и на роутере выключен AP/client isolation. "
                    "Если включён VPN, временно отключи его для устройства ассистента. "
                    "Либо задай IP ламп вручную: WIZ_IPS=192.168.1.10,192.168.1.11"
                ) from e
            raws = []

        lamps: list[Lamp] = []
        for r in raws:
            res = r.get("result") or {}
            ip = r.get("_ip")
            if not ip:
                continue
            mac = str(res.get("mac") or "").strip() or None
            module = str(res.get("moduleName") or res.get("fwVersion") or "").strip() or None
            lamp_id = mac or f"wiz:{ip}"
            self._id_to_ip[lamp_id] = ip

            # Capabilities are best-effort; real check is by trial (or extended API).
            caps = LampCapabilities(supports_brightness=True, supports_ct=True, supports_rgb=True, supports_scenes=False)
            lamps.append(
                Lamp(
                    id=lamp_id,
                    name=(module or f"WiZ {ip}"),
                    room=None,
                    groups=[],
                    is_online=True,
                    capabilities=caps,
                    state=LampState(is_on=False, brightness_percent=50, ct_kelvin=3500, color_rgb=(255, 255, 255)),
                )
            )
        if lamps:
            return lamps

        # If broadcast returned no devices, try manual IPs (if provided).
        if self.manual_ips:
            lamps2 = await self._discover_manual_ips()
            if lamps2:
                return lamps2

        # Still nothing. Provide a helpful hint.
        raise GaussAdapterError(
            "Ламп WiZ в сети не нашёл. Проверь, что в приложении WiZ включено локальное управление, "
            "что ассистент и лампы в одной Wi‑Fi сети, и что роутер не режет broadcast. "
            "Можно задать IP ламп вручную: WIZ_IPS=192.168.1.10,192.168.1.11"
        )

        # 2) Fallback: manual IP list.

    async def _discover_manual_ips(self) -> list[Lamp]:
        ips = list(self.manual_ips or [])
        lamps: list[Lamp] = []
        for ip in ips:
            try:
                obj = await asyncio.to_thread(
                    _udp_request,
                    ip,
                    {"method": "getSystemConfig", "params": {}},
                    timeout_sec=self.timeout_sec,
                    retries=self.retries,
                )
            except Exception:
                continue
            res = obj.get("result") or {}
            mac = str(res.get("mac") or "").strip() or None
            module = str(res.get("moduleName") or res.get("fwVersion") or "").strip() or None
            lamp_id = mac or f"wiz:{ip}"
            self._id_to_ip[lamp_id] = ip
            caps = LampCapabilities(supports_brightness=True, supports_ct=True, supports_rgb=True, supports_scenes=False)
            lamps.append(
                Lamp(
                    id=lamp_id,
                    name=(module or f"WiZ {ip}"),
                    room=None,
                    groups=[],
                    is_online=True,
                    capabilities=caps,
                    state=LampState(is_on=False, brightness_percent=50, ct_kelvin=3500, color_rgb=(255, 255, 255)),
                )
            )
        return lamps


    def _ip_for(self, lamp_id: str) -> str:
        ip = self._id_to_ip.get(lamp_id)
        if ip:
            return ip
        # Best-effort: allow id to be wiz:<ip>
        if lamp_id.startswith("wiz:"):
            return lamp_id.split(":", 1)[1]
        raise GaussAdapterError("unknown_lamp_id")

    async def _ensure_ip(self, lamp_id: str) -> str:
        """Resolve lamp_id -> ip.

        Problem this solves:
        - We persist devices by stable id (often MAC), but adapter instance is
          created fresh per command. Without discovery, id->ip mapping is empty.

        Strategy:
        1) use in-memory mapping
        2) if lamp_id is wiz:<ip>, use it
        3) if manual IPs were provided, probe them and match by MAC
        4) as a last resort, run discovery and retry
        """

        try:
            return self._ip_for(lamp_id)
        except GaussAdapterError as e:
            if str(e) != "unknown_lamp_id":
                raise

        # 3) Probe manual IPs and match by mac.
        if self.manual_ips:
            for ip in list(self.manual_ips or []):
                try:
                    obj = await asyncio.to_thread(
                        _udp_request,
                        ip,
                        {"method": "getSystemConfig", "params": {}},
                        timeout_sec=self.timeout_sec,
                        retries=max(1, self.retries),
                    )
                except Exception:
                    continue
                res = obj.get("result") or {}
                mac = str(res.get("mac") or "").strip().lower()
                if mac and mac == lamp_id.strip().lower():
                    self._id_to_ip[lamp_id] = ip
                    return ip

        # 4) Last resort: run discovery (tries several broadcasts) and retry.
        try:
            _ = await self.discover()
            return self._ip_for(lamp_id)
        except Exception:
            raise GaussAdapterError(
                "Лампа известна в состоянии, но её IP не определён. "
                "Скажи: 'найди лампы'. Если discovery не работает — укажи IP вручную через WIZ_IPS=..."
            )

    async def get_state(self, lamp_id: str) -> LampState:
        ip = await self._ensure_ip(lamp_id)
        try:
            obj = await asyncio.to_thread(
                _udp_request,
                ip,
                {"method": "getPilot", "params": {}},
                timeout_sec=self.timeout_sec,
                retries=self.retries,
            )
        except Exception as e:
            raise GaussDeviceOffline(str(e))

        res = obj.get("result") or {}
        on = bool(res.get("state", True))
        dim = res.get("dimming")
        temp = res.get("temp")
        r = res.get("r")
        g = res.get("g")
        b = res.get("b")

        st = LampState(
            is_on=on,
            brightness_percent=(int(dim) if isinstance(dim, (int, float)) else None),
            ct_kelvin=(int(temp) if isinstance(temp, (int, float)) else None),
            color_rgb=(int(r), int(g), int(b)) if all(isinstance(x, (int, float)) for x in (r, g, b)) else None,
            color_mode="white",
            night_mode=False,
            scene=None,
            updated_at=time.time(),
        )
        if st.color_rgb is not None:
            st.color_mode = "rgb"
        elif st.ct_kelvin is not None:
            st.color_mode = "ct"
        return st

    async def set_power(self, lamp_id: str, *, on: bool) -> None:
        ip = await self._ensure_ip(lamp_id)
        try:
            await asyncio.to_thread(
                _udp_request,
                ip,
                {"method": "setPilot", "params": {"state": bool(on)}},
                timeout_sec=self.timeout_sec,
                retries=self.retries,
            )
        except Exception as e:
            raise GaussDeviceOffline(str(e))

    async def set_brightness(self, lamp_id: str, *, percent: int) -> None:
        ip = await self._ensure_ip(lamp_id)
        p = _clamp(int(percent), 0, 100)
        params: dict = {"state": p > 0}
        # WiZ dimming expects 10..100 for many devices; keep 1..100 to be permissive.
        params["dimming"] = max(1, p)
        try:
            await asyncio.to_thread(
                _udp_request,
                ip,
                {"method": "setPilot", "params": params},
                timeout_sec=self.timeout_sec,
                retries=self.retries,
            )
        except Exception as e:
            raise GaussDeviceOffline(str(e))

    async def set_ct(self, lamp_id: str, *, kelvin: int) -> None:
        ip = await self._ensure_ip(lamp_id)
        k = _clamp(int(kelvin), 2200, 6500)
        try:
            await asyncio.to_thread(
                _udp_request,
                ip,
                {"method": "setPilot", "params": {"state": True, "temp": k}},
                timeout_sec=self.timeout_sec,
                retries=self.retries,
            )
        except Exception as e:
            raise GaussDeviceOffline(str(e))

    async def set_rgb(self, lamp_id: str, *, rgb: tuple[int, int, int]) -> None:
        ip = await self._ensure_ip(lamp_id)
        r, g, b = rgb
        params = {"state": True, "r": _clamp(r, 0, 255), "g": _clamp(g, 0, 255), "b": _clamp(b, 0, 255)}
        try:
            await asyncio.to_thread(
                _udp_request,
                ip,
                {"method": "setPilot", "params": params},
                timeout_sec=self.timeout_sec,
                retries=self.retries,
            )
        except Exception as e:
            raise GaussDeviceOffline(str(e))

    async def set_night_mode(self, lamp_id: str, *, on: bool) -> None:
        # No universal night mode in WiZ LAN; emulate by dimming.
        if not on:
            return
        await self.set_brightness(lamp_id, percent=10)

    async def apply_scene(self, lamp_id: str, *, scene_name: str) -> None:
        # Assistant-level scenes (portable): translate to brightness/ct.
        s = (scene_name or "").strip().lower()
        if s in ("кино", "movie"):
            await self.set_ct(lamp_id, kelvin=2700)
            await self.set_brightness(lamp_id, percent=20)
            return
        if s in ("чтение", "reading"):
            await self.set_ct(lamp_id, kelvin=4000)
            await self.set_brightness(lamp_id, percent=75)
            return
        if s in ("сон", "sleep"):
            await self.set_ct(lamp_id, kelvin=2700)
            await self.set_brightness(lamp_id, percent=10)
            return
        raise GaussAdapterError("scene not supported")

    async def capabilities(self, lamp_id: str) -> LampCapabilities:
        # Best-effort.
        _ = lamp_id
        return LampCapabilities(supports_brightness=True, supports_ct=True, supports_rgb=True, supports_scenes=False)
