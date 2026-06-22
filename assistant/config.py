"""Configuration for the assistant.

We keep configuration simple and deployment-friendly:
- env vars are the primary source of truth
- config dataclasses make the rest of code type-safe and testable
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .env import load_env_file


load_env_file()


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else float(default)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else int(default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    v = v.strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _env_float_opt(name: str, default: float | None) -> float | None:
    """Optional float env.

    Accepts empty/'default'/'none'/'null' to mean None.
    """

    v = os.environ.get(name)
    if v is None:
        return default
    v2 = v.strip().lower()
    if v2 in ("", "default", "none", "null"):
        return None
    return float(v)


@dataclass(slots=True)
class CloudConfig:
    """Cloud providers config."""

    provider: str = os.environ.get("CLOUD_PROVIDER", "openai").lower().strip()

    # OpenAI
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    openai_base_url: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_stt_model: str = os.environ.get("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
    openai_llm_model: str = os.environ.get("OPENAI_LLM_MODEL", "gpt-5-mini")
    # Some models only support the default temperature. Set env to "default" to omit.
    openai_llm_temperature: float | None = _env_float_opt("OPENAI_LLM_TEMPERATURE", None)
    openai_tts_model: str = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    openai_tts_voice: str = os.environ.get("OPENAI_TTS_VOICE", "alloy")
    openai_timeout_sec: float = _env_float("OPENAI_TIMEOUT_SEC", 25.0)

    # Reliability
    cloud_max_in_flight: int = _env_int("CLOUD_MAX_IN_FLIGHT", 2)
    cloud_max_attempts: int = _env_int("CLOUD_MAX_ATTEMPTS", 3)
    cloud_base_delay_sec: float = _env_float("CLOUD_BASE_DELAY_SEC", 0.4)
    cloud_max_delay_sec: float = _env_float("CLOUD_MAX_DELAY_SEC", 6.0)


@dataclass(slots=True)
class AudioConfig:
    input_device: int | None = int(os.environ["INPUT_DEVICE"]) if os.environ.get("INPUT_DEVICE") else None
    output_device: int | None = int(os.environ["OUTPUT_DEVICE"]) if os.environ.get("OUTPUT_DEVICE") else None

    # We keep internal processing at 16 kHz for wake/VAD.
    sample_rate: int = _env_int("SAMPLE_RATE", 16000)
    frame_ms: int = _env_int("FRAME_MS", 30)

    # Playback
    playback_sample_rate: int = _env_int("PLAYBACK_SR", 24000)
    # If True, mic input is ignored while TTS is playing (prevents TTS->wake feedback).
    suppress_mic_during_tts: bool = _env_bool("SUPPRESS_MIC_DURING_TTS", True)


@dataclass(slots=True)
class WakeConfig:
    wake_phrase: str = os.environ.get("WAKE_PHRASE", "олег")
    wake_match_threshold: float = _env_float("WAKE_MATCH_THRESHOLD", 0.60)
    wake_min_speech_ratio: float = _env_float("WAKE_MIN_SPEECH_RATIO", 0.10)
    wake_tail_drop_ms: int = _env_int("WAKE_TAIL_DROP_MS", 250)
    wake_cooldown_sec: float = _env_float("WAKE_COOLDOWN_SEC", 1.0)


@dataclass(slots=True)
class TurnConfig:
    """Command recording and dialog behavior."""

    command_start_timeout_sec: float = _env_float("COMMAND_START_TIMEOUT_SEC", 5.0)
    max_silence_sec: float = _env_float("MAX_SILENCE_SEC", 0.8)
    max_phrase_sec: float = _env_float("MAX_PHRASE_SEC", 12.0)
    min_phrase_sec: float = _env_float("MIN_PHRASE_SEC", 0.35)

    # LLM
    system_prompt: str = os.environ.get(
        "SYSTEM_PROMPT",
        "Ты голосовой ассистент. Отвечай кратко, по делу, на русском языке.",
    )
    max_context_turns: int = _env_int("MAX_CONTEXT_TURNS", 6)

    # Stop-word / barge-in (best-effort without AEC)
    enable_stop_word: bool = _env_bool("ENABLE_STOP_WORD", True)
    stop_words: str = os.environ.get("STOP_WORDS", "стоп,хватит,отмена")
    stop_word_cooldown_sec: float = _env_float("STOP_WORD_COOLDOWN_SEC", 1.0)


@dataclass(slots=True)
class WeatherConfig:
    """Weather/geo skill configuration (env-driven).

    This skill is intended to be deterministic (no LLM) for:
    - speed
    - cost
    - privacy
    """

    enabled: bool = _env_bool("WEATHER_ENABLED", True)

    # Geo strategy:
    # - auto: fixed -> ip
    # - fixed: always use GEO_FIXED_LAT/LON
    # - ip: use IP geolocation
    geo_mode: str = os.environ.get("GEO_MODE", "auto").lower().strip()
    fixed_lat: float | None = _env_float_opt("GEO_FIXED_LAT", None)
    fixed_lon: float | None = _env_float_opt("GEO_FIXED_LON", None)

    # IP geolocation endpoint (no API key by default).
    # ipapi.co returns {"latitude":..,"longitude":..,"city":..}.
    ip_geo_url: str = os.environ.get("IP_GEO_URL", "https://ipapi.co/json/").strip()

    # Weather provider (currently: open-meteo).
    provider: str = os.environ.get("WEATHER_PROVIDER", "open-meteo").lower().strip()

    # Timeouts
    geo_timeout_sec: float = _env_float("GEO_TIMEOUT_SEC", 4.0)
    weather_timeout_sec: float = _env_float("WEATHER_TIMEOUT_SEC", 6.0)

    # Caches
    geo_cache_ttl_sec: float = _env_float("GEO_CACHE_TTL_SEC", 12 * 60 * 60)  # 12h
    weather_cache_ttl_sec: float = _env_float("WEATHER_CACHE_TTL_SEC", 5 * 60)  # 5m

    # NLG
    units: str = os.environ.get("WEATHER_UNITS", "metric").lower().strip()


@dataclass(slots=True)
class SpotifyConfig:
    """Spotify integration config (env-driven)."""

    enabled: bool = _env_bool("SPOTIFY_ENABLED", False)
    client_id: str = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
    redirect_uri: str = os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:17845/callback").strip()
    token_path: str = os.environ.get(
        "SPOTIFY_TOKEN_PATH",
        os.path.join(os.path.dirname(__file__), "..", ".secrets", "spotify_tokens.json"),
    )
    # Optional: remembered device name to reduce "нет активного устройства" friction.
    preferred_device_name: str = os.environ.get("SPOTIFY_PREFERRED_DEVICE", "").strip()


@dataclass(slots=True)
class LightsConfig:
    """Smart lights integration config (env-driven)."""

    enabled: bool = _env_bool("LIGHTS_ENABLED", True)

    # Adapter selection:
    # - mock: built-in demo adapter
    # - wiz_lan: WiZ LAN UDP (fits many Gauss Wi‑Fi bulbs controlled by WiZ app)
    adapter: str = os.environ.get("GAUSS_ADAPTER", "wiz_lan").lower().strip()

    # Persistent state storage
    state_path: str = os.environ.get(
        "LIGHTS_STATE_PATH",
        os.path.join(os.path.dirname(__file__), "..", ".state", "lights.json"),
    )

    request_timeout_sec: float = _env_float("LIGHTS_REQUEST_TIMEOUT_SEC", 4.0)
    state_cache_ttl_sec: float = _env_float("LIGHTS_STATE_CACHE_TTL_SEC", 6.0)

    # Target defaults (to make short commands usable):
    # - default_scope=all: apply to all lamps when user didn't specify a target
    # - default_scope=room: apply to default_room (if set)
    default_scope: str = os.environ.get("LIGHTS_DEFAULT_SCOPE", "room").lower().strip()
    default_room: str = os.environ.get("LIGHTS_DEFAULT_ROOM", "").strip()


@dataclass(slots=True)
class AppConfig:
    # Use default_factory to avoid mutable defaults (required on Python 3.11+).
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    turn: TurnConfig = field(default_factory=TurnConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    spotify: SpotifyConfig = field(default_factory=SpotifyConfig)
    lights: LightsConfig = field(default_factory=LightsConfig)
