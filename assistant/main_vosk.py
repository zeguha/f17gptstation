"""Improved Vosk-based assistant entrypoint.

Implements a robust pipeline:
wait wake → confirm wake → record command (VAD) → ASR command → postprocess.

Key guarantee: wake phrase never leaks into the recognized command.
"""

from __future__ import annotations

import logging
import os

from .asr_engines import VoskEngine, WhisperCliEngine
from .audio_stream import AudioStream
from .chat import ask_gpt
from .config import AppConfig
from .intent_lights import detect_lights_intent
from .intent_weather import detect_weather_intent
from .intent_spotify import detect_spotify_intent
from .utils import normalize_text
from .weather_skill import WeatherSkillError, handle_weather_intent
from .lights_adapter_factory import build_gauss_adapter
from .lights_device_manager import LightsManager, LightsManagerConfig
from .lights_skill import LightsSkillDeps, LightsSkillError, handle_lights_intent
from .spotify_client import SpotifyClient, SpotifyClientConfig
from .spotify_skill import SpotifySkillError, handle_spotify_intent
from .pipeline_vosk import PipelineConfig, WakeCommandPipeline
from .speech import SPEAKING_EVENT, get_default_input_device, speak, speak_async, stop_speech


STOP_WORD = "стоп"


def _configure_logging() -> None:
    level = os.environ.get("ASSISTANT_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    _configure_logging()
    log = logging.getLogger("assistant.main_vosk")

    app_cfg = AppConfig()

    cfg = PipelineConfig(
        wake_phrase=os.environ.get("WAKE_PHRASE", "олег"),
        sample_rate=int(os.environ.get("SAMPLE_RATE", "16000")),
        vad_mode=int(os.environ.get("VAD_MODE", "2")),
        # Single-word wake phrases often need a lower threshold due to ASR variability.
        wake_match_threshold=float(os.environ.get("WAKE_MATCH_THRESHOLD", "0.60")),
        wake_min_speech_ratio=float(os.environ.get("WAKE_MIN_SPEECH_RATIO", "0.10")),
    )

    # Vosk model (local)
    model_path = os.environ.get("VOSK_MODEL_PATH") or os.path.join(
        os.path.dirname(__file__), "models", "vosk-model-ru"
    )

    # Optional whisper.cpp backend (higher quality, slower). Enable via ASR_BACKEND=whispercpp
    asr_backend = os.environ.get("ASR_BACKEND", "vosk").lower().strip()

    try:
        # Allow manual override for problematic macOS setups.
        env_dev = os.environ.get("INPUT_DEVICE")
        device_id = int(env_dev) if env_dev is not None else get_default_input_device()
    except Exception as e:
        print("❌ Не найден микрофон:", e)
        return

    # We use fixed 30ms frames at 16k for webrtcvad.
    frame_ms = cfg.frame_ms
    frame_samples = int(cfg.sample_rate * frame_ms / 1000)

    # NOTE: Some Vosk models (including some RU packs) do not support runtime grammars:
    # they print: "Runtime graphs are not supported by this model".
    # Wake detection is therefore implemented as fuzzy prefix matching in the pipeline.
    wake_engine = VoskEngine(model_path=model_path, sample_rate=cfg.sample_rate)

    if asr_backend == "whispercpp":
        whisper_bin = os.environ.get("WHISPER_BIN") or os.path.expanduser(
            "~/gpt-station/whisper.cpp/build/bin/whisper-cli"
        )
        whisper_model = os.environ.get("WHISPER_MODEL") or os.path.expanduser(
            "~/gpt-station/whisper.cpp/models/ggml-small.bin"
        )
        command_engine = WhisperCliEngine(
            whisper_bin=whisper_bin,
            model_path=whisper_model,
            language="ru",
            sample_rate=cfg.sample_rate,
            timeout_sec=60,
        )
        log.info("ASR backend: whisper.cpp (%s)", whisper_model)
    else:
        command_engine = VoskEngine(model_path=model_path, sample_rate=cfg.sample_rate)
        log.info("ASR backend: Vosk (%s)", model_path)

    print(f"🎤 Ассистент готов. Скажи '{cfg.wake_phrase}' для активации...")
    try:
        import sounddevice as sd

        dev = sd.query_devices(device_id, kind="input")
        print("Использован микрофон:", device_id, "|", dev.get("name"), "| default_sr", dev.get("default_samplerate"))
    except Exception:
        print("Использован микрофон (device id):", device_id)

    with AudioStream(
        sample_rate=cfg.sample_rate,
        frame_samples=frame_samples,
        device=device_id,
        ignore_event=SPEAKING_EVENT,
        queue_max_chunks=400,
    ) as audio:
        pipeline = WakeCommandPipeline(cfg=cfg, audio=audio, wake_engine=wake_engine, command_engine=command_engine)

        try:
            while True:
                interaction = pipeline.run_once()
                if interaction is None:
                    # false wake or too-short recording
                    continue

                # Metrics / logging
                log.info(
                    "wake: t=%.2fs conf=%s speech_ratio=%.2f | cmd: dur=%.2fs reason=%s | asr_conf=%s | text='%s'",
                    interaction.wake.time_to_wake_sec,
                    None if interaction.wake.confidence is None else round(interaction.wake.confidence, 3),
                    interaction.wake.speech_ratio,
                    interaction.command.duration_sec,
                    interaction.command.stop_reason,
                    None if interaction.asr.avg_confidence is None else round(interaction.asr.avg_confidence, 3),
                    interaction.final_text,
                )

                text = interaction.final_text
                if not text:
                    speak("Не расслышал команду")
                    continue

                if STOP_WORD in text:
                    stop_speech()
                    speak("Отмена")
                    continue

                speak_async("Секунду")

                # Routing probe (helps diagnose intent collisions).
                # Enable by setting ASSISTANT_ROUTE_DEBUG=1.
                nrm = normalize_text(text)
                route_dbg = os.environ.get("ASSISTANT_ROUTE_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
                li_probe = detect_lights_intent(nrm) if app_cfg.lights.enabled else None
                si_probe = detect_spotify_intent(nrm) if app_cfg.spotify.enabled else None
                wi_probe = detect_weather_intent(nrm) if app_cfg.weather.enabled else None
                if route_dbg:
                    log.info(
                        "ROUTE probe: lights=%s spotify=%s weather=%s | text='%s'",
                        None if li_probe is None else li_probe.action,
                        None if si_probe is None else si_probe.action,
                        None if wi_probe is None else wi_probe.intent,
                        nrm,
                    )

                # Deterministic skill routing (lights)
                if app_cfg.lights.enabled:
                    li = li_probe
                    if li is not None:
                        log.info("LIGHTS intent=%s", li.action)
                        try:
                            import asyncio

                            adapter = build_gauss_adapter(app_cfg.lights.adapter)
                            mgr = LightsManager(
                                adapter=adapter,
                                cfg=LightsManagerConfig(
                                    state_path=app_cfg.lights.state_path,
                                    state_cache_ttl_sec=app_cfg.lights.state_cache_ttl_sec,
                                ),
                            )
                            answer = asyncio.run(
                                handle_lights_intent(
                                    li,
                                    deps=LightsSkillDeps(
                                        manager=mgr,
                                        request_timeout_sec=app_cfg.lights.request_timeout_sec,
                                        default_scope=app_cfg.lights.default_scope,
                                        default_room=(app_cfg.lights.default_room or None),
                                    ),
                                )
                            )
                        except LightsSkillError as e:
                            answer = str(e)
                        except Exception:
                            log.exception("LIGHTS skill failed")
                            answer = "Не смог управлять светом. Попробуй чуть позже."
                        speak(answer)
                        continue

                # Deterministic skill routing (spotify)
                if app_cfg.spotify.enabled:
                    si = si_probe
                    if si is not None:
                        log.info("SPOTIFY intent=%s", si.action)
                        try:
                            import asyncio

                            sp = SpotifyClient(
                                SpotifyClientConfig(
                                    client_id=app_cfg.spotify.client_id,
                                    token_store_path=app_cfg.spotify.token_path,
                                )
                            )
                            answer, _new_dev = asyncio.run(
                                handle_spotify_intent(
                                    si,
                                    client=sp,
                                    normalized_text=normalize_text(text),
                                    preferred_device_name=(app_cfg.spotify.preferred_device_name or None),
                                )
                            )
                        except SpotifySkillError as e:
                            answer = str(e)
                        except Exception:
                            log.exception("SPOTIFY skill failed")
                            answer = "Не смог выполнить команду Spotify. Попробуй чуть позже."
                        speak(answer)
                        continue

                # Deterministic skill routing (weather)
                if app_cfg.weather.enabled:
                    wi = wi_probe
                    if wi is not None:
                        log.info("WEATHER intent=%s place=%s", wi.intent, wi.place_text)
                        try:
                            import asyncio

                            answer = asyncio.run(handle_weather_intent(wi, cfg=app_cfg.weather))
                        except WeatherSkillError as e:
                            answer = str(e)
                        except Exception:
                            log.exception("WEATHER skill failed")
                            answer = "Не смог получить погоду. Попробуй чуть позже."
                        speak(answer)
                        continue

                try:
                    answer = ask_gpt(text)
                except Exception as e:
                    log.exception("LLM error: %s", e)
                    speak("Произошла ошибка")
                    continue

                print("🤖 Ассистент:", answer)
                speak(answer)

        except KeyboardInterrupt:
            print("\nВыход по Ctrl+C")
        finally:
            log.info("false_wakes=%d", pipeline.false_wakes)


if __name__ == "__main__":
    main()
