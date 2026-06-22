"""Cloud-first assistant entrypoint for Raspberry Pi.

Local:
- audio capture
- VAD + wake state machine

Cloud:
- STT, LLM, TTS

This keeps quality high and CPU usage low for Raspberry Pi 3B.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import wave
import io

from .audio_playback import play_wav_bytes
from .audio_stream import AudioStream
from .cloud_openai import OpenAILlm, OpenAIStt, OpenAITts
from .config import AppConfig
from .dialog import DialogState
from .intent_lights import detect_lights_intent
from .intent_weather import detect_weather_intent
from .intent_spotify import detect_spotify_intent
from .pipeline_vosk import PipelineConfig, WakeCommandPipeline
from .speech import SPEAKING_EVENT, get_default_input_device
from .asr_engines import VoskEngine
from .utils import normalize_text
from .weather_skill import WeatherSkillError, handle_weather_intent
from .lights_adapter_factory import build_gauss_adapter
from .lights_device_manager import LightsManager, LightsManagerConfig
from .lights_skill import LightsSkillDeps, LightsSkillError, handle_lights_intent
from .spotify_client import SpotifyClient, SpotifyClientConfig
from .spotify_skill import SpotifySkillError, handle_spotify_intent


log = logging.getLogger("assistant.main_cloud")


def pcm16_to_wav_bytes(pcm16: bytes, *, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm16)
    return buf.getvalue()


async def run() -> None:
    cfg = AppConfig()

    level = os.environ.get("ASSISTANT_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Audio devices
    device_in = cfg.audio.input_device
    if device_in is None:
        device_in = get_default_input_device()

    frame_samples = int(cfg.audio.sample_rate * cfg.audio.frame_ms / 1000)

    # Wake pipeline uses Vosk only to get a rough text for fuzzy matching.
    model_path = os.environ.get("VOSK_MODEL_PATH") or os.path.join(
        os.path.dirname(__file__), "models", "vosk-model-ru"
    )
    wake_engine = VoskEngine(model_path=model_path, sample_rate=cfg.audio.sample_rate)

    # Cloud mode: we record command audio locally and do STT in the cloud.
    # Use the same Vosk engine only for wake; command_engine is ignored when skip_command_asr=True.
    command_engine_stub = wake_engine

    pipe_cfg = PipelineConfig(
        sample_rate=cfg.audio.sample_rate,
        frame_ms=cfg.audio.frame_ms,
        wake_phrase=cfg.wake.wake_phrase,
        wake_match_threshold=cfg.wake.wake_match_threshold,
        wake_min_speech_ratio=cfg.wake.wake_min_speech_ratio,
        wake_tail_drop_ms=cfg.wake.wake_tail_drop_ms,
        wake_cooldown_sec=cfg.wake.wake_cooldown_sec,
        command_start_timeout_sec=cfg.turn.command_start_timeout_sec,
        max_silence_sec=cfg.turn.max_silence_sec,
        max_phrase_sec=cfg.turn.max_phrase_sec,
        min_phrase_sec=cfg.turn.min_phrase_sec,
        skip_command_asr=True,
    )

    # Cloud clients (default: OpenAI)
    stt = OpenAIStt(cfg.cloud)
    llm = OpenAILlm(cfg.cloud)
    tts = OpenAITts(cfg.cloud)

    dialog = DialogState(system_prompt=cfg.turn.system_prompt, max_turns=cfg.turn.max_context_turns)

    ignore_event = SPEAKING_EVENT if cfg.audio.suppress_mic_during_tts else None

    print(f"🎤 Cloud assistant ready. Wake phrase: '{cfg.wake.wake_phrase}'")
    print(f"Input device id: {device_in}")

    with AudioStream(
        sample_rate=cfg.audio.sample_rate,
        frame_samples=frame_samples,
        device=device_in,
        ignore_event=ignore_event,
        queue_max_chunks=500,
    ) as audio:
        pipeline = WakeCommandPipeline(
            cfg=pipe_cfg,
            audio=audio,
            wake_engine=wake_engine,
            command_engine=command_engine_stub,
        )

        while True:
            # Run blocking wake/record in a thread to keep asyncio responsive.
            interaction = await asyncio.to_thread(pipeline.run_once)
            if interaction is None:
                continue

            user_text = (interaction.final_text or "").strip()

            # 1) If wake+command were merged, we already have text.
            # 2) Otherwise: do cloud STT on recorded PCM.
            if not user_text:
                if interaction.command_pcm16:
                    wav_bytes = pcm16_to_wav_bytes(
                        interaction.command_pcm16,
                        sample_rate=interaction.command_sample_rate,
                    )
                    t0 = time.monotonic()
                    stt_res = await stt.transcribe_wav(wav_bytes)
                    dt = time.monotonic() - t0
                    user_text = (stt_res.text or "").strip()
                    log.info("STT latency=%.2fs | text='%s'", dt, user_text)

            if not user_text:
                log.info("Empty text after STT; ask user to repeat")
                SPEAKING_EVENT.set()
                try:
                    wav = await tts.synthesize("Не расслышал. Повтори, пожалуйста.")
                    play_wav_bytes(wav.wav_bytes, device=cfg.audio.output_device)
                finally:
                    SPEAKING_EVENT.clear()
                continue

            # Deterministic skill routing (weather)
            # Deterministic skill routing (spotify)
            # Deterministic skill routing (lights)
            if cfg.lights.enabled:
                li = detect_lights_intent(normalize_text(user_text))
                if li is not None:
                    try:
                        adapter = build_gauss_adapter(cfg.lights.adapter)
                        mgr = LightsManager(
                            adapter=adapter,
                            cfg=LightsManagerConfig(
                                state_path=cfg.lights.state_path,
                                state_cache_ttl_sec=cfg.lights.state_cache_ttl_sec,
                            ),
                        )
                        answer = await handle_lights_intent(
                            li,
                            deps=LightsSkillDeps(
                                manager=mgr,
                                request_timeout_sec=cfg.lights.request_timeout_sec,
                                default_scope=cfg.lights.default_scope,
                                default_room=(cfg.lights.default_room or None),
                            ),
                        )
                    except LightsSkillError as e:
                        answer = str(e)
                    except Exception:
                        log.exception("LIGHTS skill failed")
                        answer = "Не смог управлять светом. Попробуй чуть позже."

                    SPEAKING_EVENT.set()
                    try:
                        wav = await tts.synthesize(answer)
                        play_wav_bytes(wav.wav_bytes, device=cfg.audio.output_device)
                    finally:
                        SPEAKING_EVENT.clear()
                    continue

            if cfg.spotify.enabled:
                si = detect_spotify_intent(normalize_text(user_text))
                if si is not None:
                    try:
                        sp = SpotifyClient(
                            SpotifyClientConfig(
                                client_id=cfg.spotify.client_id,
                                token_store_path=cfg.spotify.token_path,
                            )
                        )
                        answer, _new_dev = await handle_spotify_intent(
                            si,
                            client=sp,
                            normalized_text=normalize_text(user_text),
                            preferred_device_name=(cfg.spotify.preferred_device_name or None),
                        )
                    except SpotifySkillError as e:
                        answer = str(e)
                    except Exception:
                        log.exception("SPOTIFY skill failed")
                        answer = "Не смог выполнить команду Spotify. Попробуй чуть позже."

                    SPEAKING_EVENT.set()
                    try:
                        wav = await tts.synthesize(answer)
                        play_wav_bytes(wav.wav_bytes, device=cfg.audio.output_device)
                    finally:
                        SPEAKING_EVENT.clear()
                    continue

            if cfg.weather.enabled:
                wi = detect_weather_intent(normalize_text(user_text))
                if wi is not None:
                    log.info("WEATHER intent=%s place=%s", wi.intent, wi.place_text)
                    try:
                        answer = await handle_weather_intent(wi, cfg=cfg.weather)
                    except WeatherSkillError as e:
                        answer = str(e)
                    except Exception:
                        log.exception("WEATHER skill failed")
                        answer = "Не смог получить погоду. Попробуй чуть позже."

                    SPEAKING_EVENT.set()
                    try:
                        wav = await tts.synthesize(answer)
                        play_wav_bytes(wav.wav_bytes, device=cfg.audio.output_device)
                    finally:
                        SPEAKING_EVENT.clear()
                    continue

            # LLM
            messages = dialog.build_messages(user_text)
            t0 = time.monotonic()
            llm_res = await llm.chat(messages)
            dt = time.monotonic() - t0
            answer = llm_res.text or ""
            dialog.add_turn(user_text, answer)
            log.info("LLM latency=%.2fs | user='%s' | answer='%s'", dt, user_text, answer)

            # TTS
            if not answer:
                continue
            SPEAKING_EVENT.set()
            try:
                audio_res = await tts.synthesize(answer)
                play_wav_bytes(audio_res.wav_bytes, device=cfg.audio.output_device)
            finally:
                SPEAKING_EVENT.clear()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nExit")


if __name__ == "__main__":
    main()
