"""Cloud assistant with asyncio orchestrator.

Why
---
`assistant/main_cloud.py` is intentionally simple, but for real-world stability we want:

- queues between stages (wake/record → STT → LLM → TTS → playback)
- ability to interrupt playback with stop-word (best-effort, no AEC)
- bounded concurrency and clean shutdown

This entrypoint is the "production" variant.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
import threading
import time
import wave

from .asr_engines import VoskEngine
from .audio_playback import PlaybackInfo, start_playback, stop_playback, wait_playback
from .audio_stream import AudioStream
from .cloud_openai import OpenAILlm, OpenAIStt, OpenAITts
from .config import AppConfig
from .dialog import DialogState
from .intent_lights import detect_lights_intent
from .intent_weather import detect_weather_intent
from .intent_spotify import detect_spotify_intent
from .pipeline_vosk import PipelineCancelled, PipelineConfig, WakeCommandPipeline
from .postprocess import clean_for_speech
from .speech import SPEAKING_EVENT, get_default_input_device
from .stop_word import StopWordConfig, StopWordDetector
from .tts_cache import TtsCache
from .utils import normalize_text
from .weather_skill import WeatherSkillError, handle_weather_intent
from .lights_adapter_factory import build_gauss_adapter
from .lights_device_manager import LightsManager, LightsManagerConfig
from .lights_skill import LightsSkillDeps, LightsSkillError, handle_lights_intent
from .spotify_client import SpotifyClient, SpotifyClientConfig
from .spotify_skill import SpotifySkillError, handle_spotify_intent


log = logging.getLogger("assistant.main_cloud_orchestrated")

# Bump this when troubleshooting to confirm the running file version in logs.
BUILD_ID = "2026-02-03.2"


async def _put_latest(q: asyncio.Queue, item) -> None:
    """Put to queue without ever blocking the audio loop.

    If queue is full we drop the oldest item.
    """

    if q.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            _ = q.get_nowait()
    with contextlib.suppress(asyncio.QueueFull):
        q.put_nowait(item)


async def _health_log_loop(
    *,
    q0: asyncio.Queue,
    q1: asyncio.Queue,
    q2: asyncio.Queue,
    q3: asyncio.Queue,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        log.debug("health: q0=%d q1=%d q2=%d q3=%d", q0.qsize(), q1.qsize(), q2.qsize(), q3.qsize())
        await asyncio.sleep(2.0)


def pcm16_to_wav_bytes(pcm16: bytes, *, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm16)
    return buf.getvalue()


class PlaybackController:
    def __init__(self):
        # Serializes play_blocking() calls so two playbacks never overlap.
        # Deliberately NOT used by stop(): play_blocking() holds it for the
        # whole duration of playback, so stop() waiting on the same lock
        # would just block until playback already finished on its own —
        # confirmed on real hardware: stop-word detection fired correctly
        # mid-playback, but had nothing to interrupt (see also the
        # start_playback/wait_playback split in audio_playback.py).
        self._play_lock = asyncio.Lock()
        # Brief-hold lock protecting _current/_playing reads/writes.
        self._state_lock = asyncio.Lock()
        self._current: PlaybackInfo | None = None
        self._playing = False

    @property
    def playing(self) -> bool:
        return bool(self._playing)

    async def play_blocking(self, wav_bytes: bytes, *, device: int | None) -> None:
        async with self._play_lock:
            # Starting is quick (just spawns the player); publish `_current`
            # before the long wait, not after, so stop() has something to act on.
            info = await asyncio.to_thread(start_playback, wav_bytes, device=device)
            async with self._state_lock:
                self._current = info
                self._playing = True
            try:
                await asyncio.to_thread(wait_playback, info)
            finally:
                async with self._state_lock:
                    self._playing = False
                    self._current = None

    async def stop(self) -> None:
        async with self._state_lock:
            info = self._current
        if info is None:
            return
        await asyncio.to_thread(stop_playback, info)


async def _play_ack(*, playback: PlaybackController, cfg: AppConfig, ack_wav: bytes) -> None:
    SPEAKING_EVENT.set()
    try:
        await playback.play_blocking(ack_wav, device=cfg.audio.output_device)
    except Exception:
        log.exception("ack playback failed")
    finally:
        SPEAKING_EVENT.clear()


async def _audio_loop(
    *,
    pipeline: WakeCommandPipeline,
    out_q: asyncio.Queue,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        try:
            interaction = await asyncio.to_thread(pipeline.run_once)
        except PipelineCancelled:
            log.info("audio_loop: cancelled, stopping")
            break
        if interaction is None:
            continue
        log.info(
            "interaction: wake_ratio=%.3f stop=%s embedded=%s pcm=%d bytes",
            getattr(interaction.wake, "wake_ratio", 0.0),
            interaction.command.stop_reason,
            bool(interaction.final_text),
            len(interaction.command_pcm16 or b""),
        )
        await _put_latest(out_q, interaction)


async def _worker_stt(
    *,
    in_q: asyncio.Queue,
    out_q: asyncio.Queue,
    stt: OpenAIStt,
    stop_event: asyncio.Event,
    playback: PlaybackController | None = None,
    cfg: AppConfig | None = None,
    ack_wav: bytes = b"",
) -> None:
    while not stop_event.is_set():
        interaction = await in_q.get()
        try:
            user_text = (interaction.final_text or "").strip()
            if user_text:
                log.info("STT skipped (embedded text): '%s'", user_text)
            if not user_text and interaction.command_pcm16:
                wav_bytes = pcm16_to_wav_bytes(
                    interaction.command_pcm16,
                    sample_rate=interaction.command_sample_rate,
                )
                log.info("STT start (wav_bytes=%d)", len(wav_bytes))
                t0 = time.monotonic()
                stt_res = await stt.transcribe_wav(wav_bytes)
                log.info("STT latency=%.2fs", time.monotonic() - t0)
                user_text = (stt_res.text or "").strip()
                log.info("STT text: '%s'", user_text)
            if user_text and ack_wav and playback is not None and cfg is not None:
                # Fire-and-forget: lets the user know the command was heard
                # while the (possibly slow — LLM web search, weather retries)
                # real answer is still being worked on. PlaybackController's
                # play lock naturally sequences the real answer right after
                # this, whenever _player() gets to it.
                asyncio.create_task(_play_ack(playback=playback, cfg=cfg, ack_wav=ack_wav), name="ack")
            await _put_latest(out_q, (interaction, user_text))
        except Exception:
            log.exception("STT worker failed")
        finally:
            in_q.task_done()


async def _worker_llm(
    *,
    in_q: asyncio.Queue,
    out_q: asyncio.Queue,
    llm: OpenAILlm,
    dialog: DialogState,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        interaction, user_text = await in_q.get()
        try:
            if not user_text:
                await _put_latest(out_q, (interaction, user_text, ""))
                continue

            # Deterministic skill routing (weather)
            cfg = AppConfig()  # env-backed; fine here; can be hoisted later if needed
            norm = normalize_text(user_text)

            # Routing probe (helps diagnose intent collisions).
            # Enable by setting ASSISTANT_ROUTE_DEBUG=1.
            route_dbg = os.environ.get("ASSISTANT_ROUTE_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
            li_probe = detect_lights_intent(norm) if cfg.lights.enabled else None
            si_probe = detect_spotify_intent(norm) if cfg.spotify.enabled else None
            wi_probe = detect_weather_intent(norm) if cfg.weather.enabled else None
            if route_dbg:
                log.info(
                    "ROUTE probe: lights=%s spotify=%s weather=%s | text='%s'",
                    None if li_probe is None else li_probe.action,
                    None if si_probe is None else si_probe.action,
                    None if wi_probe is None else wi_probe.intent,
                    norm,
                )
                if si_probe is not None:
                    log.info(
                        "ROUTE probe spotify detail: action=%s content_kind=%s query=%r uri_or_url=%r",
                        si_probe.action,
                        si_probe.content_kind,
                        si_probe.query,
                        si_probe.uri_or_url,
                    )

            # Deterministic skill routing (lights)
            if cfg.lights.enabled:
                li = li_probe
                if li is not None:
                    try:
                        # For now we only have a mock adapter. Real adapter can be selected via cfg.lights.adapter.
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
                        await _put_latest(out_q, (interaction, user_text, answer))
                        continue
                    except LightsSkillError as e:
                        await _put_latest(out_q, (interaction, user_text, str(e)))
                        continue
                    except Exception:
                        log.exception("LIGHTS skill failed")
                        await _put_latest(out_q, (interaction, user_text, "Не смог управлять светом. Попробуй чуть позже."))
                        continue

            # Deterministic skill routing (spotify)
            if cfg.spotify.enabled:
                si = si_probe
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
                            normalized_text=norm,
                            preferred_device_name=(cfg.spotify.preferred_device_name or None),
                        )
                        await _put_latest(out_q, (interaction, user_text, answer))
                        continue
                    except SpotifySkillError as e:
                        await _put_latest(out_q, (interaction, user_text, str(e)))
                        continue
                    except Exception:
                        log.exception("SPOTIFY skill failed")
                        await _put_latest(out_q, (interaction, user_text, "Не смог выполнить команду Spotify. Попробуй чуть позже."))
                        continue

            if cfg.weather.enabled:
                wi = wi_probe
                if wi is not None:
                    log.info("WEATHER intent=%s place=%s", wi.intent, wi.place_text)
                    try:
                        answer = await handle_weather_intent(wi, cfg=cfg.weather)
                        await _put_latest(out_q, (interaction, user_text, answer))
                        continue
                    except WeatherSkillError as e:
                        await _put_latest(out_q, (interaction, user_text, str(e)))
                        continue
                    except Exception:
                        log.exception("WEATHER skill failed")
                        await _put_latest(out_q, (interaction, user_text, "Не смог получить погоду. Попробуй чуть позже."))
                        continue

            log.info("LLM start: '%s'", user_text)
            msgs = dialog.build_messages(user_text)
            t0 = time.monotonic()
            llm_res = await llm.chat(msgs)
            answer = (llm_res.text or "").strip()
            dialog.add_turn(user_text, answer)
            log.info("LLM latency=%.2fs", time.monotonic() - t0)
            log.info("LLM answer: '%s'", answer)
            await _put_latest(out_q, (interaction, user_text, answer))
        except Exception:
            log.exception("LLM worker failed")
        finally:
            in_q.task_done()


async def _worker_tts(
    *,
    in_q: asyncio.Queue,
    out_q: asyncio.Queue,
    tts: OpenAITts,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        interaction, user_text, answer = await in_q.get()
        try:
            if not answer:
                await _put_latest(out_q, (interaction, user_text, answer, b""))
                continue
            # Web-search-grounded LLM answers embed markdown citations that
            # read fine but sound like noise out loud; dialog history keeps
            # the original (already stored before this stage).
            answer = clean_for_speech(answer)
            if not answer:
                await _put_latest(out_q, (interaction, user_text, answer, b""))
                continue
            log.info("TTS start (len=%d)", len(answer))
            t0 = time.monotonic()
            audio_res = await tts.synthesize(answer)
            log.info("TTS latency=%.2fs", time.monotonic() - t0)
            await _put_latest(out_q, (interaction, user_text, answer, audio_res.wav_bytes))
        except Exception:
            log.exception("TTS worker failed")
        finally:
            in_q.task_done()


async def _player(
    *,
    in_q: asyncio.Queue,
    playback: PlaybackController,
    cfg: AppConfig,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        interaction, user_text, answer, wav_bytes = await in_q.get()
        try:
            if not user_text:
                # user said nothing intelligible
                wav_bytes = (await OpenAITts(cfg.cloud).synthesize("Не расслышал. Повтори, пожалуйста.")).wav_bytes
            if wav_bytes:
                SPEAKING_EVENT.set()
                try:
                    log.info("playback start (wav_bytes=%d)", len(wav_bytes))
                    await playback.play_blocking(wav_bytes, device=cfg.audio.output_device)
                    log.info("playback done")
                finally:
                    SPEAKING_EVENT.clear()
        except Exception:
            log.exception("player failed")
        finally:
            in_q.task_done()


async def _stop_word_loop(
    *,
    audio: AudioStream,
    detector: StopWordDetector,
    playback: PlaybackController,
    stop_event: asyncio.Event,
) -> None:
    # Dedicated subscriber so pipeline.clear() doesn't starve us.
    q = audio.subscribe(queue_max_chunks=400)
    try:
        while not stop_event.is_set():
            # IMPORTANT: AudioStream.read() is blocking; run it in a worker thread so
            # we don't freeze the asyncio event loop.
            ch = await asyncio.to_thread(audio.read, 0.2, q=q)
            if ch is None:
                continue
            # Only listen while something is playing.
            if not playback.playing:
                continue
            if detector.accept_frame(ch.pcm16):
                log.info("stop-word: stopping playback")
                await playback.stop()
            # be a good asyncio citizen
            await asyncio.sleep(0)
    finally:
        audio.unsubscribe(q)


async def run() -> None:
    cfg = AppConfig()

    level = os.environ.get("ASSISTANT_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    device_in = cfg.audio.input_device
    if device_in is None:
        device_in = get_default_input_device()
    frame_samples = int(cfg.audio.sample_rate * cfg.audio.frame_ms / 1000)

    model_path = os.environ.get("VOSK_MODEL_PATH") or os.path.join(
        os.path.dirname(__file__), "models", "vosk-model-ru"
    )

    wake_engine = VoskEngine(model_path=model_path, sample_rate=cfg.audio.sample_rate)

    pipe_cfg = PipelineConfig(
        sample_rate=cfg.audio.sample_rate,
        frame_ms=cfg.audio.frame_ms,
        wake_phrase=cfg.wake.wake_phrase,
        wake_match_threshold=cfg.wake.wake_match_threshold,
        wake_min_speech_ratio=cfg.wake.wake_min_speech_ratio,
        wake_min_conf=cfg.wake.wake_min_conf,
        wake_tail_drop_ms=cfg.wake.wake_tail_drop_ms,
        wake_cooldown_sec=cfg.wake.wake_cooldown_sec,
        command_start_timeout_sec=cfg.turn.command_start_timeout_sec,
        max_silence_sec=cfg.turn.max_silence_sec,
        max_phrase_sec=cfg.turn.max_phrase_sec,
        min_phrase_sec=cfg.turn.min_phrase_sec,
        skip_command_asr=True,
    )

    stt = OpenAIStt(cfg.cloud)
    llm = OpenAILlm(cfg.cloud)
    tts = OpenAITts(cfg.cloud)
    dialog = DialogState(system_prompt=cfg.turn.system_prompt, max_turns=cfg.turn.max_context_turns)

    # Pre-synthesize the "heard you, one sec" ack once so playing it later
    # adds ~no latency of its own (see ACK_ENABLED). Cached to disk too, so
    # most restarts don't even need an API call for it.
    ack_wav = b""
    if cfg.turn.ack_enabled and cfg.turn.ack_phrase.strip():
        tts_cache = TtsCache.default()
        cache_kwargs = dict(
            provider="openai",
            model=cfg.cloud.openai_tts_model,
            voice=cfg.cloud.openai_tts_voice,
            text=cfg.turn.ack_phrase,
        )
        ack_wav = tts_cache.get(**cache_kwargs) or b""
        if ack_wav:
            log.info("ack phrase loaded from cache (%d bytes)", len(ack_wav))
        else:
            try:
                ack_res = await tts.synthesize(cfg.turn.ack_phrase)
                ack_wav = ack_res.wav_bytes
                tts_cache.put(**cache_kwargs, wav_bytes=ack_wav)
                log.info("ack phrase synthesized and cached (%d bytes)", len(ack_wav))
            except Exception:
                log.exception("failed to pre-synthesize ack phrase; ack disabled for this run")
                ack_wav = b""

    # If suppress_mic_during_tts=True, stop-word cannot work.
    # So we override ignore_event based on ENABLE_STOP_WORD.
    ignore_event = None
    if cfg.audio.suppress_mic_during_tts and not cfg.turn.enable_stop_word:
        ignore_event = SPEAKING_EVENT

    stop_event = asyncio.Event()
    # `_audio_loop` calls `pipeline.run_once()` via `asyncio.to_thread`, which can
    # block indefinitely inside `wait_for_wake()` if the wake phrase is never heard.
    # `stop_event` (asyncio.Event) isn't visible/safe to poll from that worker
    # thread, so we use a plain threading.Event the pipeline checks internally to
    # unblock promptly on shutdown instead of hanging process exit.
    cancel_event = threading.Event()
    playback = PlaybackController()

    stop_words = [w.strip() for w in cfg.turn.stop_words.split(",") if w.strip()]
    sw_cfg = StopWordConfig(
        sample_rate=cfg.audio.sample_rate,
        frame_ms=cfg.audio.frame_ms,
        words=stop_words,
        cooldown_sec=cfg.turn.stop_word_cooldown_sec,
    )
    # Important: share the loaded Vosk model to avoid loading it twice (RAM on RPi 3B).
    # Free-form decoding, not grammar-constrained: `UpdateGrammarFst` requires a
    # lexicon the small Vosk model doesn't ship, so a grammar-locked recognizer
    # silently never matches *any* word (confirmed empirically), which is why
    # stop-words previously never fired. StopWordDetector already fuzzy-matches
    # the free-form text against `cfg.words`, so this needs no other changes.
    stop_engine = wake_engine.clone_with(sample_rate=cfg.audio.sample_rate)
    stop_detector = StopWordDetector(cfg=sw_cfg, engine=stop_engine)

    log.info("Cloud orchestrated assistant ready (build=%s). Wake='%s'", BUILD_ID, cfg.wake.wake_phrase)
    log.info("Input device id: %s", device_in)
    log.info("Stop-word enabled: %s (%s)", cfg.turn.enable_stop_word, ",".join(stop_words))

    q0: asyncio.Queue = asyncio.Queue(maxsize=2)
    q1: asyncio.Queue = asyncio.Queue(maxsize=2)
    q2: asyncio.Queue = asyncio.Queue(maxsize=2)
    q3: asyncio.Queue = asyncio.Queue(maxsize=2)

    with AudioStream(
        sample_rate=cfg.audio.sample_rate,
        frame_samples=frame_samples,
        device=device_in,
        ignore_event=ignore_event,
        queue_max_chunks=500,
        input_gain=cfg.audio.input_gain,
    ) as audio:
        pipeline = WakeCommandPipeline(
            cfg=pipe_cfg,
            audio=audio,
            wake_engine=wake_engine,
            command_engine=wake_engine,
            cancel_event=cancel_event,
        )

        tasks = []

        def _log_task_done(t: asyncio.Task) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is None:
                return
            log.error("task crashed: %s: %r", t.get_name(), exc)
            tb = getattr(exc, "__traceback__", None)
            if tb is not None:
                log.error("task traceback:", exc_info=(type(exc), exc, tb))

        for t in [
            asyncio.create_task(_audio_loop(pipeline=pipeline, out_q=q0, stop_event=stop_event), name="audio_loop"),
            asyncio.create_task(
                _worker_stt(
                    in_q=q0,
                    out_q=q1,
                    stt=stt,
                    stop_event=stop_event,
                    playback=playback,
                    cfg=cfg,
                    ack_wav=ack_wav,
                ),
                name="stt",
            ),
            asyncio.create_task(_worker_llm(in_q=q1, out_q=q2, llm=llm, dialog=dialog, stop_event=stop_event), name="llm"),
            asyncio.create_task(_worker_tts(in_q=q2, out_q=q3, tts=tts, stop_event=stop_event), name="tts"),
            asyncio.create_task(_player(in_q=q3, playback=playback, cfg=cfg, stop_event=stop_event), name="player"),
            asyncio.create_task(_health_log_loop(q0=q0, q1=q1, q2=q2, q3=q3, stop_event=stop_event), name="health"),
        ]:
            t.add_done_callback(_log_task_done)
            tasks.append(t)
        if cfg.turn.enable_stop_word:
            swt = asyncio.create_task(
                _stop_word_loop(audio=audio, detector=stop_detector, playback=playback, stop_event=stop_event),
                name="stop_word",
            )
            swt.add_done_callback(_log_task_done)
            tasks.append(swt)

        try:
            log.info("tasks started")
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            raise
        except KeyboardInterrupt:
            pass
        except Exception:
            log.exception("asyncio.gather failed")
        finally:
            stop_event.set()
            cancel_event.set()
            for t in tasks:
                t.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nExit")


if __name__ == "__main__":
    main()
