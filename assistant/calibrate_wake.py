"""Wake calibration helper.

Goal
----
Semi-automatic tuning of:

- WAKE_MATCH_THRESHOLD
- WAKE_MIN_SPEECH_RATIO

Approach
--------
We run the real wake detector in "probe" mode for a fixed time and log candidate
stats (ratio/conf/speech_ratio). Then we propose thresholds:

- threshold = percentile of wake candidate ratios minus margin
- min_speech_ratio = percentile of speech_ratio in wake candidates minus margin

This isn't perfect ML calibration, but it's practical and robust.
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import time

from .asr_engines import VoskEngine
from .audio_stream import AudioStream
from .config import AppConfig
from .pipeline_vosk import PipelineConfig, WakeCommandPipeline
from .speech import get_default_input_device


log = logging.getLogger("assistant.calibrate_wake")


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs2 = sorted(xs)
    i = int(round((len(xs2) - 1) * p))
    i = max(0, min(len(xs2) - 1, i))
    return float(xs2[i])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = AppConfig()
    device_in = args.device
    if device_in is None:
        device_in = cfg.audio.input_device
    if device_in is None:
        device_in = get_default_input_device()

    model_path = os.environ.get("VOSK_MODEL_PATH") or os.path.join(
        os.path.dirname(__file__), "models", "vosk-model-ru"
    )
    wake_engine = VoskEngine(model_path=model_path, sample_rate=cfg.audio.sample_rate)

    pipe_cfg = PipelineConfig(
        sample_rate=cfg.audio.sample_rate,
        frame_ms=cfg.audio.frame_ms,
        wake_phrase=cfg.wake.wake_phrase,
        wake_match_threshold=0.0,  # probe: accept everything that produces text
        wake_min_speech_ratio=0.0,
        wake_tail_drop_ms=cfg.wake.wake_tail_drop_ms,
        wake_cooldown_sec=0.0,
        command_start_timeout_sec=0.2,
    )

    frame_samples = int(cfg.audio.sample_rate * cfg.audio.frame_ms / 1000)

    ratios: list[float] = []
    speech_ratios: list[float] = []
    confs: list[float] = []

    log.info("Calibration started for %.1fs. Say wake '%s' ~20 times in разных условиях.", args.seconds, cfg.wake.wake_phrase)

    deadline = time.monotonic() + float(args.seconds)
    with AudioStream(sample_rate=cfg.audio.sample_rate, frame_samples=frame_samples, device=device_in, queue_max_chunks=500) as audio:
        pipeline = WakeCommandPipeline(cfg=pipe_cfg, audio=audio, wake_engine=wake_engine, command_engine=wake_engine)
        while time.monotonic() < deadline:
            wm = pipeline.wait_for_wake()
            speech_ratios.append(float(wm.speech_ratio))
            if wm.confidence is not None:
                confs.append(float(wm.confidence))
            ratios.append(float(wm.wake_ratio))
            log.info(
                "wake observed: ratio=%.3f speech_ratio=%.3f conf=%s text='%s' embedded='%s'",
                wm.wake_ratio,
                wm.speech_ratio,
                wm.confidence,
                wm.wake_text,
                wm.embedded_command_text,
            )

    # Propose thresholds.
    # We pick low percentiles to keep recall high, then subtract a small margin.
    # You can raise them later to reduce false accepts.
    if speech_ratios:
        p10 = _percentile(speech_ratios, 0.10)
        p25 = _percentile(speech_ratios, 0.25)
        proposed_speech = max(0.0, min(0.25, p10 - 0.02))
    else:
        proposed_speech = cfg.wake.wake_min_speech_ratio

    if ratios:
        r10 = _percentile(ratios, 0.10)
        proposed_ratio = max(0.45, min(0.85, r10 - 0.03))
    else:
        proposed_ratio = cfg.wake.wake_match_threshold

    log.info("\nProposed env (start point):")
    log.info("  export WAKE_MATCH_THRESHOLD='%.2f'", proposed_ratio)
    log.info("  export WAKE_MIN_SPEECH_RATIO='%.2f'", proposed_speech)
    if ratios:
        log.info("  # ratio stats: min=%.3f p10=%.3f median=%.3f max=%.3f", min(ratios), _percentile(ratios, 0.10), statistics.median(ratios), max(ratios))
    if speech_ratios:
        log.info(
            "  # speech_ratio stats: min=%.3f p10=%.3f median=%.3f max=%.3f",
            min(speech_ratios),
            _percentile(speech_ratios, 0.10),
            statistics.median(speech_ratios),
            max(speech_ratios),
        )


if __name__ == "__main__":
    main()
