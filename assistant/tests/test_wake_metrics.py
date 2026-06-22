from __future__ import annotations


def test_wake_metrics_has_ratio_fields():
    # Ensure WakeMetrics remains backward-compatible and includes fields used by calibration.
    from assistant.pipeline_vosk import WakeMetrics

    wm = WakeMetrics(time_to_wake_sec=0.1, confidence=None, speech_ratio=0.5)
    assert hasattr(wm, "wake_text")
    assert hasattr(wm, "wake_ratio")
    assert wm.wake_ratio == 0.0

