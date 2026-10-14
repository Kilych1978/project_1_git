# ============================================================
# SELF TEST - MODULAR SYNC v6.9.1 MICRO WAVEFORM
# ============================================================

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import Progress
from micro_refiner import micro_search_at_time, refine_sync_map_with_micro_waveform, _is_strong_micro_candidate
from reports import _translate_report_line_to_russian


def _write_s16(path: Path, values: np.ndarray):
    clipped = np.clip(values, -1.0, 1.0)
    (clipped * 32767.0).astype("<i2").tofile(path)


def test_micro_search_recovers_known_shift():
    sample_rate = 48000
    duration = 8.0
    samples = int(sample_rate * duration)
    rng = np.random.default_rng(20260925)
    time_axis = np.arange(samples, dtype=np.float64) / sample_rate

    signal = (
        0.12 * np.sin(2.0 * np.pi * 233.0 * time_axis)
        + 0.07 * np.sin(2.0 * np.pi * 517.0 * time_axis + 0.4)
        + 0.025 * rng.normal(size=samples)
    )
    envelope = 0.15 + 0.85 * (
        0.5 + 0.5 * np.sin(2.0 * np.pi * 0.31 * time_axis + 0.8)
    )
    for center, width, amplitude in (
        (1.25, 0.07, 2.0),
        (2.35, 0.12, 1.7),
        (3.75, 0.05, 2.3),
        (5.15, 0.09, 1.8),
        (6.55, 0.11, 2.1),
    ):
        envelope += amplitude * np.exp(
            -0.5 * ((time_axis - center) / width) ** 2
        )
    reference = signal * envelope

    # TARGET is delayed by exactly 7 ms. Therefore the reference event
    # corresponding to a target event is 7 ms earlier, i.e. offset = -7 ms.
    shift_samples = int(round(0.007 * sample_rate))
    target = np.concatenate(
        [
            np.zeros(shift_samples, dtype=np.float32),
            reference[:-shift_samples],
        ]
    ).astype(np.float32)

    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        reference_path = temporary / "reference.pcm"
        target_path = temporary / "target.pcm"
        _write_s16(reference_path, reference.astype(np.float32))
        _write_s16(target_path, target)

        result = micro_search_at_time(
            reference_path,
            target_path,
            target_center_sec=4.0,
            predicted_offset_sec=0.0,
        )

    assert result is not None, "micro search returned no result"
    assert abs(result["refined_offset_sec"] + 0.007) <= 0.0015, result
    assert result["best_score"] >= 0.90, result



def test_micro_refinement_keeps_true_local_shift():
    """Проверяет, что ложный локальный пик не проходит обратную микропроверку."""
    sample_rate = 48000
    duration = 8.0
    samples = int(sample_rate * duration)
    rng = np.random.default_rng(44)
    time_axis = np.arange(samples, dtype=np.float64) / sample_rate

    signal = (
        0.12 * np.sin(2.0 * np.pi * 251.0 * time_axis)
        + 0.09 * np.sin(2.0 * np.pi * 613.0 * time_axis + 0.3)
        + 0.025 * rng.normal(size=samples)
    )
    envelope = 0.2 + 0.8 * (
        0.5 + 0.5 * np.sin(2.0 * np.pi * 0.4 * time_axis + 0.2)
    )
    for center, width, amplitude in (
        (2.0, 0.05, 2.5),
        (3.3, 0.08, 2.0),
        (4.6, 0.06, 3.0),
        (5.7, 0.10, 2.0),
    ):
        envelope += amplitude * np.exp(
            -0.5 * ((time_axis - center) / width) ** 2
        )

    reference = (signal * envelope).astype(np.float32)
    shift_samples = int(round(0.004 * sample_rate))
    target = np.concatenate(
        [
            np.zeros(shift_samples, dtype=np.float32),
            reference[:-shift_samples],
        ]
    ).astype(np.float32)

    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        reference_path = temporary / "reference.pcm"
        target_path = temporary / "target.pcm"
        output_dir = temporary / "reports"
        output_dir.mkdir()
        _write_s16(reference_path, reference)
        _write_s16(target_path, target)

        analysis = {
            "reference_pcm": str(reference_path),
            "target_pcm": str(target_path),
        }
        base_times = np.arange(0.0, duration, 0.5)
        base_map = {
            "target_time": base_times,
            "offset": np.zeros_like(base_times),
            "reference_time": base_times.copy(),
            "confidence": np.ones_like(base_times),
        }
        dense_results = [
            {
                "target_time": float(time_value),
                "after_error": 0.004,
                "before_error": 0.004,
                "after_offset": -0.004,
            }
            for time_value in np.arange(1.0, 7.0, 0.5)
        ]

        import micro_refiner
        old_frame = micro_refiner.MICRO_FEATURE_FRAME_MS
        micro_refiner.MICRO_FEATURE_FRAME_MS = 2.0
        try:
            result = refine_sync_map_with_micro_waveform(
                analysis,
                base_map,
                dense_results,
                output_dir,
                Progress(),
            )
        finally:
            micro_refiner.MICRO_FEATURE_FRAME_MS = old_frame

    accepted_times = [
        float(item["target_time_sec"])
        for item in result["accepted"]
    ]
    assert result["accepted"], "true micro correction was not accepted"
    assert all(
        abs(float(item["correction_sec"]) + 0.004) < 0.0015
        for item in result["accepted"]
    ), result["accepted"]
    assert all(
        abs(float(item["correction_sec"])) <= 0.020
        for item in result["accepted"]
    ), result["accepted"]


def test_micro_candidate_must_reduce_measured_error():
    good = {
        "best_score": 0.75,
        "rms_correlation": 0.70,
        "edge_correlation": 0.40,
        "score_gain": 0.08,
        "correction_sec": -0.008,
        "refined_offset_sec": 0.002,
        "peak_margin": 0.05,
        "reverse_score": 0.75,
        "reverse_rms_correlation": 0.70,
        "reverse_disagreement_sec": 0.001,
    }
    bad = dict(good)
    bad["refined_offset_sec"] = 0.009
    assert _is_strong_micro_candidate(good, -0.010)
    assert not _is_strong_micro_candidate(bad, 0.010)


def test_micro_score_gain_cannot_override_timing_worsening():
    # Визуально похожий фрагмент может дать высокий score, но если
    # абсолютная временная ошибка после коррекции становится больше,
    # микрокоррекция обязана быть отклонена.
    candidate = {
        "best_score": 0.82,
        "rms_correlation": 0.88,
        "edge_correlation": 0.55,
        "score_gain": 0.20,
        "correction_sec": 0.005668,
        "refined_offset_sec": 0.002802,
        "peak_margin": 0.09,
        "reverse_score": 0.84,
        "reverse_rms_correlation": 0.88,
        "reverse_disagreement_sec": 0.001,
    }
    # Current map residual is -1.366 ms; adding +5.668 ms moves it to
    # about -7.034 ms, so this is a regression despite the better shape score.
    assert not _is_strong_micro_candidate(candidate, -0.001366)


def test_micro_large_jump_is_rejected():
    candidate = {
        "best_score": 0.90,
        "rms_correlation": 0.90,
        "edge_correlation": 0.70,
        "score_gain": 0.20,
        "correction_sec": 0.030,
        "refined_offset_sec": 0.005,
        "peak_margin": 0.10,
        "reverse_score": 0.90,
        "reverse_rms_correlation": 0.90,
        "reverse_disagreement_sec": 0.001,
    }
    assert not _is_strong_micro_candidate(candidate, 0.020)


def test_report_translation_has_no_known_english_fragments():
    samples = [
        "КАРТА DIFFERENCE FROM PREVIOUS VERSION:",
        "ТОЛКОВАНИЕ ДЛИТЕЛЬНОСТИ: ДЛИТЕЛЬНОСТЬ ЗАКОДИРОВАННОГО ВЫХОДА is reported separately from synchronization residuals.",
        "Подтверждённых расхождений means both validation sides independently support a deviation from the КАРТА.",
        "Неоднозначных CANDIDATE means the waveform search did not provide enough independent evidence to confirm a КАРТА failure.",
        "Неоднозначных candidates do not modify the synchronization КАРТА and are diagnostic only.",
        "The guard compares candidate and SAFE КАРТА on the SAME dense independent validation grid. Rejected ТОЧЕК are hard barriers; correction cannot interpolate through them.",
        "ROBUST ВЕРДИКТ КАРТЫ:",
        "APPLY КАРТА:",
        "min tempo: 1.000000000",
        "max tempo: 1.000000000",
        "HARD OUTLIERS:",
        "HARD candidates: 0",
        "Warning outliers: 0",
        "Raw ТОЧЕК: 169",
    ]
    forbidden = (
        "DIFFERENCE FROM PREVIOUS VERSION",
        "is reported separately from synchronization residuals",
        "means both validation sides independently support",
        "CANDIDATE means",
        "candidates do not modify",
        "The guard compares",
        "ROBUST",
        "APPLY",
        "min tempo",
        "max tempo",
        "HARD OUTLIERS",
        "HARD candidates",
        "Warning outliers",
        "Raw ТОЧЕК",
    )
    for source in samples:
        translated = _translate_report_line_to_russian(source)
        assert not any(
            fragment.lower() in translated.lower()
            for fragment in forbidden
        ), (source, translated)

    # Report artifact names must remain byte-for-byte identical.
    filename_line = _translate_report_line_to_russian(
        "Источник: ALIGNMENT_VALIDATION_REPORT.txt"
    )
    assert filename_line.endswith("ALIGNMENT_VALIDATION_REPORT.txt"), filename_line
    path_line = _translate_report_line_to_russian(
        r"REFERENCE: C:\work\The.Beyond.Original_VERSION.wav"
    )
    assert path_line.endswith(r"C:\work\The.Beyond.Original_VERSION.wav"), path_line


def main():
    tests = [
        test_micro_search_recovers_known_shift,
        test_micro_refinement_keeps_true_local_shift,
        test_micro_candidate_must_reduce_measured_error,
        test_micro_score_gain_cannot_override_timing_worsening,
        test_micro_large_jump_is_rejected,
        test_report_translation_has_no_known_english_fragments,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("ALL v6.9.1 MICRO SELF TESTS PASSED")


if __name__ == "__main__":
    main()
