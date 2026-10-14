# ============================================================
# SELF TEST - MODULAR SYNC v6.8.1
# ============================================================

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import Progress
from confidence import classify_local_measurements, build_confidence_aware_anchors, build_confidence_aware_map
from safety import build_non_regressive_sync_map
from render_guard import compare_rendered_against_original
from validator import find_validation_outliers, validation_search_offset


def _result(time_sec, predicted, before_offset, before_error=None, after_error=None, corr=0.90):
    if before_error is None:
        before_error = abs(float(before_offset) - float(predicted))
    if after_error is None:
        after_error = before_error
    return {
        "target_time": float(time_sec),
        "predicted_offset": float(predicted),
        "before_offset": float(before_offset),
        "before_correlation": float(corr),
        "before_bidirectional_offset": float(before_offset),
        "before_bidirectional_correlation": float(corr),
        "before_bidirectional_status": "CONFIRMED_MAP",
        "before_error": float(before_error),
        "after_offset": float(before_offset),
        "after_correlation": float(corr),
        "after_error": float(after_error),
        "raw_after_offset": float(before_offset),
        "raw_after_correlation": float(corr),
        "bidir_left_offset": float(before_offset),
        "bidir_left_correlation": float(corr),
        "bidir_right_offset": float(before_offset),
        "bidir_right_correlation": float(corr),
        "bidir_consensus_offset": float(before_offset),
        "bidir_consensus_correlation": float(corr),
        "bidir_disagreement": 0.0,
        "bidir_prediction_error": abs(float(before_offset) - float(predicted)),
        "bidir_status": "CONFIRMED_MAP",
    }


def test_independent_search():
    rng = np.random.default_rng(12345)
    envelope = rng.normal(size=10000)
    result = validation_search_offset(
        envelope,
        envelope,
        100.0,
        0.600,
        20.0,
        3.0,
        0.01,
        0.01,
    )
    assert result["offset"] is not None
    assert abs(result["offset"]) < 0.020, result


def test_singleton_strong_jump_is_rejected():
    local = [
        {"target_time": 1165.009, "offset": 0.014, "correlation": 0.90, "margin": 0.10, "accepted_for_tracking": True},
        {"target_time": 1195.009, "offset": 0.766, "correlation": 0.90, "margin": 0.10, "accepted_for_tracking": True},
        {"target_time": 1225.009, "offset": -0.004, "correlation": 0.90, "margin": 0.10, "accepted_for_tracking": True},
    ]
    classified = classify_local_measurements(local)
    result = build_confidence_aware_anchors(classified, Progress())
    assert result["classes"][1] == "STRONG"
    assert not bool(result["keep"][1])
    assert bool(result["strong_continuity_rejected"][1])
    sync_map = build_confidence_aware_map(result, 1255.0, Progress())
    assert float(np.max(np.abs(sync_map["offset"]))) < 0.100


def test_already_synced_micro_correction_is_suppressed():
    target_time = np.arange(0.0, 101.0, 1.0)
    candidate_offset = np.full_like(target_time, -0.0037)
    candidate_map = {
        "target_time": target_time,
        "offset": candidate_offset,
        "reference_time": target_time + candidate_offset,
        "confidence": np.ones_like(target_time),
    }
    validation = []
    for t in np.arange(10.0, 91.0, 10.0):
        validation.append(_result(
            t,
            predicted=-0.0037,
            before_offset=-0.0035,
            before_error=0.0002,
            after_error=0.0002,
            corr=0.80,
        ))
    safe_map, report = build_non_regressive_sync_map(
        candidate_map,
        validation,
        100.0,
        Progress(),
    )
    assert report["already_synced"] is True, report
    assert float(np.max(np.abs(safe_map["offset"]))) <= 1e-9, float(np.max(np.abs(safe_map["offset"])))


def test_meaningful_correction_is_kept():
    target_time = np.arange(0.0, 101.0, 1.0)
    candidate_offset = np.full_like(target_time, -0.018)
    candidate_map = {
        "target_time": target_time,
        "offset": candidate_offset,
        "reference_time": target_time + candidate_offset,
        "confidence": np.ones_like(target_time),
    }
    validation = []
    for t in np.arange(10.0, 91.0, 10.0):
        validation.append(_result(
            t,
            predicted=-0.018,
            before_offset=-0.018,
            before_error=0.018,
            after_error=0.0,
            corr=0.90,
        ))
    safe_map, report = build_non_regressive_sync_map(candidate_map, validation, 100.0, Progress())
    assert report["already_synced"] is True
    assert float(np.max(np.abs(safe_map["offset"]))) > 0.010


def test_render_comparison_detects_local_regression():
    baseline = [
        _result(10.0, 0.0, -0.005, before_error=0.005, after_error=0.005, corr=0.90),
        _result(20.0, 0.0, -0.005, before_error=0.005, after_error=0.005, corr=0.90),
        _result(30.0, 0.0, -0.004, before_error=0.004, after_error=0.004, corr=0.90),
    ]
    rendered = []
    for item in baseline:
        row = dict(item)
        row["bidir_consensus_offset"] = item["before_bidirectional_offset"]
        rendered.append(row)
    rendered[1]["bidir_consensus_offset"] = 0.009
    rendered[1]["bidir_consensus_correlation"] = 0.70
    comparison = compare_rendered_against_original(baseline, rendered)
    assert comparison["regression_count"] == 1, comparison
    assert comparison["regressions"][0]["target_time_sec"] == 20.0




def test_user_3318_micro_regression_case():
    """Точная модель наблюдаемой точки около 00:33:18.869."""
    target_time = np.arange(0.0, 101.0, 1.0)
    candidate_map = {
        "target_time": target_time,
        "offset": np.full_like(target_time, -0.003701),
        "reference_time": target_time - 0.003701,
        "confidence": np.ones_like(target_time),
    }
    validation = []
    for t in np.arange(10.0, 91.0, 10.0):
        validation.append(_result(
            t,
            predicted=-0.003701,
            before_offset=-0.0035,
            before_error=0.000201,
            after_error=0.000201,
            corr=0.80,
        ))
    safe_map, report = build_non_regressive_sync_map(
        candidate_map, validation, 100.0, Progress()
    )
    assert report["already_synced"] is True, report
    assert report["validation_decisions"], report
    assert all(d["weight"] == 0.0 for d in report["validation_decisions"]), report
    assert float(np.max(np.abs(safe_map["offset"]))) <= 1e-9




def test_render_guard_zero_map_threshold_import():
    import render_guard
    assert float(render_guard.NO_REGRESSION_ZERO_MAP_SEC) == 0.0015

def test_report_translation_is_russian():
    from reports import _translate_report_line_to_russian
    samples = {
        "Validation outlier candidates: 1": "КАНДИДАТОВ НА ВЫБРОСЫ ПРОВЕРКИ",
        "ROBUST VERDICT: EXCELLENT": "УСТОЙЧИВЫЙ ВЕРДИКТ: ОТЛИЧНО",
        "SOURCE TARGET has NOT been modified.": "Исходный TARGET НЕ изменялся.",
        "Minimum meaningful correction for already-synchronized TARGET: 0.006000 s": "Минимальная значимая коррекция",
        "Local quarantine may exclude a small problematic region without cancelling unrelated confirmed improvements.": "Локальный карантин",
    }
    for source, expected in samples.items():
        translated = _translate_report_line_to_russian(source)
        assert expected in translated, (source, translated)
        english_tokens = ("Validation outlier candidates", "ROBUST VERDICT", "has NOT been modified", "Local quarantine may exclude")
        assert not any(token.lower() in translated.lower() for token in english_tokens), (source, translated)


def test_confirmed_discrepancy_is_exportable():
    validation = [{
        "target_time": 2418.0,
        "predicted_offset": 0.0,
        "after_offset": 0.174,
        "after_error": 0.174,
        "after_correlation": 0.9,
        "bidir_status": "CONFIRMED_DISCREPANCY",
    }]
    outliers = find_validation_outliers(validation)
    assert len(outliers) == 1
    assert outliers[0]["severity"] == "CONFIRMED_DISCREPANCY"


def main():
    tests = [
        test_independent_search,
        test_singleton_strong_jump_is_rejected,
        test_already_synced_micro_correction_is_suppressed,
        test_meaningful_correction_is_kept,
        test_render_comparison_detects_local_regression,
        test_user_3318_micro_regression_case,
        test_render_guard_zero_map_threshold_import,
        test_report_translation_is_russian,
        test_confirmed_discrepancy_is_exportable,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("ALL v6.8.2 SELF TESTS PASSED")


if __name__ == "__main__":
    main()
