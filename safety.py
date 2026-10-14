# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC — MODULAR COMPONENT
# MODULE: safety.py
# ============================================================
# v6.7 DENSE NON-REGRESSION GUARD
#
# Principle:
#   The candidate map is never trusted merely because a sparse validation
#   point looks good. Candidate and SAFE MAP are evaluated on the SAME dense
#   independent validation grid. Corrections are accepted only in contiguous
#   runs of independently supported points. Rejected points form hard gaps;
#   correction cannot interpolate through them.
# ============================================================

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from config import *
from common import *

# ---------------------------------------------------------------------------
# Conservative safety thresholds.
# ---------------------------------------------------------------------------

NO_REGRESSION_TOLERANCE_SEC = 0.0015
NO_REGRESSION_CLEAR_IMPROVEMENT_SEC = 0.0015
NO_REGRESSION_ALREADY_SYNCED_SEC = 0.020
NO_REGRESSION_ZERO_MAP_SEC = 0.0015

# v6.8.1: already-synchronized TARGETs receive minimal intervention.
# Very small corrections can change the local waveform shape without providing
# a robust audible benefit. They are therefore treated as micro-corrections and
# suppressed unless both the correction and the measured improvement are large
# enough to justify changing an already good source.
NO_REGRESSION_MIN_MEANINGFUL_CORRECTION_SEC = 0.006
NO_REGRESSION_MIN_MEANINGFUL_IMPROVEMENT_SEC = 0.004

# v6.7: dense safety grid and continuity requirement.
SAFETY_DENSE_VALIDATION_INTERVAL_SEC = 10.0
SAFETY_MIN_ACCEPTED_RUN_POINTS = 3
SAFETY_MIN_RUN_SPAN_SEC = 20.0

# v6.7.2: local quarantine for dense points that demonstrably worsen.
# The quarantine is wider than one validation sample because validation uses
# a finite analysis window around each timestamp.
SAFETY_QUARANTINE_RADIUS_SEC = max(
    float(VALIDATION_WINDOW_SEC),
    2.0 * SAFETY_DENSE_VALIDATION_INTERVAL_SEC,
)
SAFETY_MAX_QUARANTINE_FRACTION = 0.20

NON_REGRESSION_REPORT = "NON_REGRESSION_GUARD_REPORT.txt"
CANDIDATE_VALIDATION_JSON = "CANDIDATE_VALIDATION.json"
SAFE_VALIDATION_JSON = "SAFE_MAP_VALIDATION.json"
DENSE_SAFETY_VALIDATION_CSV = "DENSE_SAFETY_VALIDATION.csv"
DENSE_SAFETY_CANDIDATE_JSON = "DENSE_SAFETY_CANDIDATE.json"
DENSE_SAFETY_FINAL_JSON = "DENSE_SAFETY_FINAL.json"


def build_zero_sync_map(target_duration: float):
    """Карта без временной коррекции."""
    target_time = build_exact_time_grid(
        0.0,
        float(target_duration),
        MAP_STEP_SEC,
    )
    return {
        "target_time": target_time,
        "offset": np.zeros_like(target_time, dtype=np.float64),
        "reference_time": target_time.copy(),
        "confidence": np.ones_like(target_time, dtype=np.float64),
        "anchor_time": np.asarray([], dtype=np.float64),
        "anchor_offset": np.asarray([], dtype=np.float64),
    }


def _validation_weights(validation_results):
    """
    Локальное решение по каждой dense independent validation point.

    В отличие от v6.6, статус CONFIRMED_MAP сам по себе уже недостаточен.
    Нужна измеримая положительная разница относительно текущего TARGET.
    Для already-synchronized материала дополнительно требуется сильная
    двусторонняя поддержка.
    """
    records = []
    for result in validation_results:
        t = result.get("target_time")
        before_error = result.get("before_error")
        after_error = result.get("after_error")
        status = str(result.get("bidir_status", "UNRESOLVED"))
        if t is None or before_error is None or after_error is None:
            continue

        left_corr = float(result.get("bidir_left_correlation", 0.0) or 0.0)
        right_corr = float(result.get("bidir_right_correlation", 0.0) or 0.0)
        disagreement = result.get("bidir_disagreement")
        try:
            disagreement = float(disagreement)
        except Exception:
            disagreement = float("inf")

        try:
            predicted_offset = abs(float(result.get("predicted_offset", 0.0) or 0.0))
        except Exception:
            predicted_offset = 0.0

        records.append({
            "time": float(t),
            "before": abs(float(before_error)),
            "after": abs(float(after_error)),
            "predicted_offset": predicted_offset,
            "status": status,
            "left_corr": left_corr,
            "right_corr": right_corr,
            "disagreement": disagreement,
        })

    if not records:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            [],
            [],
            False,
        )

    # IMPORTANT v6.8: determine whether the TARGET is already synchronized
    # from the actual measured offset, not from the error relative to the
    # candidate map. The latter can be tiny even when the source still has a
    # meaningful absolute timing displacement.
    baseline_actual_values = []
    for result in validation_results:
        actual = result.get("before_bidirectional_offset")
        if actual is None:
            actual = result.get("before_offset")
        if actual is None:
            continue
        try:
            baseline_actual_values.append(abs(float(actual)))
        except Exception:
            continue

    if baseline_actual_values:
        baseline_median = float(np.median(np.asarray(baseline_actual_values, dtype=np.float64)))
        baseline_p95 = float(np.percentile(np.asarray(baseline_actual_values, dtype=np.float64), 95))
    else:
        baseline_median = float(np.median(before_values))
        baseline_p95 = float(np.percentile(before_values, 95))
    already_synced = (
        baseline_median <= NO_REGRESSION_ALREADY_SYNCED_SEC
        and baseline_p95 <= 0.050
    )

    times = []
    weights = []
    reasons = []
    decisions = []

    for item in records:
        before = item["before"]
        after = item["after"]
        status = item["status"]

        clear_improvement = (
            after <= before - NO_REGRESSION_CLEAR_IMPROVEMENT_SEC
        )
        confirmed_map = status == "CONFIRMED_MAP"
        strong_two_sided = (
            item["left_corr"] >= MICRO_MIN_SIDE_CORRELATION
            and item["right_corr"] >= MICRO_MIN_SIDE_CORRELATION
            and item["disagreement"] <= MICRO_MAX_SIDE_DISAGREEMENT_SEC
        )
        confirmed_discrepancy = status == "CONFIRMED_DISCREPANCY"
        candidate_offset = abs(float(item.get("predicted_offset", 0.0) or 0.0))
        meaningful_change = candidate_offset >= NO_REGRESSION_MIN_MEANINGFUL_CORRECTION_SEC
        meaningful_benefit = (before - after) >= NO_REGRESSION_MIN_MEANINGFUL_IMPROVEMENT_SEC

        if confirmed_discrepancy:
            weight = 0.0
            reason = "подтверждённое расхождение"
        elif not confirmed_map:
            weight = 0.0
            reason = f"статус независимой проверки: {status}"
        elif not clear_improvement:
            weight = 0.0
            reason = "нет явного независимого улучшения"
        elif already_synced and not strong_two_sided:
            weight = 0.0
            reason = "уже синхронная дорожка без сильного двустороннего подтверждения"
        elif already_synced and not meaningful_change:
            weight = 0.0
            reason = (
                "слишком малая коррекция для уже синхронной дорожки "
                f"(< {NO_REGRESSION_MIN_MEANINGFUL_CORRECTION_SEC*1000.0:.1f} мс)"
            )
        elif already_synced and not meaningful_benefit:
            weight = 0.0
            reason = (
                "слишком малое улучшение для уже синхронной дорожки "
                f"(< {NO_REGRESSION_MIN_MEANINGFUL_IMPROVEMENT_SEC*1000.0:.1f} мс)"
            )
        else:
            weight = 1.0
            reason = "явное независимо подтверждённое улучшение"

        times.append(item["time"])
        weights.append(weight)
        reasons.append(reason)
        decisions.append({
            "target_time": item["time"],
            "before_error": before,
            "after_error": after,
            "delta": after - before,
            "status": status,
            "left_correlation": item["left_corr"],
            "right_correlation": item["right_corr"],
            "disagreement": item["disagreement"],
            "weight": weight,
            "candidate_offset": candidate_offset,
            "micro_correction_suppressed": bool(
                already_synced
                and (not meaningful_change or not meaningful_benefit)
            ),
            "reason": reason,
        })

    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        reasons,
        decisions,
        already_synced,
    )


def _accepted_runs(validation_time, validation_weight, min_points=SAFETY_MIN_ACCEPTED_RUN_POINTS):
    """Возвращает непрерывные серии accepted dense points."""
    accepted = np.asarray(validation_weight > 0.5, dtype=bool)
    runs = []
    if accepted.size == 0:
        return runs

    start = None
    for index, flag in enumerate(accepted):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            end = index - 1
            count = end - start + 1
            span = float(validation_time[end] - validation_time[start]) if count > 1 else 0.0
            if count >= min_points and span >= min(
                SAFETY_MIN_RUN_SPAN_SEC,
                float(validation_time[1] - validation_time[0]) * max(1, min_points - 1)
                if len(validation_time) > 1 else SAFETY_MIN_RUN_SPAN_SEC,
            ):
                runs.append((start, end, count, span))
            start = None

    if start is not None:
        end = len(accepted) - 1
        count = end - start + 1
        span = float(validation_time[end] - validation_time[start]) if count > 1 else 0.0
        if count >= min_points and span >= min(
            SAFETY_MIN_RUN_SPAN_SEC,
            float(validation_time[1] - validation_time[0]) * max(1, min_points - 1)
            if len(validation_time) > 1 else SAFETY_MIN_RUN_SPAN_SEC,
        ):
            runs.append((start, end, count, span))

    return runs


def _build_interval_gate(target_time, validation_time, validation_weight):
    """
    Строит непрерывную gate по accepted-run интервалам.

    Главное правило:
      rejected validation point = жёсткий барьер.

    Поэтому correction не может "переехать" через одиночный плохой участок
    только потому, что соседние точки хорошие.
    """
    gate = np.zeros_like(target_time, dtype=np.float64)
    runs = _accepted_runs(validation_time, validation_weight)

    if len(validation_time) == 0:
        return gate, runs

    for start, end, _count, _span in runs:
        first_time = float(validation_time[start])
        last_time = float(validation_time[end])

        # Границы устанавливаются в середины ближайших reject/accept intervals.
        if start > 0:
            left_boundary = (
                float(validation_time[start - 1]) + first_time
            ) / 2.0
        else:
            # The first verified interval has no preceding rejected point.
            # Returning from zero in one map step creates an artificial tempo
            # jump, even for a constant 265 ms delay. Taper within the first
            # validation window instead of jumping at its centre.
            left_boundary = max(float(target_time[0]), first_time - SAFETY_DENSE_VALIDATION_INTERVAL_SEC / 2)

        if end + 1 < len(validation_time):
            right_boundary = (
                last_time + float(validation_time[end + 1])
            ) / 2.0
        else:
            right_boundary = min(float(target_time[-1]), last_time + SAFETY_DENSE_VALIDATION_INTERVAL_SEC / 2)

        inside = (
            (target_time >= left_boundary)
            & (target_time <= right_boundary)
        )
        if not np.any(inside):
            continue

        t = target_time[inside]
        local_gate = np.ones_like(t, dtype=np.float64)

        if first_time > left_boundary:
            left_mask = t < first_time
            local_gate[left_mask] = (
                (t[left_mask] - left_boundary)
                / (first_time - left_boundary)
            )

        if right_boundary > last_time:
            right_mask = t > last_time
            local_gate[right_mask] = (
                (right_boundary - t[right_mask])
                / (right_boundary - last_time)
            )

        local_gate = np.clip(local_gate, 0.0, 1.0)
        gate[inside] = np.maximum(gate[inside], local_gate)

        # Внутри run, в самих accepted validation points, gate обязан быть 1.
        center_mask = (
            (target_time >= first_time)
            & (target_time <= last_time)
        )
        gate[center_mask] = 1.0

    return gate, runs


def _is_in_quarantine(target_time: float, quarantine_times, radius_sec: float) -> bool:
    """Возвращает True, если время попадает в локальную зону карантина."""
    if not quarantine_times:
        return False
    time_value = float(target_time)
    radius = float(radius_sec)
    return any(abs(time_value - float(center)) <= radius for center in quarantine_times)


def assess_safe_validation(
    validation_results,
    excluded_times=None,
    excluded_radius_sec=SAFETY_QUARANTINE_RADIUS_SEC,
):
    """
    Проверяет safe map против текущего TARGET.

    v6.7.2: локальные зоны карантина не считаются провалом всей карты.
    Внутри такой зоны коррекция принудительно возвращается к нулю, поэтому
    там сохраняется исходный TARGET. Эти точки остаются в отчёте как
    диагностические, но не уничтожают другие подтверждённые улучшения.
    """
    excluded_times = [float(value) for value in (excluded_times or [])]
    excluded_points = []
    worsened_points = []
    improved_points = []
    unchanged_points = []

    for result in validation_results:
        target_time = result.get("target_time")
        before = result.get("before_error")
        after = result.get("after_error")

        if target_time is None or before is None or after is None:
            continue

        target_time = float(target_time)
        before = abs(float(before))
        after = abs(float(after))
        delta = after - before

        record = {
            "target_time": target_time,
            "before_error": before,
            "after_error": after,
            "delta": delta,
            "bidir_status": result.get("bidir_status", "UNRESOLVED"),
        }

        if _is_in_quarantine(target_time, excluded_times, excluded_radius_sec):
            record["quarantined"] = True
            excluded_points.append(record)
            continue

        if delta > NO_REGRESSION_TOLERANCE_SEC:
            worsened_points.append(record)
        elif delta < -NO_REGRESSION_CLEAR_IMPROVEMENT_SEC:
            improved_points.append(record)
        else:
            unchanged_points.append(record)

    total_considered = len(worsened_points) + len(improved_points) + len(unchanged_points)
    total_examined = total_considered + len(excluded_points)
    passed = len(worsened_points) == 0

    if not excluded_points:
        if passed and improved_points:
            reason = (
                "SAFE MAP passes dense same-grid non-regression: no independent "
                "validation point was materially worsened."
            )
        elif passed:
            reason = (
                "SAFE MAP passes dense same-grid non-regression, but no material "
                "independent improvement was demonstrated."
            )
        else:
            reason = (
                "SAFE MAP would materially worsen at least one dense independent "
                "validation point."
            )
    else:
        if passed and improved_points:
            reason = (
                "SAFE MAP passes outside local quarantine zones: all non-quarantined "
                "independent validation points are non-regressive and material "
                "improvements remain."
            )
        elif passed:
            reason = (
                "SAFE MAP passes outside local quarantine zones, but no material "
                "independent improvement remains outside those zones."
            )
        else:
            reason = (
                "SAFE MAP still materially worsens at least one non-quarantined "
                "dense independent validation point."
            )

    return {
        "passed": bool(passed),
        "reason": reason,
        "worsened_count": len(worsened_points),
        "improved_count": len(improved_points),
        "unchanged_count": len(unchanged_points),
        "excluded_count": len(excluded_points),
        "total_points": total_examined,
        "considered_points": total_considered,
        "worsened_points": worsened_points,
        "improved_points": improved_points,
        "unchanged_points": unchanged_points,
        "excluded_points": excluded_points,
        "max_worsening_sec": max(
            (item["delta"] for item in worsened_points),
            default=0.0,
        ),
        "max_improvement_sec": max(
            (-item["delta"] for item in improved_points),
            default=0.0,
        ),
    }

def build_non_regressive_sync_map(
    candidate_map,
    validation_results,
    target_duration: float,
    progress: Progress,
    forced_reject_times=None,
    local_radius_sec=None,
    quarantine_times=None,
    quarantine_radius_sec=SAFETY_QUARANTINE_RADIUS_SEC,
):
    """
    Строит SAFE MAP из одной и той же dense validation-сетки.

    В v6.6 correction создавалась как сумма локальных треугольников вокруг
    отдельных accepted points. Это позволяло одиночной ошибке сформировать
    большую локальную "иглу", которую sparse validation просто не замечала.

    v6.7:
      1. accepted points должны образовывать непрерывный run;
      2. run должен иметь минимум 3 dense точки;
      3. каждый rejected point является барьером;
      4. correction внутри run берётся из полного candidate map;
      5. на границах run correction плавно возвращается к нулю.
    """
    progress.stage("ЗАЩИТА ОТ РЕГРЕССИЙ: DENSE GRID")

    target_time = np.asarray(candidate_map["target_time"], dtype=np.float64)
    candidate_offset = np.asarray(candidate_map["offset"], dtype=np.float64)

    (
        validation_time,
        validation_weight,
        reasons,
        decisions,
        already_synced,
    ) = _validation_weights(validation_results)

    if len(validation_time) < 2:
        progress.done(100.0, "Недостаточно dense validation-точек")
        raise RuntimeError(
            "Недостаточно независимых dense validation-точек для безопасной защиты карты."
        )

    order = np.argsort(validation_time)
    validation_time = validation_time[order]
    validation_weight = validation_weight[order]
    reasons = [reasons[i] for i in order]
    decisions = [decisions[i] for i in order]

    unique_time, unique_indices = np.unique(
        validation_time,
        return_index=True,
    )
    validation_time = unique_time
    validation_weight = validation_weight[unique_indices]
    reasons = [reasons[i] for i in unique_indices]
    decisions = [decisions[i] for i in unique_indices]

    # v6.7.2: merge explicit rollback points and persistent local quarantine
    # centers. A quarantine removes candidate correction from a finite window
    # rather than invalidating the rest of the film.
    quarantine_centers = []
    for value in (quarantine_times or []):
        quarantine_centers.append(float(value))
    for value in (forced_reject_times or []):
        quarantine_centers.append(float(value))

    # Deduplicate nearby centers so overlapping zones remain a single local
    # exclusion region in the report.
    quarantine_centers = sorted(quarantine_centers)
    merged_quarantine_centers = []
    merge_distance = max(float(quarantine_radius_sec), SAFETY_DENSE_VALIDATION_INTERVAL_SEC)
    for value in quarantine_centers:
        if not merged_quarantine_centers or abs(value - merged_quarantine_centers[-1]) > merge_distance:
            merged_quarantine_centers.append(value)
        else:
            merged_quarantine_centers[-1] = (merged_quarantine_centers[-1] + value) / 2.0
    quarantine_centers = merged_quarantine_centers

    quarantined_count = 0
    for index, center in enumerate(validation_time):
        if _is_in_quarantine(float(center), quarantine_centers, float(quarantine_radius_sec)):
            if validation_weight[index] > 0.5:
                quarantined_count += 1
            validation_weight[index] = 0.0
            reasons[index] = "local quarantine"
            if index < len(decisions):
                decisions[index]["weight"] = 0.0
                decisions[index]["reason"] = reasons[index]

    gate, runs = _build_interval_gate(
        target_time,
        validation_time,
        validation_weight,
    )

    candidate_at_validation = np.interp(
        validation_time,
        target_time,
        candidate_offset,
    )

    safe_offset = candidate_offset * gate

    # Explicit point constraints: accepted -> candidate, rejected -> zero.
    for index, center in enumerate(validation_time):
        nearest = int(np.argmin(np.abs(target_time - float(center))))
        if validation_weight[index] > 0.5:
            safe_offset[nearest] = candidate_at_validation[index]
        else:
            safe_offset[nearest] = 0.0

    # Quarantine is a hard local barrier. The whole finite analysis neighborhood
    # is kept on the original TARGET timing so the excluded point cannot be
    # influenced by nearby candidate deformation.
    if quarantine_centers:
        for center in quarantine_centers:
            barrier = np.abs(target_time - float(center)) <= float(quarantine_radius_sec)
            safe_offset[barrier] = 0.0

    reference_time = target_time + safe_offset
    confidence = np.asarray(
        candidate_map.get("confidence", np.ones_like(target_time)),
        dtype=np.float64,
    )

    result = {
        "target_time": target_time,
        "offset": safe_offset,
        "reference_time": reference_time,
        "confidence": confidence,
        "anchor_time": np.asarray(candidate_map.get("anchor_time", []), dtype=np.float64),
        "anchor_offset": np.asarray(candidate_map.get("anchor_offset", []), dtype=np.float64),
    }

    max_abs_candidate = (
        float(np.max(np.abs(candidate_offset)))
        if candidate_offset.size else 0.0
    )
    max_abs_safe = (
        float(np.max(np.abs(safe_offset)))
        if safe_offset.size else 0.0
    )
    accepted_count = int(np.sum(validation_weight > 0.5))
    rejected_count = int(np.sum(validation_weight <= 0.5))
    total_count = int(len(validation_weight))

    before_values = [
        abs(float(item.get("before_error")))
        for item in validation_results
        if item.get("before_error") is not None
    ]
    baseline_median = float(np.median(before_values)) if before_values else None
    baseline_p95 = float(np.percentile(before_values, 95)) if before_values else None

    no_change = max_abs_safe <= NO_REGRESSION_ZERO_MAP_SEC
    if no_change:
        mode = "NO_CHANGE"
    elif quarantine_centers:
        mode = "LOCAL_QUARANTINE_SAFE_MAP"
    elif accepted_count == total_count:
        mode = "FULL_CANDIDATE"
    elif accepted_count == 0:
        mode = "PRESERVE_TARGET"
    else:
        mode = "MIXED_SAFE_MAP_DENSE"

    progress.update(
        85.0,
        (
            f"Runs={len(runs)} | accepted={accepted_count}/{total_count} | "
            f"max candidate={max_abs_candidate:.6f}s | max safe={max_abs_safe:.6f}s"
        ),
    )

    report = {
        "mode": mode,
        "already_synced": bool(already_synced),
        "baseline_median_error_sec": baseline_median,
        "baseline_p95_error_sec": baseline_p95,
        "candidate_max_abs_offset_sec": max_abs_candidate,
        "safe_max_abs_offset_sec": max_abs_safe,
        "accepted_validation_points": accepted_count,
        "rejected_validation_points": rejected_count,
        "total_validation_points": total_count,
        "quarantine_centers": [float(value) for value in quarantine_centers],
        "quarantine_radius_sec": float(quarantine_radius_sec),
        "quarantined_validation_points": int(quarantined_count),
        "quarantine_fraction": (
            float(quarantined_count) / float(total_count) if total_count else 0.0
        ),
        "max_quarantine_fraction": float(SAFETY_MAX_QUARANTINE_FRACTION),
        "validation_interval_sec": float(SAFETY_DENSE_VALIDATION_INTERVAL_SEC),
        "same_grid_candidate_and_safe": True,
        "min_accepted_run_points": int(SAFETY_MIN_ACCEPTED_RUN_POINTS),
        "min_accepted_run_span_sec": float(SAFETY_MIN_RUN_SPAN_SEC),
        "accepted_runs": [
            {
                "start_index": int(start),
                "end_index": int(end),
                "points": int(count),
                "span_sec": float(span),
                "start_time_sec": float(validation_time[start]),
                "end_time_sec": float(validation_time[end]),
            }
            for start, end, count, span in runs
        ],
        "tolerance_sec": float(NO_REGRESSION_TOLERANCE_SEC),
        "clear_improvement_sec": float(NO_REGRESSION_CLEAR_IMPROVEMENT_SEC),
        "already_synced_threshold_sec": float(NO_REGRESSION_ALREADY_SYNCED_SEC),
        "zero_map_threshold_sec": float(NO_REGRESSION_ZERO_MAP_SEC),
        "min_meaningful_correction_sec": float(NO_REGRESSION_MIN_MEANINGFUL_CORRECTION_SEC),
        "min_meaningful_improvement_sec": float(NO_REGRESSION_MIN_MEANINGFUL_IMPROVEMENT_SEC),
        "forced_reject_times": [float(value) for value in (forced_reject_times or [])],
        "validation_decisions": decisions,
        "gate_max": float(np.max(gate)) if gate.size else 0.0,
        "gate_nonzero_seconds_approx": float(np.sum(gate > 0.0)),
    }

    progress.done(100.0, f"Защита от регрессий завершена: {mode}")
    return result, report


def save_non_regression_report(
    output_dir: Path,
    candidate_metrics: dict,
    safe_metrics: dict | None,
    guard_report: dict,
    safe_assessment: dict | None,
):
    path = output_dir / NON_REGRESSION_REPORT

    with path.open("w", encoding="utf-8") as file:
        file.write("NON-REGRESSION GUARD REPORT\n")
        file.write("=" * 78 + "\n\n")
        file.write(
            "The guard compares candidate and SAFE MAP on the SAME dense independent "
            "validation grid. Rejected points are hard barriers; correction cannot "
            "interpolate through them.\n\n"
        )
        file.write(f"MODE: {guard_report['mode']}\n")
        file.write(f"ALREADY SYNCHRONIZED: {guard_report['already_synced']}\n")
        file.write(
            f"BASELINE MEDIAN ERROR: {guard_report['baseline_median_error_sec']} s\n"
        )
        file.write(
            f"BASELINE P95 ERROR: {guard_report['baseline_p95_error_sec']} s\n"
        )
        file.write(
            f"CANDIDATE MAX ABS OFFSET: {guard_report['candidate_max_abs_offset_sec']:.6f} s\n"
        )
        file.write(
            f"SAFE MAP MAX ABS OFFSET: {guard_report['safe_max_abs_offset_sec']:.6f} s\n"
        )
        file.write(
            f"DENSE VALIDATION INTERVAL: {guard_report['validation_interval_sec']:.3f} s\n"
        )
        file.write(
            f"SAME GRID FOR CANDIDATE AND SAFE: {guard_report['same_grid_candidate_and_safe']}\n"
        )
        file.write(
            f"ACCEPTED VALIDATION POINTS: {guard_report['accepted_validation_points']}\n"
        )
        file.write(
            f"REJECTED VALIDATION POINTS: {guard_report['rejected_validation_points']}\n"
        )
        file.write(
            f"TOTAL VALIDATION POINTS: {guard_report['total_validation_points']}\n"
        )
        file.write(
            f"ACCEPTED CONTIGUOUS RUNS: {len(guard_report.get('accepted_runs', []))}\n"
        )
        file.write(
            f"MINIMUM ACCEPTED RUN: {guard_report['min_accepted_run_points']} points / "
            f"{guard_report['min_accepted_run_span_sec']:.1f} s\n"
        )
        file.write(f"REPAIR PASSES: {guard_report.get('repair_passes', 0)}\n")
        file.write(
            f"FORCED LOCAL ROLLBACKS: {len(guard_report.get('forced_reject_times', []))}\n"
        )
        file.write(
            f"LOCAL QUARANTINE CENTERS: {len(guard_report.get('quarantine_centers', []))}\n"
        )
        file.write(
            f"QUARANTINE RADIUS: {guard_report.get('quarantine_radius_sec', SAFETY_QUARANTINE_RADIUS_SEC):.3f} s\n"
        )
        file.write(
            f"QUARANTINED VALIDATION POINTS: {guard_report.get('quarantined_validation_points', 0)}\n"
        )
        file.write(
            f"QUARANTINE FRACTION: {guard_report.get('quarantine_fraction', 0.0) * 100.0:.2f}%\n"
        )
        centers = guard_report.get('quarantine_centers', [])
        if centers:
            file.write("QUARANTINE CENTERS (s):\n")
            file.write("  " + ", ".join(f"{float(value):.3f}" for value in centers) + "\n")

        file.write("\nTHRESHOLDS:\n")
        file.write(f"  Non-regression tolerance: {NO_REGRESSION_TOLERANCE_SEC:.6f} s\n")
        file.write(f"  Clear improvement: {NO_REGRESSION_CLEAR_IMPROVEMENT_SEC:.6f} s\n")
        file.write(
            f"  Already synchronized median threshold: {NO_REGRESSION_ALREADY_SYNCED_SEC:.6f} s\n"
        )
        file.write(f"  Zero-map threshold: {NO_REGRESSION_ZERO_MAP_SEC:.6f} s\n")
        file.write(f"  Minimum meaningful correction for already-synchronized TARGET: {NO_REGRESSION_MIN_MEANINGFUL_CORRECTION_SEC:.6f} s\n")
        file.write(f"  Minimum meaningful improvement for already-synchronized TARGET: {NO_REGRESSION_MIN_MEANINGFUL_IMPROVEMENT_SEC:.6f} s\n")
        file.write("  Micro-correction rule: tiny corrections are suppressed on an already synchronized TARGET.\n")
        file.write(f"  Dense interval: {SAFETY_DENSE_VALIDATION_INTERVAL_SEC:.3f} s\n")
        file.write(f"  Minimum accepted run points: {SAFETY_MIN_ACCEPTED_RUN_POINTS}\n")

        file.write("\nCANDIDATE VALIDATION:\n")
        file.write(f"  Mean before: {candidate_metrics.get('before_mean_error')} s\n")
        file.write(f"  Mean candidate after: {candidate_metrics.get('after_mean_error')} s\n")
        file.write(f"  Median before: {candidate_metrics.get('before_median_error')} s\n")
        file.write(f"  Median candidate after: {candidate_metrics.get('after_median_error')} s\n")

        if safe_metrics:
            file.write("\nSAFE MAP VALIDATION:\n")
            file.write(f"  Mean before: {safe_metrics.get('before_mean_error')} s\n")
            file.write(f"  Mean safe after: {safe_metrics.get('after_mean_error')} s\n")
            file.write(f"  Median before: {safe_metrics.get('before_median_error')} s\n")
            file.write(f"  Median safe after: {safe_metrics.get('after_median_error')} s\n")

        if safe_assessment:
            file.write("\nFINAL ASSESSMENT:\n")
            file.write(f"  PASSED: {safe_assessment.get('passed')}\n")
            file.write(f"  REASON: {safe_assessment.get('reason')}\n")
            file.write(
                f"  WORSENED POINTS OUTSIDE QUARANTINE: {safe_assessment.get('worsened_count', 0)} / "
                f"{safe_assessment.get('considered_points', safe_assessment.get('total_points', 0))}\n"
            )
            file.write(
                f"  QUARANTINED POINTS EXCLUDED FROM DECISION: {safe_assessment.get('excluded_count', 0)}\n"
            )

        file.write("\nIMPORTANT:\n")
        file.write(
            "Candidate corrections are allowed into the final map only when dense "
            "independent validation demonstrates a clear improvement and enough "
            "neighboring validation points support the same correction. Local "
            "quarantine may exclude a small problematic region without cancelling "
            "unrelated confirmed improvements.\n"
        )

    return path


def save_candidate_validation_snapshot(output_dir: Path, validation_results, validation_metrics):
    path = output_dir / CANDIDATE_VALIDATION_JSON
    data = {"validation_metrics": validation_metrics, "results": validation_results}
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=float)
    return path


def save_safe_validation_snapshot(output_dir: Path, validation_results, validation_metrics):
    path = output_dir / SAFE_VALIDATION_JSON
    data = {"validation_metrics": validation_metrics, "results": validation_results}
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=float)
    return path


def save_dense_safety_validation_csv(
    output_dir: Path,
    candidate_results,
    safe_results,
    quarantine_times=None,
    quarantine_radius_sec=SAFETY_QUARANTINE_RADIUS_SEC,
):
    """Пишет точка-за-точкой ДО/кандидат/безопасную карту на одной сетке."""
    path = output_dir / DENSE_SAFETY_VALIDATION_CSV
    safe_by_time = {
        float(item["target_time"]): item
        for item in safe_results
    }

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow([
            "target_time_sec",
            "candidate_offset_sec",
            "candidate_after_error_sec",
            "candidate_bidir_status",
            "safe_offset_sec",
            "safe_after_error_sec",
            "safe_bidir_status",
            "before_error_sec",
            "candidate_delta_sec",
            "safe_delta_sec",
            "quarantined",
        ])

        for item in candidate_results:
            t = float(item["target_time"])
            safe = safe_by_time.get(t)
            if safe is None:
                continue
            before = item.get("before_error")
            cand_after = item.get("after_error")
            safe_after = safe.get("after_error")
            writer.writerow([
                f"{t:.3f}",
                f"{float(item.get('predicted_offset', 0.0)):.6f}",
                "" if cand_after is None else f"{float(cand_after):.6f}",
                item.get("bidir_status", ""),
                f"{float(safe.get('predicted_offset', 0.0)):.6f}",
                "" if safe_after is None else f"{float(safe_after):.6f}",
                safe.get("bidir_status", ""),
                "" if before is None else f"{float(before):.6f}",
                "" if before is None or cand_after is None else f"{float(cand_after)-float(before):.6f}",
                "" if before is None or safe_after is None else f"{float(safe_after)-float(before):.6f}",
                "YES" if _is_in_quarantine(t, quarantine_times or [], quarantine_radius_sec) else "NO",
            ])

    return path


def save_dense_safety_snapshot(output_dir: Path, filename: str, validation_results, validation_metrics, guard_report=None):
    path = output_dir / filename
    data = {
        "validation_metrics": validation_metrics,
        "validation_interval_sec": SAFETY_DENSE_VALIDATION_INTERVAL_SEC,
        "results": validation_results,
    }
    if guard_report is not None:
        data["guard_report"] = guard_report
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=float)
    return path
