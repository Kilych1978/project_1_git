# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC - MODULAR COMPONENT
# MODULE: render_guard.py
# ============================================================
# v6.9.1 RENDER-AWARE NON-REGRESSION GUARD
#
# The mathematical sync map is not the final signal. A real ffmpeg render can
# introduce small local timing changes when a segment is processed by atempo,
# trimmed, and concatenated. Therefore v6.8 validates the ACTUAL RENDERED
# waveform against the ORIGINAL TARGET on the SAME independent dense grid.
#
# If a rendered point is materially worse than the original TARGET, that local
# region is added to the quarantine and the map is rendered again. Unrelated
# improvements remain eligible. If the problem cannot be isolated after the
# configured number of passes, the original TARGET is preserved.
# ============================================================

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from config import *
from common import *
from applier import build_apply_map
from renderer import (
    render_target_with_sync_map,
    copy_target_without_reencode,
    verify_rendered_sync,
)
from safety import (
    build_non_regressive_sync_map,
    build_zero_sync_map,
    assess_safe_validation,
    SAFETY_DENSE_VALIDATION_INTERVAL_SEC,
    SAFETY_QUARANTINE_RADIUS_SEC,
    NO_REGRESSION_TOLERANCE_SEC,
    NO_REGRESSION_ZERO_MAP_SEC,
)


RENDER_GUARD_MAX_PASSES = 3
RENDER_GUARD_DENSE_VALIDATION_INTERVAL_SEC = SAFETY_DENSE_VALIDATION_INTERVAL_SEC
RENDER_GUARD_TOLERANCE_SEC = NO_REGRESSION_TOLERANCE_SEC
RENDER_GUARD_CORRELATION_TOLERANCE = 0.050
RENDER_GUARD_MICRO_TOLERANCE_SEC = 0.0015
RENDER_GUARD_MICRO_CORRELATION_TOLERANCE = 0.050
RENDER_GUARD_QUARANTINE_RADIUS_SEC = max(
    SAFETY_QUARANTINE_RADIUS_SEC,
    30.0,
)

RENDER_GUARD_REPORT = "RENDER_AWARE_NON_REGRESSION_REPORT.txt"
RENDER_GUARD_CSV = "RENDER_AWARE_NON_REGRESSION.csv"
RENDER_GUARD_JSON = "RENDER_AWARE_NON_REGRESSION.json"


def _baseline_error(record: dict) -> float | None:
    """Фактическая ошибка исходного TARGET относительно нулевого смещения."""
    for key in ("before_bidirectional_offset", "before_offset"):
        value = record.get(key)
        if value is None:
            continue
        try:
            return abs(float(value))
        except Exception:
            continue
    return None


def _baseline_correlation(record: dict) -> float | None:
    for key in ("before_bidirectional_correlation", "before_correlation"):
        value = record.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _rendered_error(record: dict) -> float | None:
    """Фактическая ошибка готового файла относительно нулевого смещения."""
    value = record.get("bidir_consensus_offset")
    if value is None:
        value = record.get("after_offset")
    if value is None:
        value = record.get("before_offset")
    if value is None:
        return None
    try:
        return abs(float(value))
    except Exception:
        return None


def _rendered_correlation(record: dict) -> float | None:
    for key in ("bidir_consensus_correlation", "after_correlation", "before_correlation"):
        value = record.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _same_time_records(baseline_results: list[dict], rendered_results: list[dict]):
    baseline_by_time = {}
    for item in baseline_results:
        try:
            baseline_by_time[round(float(item["target_time"]), 6)] = item
        except Exception:
            continue

    pairs = []
    for item in rendered_results:
        try:
            time_sec = round(float(item["target_time"]), 6)
        except Exception:
            continue
        baseline = baseline_by_time.get(time_sec)
        if baseline is not None:
            pairs.append((baseline, item))
    return pairs


def compare_rendered_against_original(
    baseline_results: list[dict],
    rendered_results: list[dict],
):
    """
    Сравнивает реальный результат рендера с ИСХОДНЫМ TARGET.

    Это не проверка математической карты. Это именно проверка того, что реально
    получилось после ffmpeg. Порог ошибки - 1.5 ms. Дополнительно отслеживается
    заметное падение корреляции, если одновременно нет существенного выигрыша
    по абсолютной ошибке.
    """
    rows = []
    regressions = []
    improvements = []

    for baseline, rendered in _same_time_records(baseline_results, rendered_results):
        time_sec = float(rendered["target_time"])
        baseline_error = _baseline_error(baseline)
        rendered_error = _rendered_error(rendered)
        baseline_corr = _baseline_correlation(baseline)
        rendered_corr = _rendered_correlation(rendered)

        if baseline_error is None or rendered_error is None:
            continue

        delta = rendered_error - baseline_error
        improvement = baseline_error - rendered_error

        error_regression = delta > RENDER_GUARD_TOLERANCE_SEC
        correlation_regression = False
        if (
            baseline_corr is not None
            and rendered_corr is not None
            and baseline_corr >= 0.45
            and rendered_corr < baseline_corr - RENDER_GUARD_CORRELATION_TOLERANCE
            and improvement < 0.005
        ):
            correlation_regression = True

        regression = bool(error_regression or correlation_regression)
        row = {
            "target_time_sec": time_sec,
            "baseline_error_sec": baseline_error,
            "rendered_error_sec": rendered_error,
            "error_delta_sec": delta,
            "baseline_correlation": baseline_corr,
            "rendered_correlation": rendered_corr,
            "correlation_delta": (
                None
                if baseline_corr is None or rendered_corr is None
                else rendered_corr - baseline_corr
            ),
            "error_regression": error_regression,
            "correlation_regression": correlation_regression,
            "regression": regression,
        }
        rows.append(row)

        if regression:
            regressions.append(row)
        elif improvement > RENDER_GUARD_TOLERANCE_SEC:
            improvements.append(row)

    considered = len(rows)
    return {
        "rows": rows,
        "regressions": regressions,
        "improvements": improvements,
        "considered_points": considered,
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "passed": len(regressions) == 0,
    }


def compare_micro_rendered_against_original(
    analysis,
    verification: dict,
    micro_records: list[dict] | None,
):
    """
    Проверяет фактически отрендерированный сигнал на микроскопическом масштабе.

    Для каждой принятой микрокоррекции сравнивается:
      1) исходный TARGET относительно уже найденного offset;
      2) реально отрендерированный файл относительно нулевого offset.

    Разрешение поиска около 1 мс. Это отдельная проверка от обычной 10-мс
    огибающей и именно она отвечает за локальные ошибки порядка 3–10 мс.
    """
    if not micro_records:
        return {
            "rows": [],
            "regressions": [],
            "improvements": [],
            "considered_points": 0,
            "regression_count": 0,
            "improvement_count": 0,
            "passed": True,
        }

    rendered_pcm = verification.get("rendered_first_channel_pcm")
    if rendered_pcm is None:
        return {
            "rows": [],
            "regressions": [],
            "improvements": [],
            "considered_points": 0,
            "regression_count": 0,
            "improvement_count": 0,
            "passed": False,
            "error": "У готового файла отсутствует временный PCM первого канала для микропроверки.",
        }

    from micro_refiner import micro_search_at_time

    reference_pcm = Path(analysis["reference_pcm"])
    target_pcm = Path(analysis["target_pcm"])
    rendered_pcm = Path(rendered_pcm)

    rows = []
    regressions = []
    improvements = []

    for item in micro_records:
        if item.get("status") != "ACCEPTED_MICRO":
            continue

        target_time = float(item["target_time_sec"])
        predicted_offset = float(item["predicted_offset_sec"])

        try:
            # Render-aware comparison is against the UNMODIFIED TARGET at
            # zero additional offset. The previous implementation
            # compared the render against the proposed micro-correction, so
            # a quarantined point could never become "good enough" on a later
            # pass. The guard now asks the correct question: did the rendered
            # file improve or worsen relative to the original signal itself?
            baseline = micro_search_at_time(
                reference_pcm,
                target_pcm,
                target_time,
                0.0,
            )
            rendered = micro_search_at_time(
                reference_pcm,
                rendered_pcm,
                target_time,
                0.0,
            )
        except Exception:
            baseline = None
            rendered = None

        if baseline is None or rendered is None:
            continue

        baseline_error = abs(float(baseline["refined_offset_sec"]))
        rendered_error = abs(float(rendered["refined_offset_sec"]))
        delta = rendered_error - baseline_error
        improvement = baseline_error - rendered_error

        correlation_delta = float(
            rendered["best_score"] - baseline["best_score"]
        )
        correlation_regression = bool(
            correlation_delta < -RENDER_GUARD_MICRO_CORRELATION_TOLERANCE
            and improvement < RENDER_GUARD_MICRO_TOLERANCE_SEC
        )
        error_regression = bool(
            delta > RENDER_GUARD_MICRO_TOLERANCE_SEC
        )
        regression = bool(error_regression or correlation_regression)

        row = {
            "target_time_sec": target_time,
            "baseline_micro_error_sec": baseline_error,
            "rendered_micro_error_sec": rendered_error,
            "micro_error_delta_sec": delta,
            "baseline_micro_score": float(baseline["best_score"]),
            "rendered_micro_score": float(rendered["best_score"]),
            "micro_score_delta": correlation_delta,
            "error_regression": error_regression,
            "correlation_regression": correlation_regression,
            "regression": regression,
        }
        rows.append(row)

        if regression:
            regressions.append(row)
        elif improvement > RENDER_GUARD_MICRO_TOLERANCE_SEC:
            improvements.append(row)

    return {
        "rows": rows,
        "regressions": regressions,
        "improvements": improvements,
        "considered_points": len(rows),
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "passed": len(regressions) == 0,
    }


def save_micro_render_guard_csv(output_dir: Path, comparison: dict, pass_index: int):
    if not comparison.get("rows"):
        return
    path = output_dir / "RENDER_AWARE_MICRO_NON_REGRESSION.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow([
            "pass",
            "target_time_sec",
            "baseline_micro_error_sec",
            "rendered_micro_error_sec",
            "micro_error_delta_sec",
            "baseline_micro_score",
            "rendered_micro_score",
            "micro_score_delta",
            "error_regression",
            "correlation_regression",
            "regression",
        ])
        for row in comparison.get("rows", []):
            writer.writerow([
                pass_index,
                f"{float(row['target_time_sec']):.6f}",
                f"{float(row['baseline_micro_error_sec']):.6f}",
                f"{float(row['rendered_micro_error_sec']):.6f}",
                f"{float(row['micro_error_delta_sec']):+.6f}",
                f"{float(row['baseline_micro_score']):.6f}",
                f"{float(row['rendered_micro_score']):.6f}",
                f"{float(row['micro_score_delta']):+.6f}",
                "ДА" if row["error_regression"] else "НЕТ",
                "ДА" if row["correlation_regression"] else "НЕТ",
                "ДА" if row["regression"] else "НЕТ",
            ])


def _merge_centers(times: list[float], radius_sec: float) -> list[float]:
    values = sorted(float(t) for t in times)
    merged: list[float] = []
    for value in values:
        if not merged or abs(value - merged[-1]) > float(radius_sec):
            merged.append(value)
        else:
            merged[-1] = (merged[-1] + value) / 2.0
    return merged


def save_render_guard_csv(output_dir: Path, comparison: dict, pass_index: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / RENDER_GUARD_CSV
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow([
            "pass",
            "target_time_sec",
            "baseline_error_sec",
            "rendered_error_sec",
            "error_delta_sec",
            "baseline_correlation",
            "rendered_correlation",
            "correlation_delta",
            "error_regression",
            "correlation_regression",
            "regression",
        ])
        for row in comparison["rows"]:
            writer.writerow([
                pass_index,
                f"{row['target_time_sec']:.3f}",
                f"{row['baseline_error_sec']:.6f}",
                f"{row['rendered_error_sec']:.6f}",
                f"{row['error_delta_sec']:+.6f}",
                "" if row["baseline_correlation"] is None else f"{row['baseline_correlation']:.6f}",
                "" if row["rendered_correlation"] is None else f"{row['rendered_correlation']:.6f}",
                "" if row["correlation_delta"] is None else f"{row['correlation_delta']:+.6f}",
                "ДА" if row["error_regression"] else "НЕТ",
                "ДА" if row["correlation_regression"] else "НЕТ",
                "ДА" if row["regression"] else "НЕТ",
            ])
    return path


def _write_pass_summary(pass_dir: Path, pass_index: int, comparison: dict, quarantine_times: list[float]):
    pass_dir.mkdir(parents=True, exist_ok=True)
    path = pass_dir / "RENDER_AWARE_PASS.txt"
    with path.open("w", encoding="utf-8") as file:
        file.write("ПРОВЕРКА РЕАЛЬНОГО РЕНДЕРА НА УХУДШЕНИЕ\n")
        file.write("=" * 78 + "\n\n")
        file.write(f"ПРОХОД: {pass_index}\n")
        file.write(f"ПРОВЕРЕНО ТОЧЕК: {comparison['considered_points']}\n")
        file.write(f"УЛУЧШЕНИЙ: {comparison['improvement_count']}\n")
        file.write(f"УХУДШЕНИЙ: {comparison['regression_count']}\n")
        file.write(f"ЦЕНТРОВ КАРАНТИНА: {len(quarantine_times)}\n")
        if quarantine_times:
            file.write("ЦЕНТРЫ КАРАНТИНА (с):\n")
            file.write("  " + ", ".join(f"{x:.3f}" for x in quarantine_times) + "\n")
        file.write("\n")
        if comparison["regressions"]:
            file.write("НАЙДЕННЫЕ УХУДШЕНИЯ:\n")
            for row in comparison["regressions"]:
                file.write(
                    f"  {row['target_time_sec']:.3f} с: "
                    f"{row['baseline_error_sec']*1000.0:.3f} мс -> "
                    f"{row['rendered_error_sec']*1000.0:.3f} мс, "
                    f"изменение {row['error_delta_sec']*1000.0:+.3f} мс\n"
                )
        else:
            file.write("Ухудшений не обнаружено.\n")
    return path


def _discard_rejected_files(result,guard_root,source):
    """Only this run's generated rejected WAV/PCM files may be removed."""
    removed=[]
    guard_root=guard_root.resolve()
    source=source.resolve()
    for key in ('output_path','verified_first_channel_pcm'):
        if not result.get(key):continue
        path=Path(result[key])
        resolved=path.resolve()
        if resolved!=source and resolved.is_relative_to(guard_root) and path.is_file():
            path.unlink()
            removed.append(str(path))
    return removed


def _publish_guard_output(result,output_dir,source):
    """Keep the selected audio at the result root, with no overwriting."""
    path=Path(result['output_path'])
    guard_root=(output_dir/'_render_guard').resolve()
    if path.resolve()==source.resolve() or not path.resolve().is_relative_to(guard_root):
        return
    destination=output_dir/path.name
    if destination.exists():
        raise RuntimeError('Путь итогового аудио уже занят; перезапись отменена: '+str(destination))
    path.rename(destination)
    result['output_path']=destination


def run_render_aware_safety_loop(
    selected,
    analysis,
    candidate_map,
    initial_safe_map,
    dense_candidate_validation_results,
    initial_quarantine_times,
    output_dir: Path,
    codec_plan: dict,
    progress: Progress,
    micro_records: list[dict] | None = None,
):
    """
    Итеративно рендерит и проверяет реальный сигнал.

    При каждом обнаруженном ухудшении строится новая карта с локальным
    карантином и выполняется новый полный рендер исходного TARGET.
    """
    quarantine_times = _merge_centers(
        list(initial_quarantine_times or []),
        RENDER_GUARD_QUARANTINE_RADIUS_SEC,
    )
    current_safe_map = initial_safe_map
    pass_records = []
    final_render = None
    final_comparison = None
    final_safe_map = current_safe_map
    selected_safe_result = False

    guard_root = output_dir / "_render_guard"
    guard_root.mkdir(parents=True, exist_ok=True)

    for pass_index in range(1, RENDER_GUARD_MAX_PASSES + 1):
        context = f"РЕНДЕР: ПРОХОД {pass_index}/{RENDER_GUARD_MAX_PASSES}"
        print()
        print("=" * 78)
        print(context)
        print("Проверяется вся дорожка; после отказов отменяются локальные поправки.")
        print(f"Участков карантина перед проходом: {len(quarantine_times)}")
        print("=" * 78)
        pass_dir = guard_root / f"pass_{pass_index}"
        pass_dir.mkdir(parents=True, exist_ok=True)

        # Never modify the source TARGET; every pass starts from the original.
        current_safe_map, pre_render_report = build_non_regressive_sync_map(
            candidate_map,
            dense_candidate_validation_results,
            selected.target.duration,
            Progress(context),
            quarantine_times=quarantine_times,
            quarantine_radius_sec=RENDER_GUARD_QUARANTINE_RADIUS_SEC,
        )

        max_abs = (
            float(np.max(np.abs(current_safe_map["offset"])))
            if len(current_safe_map["offset"])
            else 0.0
        )

        progress.update(
            10.0 + 20.0 * (pass_index - 1),
            f"Проверка реального рендера: проход {pass_index}/{RENDER_GUARD_MAX_PASSES}",
        )

        if max_abs <= NO_REGRESSION_ZERO_MAP_SEC:
            render_result = copy_target_without_reencode(
                selected.target,
                codec_plan,
                pass_dir,
            )
        else:
            apply_map = build_apply_map(
                current_safe_map,
                selected.target.duration,
                Progress(context),
            )
            render_result = render_target_with_sync_map(
                selected,
                apply_map,
                pass_dir,
                Progress(context),
                codec_plan=codec_plan,
            )

        rendered_path = render_result["output_path"]
        verification = verify_rendered_sync(
            selected,
            analysis,
            rendered_path,
            pass_dir,
            validation_interval_sec=RENDER_GUARD_DENSE_VALIDATION_INTERVAL_SEC,
            progress=Progress(context),
        )
        render_result["verified_first_channel_pcm"] = verification["rendered_first_channel_pcm"]

        comparison = compare_rendered_against_original(
            dense_candidate_validation_results,
            verification["validation_points"],
        )

        micro_comparison = compare_micro_rendered_against_original(
            analysis,
            verification,
            micro_records,
        )
        comparison["micro"] = micro_comparison

        # Micro regressions are promoted into the same quarantine mechanism as
        # ordinary dense regressions, so a bad local interval is isolated rather
        # than cancelling unrelated improvements.
        for micro_row in micro_comparison.get("regressions", []):
            promoted = {
                "target_time_sec": micro_row["target_time_sec"],
                "baseline_error_sec": micro_row["baseline_micro_error_sec"],
                "rendered_error_sec": micro_row["rendered_micro_error_sec"],
                "error_delta_sec": micro_row["micro_error_delta_sec"],
                "baseline_correlation": micro_row["baseline_micro_score"],
                "rendered_correlation": micro_row["rendered_micro_score"],
                "correlation_delta": micro_row["micro_score_delta"],
                "error_regression": micro_row["error_regression"],
                "correlation_regression": micro_row["correlation_regression"],
                "regression": True,
                "micro_regression": True,
            }
            comparison["rows"].append(promoted)
            comparison["regressions"].append(promoted)

        for micro_row in micro_comparison.get("improvements", []):
            promoted = {
                "target_time_sec": micro_row["target_time_sec"],
                "baseline_error_sec": micro_row["baseline_micro_error_sec"],
                "rendered_error_sec": micro_row["rendered_micro_error_sec"],
                "error_delta_sec": micro_row["micro_error_delta_sec"],
                "baseline_correlation": micro_row["baseline_micro_score"],
                "rendered_correlation": micro_row["rendered_micro_score"],
                "correlation_delta": micro_row["micro_score_delta"],
                "error_regression": False,
                "correlation_regression": False,
                "regression": False,
                "micro_improvement": True,
            }
            comparison["rows"].append(promoted)
            comparison["improvements"].append(promoted)

        comparison["regression_count"] = len(comparison["regressions"])
        comparison["improvement_count"] = len(comparison["improvements"])
        comparison["considered_points"] = len(comparison["rows"])
        comparison["passed"] = bool(
            comparison.get("passed", False)
            and micro_comparison.get("passed", True)
        )

        save_render_guard_csv(pass_dir, comparison, pass_index)
        save_micro_render_guard_csv(pass_dir, micro_comparison, pass_index)
        _write_pass_summary(pass_dir, pass_index, comparison, quarantine_times)

        pass_records.append({
            "pass": pass_index,
            "output": str(rendered_path),
            "max_abs_safe_offset_sec": max_abs,
            "comparison": comparison,
            "pre_render_report": pre_render_report,
        })
        final_render = render_result
        final_comparison = comparison
        final_safe_map = current_safe_map

        if comparison["passed"] and (comparison['improvement_count'] > 0
                                     or render_result.get('copied_without_reencode')):
            selected_safe_result = True
            print(f"\n{context}: проверка пройдена.")
            break

        pass_records[-1]['rejected_files_removed']=_discard_rejected_files(
            render_result,guard_root,selected.target.path)
        if comparison['passed']:
            pass_records[-1]['selection_reason'] = 'NO_CONFIRMED_IMPROVEMENT'
            print(f"\n{context}: улучшение не подтверждено; аудиокандидат удалён.")
            break

        bad_times = [row["target_time_sec"] for row in comparison["regressions"]]
        print(f"\n{context}: выявлено ухудшений — {len(bad_times)}.")
        preview = ", ".join(format_seconds(t) for t in bad_times[:12])
        print("Проблемные моменты: " + preview + (" …" if len(bad_times) > 12 else ""))
        new_times = []
        for bad_time in bad_times:
            if not any(
                abs(float(bad_time) - float(existing)) <= RENDER_GUARD_QUARANTINE_RADIUS_SEC
                for existing in quarantine_times
            ):
                new_times.append(float(bad_time))

        if not new_times:
            # The renderer keeps producing a regression at a location already
            # quarantined. A larger quarantine is needed; place auxiliary centers
            # just outside the existing center to expand the excluded interval.
            for bad_time in bad_times:
                for delta in (-RENDER_GUARD_QUARANTINE_RADIUS_SEC, RENDER_GUARD_QUARANTINE_RADIUS_SEC):
                    candidate = float(bad_time) + float(delta)
                    if 0.0 < candidate < selected.target.duration:
                        new_times.append(candidate)

        if not new_times or pass_index >= RENDER_GUARD_MAX_PASSES:
            break

        quarantine_times = _merge_centers(
            quarantine_times + new_times,
            RENDER_GUARD_QUARANTINE_RADIUS_SEC,
        )

    passed = selected_safe_result
    if not passed:
        print("\nРЕНДЕР: ПРОВЕРКА ИСХОДНОЙ ДОРОЖКИ ПОСЛЕ ОТМЕНЫ ОСНОВНОЙ КАРТЫ")
        print("Далее PCM-уточнение сможет независимо исправить остаточные сдвиги.")
        # Last safety net: use the original TARGET directly, without a copy.
        fallback_dir = guard_root / "fallback_baseline"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        final_safe_map = build_zero_sync_map(selected.target.duration)
        final_render = copy_target_without_reencode(
            selected.target,
            codec_plan,
            fallback_dir,
        )
        final_verification = verify_rendered_sync(
            selected,
            analysis,
            final_render["output_path"],
            fallback_dir,
            validation_interval_sec=RENDER_GUARD_DENSE_VALIDATION_INTERVAL_SEC,
            progress=Progress("РЕНДЕР: ИСХОДНАЯ ДОРОЖКА"),
        )
        final_render["verified_first_channel_pcm"] = final_verification["rendered_first_channel_pcm"]
        final_comparison = compare_rendered_against_original(
            dense_candidate_validation_results,
            final_verification["validation_points"],
        )
        final_micro_comparison = compare_micro_rendered_against_original(
            analysis,
            final_verification,
            micro_records,
        )
        final_comparison["micro"] = final_micro_comparison
        for micro_row in final_micro_comparison.get("regressions", []):
            final_comparison["regressions"].append({
                "target_time_sec": micro_row["target_time_sec"],
                "baseline_error_sec": micro_row["baseline_micro_error_sec"],
                "rendered_error_sec": micro_row["rendered_micro_error_sec"],
                "error_delta_sec": micro_row["micro_error_delta_sec"],
                "baseline_correlation": micro_row["baseline_micro_score"],
                "rendered_correlation": micro_row["rendered_micro_score"],
                "correlation_delta": micro_row["micro_score_delta"],
                "regression": True,
                "micro_regression": True,
            })
        for micro_row in final_micro_comparison.get("improvements", []):
            final_comparison["improvements"].append({
                "target_time_sec": micro_row["target_time_sec"],
                "baseline_error_sec": micro_row["baseline_micro_error_sec"],
                "rendered_error_sec": micro_row["rendered_micro_error_sec"],
                "error_delta_sec": micro_row["micro_error_delta_sec"],
                "baseline_correlation": micro_row["baseline_micro_score"],
                "rendered_correlation": micro_row["rendered_micro_score"],
                "correlation_delta": micro_row["micro_score_delta"],
                "regression": False,
                "micro_improvement": True,
            })
        final_comparison["regression_count"] = len(final_comparison["regressions"])
        final_comparison["improvement_count"] = len(final_comparison["improvements"])
        final_comparison["considered_points"] = len(final_comparison["rows"])
        final_comparison["passed"] = bool(
            final_comparison.get("passed", False)
            and final_micro_comparison.get("passed", True)
        )
        final_render["fallback_to_baseline"] = True
        final_render["guard_passed"] = bool(final_comparison["passed"])
        pass_records.append({
            "pass": "fallback_baseline",
            "output": str(final_render["output_path"]),
            "comparison": final_comparison,
        })
    else:
        final_render["fallback_to_baseline"] = False
        final_render["guard_passed"] = True

    selected_pass_path=str(final_render['output_path'])
    _publish_guard_output(final_render,output_dir,selected.target.path)
    for record in pass_records:
        if record['output']==selected_pass_path:
            record['output']=str(final_render['output_path'])
    report = {
        "passed": bool(final_render.get("guard_passed", False)),
        "fallback_to_baseline": bool(final_render.get("fallback_to_baseline", False)),
        "passes": len(pass_records),
        "validation_interval_sec": RENDER_GUARD_DENSE_VALIDATION_INTERVAL_SEC,
        "tolerance_sec": RENDER_GUARD_TOLERANCE_SEC,
        "correlation_tolerance": RENDER_GUARD_CORRELATION_TOLERANCE,
        "quarantine_radius_sec": RENDER_GUARD_QUARANTINE_RADIUS_SEC,
        "quarantine_centers": [float(x) for x in quarantine_times],
        "final_comparison": final_comparison,
        "micro_tolerance_sec": RENDER_GUARD_MICRO_TOLERANCE_SEC,
        "micro_regression_count": int((final_comparison or {}).get("micro", {}).get("regression_count", 0)),
        "micro_improvement_count": int((final_comparison or {}).get("micro", {}).get("improvement_count", 0)),
        "micro_considered_points": int((final_comparison or {}).get("micro", {}).get("considered_points", 0)),
        "pass_records": pass_records,
        "render_output": str(final_render["output_path"]),
        "output_created": not bool(final_render.get('copied_without_reencode')),
    }

    json_path = output_dir / RENDER_GUARD_JSON
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    report_path = output_dir / RENDER_GUARD_REPORT
    with report_path.open("w", encoding="utf-8") as file:
        file.write("RENDER-AWARE NON-REGRESSION REPORT\n")
        file.write("=" * 78 + "\n\n")
        file.write(f"РЕЗУЛЬТАТ: {'ПРОЙДЕНО' if report['passed'] else 'ВОЗВРАТ К ИСХОДНОЙ ДОРОЖКЕ'}\n")
        file.write(f"ПРОХОДОВ ПОЛНОГО РЕНДЕРА: {report['passes']}\n")
        file.write(f"ИНТЕРВАЛ ПЛОТНОЙ ПРОВЕРКИ: {report['validation_interval_sec']:.1f} с\n")
        file.write(f"ДОПУСК БЕЗ УХУДШЕНИЯ: {report['tolerance_sec']*1000.0:.1f} мс\n")
        file.write(f"РАДИУС КАРАНТИНА ПОСЛЕ РЕНДЕРА: {report['quarantine_radius_sec']:.1f} с\n")
        file.write(f"ЦЕНТРОВ КАРАНТИНА: {len(report['quarantine_centers'])}\n")
        if report["quarantine_centers"]:
            file.write("ЦЕНТРЫ КАРАНТИНА (с):\n")
            file.write("  " + ", ".join(f"{x:.3f}" for x in report["quarantine_centers"]) + "\n")

        comparison = final_comparison or {}
        file.write("\nПОСЛЕДНЯЯ ПРОВЕРКА РЕАЛЬНОГО ФАЙЛА:\n")
        file.write(f"  Проверено точек: {comparison.get('considered_points', 0)}\n")
        file.write(f"  Улучшений: {comparison.get('improvement_count', 0)}\n")
        file.write(f"  Ухудшений: {comparison.get('regression_count', 0)}\n")
        if comparison.get("regressions"):
            file.write("\n  Оставшиеся ухудшения:\n")
            for row in comparison["regressions"]:
                file.write(
                    f"    {row['target_time_sec']:.3f} с: "
                    f"{row['baseline_error_sec']*1000.0:.3f} -> "
                    f"{row['rendered_error_sec']*1000.0:.3f} мс\n"
                )

        if not report['output_created']:
            file.write("\nУлучшение не подтверждено. Новый аудиофайл не создан.\n")
            file.write("Исходный TARGET используется напрямую; PCM-уточнение проверяется отдельно.\n")
        else:
            file.write("\nГотовый файл прошёл проверку реального рендера без материального ухудшения вне допустимой погрешности.\n")

    return final_safe_map, final_render, report, final_comparison
