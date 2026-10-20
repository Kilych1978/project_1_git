# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC — MODULAR COMPONENT
# MODULE: launcher.py
# ============================================================
# Pipeline orchestrator. This file should rarely need changes; it wires the modules together.
#
# This module is part of the modular version of the project.
# Do not run it directly; start the pipeline with start_sync.py.
# ============================================================

from __future__ import annotations

import json
import sys
import time

from config import *
from common import *
from scanner import *
from analyzer import *
from aligner import *
from mapper import *
from validator import *
from applier import *
from confidence import *
from renderer import *
from renderer import _get_render_codec_plan
from reports import *
from safety import *
from render_guard import *
from micro_refiner import *
from pcm_refiner import refine_rendered_audio

def main():

    total_start = time.time()

    print()
    print("=" * 78)
    print("UNIVERSAL AUDIO WAVEFORM SYNC")
    print("CONFIDENCE-AWARE NONLINEAR SYNCHRONIZATION")
    print("=" * 78)
    print()

    # ========================================================
    # ШАГ 1 — ПОИСК И ВЫБОР
    # ========================================================

    print("ШАГ 1/12")
    print("-" * 78)

    check_ffprobe()
    progress = Progress()
    audio_files = scan_audio_files(progress)

    if len(audio_files) < 2:
        raise RuntimeError("Для работы необходимо минимум два аудиофайла.")

    print()
    print(f"Найдено аудиофайлов: {len(audio_files)}")

    reference = ask_selection(
        audio_files,
        "Выбери эталонную дорожку",
    )

    target = ask_selection(
        audio_files,
        "Выбери дорожку с рассинхроном",
        excluded=reference,
    )

    selected = SelectedTracks(
        reference=reference,
        target=target,
    )

    validate_selected_tracks(selected)
    confirm_selected_tracks(selected)
    print_pair_summary(selected)

    output_dir = create_output_directory(selected.target)

    print()
    print("=" * 78)
    print("РАБОЧАЯ ПАПКА")
    print("=" * 78)
    print()
    print(output_dir)
    print()

    save_initial_metadata(output_dir, selected)
    print_algorithm_plan()

    # ========================================================
    # ШАГ 2 — PREPROCESSING
    # ========================================================

    print()
    print("ШАГ 2/12")
    print("-" * 78)

    analysis = prepare_audio_analysis(
        selected,
        output_dir,
        progress,
    )

    # После реального decode duration уже известна даже для raw DTS.
    save_initial_metadata(output_dir, selected)

    print()
    print(
        f"Реальные длительности после decode: "
        f"Reference={selected.reference.duration:.3f}s, "
        f"Target={selected.target.duration:.3f}s"
    )

    # ========================================================
    # ШАГ 3 — ALIGNMENT
    # ========================================================

    print()
    print("ШАГ 3/12")
    print("-" * 78)

    alignment = run_alignment_stage(
        selected,
        output_dir,
        analysis,
        progress,
    )

    print_initial_alignment_report(alignment, output_dir)

    # ========================================================
    # ШАГ 4 — BASE MAP
    # ========================================================

    print()
    print("ШАГ 4/12")
    print("-" * 78)

    base_map_result = build_final_sync_map(
        alignment,
        selected.target.duration,
        progress,
    )

    save_map_quality_jpeg(output_dir, base_map_result, progress)

    base_map_analysis = finalize_map_analysis(
        selected,
        output_dir,
        alignment,
        base_map_result,
        progress,
    )

    print_final_verdict(
        base_map_result,
        base_map_analysis,
        output_dir,
    )

    # ========================================================
    # ШАГ 5 — CONFIDENCE
    # ========================================================

    print()
    print("ШАГ 5/12")
    print("-" * 78)

    confidence_progress = Progress()
    confidence_result = classify_local_measurements(
        alignment["local_results"]
    )

    classes = confidence_result["class"]
    strong_count = sum(value == "STRONG" for value in classes)
    acceptable_count = sum(value == "ACCEPTABLE" for value in classes)
    weak_count = sum(value == "WEAK" for value in classes)
    invalid_count = sum(value == "INVALID" for value in classes)

    print()
    print("=" * 78)
    print("CONFIDENCE CLASSIFICATION")
    print("=" * 78)
    print()
    print(f"STRONG      : {strong_count}")
    print(f"ACCEPTABLE  : {acceptable_count}")
    print(f"WEAK        : {weak_count}")
    print(f"INVALID     : {invalid_count}")

    trusted = build_confidence_aware_anchors(
        confidence_result,
        confidence_progress,
    )

    trusted_count = len(trusted["anchor_times"])
    print()
    print(f"Trusted anchors: {trusted_count}")

    if trusted_count < 5:
        raise RuntimeError(
            "После confidence-фильтра осталось слишком мало надёжных якорей: "
            f"{trusted_count}."
        )

    # ========================================================
    # ШАГ 6 — CONFIDENCE-AWARE MAP
    # ========================================================

    print()
    print("ШАГ 6/12")
    print("-" * 78)

    confidence_map = build_confidence_aware_map(
        trusted,
        selected.target.duration,
        confidence_progress,
    )

    comparison = compare_old_new_maps(
        base_map_result["map"],
        confidence_map,
    )

    import numpy as np

    old_offsets = comparison["old"]
    new_offsets = comparison["new"]
    differences = comparison["difference"]

    base_start = float(old_offsets[0])
    confidence_start = float(new_offsets[0])
    base_end = float(old_offsets[-1])
    confidence_end = float(new_offsets[-1])
    mean_change = float(np.mean(np.abs(differences)))
    max_change = float(np.max(np.abs(differences)))

    print()
    print("=" * 78)
    print("СРАВНЕНИЕ BASE / CONFIDENCE-AWARE")
    print("=" * 78)
    print()
    print(f"Base start offset       : {base_start:+.6f} s")
    print(f"Confidence start offset : {confidence_start:+.6f} s")
    print(f"Base end offset         : {base_end:+.6f} s")
    print(f"Confidence end offset   : {confidence_end:+.6f} s")
    print(f"Mean absolute change    : {mean_change:.6f} s")
    print(f"Maximum absolute change : {max_change:.6f} s")

    save_confidence_points_csv(output_dir, trusted, confidence_progress)
    save_confidence_map_csv(output_dir, confidence_map, confidence_progress)
    save_confidence_map_jpeg(
        output_dir,
        base_map_result["map"],
        confidence_map,
        trusted,
        confidence_progress,
    )
    save_confidence_report(
        output_dir,
        trusted,
        base_map_result["map"],
        confidence_map,
    )

    final_sync_map = confidence_map

    # ========================================================
    # ШАГ 7 — МИКРО-УТОЧНЕНИЕ ФОРМЫ ВОЛНЫ
    # ========================================================

    print()
    print("ШАГ 7/12")
    print("-" * 78)
    print()
    print("МИКРО-УТОЧНЕНИЕ: локальная проверка формы волны с детализацией 1 мс")
    print("Сравнение: 5-мс окна с шагом 1 мс вокруг уже найденной карты.")

    micro_dense_progress = Progress()
    micro_dense_results = validate_sync_map(
        analysis,
        final_sync_map,
        selected.target.duration,
        micro_dense_progress,
        validation_interval_sec=SAFETY_DENSE_VALIDATION_INTERVAL_SEC,
    )

    micro_result = refine_sync_map_with_micro_waveform(
        analysis,
        final_sync_map,
        micro_dense_results,
        output_dir,
        Progress(),
    )

    final_sync_map = micro_result["sync_map"]

    final_map_result = {
        "cleaned": base_map_result["cleaned"],
        "map": final_sync_map,
        "metrics": base_map_result["metrics"],
    }

    # ========================================================
    # ШАГ 8 — КАНОНИЧЕСКАЯ ПРОВЕРКА КАРТЫ-КАНДИДАТА
    # ========================================================

    print()
    print("ШАГ 8/12")
    print("-" * 78)

    validation_progress = Progress()
    candidate_validation_results = validate_sync_map(
        analysis,
        final_sync_map,
        selected.target.duration,
        validation_progress,
        validation_interval_sec=VALIDATION_INTERVAL_SEC,
    )

    candidate_validation_metrics = calculate_validation_metrics(
        candidate_validation_results
    )

    save_candidate_validation_snapshot(
        output_dir,
        candidate_validation_results,
        candidate_validation_metrics,
    )

    # ========================================================
    # ШАГ 9 — ПЛОТНАЯ ПРОВЕРКА БЕЗ УХУДШЕНИЯ
    # ========================================================
    #
    # v6.7 preserves the canonical 60-second report grid and adds a
    # separate dense 10-second grid used exclusively for safety. Candidate
    # and SAFE MAP are measured by exactly the same independent validator
    # at exactly the same timestamps.

    print()
    print("ШАГ 9/12")
    print("-" * 78)
    print(
        "Плотная защита: независимая проверка каждые "
        f"{SAFETY_DENSE_VALIDATION_INTERVAL_SEC:.0f} s | "
        "одна и та же сетка для candidate и SAFE MAP"
    )

    dense_candidate_progress = Progress()
    dense_candidate_validation_results = validate_sync_map(
        analysis,
        final_sync_map,
        selected.target.duration,
        dense_candidate_progress,
        validation_interval_sec=SAFETY_DENSE_VALIDATION_INTERVAL_SEC,
    )

    dense_candidate_validation_metrics = calculate_validation_metrics(
        dense_candidate_validation_results
    )

    save_dense_safety_snapshot(
        output_dir,
        DENSE_SAFETY_CANDIDATE_JSON,
        dense_candidate_validation_results,
        dense_candidate_validation_metrics,
    )

    # --------------------------------------------------------
    # v6.7.2 — LOCAL QUARANTINE / PARTIAL ACCEPTANCE
    # --------------------------------------------------------
    # One bad dense point must not cancel unrelated improvements.
    # A failing point is quarantined together with its finite validation
    # neighborhood; the rest of the independently supported map remains
    # eligible for application.
    quarantine_times = []
    repair_passes = 0
    max_repair_passes = 6

    while True:
        safety_progress = Progress()
        safe_sync_map, guard_report = build_non_regressive_sync_map(
            final_sync_map,
            dense_candidate_validation_results,
            selected.target.duration,
            safety_progress,
            quarantine_times=quarantine_times,
        )

        dense_safe_progress = Progress()
        dense_safe_validation_results = validate_sync_map(
            analysis,
            safe_sync_map,
            selected.target.duration,
            dense_safe_progress,
            validation_interval_sec=SAFETY_DENSE_VALIDATION_INTERVAL_SEC,
        )

        dense_safe_validation_metrics = calculate_validation_metrics(
            dense_safe_validation_results
        )

        dense_safe_assessment = assess_safe_validation(
            dense_safe_validation_results,
            excluded_times=quarantine_times,
            excluded_radius_sec=SAFETY_QUARANTINE_RADIUS_SEC,
        )

        quarantine_fraction = float(
            guard_report.get("quarantine_fraction", 0.0) or 0.0
        )

        # Once too much of the film is quarantined, the safety system stops
        # trying to carve the map into tiny pieces and returns to baseline.
        if quarantine_fraction > SAFETY_MAX_QUARANTINE_FRACTION:
            dense_safe_assessment = dict(dense_safe_assessment)
            dense_safe_assessment["passed"] = False
            dense_safe_assessment["reason"] = (
                "Local quarantine covers too much of the validation grid; "
                "baseline TARGET is safer."
            )

        if dense_safe_assessment["passed"]:
            break

        bad_times = [
            float(item["target_time"])
            for item in dense_safe_assessment.get("worsened_points", [])
        ]

        if not bad_times or repair_passes >= max_repair_passes:
            break

        # Only add genuinely new bad regions. Existing quarantines are already
        # excluded from the assessment and therefore cannot trigger an endless
        # loop.
        new_bad_times = []
        for bad_time in bad_times:
            if not any(
                abs(bad_time - existing) <= SAFETY_QUARANTINE_RADIUS_SEC
                for existing in quarantine_times
            ):
                new_bad_times.append(bad_time)

        if not new_bad_times:
            break

        repair_passes += 1
        quarantine_times.extend(new_bad_times)
        quarantine_times.sort()

        print()
        print(
            f"Локальный карантин {repair_passes}/{max_repair_passes}: "
            f"исключаются {len(new_bad_times)} новых проблемных точек, "
            f"радиус {SAFETY_QUARANTINE_RADIUS_SEC:.0f} с"
        )

    fallback_to_baseline = not dense_safe_assessment["passed"]

    if not fallback_to_baseline and quarantine_times:
        print()
        print(
            "Локальная защита сработала: подтверждённые улучшения сохранены, "
            f"а {len(quarantine_times)} конфликтных участков оставлены без коррекции."
        )

    # If the local partial map could not be made safe, preserve the original
    # TARGET exactly as before. This is the final safety net.
    if fallback_to_baseline:
        print()
        print("ПЛОТНАЯ БЕЗОПАСНАЯ КАРТА НЕ ПРОШЛА ПРОВЕРКУ БЕЗ УХУДШЕНИЯ")
        print(
            "После локального карантина сохраняем исходный TARGET "
            "без дополнительной коррекции."
        )

        safe_sync_map = build_zero_sync_map(
            selected.target.duration
        )

        dense_safe_progress = Progress()
        dense_safe_validation_results = validate_sync_map(
            analysis,
            safe_sync_map,
            selected.target.duration,
            dense_safe_progress,
            validation_interval_sec=SAFETY_DENSE_VALIDATION_INTERVAL_SEC,
        )
        dense_safe_validation_metrics = calculate_validation_metrics(
            dense_safe_validation_results
        )
        dense_safe_assessment = assess_safe_validation(
            dense_safe_validation_results,
            excluded_times=[],
            excluded_radius_sec=SAFETY_QUARANTINE_RADIUS_SEC,
        )
        guard_report["mode"] = "FALLBACK_BASELINE"

    guard_report["repair_passes"] = repair_passes
    guard_report["fallback_to_baseline"] = bool(fallback_to_baseline)
    guard_report["final_assessment"] = dense_safe_assessment
    guard_report["quarantine_centers"] = [float(value) for value in quarantine_times]
    guard_report["quarantine_radius_sec"] = float(SAFETY_QUARANTINE_RADIUS_SEC)
    guard_report["local_partial_acceptance"] = bool(
        (not fallback_to_baseline) and bool(quarantine_times)
    )

    save_non_regression_report(
        output_dir,
        dense_candidate_validation_metrics,
        dense_safe_validation_metrics,
        guard_report,
        dense_safe_assessment,
    )

    save_dense_safety_snapshot(
        output_dir,
        DENSE_SAFETY_FINAL_JSON,
        dense_safe_validation_results,
        dense_safe_validation_metrics,
        guard_report,
    )

    save_dense_safety_validation_csv(
        output_dir,
        dense_candidate_validation_results,
        dense_safe_validation_results,
        quarantine_times=quarantine_times,
        quarantine_radius_sec=SAFETY_QUARANTINE_RADIUS_SEC,
    )

    # Финальная каноническая 60-секундная validation уже выбранной карты.
    final_canonical_validation_progress = Progress()
    final_canonical_validation_results = validate_sync_map(
        analysis,
        safe_sync_map,
        selected.target.duration,
        final_canonical_validation_progress,
        validation_interval_sec=VALIDATION_INTERVAL_SEC,
    )

    final_canonical_validation_metrics = calculate_validation_metrics(
        final_canonical_validation_results
    )

    save_safe_validation_snapshot(
        output_dir,
        final_canonical_validation_results,
        final_canonical_validation_metrics,
    )

    final_sync_map = safe_sync_map
    final_map_result = {
        "cleaned": base_map_result["cleaned"],
        "map": final_sync_map,
        "metrics": base_map_result["metrics"],
    }

    # FINAL_SYNC_REPORT must describe the map that will actually be applied.
    final_map_progress = Progress()
    finalize_map_analysis(
        selected,
        output_dir,
        alignment,
        final_map_result,
        final_map_progress,
    )

    validation_results = final_canonical_validation_results
    validation_metrics = final_canonical_validation_metrics

    save_before_after_jpeg(
        output_dir,
        validation_results,
        final_canonical_validation_progress,
    )
    save_alignment_error_jpeg(
        output_dir,
        validation_results,
        final_canonical_validation_progress,
    )
    save_validation_report(output_dir, validation_metrics)
    save_validation_json(output_dir, validation_results, validation_metrics)
    save_validation_points_csv(
        output_dir,
        validation_results,
        final_canonical_validation_progress,
    )

    validation_outliers = find_validation_outliers(validation_results)
    save_validation_outliers_csv(
        output_dir,
        validation_outliers,
        final_canonical_validation_progress,
    )

    validation_verdict = evaluate_validation(validation_metrics)
    print_validation_result(validation_metrics, validation_verdict, output_dir)

    # ========================================================
    # ШАГ 10 — УСТОЙЧИВАЯ ПРОВЕРКА + ДИАГНОСТИКА
    # ========================================================

    print()
    print("ШАГ 10/12")
    print("-" * 78)

    robust_progress = Progress()
    robust_result = finalize_robust_validation(
        output_dir,
        final_map_result,
        validation_results,
        robust_progress,
    )

    print_final_robust_verdict(robust_result, output_dir)

    hard_points = extract_manual_review_points(
        validation_results,
        final_sync_map,
    )

    save_manual_audacity_report(
        output_dir,
        validation_results,
        final_sync_map,
    )

    print()
    print("=" * 78)
    print("VALIDATION DIAGNOSTICS")
    print("=" * 78)
    print()
    all_candidates = find_validation_outliers(validation_results)
    candidate_hard = [
        item for item in all_candidates if item.get("severity") == "HARD"
    ]
    confirmed_hard = [
        item for item in candidate_hard
        if item.get("bidir_status") == "CONFIRMED_DISCREPANCY"
    ]
    ambiguous_hard = [
        item for item in candidate_hard
        if item.get("bidir_status") != "CONFIRMED_DISCREPANCY"
    ]
    print(f"Validation outlier candidates : {len(all_candidates)}")
    print(f"Candidate HARD outliers       : {len(candidate_hard)}")
    print(f"Confirmed discrepancies       : {len(confirmed_hard)}")
    print(f"Ambiguous HARD candidates     : {len(ambiguous_hard)}")
    print("Safe-map guard status         : " + ("PASSED" if dense_safe_assessment.get("passed") else "BASELINE FALLBACK"))
    print(f"Local quarantine centers      : {len(quarantine_times)}")
    print(f"Local guard repair passes     : {guard_report.get('repair_passes', 0)}")
    if confirmed_hard:
        print()
        print("Подтверждённые HARD-расхождения:")
        for index, item in enumerate(confirmed_hard, 1):
            print()
            print(f"#{index}: {format_time_millis(item['target_time'])}")
            print(f"  predicted offset : {item['predicted_offset']:+.6f} s")
            actual_text = "N/A" if item["actual_offset"] is None else f"{item['actual_offset']:+.6f} s"
            print("  actual offset    : " + actual_text)
            print(f"  residual          : {item['residual']:.6f} s")
            print(f"  correlation       : {item['correlation']:.6f}")
            print(f"  bidirectional     : {item.get('bidir_status', 'UNRESOLVED')}")
    else:
        print()
        print("Подтверждённых HARD-расхождений после двусторонней проверки не обнаружено.")
    if ambiguous_hard:
        print()
        print("Неоднозначные HARD-кандидаты остаются только диагностикой и карту не изменяют.")

    # ========================================================
    # ПРЕД-APPLY: эффективность + выбор формата + подтверждение
    # ========================================================

    validation_effectiveness_report = save_sync_effectiveness_report(
        output_dir,
        selected,
        validation_metrics,
        final_sync_map,
    )

    codec_plan = _get_render_codec_plan(selected.target)
    # A changed compressed track must not be encoded lossily twice (coarse,
    # then fine). Use a lossless intermediate/final container for a correction.
    if (not selected.target.codec_name.lower().startswith('pcm_')
            and float(np.max(np.abs(final_sync_map['offset']))) > NO_REGRESSION_ZERO_MAP_SEC):
        codec_plan = {**codec_plan, 'encoder': 'pcm_f32le', 'container': 'wav',
                      'extension': '.wav', 'pcm_fallback': True,
                      'reason': 'Исправление времени без повторного кодирования с потерями: PCM WAV/RF64.'}
    print('После основного рендера: независимая проверка остаточного сдвига по PCM.')
    print('При PCM-уточнении сохраняются число каналов, частота и исходная разрядность PCM.')

    print()
    print("Защита от ухудшения сохраняет хорошие участки без необоснованной повторной коррекции.")
    print(f"Подтверждённых HARD-расхождений: {len(confirmed_hard)}")
    print(f"Неоднозначных HARD-кандидатов: {len(ambiguous_hard)}")

    if not prompt_before_apply(
        selected,
        final_sync_map,
        validation_metrics,
        codec_plan,
        guard_report=guard_report,
    ):
        print()
        print("Пользователь отменил применение карты.")
        print("Исходный TARGET не изменён.")
        print(f"Отчёт эффективности: {validation_effectiveness_report}")
        create_russian_report_copies(output_dir)
        return

    # ========================================================
    # ШАГ 11 — ПРИМЕНЕНИЕ + РЕАЛЬНЫЙ РЕНДЕР С ЗАЩИТОЙ
    # ========================================================

    print()
    print("ШАГ 11/12")
    print("-" * 78)
    print()
    print("Проверяем не только математическую карту, но и фактически отрендерированный сигнал.")
    print(
        f"Плотная проверка готового файла: каждые "
        f"{RENDER_GUARD_DENSE_VALIDATION_INTERVAL_SEC:.0f} с"
    )

    (
        final_sync_map,
        render_result,
        render_guard_report,
        render_guard_comparison,
    ) = run_render_aware_safety_loop(
        selected,
        analysis,
        final_sync_map,
        final_sync_map,
        dense_candidate_validation_results,
        quarantine_times,
        output_dir,
        codec_plan,
        Progress(),
        micro_records=micro_result.get("accepted", []),
    )

    # Render-aware protection can only remove/neutralize correction; it never
    # creates a stronger correction than the pre-apply map the user approved.
    guard_report["render_aware"] = render_guard_report
    quarantine_times = list(render_guard_report.get("quarantine_centers", []))

    coarse_sync_map = final_sync_map
    final_sync_map, render_result, pcm_report = refine_rendered_audio(
        selected, analysis, coarse_sync_map, render_result, output_dir,
    )
    codec_plan = render_result['codec_plan']

    # The map that is actually present in the output must become the canonical
    # map used by all final reports. Revalidate it before writing final reports.
    final_map_result = {
        "cleaned": base_map_result["cleaned"],
        "map": final_sync_map,
        "metrics": base_map_result["metrics"],
    }

    final_map_progress = Progress()
    finalize_map_analysis(
        selected,
        output_dir,
        alignment,
        final_map_result,
        final_map_progress,
    )

    final_canonical_validation_progress = Progress()
    final_canonical_validation_results = validate_sync_map(
        analysis,
        final_sync_map,
        selected.target.duration,
        final_canonical_validation_progress,
        validation_interval_sec=VALIDATION_INTERVAL_SEC,
    )
    final_canonical_validation_metrics = calculate_validation_metrics(
        final_canonical_validation_results
    )

    # v6.8.1: if render-aware protection changed the map, refresh the dense
    # safety validation so the saved safety report describes the actual final
    # map rather than the pre-render-safe map.
    dense_safe_progress = Progress()
    dense_safe_validation_results = validate_sync_map(
        analysis,
        final_sync_map,
        selected.target.duration,
        dense_safe_progress,
        validation_interval_sec=RENDER_GUARD_DENSE_VALIDATION_INTERVAL_SEC,
    )
    dense_safe_validation_metrics = calculate_validation_metrics(
        dense_safe_validation_results
    )
    dense_safe_assessment = assess_safe_validation(
        dense_safe_validation_results,
        excluded_times=quarantine_times,
        excluded_radius_sec=RENDER_GUARD_DENSE_VALIDATION_INTERVAL_SEC
        if not quarantine_times
        else float(guard_report.get("quarantine_radius_sec", SAFETY_QUARANTINE_RADIUS_SEC)),
    )

    validation_results = final_canonical_validation_results
    validation_metrics = final_canonical_validation_metrics

    save_before_after_jpeg(
        output_dir,
        validation_results,
        final_canonical_validation_progress,
    )
    save_alignment_error_jpeg(
        output_dir,
        validation_results,
        final_canonical_validation_progress,
    )
    save_validation_report(output_dir, validation_metrics)
    save_validation_json(output_dir, validation_results, validation_metrics)
    save_validation_points_csv(
        output_dir,
        validation_results,
        final_canonical_validation_progress,
    )

    validation_outliers = find_validation_outliers(validation_results)
    save_validation_outliers_csv(
        output_dir,
        validation_outliers,
        final_canonical_validation_progress,
    )

    validation_verdict = evaluate_validation(validation_metrics)
    print_validation_result(validation_metrics, validation_verdict, output_dir)

    robust_progress = Progress()
    robust_result = finalize_robust_validation(
        output_dir,
        final_map_result,
        validation_results,
        robust_progress,
    )
    print_final_robust_verdict(robust_result, output_dir)

    hard_points = extract_manual_review_points(
        validation_results,
        final_sync_map,
    )
    save_manual_audacity_report(
        output_dir,
        validation_results,
        final_sync_map,
    )

    save_sync_effectiveness_report(
        output_dir,
        selected,
        validation_metrics,
        final_sync_map,
    )
    save_non_regression_report(
        output_dir,
        dense_candidate_validation_metrics,
        dense_safe_validation_metrics,
        guard_report,
        dense_safe_assessment,
    )
    save_dense_safety_snapshot(
        output_dir,
        DENSE_SAFETY_FINAL_JSON,
        dense_safe_validation_results,
        dense_safe_validation_metrics,
        guard_report,
    )
    save_dense_safety_validation_csv(
        output_dir,
        dense_candidate_validation_results,
        dense_safe_validation_results,
        quarantine_times=quarantine_times,
        quarantine_radius_sec=float(guard_report.get("quarantine_radius_sec", SAFETY_QUARANTINE_RADIUS_SEC)),
    )

    # Build and save the final apply map only after render-aware protection has
    # settled the final timing map. This prevents stale APPLY reports.
    apply_progress = Progress()
    apply_map = build_apply_map(
        final_sync_map,
        selected.target.duration,
        apply_progress,
    )
    validate_apply_map_limits(apply_map)

    save_pre_apply_map_csv(
        output_dir,
        apply_map,
        apply_progress,
    )
    save_apply_plan_json(
        output_dir,
        apply_map,
        robust_result,
        apply_progress,
    )
    save_apply_plan_jpeg(
        output_dir,
        apply_map,
        apply_progress,
    )
    save_pre_apply_report(
        output_dir,
        selected,
        robust_result,
        apply_map,
        hard_points,
    )

    rendered_path = render_result["output_path"]

    # ========================================================
    # ШАГ 12 — ФИНАЛЬНАЯ ПРОВЕРКА РЕЗУЛЬТАТА
    # ========================================================

    print()
    print("ШАГ 12/12")
    print("-" * 78)

    rendered_info = verify_rendered_format(
        rendered_path,
        selected.target,
        codec_plan=codec_plan,
    )

    final_verification = verify_rendered_sync(
        selected,
        analysis,
        rendered_path,
        output_dir,
        validation_interval_sec=RENDER_GUARD_DENSE_VALIDATION_INTERVAL_SEC,
    )

    final_effectiveness_report = save_final_effectiveness_report(
        output_dir,
        selected,
        validation_metrics,
        final_verification["metrics"],
        final_verification,
    )

    # ========================================================
    # FINAL JSON
    # ========================================================

    final_map_json = output_dir / "FINAL_SYNC_MAP.json"

    final_json = {
        "reference_file": str(selected.reference.path),
        "target_file": str(selected.target.path),
        "output_file": str(rendered_path),
        "reference_duration": float(selected.reference.duration),
        "target_duration": float(selected.target.duration),
        "output_duration": float(rendered_info.duration),
        "reference_time_definition": "REFERENCE = TARGET + OFFSET",
        "target_time": [float(x) for x in final_sync_map["target_time"]],
        "offset": [float(x) for x in final_sync_map["offset"]],
        "reference_time": [float(x) for x in final_sync_map["reference_time"]],
        "confidence": [float(x) for x in final_sync_map["confidence"]],
        "apply_min_tempo": float(apply_map["min_tempo"]),
        "apply_max_tempo": float(apply_map["max_tempo"]),
        "trusted_anchor_count": int(trusted_count),
        "strong_count": int(strong_count),
        "acceptable_count": int(acceptable_count),
        "weak_count": int(weak_count),
        "invalid_count": int(invalid_count),
        "validation_metrics": validation_metrics,
        "candidate_validation_metrics": candidate_validation_metrics,
        "non_regression_guard": guard_report,
        "safe_validation_assessment": dense_safe_assessment,
        "render_aware_non_regression": render_guard_report,
        "pcm_micro_refinement": pcm_report,
        "coarse_map_before_pcm": {
            'target_time': [float(x) for x in coarse_sync_map['target_time']],
            'reference_time': [float(x) for x in coarse_sync_map['reference_time']],
            'offset': [float(x) for x in coarse_sync_map['offset']],
        },
        "robust_verdict": robust_result["quality"]["verdict"],
        "robust_score": float(robust_result["quality"]["score"]),
        "render_encoder": render_result["encoder"],
        "render_codec_plan": render_result["codec_plan"],
        "render_segment_count": len(render_result["segments"]),
        "render_max_mapping_error_sec": max(
            (float(item["mapping_error"]) for item in render_result["segments"]),
            default=0.0,
        ),
        "final_render_validation": final_verification["metrics"],
        "final_render_duration_difference_from_reference": float(final_verification["duration_difference"]),
        "final_render_duration_check": bool(final_verification["duration_ok"]),
        "final_render_duration_diagnostics": final_verification.get("duration_diagnostics", {}),
        "final_effectiveness_report": str(final_effectiveness_report),
        "audio_rendered": not bool(render_result.get('copied_without_reencode', False)),
        "result_status": pcm_report['status'] if pcm_report['status']=='PCM_REFINED' else (
            'BASELINE_KEPT' if render_result.get('copied_without_reencode') else 'COARSE_REFINED'),
        "generated_by": Path(sys.argv[0]).name,
        "algorithm": "confidence-aware nonlinear waveform map + dense safety + render-aware non-regression",
    }

    with final_map_json.open("w", encoding="utf-8") as file:
        json.dump(final_json, file, ensure_ascii=False, indent=2)

    # Английские отчёты остаются основными. Дополнительно создаём
    # отдельные русские копии всех TXT-отчётов.
    russian_reports = create_russian_report_copies(output_dir)

    elapsed = time.time() - total_start
    start_offset = float(final_sync_map["offset"][0])
    end_offset = float(final_sync_map["offset"][-1])

    print()
    print("=" * 78)
    print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ")
    print("=" * 78)
    print()
    print("Reference:", selected.reference.path.name)
    print("Target   :", selected.target.path.name)
    print()
    print(f"Start offset : {start_offset:+.6f} s")
    print(f"End offset   : {end_offset:+.6f} s")
    print(f"Map change   : {end_offset - start_offset:+.6f} s")
    print()
    print(f"Trusted anchors : {trusted_count}")
    print(f"STRONG          : {strong_count}")
    print(f"ACCEPTABLE      : {acceptable_count}")
    print(f"WEAK            : {weak_count}")
    print(f"INVALID         : {invalid_count}")
    print()
    print("Robust verdict :", robust_result["quality"]["verdict"])
    print(f"Robust score   : {robust_result['quality']['score']:.2f}/100")
    print()
    print("Final output:")
    print(rendered_path)
    print()
    print("Output format:", rendered_info.codec_name, rendered_info.sample_rate, "Hz", rendered_info.channels, "ch")
    print("Final rendered duration:", f"{rendered_info.duration:.3f} s")
    print("Difference from Reference:", f"{final_verification['duration_difference']:+.6f} s")
    if codec_plan.get("pcm_fallback"):
        print("Lossless PCM fallback: YES")
    print("Final sync median residual:", final_verification["metrics"].get("after_median_error"))
    print("Final sync mean residual  :", final_verification["metrics"].get("after_mean_error"))
    print("Final sync max residual   :", final_verification["metrics"].get("after_max_error"))
    print()
    print("PCM-уточнение:", pcm_report['reason'])
    checked_pcm = [r for r in pcm_report.get('actual_output_windows', [])
                   if r['status'] == 'VERIFIED' and r['micro_changed']]
    if checked_pcm and pcm_report['status'] == 'PCM_REFINED':
        residuals = np.array([abs(r['residual_sec']) for r in checked_pcm])*1000
        print(f"Остаток PCM в проверенных изменённых окнах: медиана {np.median(residuals):.6f} мс; максимум {np.max(residuals):.6f} мс")
        print("Проверенных окон с остатком больше 0,1 мс:",len(pcm_report.get('remaining_windows', [])))
        print("Оставшиеся расхождения: PCM_MICRO_REMAINING.csv (если такие окна есть).")
    print()
    print("Исходный TARGET НЕ изменён.")
    if render_result.get('copied_without_reencode'):
        print("Безопасное улучшение не подтверждено. Создана точная копия исходной дорожки.")
    else:
        print("Готовый файл создан отдельно; результат проверок приведён в отчётах.")
    print()
    print("FINAL_SYNC_MAP.json:")
    print(final_map_json)
    print()
    print(f"Русских копий TXT-отчётов создано: {len(russian_reports)}")
    print()
    print(f"Общее время: {elapsed / 60.0:.2f} min")
    print()
    print("=" * 78)
    print("ГОТОВО")
    print("=" * 78)
    print()
