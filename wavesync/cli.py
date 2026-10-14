import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
from dataclasses import replace
import numpy as np
from . import __version__
from .alignment import AlignmentError, coarse_alignment, envelope_match, validate_coarse
from .audio import check_disk, decode, describe, find_tools, probe
from .config import Settings
from .features import envelope
from .guard import gate_micro, render_verdict, same_map, verify_render
from .micro import collect
from .model import TimeMap
from .render import render
from .reports import csv_write, json_write, plots, write_report


AUDIO_EXTENSIONS = {".wav", ".wave", ".flac", ".ac3", ".eac3", ".dts", ".mka", ".m4a", ".mp3", ".aac", ".ogg", ".opus", ".ape", ".wv", ".mkv", ".mp4"}


def parser():
    p = argparse.ArgumentParser(description="Синхронизация звука по первому каналу с проверкой фактического результата.")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--check", action="store_true", help="Проверить инструменты и Python-пакеты без обработки аудио")
    p.add_argument("--directory", type=Path, default=Path.cwd(), help="Папка для поиска дорожек")
    p.add_argument("--reference", type=Path, help="Эталонная дорожка")
    p.add_argument("--target", type=Path, help="Дорожка для исправления")
    p.add_argument("--reference-stream", type=int, help="Индекс звукового потока в эталонном контейнере")
    p.add_argument("--target-stream", type=int, help="Индекс звукового потока в исходном контейнере")
    p.add_argument("--output", type=Path, help="Новая папка результатов; существующая не перезаписывается")
    p.add_argument("--refine-only", action="store_true", help="Уточнить уже сведённую дорожку: начальная карта тождественная")
    p.add_argument("--map-json", type=Path, help="Явная карта времени, включая границы монтажа (см. документацию)")
    p.add_argument("--step", type=float, default=5., help="Шаг анализа в секундах: 0,5–30; по умолчанию 5")
    p.add_argument("--search-radius", type=float, default=120., help="Радиус поиска крупного смещения в секундах")
    p.add_argument("--keep-work", action="store_true", help="Сохранить временные моно PCM-файлы для диагностики")
    p.add_argument("--no-plots", action="store_true", help="Не создавать графики")
    return p


def select_pair(directory):
    paths = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS), key=lambda p: p.name.casefold())
    files = []
    for path in paths:
        try:
            files.append(probe(path))
        except (RuntimeError, ValueError) as exc:
            print(f"  Пропущен {path.name}: {exc}")
    if len(files) < 2:
        raise ValueError("В папке нужно минимум два аудиофайла. Для контейнеров с несколькими потоками используйте параметры командной строки.")
    def choose(title, items):
        print("\n" + title)
        for i, info in enumerate(items, 1):
            print(f"{i}. {describe(info)}")
        while True:
            answer = input("Номер дорожки (q — выход): ").strip()
            if answer.lower() in {"q", "й"}:
                raise KeyboardInterrupt
            try:
                index = int(answer) - 1
                if 0 <= index < len(items):
                    return items[index]
            except ValueError:
                pass
            print("Введите номер из списка.")
    reference = choose("Выберите эталонную дорожку без рассинхрона:", files)
    target = choose("Выберите дорожку с рассинхроном (нумерация начинается заново):", [p for p in files if p.path != reference.path])
    return reference, target


def output_directory(target, requested):
    if requested:
        directory = requested.expanduser().resolve()
    else:
        # Windows has no bundled IANA timezone database; fixed Samara UTC+4 works here.
        stamp = datetime.now(timezone(timedelta(hours=4))).strftime("%Y%m%d_%H%M%S")
        stem = target.path.stem[:60]
        directory = target.path.parent / f"{stem}_WAVEFORM_SYNC_{stamp}"
        suffix = 1
        while directory.exists():
            directory = target.path.parent / f"{stem}_WAVEFORM_SYNC_{stamp}_{suffix}"
            suffix += 1
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def verify_output_envelope(reference_env, actual_env, duration, settings):
    rows = []
    window = min(3., duration / 8)
    for t in np.arange(window, duration - window, settings.anchor_step):
        m = envelope_match(reference_env, actual_env, t, 1., t, window, .100, settings)
        if m and m["score"] >= settings.min_coarse_score and m["margin"] >= settings.min_coarse_margin:
            rows.append({**m, "status": "VERIFIED" if abs(m["offset"]) < .020 else "FAILED"})
        else:
            rows.append({"target_time": float(t), "status": "UNVERIFIABLE"})
    good = sum(r["status"] == "VERIFIED" for r in rows)
    return good >= 3 and good / max(1, len(rows)) >= .60 and not any(r["status"] == "FAILED" for r in rows), rows


def synchronize(reference_info, target_info, args, settings, log=print):
    started = time.monotonic()
    directory = output_directory(target_info, args.output)
    work = directory / "work"
    work.mkdir()
    os.environ.setdefault("MPLCONFIGDIR", str(work / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(work / "cache"))
    report = {"version": __version__, "status": "BLOCKED", "reference": reference_info.to_dict(),
              "target": target_info.to_dict(), "output": None, "mode": "refine-only" if args.refine_only else "automatic",
              "settings": settings.__dict__, "validation_scope": "sampled checkpoints; not a whole-film guarantee"}
    reference = target = actual = None
    ref_env = tar_env = None
    base = final = None
    measurements = []
    validation = []
    candidate_path = None
    try:
        log("\n1/6. Декодирование первого канала и построение огибающих.")
        required = ((reference_info.duration + target_info.duration) * settings.analysis_rate * 4) * 1.2 + 128 * 2**20
        check_disk(work, int(required))
        reference = decode(reference_info, work / "reference_channel1.f32", settings.analysis_rate, True)
        target = decode(target_info, work / "target_channel1.f32", settings.analysis_rate, True)
        reference_info.duration = len(reference) / settings.analysis_rate
        target_info.duration = len(target) / settings.analysis_rate
        report["reference"], report["target"] = reference_info.to_dict(), target_info.to_dict()
        if min(reference_info.duration, target_info.duration) < 1.6:
            raise AlignmentError("Для анализа требуется минимум 1,6 секунды аудио.")
        ref_env, tar_env = envelope(reference, settings.analysis_rate, settings), envelope(target, settings.analysis_rate, settings)
        log("\n2/6. Построение крупной карты времени.")
        if args.map_json:
            base = TimeMap.from_dict(json.loads(args.map_json.read_text(encoding="utf-8-sig"))).validate(settings.max_rate_change)
            if any(p.target[0] < 0 or p.target[-1] > target_info.duration + 1 / settings.analysis_rate for p in base.pieces):
                raise ValueError("Ручная карта выходит за пределы исходной дорожки.")
            report["mode"] = "explicit-map"
        elif args.refine_only:
            base = TimeMap.affine(target_info.duration)
        else:
            base, coarse_records = coarse_alignment(ref_env, tar_env, reference_info.duration, target_info.duration, settings, log)
            report["coarse_measurements"] = coarse_records
            csv_write(directory / "COARSE_MEASUREMENTS.csv", coarse_records)
        if not args.refine_only:
            coarse_validation = validate_coarse(base, ref_env, tar_env, target_info.duration, settings)
            report["coarse_holdout_validation"] = coarse_validation
            csv_write(directory / "COARSE_HOLDOUT.csv", coarse_validation)
        json_write(directory / "BASE_MAP.json", base.to_dict())
        log("\n3/6. Уточнение по PCM: поиск дробного лага с учётом скорости карты.")
        measurements = collect(reference, target, base, settings, log)
        report["micro_measurements"] = [m.to_dict() for m in measurements]
        csv_write(directory / "MICRO_MEASUREMENTS.csv", report["micro_measurements"])
        log("\n4/6. Проверка поправок на отдельных окнах и локальная отмена сомнительных участков.")
        final, validation, audit = gate_micro(reference, target, base, measurements, settings, log)
        adaptive_steps = [settings.anchor_step]
        for _ in range(2):
            remaining = [abs(r["after_residual_sec"]) for r in validation
                         if r["status"] == "IMPROVED" and r["after_residual_sec"] is not None]
            if not remaining or max(remaining) <= 1 / settings.analysis_rate or settings.anchor_step <= 1.25:
                break
            settings = replace(settings, anchor_step=max(.5, settings.anchor_step / 2))
            adaptive_steps.append(settings.anchor_step)
            log(f"  Остался измеримый нелинейный остаток: уменьшаю шаг до {settings.anchor_step:g} с и использую новую проверочную сетку.")
            measurements = collect(reference, target, base, settings, log)
            final, validation, more_audit = gate_micro(reference, target, base, measurements, settings, log)
            audit += more_audit
        report["adaptive_steps_sec"] = adaptive_steps
        report["settings"] = settings.__dict__
        report["micro_measurements"] = [m.to_dict() for m in measurements]
        csv_write(directory / "MICRO_MEASUREMENTS.csv", report["micro_measurements"])
        report["micro_holdout_validation"] = validation
        report["micro_guard_audit"] = audit
        csv_write(directory / "MICRO_HOLDOUT.csv", audit)
        json_write(directory / "FINAL_MAP.json", final.to_dict())
        report["edit_gaps"] = [{"target_start": float(left.target[-1]), "target_end": float(right.target[0]),
                               "reference_start": float(left.reference[-1]), "reference_end": float(right.reference[0]),
                               "target_gap_sec": float(right.target[0] - left.target[-1]),
                               "reference_gap_sec": float(right.reference[0] - left.reference[-1])}
                              for left, right in zip(final.pieces, final.pieces[1:])]
        report["reference_padding"] = []
        if final.pieces[0].reference[0] > 0:
            report["reference_padding"].append([0., min(float(final.pieces[0].reference[0]), reference_info.duration)])
        if final.pieces[-1].reference[-1] < reference_info.duration:
            report["reference_padding"].append([max(0., float(final.pieces[-1].reference[-1])), reference_info.duration])
        map_rows = []
        for i, p in enumerate(final.pieces):
            for t, r in zip(p.target, p.reference):
                map_rows.append({"piece": i, "target_time": float(t), "reference_time": float(r),
                                 "offset_sec": float(r - t), "micro_sec": float(r - base.forward([t])[0])})
        csv_write(directory / "FINAL_MAP.csv", map_rows)
        if args.refine_only and same_map(base, final):
            report["status"] = "NO_CONFIRMED_IMPROVEMENT"
            log("Надёжное улучшение не подтверждено. Исходная дорожка сохранена без перекодирования.")
        else:
            log("\n5/6. Однократный рендер всех каналов в WAV/RF64.")
            candidate_path = render(target_info, final, reference_info.duration, directory, settings, log)
            actual_info = probe(candidate_path)
            if actual_info.channels != target_info.channels or actual_info.sample_rate != target_info.sample_rate:
                raise RuntimeError("Выходной файл не сохранил число каналов или частоту дискретизации.")
            if target_info.channel_layout and actual_info.channel_layout != target_info.channel_layout:
                raise RuntimeError("Выходной файл не сохранил раскладку каналов.")
            log("\n6/6. Декодирование и проверка фактического выходного файла.")
            actual = decode(actual_info, work / "output_channel1.f32", settings.analysis_rate, True)
            duration_error = len(actual) / settings.analysis_rate - reference_info.duration
            if abs(duration_error) > 2 / settings.analysis_rate:
                raise RuntimeError("Длительность фактического результата не совпала с эталоном.")
            checks = verify_render(reference, target, actual, base, final, settings)
            report["output_pcm_validation"] = checks
            report["duration_error_sec"] = duration_error
            csv_write(directory / "ACTUAL_OUTPUT_VALIDATION.csv", checks)
            micro_applied = not same_map(base, final)
            verdict = render_verdict(checks, require_micro=micro_applied or args.refine_only)
            env_good, env_rows = verify_output_envelope(ref_env, envelope(actual, settings.analysis_rate, settings), reference_info.duration, settings)
            report["output_envelope_validation"] = env_rows
            csv_write(directory / "ACTUAL_OUTPUT_ENVELOPE.csv", env_rows)
            if verdict == "VERIFIED_PCM":
                destination = directory / "SYNCED_TARGET.wav"
            elif verdict == "VERIFIED_ENVELOPE_ONLY" and env_good:
                destination = directory / "COARSE_ALIGNED_REVIEW.wav"
            else:
                destination = None
                verdict = "REJECTED"
            if destination:
                # Files are in a newly created output directory; no input or old result is replaced.
                candidate_path.rename(destination)
                report["output"] = str(destination)
            report["status"] = verdict
            report["renders"] = 1
        if not args.no_plots:
            log("Создание графиков и увеличенных сравнений звуковой волны.")
            try:
                plots(directory, reference, target, ref_env, tar_env, base, final, measurements, validation, actual, settings,
                      report.get("output_pcm_validation", []))
            except Exception as exc:
                report["plot_error"] = str(exc)
                log("Не удалось завершить создание графиков: " + str(exc))
    except (AlignmentError, ValueError, RuntimeError, OSError) as exc:
        report["status"] = "BLOCKED"
        report["error"] = str(exc)
        if isinstance(exc, AlignmentError):
            report["failed_stage_measurements"] = exc.records
            csv_write(directory / "FAILED_STAGE.csv", exc.records)
        log("Остановлено: " + str(exc))
        if reference is not None and target is not None and ref_env is not None and not args.no_plots:
            try:
                plots(directory, reference, target, ref_env, tar_env, base, final, measurements, validation, actual, settings,
                      report.get("output_pcm_validation", []))
            except Exception as plot_exc:
                report["plot_error"] = str(plot_exc)
    except KeyboardInterrupt:
        report["status"] = "BLOCKED"
        report["error"] = "Остановлено пользователем."
        if report.get("output"):
            Path(report["output"]).unlink(missing_ok=True)
            report["output"] = None
        raise
    finally:
        report["elapsed_sec"] = time.monotonic() - started
        write_report(directory, report)
        if candidate_path and candidate_path.exists():
            candidate_path.unlink()
        for data in (reference, target, actual):
            if data is not None:
                data._mmap.close()
        if not args.keep_work:
            shutil.rmtree(work)
    return report, directory


def main(argv=None):
    args = parser().parse_args(argv)
    tools = find_tools()
    if args.check:
        import scipy
        import matplotlib
        print(f"Python {sys.version.split()[0]}; NumPy {np.__version__}; SciPy {scipy.__version__}; Matplotlib {matplotlib.__version__}")
        for name, path in tools.items():
            print(f"{name}: {path}")
        print("Инструменты доступны.")
        return 0
    if args.refine_only and args.map_json:
        raise ValueError("Параметры --refine-only и --map-json нельзя использовать одновременно.")
    settings = Settings(anchor_step=args.step, search_radius=args.search_radius)
    if bool(args.reference) != bool(args.target):
        raise ValueError("Укажите одновременно --reference и --target либо выберите обе дорожки в меню.")
    if args.reference:
        reference, target = probe(args.reference, args.reference_stream), probe(args.target, args.target_stream)
    else:
        if args.reference_stream is not None or args.target_stream is not None:
            raise ValueError("Для индексов потоков нужно указать пути --reference и --target.")
        reference, target = select_pair(args.directory)
    if reference.path == target.path or reference.path.samefile(target.path):
        raise ValueError("Эталон и исправляемая дорожка должны быть разными файлами.")
    print("\nЭталон:", describe(reference))
    print("Исправляемая дорожка:", describe(target))
    report, directory = synchronize(reference, target, args, settings)
    print("\n" + report["status"] + ": " + str(report["output"] or report.get("error", "Новая дорожка не создана.")))
    print("Отчёт и графики:", directory)
    return 0 if report["status"] in {"VERIFIED_PCM", "NO_CONFIRMED_IMPROVEMENT"} else 2
