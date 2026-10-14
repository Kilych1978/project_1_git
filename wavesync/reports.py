"""Russian reports and plots; overlays use the decoded output, never a hypothetical shift."""
import csv
import json
from pathlib import Path
import numpy as np
from .dsp import mapped_window


def json_write(path, value):
    def plain(x):
        if isinstance(x, dict):
            return {k: plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [plain(v) for v in x]
        if isinstance(x, np.ndarray):
            return plain(x.tolist())
        if isinstance(x, np.generic):
            return plain(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        if isinstance(x, Path):
            return str(x)
        return x
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv_write(path, rows):
    if not rows:
        path.write_text("Нет измерений.\n", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


VERDICTS = {
    "VERIFIED_PCM": "Результат проверен по PCM на контрольных участках.",
    "VERIFIED_ENVELOPE_ONLY": "Проверено крупное выравнивание по огибающей. Точность по отсчётам не подтверждена; требуется прослушивание.",
    "NO_CONFIRMED_IMPROVEMENT": "Надёжное улучшение не подтверждено. Новая дорожка не создана; используйте исходную.",
    "REJECTED": "Результат не прошёл проверку. Файл не выдан как синхронизированный.",
    "BLOCKED": "Синхронизация остановлена: недостаточно надёжных данных или требуется ручная граница монтажа.",
}


def write_report(directory, report):
    json_write(directory / "REPORT.json", report)
    checks = report.get("output_pcm_validation", [])
    residuals = [abs(r["residual_sec"]) * 1000 for r in checks if r["status"] == "VERIFIED"]
    lines = ["UNIVERSAL AUDIO WAVEFORM SYNC 8.0", "", VERDICTS.get(report["status"], report["status"]), "",
             "Эталон: " + report["reference"]["path"], "Исходная дорожка: " + report["target"]["path"],
             "Результат: " + str(report.get("output") or "не создан"), "",
             "Анализ: только первый канал выбранного звукового потока.",
             "Рендер: все каналы исходной дорожки, с одной общей картой времени.",
             "Исходные файлы не изменяются. Выход: WAV/RF64, 32-битный PCM с плавающей точкой.",
             "Изменение скорости ресемплированием меняет и высоту тона; отдельного сохранения высоты тона здесь нет.", "",
             f"Измерений микропоиска: {len(report.get('micro_measurements', []))}.",
             f"Независимых проверок фактического выхода: {len(checks)}.",
             f"Участков с подтверждённым остатком: {len(residuals)}.",
             f"Неподтверждённых участков: {sum(r['status'] == 'UNVERIFIABLE' for r in checks)}."]
    if residuals:
        lines += [f"Медиана абсолютного остатка: {np.median(residuals):.6f} мс.",
                  f"Максимум на подтверждённых участках: {max(residuals):.6f} мс."]
    if report.get("error"):
        lines += ["", "Причина остановки: " + report["error"]]
    if report.get("plot_error"):
        lines += ["", "Не удалось завершить графики: " + report["plot_error"]]
    for gap in report.get("edit_gaps", []):
        if gap["reference_gap_sec"] > 1e-9:
            lines += [f"Разрыв монтажа по эталону: {gap['reference_start']:.3f}–{gap['reference_end']:.3f} с; заполнен тишиной."]
        if gap["target_gap_sec"] > 1e-9:
            lines += [f"Исключён фрагмент исходной дорожки: {gap['target_start']:.3f}–{gap['target_end']:.3f} с."]
    for start, end in report.get("reference_padding", []):
        lines += [f"Вне доступной исходной дорожки: {start:.6f}–{end:.6f} с по эталону; заполнено тишиной."]
    lines += ["", "Проверка выборочная. Она не доказывает отсутствие ошибки между контрольными точками.",
              "Разные переводы, миксы, эхо, кодеки и монтаж ограничивают измеримую точность.",
              "Один отсчёт при 48 кГц — 20,833 мкс. Дробный лаг — оценка, а не гарантия точности 1 мкс.",
              "CSV: target_time — время исходной дорожки; reference_time — соответствующее время эталона;",
              "correction_sec — поправка к карте; residual_sec — остаток после применения карты.",
              "UNVERIFIABLE означает отсутствие надёжного измерения; это не нулевая ошибка.",
              "Графики микросовмещения нормированы только для отображения; громкость выходного файла не нормируется."]
    (directory / "REPORT_RU.txt").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def plots(directory, reference, target, ref_env, tar_env, base, final, measurements, validations, actual, settings, output_checks=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": .2})
    images = directory / "graphs"
    images.mkdir(exist_ok=True)
    sr = settings.analysis_rate
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(images / name, dpi=130)
        plt.close(fig)
    def overview(pcm):
        hop = max(1, len(pcm) // 12000)
        starts = np.arange(0, len(pcm), hop)
        low, high = [], []
        for start in starts:
            block = pcm[start:start + hop]
            low.append(float(np.min(block)))
            high.append(float(np.max(block)))
        return starts / sr, low, high
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    for ax, data, label in zip(axes, (reference, target), ("Эталон — первый канал", "Исходная дорожка — первый канал")):
        t, lo, hi = overview(data)
        ax.fill_between(t, lo, hi, alpha=.65)
        ax.set(title=label, ylabel="Амплитуда", xlabel="Время, с")
    save(fig, "01_WAVEFORMS.jpg")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    for ax, env, label in zip(axes, (ref_env, tar_env), ("Огибающая эталона", "Огибающая исходной дорожки")):
        step = max(1, len(env) // 16000)
        ax.plot(np.arange(0, len(env), step) * settings.envelope_hop, env[::step], linewidth=.6)
        ax.set(title=label, xlabel="Время, с", ylabel="Огибающая")
    save(fig, "02_ENVELOPES.jpg")
    if base is None or final is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for mapping, label in [(base, "Крупная карта"), (final, "Итоговая карта")]:
        for k, p in enumerate(mapping.pieces):
            axes[0].plot(p.target, (p.reference - p.target) * 1000, label=label if k == 0 else None)
    for p in final.pieces:
        ts = np.unique(np.r_[p.target, np.linspace(p.target[0], p.target[-1], 1000)])
        axes[1].plot(ts, (final.forward(ts) - base.forward(ts)) * 1000)
    axes[0].set(ylabel="Смещение, мс", title="Крупная и итоговая карты времени")
    axes[0].legend()
    axes[1].set(xlabel="Время исходной дорожки, с", ylabel="Микропоправка, мс")
    save(fig, "03_TIME_MAP_AND_MICRO.jpg")
    fig, ax = plt.subplots(figsize=(14, 5))
    for p in final.pieces:
        ax.step(p.target[:-1], p.rate, where="post")
    ax.set(xlabel="Время исходной дорожки, с", ylabel="Скорость карты", title="Изменение скорости: 1 — без растяжения")
    save(fig, "04_RATE_AND_DRIFT.jpg")
    rows = [m.to_dict() if hasattr(m, "to_dict") else m for m in measurements]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for status, color, label in [("MEASURED", "green", "Надёжно измерено"), ("UNVERIFIABLE", "gray", "Не подтверждено"), ("INCONSISTENT", "orange", "Окна расходятся")]:
        selected = [r for r in rows if r["status"] == status]
        if selected:
            axes[0].scatter([r["target_time"] for r in selected], [r["score"] for r in selected], s=10, color=color, label=label)
    axes[0].legend()
    axes[0].set(ylabel="Сходство PCM", title="Достоверность микропоиска")
    selected = [r for r in rows if r["correction_sec"] is not None]
    axes[1].scatter([r["target_time"] for r in selected], [r["correction_sec"] * 1000 for r in selected], s=10)
    axes[1].set(xlabel="Время исходной дорожки, с", ylabel="Измеренная поправка, мс")
    save(fig, "05_CONFIDENCE_AND_CANDIDATES.jpg")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    valid = [r for r in validations if r.get("before_residual_sec") is not None and r.get("after_residual_sec") is not None]
    axes[0].plot([r["target_time"] for r in valid], [abs(r["before_residual_sec"]) * 1000 for r in valid], "o-", markersize=3, label="До уточнения")
    axes[0].plot([r["target_time"] for r in valid], [abs(r["after_residual_sec"]) * 1000 for r in valid], "o-", markersize=3, label="После уточнения")
    axes[0].set(xlabel="Время исходной дорожки, с", ylabel="Остаток, мс", title="Проверка на отдельных контрольных окнах")
    axes[0].legend()
    values = [abs(r["after_residual_sec"]) * 1000 for r in valid]
    if values:
        axes[1].hist(values, bins=min(30, max(5, len(values))), alpha=.7)
    axes[1].set(xlabel="Абсолютный остаток, мс", ylabel="Число точек", title="Распределение подтверждённых остатков")
    save(fig, "06_BEFORE_AFTER_AND_DISTRIBUTION.jpg")
    if actual is None:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    actual_rows = [(r["reference_time"], r["residual_sec"] * 1000) for r in (output_checks or []) if r.get("residual_sec") is not None]
    if actual_rows:
        ax.scatter(*np.array(actual_rows).T, s=12)
    ax.set(xlabel="Время эталона, с", ylabel="Остаток, мс", title="Измеренные остатки декодированного выходного файла")
    save(fig, "07_ACTUAL_OUTPUT_RESIDUALS.jpg")
    chosen = sorted(selected, key=lambda r: abs(r["correction_sec"]), reverse=True)[:settings.max_micro_plots]
    micros = images / "micro_overlays"
    micros.mkdir(exist_ok=True)
    for i, row in enumerate(chosen):
        t = row["target_time"]
        center = float(final.forward([t])[0])
        length = round(.04 * sr)
        before, start = mapped_window(target, base, center, length, sr, settings.sinc_radius)
        if before is None or start < 0 or start + length > min(len(reference), len(actual)):
            continue
        ref = reference[start:start + length]
        after = actual[start:start + length]
        def norm(x):
            x = np.asarray(x, dtype=np.float64)
            x = x - x.mean()
            return x / max(np.percentile(np.abs(x), 98), 1e-10)
        xx = (np.arange(length) - length / 2) / sr * 1000
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        axes[0].plot(xx, norm(ref), label="Эталон", linewidth=.7)
        axes[0].plot(xx, norm(before), label="До микропоправки", linewidth=.7, alpha=.8)
        axes[1].plot(xx, norm(ref), label="Эталон", linewidth=.7)
        axes[1].plot(xx, norm(after), label="Фактический выходной файл", linewidth=.7, alpha=.8)
        for ax in axes:
            ax.legend()
            ax.set(ylabel="Нормированная амплитуда")
        applied = float(final.forward([t])[0] - base.forward([t])[0]) * 1000
        axes[0].set_title(f"Контрольный участок: {center:.3f} с по эталону; применённая микропоправка {applied:+.6f} мс")
        axes[1].set_xlabel("Относительное время, мс")
        fig.tight_layout()
        fig.savefig(micros / f"{i + 1:03d}_{center:.3f}s.jpg", dpi=145)
        plt.close(fig)
