"""Fine residual correction after the original v6.9.1 render/guard.

This module never builds a coarse map and cannot invalidate a good coarse
result. It compares the actual rendered baseline with a lossless candidate.
"""
from __future__ import annotations

import csv
from dataclasses import replace
import json
import time
import wave
import zipfile
from pathlib import Path

import numpy as np

from common import Progress, format_seconds
from pcm_core.audio import probe, decode
from pcm_core.config import Settings
from pcm_core.guard import gate_micro, same_map, verify_render, render_verdict
from pcm_core.micro import collect
from pcm_core.model import TimeMap
from pcm_core.render import render


class _ProgressCallback:
    def __init__(self, title, seconds=False):
        self.progress = Progress()
        self.title = title
        self.last = 0.
        self.seconds = seconds
        self.progress.stage(title)

    def __call__(self, done, total):
        now = time.monotonic()
        if now - self.last < .5 and done < total:
            return
        self.last = now
        percent = 100. * done / total if total > 0 else 0.
        eta = format_seconds(self.progress.eta(done, total))
        if self.seconds:
            self.progress.update(percent, self.title + ' | ETA ' + eta, done, total,
                                 self.progress.speed(done))
        else:
            self.progress.update(percent, f'{self.title} | {done:.0f}/{total:.0f} точек | ETA {eta}')
        if total > 0 and done >= total:
            print()

    def finish(self):
        self.progress.done(100., self.title)


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(output_dir, measurements, holdouts, actual):
    # Preserve the original graph set; add a separate graph of the PCM check.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    measured = [m for m in measurements if m.status == 'MEASURED']
    axes[0].plot([m.target_time / 60 for m in measured],
                 [m.correction_sec * 1000 for m in measured], '.', ms=3,
                 label='Остаточный сдвиг до уточнения')
    checked = [r for r in actual if r.get('residual_sec') is not None]
    axes[0].plot([r['reference_time'] / 60 for r in checked],
                 [r['residual_sec'] * 1000 for r in checked], '.', ms=3,
                 label='Остаточный сдвиг проверяемого файла')
    axes[0].set_ylabel('Сдвиг, мс')
    axes[0].legend()
    statuses = {'IMPROVED': 1, 'UNCHANGED': 0, 'UNVERIFIABLE': -1, 'REGRESSION': -2}
    axes[1].scatter([r['target_time'] / 60 for r in holdouts],
                    [statuses[r['status']] for r in holdouts], s=8)
    axes[1].set_yticks([-2, -1, 0, 1], ['Ухудшение', 'Не подтверждено', 'Без изменения', 'Улучшение'])
    axes[1].set_xlabel('Время, мин')
    axes[1].set_title('Независимые окна проверки микропоправок')
    for ax in axes:
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output_dir / '14_pcm_micro_validation.jpg', dpi=150)
    plt.close(fig)


def export_diagnostic_clips(reference, baseline, output_dir, sample_rate=48000, times=None):
    """Small mono PCM excerpts, same absolute timeline; never upload them."""
    duration = min(len(reference), len(baseline)) / sample_rate
    if duration < 5:
        return None
    centers = np.linspace(2., duration - 2., 6).tolist()
    centers += list(times or [])[:4]
    centers = sorted(set(round(float(t), 3) for t in centers if 2 <= t <= duration - 2))
    clip_dir = output_dir / '_diagnostic_clips'
    clip_dir.mkdir(exist_ok=True)
    manifest = {'sample_rate': sample_rate, 'channels': 1,
                'timeline': 'reference and coarse-rendered target, same absolute start time',
                'clips': []}
    for index, center in enumerate(centers, 1):
        start = round((center - 2.) * sample_rate)
        count = 4 * sample_rate
        entry = {'start_sec': start / sample_rate, 'duration_sec': 4., 'files': {}}
        for name, pcm in (('reference', reference), ('coarse_target', baseline)):
            path = clip_dir / f'{index:02d}_{name}.wav'
            with wave.open(str(path), 'wb') as file:
                file.setnchannels(1)
                file.setsampwidth(2)
                file.setframerate(sample_rate)
                file.writeframes(np.asarray(pcm[start:start + count], dtype='<i2').tobytes())
            entry['files'][name] = path.name
        manifest['clips'].append(entry)
    (clip_dir / 'CLIPS.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    destination = output_dir / 'DIAGNOSTIC_CLIPS.zip'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(clip_dir.iterdir()):
            archive.write(path, path.name)
    return destination


def refine_rendered_audio(selected, analysis, coarse_map, render_result, output_dir,
                          settings=None, make_plots=True):
    settings = settings or Settings(min_improvement=1. / 48000, max_rate_change=.02)
    fine_dir = output_dir / '_pcm_micro'
    fine_dir.mkdir(parents=True, exist_ok=False)
    baseline_path = Path(render_result['output_path'])
    report = {'status': 'BASELINE_KEPT', 'baseline_file': str(baseline_path),
              'output_file': str(baseline_path), 'sample_rate': settings.analysis_rate,
              'independent_windows': [], 'actual_output_windows': [], 'audit': [],
              'coarse_algorithm': 'original v6.9.1', 'measurements': []}
    reference = baseline = actual = None
    measurements, holdouts, actual_rows = [], [], []
    candidate_path = None
    original_map, original_result = coarse_map, render_result
    published_candidate = None
    try:
        reference = np.memmap(analysis['reference_pcm'], mode='r', dtype='<i2')
        baseline_pcm = render_result.get('verified_first_channel_pcm')
        if not baseline_pcm:
            # Standalone invocation / tests; production reuses the coarse guard decode.
            from analyzer import decode_first_channel_stream
            from scanner import probe_audio_file
            baseline_pcm = fine_dir / 'baseline_s16le.pcm'
            decode_first_channel_stream(probe_audio_file(baseline_path), baseline_pcm, Progress(), 0., 100.)
        baseline = np.memmap(baseline_pcm, mode='r', dtype='<i2')
        duration = len(baseline) / settings.analysis_rate
        base = TimeMap.affine(duration)
        report['baseline_duration'] = duration
        callback = _ProgressCallback('ПОИСК ОСТАТОЧНОГО СДВИГА ПО PCM-ОТСЧЁТАМ')
        measurements = collect(reference, baseline, base, settings, log=lambda _: None, progress=callback)
        callback.finish()
        report['measurements'] = [m.to_dict() for m in measurements]
        callback = _ProgressCallback('НЕЗАВИСИМАЯ ПРОВЕРКА МИКРОПОПРАВОК')
        fine, holdouts, audit = gate_micro(reference, baseline, base, measurements,
                                          settings, log=lambda _: None, progress=callback)
        callback.finish()
        steps = [settings.anchor_step]
        # Refine temporal sampling when a curved local residual survives the
        # first pass. Recompute estimates and independent windows at each step.
        for step in (2., 1.):
            if step >= settings.anchor_step:
                continue
            evidence = audit[-len(holdouts):] if holdouts else audit
            needs_dense = any(r.get('before_residual_sec') is not None
                              and abs(r.get('after_residual_sec') or 0.) > 2 / settings.analysis_rate
                              and abs(r['before_residual_sec']) > .0005 for r in evidence)
            if not needs_dense:
                break
            settings = replace(settings, anchor_step=step)
            steps.append(step)
            callback = _ProgressCallback('УТОЧНЕНИЕ УЧАСТКОВ С МЕНЯЮЩИМСЯ ОСТАТОЧНЫМ СДВИГОМ')
            measurements = collect(reference, baseline, base, settings, log=lambda _: None, progress=callback)
            callback.finish()
            callback = _ProgressCallback('НЕЗАВИСИМАЯ ПРОВЕРКА УТОЧНЁННОЙ СЕТКИ PCM')
            fine, holdouts, next_audit = gate_micro(reference, baseline, base, measurements,
                                                   settings, log=lambda _: None, progress=callback)
            callback.finish()
            audit += next_audit
        report['analysis_steps_sec'] = steps
        report['measurements'] = [m.to_dict() for m in measurements]
        report.update(independent_windows=holdouts, audit=audit, fine_map=fine.to_dict())
        if same_map(base, fine):
            report['reason'] = 'Надёжное улучшение не подтверждено; основной результат сохранён.'
        else:
            # Fine output is lossless float32 WAV/RF64, all channels in source order.
            info = probe(baseline_path, selected.target.stream_index if render_result.get('copied_without_reencode') else 0)
            info.duration = duration
            decoding = _ProgressCallback('ДЕКОДИРОВАНИЕ ВСЕХ КАНАЛОВ ДЛЯ МИКРОУТОЧНЕНИЯ', seconds=True)
            rendering = _ProgressCallback('ПРИМЕНЕНИЕ ПОДТВЕРЖДЁННЫХ МИКРОПОПРАВОК', seconds=True)
            candidate_path = render(info, fine, duration, fine_dir, settings, log=lambda _: None,
                                    progress=rendering, decode_progress=decoding)
            rendering.finish()
            candidate_info = probe(candidate_path)
            if (candidate_info.channels != selected.target.channels
                    or candidate_info.sample_rate != selected.target.sample_rate
                    or abs(candidate_info.duration - duration) > 1 / selected.target.sample_rate):
                raise RuntimeError('Микроуточнение изменило формат или длительность дорожки.')
            callback = _ProgressCallback('ПРОВЕРКА PCM ГОТОВОГО ФАЙЛА', seconds=True)
            actual = decode(candidate_info, fine_dir / 'candidate_first.f32',
                            sample_rate=settings.analysis_rate, first_channel=True, progress=callback)
            callback.finish()
            # Reference/baseline caches use S16 units; decoded float uses [-1, 1].
            # Correlation itself is scale independent, so no whole-film conversion.
            callback = _ProgressCallback('СРАВНЕНИЕ ГОТОВОГО ФАЙЛА С ОСНОВНЫМ РЕЗУЛЬТАТОМ')
            actual_rows = verify_render(reference, baseline, actual, base, fine, settings, progress=callback)
            callback.finish()
            report['actual_output_windows'] = actual_rows
            report['candidate_verdict'] = render_verdict(actual_rows)
            if report['candidate_verdict'] == 'VERIFIED_PCM':
                destination = output_dir / (selected.target.path.stem + '_SYNCED.wav')
                if destination.exists():
                    raise RuntimeError('Путь итогового файла уже занят; перезапись отменена.')
                candidate_path.rename(destination)
                candidate_path = None
                published_candidate = destination
                plan = {'source_codec': selected.target.codec_name, 'encoder': 'pcm_f32le',
                        'container': 'wav', 'extension': '.wav', 'pcm_fallback': True,
                        'reason': 'Подтверждённое PCM-микроуточнение; WAV/RF64 без потерь при кодировании.'}
                render_result = {**render_result, 'output_path': destination, 'encoder': 'pcm_f32le',
                                 'codec_plan': plan, 'copied_without_reencode': False,
                                 'coarse_output_path': baseline_path, 'pcm_refined': True}
                # Exact two-stage maps are saved separately. Dense canonical map
                # is the composition for original TARGET -> final REFERENCE.
                times = np.asarray(coarse_map['target_time'], dtype=float)
                old_reference = np.asarray(coarse_map['reference_time'], dtype=float)
                new_reference = fine.forward(old_reference)
                valid = np.isfinite(new_reference)
                new_reference[~valid] = old_reference[~valid]
                coarse_map = {**coarse_map, 'reference_time': new_reference,
                              'offset': new_reference - times}
                report.update(status='PCM_REFINED', output_file=str(destination),
                              reason='Улучшение подтверждено на независимых окнах и на записанном файле.')
            else:
                report['reason'] = 'Проверка записанного кандидата не подтвердила безопасное улучшение; основной результат сохранён.'
    except Exception as exc:
        # A local fine-analysis problem cannot destroy the verified baseline.
        coarse_map, render_result = original_map, original_result
        if published_candidate is not None:
            published_candidate.unlink(missing_ok=True)
        report.update(status='BASELINE_KEPT_ERROR', reason=str(exc))
        report['output_file'] = str(baseline_path)
    finally:
        if candidate_path is not None:
            candidate_path.unlink(missing_ok=True)
        if reference is not None and baseline is not None:
            try:
                ambiguous = [m.target_time for m in measurements if m.status != 'MEASURED']
                clips = export_diagnostic_clips(reference, baseline, output_dir, settings.analysis_rate, ambiguous)
                report['diagnostic_clips'] = str(clips) if clips else None
            except Exception as exc:
                report['diagnostic_export_error'] = str(exc)
        for pcm in (reference, baseline, actual):
            if pcm is not None:
                pcm._mmap.close()
        (fine_dir / 'candidate_first.f32').unlink(missing_ok=True)
    _write_csv(output_dir / 'PCM_MICRO_POINTS.csv', report['measurements'])
    _write_csv(output_dir / 'PCM_MICRO_HOLDOUT.csv', holdouts)
    _write_csv(output_dir / 'PCM_MICRO_ACTUAL.csv', actual_rows)
    if make_plots:
        _plot(output_dir, measurements, holdouts, actual_rows)
    (output_dir / 'PCM_MICRO_REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    improved = [r for r in actual_rows if r['micro_changed'] and r['status'] == 'VERIFIED']
    text = ['МИКРОУТОЧНЕНИЕ ПО PCM-ОТСЧЁТАМ', '=' * 78, '', report['reason'],
            f"Подтверждённых измерений: {sum(m.status == 'MEASURED' for m in measurements)} / {len(measurements)}",
            f"Независимых проверок: {len(holdouts)}", f"Проверок записанного файла: {len(actual_rows)}",
            f"Применено микроуточнение: {'да' if report['status'] == 'PCM_REFINED' else 'нет'}",
            f"Итоговый файл: {report['output_file']}", '',
            'Измерения относятся только к проверенным окнам с общим звуком.',
            'Тишина и различающийся звук не подтверждают точность синхронизации.',
            'Исходная дорожка не изменена.']
    if improved and report['status'] == 'PCM_REFINED':
        text.insert(8, f"Медианный остаточный сдвиг в проверенных изменённых окнах: {np.median([abs(r['residual_sec']) for r in improved]) * 1000:.6f} мс")
    (output_dir / 'PCM_MICRO_REPORT_RU.txt').write_text('\n'.join(text) + '\n', encoding='utf-8-sig')
    print()
    print(report['reason'])
    return coarse_map, render_result, report
