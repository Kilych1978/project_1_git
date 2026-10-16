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
from pcm_core.guard import (gate_micro, same_map, verify_render, render_verdict,
                            cancel_unsafe_components, verify_candidate_pcm)
from pcm_core.micro import collect, Measurement
from pcm_core.model import TimeMap
from pcm_core.render import render


class _ProgressCallback:
    def __init__(self, title, seconds=False, timings=None):
        self.progress = Progress()
        self.title = title
        self.base_title = title
        self.last = 0.
        self.seconds = seconds
        self.started = time.monotonic()
        self.timings = timings
        self.progress.stage(title)

    def begin_pass(self, index, total):
        self.title = f'{self.base_title} | проход {index}/{total}'
        self.progress = Progress()
        self.last = 0.

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
        if self.timings is not None:
            self.timings.append({'stage': self.title, 'elapsed_sec': time.monotonic()-self.started})


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


def _plot_pcm_overlays(reference, baseline, final, rows, output_dir, sample_rate):
    """Actual sample waveforms, same reference timeline, before and after."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    usable = sorted([r for r in rows if r['status'] == 'VERIFIED' and r['micro_changed']],
                    key=lambda r: r['reference_time'])
    if not usable:
        return
    chosen = [usable[i] for i in sorted(set(np.linspace(0, len(usable)-1, 3).astype(int)))]
    fig, axes = plt.subplots(len(chosen), 2, figsize=(16, 3*len(chosen)), squeeze=False)
    count = round(.040 * sample_rate)
    for row_axes, row in zip(axes, chosen):
        begin = round(row['reference_time'] * sample_rate - count/2)
        if begin < 0 or begin+count > min(len(reference), len(baseline), len(final)):
            continue
        times = (np.arange(count)/sample_rate - count/(2*sample_rate))*1000
        ref = np.asarray(reference[begin:begin+count], float)/32768
        before = np.asarray(baseline[begin:begin+count], float)/32768
        after = np.asarray(final[begin:begin+count], float)
        for ax, values, label, error in zip(row_axes, [before, after], ['До уточнения', 'После уточнения'],
                                           [row['baseline_residual_sec'], row['residual_sec']]):
            ax.plot(times, ref, color='#333333', lw=1, label='Эталон')
            ax.plot(times, values, color='#d04b4b' if label.startswith('До') else '#2878c8',
                    lw=.9, alpha=.8, label=label)
            ax.set_title(f"{row['reference_time']:.3f} с | {label} | остаток {abs(error)*1000:.6f} мс")
            ax.set_xlabel('Время относительно центра окна, мс')
            ax.grid(alpha=.25)
            ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(output_dir/'15_pcm_waveform_overlays.jpg', dpi=160)
    plt.close(fig)


def export_diagnostic_clips(reference, baseline, output_dir, sample_rate=48000, times=None, final=None):
    """Small mono PCM excerpts, same absolute timeline; never upload them."""
    duration = min(len(reference), len(baseline)) / sample_rate
    if duration < 5:
        return None
    clip_length = min(8., duration)
    margin = clip_length / 2
    centers = np.linspace(max(margin, duration*.1), min(duration - margin, duration*.9), 6).tolist()
    if duration < clip_length * 3:
        centers = [duration / 2]
    centers += list(times or [])[:4]
    centers = sorted(set(round(float(t), 3) for t in centers if margin <= t <= duration - margin))
    clip_dir = output_dir / '_diagnostic_clips'
    clip_dir.mkdir(exist_ok=True)
    manifest = {'sample_rate': sample_rate, 'channels': 1,
                'timeline': 'reference and coarse-rendered target, same absolute start time',
                'clips': []}
    for index, center in enumerate(centers, 1):
        start = round((center - margin) * sample_rate)
        count = round(clip_length * sample_rate)
        entry = {'start_sec': start / sample_rate, 'duration_sec': clip_length, 'files': {}}
        streams = [('reference', reference), ('coarse_target', baseline)]
        if final is not None:
            streams.append(('final_target', final))
        for name, pcm in streams:
            path = clip_dir / f'{index:02d}_{name}.wav'
            with wave.open(str(path), 'wb') as file:
                file.setnchannels(1)
                file.setsampwidth(2)
                file.setframerate(sample_rate)
                values = np.asarray(pcm[start:start + count])
                if np.issubdtype(values.dtype, np.floating):
                    values = np.clip(values * 32768, -32768, 32767)
                file.writeframes(values.astype('<i2').tobytes())
            entry['files'][name] = path.name
        manifest['clips'].append(entry)
    (clip_dir / 'CLIPS.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    destination = output_dir / 'DIAGNOSTIC_CLIPS.zip'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(clip_dir.iterdir()):
            archive.write(path, path.name)
    return destination


def _analyze_candidate(reference, baseline, base, settings, report, progress):
    duration = len(baseline) / settings.analysis_rate
    callback = progress('ПОИСК ОСТАТОЧНОГО СДВИГА ПО PCM-ОТСЧЁТАМ')
    measurements = collect(reference, baseline, base, settings, log=lambda _: None, progress=callback)
    callback.finish()
    report['measurements'] = [m.to_dict() for m in measurements]
    callback = progress('НЕЗАВИСИМАЯ ПРОВЕРКА МИКРОПОПРАВОК')
    fine, holdouts, audit = gate_micro(reference, baseline, base, measurements,
                                      settings, log=lambda _: None, progress=callback)
    callback.finish()
    def value(rows, step):
        independent = sorted([r for r in rows if r.get('probe_kind') != 'anchor'], key=lambda r: r['target_time'])
        if not independent:
            return 0., 0.
        times = np.array([r['target_time'] for r in independent])
        borders = np.r_[0., (times[:-1] + times[1:]) / 2, duration]
        weights = np.maximum(0., np.diff(borders))
        precise = gain = 0.
        for row, weight in zip(independent, weights):
            if row['status'] != 'IMPROVED':
                continue
            remaining = abs(row['after_residual_sec'])
            improvement = max(0., abs(row['before_residual_sec']) - remaining)
            precise += improvement * weight / (1 + remaining * settings.analysis_rate)
            gain += improvement * weight
        return precise, gain
    best = (fine, holdouts, measurements, settings)
    best_value = value(holdouts, settings.anchor_step) if not same_map(base, fine) else (0., 0.)
    steps = [settings.anchor_step]
    # Refine temporal sampling when a curved local residual survives the
    # first pass. Recompute estimates and independent windows at each step.
    for step in (2., 1.):
        if step >= settings.anchor_step:
            continue
        evidence = holdouts if holdouts else audit
        needs_dense = any(r.get('before_residual_sec') is not None
                          and abs(r.get('after_residual_sec') or 0.) > 2 / settings.analysis_rate
                          and abs(r['before_residual_sec']) > .0005 for r in evidence)
        needs_dense = needs_dense or sum(
            m.status == 'MEASURED' and abs(m.correction_sec) > .0005
            and abs(float(fine.forward([m.target_time])[0] - m.target_time)) < .9 * abs(m.correction_sec)
            for m in measurements) >= 3
        if not needs_dense:
            break
        settings = replace(settings, anchor_step=step)
        steps.append(step)
        callback = progress('УТОЧНЕНИЕ УЧАСТКОВ С МЕНЯЮЩИМСЯ ОСТАТОЧНЫМ СДВИГОМ')
        measurements = collect(reference, baseline, base, settings, log=lambda _: None, progress=callback)
        callback.finish()
        callback = progress('НЕЗАВИСИМАЯ ПРОВЕРКА УТОЧНЁННОЙ СЕТКИ PCM')
        fine, holdouts, next_audit = gate_micro(reference, baseline, base, measurements,
                                               settings, log=lambda _: None, progress=callback)
        callback.finish()
        audit += next_audit
        score = value(holdouts, settings.anchor_step) if not same_map(base, fine) else (0., 0.)
        if score > best_value:
            best = (fine, holdouts, measurements, settings)
            best_value = score
    fine, holdouts, measurements, settings = best
    verification_schedules = [(settings, None)]
    # Zoom only on unresolved, measurable regions. The already verified
    # map is the baseline; a failed zoom cannot discard it or affect other
    # parts of the movie. Extra speed changes require stronger PCM proof.
    hotspots = [r['target_time'] for r in holdouts if r['status'] == 'IMPROVED'
                and r.get('after_residual_sec') is not None
                and abs(r['after_residual_sec']) > .0001
                and r.get('before_residual_sec') is not None
                and abs(r['before_residual_sec']) > .0001]
    zones = []
    for t in sorted(hotspots):
        lo, hi = max(.25, t - 1.5), min(duration - .25, t + 1.5)
        if zones and lo <= zones[-1][1]:
            zones[-1] = (zones[-1][0], max(hi, zones[-1][1]))
        else:
            zones.append((lo, hi))
    if zones:
        zoom = replace(settings, anchor_step=.125, max_rate_change=.08)
        centers, probes, barriers = [], [], []
        for lo, hi in zones:
            values = np.arange(lo + zoom.anchor_step / 2, hi, zoom.anchor_step)
            centers.extend(values.tolist())
            probes.extend(np.arange(lo, hi + 1e-9, zoom.anchor_step / 2).tolist())
            barriers += [Measurement(lo, float(fine.forward([lo])[0]), None, method='analysis-boundary'),
                         Measurement(hi, float(fine.forward([hi])[0]), None, method='analysis-boundary')]
        callback = progress('ТОЧЕЧНОЕ УТОЧНЕНИЕ ОСТАВШИХСЯ ПЕРЕХОДОВ')
        local_rows = collect(reference, baseline, fine, zoom, log=lambda _: None,
                             progress=callback, centers=centers)
        callback.finish()
        local_rows = sorted([*local_rows, *barriers], key=lambda m: m.target_time)
        callback = progress('НЕЗАВИСИМАЯ ПРОВЕРКА ТОЧЕЧНЫХ ПОПРАВОК')
        candidate, zoom_rows, zoom_audit = gate_micro(
            reference, baseline, fine, local_rows, zoom, log=lambda _: None,
            progress=callback, probe_times=probes, strong_fast=True)
        callback.finish()
        audit.extend(zoom_audit)
        report['local_zones'] = zones
        report['local_measurements'] = [m.to_dict() for m in local_rows]
        if not same_map(fine, candidate):
            fine = candidate
            verification_schedules.append((zoom, probes))
            settings = zoom
            holdouts += zoom_rows
            report['local_zoom_applied'] = True
    report['selected_step_sec'] = settings.anchor_step
    report['analysis_steps_sec'] = steps
    report['measurements'] = [m.to_dict() for m in measurements]
    report.update(independent_windows=[r for r in holdouts if r.get('probe_kind') != 'anchor'],
                  anchor_validation_windows=[r for r in holdouts if r.get('probe_kind') == 'anchor'],
                  audit=audit, fine_map=fine.to_dict())
    return fine, holdouts, measurements, settings, audit, verification_schedules


def refine_rendered_audio(selected, analysis, coarse_map, render_result, output_dir,
                          settings=None, make_plots=True, resume_report=None):
    started = time.monotonic()
    settings = settings or Settings(anchor_step=1., min_improvement=1. / 48000, max_rate_change=.02)
    fine_dir = output_dir / '_pcm_micro'
    fine_dir.mkdir(parents=True, exist_ok=False)
    baseline_path = Path(render_result['output_path'])
    report = {'status': 'BASELINE_KEPT', 'baseline_file': str(baseline_path),
              'output_file': str(baseline_path), 'sample_rate': settings.analysis_rate,
              'independent_windows': [], 'actual_output_windows': [], 'audit': [],
              'coarse_algorithm': 'original v6.9.1', 'measurements': [], 'stage_timings_sec': []}
    def progress(title, seconds=False):
        return _ProgressCallback(title, seconds, report['stage_timings_sec'])
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
        reference_duration = len(reference) / settings.analysis_rate
        output_duration = reference_duration if abs(reference_duration - duration) <= .1 else duration
        base = TimeMap.affine(duration)
        report['baseline_duration'] = duration
        report['expected_output_duration'] = output_duration
        if resume_report is None:
            fine, holdouts, measurements, settings, audit, verification_schedules = _analyze_candidate(
                reference, baseline, base, settings, report, progress)
        else:
            fine = TimeMap.from_dict(resume_report['fine_map'])
            if abs(fine.pieces[-1].target[-1] - duration) > 1/settings.analysis_rate:
                raise ValueError('Длительность сохранённой карты не совпадает с PCM.')
            settings = replace(settings, anchor_step=float(resume_report['selected_step_sec']),
                               max_rate_change=.08 if resume_report.get('local_zoom_applied') else .02)
            fine.validate(settings.max_rate_change)
            measurements = [Measurement(**m) for m in resume_report['measurements']]
            holdouts = resume_report['independent_windows'] + resume_report.get('anchor_validation_windows', [])
            schedules = {}
            for row in resume_report.get('actual_output_windows', []):
                step = float(row.get('analysis_step_sec', settings.anchor_step))
                schedules.setdefault(step, set()).add(float(row['target_time']))
            if not schedules:
                schedules[settings.anchor_step] = {m.target_time for m in measurements} | {r['target_time'] for r in holdouts}
            verification_schedules = [(replace(settings, anchor_step=step), sorted(times))
                                      for step, times in schedules.items()]
            audit = report['audit']
            report.update(measurements=[m.to_dict() for m in measurements],
                          independent_windows=resume_report['independent_windows'],
                          anchor_validation_windows=resume_report.get('anchor_validation_windows', []),
                          selected_step_sec=settings.anchor_step, analysis_steps_sec=[],
                          resumed_candidate=True, previous_status=resume_report['status'])
        # Verify the EXACT waveform and acceptance rules that will be used on
        # the decoded WAV before spending time rendering every audio channel.
        # Reuse unchanged probes when local cancellation changes only a splice.
        preflight_cache, baseline_cache = {}, {}
        original_candidate = fine
        original_bad = []
        for attempt in range(4):
            if attempt:
                strategy = ('local', 'clock_rate', 'whole_component')[attempt-1]
                fine, removed = cancel_unsafe_components(base, original_candidate,
                    [r['target_time'] for r in original_bad], settings.max_rate_change,
                    whole_components=attempt == 3,
                    rollback_rate=settings.max_rate_change if attempt == 1 else .001)
                audit.append({'status': 'PREFLIGHT_LOCAL_CANCEL', 'attempt': attempt,
                              'strategy': strategy, 'failed_windows': original_bad,
                              'removed_components': removed})
            if same_map(base, fine):
                if not original_bad:
                    break
                continue
            callback = progress(f'ПРОВЕРКА PCM ПЕРЕД ЗАПИСЬЮ — {attempt+1}/4')
            predicted = []
            for schedule, probes in verification_schedules:
                rows = verify_candidate_pcm(reference, baseline, base, fine, schedule,
                    progress=callback, target_times=probes, row_cache=preflight_cache,
                    baseline_cache=baseline_cache)
                predicted += [{**r, 'analysis_step_sec': schedule.anchor_step} for r in rows]
            callback.finish()
            report['preflight_windows'] = predicted
            preflight_verdict = render_verdict(predicted)
            if preflight_verdict == 'VERIFIED_PCM':
                break
            bad = [r for r in predicted if r['status'] in {'REGRESSION', 'RESIDUAL_TOO_LARGE'}
                   or (r['micro_changed'] and r['status'] == 'UNVERIFIABLE')]
            if not bad or attempt == 3:
                fine = base
                break
            if attempt == 0:
                original_bad = bad
            # Every alternative starts from the original curve and original
            # failures. Never propagate defects introduced by a failed taper
            # into distant previously verified regions.
        report['fine_map'] = fine.to_dict()
        if same_map(base, fine):
            report['reason'] = 'Надёжное улучшение не подтверждено; основной результат сохранён.'
        else:
            # Fine output is lossless float32 WAV/RF64, all channels in source order.
            info = probe(baseline_path, selected.target.stream_index if render_result.get('copied_without_reencode') else 0)
            info.duration = duration
            # A rejected local transition must not erase independently safe
            # components. Every retry starts from the ORIGINAL baseline, then
            # checks the decoded WAV at ALL previously selected probe times.
            for attempt in range(3):
                decoding = progress('ДЕКОДИРОВАНИЕ ВСЕХ КАНАЛОВ ДЛЯ МИКРОУТОЧНЕНИЯ', seconds=True)
                rendering = progress('ПРИМЕНЕНИЕ ПОДТВЕРЖДЁННЫХ МИКРОПОПРАВОК', seconds=True)
                candidate_path = render(info, fine, output_duration, fine_dir, settings, log=lambda _: None,
                                        progress=rendering, decode_progress=decoding)
                rendering.finish()
                candidate_info = probe(candidate_path)
                if (candidate_info.channels != selected.target.channels
                        or candidate_info.sample_rate != selected.target.sample_rate
                        or abs(candidate_info.duration - output_duration) > 1 / selected.target.sample_rate):
                    raise RuntimeError('Микроуточнение изменило формат или длительность дорожки.')
                callback = progress('ПРОВЕРКА PCM ГОТОВОГО ФАЙЛА', seconds=True)
                actual = decode(candidate_info, fine_dir / 'candidate_first.f32',
                                sample_rate=settings.analysis_rate, first_channel=True, progress=callback)
                callback.finish()
                # Correlation is scale independent: baseline S16 vs float WAV.
                callback = progress('СРАВНЕНИЕ ГОТОВОГО ФАЙЛА С ОСНОВНЫМ РЕЗУЛЬТАТОМ')
                actual_rows = []
                for schedule, probes in verification_schedules:
                    checked = verify_render(reference, baseline, actual, base, fine, schedule,
                                            progress=callback, target_times=probes)
                    actual_rows += [{**r, 'analysis_step_sec': schedule.anchor_step} for r in checked]
                callback.finish()
                verdict = render_verdict(actual_rows)
                report.update(actual_output_windows=actual_rows, candidate_verdict=verdict)
                if verdict != 'REJECTED' or attempt == 2:
                    break
                bad = [r for r in actual_rows if r['status'] in {'REGRESSION', 'RESIDUAL_TOO_LARGE'}
                       or (r['micro_changed'] and r['status'] == 'UNVERIFIABLE')]
                if not bad:
                    break
                reduced, removed = cancel_unsafe_components(base, fine, [r['target_time'] for r in bad],
                                                            settings.max_rate_change)
                if not removed or same_map(base, reduced):
                    break
                audit.append({'status': 'ACTUAL_LOCAL_CANCEL', 'attempt': attempt + 1,
                              'removed_components': removed, 'failed_windows': bad})
                actual._mmap.close()
                actual = None
                (fine_dir / 'candidate_first.f32').unlink(missing_ok=True)
                candidate_path.unlink()
                candidate_path = None
                fine = reduced.validate(settings.max_rate_change)
            report['fine_map'] = fine.to_dict()
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
                ambiguous = [m.target_time for m in measurements if m.status == 'INCONSISTENT' and m.score >= .55]
                if len(ambiguous) > 4:
                    ambiguous = [ambiguous[i] for i in np.linspace(0, len(ambiguous) - 1, 4).astype(int)]
                final_pcm = actual if report['status'] == 'PCM_REFINED' else baseline
                clips = export_diagnostic_clips(reference, baseline, output_dir, settings.analysis_rate, ambiguous, final=final_pcm)
                report['diagnostic_clips'] = str(clips) if clips else None
                if make_plots and report['status'] == 'PCM_REFINED':
                    _plot_pcm_overlays(reference, baseline, actual, actual_rows, output_dir, settings.analysis_rate)
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
    report['remaining_windows'] = [r for r in actual_rows if r['status'] == 'VERIFIED'
                                   and r['residual_sec'] is not None and abs(r['residual_sec']) > .0001]
    report['elapsed_sec'] = time.monotonic()-started
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
        text.insert(9, f"Наибольший остаточный сдвиг в этих окнах: {max(abs(r['residual_sec']) for r in improved) * 1000:.6f} мс")
        text.insert(10, f"Проверенных окон с остатком больше 0,1 мс: {len(report['remaining_windows'])}")
    (output_dir / 'PCM_MICRO_REPORT_RU.txt').write_text('\n'.join(text) + '\n', encoding='utf-8-sig')
    bundle = output_dir / 'REPORT_BUNDLE.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and (path.name.startswith('PCM_MICRO_') or path.name in {
                    'DIAGNOSTIC_CLIPS.zip', '14_pcm_micro_validation.jpg', '15_pcm_waveform_overlays.jpg'}):
                archive.write(path, path.name)
    print()
    print(report['reason'])
    return coarse_map, render_result, report
