"""Check the complete sampling stack against both source and protected PCM.

Incremental gates are useful for building a candidate, but their results are
not transitive: a later layer can reveal an unsafe part of an earlier layer.
No multichannel WAV is written until this end-to-end check has passed.
"""
from dataclasses import replace

from .guard import verify_candidate_pcm, same_map
from .choice import verify_reference_grid
from .recovery import restore_failures


def failed(rows):
    return [r for r in rows if r['status'] == 'REGRESSION'
            or (r['micro_changed'] and r['status'] == 'UNVERIFIABLE')]


def settle(reference, target, source, candidate, fallback, schedules, settings,
           reference_times=(), progress=None, protected=False):
    """Restore failed whole windows and verify every newly exposed boundary.

    Each retry changes only a reference-local amplitude mask. Its padding
    grows when a splice keeps failing, instead of moving it by 12 ms forever.
    Cached PCM is reused only when the complete local sampling plan matches.
    """
    source_cache, source_before, protected_cache, protected_before = {}, {}, {}, {}
    fixed_cache, fixed_before = {}, {}
    history = []
    original = candidate
    for attempt in range(12):
        rows = []
        for index, (schedule, times) in enumerate(schedules, 1):
            callback = progress(f'СКВОЗНАЯ ПРОВЕРКА PCM | проход {attempt+1}/12 | исходник | сетка {index}/{len(schedules)}') if progress else None
            rows += verify_candidate_pcm(reference, target, source, candidate, schedule,
                target_times=times, row_cache=source_cache, baseline_cache=source_before, progress=callback)
            if callback: callback.finish()
            if protected:
                callback = progress(f'СКВОЗНАЯ ПРОВЕРКА PCM | проход {attempt+1}/12 | сохранность базы | сетка {index}/{len(schedules)}') if progress else None
                rows += verify_candidate_pcm(reference, target, fallback, candidate, schedule,
                    target_times=times, row_cache=protected_cache, baseline_cache=protected_before, progress=callback)
                if callback: callback.finish()
        if protected:
            for step in (.125, .5):
                callback = progress(f'СКВОЗНАЯ ПРОВЕРКА PCM | проход {attempt+1}/12 | время эталона | окно {step/8*1000:g} мс') if progress else None
                rows += verify_reference_grid(reference, target, fallback, candidate,
                    replace(settings, anchor_step=step), reference_times,
                    row_cache=fixed_cache, baseline_cache=fixed_before, progress=callback)
                if callback: callback.finish()
        bad = failed(rows)
        improved = {round(r['reference_time'], 9) for r in rows if r['micro_changed']
                    and r['status'] == 'VERIFIED' and r['baseline_residual_sec'] is not None
                    and r['residual_sec'] is not None
                    and abs(r['residual_sec'])+settings.min_improvement < abs(r['baseline_residual_sec'])}
        history.append({'attempt': attempt+1, 'failed': len(bad), 'improved': len(improved),
                        'failed_windows': bad})
        if not bad:
            return candidate, rows, history
        if attempt == 11: break
        padding = 0. if attempt == 0 else min(8., .0625*2**(attempt-1))
        candidate = restore_failures(candidate, fallback, bad, padding=padding)
        history[-1]['padding_sec'] = padding
        if same_map(original, candidate): break
    # Caller will still verify this fallback on the native written WAV.
    history.append({'status': 'COMPLETE_PLAN_FALLBACK', 'reason': 'Local recovery budget exhausted.'})
    return fallback, [], history
