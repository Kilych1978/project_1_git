"""Report synchronization evidence separately from safe processing completion.

This module selects no map, modifies no audio and relaxes no render guard.
In particular, keeping an unchanged source proves non-regression, not alignment.
"""
import json
import math
from pathlib import Path


ALIGNMENT_TOLERANCE_SEC = .020
MIN_TRUSTED_WINDOWS = 3
MIN_COMMON_FRACTION = .60
MIN_TIME_COVERAGE = .80
TIME_BINS = 6


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _identity_plan(plan):
    """A preview of a rejected non-identity proposal is not the retained audio."""
    mapping = (plan or {}).get('fine_map', {})
    pieces = mapping.get('pieces', [])
    if not pieces or mapping.get('layers'):
        return False
    for piece in pieces:
        target, reference = piece.get('target'), piece.get('reference')
        if (not isinstance(target, list) or not isinstance(reference, list)
                or len(target) != len(reference) or len(target) < 2
                or piece.get('corrections') or piece.get('layers')):
            return False
        if any(_number(a) is None or _number(b) is None or float(a) != float(b)
               for a, b in zip(target, reference)):
            return False
    return True


def _coverage(windows, scheduled, duration):
    common_fraction = len(windows) / max(1, scheduled)
    bins = set()
    duration = _number(duration)
    if duration is not None and duration > 0:
        for item in windows:
            position = _number(item.get('time'))
            if position is not None and 0 <= position <= duration:
                bins.add(min(TIME_BINS - 1, int(position / duration * TIME_BINS)))
    time_fraction = len(bins) / TIME_BINS
    sufficient = (len(windows) >= MIN_TRUSTED_WINDOWS
                  and common_fraction >= MIN_COMMON_FRACTION
                  and time_fraction >= MIN_TIME_COVERAGE)
    return {'trusted_windows': len(windows), 'scheduled_windows': scheduled,
            'common_fraction': common_fraction, 'time_bin_fraction': time_fraction,
            'time_bins': sorted(bins), 'sufficient': sufficient}


def assess_sync_outcome(verification, pcm_report=None, *, output_created=False,
                        baseline_plan=None, render_guard=None, duration=None):
    """Assess existing, absolute evidence without turning UNKNOWN into zero lag.

    ``VERIFIED`` PCM refers to a measurable common waveform, not necessarily
    to zero delay. Bidirectional ambiguity is diagnostic, never aligned proof.
    All positive claims are limited to sampled common-audio windows.
    """
    verification = verification or {}
    pcm_report = pcm_report or {}
    render_guard = render_guard or {}
    envelope_rows = verification.get('validation_points', [])
    if duration is None:
        duration = verification.get('output_duration')
        info = verification.get('output_info')
        if duration is None and info is not None:
            duration = getattr(info, 'duration', None)

    pcm_rows = pcm_report.get('actual_output_windows', [])
    pcm_source = 'actual_output_windows'
    if not pcm_rows and _identity_plan(baseline_plan):
        pcm_rows = baseline_plan.get('native_preview_windows', [])
        pcm_source = 'identity_baseline_preview'
    settings = pcm_report.get('selected_settings') or pcm_report.get('initial_settings') or {}
    min_score = _number(settings.get('min_raw_score', .55)) or .55
    pcm = []
    improved = []
    for row in pcm_rows:
        residual = _number(row.get('residual_sec'))
        score = _number(row.get('score'))
        if row.get('status') != 'VERIFIED' or residual is None or score is None or score < min_score:
            continue
        record = {'time': row.get('reference_time', row.get('target_time')),
                  'residual_sec': residual, 'source': 'pcm'}
        pcm.append(record)
        before = _number(row.get('baseline_residual_sec'))
        if row.get('micro_changed') and before is not None and abs(residual) + .25 / 48000 < abs(before):
            improved.append(record)

    envelope = []
    ambiguous_count = 0
    weak_count = 0
    for row in envelope_rows:
        status = row.get('bidir_status')
        if status not in ('CONFIRMED_MAP', 'CONFIRMED_DISCREPANCY'):
            if status == 'CONSISTENT_BUT_WEAK':
                weak_count += 1
            else:
                ambiguous_count += 1
            continue
        residual = _number(row.get('bidir_consensus_offset'))
        left = _number(row.get('bidir_left_correlation'))
        right = _number(row.get('bidir_right_correlation'))
        disagreement = _number(row.get('bidir_disagreement'))
        if (residual is None or left is None or right is None or min(left, right) < .30
                or disagreement is None or disagreement > .080):
            continue
        envelope.append({'time': row.get('target_time'), 'residual_sec': residual,
                         'source': 'bidirectional_envelope'})

    pcm_coverage = _coverage(pcm, len(pcm_rows), duration)
    envelope_coverage = _coverage(envelope, len(envelope_rows), duration)
    pcm_mismatches = [row for row in pcm if abs(row['residual_sec']) > ALIGNMENT_TOLERANCE_SEC]
    envelope_mismatches = [row for row in envelope if abs(row['residual_sec']) > ALIGNMENT_TOLERANCE_SEC]
    mismatches = pcm_mismatches + envelope_mismatches
    pcm_bad = any(row.get('status') in ('REGRESSION', 'RESIDUAL_TOO_LARGE') for row in pcm_rows)
    verified_pcm_improvement = (pcm_report.get('status') == 'PCM_REFINED'
                                and len(improved) >= 3 and not pcm_bad)
    comparison = render_guard.get('final_comparison') or {}
    verified_coarse_improvement = (render_guard.get('passed') is True
                                   and not render_guard.get('fallback_to_baseline')
                                   and int(comparison.get('improvement_count', 0) or 0) > 0)
    improvement_verified = bool(output_created and (verified_pcm_improvement or verified_coarse_improvement))
    common_sufficient = pcm_coverage['sufficient'] or envelope_coverage['sufficient']
    alignment_confirmed = bool(common_sufficient and not mismatches and not pcm_bad)
    if mismatches or pcm_bad:
        status = 'IMPROVED_NOT_FULLY_ALIGNED' if improvement_verified else 'NOT_SYNCHRONIZED'
    elif alignment_confirmed:
        status = 'SYNCHRONIZED' if output_created else 'ALREADY_ALIGNED'
    elif improvement_verified:
        status = 'IMPROVED_UNVERIFIED'
    else:
        status = 'UNVERIFIED'

    messages = {
        'NOT_SYNCHRONIZED': ('СИНХРОНИЗАЦИЯ НЕ ДОСТИГНУТА',
            'В надёжно измеренных общих звуковых окнах остаются расхождения. '
            'Отсутствие ухудшения не означает совпадения с эталоном.'),
        'UNVERIFIED': ('СИНХРОНИЗАЦИЯ НЕ ПОДТВЕРЖДЕНА',
            'Недостаточно надёжных общих звуковых окон для подтверждения совмещения. '
            'Тишина, слабое совпадение и неизвестный сдвиг не считаются нулевым остатком.'),
        'ALREADY_ALIGNED': ('ИСХОДНАЯ ДОРОЖКА УЖЕ СОВМЕЩЕНА В ПРОВЕРЕННЫХ ОКНАХ',
            'Совмещение исходной дорожки подтверждено в распределённых по времени '
            'общих звуковых окнах. Новый аудиофайл не требуется.'),
        'SYNCHRONIZED': ('СОВМЕЩЕНИЕ ПОДТВЕРЖДЕНО В ПРОВЕРЕННЫХ ОКНАХ',
            'Готовая дорожка совмещена с эталоном в проверенных общих звуковых окнах.'),
        'IMPROVED_NOT_FULLY_ALIGNED': ('УЛУЧШЕНИЯ ПОДТВЕРЖДЕНЫ, РАССИНХРОН ОСТАЛСЯ',
            'Подтверждены локальные улучшения, но в других надёжно измеренных окнах '
            'остаётся рассинхрон. Полная синхронизация не достигнута.'),
        'IMPROVED_UNVERIFIED': ('ЛОКАЛЬНЫЕ УЛУЧШЕНИЯ ПОДТВЕРЖДЕНЫ',
            'Подтверждены локальные улучшения. Данных недостаточно, чтобы подтвердить '
            'совмещение всей проверяемой временной шкалы.'),
    }
    headline, message = messages[status]
    if not output_created and status not in ('ALREADY_ALIGNED',):
        message += ' Исходная дорожка сохранена без изменений; новый аудиофайл не создан.'
    return {'schema': 1, 'status': status, 'headline_ru': headline, 'message_ru': message,
            'alignment_confirmed': alignment_confirmed, 'improvement_verified': improvement_verified,
            'output_created': bool(output_created), 'scope': 'sampled_common_audio_windows',
            'alignment_tolerance_sec': ALIGNMENT_TOLERANCE_SEC,
            'confirmed_mismatch_count': len(mismatches),
            'largest_confirmed_mismatch_sec': max((abs(row['residual_sec']) for row in mismatches), default=None),
            'pcm_evidence_source': pcm_source, 'pcm_evidence': pcm_coverage,
            'envelope_evidence': envelope_coverage, 'ambiguous_envelope_windows': ambiguous_count,
            'weak_envelope_windows': weak_count,
            'non_regression_passed': render_guard.get('passed'),
            'process_completion_is_alignment_proof': False}


def saved_baseline_plan(directory):
    try:
        return json.loads((Path(directory) / 'BASELINE_PCM_PLAN.json').read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return None


def save_sync_outcome(directory, outcome):
    """Make the independent outcome prominent without rewriting measurements."""
    directory = Path(directory)
    (directory / 'SYNC_OUTCOME.json').write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [outcome['headline_ru'], '=' * 78, '', outcome['message_ru'], '',
             'Статус синхронизации: ' + outcome['status'],
             'Допуск итоговой оценки в проверенных окнах: 20 мс.',
             'Надёжных PCM-окон: ' + str(outcome['pcm_evidence']['trusted_windows']),
             'Надёжных двусторонних окон огибающей: ' + str(outcome['envelope_evidence']['trusted_windows']),
             'Подтверждённых окон с остатком выше допуска: ' + str(outcome['confirmed_mismatch_count']),
             '', 'Это оценка по измеренным общим звуковым окнам, а не гарантия совпадения неизвестного или различающегося звука.']
    (directory / 'SYNC_OUTCOME_RU.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')
    prefix = ('ИТОГ СИНХРОНИЗАЦИИ ПО НЕЗАВИСИМЫМ ЗВУКОВЫМ ПРОВЕРКАМ\n'
              + outcome['headline_ru'] + '\n' + outcome['message_ru']
              + '\nСтатус: ' + outcome['status'] + '\n\n'
              + 'Ниже приведена диагностика. Оценка формы карты и прохождение защиты '
                'от ухудшения сами по себе не подтверждают синхронизацию.\n' + '=' * 78 + '\n\n')
    for name in ('FINAL_SYNC_REPORT.txt', 'FINAL_RENDER_VALIDATION_REPORT.txt', 'FINAL_SYNC_EFFECTIVENESS_REPORT.txt'):
        path = directory / name
        if path.is_file():
            text = path.read_text(encoding='utf-8-sig')
            if not text.startswith('ИТОГ СИНХРОНИЗАЦИИ ПО НЕЗАВИСИМЫМ'):
                path.write_text(prefix + text, encoding='utf-8')
    path = directory / 'FINAL_RENDER_VALIDATION.json'
    if path.is_file():
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        data['synchronization_outcome'] = outcome
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
