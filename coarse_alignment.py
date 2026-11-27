"""Optional common-sound alignment for different mixes; the PCM path is intact.

This module uses measured evidence only. No file names, absolute timecodes or
previous run directories participate in choosing a clock.
"""
from __future__ import annotations

import json
from pathlib import Path
import traceback

import numpy as np

import aligner


class NoCommonAudio(RuntimeError):
    """Healthy refusal to make a time map without enough common sound."""


def classic_is_reliable(alignment):
    rows = alignment.get('local_results', []) if alignment else []
    if not rows:
        return False
    strong = [r for r in rows if r.get('offset') is not None
              and float(r.get('correlation', 0)) >= .65
              and float(r.get('margin', 0)) >= .025]
    return len(strong) >= 5 and len(strong) / len(rows) >= .70


def _save_report(directory, report):
    directory = Path(directory)
    (directory / 'COARSE_MATCH_REPORT.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    lines = ['ПОИСК ОБЩЕГО ЗВУКА В РАЗНЫХ МИКСАХ', '=' * 78,
             'Режим: ' + report['mode'], report.get('message', ''),
             'Это крупное сопоставление, а не доказательство точности PCM до отсчёта.']
    if 'confirmed_anchors' in report:
        lines += ['Подтверждено опорных точек: ' + str(report['confirmed_anchors'])]
    (directory / 'COARSE_MATCH_REPORT_RU.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')


def _persist_alignment(selected, directory, analysis, rows, progress):
    offsets = [float(r['offset']) for r in rows if r.get('offset') is not None]
    global_result = {'coarse_offset': float(np.median(offsets)),
                     'reference_duration': float(selected.reference.duration),
                     'target_duration': float(selected.target.duration),
                     'results': [dict(r) for r in rows],
                     'evidence_kind': 'spectral_common_audio'}
    aligner.save_global_results_csv(directory, global_result)
    aligner.save_local_anchor_csv(directory, rows)
    aligner.save_analysis_state(directory, global_result, rows)
    aligner.save_waveform_overview_jpeg(directory, analysis['reference_waveform_times'],
        analysis['reference_waveform'], analysis['target_waveform_times'], analysis['target_waveform'], progress)
    aligner.save_envelope_overview_jpeg(directory, analysis['reference_envelope'], analysis['target_envelope'], progress)
    aligner.save_global_alignment_jpeg(directory, global_result, progress)
    aligner.save_local_anchor_jpeg(directory, rows, progress)
    summary = aligner.calculate_initial_sync_summary(global_result, rows)
    summary['evidence_kind'] = 'spectral_common_audio'
    aligner.save_initial_sync_summary(directory, summary)
    return {'global_result': global_result, 'local_results': rows, 'summary': summary}


def _anchors(observations):
    result = []
    for observation in observations:
        status = observation.get('feature_status', observation.get('status'))
        if status != 'CONFIRMED':
            continue
        time = observation.get('source_target_time', observation.get('target_time'))
        offset = observation.get('observed_offset', observation.get('offset'))
        if time is None or offset is None:
            continue
        score = float(observation.get('score', observation.get('correlation', 0)))
        margin = float(observation.get('margin', 0))
        if not (np.isfinite([time, offset, score, margin]).all() and score >= .45 and margin >= .025):
            continue
        result.append({'target_time': float(time), 'offset': float(offset),
                       'correlation': score, 'margin': margin,
                       'second_correlation': score-margin,
                       'accepted_for_tracking': True,
                       'evidence_kind': 'spectral_common_audio',
                       'rate': observation.get('rate'),
                       'uncertainty_sec': observation.get('uncertainty_sec'),
                       'reference_view': observation.get('reference_view'),
                       'target_view': observation.get('target_view')})
    return sorted(result, key=lambda r: r['target_time'])


def defer_pcm_micro_until_render(mapping, directory):
    """A coarse spectral estimate is refined on the ACTUAL rendered PCM later."""
    import matplotlib.pyplot as plt
    message = ('Для спектральной крупной карты точное PCM-уточнение выполняется '
               'после записи и проверки базовой дорожки на шаге 11. '
               'Проверенный PCM-алгоритм и его пороги сохранены.')
    print(message, flush=True)
    directory=Path(directory)
    (directory/'MICRO_WAVEFORM_REFINEMENT_REPORT.txt').write_text(message+'\n',encoding='utf-8')
    figure,axis=plt.subplots(figsize=(12,4))
    axis.plot(mapping['target_time'],mapping['offset'])
    axis.set(xlabel='Время исправляемой дорожки, с',ylabel='Крупное смещение, с',
             title='Крупная карта: точное PCM-уточнение выполняется после записи')
    axis.grid(True,alpha=.3)
    figure.tight_layout()
    figure.savefig(directory/'11_MICRO_WAVEFORM_REFINEMENT.jpg',dpi=130)
    plt.close(figure)
    return {'sync_map':mapping,'accepted':[],'records':[],
            'method':'PCM refinement after confirmed coarse render'}


def run_general_alignment_stage(selected, directory, analysis, progress):
    """Keep a reliable legacy alignment, otherwise try an independent backend.

    Feature uncertainty is retained. Scores are actual normalized feature
    correlations; they are never inflated to turn weak audio into a strong
    legacy anchor. The same backend must also validate the actual output.
    """
    directory = Path(directory)
    legacy = None
    legacy_error = None
    # The legacy 120-second global template cannot serve short excerpts.
    if min(selected.reference.duration, selected.target.duration) >= 160.:
        try:
            legacy = aligner.run_alignment_stage(selected, directory, analysis, progress)
        except Exception as error:
            legacy_error = str(error)
            (directory/'CLASSIC_ALIGNMENT_TRACEBACK.txt').write_text(traceback.format_exc(), encoding='utf-8')
            print('Сопоставление по первому каналу недоступно. Проверяются общие звуковые события.', flush=True)
    if classic_is_reliable(legacy):
        _save_report(directory, dict(mode='CLASSIC',
            message='Первый канал даёт устойчивое сопоставление. Проверенный путь сохранён.'))
        return legacy

    print('\nПОИСК ОБЩЕГО ЗВУКА ПО КАНАЛАМ И ЧАСТОТАМ', flush=True)
    print('Признаки вычисляются последовательно; исходные PCM первого канала сохраняются.', flush=True)
    reference = target = evidence = None
    try:
        from coarse_features import build_features_file
        from coarse_evidence import CoarseEvidence
        from coarse_pcm_guard import PCMProtectedEvidence
        reference = build_features_file(selected.reference, directory/'_coarse_features'/'reference', progress)
        target = build_features_file(selected.target, directory/'_coarse_features'/'target', progress)
        radius = min(300., max(15., abs(reference.duration-target.duration)+15.))
        evidence = PCMProtectedEvidence(CoarseEvidence(reference, target, search_radius=radius),
                                        analysis['reference_pcm'], analysis['target_pcm'])
        duration = min(reference.duration, target.duration)
        interval = min(30., max(1.5, (duration-12.)/16.))
        observations = evidence.observations(interval_sec=interval, phase=.5)
        rows = _anchors(observations)
        from coarse_clock import fit_global_clock
        global_clock = fit_global_clock(rows, target.duration)
        if any(r.get('status') in {'AMBIGUOUS', 'BOUNDARY'} for r in observations):
            global_clock = None
        evidence.global_clock = global_clock
        report = dict(mode='SPECTRAL_COMMON_AUDIO', search_radius_sec=radius,
                      feature_frame_hop_sec=reference.hop_sec,
                      confirmed_anchors=len(rows), observations=observations,
                      global_clock=global_clock,
                      score_definition='normalized correlation of spectral features, not raw PCM',
                      message='Новые опорные точки подтверждаются отдельными звуковыми окнами.')
        if legacy_error is not None:
            report['classic_alignment_error'] = legacy_error
        if len(rows) < 5:
            report.update(mode='INSUFFICIENT_COMMON_AUDIO',
                          message='Недостаточно независимых опорных точек для новой карты времени.')
            _save_report(directory, report)
            if legacy is not None:
                return legacy
            raise NoCommonAudio(report['message'])
        _save_report(directory, report)
        result = _persist_alignment(selected, directory, analysis, rows, progress)
        analysis['coarse_evidence'] = evidence
        analysis['coarse_evidence_kind'] = 'spectral_common_audio'
        from diagnostics import register_analysis_resource
        register_analysis_resource(evidence)
        return result
    except NoCommonAudio:
        raise
    except Exception as error:
        _save_report(directory, dict(mode='FEATURE_BACKEND_UNAVAILABLE', message=str(error)))
        (directory/'COARSE_MATCH_TRACEBACK.txt').write_text(traceback.format_exc(), encoding='utf-8')
        if legacy is not None:
            print('Дополнительное сопоставление недоступно. Сохраняется исходный проверяемый путь.', flush=True)
            return legacy
        raise NoCommonAudio('Не удалось подтвердить общие звуковые опоры: '+str(error)) from error
    finally:
        if analysis.get('coarse_evidence') is not evidence or evidence is None:
            for bank in (reference, target):
                if bank is not None:
                    bank.close()
