"""Continue a saved PCM candidate, retaining the previous run and source audio."""
import json
from pathlib import Path, PureWindowsPath

import numpy as np

from common import Progress, SelectedTracks
from pcm_core.config import Settings
from pcm_refiner import refine_rendered_audio
from scanner import (probe_audio_file, create_output_directory, save_initial_metadata,
                     confirm_selected_tracks, print_pair_summary)


def _read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _source(record, previous):
    path = Path(record['path'])
    if not path.is_file():
        path = previous.parent / record['filename']
    if not path.is_file() or path.stat().st_size != record['size_bytes']:
        raise ValueError(f"Исходный файл отсутствует или изменился: {record['filename']}")
    info = probe_audio_file(path)
    if not info or (info.channels, info.sample_rate) != (record['channels'], record['sample_rate']):
        raise ValueError(f"Изменились параметры исходника: {record['filename']}")
    return info


def resume(previous):
    previous = Path(previous).resolve()
    saved = _read(previous/'PCM_MICRO_REPORT.json')
    meta = _read(previous/'initial_metadata.json')
    analysis_info = _read(previous/'analysis_info.json')
    summary = _read(previous/'FINAL_SYNC_MAP.json')
    if analysis_info['analysis_sample_rate'] != 48000 or analysis_info['pcm_format'] != 's16le':
        raise ValueError('Сохранённые кэши должны иметь формат S16, 48 кГц.')
    caches = {}
    for role in ('reference', 'target'):
        path = previous/f'{role}_first_channel_s16le.pcm'
        if not path.is_file():
            path = Path(analysis_info[f'{role}_decode']['path'])
        if not path.is_file() or path.stat().st_size != analysis_info[f'{role}_decode']['size_bytes']:
            raise ValueError(f'Отсутствует или повреждён кэш: {path.name}')
        caches[f'{role}_pcm'] = path
    selected = SelectedTracks(_source(meta['reference'], previous), _source(meta['target'], previous))
    selected.reference.duration = analysis_info['reference_decode']['duration']
    selected.target.duration = analysis_info['target_decode']['duration']
    print_pair_summary(selected)
    confirm_selected_tracks(selected)
    baseline = Path(saved['baseline_file'])
    if not baseline.is_file():
        # Support moving an entire old result folder, on both Windows and POSIX.
        old_root = PureWindowsPath(analysis_info['reference_decode']['path']).parent.name
        parts = PureWindowsPath(saved['baseline_file']).parts
        if old_root in parts:
            tail = parts[parts.index(old_root)+1:]
            if '..' in tail:
                raise ValueError('Недопустимый путь сохранённого результата.')
            baseline = previous.joinpath(*tail)
    zero = not np.any(np.abs(summary.get('coarse_map_before_pcm', {}).get('offset', [0.])) > 1e-9)
    if not baseline.is_file() and zero:
        baseline = selected.target.path
    if not baseline.is_file():
        raise ValueError('Не найден основной аудиорезультат предыдущего запуска.')
    output = create_output_directory(selected.target)
    print('НОВАЯ ПАПКА РЕЗУЛЬТАТА:', output)
    print('Повторный крупный анализ и перебор сеток пропущены. Проверяем сохранённую карту.')
    save_initial_metadata(output, selected)
    (output/'analysis_info.json').write_text(json.dumps(analysis_info, ensure_ascii=False, indent=2), encoding='utf-8')
    coarse = summary.get('coarse_map_before_pcm')
    if coarse is None:
        times = np.array([0., selected.target.duration])
        coarse = {'target_time': times, 'reference_time': times.copy(), 'offset': np.zeros_like(times)}
    else:
        coarse = {k: np.asarray(v, float) for k, v in coarse.items()}
    coarse['confidence'] = np.ones_like(coarse['target_time'])
    copied = zero and baseline.stat().st_size == selected.target.path.stat().st_size
    result = {'output_path': baseline, 'encoder': 'copy', 'segments': [],
              'copied_without_reencode': copied,
              'codec_plan': {'source_codec': selected.target.codec_name, 'encoder': 'copy', 'pcm_fallback': False}}
    if copied:
        result['verified_first_channel_pcm'] = caches['target_pcm']
    mapping, result, report = refine_rendered_audio(selected, caches, coarse, result, output,
        Settings(anchor_step=1., min_improvement=1/48000, max_rate_change=.02), resume_report=saved)
    snapshot = {'reference_file': str(selected.reference.path), 'target_file': str(selected.target.path),
                'output_file': str(result['output_path']), 'mode': 'resume', 'resumed_from': str(previous),
                'pcm_micro_refinement': report, 'coarse_map_before_pcm': {
                    k: coarse[k].tolist() for k in ('target_time','reference_time','offset')},
                'audio_rendered': report['status'] == 'PCM_REFINED',
                **{k: mapping[k].tolist() for k in ('target_time','reference_time','offset')}}
    (output/'FINAL_SYNC_MAP.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
    Progress().done(100., 'Продолжение завершено')
    print(report['reason'])
    print('Итоговый файл:', result['output_path'])
    print('Отчёты:', output)
    return snapshot


def main(folder=None):
    print('\nПРОДОЛЖЕНИЕ СОХРАНЁННОГО PCM-УТОЧНЕНИЯ\n')
    if folder is None:
        folder = input('Вставь полный путь к прежней папке _WAVEFORM_SYNC: ').strip().strip('"')
    return resume(folder)
