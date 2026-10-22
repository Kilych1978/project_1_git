"""Original selection/progress shell; refine a previously aligned track directly."""
import json
import time
from pathlib import Path

import numpy as np

from analyzer import prepare_audio_analysis
from common import Progress, SelectedTracks, build_exact_time_grid
from pcm_core.config import Settings
from pcm_refiner import refine_rendered_audio
from diagnostics import diagnostic_session
from scanner import (ask_selection, check_ffprobe, confirm_selected_tracks,
                     create_output_directory, print_pair_summary,
                     save_initial_metadata, scan_audio_files, validate_selected_tracks)


@diagnostic_session
def main():
    started = time.monotonic()
    print()
    print('=' * 78)
    print('UNIVERSAL AUDIO WAVEFORM SYNC')
    print('УТОЧНЕНИЕ УЖЕ СИНХРОНИЗИРОВАННОЙ ДОРОЖКИ')
    print('=' * 78)
    print()
    print('ШАГ 1/4')
    print('-' * 78)
    check_ffprobe()
    files = scan_audio_files(Progress())
    reference = ask_selection(files, 'Выбери эталонную дорожку')
    target = ask_selection(files, 'Выбери дорожку с рассинхроном', excluded=reference)
    selected = SelectedTracks(reference, target)
    validate_selected_tracks(selected)
    confirm_selected_tracks(selected)
    print_pair_summary(selected)
    output = create_output_directory(target)
    print('РАБОЧАЯ ПАПКА:', output)
    save_initial_metadata(output, selected)

    print()
    print('ШАГ 2/4')
    print('-' * 78)
    analysis = prepare_audio_analysis(selected, output, Progress())
    save_initial_metadata(output, selected)

    print()
    print('ШАГ 3/4')
    print('-' * 78)
    t = build_exact_time_grid(0., selected.target.duration, .25)
    zero = {'target_time': t, 'reference_time': t.copy(), 'offset': np.zeros_like(t),
            'confidence': np.ones_like(t)}
    baseline = {'output_path': target.path, 'encoder': 'copy', 'segments': [],
                'output_created': False, 'uses_original_directly': True,
                'copied_without_reencode': True, 'verified_first_channel_pcm': analysis['target_pcm'],
                'codec_plan': {'source_codec': target.codec_name, 'encoder': 'copy',
                               'pcm_fallback': False, 'container': target.format_name}}
    final_map, result, report = refine_rendered_audio(
        selected, analysis, zero, baseline, output,
        Settings(anchor_step=1., min_improvement=1 / 48000, max_rate_change=.02))

    print()
    print('ШАГ 4/4')
    print('-' * 78)
    progress = Progress()
    progress.stage('СОХРАНЕНИЕ ПРОВЕРЕННОГО РЕЗУЛЬТАТА')
    snapshot = {'reference_file': str(reference.path), 'target_file': str(target.path),
                'output_file': report['output_file'], 'retained_file': str(result['output_path']),
                'output_created': report['output_created'], 'mode': 'micro-only',
                'pcm_micro_refinement': report,
                'target_time': final_map['target_time'].tolist(),
                'reference_time': final_map['reference_time'].tolist(),
                'offset': final_map['offset'].tolist(),
                'audio_rendered': report['status'] == 'PCM_REFINED',
                'reference_time_definition': ('Nominal clock; exact PCM sampling plan: pcm_micro_refinement.fine_map'
                    if report.get('nominal_map_has_pcm_layers') else 'REFERENCE = TARGET + OFFSET')}
    with (output / 'FINAL_SYNC_MAP.json').open('w', encoding='utf-8') as file:
        json.dump(snapshot, file, ensure_ascii=False, indent=2)
    progress.done(100., 'Результат и отчёты сохранены')
    rows = [r for r in report['actual_output_windows'] if r['micro_changed'] and r['status'] == 'VERIFIED']
    print()
    print('=' * 78)
    print('ФИНАЛЬНЫЙ РЕЗУЛЬТАТ')
    print('=' * 78)
    print(report['reason'])
    if report['output_created']:
        print('Итоговый файл:', result['output_path'])
    else:
        print('Улучшение не подтверждено. Новый аудиофайл не создан.')
        print('Исходная дорожка:', result['output_path'])
        if report['status'] == 'BASELINE_KEPT_ERROR':
            print('PCM-уточнение завершилось ошибкой; подробности: PCM_ERROR_TRACEBACK.txt.')
    print('Отчёты:', output)
    print('Подтверждённых изменённых окон:', len(rows))
    if rows and report['status'] == 'PCM_REFINED':
        residual = np.array([abs(r['residual_sec']) for r in rows]) * 1000
        print(f'Остаточный сдвиг: медиана {np.median(residual):.6f} мс; P95 {np.percentile(residual, 95):.6f} мс')
        print(f'Наибольший остаток в проверенных изменённых окнах: {np.max(residual):.6f} мс')
        print('Проверенных окон с остатком больше 0,1 мс:', len(report.get('remaining_windows', [])))
    print('Исходные дорожки не изменены.')
    print(f'Общее время: {(time.monotonic() - started) / 60:.2f} мин')
