"""Export exact PCM around user-specified Audacity times, without a sync run."""
import json
import math
import numpy as np

from analyzer import decode_first_channel_stream
from common import Progress, SelectedTracks
from diagnostics import diagnostic_session
from pcm_refiner import export_diagnostic_clips
from scanner import (ask_selection, check_ffprobe, confirm_selected_tracks,
                     create_output_directory, print_pair_summary, save_initial_metadata,
                     scan_audio_files, validate_selected_tracks)


def parse_time(value):
    fields = value.replace(',', '.').split(':')
    if not 1 <= len(fields) <= 3:
        raise ValueError('Введите время как 7:28.755 или 00:07:28.755.')
    numbers = [float(x) for x in fields]
    if any(not math.isfinite(x) or x < 0 for x in numbers) or any(x >= 60 for x in numbers[1:]):
        raise ValueError('Некорректный таймкод: ' + value)
    return sum(x*60**i for i,x in enumerate(reversed(numbers)))


@diagnostic_session
def main(times=None):
    print('\nДИАГНОСТИКА УКАЗАННЫХ ТОЧЕК AUDACITY')
    print('Создаются только короткие монофрагменты. Синхронизация фильма не запускается.')
    check_ffprobe()
    files = scan_audio_files(Progress())
    reference = ask_selection(files, 'Выбери эталонную дорожку')
    target = ask_selection(files, 'Выбери дорожку с рассинхроном', excluded=reference)
    selected = SelectedTracks(reference, target)
    validate_selected_tracks(selected)
    confirm_selected_tracks(selected)
    print_pair_summary(selected)
    if not times:
        times = input('Точки через пробел (например, 7:28.755 52:05.916 52:15.301): ').replace(';', ' ').split()
    if not 1 <= len(times) <= 8:
        raise ValueError('Укажите от одной до восьми точек.')
    points = [parse_time(value) for value in times]
    output = create_output_directory(target)
    save_initial_metadata(output, selected)
    work = output/'_point_decode'
    work.mkdir()
    pcms, paths = [], []
    try:
        # Decode sequentially from the beginning. Seeking in elementary DTS
        # can change its timestamp origin; PCM sample counters cannot.
        for index, info in enumerate((reference, target)):
            path = work/f'{index}_first_channel_s16le.pcm'
            paths.append(path)
            decoded = decode_first_channel_stream(info, path, Progress(), index*50., (index+1)*50.,
                estimated_duration=max(reference.duration, target.duration))
            info.duration = decoded['duration']
            pcms.append(np.memmap(path, mode='r', dtype='<i2'))
        duration = min(len(pcm) for pcm in pcms)/48000
        if duration < 5 or any(t >= duration for t in points):
            raise ValueError('Одна из точек находится вне общего времени дорожек; длительность '
                             f'{duration:.3f} с.')
        margin = min(8., duration)/2
        centers = [min(duration-margin, max(margin,t)) for t in points]
        export_diagnostic_clips(pcms[0],pcms[1],output,times=centers,include_uniform=False)
        info = {'mode': 'diagnostic-points-only', 'requested_points_sec': points,
                'clip_centers_sec': centers, 'source_audio_unchanged': True,
                'sample_rate': 48000, 'analysis_channel': 0}
        (output/'REQUESTED_POINTS.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        save_initial_metadata(output, selected)
        print('\nГотово. Полные дорожки и промежуточные PCM в архив не включаются.')
    finally:
        for pcm in pcms:
            pcm._mmap.close()
        for path in paths:
            path.unlink(missing_ok=True)
