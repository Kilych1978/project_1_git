"""Standalone exact first-channel excerpt collector: Python + FFmpeg only."""
from __future__ import annotations

import datetime
import hashlib
import json
import math
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import wave
import zipfile

RATE = 48000
EXTENSIONS = {'.wav','.wave','.flac','.dts','.ac3','.eac3','.mka','.m4a','.aac','.mp3','.w64','.ogg','.opus'}


def parse_time(value):
    fields = [float(x) for x in value.replace(',', '.').split(':')]
    if (not 1 <= len(fields) <= 3 or any(not math.isfinite(x) or x < 0 for x in fields)
            or any(x >= 60 for x in fields[1:])):
        raise ValueError('Некорректный таймкод: ' + value)
    return sum(x*60**i for i,x in enumerate(reversed(fields)))


def find_ffmpeg(root):
    for path in (root/'ffmpeg.exe',root/'tools/ffmpeg/bin/ffmpeg.exe',root/'ffmpeg/bin/ffmpeg.exe'):
        if path.is_file():
            return str(path)
    path = shutil.which('ffmpeg')
    if not path:
        raise RuntimeError('FFmpeg не найден. Положите сборщик в папку программы, где работает FFmpeg, '
                           'или добавьте ffmpeg.exe в эту папку.')
    return path


def extract_first_channel(source, destinations, ffmpeg, log_path, show_progress=True):
    """Sequential sample counting: no imprecise elementary-DTS seeking."""
    records = [(round(start*RATE),round(duration*RATE),path) for start,duration,path in destinations]
    last_sample = max(start+count for start,count,_ in records)
    writers, counts = [], [0]*len(records)
    chunks = queue.Queue(maxsize=8)
    stop = threading.Event()
    process = None
    worker = None
    decoded = 0
    last_print = 0.
    try:
        for _,_,path in records:
            writer = wave.open(str(path),'wb')
            writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(RATE)
            writers.append(writer)
        with log_path.open('wb') as errors:
            process = subprocess.Popen([ffmpeg,'-hide_banner','-nostdin','-v','warning','-i',str(source),
                '-map','0:a:0','-af','pan=mono|c0=c0','-ar',str(RATE),'-ac','1','-f','s16le','pipe:1'],
                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=errors)

            def receive():
                try:
                    while not stop.is_set():
                        block = process.stdout.read(131072)
                        while not stop.is_set():
                            try:
                                chunks.put(block,timeout=.2)
                                break
                            except queue.Full:
                                pass
                        if not block:
                            break
                except Exception as exc:
                    if not stop.is_set():
                        chunks.put(exc)

            worker = threading.Thread(target=receive,daemon=True)
            worker.start()
            while decoded < last_sample:
                try:
                    block = chunks.get(timeout=90)
                except queue.Empty:
                    raise RuntimeError('FFmpeg не выдал звук за 90 секунд. См. журнал декодирования.')
                if isinstance(block,Exception):
                    raise block
                if not block:
                    raise RuntimeError('Дорожка закончилась до одной из указанных точек.')
                if len(block)%2:
                    raise RuntimeError('Получен неполный PCM-отсчёт.')
                end = decoded+len(block)//2
                for i,(begin,count,_) in enumerate(records):
                    lo,hi = max(decoded,begin),min(end,begin+count)
                    if hi > lo:
                        writers[i].writeframesraw(block[(lo-decoded)*2:(hi-decoded)*2])
                        counts[i] += hi-lo
                decoded = end
                if show_progress and time.monotonic()-last_print >= .5:
                    last_print = time.monotonic()
                    message = f'[{100*min(decoded,last_sample)/last_sample:6.2f}%] '
                    message += f'Декодировано {min(decoded,last_sample)/RATE:.1f} / {last_sample/RATE:.1f} с'
                    width = max(1,shutil.get_terminal_size().columns-1)
                    print('\r'+message[:width].ljust(width),end='',flush=True)
            if counts != [count for _,count,_ in records]:
                raise RuntimeError('Один из фрагментов записан не полностью.')
    finally:
        stop.set()
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
            if worker is not None:
                worker.join(timeout=3)
            process.stdout.close()
        for writer in writers:
            writer.close()
        if show_progress:
            print()


def choose(files,title,excluded=None):
    options = [path for path in files if path != excluded]
    print('\n'+title)
    for index,path in enumerate(options,1):
        print(f'{index}. {path.name}')
    while True:
        reply = input('Номер (q — выход): ').strip()
        if reply.lower() == 'q':
            raise KeyboardInterrupt
        if reply.isdigit() and 1 <= int(reply) <= len(options):
            return options[int(reply)-1]
        print('Введите номер из списка.')


def main():
    root = Path(__file__).resolve().parent
    print('СБОР ЗВУКОВЫХ ФРАГМЕНТОВ ДЛЯ ДИАГНОСТИКИ')
    print('Синхронизация не запускается. Исходные дорожки не изменяются.')
    ffmpeg = find_ffmpeg(root)
    files = sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS],key=lambda p:p.name.lower())
    if len(files)<2:
        raise RuntimeError('Положите два аудиофайла рядом со сборщиком либо распакуйте сборщик в папку этих файлов.')
    reference = choose(files,'Выберите эталон:')
    target = choose(files,'Выберите входную дорожку после v3.3:',excluded=reference)
    values = sys.argv[1:]
    if not values:
        values = input('Точки через пробел (7:28.755 52:05.916 52:15.301): ').split()
    if not 1 <= len(values) <= 8:
        raise ValueError('Укажите от одной до восьми точек.')
    points = sorted(set(parse_time(value) for value in values))
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output = root/('AUDIO_POINT_DIAGNOSTICS_'+stamp)
    output.mkdir(exist_ok=False)
    manifest = {'sample_rate':RATE,'channels':1,'timeline':'same absolute PCM timeline; first physical channel',
                'requested_points_sec':points,'clips':[],'status':'ERROR',
                'sources':{'reference':str(reference),'coarse_target':str(target)}}
    for index,point in enumerate(points,1):
        start = round(max(0.,point-4.)*RATE)/RATE
        manifest['clips'].append({'start_sec':start,'duration_sec':8.,'requested_time_sec':point,
            'files':{role:f'{index:02d}_{role}.wav' for role in ('reference','coarse_target')}})
    try:
        for role,source in [('reference',reference),('coarse_target',target)]:
            print('\nПервый канал:',source.name)
            destinations = [(clip['start_sec'],clip['duration_sec'],output/clip['files'][role])
                            for clip in manifest['clips']]
            extract_first_channel(source,destinations,ffmpeg,output/(role+'_ffmpeg.log'))
        manifest['status'] = 'COMPLETE'
    except BaseException:
        (output/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8-sig')
        raise
    finally:
        (output/'CLIPS.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        checksums = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
        (output/'SHA256.json').write_text(json.dumps(checksums,indent=2),encoding='utf-8')
        archive = root/(output.name+'.zip')
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(output.iterdir()):
                bundle.write(path,path.name)
        print('\nАрхив для отправки:',archive)
        print(f'Размер: {archive.stat().st_size/1024/1024:.2f} МиБ. Автоматической отправки нет.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nОстановлено пользователем.')
        sys.exit(1)
    except Exception as exc:
        print('\nОШИБКА:',exc)
        sys.exit(1)
