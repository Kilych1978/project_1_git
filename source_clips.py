"""Collect all source channels around a few times, without synchronization."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import wave
import zipfile

from tools.collect_audio_points import EXTENSIONS,RATE,choose,extract_channels,find_ffmpeg,parse_time


CHANNEL_ORDER={
    'mono':['FC'],'stereo':['FL','FR'],'2.1':['FL','FR','LFE'],
    '3.0':['FL','FR','FC'],'3.0(back)':['FL','FR','BC'],
    '4.0':['FL','FR','FC','BC'],'quad':['FL','FR','BL','BR'],
    'quad(side)':['FL','FR','SL','SR'],'5.0':['FL','FR','FC','BL','BR'],
    '5.0(side)':['FL','FR','FC','SL','SR'],'5.1':['FL','FR','FC','LFE','BL','BR'],
    '5.1(side)':['FL','FR','FC','LFE','SL','SR'],
    '6.1':['FL','FR','FC','LFE','BC','SL','SR'],
    '7.1':['FL','FR','FC','LFE','BL','BR','SL','SR'],
    '7.1(wide)':['FL','FR','FC','LFE','FLC','FRC','SL','SR'],
}
WINDOW_SECONDS=32.
PAIR_ARCHIVE_LIMIT=25*1024**2


def find_ffprobe(ffmpeg):
    executable=Path(ffmpeg)
    sibling=executable.with_name('ffprobe.exe' if executable.suffix.lower()=='.exe' else 'ffprobe')
    if sibling.is_file():return str(sibling)
    found=shutil.which('ffprobe')
    if not found:
        raise RuntimeError('FFprobe не найден. Он нужен рядом с ffmpeg.exe или в PATH для проверки числа и порядка каналов.')
    return found


def identity(path):
    value=Path(path).stat()
    return {'size_bytes':value.st_size,'mtime_ns':value.st_mtime_ns}


def source_info(path,ffprobe):
    command=[ffprobe,'-v','error','-select_streams','a:0','-show_entries',
        'stream=index,codec_name,sample_rate,channels,channel_layout,sample_fmt,bits_per_sample,bits_per_raw_sample,duration:format=duration,format_name',
        '-of','json',str(path)]
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
    if result.returncode:
        raise RuntimeError('FFprobe не прочитал источник: '+result.stderr.decode('utf-8',errors='replace').strip())
    data=json.loads(result.stdout)
    streams=data.get('streams',[])
    if not streams:raise ValueError('В файле нет звуковой дорожки: '+str(path))
    stream=streams[0];channels=int(stream.get('channels',0))
    if not 1<=channels<=64:raise ValueError('Не определено число каналов: '+str(path))
    layout=stream.get('channel_layout') or 'unknown'
    order=CHANNEL_ORDER.get(layout)
    if order is None or len(order)!=channels:
        order=[f'channel_{i}' for i in range(channels)]
    duration=None
    for value in (stream.get('duration'),data.get('format',{}).get('duration')):
        try:
            seconds=float(value)
            if math.isfinite(seconds) and seconds>0:
                duration=seconds;break
        except (TypeError,ValueError):pass
    return {'path':str(Path(path).resolve()),'filename':Path(path).name,'identity':identity(path),
        'codec_name':stream.get('codec_name'),'source_sample_rate':stream.get('sample_rate'),
        'source_sample_format':stream.get('sample_fmt'),'source_bits_per_raw_sample':stream.get('bits_per_raw_sample'),
        'channels':channels,'channel_layout':layout,'decoded_channel_order':order,
        'channel_order_basis':'FFmpeg native channel layout' if layout in CHANNEL_ORDER else 'Unknown semantic layout; physical decoded channel indices only',
        'duration_sec':duration,'selected_stream_index':stream.get('index'),
        'output_sample_rate':RATE,'output_sample_format':'s16le','output_wav_channel_layout':'not embedded; use decoded_channel_order from this manifest'}


def checksum(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _package(output,manifest):
    for clip in manifest['clips']:
        clip['audio_files']={}
        for role,name in clip['files'].items():
            path=output/name
            if not path.is_file():continue
            with wave.open(str(path),'rb') as wav:
                details={'channels':wav.getnchannels(),'sample_rate':wav.getframerate(),
                         'frames':wav.getnframes(),'sample_width_bytes':wav.getsampwidth()}
            details.update(bytes=path.stat().st_size,sha256=checksum(path))
            clip['audio_files'][role]=details
        clip['status']='COMPLETE' if (manifest['status']=='COMPLETE' and len(clip['audio_files'])==2
            and all(v['frames']==clip['frame_count'] for v in clip['audio_files'].values())) else 'ERROR'
    (output/'CLIPS.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archives=[]
    for index,clip in enumerate(manifest['clips'],1):
        path=output/f'SOURCE_CLIPS_{index:02d}.zip'
        per_clip=dict(manifest,clips=[clip],requested_points_sec=[clip['requested_time_sec']],archives=[path.name])
        hashes={name:checksum(output/name) for name in clip['files'].values() if (output/name).is_file()}
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
            archive.writestr('CLIPS.json',json.dumps(per_clip,ensure_ascii=False,indent=2)+'\n')
            archive.writestr('SHA256.json',json.dumps(hashes,indent=2)+'\n')
            for name in hashes:archive.write(output/name,name)
            if (output/'ERROR.txt').is_file():archive.write(output/'ERROR.txt','ERROR.txt')
        if sum(s['channels'] for s in manifest['sources'].values())<=8 and path.stat().st_size>PAIR_ARCHIVE_LIMIT:
            raise RuntimeError('Диагностический архив превысил 25 МиБ: '+str(path))
        archives.append(path)
    return archives


def collect_source_clips(reference,target,points,output_parent,ffmpeg=None,ffprobe=None,show_progress=True):
    """One sequential decode per source; only short all-channel PCM files on disk."""
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output=Path(output_parent).resolve()/('SOURCE_CHANNEL_DIAGNOSTICS_'+stamp)
    output.mkdir(parents=True,exist_ok=False)
    reference,target=Path(reference).resolve(),Path(target).resolve()
    manifest={'schema':1,'mode':'original-source-all-channel-diagnostics','status':'ERROR',
        'created':datetime.datetime.now().isoformat(),'sample_rate':RATE,'sample_format':'s16le',
        'requested_window_sec':WINDOW_SECONDS,
        'timeline':'same absolute sequentially decoded source PCM time; no synchronization or time-map application',
        'source_files_modified':False,'requested_points_sec':list(points),'sources':{},'clips':[],
        'whole_source_hashing':False,'source_identity_check':'size and mtime before/after; SHA256 applies only to exported clips',
        'decoder_logs':['reference_ffmpeg.log','target_ffmpeg.log'],'archives':[]}
    try:
        if reference==target:raise ValueError('Эталон и исправляемый источник должны быть разными файлами.')
        if not 1<=len(points)<=8 or any(not math.isfinite(p) or p<0 for p in points):
            raise ValueError('Укажите от одной до восьми неотрицательных точек.')
        ffmpeg=ffmpeg or find_ffmpeg(Path(__file__).resolve().parent)
        ffprobe=ffprobe or find_ffprobe(ffmpeg)
        for role,path in (('reference',reference),('target',target)):
            manifest['sources'][role]=source_info(path,ffprobe)
        durations=[s['duration_sec'] for s in manifest['sources'].values() if s['duration_sec'] is not None]
        common=min(durations) if durations else None
        duration=min(WINDOW_SECONDS,common) if common is not None else WINDOW_SECONDS
        if duration<=0:raise ValueError('Источники не содержат звука.')
        for index,point in enumerate(sorted(set(points)),1):
            if common is not None and point>=common:
                raise ValueError(f'Точка {point:.3f} с находится за пределами известной длительности {common:.3f} с.')
            start=max(0.,point-WINDOW_SECONDS/2)
            if common is not None:start=min(start,max(0.,common-duration))
            begin,count=round(start*RATE),round(duration*RATE)
            manifest['clips'].append({'requested_time_sec':point,'start_sec':begin/RATE,
                'duration_sec':count/RATE,'start_frame':begin,'frame_count':count,
                'files':{role:f'{index:02d}_{role}_all_channels.wav' for role in ('reference','target')}})
        for role,path in (('reference',reference),('target',target)):
            channels=manifest['sources'][role]['channels']
            print(f'\nИсточник: {path.name}; сохраняются все каналы: {channels}.',flush=True)
            print('Последовательное декодирование от начала только до последнего фрагмента; синхронизация не запускается.',flush=True)
            destinations=[(c['start_sec'],c['duration_sec'],output/c['files'][role]) for c in manifest['clips']]
            manifest['sources'][role]['decode']=extract_channels(path,destinations,ffmpeg,output/(role+'_ffmpeg.log'),channels,show_progress)
            manifest['sources'][role]['identity_after']=identity(path)
            if manifest['sources'][role]['identity_after']!=manifest['sources'][role]['identity']:
                raise RuntimeError('Исходник изменился во время сбора: '+str(path))
        for role,path in (('reference',reference),('target',target)):
            manifest['sources'][role]['identity_after']=identity(path)
            if manifest['sources'][role]['identity_after']!=manifest['sources'][role]['identity']:
                raise RuntimeError('Исходник изменился во время сбора: '+str(path))
        manifest['status']='COMPLETE'
    except BaseException:
        (output/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8-sig')
        print('\nДиагностическая папка с описанием ошибки: '+str(output),flush=True)
        raise
    finally:
        archives=_package(output,manifest)
        manifest['archives']=[p.name for p in archives]
        (output/'CLIPS.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        for path in archives:
            print(f'Архив для отправки: {path} ({path.stat().st_size/1024**2:.2f} МиБ)',flush=True)
        if archives:print('Автоматической отправки нет. Прикрепите архивы вручную.',flush=True)
    return output,manifest


def main(argv=None):
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description='Короткие фрагменты исходных дорожек со всеми каналами; без синхронизации.')
    parser.add_argument('times',nargs='*',help='Точки: например 1:20 15:40 50:45')
    parser.add_argument('--reference',type=Path);parser.add_argument('--target',type=Path)
    parser.add_argument('--scan-dir',type=Path,default=root);parser.add_argument('--output-dir',type=Path,default=root)
    args=parser.parse_args(argv)
    print('ДИАГНОСТИКА ИСХОДНЫХ ДОРОЖЕК: ВСЕ КАНАЛЫ')
    print('Исходники не изменяются. Полная синхронизация и построение карты не запускаются.')
    files=sorted((p for p in args.scan_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),key=lambda p:p.name.casefold())
    reference=args.reference.expanduser().resolve() if args.reference is not None else None
    if reference is None:
        if len(files)<2:raise ValueError('Положите исходные аудиофайлы рядом с COLLECT_SOURCE_CLIPS.cmd.')
        reference=choose(files,'Выберите исходную эталонную дорожку:')
    target=args.target.expanduser().resolve() if args.target is not None else choose(files,'Выберите исходную исправляемую дорожку (не готовый результат):',excluded=reference)
    values=args.times or input('Точки через пробел (например 1:20 15:40 50:45): ').replace(';',' ').split()
    points=sorted(set(parse_time(value) for value in values))
    collect_source_clips(reference,target,points,args.output_dir)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nОстановлено пользователем. Исходные файлы не изменены.');raise SystemExit(2)
    except Exception as error:
        print('\nОШИБКА: '+str(error));raise SystemExit(2)
