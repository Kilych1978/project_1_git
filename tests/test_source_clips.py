import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import wave
import zipfile

import numpy as np

from source_clips import collect_source_clips,find_ffprobe,PAIR_ARCHIVE_LIMIT
from tools.collect_audio_points import extract_channels,find_ffmpeg,RATE


def make_source(path,channels,rate,duration,codec):
    expressions='|'.join(f'0.05*sin(2*PI*{211+127*i}*t)+0.01*sin(2*PI*{411+79*i}*t*t)' for i in range(channels))
    layout='5.1' if channels==6 else 'stereo'
    command=['ffmpeg','-v','error','-nostdin','-n','-f','lavfi','-i',
             f'aevalsrc={expressions}:s={rate}:d={duration}:c={layout}',
             '-c:a',codec]
    command+=['-strict','-2','-b:a','1411200'] if codec=='dca' else ['-b:a','192000']
    subprocess.run([*command,str(path)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)


def independent_pcm(path,channels):
    # No -ac or pan filter here: independently preserve the decoder's channels.
    result=subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(path),'-map','0:a:0',
        '-ar',str(RATE),'-f','s16le','pipe:1'],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    return np.frombuffer(result.stdout,dtype='<i2').reshape(-1,channels)


class SourceClipTests(unittest.TestCase):
    def test_six_channel_dts_fractional_overlapping_slices_match_independent_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'six channels.dts'
            make_source(source,6,48000,4.2,'dca')
            expected=independent_pcm(source,6)
            windows=[(.123456,.7654321,root/'one.wav'),(.567891,1.2345678,root/'overlap.wav')]
            before=(source.stat().st_size,source.stat().st_mtime_ns)
            result=extract_channels(source,windows,find_ffmpeg(root),root/'decode.log',6,False)
            self.assertTrue(result['sequential_decode'])
            self.assertEqual(before,(source.stat().st_size,source.stat().st_mtime_ns))
            for start,duration,path in windows:
                begin,count=round(start*RATE),round(duration*RATE)
                with wave.open(str(path)) as data:
                    self.assertEqual((data.getnchannels(),data.getframerate(),data.getsampwidth(),data.getnframes()),(6,RATE,2,count))
                    actual=np.frombuffer(data.readframes(count),dtype='<i2').reshape(-1,6)
                self.assertTrue(np.array_equal(actual,expected[begin:begin+count]))

    def test_stereo_ac3_resampling_uses_one_exact_48k_timeline(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'stereo.ac3'
            make_source(source,2,44100,4.2,'ac3')
            expected=independent_pcm(source,2)
            windows=[(.00003125,.234567,root/'early.wav'),(1.234567,.765432,root/'later.wav')]
            extract_channels(source,windows,find_ffmpeg(root),root/'decode.log',2,False)
            for start,duration,path in windows:
                begin,count=round(start*RATE),round(duration*RATE)
                with wave.open(str(path)) as data:
                    self.assertEqual(data.getnchannels(),2)
                    actual=np.frombuffer(data.readframes(data.getnframes()),dtype='<i2').reshape(-1,2)
                self.assertTrue(np.array_equal(actual,expected[begin:begin+count]))

    def test_collector_archives_32_seconds_all_channels_with_identity_order_and_hashes(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);reference=root/'reference.dts';target=root/'target.ac3'
            make_source(reference,6,48000,34.,'dca');make_source(target,2,48000,34.,'ac3')
            before={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in (reference,target)}
            ffmpeg=find_ffmpeg(root)
            with contextlib.redirect_stdout(io.StringIO()):
                output,manifest=collect_source_clips(reference,target,[17.123456],root,ffmpeg,find_ffprobe(ffmpeg),False)
            self.assertEqual(manifest['status'],'COMPLETE')
            self.assertEqual(manifest['requested_window_sec'],32.)
            self.assertEqual(manifest['sources']['reference']['channels'],6)
            self.assertEqual(manifest['sources']['target']['channels'],2)
            self.assertEqual(manifest['sources']['reference']['decoded_channel_order'][:4],['FL','FR','FC','LFE'])
            self.assertEqual(manifest['sources']['target']['decoded_channel_order'],['FL','FR'])
            archive=output/'SOURCE_CLIPS_01.zip'
            self.assertLessEqual(archive.stat().st_size,PAIR_ARCHIVE_LIMIT)
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                inside=json.loads(z.read('CLIPS.json'));self.assertEqual(len(inside['clips']),1)
                clip=inside['clips'][0]
                self.assertEqual(clip['frame_count'],32*RATE)
                self.assertEqual(clip['start_frame'],round(1.123456*RATE))
                hashes=json.loads(z.read('SHA256.json'))
                for role,source in (('reference',reference),('target',target)):
                    payload=z.read(clip['files'][role])
                    self.assertEqual(hashlib.sha256(payload).hexdigest(),hashes[clip['files'][role]])
                    with wave.open(io.BytesIO(payload)) as wav:
                        actual=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').reshape(-1,wav.getnchannels())
                    expected=independent_pcm(source,actual.shape[1])
                    begin=clip['start_frame'];count=clip['frame_count']
                    self.assertTrue(np.array_equal(actual,expected[begin:begin+count]))
            for path in (reference,target):
                self.assertEqual(before[path.name],(path.stat().st_size,path.stat().st_mtime_ns))


if __name__=='__main__':unittest.main()
