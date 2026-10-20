import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pcm_core.audio import probe
from pcm_core.config import Settings
from pcm_core.guard import cancel_unsafe_components, gate_micro, verify_candidate_pcm, render_verdict
from pcm_core.micro import Measurement, collect, refined_map
from pcm_core.model import Piece, TimeMap
from pcm_core.render import render
from pcm_core.wav import finalize_wave
from .helpers import SR, soundtrack, warp, write_audio


class PCMRecoveryTests(unittest.TestCase):
    def test_rf64_fact_uses_actual_frames_instead_of_24_hour_sentinel(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'float.wav'
            subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
                            'anullsrc=r=48000:cl=5.1', '-t', '0.25', '-c:a', 'pcm_f32le',
                            '-rf64', 'always', str(path)], check=True, capture_output=True)
            data = path.read_bytes()
            fact = data.index(b'fact')+8
            self.assertEqual(struct.unpack_from('<I', data, fact)[0], 0xFFFFFFFF)
            frames = SR//4
            result = finalize_wave(path, frames, SR, 6)
            actual = path.read_bytes()
            self.assertEqual(struct.unpack_from('<I', actual, fact)[0], frames)
            ds64 = actual.index(b'ds64')+8
            self.assertEqual(struct.unpack_from('<Q', actual, ds64+16)[0], frames)
            self.assertEqual(result['duration_sec'], .25)
            data_start = actual.index(b'data')+8
            self.assertEqual(actual[data_start:], data[data_start:])
            # A sparse movie-size RF64 exercises 64-bit sizes without storing
            # gigabytes of generated audio. Audio duration is 87:32.010667.
            with self.subTest('movie-size sparse RF64 header'):
                if not hasattr(path.stat(), 'st_blocks'):
                    self.skipTest('Большой разреженный файл проверяется только на POSIX-файловой системе.')
                frames = 252096512
                data_bytes = frames*6*4
                with path.open('r+b') as file:
                    file.truncate(data_start+data_bytes)
                    file.seek(ds64)
                    file.write(struct.pack('<QQQ',data_start+data_bytes-8,data_bytes,0))
                    file.seek(fact)
                    file.write(struct.pack('<I',0xFFFFFFFF))
                result = finalize_wave(path,frames,SR,6)
                with path.open('rb') as file:
                    header = file.read(data_start)
                self.assertEqual(struct.unpack_from('<I',header,fact)[0],frames)
                self.assertEqual(struct.unpack_from('<Q',header,ds64+16)[0],frames)
                self.assertAlmostEqual(result['duration_sec'],5252.010666666667)

    def test_s16_six_channel_render_keeps_precision_size_and_riff_timing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            t = np.arange(SR)/SR
            values = np.stack([.1*np.sin(2*np.pi*(150+i*310)*t) for i in range(6)], axis=1)
            source = root/'source.wav'
            write_audio(source, values, layout='5.1', codec='pcm_s16le')
            before = source.read_bytes()
            work = root/'out'
            work.mkdir()
            result = render(probe(source), TimeMap.affine(1.), 1., work, Settings(), log=lambda _:None)
            self.assertEqual(probe(result).codec, 'pcm_s16le')
            self.assertEqual(result.read_bytes()[:4], b'RIFF')
            with wave.open(str(result)) as output:
                self.assertEqual((output.getnchannels(), output.getsampwidth(), output.getnframes()), (6, 2, SR))
                output_frames = output.readframes(SR)
            with wave.open(str(source)) as original:
                self.assertEqual(output_frames,original.readframes(SR))
            self.assertLess(abs(len(before)-result.stat().st_size), 512)
            self.assertEqual(source.read_bytes(), before)

    def test_zoom_rollback_preserves_verified_base_curve(self):
        t = np.array([0., 2., 6., 8., 12., 14.])
        main = {'target': t.tolist(), 'delta': [-.006]*len(t)}
        base = TimeMap([Piece(t, t-.006, [main])])
        x = np.array([0., 1., 5., 6., 8., 9., 13., 14.])
        delta = np.array([0., -.002, -.002, 0., 0., -.002, -.002, 0.])
        zoom = {'target': x.tolist(), 'delta': delta.tolist()}
        candidate = TimeMap([Piece(x, base.forward(x)+delta, [main, zoom])])
        final, removed = cancel_unsafe_components(base, candidate, [10.], .02, whole_components=True)
        self.assertTrue(removed)
        np.testing.assert_allclose(final.forward([9.,10.,11.]), base.forward([9.,10.,11.]), atol=1e-12)
        np.testing.assert_allclose(final.forward([2.,3.,4.]), candidate.forward([2.,3.,4.]), atol=1e-12)
        final.validate(.02)

    def test_bounded_gate_keeps_independent_good_curve_after_eight_failures(self):
        x = soundtrack(14., 921)
        t = np.array([0.,1.,5.,6.,8.,9.,13.,14.])
        delta = np.array([0.,-.006,-.006,0.,0.,-.006,-.006,0.])
        proposed = TimeMap([Piece(t,t+delta,[{'target':t.tolist(),'delta':delta.tolist()}])])
        true_delta = delta.copy()
        true_delta[4:] = 0.
        true = TimeMap([Piece(t,t+true_delta,[{'target':t.tolist(),'delta':true_delta.tolist()}])])
        y = warp(x,true.forward(np.arange(len(x))/SR)*SR)
        base = TimeMap.affine(14.)
        settings = Settings(anchor_step=1., max_rate_change=.02, min_improvement=1/SR)
        measurements = collect(x,y,base,settings,log=lambda _:None)
        probes = [2.,3.,4.,9.,10.,11.,12.]
        # Reproduce a proposal that remains locally unsafe after the bounded
        # iterations. Acceptance still uses real PCM, with no mocked verdict.
        with patch('pcm_core.guard.refined_map',return_value=proposed):
            final, rows, audit = gate_micro(x,y,base,measurements,settings,log=lambda _:None,probe_times=probes)
        self.assertTrue(any(r['status']=='GATE_LOCAL_CANCEL' for r in audit))
        self.assertAlmostEqual(float(final.forward([3.])[0])-3.,-.006,delta=1/SR)
        self.assertAlmostEqual(float(final.forward([10.])[0])-10.,0.,delta=1e-10)
        checks = verify_candidate_pcm(x,y,base,final,settings,target_times=probes)
        self.assertEqual(render_verdict(checks),'VERIFIED_PCM')

    def test_consistent_clock_bridges_one_ambiguous_anchor_without_claiming_a_measurement(self):
        base = TimeMap.affine(12.)
        rows = [Measurement(float(t),float(t),-.006,.98,.5,0.,3,1,'MEASURED','raw')
                for t in np.arange(.5,12.,1.)]
        rows[5] = Measurement(5.5,5.5,None)
        audit = []
        final = refined_map(base,rows,Settings(anchor_step=1.,max_rate_change=.02),
                            {m.target_time for m in rows if m.status=='MEASURED'},audit)
        self.assertTrue(any(r['status']=='CLOCK_GAP_PREDICTION' for r in audit))
        self.assertEqual(rows[5].status,'UNVERIFIABLE')
        np.testing.assert_allclose(final.forward([5.,5.5,6.])-[5.,5.5,6.],-.006,atol=1e-12)


if __name__ == '__main__':
    unittest.main()
