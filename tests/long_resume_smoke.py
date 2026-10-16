"""Full movie-length PCM resume/render/verification; generated audio only.

python -m tests.long_resume_smoke /tmp/wavesync-long 5252
One channel limits disk usage; multichannel coherence is checked by unittest.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
from scipy.signal import butter, sosfilt

from common import SelectedTracks
from pcm_core.model import TimeMap, Piece
from pcm_core.micro import grid
from pcm_core.config import Settings
from resume_launcher import resume
from scanner import probe_audio_file, save_initial_metadata


def _hash(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    started = time.monotonic()
    root = Path(sys.argv[1]).resolve()
    root.mkdir(exist_ok=False, parents=True)
    duration = float(sys.argv[2]) if len(sys.argv)>2 else 5252.
    sr = 48000
    previous = root/'saved_WAVEFORM_SYNC'
    previous.mkdir()
    frames = round(duration*sr)
    reference = np.memmap(previous/'reference_first_channel_s16le.pcm',dtype='<i2',mode='w+',shape=(frames,))
    target = np.memmap(previous/'target_first_channel_s16le.pcm',dtype='<i2',mode='w+',shape=(frames,))
    rng = np.random.default_rng(695)
    filt = butter(4, 5000, fs=sr, output='sos')
    state = np.zeros((len(filt),2))
    for begin in range(0,frames,1_000_000):
        end = min(frames,begin+1_000_000)
        values,state = sosfilt(filt,rng.normal(0,.12,end-begin),zi=state)
        reference[begin:end] = np.rint(np.clip(values,-.9,.9)*32767).astype('<i2')
    xs,ys = [0.],[-.006]
    for index,t in enumerate(np.arange(20.,duration-1.,20.)):
        value = -.010 if index%2==0 else -.006
        xs += [float(t),float(t+.5)]
        ys += [ys[-1],value]
    xs.append(duration);ys.append(ys[-1])
    xs,ys = np.array(xs),np.array(ys)
    candidate = TimeMap([Piece(xs,xs+ys,[{'target':xs.tolist(),'delta':ys.tolist()}])]).validate(.02)
    probes = grid(TimeMap.affine(duration),Settings(anchor_step=1.))
    bad_times = [min(probes,key=lambda t:abs(t-x)) for x in np.arange(107.,duration-5,250.)]
    for begin in range(0,frames,1_000_000):
        end = min(frames,begin+1_000_000)
        t = np.arange(begin,end)/sr
        positions = candidate.forward(t)*sr
        for bad in bad_times:
            at = np.abs(t-bad)<.3
            positions[at] = t[at]*sr
        index = np.floor(positions).astype(np.int64)
        fraction = positions-index
        good = (index>=0)&(index<frames-1)
        values = np.zeros(end-begin)
        values[good] = ((1-fraction[good])*reference[index[good]]+fraction[good]*reference[index[good]+1])
        target[begin:end] = np.rint(values).astype('<i2')
    reference.flush();target.flush()
    for role in ('reference','target'):
        subprocess.run(['ffmpeg','-v','error','-nostdin','-n','-f','s16le','-ar',str(sr),'-ac','1',
            '-i',str(previous/f'{role}_first_channel_s16le.pcm'),'-c:a','pcm_s16le',str(root/f'{role}.wav')],check=True)
    selected = SelectedTracks(probe_audio_file(root/'reference.wav'),probe_audio_file(root/'target.wav'))
    save_initial_metadata(previous,selected)
    info={'analysis_sample_rate':sr,'pcm_format':'s16le'}
    for role in ('reference','target'):
        p=previous/f'{role}_first_channel_s16le.pcm'
        info[f'{role}_decode']={'path':str(p),'size_bytes':p.stat().st_size,'duration':duration}
    (previous/'analysis_info.json').write_text(json.dumps(info))
    t=np.r_[np.arange(0,duration,1.),duration].tolist()
    coarse={'target_time':t,'reference_time':t,'offset':[0.]*len(t)}
    (previous/'FINAL_SYNC_MAP.json').write_text(json.dumps({'coarse_map_before_pcm':coarse}))
    saved={'status':'BASELINE_KEPT_ERROR','baseline_file':str(root/'target.wav'),
        'selected_step_sec':1.,'fine_map':candidate.to_dict(),'measurements':[],
        'independent_windows':[],'actual_output_windows':[{'target_time':t,'analysis_step_sec':1.} for t in probes]}
    (previous/'PCM_MICRO_REPORT.json').write_text(json.dumps(saved))
    hashes=[_hash(p) for p in (root/'reference.wav',root/'target.wav')]
    reference._mmap.close();target._mmap.close()
    ready=time.monotonic()
    with patch('builtins.input',return_value='y'), patch('pcm_refiner._analyze_candidate',side_effect=AssertionError('analysis repeated')):
        result=resume(previous)
    report=result['pcm_micro_refinement']
    assert report['status']=='PCM_REFINED',report['reason']
    assert not report['analysis_steps_sec'] and report['resumed_candidate']
    assert any(x['status']=='PREFLIGHT_LOCAL_CANCEL' for x in report['audit'])
    assert len(report['actual_output_windows'])>duration*.8
    assert hashes==[_hash(p) for p in (root/'reference.wav',root/'target.wav')]
    rows=[r for r in report['actual_output_windows'] if r['micro_changed'] and r['status']=='VERIFIED']
    residual=np.median([abs(r['residual_sec']) for r in rows])
    assert residual<1/sr,residual
    summary={'duration_sec':duration,'native_sample_rate':sr,'channels':1,'verified_changed_windows':len(rows),
             'median_residual_sec':float(residual),'resume_elapsed_sec':time.monotonic()-ready,
             'total_elapsed_sec':time.monotonic()-started,'source_hashes_unchanged':True}
    (root/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    print('LONG RESUME SMOKE PASSED:',json.dumps(summary))


if __name__=='__main__':
    main()
