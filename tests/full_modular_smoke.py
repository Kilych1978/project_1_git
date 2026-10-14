"""Run all 12 original interactive stages on 6-channel audio, including raw DTS.

Run from the repository: python -m tests.full_modular_smoke /tmp/sync-smoke
Keeps logs/reports for inspection. Audio is generated, never user-provided.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .helpers import SR, soundtrack, write_audio


def main():
    directory = Path(sys.argv[1]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    duration = 240.
    delay_sec = float(sys.argv[2]) if len(sys.argv) > 2 else .006
    reference = soundtrack(duration, 583)
    n = round(delay_sec * SR)
    target = np.r_[np.zeros(n), reference[:-n]]
    rp, tp = directory / '01_reference.dts', directory / '02_target.wav'
    # Central channels intentionally differ; only front-left is common.
    multichannel = np.stack([reference, reference * .5, reference * .3,
                            reference * .1, reference * .7, reference * .4], axis=1).astype('<f4')
    encoded = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'f32le',
                              '-ar', str(SR), '-ac', '6', '-channel_layout', '5.1', '-i', 'pipe:0',
                              '-c:a', 'dca', '-strict', '-2', '-b:a', '1411200', str(rp)],
                             input=multichannel.tobytes(), capture_output=True)
    if encoded.returncode:
        raise RuntimeError(encoded.stderr.decode())
    del multichannel
    target_channels = np.stack([target, target * .5, soundtrack(duration, 901),
                               target * .1, target * .7, target * .4], axis=1)
    write_audio(tp, target_channels, layout='5.1', codec='pcm_s16le')
    del target_channels
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (rp, tp)]
    import scanner
    import launcher
    original_probe = scanner.probe_audio_file
    def unknown_dts_duration(path):
        info = original_probe(path)
        if info and Path(path).suffix == '.dts':
            info.duration = 0.  # Exercise the user's missing-duration scenario.
        return info
    with patch.object(scanner, 'BASE_DIR', directory), patch.object(scanner, 'probe_audio_file', unknown_dts_duration), \
         patch('builtins.input', side_effect=['1', '1', 'y', 'y']):
        launcher.main()
    outputs = list(directory.glob('*_WAVEFORM_SYNC*/FINAL_SYNC_MAP.json'))
    assert len(outputs) == 1, 'All 12 stages must complete and save the final report'
    report = json.loads(outputs[0].read_text())
    fine = report['pcm_micro_refinement']
    assert fine['status'] == 'PCM_REFINED', fine.get('reason')
    assert report['output_duration'] > 239.9
    assert report['final_render_duration_check']
    verified = [r for r in fine['actual_output_windows'] if r['micro_changed'] and r['status'] == 'VERIFIED']
    assert len(verified) > 20
    residual = np.median([abs(r['residual_sec']) for r in verified])
    assert residual < 1 / SR, residual
    assert hashes == [hashlib.sha256(p.read_bytes()).hexdigest() for p in (rp, tp)]
    assert len(list(outputs[0].parent.glob('*.jpg'))) >= 13
    print('FULL MODULAR SMOKE PASSED: 12 stages; DTS + 6-channel WAV; source hashes unchanged;',
          len(verified), 'verified fine windows; median residual', residual, 'seconds')


if __name__ == '__main__':
    main()
