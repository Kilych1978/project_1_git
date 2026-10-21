"""Private, reproducible tests on a DIAGNOSTIC_CLIPS folder; never upload audio.

python -m tools.verify_diagnostic_clips EXTRACTED_CLIPS OUTPUT_DIRECTORY
"""
import contextlib
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from common import SelectedTracks
from pcm_core.config import Settings
from pcm_refiner import refine_rendered_audio
from scanner import probe_audio_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    parser.add_argument('indices',nargs='*',type=int)
    parser.add_argument('--step',type=float,default=.5)
    args=parser.parse_args()
    source,destination=args.source.resolve(),args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / 'CLIPS.json').read_text(encoding='utf-8-sig'))
    summary = []
    selected_indices = set(args.indices)
    for index, clip in enumerate(manifest['clips'], 1):
        if selected_indices and index not in selected_indices:
            continue
        names = [clip['files'][key] for key in ('reference', 'coarse_target')]
        if not all(Path(name).name == name and not Path(name).is_absolute() for name in names):
            raise ValueError('Недопустимый путь в описании фрагментов.')
        rp, tp = [source / name for name in names]
        rate, r = wavfile.read(rp)
        target_rate, t = wavfile.read(tp)
        if (rate != 48000 or rate != target_rate or r.dtype != np.int16 or t.dtype != np.int16
                or r.ndim != 1 or t.ndim != 1):
            raise ValueError('Ожидаются диагностические монофрагменты S16, 48 кГц.')
        directory = destination / f'{index:02d}'
        directory.mkdir()
        r.astype('<i2').tofile(directory / 'reference.pcm')
        t.astype('<i2').tofile(directory / 'target.pcm')
        duration = len(t) / rate
        times = np.r_[np.arange(0., duration, .05), duration]
        identity = {'target_time': times, 'reference_time': times.copy(),
                    'offset': np.zeros_like(times), 'confidence': np.ones_like(times)}
        selected = SelectedTracks(probe_audio_file(rp), probe_audio_file(tp))
        original_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (rp, tp)]
        baseline = {'output_path': tp, 'encoder': 'copy', 'segments': [],
                    'copied_without_reencode': True, 'verified_first_channel_pcm': directory / 'target.pcm',
                    'codec_plan': {'encoder': 'copy', 'pcm_fallback': False}}
        with contextlib.redirect_stdout(io.StringIO()):
            _, result, report = refine_rendered_audio(
                selected, {'reference_pcm': directory / 'reference.pcm'}, identity, baseline, directory,
                Settings(anchor_step=args.step, min_improvement=1 / rate, max_rate_change=.02), make_plots=False)
        assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in (rp, tp)] == original_hashes
        rows = [row for row in report['actual_output_windows'] if row['micro_changed'] and row['status'] == 'VERIFIED']
        item = {'start_sec': clip['start_sec'], 'status': report['status'],
                'verified_windows': len(rows), 'source_unchanged': True}
        if rows and report['status'] == 'PCM_REFINED':
            item.update(before_median_ms=float(np.median([abs(row['baseline_residual_sec']) for row in rows]) * 1000),
                        after_median_ms=float(np.median([abs(row['residual_sec']) for row in rows]) * 1000),
                        after_max_ms=float(max(abs(row['residual_sec']) for row in rows) * 1000))
        summary.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    (destination / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
