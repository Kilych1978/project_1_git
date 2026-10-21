"""Compare actual WAVs at identical absolute times, independent of chosen maps.

python -m tools.compare_diagnostic_runs CLIPS_DIR OLD_RUN NEW_RUN RESULT_JSON
"""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

from pcm_core.config import Settings
from pcm_core.micro import measure_window
from pcm_core.model import TimeMap


def pcm(path):
    rate, values = wavfile.read(path)
    if rate != 48000 or values.ndim != 1:
        raise ValueError('Нужен первый канал 48 кГц.')
    if np.issubdtype(values.dtype, np.integer):
        values = values.astype(np.float32)/(-np.iinfo(values.dtype).min)
    return values


def valid(row, settings):
    return bool(row and not row['near_edge'] and row['score'] >= settings.min_raw_score
                and row['margin'] >= settings.min_raw_margin)


def main():
    source, old, new, destination = map(Path,sys.argv[1:])
    manifest = json.loads((source/'CLIPS.json').read_text(encoding='utf-8-sig'))
    result = []
    for index, clip in enumerate(manifest['clips'],1):
        original = pcm(source/clip['files']['coarse_target'])
        reference = pcm(source/clip['files']['reference'])
        waves = []
        for run in (old,new):
            report = json.loads((run/f'{index:02d}'/'PCM_MICRO_REPORT.json').read_text())
            waves.append(pcm(Path(report['output_file'])))
        duration = min(map(len,[reference,original,*waves]))/48000
        mapping = TimeMap.affine(duration)
        settings = Settings(anchor_step=.125)
        rows = []
        for t in np.arange(.25,duration-.25,.0625):
            measurements = [measure_window(reference,wave,mapping,float(t),settings) for wave in [original,*waves]]
            rows.append([m['correction_sec'] if valid(m,settings) else None for m in measurements])
        common = np.array([r for r in rows if all(v is not None for v in r)])
        item = {'clip_start_sec':clip['start_sec'],'fixed_probes':len(rows),
                'common_measurable_probes':len(common),
                'old_unmeasurable':sum(r[1] is None for r in rows),
                'new_unmeasurable':sum(r[2] is None for r in rows)}
        if len(common):
            absolute = np.abs(common)*1000
            item.update(before_median_ms=float(np.median(absolute[:,0])),
                        old_median_ms=float(np.median(absolute[:,1])),
                        new_median_ms=float(np.median(absolute[:,2])),
                        improved_vs_old=int(np.sum(absolute[:,2]+1000/48000<absolute[:,1])),
                        worse_vs_old=int(np.sum(absolute[:,2]>absolute[:,1]+1000/48000)),
                        improved_vs_source=int(np.sum(absolute[:,2]+1000/48000<absolute[:,0])),
                        worse_vs_source=int(np.sum(absolute[:,2]>absolute[:,0]+1000/48000)))
        result.append(item)
        print(json.dumps(item,ensure_ascii=False),flush=True)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
