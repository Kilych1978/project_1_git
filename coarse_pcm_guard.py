"""Protect measurable first-channel PCM when coarse features use other mixes.

Spectral uncertainty never relaxes this guard.  A known good PCM event cannot
be sacrificed merely because another channel has a better feature score.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from pcm_core.config import Settings
from pcm_core.micro import measure
from pcm_core.model import TimeMap


class PCMProtectedEvidence:
    def __init__(self, evidence, reference_pcm, target_pcm, *, _source=None):
        self.evidence = evidence
        self.reference_pcm = Path(reference_pcm)
        self.target_pcm = Path(target_pcm)
        self._source = _source
        self._reference = self._target = self._actual = None
        self._cache = {}
        self.settings = Settings(anchor_step=5.)
        self.global_clock = None if _source is None else _source.global_clock

    def __getattr__(self, name):
        return getattr(self.evidence, name)

    def observations(self, *args, **kwargs):
        return self.evidence.observations(*args, **kwargs)

    def _original(self):
        return self if self._source is None else self._source._original()

    def _pcm(self):
        source = self._original()
        if source._reference is None:
            source._reference = np.memmap(source.reference_pcm, mode='r', dtype='<i2')
            source._target = np.memmap(source.target_pcm, mode='r', dtype='<i2')
        return source._reference, source._target

    def _baseline(self, time):
        source = self._original()
        key = round(float(time), 9)
        if key not in source._cache:
            reference, target = self._pcm()
            identity = TimeMap.affine(len(target) / self.settings.analysis_rate)
            result = measure(reference, target, identity, float(time), self.settings)
            source._cache[key] = result.to_dict()
        return copy.deepcopy(source._cache[key])

    def set_rendered_pcm(self, path):
        if self._actual is not None:
            self._actual._mmap.close()
        self._actual = np.memmap(path, mode='r', dtype='<i2')

    def for_output(self, info, directory, progress=None):
        return PCMProtectedEvidence(self.evidence.for_output(info, directory, progress),
                                    self.reference_pcm, self.target_pcm, _source=self._original())

    def validate(self, sync_map, target_duration, progress=None, interval_sec=None):
        interval = max(1., float(interval_sec or 60.))
        rows = self.evidence.validate(sync_map, target_duration, progress, interval_sec=interval)
        if self.global_clock is not None:
            for row in rows:
                if row.get('feature_backend'):
                    row['coarse_global_clock'] = dict(self.global_clock)
        reference, target = self._pcm()
        duration = min(len(reference), len(target)) / self.settings.analysis_rate
        # Half-step reference probes differ from the spectral validation grid.
        times = np.arange(max(1., interval/2), duration-1., interval)
        for time in times:
            baseline = self._baseline(time)
            if baseline['status'] != 'MEASURED' or baseline['correction_sec'] is None:
                continue
            before = float(baseline['correction_sec'])
            score = float(baseline['score'])
            reference_time = float(time) + before
            predicted = float(np.interp(time, sync_map['target_time'], sync_map['offset']))
            lost = False
            if self._source is None:
                after, after_score = before, score
                error = abs(before-predicted)
            elif self._actual is None:
                after = after_score = error = None
                lost = True
            else:
                actual = measure(reference, self._actual,
                                 TimeMap.affine(len(self._actual)/self.settings.analysis_rate),
                                 reference_time, self.settings)
                if actual.status != 'MEASURED' or actual.correction_sec is None:
                    after = after_score = error = None
                    lost = True
                else:
                    after = float(actual.correction_sec)
                    after_score = float(actual.score)
                    error = abs(after)
            status = 'UNRESOLVED' if lost else ('CONFIRMED_MAP' if error <= .020 else 'CONFIRMED_DISCREPANCY')
            row = {'target_time': float(time), 'reference_time': reference_time,
                   'comparison_key': f'pcm:{float(time):.9f}',
                   'predicted_offset': predicted if self._source is None else 0.,
                   'before_offset': before, 'before_selected_offset': before,
                   'before_correlation': score, 'before_selected_correlation': score,
                   'before_bidirectional_offset': before,
                   'before_bidirectional_correlation': score,
                   'before_bidirectional_status': 'CONFIRMED_MAP',
                   'before_error': abs(before), 'after_offset': after,
                   'after_error': error, 'after_correlation': after_score,
                   'bidir_status': status, 'bidir_consensus_offset': after,
                   'bidir_consensus_correlation': after_score,
                   'bidir_left_correlation': after_score,
                   'bidir_right_correlation': after_score,
                   'bidir_disagreement': baseline.get('spread_sec'),
                   'bidir_prediction_error': error,
                   'evidence_kind': 'protected_first_channel_pcm',
                   'feature_backend': False, 'uncertainty_sec': self.settings.raw_consensus,
                   'before_uncertainty_sec': self.settings.raw_consensus,
                   'feature_verification_lost': lost,
                   'before_estimator': 'independent_first_channel_pcm',
                   'after_estimator': 'actual_first_channel_pcm' if self._source is not None else 'mapped_pcm_event'}
            rows.append(row)
        # Preserve UNKNOWN spectral records as explicit run barriers. The
        # numerical gate gives PCM priority at an identical source time.
        return sorted(rows, key=lambda r: (r['target_time'],
            0 if r.get('evidence_kind')=='protected_first_channel_pcm' else 1))

    def close(self):
        self.evidence.close()
        if self._actual is not None:
            self._actual._mmap.close()
            self._actual = None
        if self._source is None:
            self.evidence.reference_bank.close()
            self.evidence.target_bank.close()
            for name in ('_reference', '_target'):
                value = getattr(self, name)
                if value is not None:
                    value._mmap.close()
                    setattr(self, name, None)
