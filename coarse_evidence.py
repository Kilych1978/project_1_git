"""Map-independent spectral observations on a fixed reference clock.

This is optional coarse evidence, not a sample-accurate PCM estimator.  A
reference window selects its source event once.  Candidate maps only move that
event mathematically; actual renders must match the same reference window with
the same frozen channels and frequency band.
"""
from __future__ import annotations

from dataclasses import replace
import copy
import math
from pathlib import Path

import numpy as np

from coarse_features import build_features_file
from coarse_matcher import match_window


EVIDENCE_KIND = 'spectral_common_audio'


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _selection(result):
    return {key: result.get(key) for key in ('reference_view', 'target_view', 'band')}


class CoarseEvidence:
    """Independent coarse clock observations backed by bounded feature banks.

    ``observations`` is the proposal grid. ``validate`` uses a different phase
    on the fixed REFERENCE clock; neither search consults the proposed map.
    Source observations and view/band selections are cached across all maps.
    """

    def __init__(self, reference_bank, target_bank, search_radius=15.0,
                 window_sec=6.0, *, _source=None):
        if not math.isfinite(float(search_radius)) or search_radius <= 0:
            raise ValueError('Coarse search radius must be positive and finite.')
        if not math.isfinite(float(window_sec)) or window_sec < 3:
            raise ValueError('Coarse window must be at least three seconds.')
        self.reference_bank = reference_bank
        self.target_bank = target_bank
        self.search_radius = float(search_radius)
        self.window_sec = float(window_sec)
        self._source = _source
        self._cache = {}
        self._owns_target = False

    def _schedule(self, interval_sec, phase):
        interval = float(interval_sec)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('Validation interval must be positive and finite.')
        duration = float(self.reference_bank.duration)
        half = self.window_sec / 2.0
        if duration <= 0:
            return []
        begin, end = half, duration - half
        if end < begin:
            # Keep an explicit unverified observation for very short material.
            return [duration / 2.0]
        first = math.ceil(begin / interval - phase)
        last = math.floor(end / interval - phase)
        times = [(i + phase) * interval for i in range(first, last + 1)]
        return times or [(begin + end) / 2.0]

    def _original(self):
        return self if self._source is None else self._source._original()

    def _observe(self, reference_time):
        key = round(float(reference_time), 9)
        if key in self._cache:
            return self._cache[key]
        source = self._original()
        original = None if self._source is None else source._observe(reference_time)
        if original is not None and original['status'] != 'CONFIRMED':
            # An unknown source correspondence cannot become an improvement by
            # selecting a new channel in the output.
            result = copy.deepcopy(original)
            result['reason'] = 'The original reference/source correspondence is unverified.'
            result['rendered_target_time'] = None
        else:
            frozen = None if original is None else original['reverse_selection']
            reverse = match_window(
                self.target_bank, self.reference_bank, float(reference_time),
                search_center=0.0, search_radius=self.search_radius,
                window_sec=self.window_sec, frozen_views=frozen,
            )
            raw_offset, raw_rate = _number(reverse.get('offset')), _number(reverse.get('rate'))
            confirmed = reverse.get('status') == 'CONFIRMED' and raw_offset is not None and raw_rate is not None and raw_rate > 0
            event_time = float(reference_time) + raw_offset if confirmed else None
            result = {
                'status': 'CONFIRMED' if confirmed else reverse.get('status', 'UNVERIFIABLE'),
                'target_time': event_time,
                'reference_time': float(reference_time),
                'offset': None if event_time is None else float(reference_time) - event_time,
                'rate': None if not confirmed else 1.0 / raw_rate,
                'score': float(reverse.get('score') or 0.0),
                'margin': float(reverse.get('margin') or 0.0),
                'uncertainty_sec': (None if not confirmed else _number(reverse.get('uncertainty_sec'))),
                # Reverse search swaps the signal roles, but public labels use
                # the normal REF/TARGET convention.
                'reference_view': reverse.get('target_view'),
                'target_view': reverse.get('reference_view'),
                'band': reverse.get('band'),
                'reverse_selection': _selection(reverse),
                'probes': copy.deepcopy(reverse.get('probes', [])),
                'holdouts': copy.deepcopy(reverse.get('disjoint_holdouts', reverse.get('holdouts', []))),
                'hit_boundary': bool(reverse.get('hit_boundary', False)),
                'reason': reverse.get('reason', ''),
                'observation_direction': 'source_time_from_fixed_reference',
                'comparison_key': f'reference:{float(reference_time):.9f}',
                'evidence_kind': EVIDENCE_KIND,
            }
            if original is not None:
                result['rendered_target_time'] = event_time
                # Pairing is always in ORIGINAL source coordinates, even after
                # the output has been stretched, trimmed, or shifted.
                result['target_time'] = original['target_time']
        self._cache[key] = result
        return result

    def observations(self, interval_sec=30.0, phase=0.5):
        """Return proposal anchors without exposing mutable cached records.

        Confirmed anchors have target_time/source event, reference_time/fixed
        query, offset=REF−TARGET and rate=dREF/dTARGET. Unknown events use None.
        """
        return [copy.deepcopy(self._observe(t)) for t in self._schedule(interval_sec, phase)]

    @staticmethod
    def _mapped_offset(sync_map, source_time):
        times = np.asarray(sync_map['target_time'], dtype=float)
        offsets = np.asarray(sync_map['offset'], dtype=float)
        if not len(times) or times.shape != offsets.shape or not np.all(np.isfinite(times)) or not np.all(np.isfinite(offsets)):
            raise ValueError('A finite coarse synchronization map is required.')
        return float(np.interp(source_time, times, offsets))

    def _row(self, reference_time, sync_map):
        observation = self._observe(reference_time)
        original = self._original()._observe(reference_time)
        before_known = original['status'] == 'CONFIRMED'
        after_known = observation['status'] == 'CONFIRMED' and before_known
        source_time = original['target_time']
        predicted = None if source_time is None else self._mapped_offset(sync_map, source_time)
        before = original['offset'] if before_known else None
        after = observation['offset'] if after_known else None
        # A source-map residual is inferred; an actual-output residual is
        # measured against zero, regardless of the original map prediction.
        expected = 0.0 if self._source is not None else predicted
        error = None if after is None else abs(after - expected)
        holdouts = observation.get('holdouts', [])
        passed = [p for p in holdouts if p.get('confirms_frozen_model', True)]
        residuals = [_number(p.get('residual_sec')) for p in passed]
        residuals = [x for x in residuals if x is not None]
        # These are affine-fit residuals, not offset variation due to speed.
        disagreement = None if not residuals else (max(residuals) - min(residuals)) * (observation.get('rate') or 1.0)
        left = passed[0] if passed else {}
        right = passed[-1] if passed else {}
        uncertainty = observation.get('uncertainty_sec')
        before_uncertainty = original.get('uncertainty_sec')
        status = ('CONFIRMED_MAP' if error <= max(.1, uncertainty or 0.0)
                  else 'CONFIRMED_DISCREPANCY') if after_known else 'UNRESOLVED'
        if not after_known and observation['status'] == 'AMBIGUOUS':
            status = 'AMBIGUOUS'
        before_status = ('CONFIRMED_MAP' if abs(before) <= max(.1, before_uncertainty or 0.0)
                         else 'CONFIRMED_DISCREPANCY') if before_known else 'UNRESOLVED'
        corr = observation['score'] if after_known else None
        before_corr = original['score'] if before_known else None
        return {
            'target_time': float(reference_time if source_time is None else source_time),
            'source_event_time': source_time,
            'source_event_known': before_known,
            'reference_time': float(reference_time),
            'comparison_reference_time': float(reference_time),
            'comparison_key': original['comparison_key'],
            'rendered_target_time': observation.get('rendered_target_time'),
            'predicted_offset': 0.0 if predicted is None else predicted,
            'before_offset': before, 'before_selected_offset': before,
            'before_correlation': before_corr, 'before_selected_correlation': before_corr,
            'before_bidirectional_offset': before,
            'before_bidirectional_correlation': before_corr,
            'before_bidirectional_status': before_status,
            'before_estimator': EVIDENCE_KIND, 'after_estimator': EVIDENCE_KIND,
            'before_error': None if before is None else abs(before),
            'before_error_raw': None if before is None else abs(before),
            'after_offset': after, 'after_correlation': corr, 'after_error': error,
            'raw_after_offset': after, 'raw_after_correlation': corr,
            'bidir_left_offset': after, 'bidir_left_correlation': left.get('score', corr),
            'bidir_right_offset': after, 'bidir_right_correlation': right.get('score', corr),
            'bidir_consensus_offset': after, 'bidir_consensus_correlation': corr,
            'bidir_disagreement': disagreement, 'bidir_prediction_error': error,
            'bidir_status': status,
            'feature_backend': True, 'evidence_kind': EVIDENCE_KIND,
            'feature_status': observation['status'],
            'feature_original_status': original['status'],
            'feature_verification_lost': bool(self._source is not None and before_known and not after_known),
            'feature_actual_output': self._source is not None,
            'estimator': EVIDENCE_KIND, 'uncertainty_sec': uncertainty,
            'before_uncertainty_sec': before_uncertainty,
            'feature_rate': observation.get('rate'),
            'reference_view': original['reference_view'], 'target_view': original['target_view'],
            'feature_band': original['band'], 'frozen_selection': dict(original['reverse_selection']),
            'probes': copy.deepcopy(observation['probes']),
            'holdouts': copy.deepcopy(holdouts), 'feature_hit_boundary': observation['hit_boundary'],
            'feature_reason': observation['reason'],
        }

    def validate(self, sync_map, target_duration, progress=None, interval_sec=60.0):
        """Validate a proposed map or actual output on fixed reference windows."""
        requested_interval = 60.0 if interval_sec is None else float(interval_sec)
        reference_duration = float(self.reference_bank.duration)
        # Short excerpts need enough independent controls to establish the
        # same supported interval required for long recordings. Densify the
        # observations, never the acceptance thresholds. The unchanged REF
        # clock gives source maps and real outputs exactly the same schedule.
        effective_interval = requested_interval
        if reference_duration > self.window_sec:
            effective_interval = min(requested_interval, max(1.0, (reference_duration - self.window_sec) / 10.0))
            # On short excerpts an unlucky coarse phase can miss most usable
            # events near an edge. Measure a fixed one-second grid instead of
            # weakening the independent span/coverage requirements.
            if reference_duration <= 60.0:
                effective_interval = min(effective_interval, 1.0)
        times = self._schedule(effective_interval, 0.0)
        if progress is not None:
            progress.stage('НЕЗАВИСИМАЯ ПРОВЕРКА ОБЩИХ ЗВУКОВЫХ СОБЫТИЙ', 0.0)
        rows = []
        for index, query in enumerate(times, 1):
            row = self._row(query, sync_map)
            row['feature_validation_interval_sec'] = effective_interval
            row['feature_requested_interval_sec'] = requested_interval
            rows.append(row)
            if progress is not None:
                progress.update(100.0 * index / max(1, len(times)),
                                f'Спектральная проверка | эталон {query:.3f} с')
        if progress is not None:
            progress.done(100.0, 'Спектральная проверка завершена')
        return rows

    def for_output(self, output_info, output_dir, progress=None):
        """Decode the actual output, retaining original reference/view choices."""
        original = self._original()
        physical = original.target_bank.parameters.get('physical_channels', [])
        info = output_info
        # WAV may drop a known channel mask or acquire an inferred stereo mask.
        # Rendering preserves channel order. Keep declared physical labels, or
        # the original index-only labels, without inventing spatial identities.
        original_indices = bool(physical) and all(p.startswith('c') and p[1:].isdigit() for p in physical)
        if len(physical) == output_info.channels and (not output_info.channel_layout or original_indices):
            info = replace(output_info, channel_layout=list(physical))
        bank = build_features_file(info, Path(output_dir) / '_render_spectral_features', progress)
        evidence = CoarseEvidence(original.reference_bank, bank, self.search_radius,
                                  self.window_sec, _source=original)
        evidence._owns_target = True
        return evidence

    def close(self):
        if self._owns_target:
            self.target_bank.close()
            self._owns_target = False
