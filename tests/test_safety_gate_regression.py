"""Numeric regressions from the 2026-11-26 pre-render map failure.

The fixture contains time-map nodes and validation measurements only: no
audio, filenames, user directories, or terminal logs are included.
"""
import contextlib
import io
import json
from pathlib import Path
import unittest

import numpy as np

from applier import build_apply_map, validate_apply_map_limits
from common import Progress
from render_guard import _merge_centers
from safety import build_non_regressive_sync_map


class SafetyGateRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).parent/'fixtures/safety_gate_20261126.json'
        cls.fixture=json.loads(path.read_text(encoding='utf-8'))

    def recorded_candidate(self):
        times=np.asarray(self.fixture['target_time'],dtype=np.float64)
        offsets=np.asarray(self.fixture['candidate_offset'],dtype=np.float64)
        return {'target_time':times,'offset':offsets,'reference_time':times+offsets,
                'confidence':np.ones_like(times)}

    def build_recorded(self,radius):
        candidate=self.recorded_candidate()
        before={name:value.copy() for name,value in candidate.items()}
        # This is the same path as the render guard, including its collection
        # of persisted quarantine centers before reconstructing the safe map.
        centers=_merge_centers(self.fixture['quarantine_centers'],radius)
        with contextlib.redirect_stdout(io.StringIO()):
            mapping,report=build_non_regressive_sync_map(candidate,
                self.fixture['validation_results'],self.fixture['target_duration'],Progress(),
                quarantine_times=centers,quarantine_radius_sec=radius)
        for name,value in before.items():
            np.testing.assert_array_equal(candidate[name],value,err_msg='Source candidate was mutated: '+name)
        return mapping,report

    def test_recorded_render_rebuild_produces_monotone_renderable_map(self):
        mapping,_=self.build_recorded(30.)
        self.assertTrue(np.all(np.diff(mapping['reference_time'])>0))
        # The old explicit accepted-point override resurrected this isolated
        # 1.097164 s spike even though its interval gate was zero.
        self.assertEqual(float(np.interp(4598.,mapping['target_time'],mapping['offset'])),0.)
        with contextlib.redirect_stdout(io.StringIO()):
            apply=build_apply_map(mapping,self.fixture['target_duration'],Progress())
            validate_apply_map_limits(apply)
        self.assertTrue(np.isfinite(apply['tempo']).all())
        self.assertGreaterEqual(float(np.min(apply['tempo'])),.8)
        self.assertLessEqual(float(np.max(apply['tempo'])),1.2)

    def test_recorded_quarantines_remain_zero_across_their_whole_union(self):
        for radius in (20.,30.):
            with self.subTest(radius=radius):
                mapping,_=self.build_recorded(radius)
                times=np.asarray(self.fixture['target_time'])
                centers=np.asarray(self.fixture['quarantine_centers'])
                protected=np.any(np.abs(times[:,None]-centers[None,:])<=radius,axis=1)
                offsets=np.interp(times,mapping['target_time'],mapping['offset'])
                np.testing.assert_array_equal(offsets[protected],np.zeros(int(protected.sum())))

    def test_recorded_report_counts_all_quarantined_validation_points(self):
        for radius in (20.,30.):
            with self.subTest(radius=radius):
                _,report=self.build_recorded(radius)
                expected=self.fixture['expected_quarantine_coverage'][str(int(radius))]
                self.assertEqual(report['total_validation_points'],522)
                self.assertEqual(report['quarantined_validation_points'],expected)
                self.assertAlmostEqual(report['quarantine_fraction'],expected/522.,places=14)
        self.assertEqual(self.fixture['expected_quarantine_coverage']['20'],484)

    def test_merging_centers_does_not_move_or_shrink_protected_intervals(self):
        original=[78.,108.,148.,158.,168.,178.,78.]
        radius=30.
        merged=_merge_centers(original,radius)
        probes=np.unique(np.array([p for c in original for p in (c-radius,c,c+radius)]))
        for point in probes:
            self.assertTrue(any(abs(point-center)<=radius for center in merged),point)
        self.assertEqual(len(merged),len(set(merged)))

    @staticmethod
    def validation_row(time,accepted):
        return {'target_time':float(time),'before_error':.2,'after_error':.001 if accepted else .2,
                'before_offset':.2,'predicted_offset':-.006,'bidir_status':'CONFIRMED_MAP',
                'bidir_left_correlation':.99,'bidir_right_correlation':.99,'bidir_disagreement':0.}

    def test_isolated_accepted_point_cannot_restore_a_rejected_run(self):
        times=np.arange(101,dtype=np.float64)
        candidate={'target_time':times,'offset':np.full_like(times,-.006),
                   'reference_time':times-.006}
        rows=[self.validation_row(t,t==20) for t in range(10,100,10)]
        with contextlib.redirect_stdout(io.StringIO()):
            mapping,_=build_non_regressive_sync_map(candidate,rows,100.,Progress())
        np.testing.assert_array_equal(mapping['offset'],np.zeros_like(mapping['offset']))

    def test_isolated_point_is_removed_without_changing_supported_run(self):
        times=np.arange(101,dtype=np.float64)
        candidate={'target_time':times,'offset':np.full_like(times,-.006),
                   'reference_time':times-.006}
        rows=[self.validation_row(t,t in (20,60,70,80)) for t in range(10,100,10)]
        with contextlib.redirect_stdout(io.StringIO()):
            mapping,_=build_non_regressive_sync_map(candidate,rows,100.,Progress())
            apply=build_apply_map(mapping,100.,Progress())
            validate_apply_map_limits(apply)
        self.assertEqual(float(np.interp(20.,mapping['target_time'],mapping['offset'])),0.)
        np.testing.assert_array_equal(np.interp([60.,70.,80.],mapping['target_time'],mapping['offset']),
                                      np.full(3,-.006))
        np.testing.assert_array_equal(np.interp([10.,30.,40.,50.,90.],mapping['target_time'],mapping['offset']),
                                      np.zeros(5))


if __name__=='__main__':unittest.main()
