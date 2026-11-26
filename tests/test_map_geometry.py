import unittest

import numpy as np

from map_geometry import limit_map_geometry


class MapGeometryTests(unittest.TestCase):
    def assert_invariants(self,time,offset):
        t,o,stats=limit_map_geometry(time,offset,.8,1.2)
        original=np.interp(t,time,offset)
        self.assertTrue(np.all(np.diff(t)>0))
        self.assertTrue(np.all(np.abs(o)<=np.abs(original)+1e-12))
        self.assertTrue(np.all((o==0)|(np.sign(o)==np.sign(original))))
        for x,y in zip(time,offset):
            if y==0:
                self.assertTrue(np.any(t==x))
                self.assertEqual(o[np.flatnonzero(t==x)[0]],0.)
        dr=np.diff(t+o);self.assertTrue(np.all(dr>0))
        tempo=np.diff(t)/dr
        self.assertTrue(np.all(tempo>=.8),float(tempo.min()))
        self.assertTrue(np.all(tempo<=1.2),float(tempo.max()))
        return t,o,stats

    def test_valid_map_is_bit_for_bit_unchanged(self):
        time=np.array([0.,.7,2.,10.,12.,40.])
        offset=np.array([-0.,.02,.03,.0,-.15,-.1])
        t,o,stats=limit_map_geometry(time,offset)
        self.assertEqual(t.tobytes(),time.tobytes())
        self.assertEqual(o.tobytes(),offset.tobytes())
        self.assertFalse(stats['changed'])

    def test_positive_and_negative_needles_are_neutralized_locally(self):
        time=np.arange(20,dtype=float)
        offset=np.zeros(20);offset[3]=2.9;offset[12]=-2.3
        t,o,stats=self.assert_invariants(time,offset)
        self.assertGreater(o[3],0);self.assertLess(o[12],0)
        self.assertTrue(stats['changed'])
        self.assertLess(o[3],.17);self.assertGreater(o[12],-.17)

    def test_healthy_centres_and_other_islands_are_retained(self):
        time=np.arange(101,dtype=float)
        offset=np.zeros(101);offset[2:31]=1.1;offset[50:81]=-.6
        t,o,_=self.assert_invariants(time,offset)
        self.assertEqual(o[15],1.1);self.assertEqual(o[65],-.6)
        self.assertEqual(o[31],0);self.assertEqual(o[49],0)

    def test_sign_change_keeps_original_interpolated_zero(self):
        time=np.array([0.,2.,3.,5.,10.])
        offset=np.array([0.,2.,-1.,-.8,0.])
        t,o,stats=self.assert_invariants(time,offset)
        crossing=2.+2./3.
        index=np.argmin(np.abs(t-crossing))
        self.assertAlmostEqual(t[index],crossing,places=14)
        self.assertEqual(o[index],0.)
        self.assertGreater(stats['inserted_zero_nodes'],0)

    def test_nonuniform_nodes_and_float_roundoff(self):
        time=np.array([0.,.1,.100000001,1.3,2.1,12.8,12.80000001,18.,24.])
        offset=np.array([0.,.3,.5,.4,-.7,-.4,-.6,.2,0.])
        self.assert_invariants(time,offset)

    def test_realistic_long_track_and_quarter_second_apply_grid(self):
        time=np.arange(5248,dtype=float)
        offset=np.zeros(len(time));offset[4598]=1.097164
        offset[900:970]=1.2;offset[4600:4630]=-.7
        t,o,_=self.assert_invariants(time,offset)
        apply=np.arange(0,time[-1],.25)
        mapped=apply+np.interp(apply,t,o)
        tempo=np.diff(apply)/np.diff(mapped)
        self.assertTrue(np.all((tempo>=.8)&(tempo<=1.2)))
        self.assertEqual(o[np.flatnonzero(t==930.)[0]],1.2)

    def test_randomized_mixed_sign_caps_remain_renderable(self):
        rng=np.random.default_rng(933)
        for _ in range(40):
            time=np.r_[0.,np.cumsum(rng.uniform(.03,5.,70))]
            offset=rng.normal(0,1.5,len(time));offset[rng.random(len(time))<.15]=0.
            self.assert_invariants(time,offset)

    def test_invalid_input_is_rejected(self):
        for t,o in [([0,0],[0,1]),([0,1],[0,float('nan')]),([0,1],[0]),([0],[0])]:
            with self.assertRaises(ValueError):limit_map_geometry(t,o)
        with self.assertRaises(ValueError):limit_map_geometry([0,1],[0,1],1.2,1.3)


if __name__=='__main__':unittest.main()
