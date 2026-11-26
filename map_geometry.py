"""Conservative geometry bounds for piecewise-linear time corrections.

The input is target time and its proposed offset. Only correction magnitude
may decrease: zero barriers and the original pointwise sign are preserved.
No alignment measurements or acceptance thresholds are changed here.
"""
from __future__ import annotations

import numpy as np


def _valid(time,offset,min_tempo,max_tempo):
    dt=np.diff(time)
    dr=np.diff(time+offset)
    if np.any(dr<=0) or not np.isfinite(dr).all():return False
    tempo=dt/dr
    return bool(np.all((tempo>=min_tempo)&(tempo<=max_tempo)))


def limit_map_geometry(target_time,offset,min_tempo=.8,max_tempo=1.2):
    """Return ``(time, offset, stats)`` with renderable local time geometry.

    A valid float64 map is returned bit-for-bit unchanged. For an invalid map,
    each same-sign correction is capped by forward/backward slope envelopes.
    Positive and negative corrections use the appropriate asymmetric bounds:
    ``1/max_tempo - 1 <= d(offset)/dt <= 1/min_tempo - 1``.

    A sign change gains its original interpolated zero crossing. This prevents
    reducing either side from shifting the crossing through a protected region.
    If that interior time is not representable, both adjacent corrections are
    conservatively zeroed. A final floating-point check can neutralize a
    numerically unrepresentable component, never increase another correction.
    """
    time=np.asarray(target_time,dtype=np.float64)
    proposal=np.asarray(offset,dtype=np.float64)
    if (time.ndim!=1 or proposal.ndim!=1 or len(time)!=len(proposal) or len(time)<2
            or not np.isfinite(time).all() or not np.isfinite(proposal).all()
            or not np.isfinite(np.diff(time)).all() or np.any(np.diff(time)<=0)):
        raise ValueError('Карта должна содержать конечные значения на строго возрастающей сетке времени.')
    if not (np.isfinite(min_tempo) and np.isfinite(max_tempo) and 0<min_tempo<1<max_tempo):
        raise ValueError('Пределы скорости должны быть положительными и содержать 1 внутри диапазона.')
    stats={'changed':False,'inserted_zero_nodes':0,'changed_points':0,'precision_neutralized_components':0}
    if _valid(time,proposal,min_tempo,max_tempo):
        return time.copy(),proposal.copy(),stats

    # Keep exact original nodes, including their zero barriers. Interpolating
    # across a sign change only after clipping would move its original zero.
    times=[];values=[]
    working=proposal.copy()
    for i in range(len(time)-1):
        times.append(float(time[i]));values.append(float(working[i]))
        left,right=proposal[i],proposal[i+1]
        if left==0 or right==0 or (left>0)==(right>0):continue
        # A ratio of magnitudes avoids overflow in left/(left-right).
        scale=max(abs(left),abs(right))
        fraction=(abs(left)/scale)/(abs(left)/scale+abs(right)/scale)
        zero=float(time[i]+(time[i+1]-time[i])*fraction)
        if time[i]<zero<time[i+1]:
            times.append(zero);values.append(0.)
            stats['inserted_zero_nodes']+=1
        else:
            # Adjacent machine floats can have no representable interior node.
            values[-1]=0.;working[i+1]=0.
    times.append(float(time[-1]));values.append(float(working[-1]))
    time=np.asarray(times,dtype=np.float64)
    cap=np.asarray(values,dtype=np.float64)
    signs=np.sign(cap)
    amplitude=np.abs(cap)
    dt=np.diff(time)
    lower=1./max_tempo-1.
    upper=1./min_tempo-1.

    # Reserve numerical room inside the *same* tempo limits. It is a small
    # tightening, not a tolerance that accepts an out-of-bounds render. The
    # source magnitude also covers cancellation in reference=time+offset.
    magnitude=np.maximum(1.,np.maximum(np.abs(time),np.abs(time+cap)))
    guard=np.maximum(1e-10,128*np.finfo(np.float64).eps*np.maximum(magnitude[:-1],magnitude[1:])/dt)
    guard=np.minimum(guard,min(-lower,upper)*.25)
    rising_positive=upper-guard
    falling_positive=-lower-guard
    # Two min-plus passes give the largest feasible same-sign envelope below
    # the proposal for these slightly tightened floating-point-safe slopes.
    for i in range(1,len(time)):
        rate=rising_positive[i-1] if signs[i]>=0 else falling_positive[i-1]
        amplitude[i]=min(amplitude[i],amplitude[i-1]+rate*dt[i-1])
    for i in range(len(time)-2,-1,-1):
        rate=falling_positive[i] if signs[i]>=0 else rising_positive[i]
        amplitude[i]=min(amplitude[i],amplitude[i+1]+rate*dt[i])
    result=signs*amplitude

    # Extreme timestamp magnitudes can defeat even a tightened analytic bound
    # through float addition. Drop only connected nonzero components incident
    # to a bad edge; all other already-valid islands remain intact.
    if not _valid(time,result,min_tempo,max_tempo):
        dr=np.diff(time+result)
        with np.errstate(divide='ignore',invalid='ignore'):
            tempo=dt/dr
        bad=(dr<=0)|~np.isfinite(dr)|(tempo<min_tempo)|(tempo>max_tempo)
        starts=np.flatnonzero((result!=0)&np.r_[True,result[:-1]==0])
        ends=np.flatnonzero((result!=0)&np.r_[result[1:]==0,True])+1
        for start,end in zip(starts,ends):
            if np.any(bad[max(0,start-1):min(len(bad),end)]):
                result[start:end]=0.
                stats['precision_neutralized_components']+=1
    if not _valid(time,result,min_tempo,max_tempo):
        # All corrections can always be neutralized without changing the source
        # timeline. This final fail-closed path is only for arithmetic extremes.
        result[:]=0.
        stats['precision_neutralized_components']+=1
    original=np.interp(time,np.asarray(target_time,dtype=np.float64),proposal)
    stats.update(changed=True,changed_points=int(np.count_nonzero(result!=original)))
    return time,result,stats
