"""A globally confirmed affine clock needs no artificial zero-offset end caps.

Training learns the model once. Separate reference-grid observations must
confirm it before it can bypass LOCAL tapering. Quarantine or a protected PCM
regression always prevents this path. No measured errors are filled with zero.
"""
from __future__ import annotations

import itertools
import math
import numpy as np


def fit_global_clock(anchors, duration, *, window_sec=6.):
    if len(anchors) < 5 or duration <= window_sec:
        return None
    times=np.asarray([r['target_time'] for r in anchors],dtype=float)
    offsets=np.asarray([r['offset'] for r in anchors],dtype=float)
    uncertainty=np.asarray([r.get('uncertainty_sec') or .020 for r in anchors],dtype=float)
    if not np.all(np.isfinite(np.r_[times,offsets,uncertainty])) or np.ptp(times)<=0:
        return None
    if np.ptp(times) < .70*max(1.,duration-window_sec):
        return None
    matrix=np.column_stack([times,np.ones(len(times))])
    slope,intercept=np.linalg.lstsq(matrix,offsets,rcond=None)[0]
    error=offsets-(slope*times+intercept)
    tolerance=max(.010,float(np.median(uncertainty)))
    if abs(slope)>.08 or np.max(np.abs(error))>tolerance:
        return None
    return {'rate':float(1+slope),'intercept_sec':float(intercept),
            'uncertainty_sec':max(.010,float(np.median(uncertainty)),float(np.max(np.abs(error)))),
            'training_points':len(anchors),'training_span_sec':float(np.ptp(times)),
            'training_max_error_sec':float(np.max(np.abs(error))),
            'target_duration':float(duration),'window_sec':float(window_sec),
            'source':'independent_spectral_proposal_anchors'}


def map_with_global_clock(mapping, model):
    if model is None:
        return mapping
    result=dict(mapping)
    times=np.asarray(mapping['target_time'],dtype=float)
    result['offset']=(model['rate']-1)*times+model['intercept_sec']
    result['reference_time']=times+result['offset']
    return result


def verify_global_clock(candidate, records, duration, *, quarantined=False):
    if quarantined:
        return None
    feature=[r for r in records if r.get('feature_backend')]
    model=next((r.get('coarse_global_clock') for r in feature if r.get('coarse_global_clock')),None)
    if not model or model.get('source')!='independent_spectral_proposal_anchors':
        return None
    # A positive competing clock is different from absence of sound. The
    # global exception must never bridge a contradictory edit, even when the
    # remaining controls exceed the ordinary coverage threshold.
    if any(r.get('feature_status') in {'AMBIGUOUS', 'BOUNDARY'}
           or r.get('feature_verification_lost') for r in feature):
        return None
    try:
        rate,intercept,u=map(float,(model['rate'],model['intercept_sec'],model['uncertainty_sec']))
        if not np.isfinite([rate,intercept,u]).all() or not .92<=rate<=1.08 or u<0:
            return None
        if not math.isfinite(float(duration)) or duration<=0 or abs(float(model['target_duration'])-duration)>.001:
            return None
        for row in feature:
            # Even a window without a complete local fit can contain a strong
            # competing clock. A lone peak cannot prove one: isolated false
            # peaks are precisely why the matcher requires several disjoint
            # events. Reverse probes measure SOURCE minus the fixed REF time.
            contrary=[]
            for probe in row.get('probes', []) + row.get('holdouts', []):
                if not probe.get('passes_peak_gate'):
                    continue
                query=float(probe['target_time'])
                observed=float(probe['offset'])
                expected=(query-intercept)/rate-query
                if not np.isfinite([query,observed]).all():
                    return None
                if abs(observed-expected)>max(.04,2*u+.0015):
                    width=max(.01,float(probe.get('window_sec',.6)))
                    contrary.append((query,observed,width))
            # Match the proposal's minimum temporal support. Repeated or
            # overlapping observations of one event cannot provide 3 votes.
            for group in itertools.combinations(contrary,3):
                xx=np.asarray([p[0] for p in group])
                yy=np.asarray([p[1] for p in group])
                if np.ptp(xx)<.5*float(model['window_sec']):
                    continue
                if any(abs(a[0]-b[0]) < (a[2]+b[2])/2-1e-9
                       or abs(a[0]+a[1]-b[0]-b[1]) < 1.08*(a[2]+b[2])/2-1e-9
                       for a,b in itertools.combinations(group,2)):
                    continue
                slope,zero=np.polyfit(xx-xx[0],yy,1)
                if abs(slope)<=.08 and np.max(np.abs(yy-(slope*(xx-xx[0])+zero)))<=.025:
                    return None
        times=np.asarray(candidate['target_time'],dtype=float)
        offsets=np.asarray(candidate['offset'],dtype=float)
        if (len(times)<2 or times.shape!=offsets.shape
                or not np.isfinite(np.r_[times,offsets]).all()
                or np.any(np.diff(times)<=0) or abs(times[0])>.001 or abs(times[-1]-duration)>.001
                or np.max(np.abs(offsets-((rate-1)*times+intercept)))>1e-8):
            return None
        known=[r for r in feature if r.get('feature_status')=='CONFIRMED'
               and r.get('source_event_time') is not None and r.get('before_error') is not None
               and r.get('after_error') is not None]
        if len(known)<5 or len(known)<.70*max(1,len(feature)):
            return None
        source_times=[float(r['source_event_time']) for r in known]
        required_span=min(20.,.75*max(1.,duration-float(model['window_sec'])))
        if np.ptp(source_times)<required_span:
            return None
        improvements=0
        for r in known:
            before=float(r['before_error']);after=float(r['after_error'])
            rowu=max(0.,float(r.get('uncertainty_sec') or 0.))
            source=float(r['source_event_time'])
            observed=float(r['before_selected_offset'])
            if abs(observed-((rate-1)*source+intercept))>u+rowu+.0015:
                return None
            if after>before+.0015 or r.get('feature_verification_lost'):
                return None
            improvements+=before-after>2*rowu+.0015
        for r in records:
            if r.get('evidence_kind')!='protected_first_channel_pcm':
                continue
            if (r.get('feature_verification_lost') or r.get('after_error') is None
                    or float(r['after_error'])>float(r['before_error'])+.0015):
                return None
        if improvements<max(3,math.ceil(.70*len(known))):
            return None
        return {**model,'independent_validation_points':len(known),
                'independent_validation_span_sec':float(np.ptp(source_times)),
                'independent_improved_points':int(improvements),
                'policy':'confirmed global clock retained through edges; no local zero-offset fade'}
    except (KeyError,TypeError,ValueError):
        return None
