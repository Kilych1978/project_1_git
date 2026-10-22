"""Propose a local replacement, preserving the verified map as the baseline.

Both maps describe the same original TARGET sample clock. Their difference
is piecewise cubic, so a Hermite curve on the union of their knots is exact.
No intermediate audio or map from an earlier run is needed.
"""
from dataclasses import replace
import numpy as np

from .micro import measure_window
from .model import Piece,TimeMap
from .guard import verify_candidate_pcm,holdout_times,grid,cancel_unsafe_components,same_map,_signature


def replace_map(base,proposal):
    if len(base.pieces)!=len(proposal.pieces):
        raise ValueError('Кандидаты должны описывать одни и те же участки исходника.')
    pieces = []
    for old,new in zip(base.pieces,proposal.pieces):
        if abs(old.target[0]-new.target[0])>1e-10 or abs(old.target[-1]-new.target[-1])>1e-10:
            raise ValueError('Области кандидатов различаются.')
        knots = np.unique(np.r_[old.target,new.target])
        curve = {'target':knots.tolist(),
                 'delta':(new.reference_at(knots)-old.reference_at(knots)).tolist(),
                 'derivative':(new.rate_at(knots)-old.rate_at(knots)).tolist()}
        pieces.append(Piece(knots,new.reference_at(knots),[*old.corrections,curve]))
    return TimeMap(pieces,list(base.notes))


def verify_reference_grid(reference,target,base,proposal,settings,times,rendered=None,progress=None,
                          row_cache=None,baseline_cache=None):
    """Before/after at the SAME absolute reference times, independent of knots."""
    identity = TimeMap.affine(len(rendered)/settings.analysis_rate) if rendered is not None else None
    rows = []
    row_cache = {} if row_cache is None else row_cache
    baseline_cache = {} if baseline_cache is None else baseline_cache
    def valid(row):
        return bool(row and not row['near_edge'] and row['score']>=settings.min_raw_score
                    and row['margin']>=settings.min_raw_margin)
    length = min(settings.micro_window,settings.anchor_step/8)
    half = length/2 + settings.sinc_radius/settings.analysis_rate
    for index,t in enumerate(times,1):
        old_time,new_time = float(base.inverse([t])[0]),float(proposal.inverse([t])[0])
        if not np.isfinite(old_time):
            continue
        cache_key = (settings,float(t),_signature(proposal,new_time,settings)) if rendered is None and np.isfinite(new_time) else None
        if cache_key is not None and cache_key in row_cache:
            rows.append(row_cache[cache_key])
            if progress:
                progress(index,len(times))
            continue
        key = (settings,float(t))
        if key not in baseline_cache:
            baseline_cache[key] = measure_window(reference,target,base,old_time,settings,
                                                  quantize_pcm16=settings.verification_pcm16)
        before = baseline_cache[key]
        expected = (measure_window(reference,target,proposal,new_time,settings,
                                  quantize_pcm16=settings.verification_pcm16)
                    if np.isfinite(new_time) else None)
        after = expected if rendered is None else measure_window(reference,rendered,identity,t,settings)
        probes = np.linspace(t-half,t+half,9)
        old_positions,new_positions = base.inverse(probes),proposal.inverse(probes)
        finite = np.isfinite(old_positions)&np.isfinite(new_positions)
        changed = (not np.array_equal(np.isfinite(old_positions),np.isfinite(new_positions)) or
                   (finite.any() and np.max(np.abs(old_positions[finite]-new_positions[finite]))>=settings.min_improvement))
        if hasattr(proposal,'extent'):
            changed=proposal.extent(base,probes)>=settings.min_improvement
        status = 'VERIFIED' if valid(after) else 'UNVERIFIABLE'
        if changed and (not valid(before) or not valid(after) or not valid(expected)):
            status = 'REGRESSION'
        if valid(before) and valid(after) and abs(after['correction_sec'])>abs(before['correction_sec'])+1/settings.analysis_rate:
            status = 'REGRESSION'
        if valid(expected) and valid(after) and expected['method']==after['method'] and after['score']<expected['score']-.025:
            status = 'REGRESSION'
        row = {'target_time':old_time,'reference_time':float(t),'micro_changed':bool(changed),
                     'baseline_residual_sec':before['correction_sec'] if valid(before) else None,
                     'residual_sec':after['correction_sec'] if valid(after) else None,
                     'status':status,'probe_kind':'absolute_reference','window_sec':length}
        rows.append(row)
        if cache_key is not None:
            row_cache[cache_key] = row
        if progress:
            progress(index,len(times))
    return rows


def choose_local(reference,target,base,proposal,schedules,settings,reference_times,log=print,progress=None):
    """Only the extra curve may be rolled back. Base remains byte-for-byte defined."""
    if same_map(base,proposal):
        return base,[],[]
    try:
        original = replace_map(base,proposal).validate(settings.max_rate_change)
    except ValueError as exc:
        return base,[],[{'status':'INVALID_OPTIONAL_MAP','reason':str(exc)}]
    candidate = original
    failed_times = set()
    history = []
    grouped = {}
    for schedule,times in schedules:
        values = times if times is not None else holdout_times(base,schedule)+grid(base,schedule)
        grouped.setdefault(schedule,set()).update(values)
    schedules = [(schedule,sorted(values)) for schedule,values in grouped.items()]
    row_cache,before_cache,fixed_cache,fixed_before = {},{},{},{}
    total_checks = sum(len(times) for _,times in schedules)+2*len(reference_times)
    # Restart each taper family from the original failures: defects introduced
    # by a rejected fast transition must not erase good audio in a slower one.
    strategies = [(settings.max_rate_change,1.,False),
                  (settings.max_rate_change,1.,False),
                  (.002,1.,False),(.002,1.,False),
                  (.001,1.,False),(.001,1.,False),
                  (.001,2.,False),(.001,2.,False),(.001,1.,True)]
    initial_failures = set()
    protected_half_window = max([min(s.micro_window,max(.5,s.anchor_step)/8)/2+s.micro_radius+
                                 s.sinc_radius/s.analysis_rate for s,_ in schedules]+[.057])
    for attempt in range(10):
        if attempt:
            if attempt in (1,3,5,7,9):
                failed_times = set(initial_failures)
            rate,scale,whole = strategies[attempt-1]
            try:
                candidate,_ = cancel_unsafe_components(base,original,sorted(failed_times),
                    settings.max_rate_change,rollback_rate=rate,taper_scale=scale,
                    whole_components=whole,protected_half_window=protected_half_window)
            except ValueError as exc:
                history.append({'attempt':attempt,'status':'INVALID_OPTIONAL_ROLLBACK','reason':str(exc)})
                continue
        if same_map(base,candidate):
            continue
        if progress is not None and hasattr(progress,'begin_pass'):
            progress.begin_pass(attempt+1,10)
        rows = []
        completed = 0
        for schedule,centers in schedules:
            rows.extend(verify_candidate_pcm(reference,target,base,candidate,schedule,target_times=centers,
                row_cache=row_cache,baseline_cache=before_cache,
                progress=(lambda done,total,offset=completed:progress(offset+done,total_checks)) if progress else None))
            completed += len(centers)
        for step in (.125,.5):
            rows.extend(verify_reference_grid(reference,target,base,candidate,
                replace(settings,anchor_step=step),reference_times,row_cache=fixed_cache,baseline_cache=fixed_before,
                progress=(lambda done,total,offset=completed:progress(offset+done,total_checks)) if progress else None))
            completed += len(reference_times)
        bad = [r for r in rows if r['status']=='REGRESSION' or
               (r['micro_changed'] and r['status']=='UNVERIFIABLE')]
        improved = [r for r in rows if r['status']=='VERIFIED' and r['micro_changed']
                    and r['baseline_residual_sec'] is not None and r['residual_sec'] is not None
                    and abs(r['residual_sec'])+settings.min_improvement<abs(r['baseline_residual_sec'])]
        history.append({'attempt':attempt,'bad':len(bad),'improved':len(improved)})
        log(history[-1])
        if not bad and len({round(r['target_time'],9) for r in improved})>=3:
            return candidate,rows,history
        if not bad:
            return base,[],history
        failed_times.update(r['target_time'] for r in bad)
        if attempt==0:
            initial_failures = set(failed_times)
    return base,[],history
