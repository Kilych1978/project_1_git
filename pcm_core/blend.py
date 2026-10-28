"""Local affine PCM patches with amplitude crossfades, never delay ramps.

A patch samples the same original target at a separately measured clock.
Weights apply to every channel identically. Outside accepted intervals the
protected baseline is copied. This is an audio sampling plan, not a single
invertible time map; the nominal timeline remains the baseline's timeline.
"""
from dataclasses import replace
import numpy as np
from .model import TimeMap
from .dsp import sinc_sample


def sample_map(source,mapping,times,sample_rate,radius):
    if isinstance(mapping,BlendMap):
        return mapping.sample(source,times,sample_rate,radius)
    positions,rate=mapping.inverse_with_rate(times)
    cutoff=np.minimum(1.,1/np.maximum(np.nan_to_num(rate,nan=1.),1.))
    return sinc_sample(source,positions*sample_rate,radius,cutoff)


class BlendMap:
    def __init__(self,base,layers):
        self.base=base
        self.layers=layers
        self.pieces=base.pieces
        self.notes=[*base.notes,'Local PCM patches use amplitude crossfades; see sampling_layers.']
        self._starts=np.array([layer['reference'][0] for layer in layers])
        self._ends=np.array([layer['reference'][-1] for layer in layers])

    def local_layers(self,times):
        times=np.asarray(times)
        finite=times[np.isfinite(times)]
        if not len(finite):return []
        lo=np.searchsorted(self._ends,np.min(finite),side='left')
        hi=np.searchsorted(self._starts,np.max(finite),side='right')
        return self.layers[lo:hi]

    def forward(self,times):return self.base.forward(times)
    def inverse(self,times):return self.base.inverse(times)
    def inverse_rate(self,times):return self.base.inverse_rate(times)
    def inverse_with_rate(self,times):return self.base.inverse_with_rate(times)
    def local_rate(self,t):return self.base.local_rate(t)

    def weight(self,layer,times):
        return np.interp(times,layer['reference'],layer['weight'],left=0.,right=0.)

    def sample(self,source,times,sr,radius):
        values=sample_map(source,self.base,times,sr,radius)
        for layer in self.local_layers(times):
            weight=self.weight(layer,times)
            active=weight>0
            if not active.any():continue
            positions=(np.asarray(times)[active]-layer['offset'])/layer['rate']*sr
            aligned=sinc_sample(source,positions,radius,min(1.,layer['rate']))
            w=weight[active] if values.ndim==1 else weight[active,None]
            values[active]=values[active]*(1-w)+aligned*w
        return values

    def validate(self,max_rate_change=.08):
        self.base.validate(max_rate_change)
        last=-np.inf
        for layer in self.layers:
            x=np.asarray(layer['reference']);w=np.asarray(layer['weight'])
            if (len(x)<2 or x.shape!=w.shape or not np.isfinite(x).all() or
                not np.isfinite(w).all() or not (np.diff(x)>0).all() or
                np.min(w)<0 or np.max(w)>1 or x[0]<last-1e-10 or
                abs(w[0])>1e-12 or abs(w[-1])>1e-12 or
                not np.isfinite(layer['offset']) or abs(layer['rate']-1)>max_rate_change):
                raise ValueError('Некорректный план локального PCM-соединения.')
            last=x[-1]
        return self

    def extent(self,other,ref_times):
        # Conservative: cancellation of nominal delays cannot hide a blend.
        a=self.base.inverse(ref_times);b=other.inverse(ref_times)
        extent=np.nan_to_num(np.abs(a-b),nan=0.)
        if isinstance(self.base,BlendMap) and self.base is not other:
            # A later refinement must not hide PCM changes in its preserved
            # inner layer merely because the nominal clock is unchanged.
            extent+=self.base.extent(other,ref_times)
        for layer in self.local_layers(ref_times):
            aligned=(np.asarray(ref_times)-layer['offset'])/layer['rate']
            extent+=self.weight(layer,ref_times)*np.nan_to_num(np.abs(aligned-a),nan=0.)
        return float(np.max(extent)) if len(extent) else 0.

    def reduced(self,bad_reference_times,half_window=.06,protected_base=None):
        layers=[];removed=[]
        bad_times=np.sort(np.asarray(bad_reference_times))
        base=self.base
        if protected_base is not None and isinstance(base,BlendMap) and base is not protected_base:
            base,inner_removed=base.reduced(bad_reference_times,half_window,protected_base)
            removed.extend(inner_removed)
        for layer in self.layers:
            x=np.asarray(layer['reference']);w=np.asarray(layer['weight'])
            bad=bad_times[np.searchsorted(bad_times,x[0]-half_window):
                          np.searchsorted(bad_times,x[-1]+half_window,side='right')]
            if len(bad):
                points=np.unique(np.r_[x,*[[t-half_window-.012,t-half_window,
                                           t+half_window,t+half_window+.012] for t in bad]])
                points=points[(points>=x[0])&(points<=x[-1])]
                weights=np.interp(points,x,w)
                for t in bad:
                    mask=np.clip((np.abs(points-t)-half_window)/.012,0.,1.)
                    weights*=mask
                    removed.append({'reference_time':float(t),'half_window_sec':half_window})
                x,w=points,weights
            if np.max(w)>0:
                layers.append({**layer,'reference':x.tolist(),'weight':w.tolist()})
        return (BlendMap(base,layers) if layers else base),removed

    def to_dict(self):
        return {'representation':'pcm_overlap_add','base':self.base.to_dict(),
                'sampling_layers':self.layers,'notes':self.notes}

    def support_components(self):
        """Separate positive-weight islands without changing a single weight.

        Local cancellation can leave several safe islands in one clock patch.
        They must remain independently eligible when another island fails.
        """
        parts=[]
        for layer in self.layers:
            x=np.asarray(layer['reference']);w=np.asarray(layer['weight'])
            active=(w[:-1]>0)|(w[1:]>0)
            edges=np.diff(np.r_[False,active,False].astype(np.int8))
            for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                parts.append({**layer,'reference':x[start:end+1].tolist(),
                              'weight':w[start:end+1].tolist()})
        return parts

    def discard_failed_components(self,failures,halo=.001):
        """Drop only support islands touching a failed PCM comparison window."""
        intervals=[]
        for row in failures:
            half=row.get('window_sec',row.get('verification_window_sec',.125))/2+halo
            intervals.append((row['reference_time']-half,row['reference_time']+half))
        kept=[];removed=[]
        for layer in self.support_components():
            lo,hi=layer['reference'][0],layer['reference'][-1]
            if any(lo<b and hi>a for a,b in intervals):
                removed.append({'start_sec':float(lo),'end_sec':float(hi)})
            else:kept.append(layer)
        return (BlendMap(self.base,kept) if kept else self.base),removed


def affine_patches(measurements,base,sample_rate,coherent_gaps=False,
                   max_anchor_gap=.251,min_band_score=.95):
    """Consecutive independently measured anchors supporting one source clock."""
    runs=[];run=[]
    tolerance=max(2/sample_rate,.00008)
    def fit(rows):
        x=np.array([r.target_time for r in rows]);y=np.array([r.reference_time+r.correction_sec for r in rows])-x
        slope,offset=np.polyfit(x-x[0],y,1)
        return slope,offset-slope*x[0],float(np.max(np.abs(y-(slope*(x-x[0])+offset))))
    def commit():
        if len(run)>=3 and run[-1].target_time-run[0].target_time>=.20:
            slope,offset,error=fit(run)
            if error<=tolerance and abs(slope)<=.003:
                runs.append((list(run),1+slope,offset,error))
    for row in measurements:
        good=(row.status=='MEASURED' and row.correction_sec is not None and row.score>=.85
              and (row.method.startswith('raw') or
                   (coherent_gaps and row.method.startswith('multiband') and row.score>=min_band_score)))
        if not good:
            if coherent_gaps and row.status=='UNVERIFIABLE':
                # This only proposes an interval between independently fitted
                # anchors. Every intervening PCM window still gates the patch.
                continue
            commit();run=[];continue
        if run and (row.target_time-run[-1].target_time>(max_anchor_gap if coherent_gaps else .126) or
                    abs((row.reference_time+row.correction_sec-row.target_time)-
                        (run[-1].reference_time+run[-1].correction_sec-run[-1].target_time))>.001):
            commit();run=[]
        if len(run)>=2:
            slope,offset,error=fit([*run,row])
            if error>tolerance or abs(slope)>.003:
                commit();run=run[-1:]
        run.append(row)
    commit()
    layers=[]
    for rows,rate,offset,error in runs:
        lo=rate*rows[0].target_time+offset
        hi=rate*rows[-1].target_time+offset
        # Fade in amplitude over 12 ms. Time positions keep their measured
        # clock throughout, so a large shift does not create a rate ramp.
        fade=min(.012,(hi-lo)/4)
        # An independently measured edge anchor permits half a grid interval
        # of extension into the edge region. All added windows still gate it.
        if rows[-1].target_time>=base.pieces[-1].target[-1]-.075:
            lo-=.0625+fade
        if rows[0].target_time<=base.pieces[0].target[0]+.075:
            hi+=.0625+fade
        if layers:
            lo=max(lo,layers[-1]['reference'][-1])
        if hi-lo<.04:continue
        layers.append({'reference':[float(lo),float(lo+fade),float(hi-fade),float(hi)],
                       'weight':[0.,1.,1.,0.],'rate':float(rate),'offset':float(offset),
                       'fit_error_sec':error,'measured_anchors':len(rows)})
    return BlendMap(base,layers) if layers else base


def gate_patches(reference,target,base,proposal,schedules,settings,reference_times,progress=None):
    from .guard import verify_candidate_pcm,same_map
    from .choice import verify_reference_grid
    candidate=proposal
    history=[]
    source_cache,before_cache,fixed_cache,fixed_before={},{},{},{}
    # The first six passes keep local waveform masks. If a few boundaries
    # remain unsafe, remove their support islands and VERIFY the retained map.
    # A rejected island must not discard unrelated improvements in the film.
    for attempt in range(9):
        if same_map(base,candidate):break
        if progress is not None and hasattr(progress,'begin_pass'):progress.begin_pass(attempt+1,9)
        rows=[]
        total=sum(len(times) for _,times in schedules)+2*len(reference_times)
        done=0
        for schedule,times in schedules:
            rows+=verify_candidate_pcm(reference,target,base,candidate,schedule,target_times=times,
                row_cache=source_cache,baseline_cache=before_cache,
                progress=(lambda n,count,offset=done:progress(offset+n,total)) if progress else None)
            done+=len(times)
        for step in (.125,.5):
            rows+=verify_reference_grid(reference,target,base,candidate,replace(settings,anchor_step=step),
                reference_times,row_cache=fixed_cache,baseline_cache=fixed_before,
                progress=(lambda n,count,offset=done:progress(offset+n,total)) if progress else None)
            done+=len(reference_times)
        bad=[row for row in rows if row['status']=='REGRESSION' or
             (row['micro_changed'] and row['status']=='UNVERIFIABLE')]
        improved={round(row['target_time'],9) for row in rows if row['status']=='VERIFIED'
                  and row['baseline_residual_sec'] is not None and row['residual_sec'] is not None
                  and abs(row['residual_sec'])+settings.min_improvement<abs(row['baseline_residual_sec'])}
        history.append({'attempt':attempt,'bad':len(bad),'improved':len(improved),
                        'mode':'local_mask' if attempt<6 else 'isolated_components'})
        if not bad:
            return (candidate,rows,history) if len(improved)>=3 else (base,[],history)
        if attempt>=5:
            history[-1]['failed_windows']=[{
                'target_time':r.get('target_time'),'reference_time':r['reference_time'],
                'status':r['status'],'window_sec':r.get('window_sec',r.get('verification_window_sec',.125)),
                'before_residual_sec':r.get('baseline_residual_sec'),'after_residual_sec':r.get('residual_sec'),
                'before_score':r.get('baseline_score'),'after_score':r.get('score')} for r in bad]
            if attempt==8 or not isinstance(candidate,BlendMap):break
            candidate,removed=candidate.discard_failed_components(bad)
            history[-1]['discarded_components']=removed
            if not removed:break
            if progress is not None:
                print(f'\nИзолировано проблемных фрагментов: {len(removed)}. Остальные поправки проверяются повторно.')
            continue
        # Copy the baseline over every failed analysis window, with a short
        # amplitude crossfade outside it. This never creates a speed ramp.
        groups={}
        for row in bad:
            length=row.get('window_sec',row.get('verification_window_sec',.125))
            groups.setdefault(length,[]).append(row['reference_time'])
        for length,times in groups.items():
            candidate,_=candidate.reduced(times,length/2+.001)
            if not isinstance(candidate,BlendMap):break
    return base,[],history
