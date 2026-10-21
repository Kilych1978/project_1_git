"""Fresh comparison plans and a protected best map; no previous-run loading."""
from dataclasses import dataclass,replace
import json
import numpy as np
from pcm_core.model import Piece,TimeMap
from pcm_core.micro import collect
from pcm_core.blend import affine_patches,gate_patches

from pcm_core.choice import choose_local,verify_reference_grid
from pcm_core.guard import (same_map,verify_candidate_pcm,verify_render,render_verdict,
                            cancel_unsafe_components,holdout_times,grid)


def bad_rows(rows):
    return [r for r in rows if r['status']=='REGRESSION' or
            (r['micro_changed'] and r['status']=='UNVERIFIABLE')]


def merge_schedules(base,groups,quantize):
    result = {}
    for schedules in groups:
        for settings,times in schedules:
            settings = replace(settings,verification_pcm16=quantize)
            result.setdefault(settings,set()).update(times if times is not None else
                holdout_times(base,settings)+grid(base,settings))
    return [(settings,sorted(times)) for settings,times in result.items()]


@dataclass
class ProtectedPlan:
    baseline: object
    schedules: list
    reference_times: list
    settings: object
    additional_confirmed: bool

    def verify_written(self,reference,target,actual,final,progress):
        rows = []
        for index,(settings,times) in enumerate(self.schedules,1):
            callback = progress(f'СОХРАННОСТЬ БАЗОВОГО РЕЗУЛЬТАТА | сетка {index}/{len(self.schedules)}')
            rows.extend(verify_render(reference,target,actual,self.baseline,final,settings,
                                       target_times=times,progress=callback))
            callback.finish()
        for step in (.125,.5):
            callback = progress(f'ПРОВЕРКА НА ОДИНАКОВОМ ВРЕМЕНИ ЭТАЛОНА | окно {min(self.settings.micro_window,step/8)*1000:g} мс')
            rows.extend(verify_reference_grid(reference,target,self.baseline,final,
                replace(self.settings,anchor_step=step),self.reference_times,rendered=actual,progress=callback))
            callback.finish()
        return rows


def build_best_plan(reference,target,base,initial_settings,output_dir,report,progress,planner):
    """Compute both candidates anew, then permit only checked local additions."""
    def tagged(label):
        return lambda title,seconds=False,target_times=None:progress(label+' | '+title,seconds,target_times)

    baseline_report = {'measurements':[],'independent_windows':[],'audit':[]}
    classic_settings = replace(initial_settings,verification_pcm16=False)
    old,old_holdouts,old_measurements,old_settings,old_audit,old_schedules = planner(
        reference,target,base,classic_settings,baseline_report,tagged('БАЗОВЫЙ ВАРИАНТ'),classic=True)
    for attempt in range(3):
        rows = []
        for index,(settings,times) in enumerate(old_schedules,1):
            native = replace(settings,verification_pcm16=initial_settings.verification_pcm16)
            callback = progress(f'ПРОВЕРКА БАЗОВОГО PCM | проход {attempt+1}/3 | сетка {index}/{len(old_schedules)}')
            rows.extend(verify_candidate_pcm(reference,target,base,old,native,target_times=times,progress=callback))
            callback.finish()
        verdict = render_verdict(rows)
        if verdict!='REJECTED' or attempt==2:
            break
        failed = bad_rows(rows)
        reduced,removed = cancel_unsafe_components(base,old,[r['target_time'] for r in failed],
            old_settings.max_rate_change,taper_scale=2.,combine_masks=False)
        if not removed or same_map(base,reduced):
            break
        old = reduced
    if verdict!='VERIFIED_PCM':
        old = base
    baseline_report.update(fine_map=old.to_dict(),native_preview_verdict=verdict,
                           native_preview_windows=rows)
    with (output_dir/'BASELINE_PCM_PLAN.json').open('w',encoding='utf-8') as file:
        json.dump(baseline_report,file,ensure_ascii=False,indent=2)
    del baseline_report,old_holdouts,old_measurements,old_audit,rows

    modern,holdouts,measurements,settings,audit,new_schedules = planner(
        reference,target,base,initial_settings,report,tagged('УТОЧНЁННЫЙ ВАРИАНТ'))
    schedules = merge_schedules(base,[old_schedules,new_schedules],initial_settings.verification_pcm16)
    settings = replace(settings,max_rate_change=max(settings.max_rate_change,old_settings.max_rate_change))
    duration = min(len(reference),len(target))/settings.analysis_rate
    reference_times = np.arange(.25,duration-.25,.0625).tolist()

    source_check = []
    for index,(schedule,times) in enumerate(schedules,1):
        callback = progress(f'ПРОВЕРКА БАЗОВОЙ КАРТЫ НА ОБЩЕЙ СЕТКЕ | сетка {index}/{len(schedules)}')
        source_check.extend(verify_candidate_pcm(reference,target,base,old,schedule,target_times=times,progress=callback))
        callback.finish()
    # Choose refinement regions from the ORIGINAL common waveform, not its
    # potentially smeared rendition. Keep every newly observed control point
    # before freezing a protected baseline.
    unresolved=sorted({r['target_time'] for r in source_check
        if r['baseline_residual_sec'] is not None and abs(r['baseline_residual_sec'])>.0005
        and (r['residual_sec'] is None or abs(r['residual_sec'])>.0001)})
    zones=[]
    for t in unresolved:
        lo,hi=max(.125,t-.5),min(duration-.125,t+.5)
        if lo>=hi:continue
        if zones and lo<=zones[-1][1]:zones[-1]=(zones[-1][0],max(hi,zones[-1][1]))
        else:zones.append((lo,hi))
    dense_rows=[]
    if zones:
        zoom=replace(settings,anchor_step=.125,max_rate_change=.08)
        centers,probes=[],[]
        for lo,hi in zones:
            centers.extend(np.arange(lo+.0625,hi,.125).tolist())
            probes.extend(np.arange(lo,hi,.0625).tolist())
        callback=progress('ИЗМЕРЕНИЕ ЛОКАЛЬНОЙ СКОРОСТИ И ЗАДЕРЖКИ PCM',target_times=centers)
        dense_rows=collect(reference,target,base,zoom,log=lambda _:None,progress=callback,centers=centers)
        callback.finish()
        # A cropped file edge can leave precisely two agreeing anchors. Take
        # one additional real PCM measurement, rather than discard the whole
        # supported edge or invent a third observation by extrapolation.
        extra=[]
        for edge,ordered in (('left',dense_rows),('right',list(reversed(dense_rows)))):
            run=[]
            for row in ordered:
                if row.status!='MEASURED' or row.score<.85 or not row.method.startswith('raw'):break
                run.append(row)
            if len(run)!=2:continue
            if (edge=='left' and run[0].target_time>.25) or (edge=='right' and run[0].target_time<duration-.25):continue
            delta=[row.reference_time+row.correction_sec-row.target_time for row in run]
            if abs((delta[1]-delta[0])/(run[1].target_time-run[0].target_time))>.003:continue
            t=.0625 if edge=='left' else duration-.0625
            if t not in centers:extra.append(t)
        if extra:
            callback=progress('ДОПОЛНИТЕЛЬНОЕ PCM-ИЗМЕРЕНИЕ У КРАЯ ФАЙЛА',target_times=extra)
            dense_rows+=collect(reference,target,base,zoom,log=lambda _:None,progress=callback,centers=extra)
            callback.finish()
            dense_rows.sort(key=lambda row:row.target_time)
            centers+=extra
        dense_times=sorted(set(probes+centers))
        callback=progress('ПРОВЕРКА БАЗОВОЙ КАРТЫ НА НОВЫХ ТОЧКАХ PCM',target_times=dense_times)
        source_check.extend(verify_candidate_pcm(reference,target,base,old,zoom,target_times=dense_times,progress=callback))
        callback.finish()
        schedules=merge_schedules(base,[schedules,[(zoom,dense_times)]],initial_settings.verification_pcm16)
    if bad_rows(source_check):
        callback = progress('ЛОКАЛЬНАЯ ПРОВЕРКА БАЗОВЫХ ПОПРАВОК')
        # Repair the newly observed SOURCE-window failures only. Absolute
        # reference probes constrain ADDITIONS relative to the established
        # result; they must not erase it merely because the original mix is
        # unmeasurable on that different reference-centred context.
        old,_,history = choose_local(reference,target,base,old,schedules,settings,[],
            log=lambda row:None,progress=callback)
        callback.finish()
        report['baseline_sanitization'] = history
    # A coherent common delay can be corrected without introducing any local
    # tempo transitions. Its admissible interval is the intersection of the
    # before/after non-regression intervals of measured residuals. It remains
    # a proposal and still undergoes every PCM check, including weaker windows.
    estimates = [r['residual_sec'] for r in source_check if r['status']=='VERIFIED'
                 and r['residual_sec'] is not None and (r.get('score') or 0.)>=.85]
    shifted_best = old
    if estimates:
        lower = max(min(0.,2*value) for value in estimates)
        upper = min(max(0.,2*value) for value in estimates)
        shift = float(np.clip(np.median(estimates),.9*lower,.9*upper))
        if abs(shift)>=2*settings.min_improvement:
            pieces = []
            for p in old.pieces:
                extra = {'target':[float(p.target[0]),float(p.target[-1])],'delta':[shift,shift]}
                pieces.append(Piece(p.target,p.reference+shift,[*p.corrections,extra]))
            proposal = TimeMap(pieces,list(old.notes))
            callback = progress('ПРОВЕРКА ОБЩЕГО ОСТАТОЧНОГО СДВИГА')
            shifted_best,_,shift_audit = choose_local(reference,target,old,proposal,schedules,settings,
                reference_times,log=lambda row:None,progress=callback)
            callback.finish()
            report.update(global_offset_candidate_sec=shift,global_offset_audit=shift_audit)
    del source_check

    callback = progress('ВЫБОР ЛОКАЛЬНЫХ УЛУЧШЕНИЙ С СОХРАНЕНИЕМ БАЗОВОГО PCM')
    selected,checked,history = choose_local(reference,target,shifted_best,modern,schedules,settings,
        reference_times,log=lambda row:None,progress=callback)
    callback.finish()
    if dense_rows:
        raw=affine_patches(dense_rows,selected,settings.analysis_rate)
        callback=progress('ПРОВЕРКА ЛОКАЛЬНЫХ PCM-СОЕДИНЕНИЙ')
        selected,checked,dense_history=gate_patches(reference,target,selected,raw,schedules,zoom,
            reference_times,progress=callback)
        callback.finish()
        report.update(residual_clock_measurements=[row.to_dict() for row in dense_rows],
                      residual_choice_audit=dense_history)
        settings=zoom
    additional = not same_map(old,selected)
    report.update(baseline_fine_map=old.to_dict(),fine_map=selected.to_dict(),
                  local_choice_audit=history,protected_preview_windows=checked,
                  additional_improvement_confirmed=additional,
                  comparison_definition='same current source; protected established candidate; identical probe schedules',
                  baseline_plan_file='BASELINE_PCM_PLAN.json')
    protection = ProtectedPlan(old,schedules,reference_times,settings,additional)
    return selected,holdouts,measurements,settings,audit,schedules,protection
