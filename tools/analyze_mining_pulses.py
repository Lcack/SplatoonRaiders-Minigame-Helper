"""Offline MINING pulse identification and exact clear-motion prediction.

No hardware imports or commands. Models use normalized viewport width/second.
Input times are actual successful commands.jsonl ``after`` times, never planned
pulse durations. Every delay/tau candidate sees exactly the same observations.
Whole segments/rounds are held out; target tasks are a further independent test.
"""
from bisect import bisect_right
from collections import defaultdict
import argparse
import json
import math
from pathlib import Path

import numpy as np


def _model(model):
    model=dict(model)
    for key in ("steady_gain", "tau_seconds", "delay_seconds"):
        if key not in model or not math.isfinite(float(model[key])):
            raise ValueError("missing finite model parameter: "+key)
    if model["steady_gain"]<=0 or model["tau_seconds"]<=0 or model["delay_seconds"]<0:
        raise ValueError("gain/tau must be positive; delay nonnegative")
    model.setdefault("neutral_velocity",0.)
    return model


def _step(x,v,u,dt,model):
    target=model["steady_gain"]*u+model.get("neutral_velocity",0.)
    tau=model["tau_seconds"];one_minus=-math.expm1(-dt/tau)
    return (x+target*dt+(v-target)*tau*one_minus,
            target+(v-target)*(1-one_minus))


def _trajectory(times,state,inputs,model):
    model=_model(model)
    if isinstance(inputs,(int,float)):
        inputs=[{"time":0.,"axis":float(inputs)}]
    changes=sorted(((float(item["time"])+model["delay_seconds"],float(item["axis"])) for item in inputs),key=lambda item:item[0])
    x=float(state["x"]);v=float(state["v"])
    if "previous_axis" not in state:
        raise ValueError("previous_axis and pending input history are required for delayed prediction")
    u=float(state["previous_axis"]);cursor=0;now=0.;result=[]
    while cursor<len(changes) and changes[cursor][0]<=0:
        u=changes[cursor][1];cursor+=1
    for goal in times:
        goal=float(goal)
        if goal<now-1e-12:raise ValueError("times must be sorted and nonnegative")
        while cursor<len(changes) and changes[cursor][0]<=goal:
            event,new_axis=changes[cursor]
            if event>now:x,v=_step(x,v,u,event-now,model)
            now=max(now,event);u=new_axis;cursor+=1
        x,v=_step(x,v,u,goal-now,model);now=goal
        result.append((x,v))
    return np.asarray(result)


def position_predict(t,state,input,model=None):
    """Predict observed x/v after t seconds, including delayed pending commands.

    ``state`` is {x,v,previous_axis}; ``input`` is a normalized axis or a list
    of {time,axis} actual/planned issue times relative to the state's timestamp.
    Include recent negative-time commands so pending effects are not lost.
    ``model`` can instead be supplied as state['model']; no implicit constants.
    This is clear-flight horizontal prediction, never collision/forward motion.
    """
    model=_model(model if model is not None else state["model"])
    if not math.isfinite(float(t)) or t<0:raise ValueError("t must be finite and nonnegative")
    x,v=_trajectory([t],state,input,model)[0]
    return {"x":float(x),"v":float(v),"seconds":float(t),"calibrated":False,
            "scope":"horizontal_clear_motion_effective_input_to_observation_model"}


def input_duration_to_target(target_x,state,axis,model,max_hold_seconds=1.2,
                             release_observation_seconds=.55,command_start_seconds=0.):
    """Solve command hold time for x at a declared time AFTER release effect.

    A release is issued at hold_seconds and takes effect after model delay.
    The target is evaluated another release_observation_seconds later, so coast
    is included. This is not 'wait until the frame reaches target then release'.
    No solution is returned for targets outside the specified hold-time range.
    """
    model=_model(model)
    if not 0<abs(axis)<=1:raise ValueError("nonzero normalized axis required")
    history=list(state.get("command_history",[]))
    def evaluate(hold):
        inputs=history+[{"time":command_start_seconds,"axis":axis},{"time":command_start_seconds+hold,"axis":0.}]
        end=command_start_seconds+hold+model["delay_seconds"]+release_observation_seconds
        return position_predict(end,state,inputs,model)
    grid=np.linspace(0,max_hold_seconds,121);last=evaluate(float(grid[0]));root=None
    if abs(last["x"]-target_x)<1e-10:root=0.
    for left,right in zip(grid,grid[1:]):
        nxt=evaluate(float(right))
        if (last["x"]-target_x)*(nxt["x"]-target_x)<=0:
            lo,hi=float(left),float(right);sign=last["x"]-target_x
            for _ in range(42):
                middle=(lo+hi)/2;value=evaluate(middle)["x"]-target_x
                if sign*value<=0:hi=middle
                else:lo=middle;sign=value
            root=(lo+hi)/2;break
        last=nxt
    if root is None:return {"reachable":False,"hold_seconds":None,"calibrated":False}
    final=evaluate(root)
    return {"reachable":True,"hold_seconds":root,"prediction":final,
            "release_observation_seconds":release_observation_seconds,
            "command_start_seconds":command_start_seconds,
            "remaining_relative_coast_widths":model["tau_seconds"]*(final["v"]-model["neutral_velocity"]),
            "duration_exceeds_observed_hold_support":root>model.get("observed_max_nonzero_hold_seconds",math.inf),
            "target_x":target_x,"calibrated":False}


def target_duration_interval(target_x,state,axis,models,initial_velocity_interval=None,**kwargs):
    models=list(models)
    velocities=[state["v"]] if initial_velocity_interval is None else sorted(set([state["v"],*initial_velocity_interval]))
    results=[]
    for model in models:
        for velocity in velocities:
            plan=input_duration_to_target(target_x,{**state,"v":velocity},axis,model,**kwargs)
            plan["initial_velocity_used"]=velocity;results.append(plan)
    durations=[r["hold_seconds"] for r in results if r["reachable"]]
    return {"target_x":target_x,"axis":axis,"model_count":len(models),"prediction_count":len(results),
            "unreachable_model_count":len(results)-len(durations),
            "hold_seconds_interval":[min(durations),max(durations)] if durations else None,
            "initial_velocity_interval":initial_velocity_interval,"individual_plans":results,"calibrated":False,
            "interval_kind":"plausible_model_envelope_not_statistical_confidence_interval"}


def _read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_segments(directories,manifest=None):
    """Read supervisor diagnostics.identification; optional explicit manifest."""
    segments=[];rejected=[]
    for directory in map(Path,directories):
        trace=_read(directory/"trace.jsonl")
        summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
        command_path=Path(summary["actual_commands"]) if summary.get("actual_commands") else Path(summary.get("session",summary.get("source_session")))/"commands.jsonl"
        commands=sorted((c for c in _read(command_path) if c.get("success")),key=lambda c:c["after"])
        groups=defaultdict(list);outcomes={}
        for row in trace:
            diag=row.get("identification",row.get("decision",{}).get("diagnostics",{}).get("identification",{}))
            sid=diag.get("segment_id")
            if sid is not None:groups[str(sid)].append((row,diag))
            for event in diag.get("events",[]):
                if not isinstance(event,dict):continue
                eid=event.get("segment_id",sid)
                if eid is not None:outcomes[str(eid)]=event
        overrides={str(t["segment_id"]):t for t in (manifest or {}).get("segments",[]) if Path(t.get("directory",directory)).name==directory.name}
        for sid,rows in groups.items():
            override=overrides.get(sid,{})
            if override.get("exclude"):rejected.append({"id":directory.name+":"+sid,"reason":"manifest_excluded"});continue
            aborted=any(d.get("segment_aborted") or d.get("segment_validity")=="aborted" for _,d in rows)
            event=outcomes.get(sid,{})
            text=str(event.get("type",event.get("event",event.get("status","")))).lower()
            completed=any(d.get("segment_validity")=="completed" for _,d in rows) or "completed" in text
            aborted=aborted or "aborted" in text
            mode=override.get("mode",next((d.get("mode") for _,d in rows if d.get("mode") in ("normal","zr")),None))
            reason="aborted" if aborted else "not_completed" if not completed else "mode_missing" if mode is None else None
            points=[];tracks=set();unreliable=False
            for row,diag in rows:
                obs=row["observation"]
                players=[e for e in obs.get("entities",[]) if e.get("kind")=="player" and not e.get("source")]
                bad_events=set(diag.get("motion_events",[]))|set(obs.get("metrics",{}).get("motion_events",[]))
                if bad_events & {"collision","wall_collision","rock_collision","egg_pickup","buff_active"}:reason="collision_or_buff_event"
                if any(e.get("kind") in ("player_hit","egg_buff") for e in obs.get("entities",[])):reason="collision_or_buff_entity"
                if (obs.get("phase")!="playing" or len(players)!=1 or not obs.get("metrics",{}).get("player_detected",True)):
                    unreliable=True;continue
                p=players[0]
                if p.get("samples",0)<3:unreliable=True;continue
                tracks.add(p.get("track_id"));points.append((float(obs["timestamp"]),float(p["x"]),int(obs["sequence"]),diag.get("stage")))
            if unreliable and not override.get("allow_isolated_missing",False):reason=reason or "missing_or_unconfirmed_body_in_segment"
            if len(tracks)!=1:reason=reason or "track_changed"
            points=sorted({p[0]:p for p in points}.values())
            if len(points)<12 or (points and points[-1][0]-points[0][0]<.7):reason=reason or "too_short"
            gap_limit=float(summary.get("config",{}).get("max_step_seconds",.12))
            if any(b[0]-a[0]>gap_limit for a,b in zip(points,points[1:])):reason=reason or "observation_gap_exceeds_declared_probe_cadence_guard"
            identity=directory.name+":"+sid
            if reason:rejected.append({"id":identity,"reason":reason});continue
            origin=points[0][0];original_origin=origin;end=points[-1][0]
            prior=[c for c in commands if c["after"]<=origin-.35]
            if not prior:rejected.append({"id":identity,"reason":"missing_input_history"});continue
            active=[prior[-1]]+[c for c in commands if origin-.35<c["after"]<=end]
            pulse_intent=min((float(d["stage_started_at"]) for _,d in rows if d.get("stage")=="pulse" and d.get("stage_started_at") is not None),default=origin)
            first_lateral=min((c["after"] for c in active if c["after"]>=pulse_intent-.01 and abs(c["state"].get("left_stick",{}).get("x",0))>1638
                               and ("ZR" in c["state"].get("buttons",[]))==(mode=="zr")),default=None)
            if first_lateral is None:
                rejected.append({"id":identity,"reason":"no_actual_lateral_pulse"});continue
            # A ZR segment intentionally connects ZR during its zero-axis
            # baseline. Accept that lead-in, but never merge mode changes
            # once nonzero lateral actuation begins.
            if any(("ZR" in c["state"].get("buttons",[]))!=(mode=="zr") for c in active if c["after"]>=first_lateral):
                rejected.append({"id":identity,"reason":"ZR_switched_after_lateral_pulse_began"});continue
            mismatches=[i for i,c in enumerate(active) if ("ZR" in c["state"].get("buttons",[]))!=(mode=="zr")]
            if mismatches:
                last_mismatch=mismatches[-1]
                if last_mismatch+1>=len(active):
                    rejected.append({"id":identity,"reason":"no_stable_mode_after_lead_in"});continue
                # Every delay candidate, including .300s, must begin AFTER
                # the previous mode's last pending force. Initial velocity is
                # a state to estimate, not a claim the trimmed frame is rest.
                clean_start=active[last_mismatch+1]["after"]+.300
                points=[p for p in points if p[0]>=clean_start]
                if len(points)<12 or points[-1][0]-points[0][0]<.7:
                    rejected.append({"id":identity,"reason":"too_short_after_common_mode_history_trim"});continue
                origin=points[0][0]
                prior=[c for c in commands if c["after"]<=origin-.35]
                active=[prior[-1]]+[c for c in commands if origin-.35<c["after"]<=end]
            changes=[];last=None
            for c in active:
                axis=c["state"].get("left_stick",{}).get("x",0)/32767
                if axis!=last:changes.append({"time":c["after"]-origin,"axis":axis});last=axis
            points=np.asarray([(p[0]-origin,p[1]) for p in points])
            segments.append({"id":identity,"round":directory.name,"segment_id":sid,"mode":mode,"times":points[:,0],"x":points[:,1],
                "input":changes,"previous_axis":active[0]["state"]["left_stick"]["x"]/32767,
                "split":override.get("split"),"target_x":override.get("target_x"),
                "first_lateral_after_segment_seconds":first_lateral-origin,
                "common_mode_history_trim_seconds":origin-original_origin,
                "protocol":rows[0][1].get("protocol"),
                "intended_axis":next((d.get("axis") for _,d in rows if d.get("stage")=="pulse"),None),
                "declared_probe_cadence_guard_seconds":gap_limit,
                "source":"completed_probe_gates_not_independent_collision_ground_truth"})
    return segments,rejected


def _basis(segment,tau,delay):
    model={"steady_gain":1.,"tau_seconds":tau,"delay_seconds":delay,"neutral_velocity":0.}
    times=segment["times"]
    forced=_trajectory(times,{"x":0.,"v":0.,"previous_axis":segment["previous_axis"]},segment["input"],model)[:,0]
    velocity=tau*(-np.expm1(-times/tau));neutral=times-velocity
    return forced,neutral,velocity


def _initial_state_from_baseline(segment,model):
    first_change=min((max(0,e["time"]) for e in segment["input"] if abs(e["axis"])>.05 and e["time"]>=0),default=.25)
    ids=np.flatnonzero(segment["times"]<=min(.25,first_change))
    if len(ids)<3:ids=np.arange(min(3,len(segment["times"])))
    f,n,v=_basis(segment,model["tau_seconds"],model["delay_seconds"])
    rhs=segment["x"][ids]-segment["x"][0]-model["steady_gain"]*f[ids]-model["neutral_velocity"]*n[ids]
    initial_velocity,offset=np.linalg.lstsq(np.column_stack((v[ids],np.ones(len(ids)))),rhs,rcond=None)[0]
    return {"x":float(segment["x"][0]+offset),"v":float(initial_velocity),"previous_axis":segment["previous_axis"]},int(ids[-1])+1


def fit_mode(segments,mode):
    segments=[s for s in segments if s["mode"]==mode];rounds=list(dict.fromkeys(s["round"] for s in segments))
    if not segments:return {"mode":mode,"status":"no_usable_segments","calibrated":False}
    explicit=any(s["split"] for s in segments)
    if explicit:
        train=[s for s in segments if s["split"] in (None,"train")];holdout=[s for s in segments if s["split"]=="holdout"]
    elif len(rounds)>=2:
        train=[s for s in segments if s["round"]!=rounds[-1]];holdout=[s for s in segments if s["round"]==rounds[-1]]
    else:
        border=max(1,math.floor(len(segments)*.75));train=segments[:border];holdout=segments[border:]
    if len(train)<2:return {"mode":mode,"status":"fewer_than_two_train_segments","calibrated":False}
    delays=np.arange(0,.301,.0125);taus=np.unique(np.r_[np.geomspace(.04,1.2,29),.1,.2,.25,1/3,.5])
    candidates=[];all_models=[];n=len(train);observed=np.concatenate([s["x"]-s["x"][0] for s in train])
    for delay in delays:
        best=None
        for tau in taus:
            design=np.zeros((len(observed),2+2*n));offset=0
            for index,segment in enumerate(train):
                count=len(segment["times"]);f,b,v=_basis(segment,float(tau),float(delay));sl=slice(offset,offset+count)
                design[sl,0]=f;design[sl,1]=b;design[sl,2+index]=v;design[sl,2+n+index]=1;offset+=count
            co=np.linalg.lstsq(design,observed,rcond=None)[0]
            if not .02<co[0]<1 or abs(co[1])>.08 or max(abs(co[2:2+n]))>.6 or max(abs(co[2+n:]))>.012:continue
            rmse=float(np.sqrt(np.mean((design@co-observed)**2)))
            model={"steady_gain":float(co[0]),"tau_seconds":float(tau),"delay_seconds":float(delay),"neutral_velocity":float(co[1]),
                   "input_gain_per_second":float(co[0]/tau),"drag_per_second":float(1/tau),"train_position_rmse":rmse}
            all_models.append(model)
            if best is None or rmse<best["train_position_rmse"]:best=model
        if best:candidates.append(best)
    if not candidates:return {"mode":mode,"status":"no_physically_compatible_candidate","calibrated":False}
    selected=min(candidates,key=lambda c:c["train_position_rmse"])
    # Explicit resolution envelope, not an accuracy/pass threshold or formal CI.
    band=.5/640
    ensemble=[c for c in all_models if c["train_position_rmse"]<=math.hypot(selected["train_position_rmse"],band)]
    holds=[];levels=[]
    for segment in train:
        changes=segment["input"]+[{"time":float(segment["times"][-1]),"axis":0.}]
        for a,b in zip(changes,changes[1:]):
            if abs(a["axis"])>.05 and b["time"]>0:
                levels.append(abs(a["axis"]));holds.append(b["time"]-max(0,a["time"]))
    for candidate in all_models:
        candidate["observed_max_nonzero_hold_seconds"]=max(holds,default=0.)
        candidate["observed_nonzero_axis_levels"]=sorted(set(levels))
    validation=[];curves=[]
    for segment in holdout:
        state,baseline_count=_initial_state_from_baseline(segment,selected)
        prediction=_trajectory(segment["times"],state,segment["input"],selected)[:,0]
        error=prediction[baseline_count:]-segment["x"][baseline_count:]
        validation.append({"id":segment["id"],"initial_state_from_observed_prefix_only":state,"initialization_samples":baseline_count,"scored_samples":len(error),
            "protocol":segment.get("protocol"),"intended_axis":segment.get("intended_axis"),
            "position_rmse":float(np.sqrt(np.mean(error**2))) if len(error) else None,
            "position_max_abs_error":float(max(abs(error))) if len(error) else None,
            "endpoint_error":float(prediction[-1]-segment["x"][-1])})
        curves.append({"id":segment["id"],"times":segment["times"].tolist(),"observed_x":segment["x"].tolist(),"predicted_x":prediction.tolist()})
    return {"mode":mode,"status":"exploratory_position_model_requires_independent_target_tasks","calibrated":False,
        "train_segment_ids":[s["id"] for s in train],"holdout_segment_ids":[s["id"] for s in holdout],
        "selected_by_training_only":selected,"delay_profile_same_observations":candidates,
        "plausible_models_noise_resolution_band":ensemble,"noise_resolution_band_widths":band,
        "parameter_envelope":{key:[min(m[key] for m in ensemble),max(m[key] for m in ensemble)]
            for key in ("steady_gain","tau_seconds","delay_seconds","neutral_velocity")},
        "selected_on_grid_boundary":bool(selected["delay_seconds"] in (float(delays[0]),float(delays[-1]))
            or selected["tau_seconds"] in (float(taus[0]),float(taus[-1]))),
        "plausible_envelope_touches_tau_grid_boundary":any(m["tau_seconds"] in (float(taus[0]),float(taus[-1])) for m in ensemble),
        "holdout_position_errors":validation,"holdout_curves":curves,"independent_target_task_results":[],
        "model_equation":"x_dot=v; v_dot=(steady_gain*u_delayed+neutral_velocity-v)/tau",
        "limitations":["probe清障门槛不是独立事件真值；碰撞/丢体/中止段整段剔除。",
            "时间是实际发送after到采集观测的有效延迟，不能拆成游戏内部和硬件链路延迟。",
            "位置空间一次拟合完整轨迹；每个delay使用相同观测，模型选择不使用holdout。",
            "分辨率带只保留近最优延迟候选，不是统计置信区间；单tau假设需反向制动与目标任务验证。",
            "ZR段若每次从松开重新接通，只覆盖该前向加速期；不得冒充任意高速ZR状态的相同横向响应。",
            "横向结果不预测前向ZR加速度，且未证明与前向速度、强化状态无关。"]}


def self_check():
    """Independent analytic identities and input/target inversion checks."""
    model={"steady_gain":.28,"tau_seconds":.25,"delay_seconds":.1,"neutral_velocity":0.}
    state={"x":.5,"v":0.,"previous_axis":0.}
    pulse=[{"time":0.,"axis":.4},{"time":.35,"axis":0.}]
    early=position_predict(.05,state,pulse,model)
    assert early["x"]==.5 and early["v"]==0
    coast=position_predict(8.,state,pulse,model)
    assert abs(coast["x"]-(.5+.28*.4*.35))<1e-10
    plan=input_duration_to_target(.54,state,.4,model,release_observation_seconds=.8)
    assert plan["reachable"] and abs(plan["prediction"]["x"]-.54)<1e-9
    assert .35<plan["hold_seconds"]<.40
    rng=np.random.default_rng(422)
    synthetic=[]
    for i in range(6):
        axis=.4 if i%2==0 else -.4
        inputs=[{"time":.3,"axis":axis},{"time":.65 if i<2 else .9,"axis":0.}]
        times=np.arange(0,1.55,1/30);initial={"x":.5,"v":0.,"previous_axis":0.}
        observed=_trajectory(times,initial,inputs,model)[:,0]+rng.normal(0,.00015,len(times))
        synthetic.append({"id":str(i),"round":"train" if i<4 else "holdout","mode":"normal","times":times,"x":observed,
                          "input":inputs,"previous_axis":0.,"split":"train" if i<4 else "holdout"})
    recovered=fit_mode(synthetic,"normal");best=recovered["selected_by_training_only"]
    assert abs(best["steady_gain"]-.28)<.025
    assert abs(best["delay_seconds"]-.1)<=.0251
    assert abs(best["tau_seconds"]-.25)<.065
    assert max(v["position_rmse"] for v in recovered["holdout_position_errors"])<.003
    return {"status":"analytic_delay_pulse_area_target_inverse_and_synthetic_whole_segment_recovery_passed",
            "synthetic_recovered":best,"calibrated":False}


def main():
    parser=argparse.ArgumentParser(description="MINING整段脉冲位置辨识；纯离线，无控制器")
    parser.add_argument("directories",nargs="*",type=Path)
    parser.add_argument("--output",type=Path,default=Path("artifacts/mining_pulse_analysis"))
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--batch",type=Path,help="root连续脉冲驱动的根目录；读取全部round-*/trace.jsonl")
    parser.add_argument("--self-check",action="store_true")
    args=parser.parse_args()
    if args.self_check:print(json.dumps(self_check()));return
    if args.batch:args.directories.extend(sorted(p.parent for p in args.batch.glob("round-*/trace.jsonl") if (p.parent/"summary.json").exists()))
    if not args.directories:parser.error("需要监督记录目录、--batch，或--self-check")
    manifest=json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    segments,rejects=load_segments(args.directories,manifest)
    report={"scope":"whole_segment_position_identification_not_calibration","calibrated":False,
        "usable_segment_count":len(segments),"rejected_segments":rejects,
        "models":{mode:fit_mode(segments,mode) for mode in ("normal","zr")}}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"pulse_position_models.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(args.output.resolve()),"usable_segments":len(segments),"rejected":len(rejects),"calibrated":False}))


if __name__=="__main__":main()
