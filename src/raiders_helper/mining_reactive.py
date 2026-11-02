"""Experimental short-horizon MINING steering, driven by observed geometry.

This controller never opens hardware and never marks a profile calibrated.
The fallback rates below are declared control hypotheses. Measured normal/ZR
models may replace them; camera-player y is never interpreted as world speed.
The player advances down the *current* viewport for planning, while the camera
keeps the real player near its upper edge. This is a relative geometry model.
"""
from dataclasses import dataclass
import math
from time import perf_counter

from .control.buttons import Button, ControllerState, StickVector
from .mining_planner import Road, contact_fraction
from .mining_forward import seed_context
from .mining_collision import ObstacleGeometry,BOX_GEOMETRY
from .models import Decision, Game


def _finite(value, default=None):
    return float(value) if type(value) in (int, float) and math.isfinite(value) else default


def _positive(parameters, key, default):
    value = _finite(parameters[key]) if key in parameters else default
    if value is None or value <= 0:
        raise ValueError(f"{key} 必须为有限正数")
    return value


@dataclass(frozen=True)
class _Motion:
    forward: float
    input_gain: float
    drag: float
    maximum_speed: float
    neutral_velocity: float = 0.0


def _motion(parameters, observation, mode):
    config = parameters.get("motion_models", {}).get(mode, {})
    # Both defaults are explicit engineering hypotheses, not game constants.
    forward = _positive(config, "forward_speed", .50 if mode == "zr" else .28)
    observed = observation.metrics
    source = "configured_or_heuristic"
    if (observed.get("forward_speed_observed") is True
            and observed.get("forward_speed_confidence", 0) >= .8
            and observed.get("forward_speed_units") == "viewport_heights_per_second"
            and _finite(observed.get("forward_speed"), 0) > 0):
        forward = float(observed["forward_speed"])
        source = str(observed.get("forward_speed_source", "explicit_measurement"))
    gain = _positive(config, "input_gain_per_second", 5.0)
    drag = _finite(config["drag_per_second"]) if "drag_per_second" in config else 5.0
    if drag is None or drag < 0:
        raise ValueError("drag_per_second 必须非负")
    speed = _positive(config, "maximum_lateral_speed", .85)
    neutral = _finite(config["neutral_velocity"]) if "neutral_velocity" in config else 0.0
    if neutral is None:
        raise ValueError("neutral_velocity 必须为有限数")
    return _Motion(forward, gain, drag, speed, neutral), source


@dataclass(frozen=True)
class _Node:
    x: float
    velocity: float
    depth: float
    cost: float
    path: tuple
    collected_predicted: frozenset = frozenset()


def _spans(road, center_y, radius_x, radius_y):
    """Center intervals fitting the complete body at the current depth."""
    depths = sorted({center_y-radius_y, center_y+radius_y,
                     *[y for y, _ in road.rows if center_y-radius_y < y < center_y+radius_y]})
    intervals = list(road.intervals(depths[0]))
    for y in depths[1:]:
        intervals = [(max(a,c), min(b,d)) for a,b in intervals for c,d in road.intervals(y)
                     if max(a,c) < min(b,d)]
    return [(a+radius_x,b-radius_x) for a,b in intervals if a+radius_x <= b-radius_x]


def _closest(spans, x):
    return min((max(a,min(b,x)) for a,b in spans),key=lambda candidate:abs(candidate-x),default=None)


def _sweep_constraints(road,y0,y1,rx,ry):
    """Compile a single-branch swept rectangle into x endpoint inequalities.

    Between road knots and body-envelope knots every bound is linear. This is
    the same complete body test as Road.swept_safe, compiled once per layer
    instead of rebuilding its geometry for every candidate control.
    Multiple branches retain Road's explicit topology check as a fallback.
    """
    if y1-y0 < 1e-10:
        return None
    depths=sorted({y0-ry,y1+ry,y0+ry,y1-ry,
                   *[y for y,_ in road.rows if y0-ry < y < y1+ry]})
    constraints={}
    for y in depths:
        spans=road.intervals(y)
        if len(spans)!=1:
            return None
        left,right=spans[0]
        for t in (max(0,min(1,(y-ry-y0)/(y1-y0))),max(0,min(1,(y+ry-y0)/(y1-y0)))):
            a,b=constraints.get(t,(-math.inf,math.inf))
            constraints[t]=(max(a,left+rx),min(b,right-rx))
    return tuple((t,a,b) for t,(a,b) in constraints.items())


def _compiled_safe(constraints,x0,x1):
    return all(left <= x0+(x1-x0)*t <= right for t,left,right in constraints)


def _empty(reason, diagnostics=None):
    return Decision(reason="MINING："+reason, diagnostics={"policy":"mining_reactive",
        "experimental":True,"path":[],"risks":[reason],**(diagnostics or {})})


def _valid_entity(entity):
    return (all(_finite(v) is not None for v in (entity.x,entity.y,entity.w,entity.h,entity.confidence))
            and entity.w > 0 and entity.h >= 0 and 0 <= entity.x <= 1 and 0 <= entity.y <= 1)


def _hit_recovery(obs, parameters, geometry=BOX_GEOMETRY):
    """A current, separately observed hit body may justify limited inward input.

    This does not relabel an unknown frame as play, extrapolate a missing body,
    or claim a collision-free path from a body that may already touch a wall.
    Every call needs the visible hit appearance and its recent blue identity.
    """
    metrics=obs.metrics
    hits=obs.of_kind("player_hit")
    identity_age=_finite(metrics.get("hit_body_identity_age"))
    if (metrics.get("collision_suspected") is not True
            or metrics.get("hit_body_detected") is not True
            or metrics.get("ambiguous_player") is True
            or metrics.get("ambiguous_hit_body") is True
            or metrics.get("hit_body_candidates") != 1
            or _finite(metrics.get("hud_anchor_score"),0) < .65
            or identity_age is None or not 0 < identity_age <= .75
            or obs.of_kind("player") or len(hits)!=1
            or not _valid_entity(hits[0]) or hits[0].confidence < .8 or hits[0].h <= 0):
        return _empty("碰撞外观或近期身份不足，保持中立")
    body=hits[0]
    corridors=[e for e in obs.of_kind("corridor") if _valid_entity(e) and e.confidence >= .8
               and 0 <= e.x-e.w/2 < e.x+e.w/2 <= 1]
    if len({round(e.y,5) for e in corridors}) < 2:
        return _empty("碰撞恢复缺少可靠道路分层")
    try:
        margin=_positive(parameters,"wall_margin",.012)
        uncertainty=_positive(parameters,"position_uncertainty",.004)
        maximum_axis=min(1.0,_positive(parameters,"maximum_axis",1.0))
        # A configured larger value cannot turn this into a full-axis action.
        recovery_axis=min(.6,_positive(parameters,"hit_recovery_axis",.45),maximum_axis)
    except (ValueError,TypeError) as error:
        return _empty(f"碰撞恢复配置无效：{error}")
    road=Road(corridors)
    rx,ry=body.w/2+margin+uncertainty,body.h/2+uncertainty
    if road.rows[0][0] > body.y-ry or road.rows[-1][0] < body.y+ry+.03:
        return _empty("碰撞恢复道路未覆盖完整机体及近处前方")
    spans=_spans(road,body.y,rx,ry)
    if len(spans)!=1:
        return _empty("碰撞恢复没有唯一可容纳机体的通道")
    left,right=spans[0]
    # Stay in the same visible branch and only move an edge-adjacent body
    # inward. A hit in the wide interior may be a boulder, not a side wall.
    inset=min(.05,(right-left)/2)
    target=max(left+inset,min(right-inset,body.x))
    delta=target-body.x
    if abs(delta) <= .005:
        return _empty("碰撞体处于通道内部，缺少脱墙方向")
    hazards=[e for e in obs.entities if e.kind in ("obstacle","rock","wall")
             and _valid_entity(e) and e.confidence >= .8]
    if any(geometry.contact_fraction(body.x,body.y,target,body.y,e,rx,ry) is not None for e in hazards):
        return _empty("碰撞恢复方向经过可见障碍，保持中立")
    control=math.copysign(recovery_axis,delta)
    diagnostics={"policy":"mining_reactive","experimental":True,
        "mode":"collision_recovery","motion_calibrated":False,"path":[],
        "target":{"x":target,"depth":body.y},"acceleration_allowed":False,
        "recovery_source":"observed_player_hit","requires_current_hit":True,
        "hit_body_identity_age":identity_age,"maximum_axis":recovery_axis,
        "risks":["紫红碰撞外观与脱墙动作尚未完成实机验证，当前输入不保证脱离碰撞",
                 "仅依据本帧可见碰撞体恢复；外观消失或身份过期后停止"]}
    return Decision(ControllerState(left_stick=StickVector(round(control*32767),0)),
        "MINING：可见碰撞体靠近侧墙，松开 ZR 并有限向通道内恢复",body.confidence,diagnostics)


def _advance(motion, initial_velocity, control, dt):
    """The same declared constant-input lateral model used by the search."""
    initial_velocity=max(-motion.maximum_speed,min(motion.maximum_speed,initial_velocity))
    drive=motion.input_gain*control+motion.drag*motion.neutral_velocity
    if motion.drag > 1e-8:
        terminal=drive/motion.drag
        limit=math.copysign(motion.maximum_speed,terminal)
        if abs(terminal)>motion.maximum_speed:
            ratio=(limit-terminal)/(initial_velocity-terminal)
            hit=-math.log(ratio)/motion.drag if ratio>0 else 0.0
            if hit < dt:
                distance=terminal*hit+(initial_velocity-terminal)*(-math.expm1(-motion.drag*hit))/motion.drag
                return distance+limit*(dt-hit),limit
        decay=math.exp(-motion.drag*dt)
        velocity=terminal+(initial_velocity-terminal)*decay
        distance=terminal*dt+(initial_velocity-terminal)*(-math.expm1(-motion.drag*dt))/motion.drag
    else:
        velocity=initial_velocity+drive*dt
        if abs(velocity)>motion.maximum_speed:
            limit=math.copysign(motion.maximum_speed,velocity)
            hit=(limit-initial_velocity)/drive
            return initial_velocity*hit+.5*drive*hit*hit+limit*(dt-hit),limit
        distance=initial_velocity*dt+.5*drive*dt*dt
    return distance,velocity


def _capture_clock_context(obs, parameters, normal, boosted, age, fallback_delay):
    """The fit delay is report.after -> capture clock, never pure hardware lag.

    A is image age at planning start; P budgets the upcoming computation.
    D is a separately declared decision-to-report submission wait, and L is
    report-to-capture response. A+P+D+L counts capture buffering once. A common
    L keeps command order causal when fitted mode-delay ranges overlap.
    """
    delay=_finite(parameters.get("effective_delay_seconds",fallback_delay))
    if delay is None or not 0<=delay<=.5:
        raise ValueError("effective_delay_seconds 必须在0到.5秒之间")
    submission=_finite(parameters.get("dispatch_delay_seconds",parameters.get("command_submission_delay_seconds",0.0)))
    if submission is None or not 0<=submission<=.5:
        raise ValueError("dispatch_delay_seconds 必须在0到.5秒之间")
    if ("dispatch_delay_seconds" in parameters and "command_submission_delay_seconds" in parameters
            and parameters["dispatch_delay_seconds"]!=parameters["command_submission_delay_seconds"]):
        raise ValueError("dispatch_delay_seconds 与旧提交等待别名冲突")
    submission_source=("configured_decision_to_backend_report_hypothesis"
                       if "dispatch_delay_seconds" in parameters or "command_submission_delay_seconds" in parameters
                       else "assumed_zero_requires_measurement")
    metrics=obs.metrics
    planning=_finite(metrics.get("planning_delay_seconds",parameters.get("planning_delay_seconds",0.0)))
    if planning is None or not 0<=planning<=.15:
        raise ValueError("planning_delay_seconds 必须在0到.15秒之间")
    planning_source=(str(metrics.get("planning_delay_source","observation_compute_budget"))
                     if "planning_delay_seconds" in metrics else "configured_compute_budget"
                     if "planning_delay_seconds" in parameters else "assumed_zero_legacy")
    prefix_current=_finite(metrics.get("current_forward_speed"))
    prefix_reliable=(metrics.get("current_forward_speed_observed") is True
              and _finite(metrics.get("current_forward_speed_confidence"),0)>=.8
              and metrics.get("current_forward_speed_units")=="viewport_heights_per_second"
              and prefix_current is not None and prefix_current>=0)
    current=_finite(metrics.get("forward_speed"))
    reliable=(metrics.get("forward_speed_observed") is True
              and metrics.get("forward_speed_confidence",0)>=.8
              and metrics.get("forward_speed_units")=="viewport_heights_per_second"
              and current is not None and current>=0)
    if prefix_reliable:
        current=prefix_current;reliable=True
        forward=current;forward_source=str(metrics.get("current_forward_speed_source","prefix_landmark_measurement"))
    elif reliable:
        forward=current;forward_source=str(metrics.get("forward_speed_source","explicit_measurement"))
    else:
        forward=(_finite(parameters["current_forward_speed_hypothesis"]) if "current_forward_speed_hypothesis" in parameters
                 else max(normal.forward,boosted.forward))
        if forward is None or forward<0:
            raise ValueError("current_forward_speed_hypothesis 必须非负")
        forward_source="configured_current_speed_hypothesis_not_observed"
    history=[];history_source="unknown_input_constant_observed_vx"
    supplied=metrics.get("mining_input_history")
    declared_source=metrics.get("mining_input_history_source")
    if supplied is not None and declared_source in {"backend_send_state","commands.jsonl.after","guard_submission_after"}:
        if not isinstance(supplied,(list,tuple)):
            raise ValueError("mining_input_history 必须为实际发送记录列表")
        previous=None
        for item in supplied:
            if not isinstance(item,dict):raise ValueError("输入历史记录必须为对象")
            timestamp=_finite(item.get("after"))
            state=item.get("state")
            if isinstance(state,dict):
                if (not set(state.get("buttons",[]))<={"ZR"}
                        or state.get("left_stick",{}).get("y",0)!=0
                        or any(state.get("right_stick",{}).get(k,0)!=0 for k in ("x","y"))):
                    raise ValueError("输入历史包含横向/ZR模型以外的按键或轴")
                axis=_finite(state.get("left_stick",{}).get("x",0))
                axis=None if axis is None else axis/32767
                zr="ZR" in state.get("buttons",[])
            else:
                axis=_finite(item.get("axis"));zr=item.get("zr")
            if (timestamp is None or axis is None or abs(axis)>1 or type(zr) is not bool
                    or timestamp>obs.timestamp+age+1e-6 or (previous is not None and timestamp<previous)):
                raise ValueError("输入历史时钟/归一化轴/模式无效")
            previous=timestamp;history.append((timestamp-obs.timestamp+delay,axis,zr))
        if history and history[0][0]<=0:
            history_source=("guard_submission_reports_approx_backend_after" if declared_source=="guard_submission_after"
                            else "actual_backend_reports_shifted_on_capture_clock")
        else:
            history=[];history_source="incomplete_history_constant_observed_vx"
    if parameters.get("require_verified_input_history") is True and not history:
        raise ValueError("缺少覆盖观测时刻的实际发送历史")
    forward_model,seed_speed,forward_diagnostics=seed_context(obs,parameters)
    if forward_model is not None and not history:
        forward_diagnostics.update(active=False,fallback_reason="missing_actual_input_history_for_prefix")
        forward_model,seed_speed=None,None
    from_modes=parameters.get("prefix_forward_from_input_history",False)
    if type(from_modes) is not bool:
        raise ValueError("prefix_forward_from_input_history 必须为布尔值")
    from_modes=bool(from_modes and history and not reliable)
    if from_modes:
        forward_source="submitted_mode_piecewise_configured_speed_not_measured_transition"
    if forward_model is not None:
        forward=seed_speed;forward_source="capture_time_seed_integrated_over_actual_submitted_modes"
        reliable=False;from_modes=False
    return {"delay":delay,"age":age,"seconds":age+planning+submission+delay,"forward":forward,
        "forward_model":forward_model,"forward_seed_speed":seed_speed,
        "forward_source":forward_source,"forward_observed":reliable,"history":history,"forward_from_modes":from_modes,
        "history_source":history_source,"normal":normal,"zr":boosted,
        "diagnostics":{"semantics":"backend_report_after_to_capture_clock_effective_delay",
            "observation_age_seconds":age,"observation_age_source":"metric" if "observation_age_seconds" in metrics else "assumed_zero",
            "effective_delay_seconds":delay,"command_submission_delay_seconds":submission,
            "command_submission_delay_source":submission_source,
            "dispatch_delay_seconds":submission,"dispatch_delay_source":submission_source,
            "planning_delay_seconds":planning,"planning_delay_source":planning_source,
            "planning_delay_is_budget_not_current_measurement":True,
            "prefix_formula":"observation_age + planning_budget + dispatch + effective_delay",
            "prefix_seconds":age+planning+submission+delay,
            "extra_capture_buffer_seconds_added":0.0,"input_history_source":history_source,
            "input_history_declared_source":declared_source,
            "input_history_for_planning_only":True,
            "unsubmitted_inflight_requests_observed":False,
            "current_scroll_speed":None if from_modes else forward,"current_scroll_source":forward_source,
            "prefix_forward_from_input_history":from_modes,
            "prefix_mode_forward_speeds":{"normal":normal.forward,"zr":boosted.forward} if from_modes else {},
            "current_scroll_observed":reliable,
            "current_scroll_observation_scope":"prefix_only" if prefix_reliable else "legacy_or_hypothesis",
            "current_scroll_confidence_kind":metrics.get("current_forward_speed_confidence_kind","heuristic_not_probability") if prefix_reliable else None,
            "forward_dynamics":forward_diagnostics,
            "future_scroll_assumption":("short_window_fitted_zr_acceleration_normal_exponential_response_no_terminal_speed"
                if forward_model is not None else "constant_current_or_configured_mode_speed_not_acceleration_model")}}


def _prediction_prefix(player, motion, settings):
    """Replay only already submitted inputs before a NEW action can act."""
    context=settings.get("capture_clock")
    if context is None:
        velocity=max(-motion.maximum_speed,min(motion.maximum_speed,player.vx))
        return [(player.x,player.y,0.0,velocity),
                (player.x+velocity*settings["latency"],player.y,settings["latency"],velocity)]
    if "prefix_points" in context:return context["prefix_points"]
    duration=context["seconds"];x=player.x;velocity=player.vx;now=0.;depth=player.y
    forward_model=context.get("forward_model");forward_speed=context.get("forward_seed_speed")
    points=[(x,player.y,0.,velocity)]
    changes=context["history"];cursor=0;axis=0.;active=context["normal"]
    while cursor<len(changes) and changes[cursor][0]<=0:
        _,axis,zr=changes[cursor];active=context["zr"] if zr else context["normal"];cursor+=1
    while now<duration-1e-10:
        end=min(duration,now+.02)
        if cursor<len(changes):end=min(end,max(now,changes[cursor][0]))
        if end>now:
            dt=end-now
            if changes:
                distance,velocity=_advance(active,velocity,axis,dt)
            else:distance=velocity*dt
            if forward_model is not None:
                ds,forward_speed=forward_model.advance(forward_speed,active is context["zr"],dt)
                depth+=ds
            else:depth+=(active.forward if context.get("forward_from_modes") else context["forward"])*dt
            x+=distance;now=end
            points.append((x,depth,now,velocity))
        while cursor<len(changes) and changes[cursor][0]<=now+1e-10:
            _,axis,zr=changes[cursor];active=context["zr"] if zr else context["normal"];cursor+=1
    context["prefix_points"]=points
    context["prefix_end_forward_speed"]=forward_speed
    return points


def _normal_depth_response(motion, settings, seconds):
    context=settings.get("capture_clock")
    if context is not None and context.get("forward_model") is not None:
        return context["forward_model"].advance(context["prefix_end_forward_speed"],False,seconds)[0]
    return motion.forward*seconds


def _mixed_deadline_context(obs, parameters, capture_clock, mixed_prefixes):
    """Constrain the NEW plan to the supervisor's already promised release.

    release_at is the host lease request clock, not console execution. The
    same dispatch D and effective L apply to the release and the new action.
    A normal decision must be solved with normal dynamics, never obtained by
    removing ZR from a fast plan's first lateral command.
    """
    enabled=parameters.get("mixed_deadline_guard",False)
    if type(enabled) is not bool:raise ValueError("mixed_deadline_guard 必须为布尔值")
    diagnostics={"enabled":enabled,"phase":"disabled","normal_replanned":False,
                 "full_fast_rearm_verified":False}
    if not enabled:return True,mixed_prefixes,diagnostics
    supplied=obs.metrics.get("mining_mixed_guard")
    if (capture_clock is None or not isinstance(supplied,dict) or supplied.get("enabled") is not True
            or supplied.get("phase") not in {"eligible","scheduled","braking"}):
        diagnostics.update(phase="invalid",reason="missing_valid_capture_clock_supervisor_guard",force_normal=True)
        return False,(),diagnostics
    phase=supplied["phase"]
    diagnostics.update(phase=phase,force_normal=False)
    if phase=="eligible":return True,mixed_prefixes,diagnostics
    if phase=="braking":
        allowed=supplied.get("allow_full_fast_rearm") is True
        diagnostics.update(allow_full_fast_rearm=allowed,force_normal=not allowed,
            reason="only_complete_pure_fast_can_rearm" if allowed else "await_normal_replan_and_observed_release")
        return allowed,(),diagnostics
    deadline=_finite(supplied.get("release_at"))
    if deadline is None:
        diagnostics.update(reason="invalid_release_clock",force_normal=True)
        return False,(),diagnostics
    dispatch=capture_clock["diagnostics"]["dispatch_delay_seconds"]
    release_effect=deadline+dispatch+capture_clock["delay"]
    action_effect=obs.timestamp+capture_clock["seconds"]
    remaining=release_effect-action_effect
    # Existing profile prefixes are upper bounds. A late frame may leave
    # less than one step; _search splits exactly at this remaining time.
    prefixes=sorted({min(seconds,remaining) for seconds in mixed_prefixes if remaining>1e-6},reverse=True)
    diagnostics.update(release_at=deadline,release_clock="host_lease_request",remaining_zr_seconds=max(0.,remaining),
        release_effect_capture_clock=release_effect,new_action_effect_capture_clock=action_effect,
        remaining_formula="release_at + dispatch + effective_delay - (capture + prefix)",
        force_normal=not bool(prefixes),reason="fixed_release_deadline" if prefixes else "release_due_before_new_action")
    # Even a newly found pure-fast path cannot silently cancel this deadline.
    return False,prefixes,diagnostics


def _time_stars(obs, road):
    """Visible candidates only; neither this list nor a route confirms +time."""
    result=[];keys=set();geometry=set()
    for entity in obs.of_kind("time_star"):
        if (not _valid_entity(entity) or entity.confidence<.8 or not 0<entity.h<=.15 or entity.w>.15
                or not any(a<=entity.x<=b for a,b in road.intervals(entity.y))):continue
        gkey=f"g:{round(entity.x/.012)}:{round(entity.y/.012)}"
        key=f"t:{entity.track_id}" if type(entity.track_id) is int and entity.track_id>=0 else gkey
        if key in keys or gkey in geometry:continue
        keys.add(key);geometry.add(gkey);result.append((key,entity))
    return tuple(result)


def _overlap_area(x, y, entity, rx, ry):
    """Rectangle overlap in the SAME conservative envelope as normal search."""
    width=max(0,min(x+rx,entity.x+entity.w/2)-max(x-rx,entity.x-entity.w/2))
    height=max(0,min(y+ry,entity.y+entity.h/2)-max(y-ry,entity.y-entity.h/2))
    return width*height


def _prefix_rock_contacts(player, hazards, prefix, rx, ry, geometry=BOX_GEOMETRY):
    current={i for i,e in enumerate(hazards) if e.kind in ("obstacle","rock")
             and geometry.overlap_area(player.x,player.y,e,rx,ry)>1e-12}
    inevitable={i for i,e in enumerate(hazards) if e.kind in ("obstacle","rock")
                and any(geometry.contact_fraction(a[0],a[1],b[0],b[1],e,rx,ry) is not None
                        for a,b in zip(prefix,prefix[1:]))}
    return current,inevitable


def _wall_recovery_obstacle_check(player, hazards, motion, settings, control):
    """The inward wall-recovery direction must not push into a visible rock.

    Old-input contacts cannot be undone. Compare their remaining overlap
    against neutral, and reject any new obstacle touched by the new action.
    This checks a short normal-model response, not a collision-free route.
    """
    rx,ry=settings["radius_x"],settings["radius_y"]
    geometry=settings.get("geometry",BOX_GEOMETRY)
    dt=min(.1,settings["step_seconds"])
    if not hazards:return True,{"checked_hazards":0,"normal_step_seconds":dt}
    prefix=_prediction_prefix(player,motion,settings)
    x,y,_,velocity=prefix[-1]
    fractions=(0.,.25,.5,.75,1.)
    def points(axis):
        return [(x+_advance(motion,velocity,axis,dt*t)[0],y+_normal_depth_response(motion,settings,dt*t)) for t in fractions]
    candidate,neutral=points(control),points(0.)
    current,inevitable=_prefix_rock_contacts(player,hazards,prefix,rx,ry,geometry)
    existing=current|inevitable
    diagnostics={"checked_hazards":len(hazards),"normal_step_seconds":dt,
        "current_rock_contacts":len(current),"uncontrollable_prefix_rock_contacts":len(inevitable-current),
        "comparison":"no_new_obstacle_and_no_worse_existing_overlap_than_neutral"}
    for i,e in enumerate(hazards):
        if i in existing:
            if any(geometry.overlap_area(a,b,e,rx,ry)>geometry.overlap_area(c,d,e,rx,ry)+1e-12
                   for (a,b),(c,d) in zip(candidate,neutral)):
                return False,{**diagnostics,"blocked_by":"increases_existing_rock_overlap","hazard_index":i}
        elif any(geometry.contact_fraction(a[0],a[1],b[0],b[1],e,rx,ry) is not None
                 for a,b in zip(candidate,candidate[1:])):
            return False,{**diagnostics,"blocked_by":"new_obstacle_contact","hazard_index":i}
    return True,diagnostics


def _contact_mitigation(obs, player, road, hazards, motion, settings, maximum_axis, base, failure):
    """Reduce existing contact relative to neutral; never claim a safe route.

    A current rock contact, a rock contact already inevitable under old inputs,
    or a side-wall overlap during input latency can open this opt-in fallback.
    Purely future blocked routes remain neutral. Geometry is checked through
    one normal-model step; the real game's response is still uncalibrated.
    """
    if (player.h <= 0 or obs.metrics.get("ambiguous_player") is True
            or obs.of_kind("player_hit")):
        return None
    rx,ry=settings["radius_x"],settings["radius_y"]
    geometry=settings.get("geometry",BOX_GEOMETRY)
    dt=min(.1,settings["step_seconds"])
    prefix=_prediction_prefix(player,motion,settings)
    if settings.get("capture_clock") is not None:
        initial_x,y0,_,velocity=prefix[-1]
    else:
        y0=player.y
        velocity=max(-motion.maximum_speed,min(motion.maximum_speed,player.vx))
        initial_x=player.x+velocity*settings["latency"]
    y1=y0+_normal_depth_response(motion,settings,dt)
    if road.rows[0][0] > y0-ry or road.rows[-1][0] < y1+ry:
        return None
    constraints=_sweep_constraints(road,y0,y1,rx,ry)
    if constraints is None:
        return None
    current_spans=_spans(road,y0,rx,ry)
    if len(current_spans)!=1:
        return None
    latency_wall=not current_spans[0][0] <= initial_x <= current_spans[0][1]
    current_rocks,prefix_rocks=_prefix_rock_contacts(player,hazards,prefix,rx,ry,geometry)
    existing=sorted(current_rocks|prefix_rocks)
    if not latency_wall and not existing:
        return None
    context=settings.get("capture_clock")
    if context is not None and context.get("forward_model") is not None:
        knot_fractions=[context["forward_model"].time_to_distance(context["prefix_end_forward_speed"],False,
                         (y1-y0)*t)/dt for t,_,_ in constraints]
    else:knot_fractions=[t for t,_,_ in constraints]
    fractions=sorted({0.0,.25,.5,.75,1.0,*[max(0.,min(1.,t)) for t in knot_fractions]})
    spans=[_spans(road,y0+_normal_depth_response(motion,settings,dt*t),rx,ry) for t in fractions]
    # A split, insufficient width, or incomplete envelope cannot justify an
    # emergency side choice even if a narrower branch happened to look closer.
    if any(len(row)!=1 for row in spans):
        return None
    def evaluate(control):
        points=[(initial_x+_advance(motion,velocity,control,dt*t)[0],y0+_normal_depth_response(motion,settings,dt*t))
                for t in fractions]
        walls=[(max(0,row[0][0]-x),max(0,x-row[0][1]))
               for (x,_),row in zip(points,spans)]
        rocks=[[geometry.overlap_area(x,y,hazards[i],rx,ry) for i in existing] for x,y in points]
        return points,walls,rocks
    neutral_points,neutral_walls,neutral_rocks=evaluate(0)
    best=None
    controls=sorted({max(-maximum_axis,min(maximum_axis,u)) for u in (-1.0,-.5,0.0,.5,1.0)})
    for control in controls:
        if not control:continue
        points,walls,rocks=evaluate(control)
        # Do not trade a new opposite-wall or second-rock contact for a smaller
        # aggregate number. Every existing contact must be no worse than neutral.
        if any(a>b+1e-12 for pair,baseline in zip(walls,neutral_walls) for a,b in zip(pair,baseline)):
            continue
        if any(a>b+1e-12 for row,baseline in zip(rocks,neutral_rocks) for a,b in zip(row,baseline)):
            continue
        if settings.get("capture_clock") is not None:
            prefix_pairs=[((a[0],a[1]),(b[0],b[1])) for a,b in zip(prefix,prefix[1:])]
            segments=[*prefix_pairs,*zip(points,points[1:])]
        else:segments=[((player.x,y0),points[0]),*zip(points,points[1:])]
        if any(geometry.contact_fraction(a[0],a[1],b[0],b[1],e,rx,ry) is not None
               for i,e in enumerate(hazards) if i not in existing for a,b in segments):
            continue
        wall_gain=sum(b-a for pair,baseline in zip(walls,neutral_walls) for a,b in zip(pair,baseline))/rx
        rock_gain=sum(b-a for row,baseline in zip(rocks,neutral_rocks) for a,b in zip(row,baseline))/(4*rx*ry)
        gain=wall_gain+rock_gain
        if gain <= 1e-6:continue
        rank=(gain,-abs(control))
        if best is None or rank>best[0]:
            best=(rank,control,points[-1],wall_gain,rock_gain)
    if best is None:return None
    _,control,end,wall_gain,rock_gain=best
    diagnostics={**base,"mode":"contact_mitigation","motion_calibrated":False,"path":[],
        "acceleration_allowed":False,"route_status":"contact_not_collision_free",
        "mitigation_trigger":failure,"latency_wall_contact":latency_wall,
        "existing_rock_contacts":len(existing),"current_rock_contacts":len(current_rocks),
        "uncontrollable_prefix_rock_contacts":len(prefix_rocks-current_rocks),"normal_step_seconds":dt,
        "predicted_end":{"x":end[0],"depth":end[1]},
        "relative_to_neutral":{"wall_overlap_reduction":wall_gain,"rock_overlap_reduction":rock_gain},
        "risks":["严格无碰撞首步已失败；本输入仅按未标定模型减轻已有接触，不保证脱困",
                 "仅本帧可见蓝色机体适用，下一帧必须重新观测和判断"]}
    return Decision(ControllerState(left_stick=StickVector(round(control*32767),0)),
        "MINING：已接触包络，松开 ZR 并选择比中立更能减轻接触的方向",
        min(obs.confidence,player.confidence),diagnostics)


def _early_route_clearance(plan, road, player, hazards, motion, settings, seconds=.3):
    """Extra clearance outside the existing full-body envelope, in widths.

    Road knots use the same swept constraints as the hard search. For rocks,
    segment/expanded-rectangle distance is minimized at ends, corner
    projections and boundary crossings; normalized y is scaled by 480/640.
    Ambiguous road branches cannot justify waiving a steering gate.
    """
    rx,ry=settings["radius_x"],settings["radius_y"]
    geometry=settings.get("geometry",BOX_GEOMETRY)
    prefix=_prediction_prefix(player,motion,settings)[-1]
    before=prefix[:3] if settings.get("capture_clock") is not None else (prefix[0],prefix[1],0.)
    deadline=before[2]+seconds;minimum=math.inf
    for x,y,t,_,_ in plan.path:
        a=before;ratio=min(1.,max(0.,(deadline-a[2])/max(1e-12,t-a[2])))
        b=(a[0]+(x-a[0])*ratio,a[1]+(y-a[1])*ratio)
        constraints=_sweep_constraints(road,a[1],b[1],rx,ry)
        if constraints is None:return None
        for fraction,left,right in constraints:
            px=a[0]+(b[0]-a[0])*fraction
            minimum=min(minimum,px-left,right-px)
        for e in hazards:
            minimum=min(minimum,geometry.segment_clearance(a[0],a[1],b[0],b[1],e,rx,ry))
        if t>=deadline-1e-9:break
        before=(x,y,t)
    return minimum if math.isfinite(minimum) else None


def _turn_authority(plan, motion, maximum_axis, parameters, boosted_seconds=None, *, clearance=None, complete=False):
    """Relative early demand, not an unreachable absolute stick threshold."""
    early=list(plan.path[:3])
    if boosted_seconds is not None:
        start=plan.path[0][2]-min(parameters.get("step_seconds",.1),boosted_seconds)
        early=[p for p in early if p[2]<=start+boosted_seconds+1e-8]
    fraction=max((abs(p[3])/maximum_axis for p in early),default=0.)
    slope=max((abs(b[0]-a[0])/max(.01,b[1]-a[1]) for a,b in zip(early,early[1:])),default=0.)
    lateral=(min(motion.maximum_speed,motion.input_gain*maximum_axis/motion.drag+abs(motion.neutral_velocity))
             if motion.drag>0 else motion.maximum_speed)
    capacity=lateral/motion.forward
    # Compare lateral distance/time directly. Fixed dy/dt would incorrectly
    # reject a route whose forward speed is still building after ZR begins.
    lateral_rate=max((abs(b[0]-a[0])/max(1e-9,b[2]-a[2]) for a,b in zip(early,early[1:])),default=0.)
    utilization=lateral_rate/max(1e-9,lateral)
    axis_exceeded=fraction>parameters.get("sharp_turn_axis_fraction",.95)
    waived=(parameters.get("allow_full_axis_with_clearance") is True and complete
            and clearance is not None and clearance>=parameters.get("full_axis_minimum_clearance",.02))
    sharp=((axis_exceeded and not waived) or utilization>parameters.get("sharp_turn_authority_fraction",1.05)
           or slope>parameters.get("sharp_turn_slope",1.05))
    return sharp,{"early_axis_fraction":fraction,"early_slope":slope,
        "available_steady_slope_hypothesis":capacity,"early_slope_authority_fraction":utilization,
        "authority_comparison":"predicted_lateral_distance_per_second_over_lateral_authority",
        "early_lateral_rate":lateral_rate,
        "axis_fraction_threshold":parameters.get("sharp_turn_axis_fraction",.95),
        "axis_threshold_exceeded":axis_exceeded,"axis_gate_waived":bool(axis_exceeded and waived),
        "early_envelope_clearance_widths":clearance,
        "required_extra_clearance_widths":parameters.get("full_axis_minimum_clearance",.02)}


def _route_reference(obs, weight):
    """Explicit old-route reprojection; never compare same-index camera rows."""
    if weight<=0:return (),"disabled"
    previous=obs.metrics.get("mining_previous_route")
    if not isinstance(previous,dict):return (),"missing_previous_route"
    timestamp=_finite(previous.get("timestamp"));speed=_finite(previous.get("forward_speed"))
    source=previous.get("scroll_source")
    if (timestamp is None or not 0<=obs.timestamp-timestamp<=.5 or speed is None or speed<0
            or source not in {"measured_scroll","configured_scroll_hypothesis"}):
        return (),"unusable_previous_route_clock_or_scroll"
    shift=(obs.timestamp-timestamp)*speed
    points=[]
    for p in previous.get("path",[]):
        if not isinstance(p,dict):return (),"invalid_previous_route"
        x,y=_finite(p.get("x")),_finite(p.get("depth"))
        if x is None or y is None:return (),"invalid_previous_route"
        points.append((y-shift,x))
    return tuple(sorted(points)),"reprojected_"+source


def _route_x(points, depth):
    for a,b in zip(points,points[1:]):
        if a[0]<=depth<=b[0] and b[0]>a[0]:
            return a[1]+(b[1]-a[1])*(depth-a[0])/(b[0]-a[0])
    return None


def _search(road, player, hazards, motion, settings, boost=False, switch_after=None, suffix=None, guide_side=None):
    rx,ry=settings["radius_x"],settings["radius_y"]
    geometry=settings.get("geometry",BOX_GEOMETRY)
    prefix=_prediction_prefix(player,motion,settings)
    for a,b in zip(prefix,prefix[1:]):
        if not road.swept_safe(a[0],a[1],b[0],b[1],rx,ry):
            return None,0,"input_latency_envelope_outside_road"
        if settings.get("capture_clock") is not None and any(
                geometry.contact_fraction(a[0],a[1],b[0],b[1],e,rx,ry) is not None for e in hazards):
            return None,0,"pre_action_prefix_hits_observed_obstacle"
    initial_x,initial_depth,prefix_seconds,initial_velocity=prefix[-1]
    if settings.get("capture_clock") is None:prefix_seconds=0.0
    pickup_x,pickup_y=settings["star_pickup_radii"]
    # A contact already inevitable under old inputs is not a reward for the
    # new choice. In particular it is never added to the observed countdown.
    stars=[(key,e) for key,e in settings["stars"] if not any(
        contact_fraction(a[0],a[1],b[0],b[1],e,pickup_x,pickup_y) is not None for a,b in zip(prefix,prefix[1:]))]
    def star_potential(x,depth,picked):
        # A telescoping potential keeps an approaching route in the beam
        # BEFORE contact. It is not another per-step pickup reward: summing
        # P(old)-P(new) cancels all intermediate values exactly.
        return sum(.35*settings["star_reward"]
            *math.exp(-(max(0,abs(x-e.x)-pickup_x-e.w/2)/.12)**2)
            *math.exp(-max(0,e.y-depth)/.6)
            for key,e in stars if key not in picked and depth-pickup_y<=e.y+e.h/2
            and e.y-e.h/2<=min(player.y+settings["lookahead_depth"],road.rows[-1][0]-ry)+pickup_y)
    start=_Node(initial_x,initial_velocity,initial_depth,0.0,())
    beam=[start]
    expanded=0
    step=settings["step_seconds"]
    final_depth=min(player.y+settings["lookahead_depth"],road.rows[-1][0]-ry)
    if final_depth-initial_depth <= 1e-9:
        return None,expanded,"insufficient_forward_coverage"
    context=settings.get("capture_clock")
    forward_model=context.get("forward_model") if context is not None else None
    forward_speed=context.get("prefix_end_forward_speed") if forward_model is not None else None
    # The search fixes the vertical mode schedule in advance, so every node
    # in a layer shares v/depth. Lateral controls do not alter that schedule.
    arrival_seconds=(forward_model.arrival_time(forward_speed,final_depth-initial_depth,boost,switch_after)
                     if forward_model is not None else None)
    complete=True
    last_beam=beam
    travel_seconds=0.
    for index in range(settings["maximum_steps"]):
        active=suffix if switch_after is not None and travel_seconds>=switch_after-1e-9 else motion
        dt=min(step,arrival_seconds-travel_seconds if forward_model is not None
               else (final_depth-last_beam[0].depth)/active.forward)
        if switch_after is not None and travel_seconds<switch_after-1e-9:
            dt=min(dt,switch_after-travel_seconds)
        if dt <= 1e-6:
            break
        if forward_model is not None:
            ds,next_forward=forward_model.advance(forward_speed,boost and active is motion,dt)
            depth=min(final_depth,last_beam[0].depth+ds)
        else:depth=min(final_depth,last_beam[0].depth+dt*active.forward)
        spans=_spans(road,depth,rx,ry)
        if not spans:
            complete=False
            break
        choices=[]
        constraints=_sweep_constraints(road,last_beam[0].depth,depth,rx,ry)
        layer_stars=[(key,e) for key,e in stars if e.y+e.h/2>=last_beam[0].depth-pickup_y and e.y-e.h/2<=depth+pickup_y]
        for node in beam:
            old_potential=star_potential(node.x,node.depth,node.collected_predicted)
            previous_control=node.path[-1][3] if node.path else settings.get("previous_axis",0.)
            levels=[-1.0,-.5,0.0,.5,1.0]
            if settings.get("steering_change_weight",0)>0:
                levels.extend((previous_control,previous_control-.15,previous_control+.15))
            controls=sorted({max(-settings["maximum_axis"],min(settings["maximum_axis"],u)) for u in levels})
            for control in controls:
                expanded+=1
                # Exact constant-input integration of dv/dt=g*u-drag*v,
                # followed by the explicitly configured lateral speed cap.
                distance,velocity=_advance(active,node.velocity,control,dt)
                x=node.x+distance
                if not any(a <= x <= b for a,b in spans):
                    continue
                if (not _compiled_safe(constraints,node.x,x) if constraints is not None
                        else not road.swept_safe(node.x,node.depth,x,depth,rx,ry)):
                    continue
                if any(geometry.contact_fraction(node.x,node.depth,x,depth,entity,rx,ry) is not None for entity in hazards):
                    continue
                span=next((a,b) for a,b in spans if a <= x <= b)
                center=(span[0]+span[1])/2
                if guide_side is not None:center=span[0]+.01 if guide_side=="left" else span[1]-.01
                margin=min(x-span[0],span[1]-x)
                future_obstacle_cost=sum(math.exp(-((x-entity.x)/max(.035,entity.w/2+rx+.035))**2)
                    *math.exp(-max(0,entity.y-entity.h/2-depth)/.18)
                    for entity in hazards if entity.y+entity.h/2 >= depth-ry)
                # Prefer wide interior geometry and low drift. Future layers
                # influence the FIRST control; no common-x intersection is used.
                center_weight=8. if guide_side is not None else settings.get("center_attraction_weight",1.8)
                rock_weight=0. if guide_side is not None else .8
                cost=node.cost+dt*(center_weight*(x-center)**2+.006*control*control+
                    .04*velocity*velocity+.00025/max(.006,margin)+rock_weight*future_obstacle_cost)
                cost+=settings.get("lateral_travel_weight",0.)*abs(distance)
                cost+=settings.get("steering_change_weight",0)*(control-previous_control)**2
                reference=_route_x(settings.get("previous_route",()),depth)
                if reference is not None:cost+=dt*settings.get("route_continuity_weight",0)*(x-reference)**2
                new_stars=frozenset(key for key,e in layer_stars if key not in node.collected_predicted
                    and contact_fraction(node.x,node.depth,x,depth,e,pickup_x,pickup_y) is not None)
                cost-=settings["star_reward"]*len(new_stars)
                cost+=old_potential-star_potential(x,depth,node.collected_predicted|new_stars)
                elapsed=(node.path[-1][2] if node.path else prefix_seconds)+dt
                path=node.path+((x,depth,elapsed,control,velocity),)
                choices.append(_Node(x,velocity,depth,cost,path,node.collected_predicted|new_stars))
        if not choices:
            complete=False
            break
        # Preserve spatial/velocity alternatives rather than filling the beam
        # with many almost-identical histories at the same location.
        choices.sort(key=lambda node:node.cost)
        unique={}
        for node in choices:
            key=(round(node.x/0.012),round(node.velocity/.12),node.collected_predicted,
                 round(node.path[-1][3]/.15) if settings.get("steering_change_weight",0)>0 else 0)
            if key not in unique:
                unique[key]=node
            if len(unique)>=settings["beam_width"]:
                break
        beam=list(unique.values())
        last_beam=beam
        if forward_model is not None:forward_speed=next_forward
        travel_seconds+=dt
    best=min(last_beam,key=lambda node:node.cost+.08*node.velocity*node.velocity)
    if not best.path:
        return None,expanded,"no_safe_next_step"
    complete=complete and best.depth >= final_depth-1e-6
    return best,expanded,"complete" if complete else "partial"


def _guided_normal_alternative(road,player,hazards,motion,settings,plan):
    """At most one opposite-side search after a default normal partial route.

    The nearby circle is a candidate explanation of the dead branch, not a
    measured collision cause. Acceptance still requires the complete horizon
    and every ordinary hard wall/shape sweep; no partial is relabelled safe.
    """
    geometry=settings.get("geometry",BOX_GEOMETRY)
    candidates=[e for e in hazards if geometry.shape(e) is not None and geometry.bounds(e)[3]>=player.y]
    if not candidates:return None,0,"not_attempted",{"attempted":False,"reason":"no_current_ring_shape"}
    endpoint=plan.path[-1]
    entity=min(candidates,key=lambda e:abs(e.y-endpoint[1]))
    circle=geometry.shape(entity);center=circle.center_x_pixels/640
    side="left" if endpoint[0]>=center else "right"
    guided_settings={**settings,"beam_width":min(settings["beam_width"],12)}
    started=perf_counter()
    alternative,expanded,status=_search(road,player,hazards,motion,guided_settings,guide_side=side)
    diagnostics={"attempted":True,"side":side,"candidate_track_id":entity.track_id,
        "default_partial_end_depth":endpoint[1],"status":status,"expanded_nodes":expanded,
        "extra_search_ms":(perf_counter()-started)*1000,"searches":1,"normal_mode_only":True,
        "center_weight":8.,"soft_rock_repulsion_weight":0.,"all_selected_geometry_constraints_checked":True,
        "wall_player_envelope_unchanged":True,
        "collision_shape_measured":False,"selected":status=="complete" and alternative is not None}
    return alternative,expanded,status,diagnostics


def decide_mining_reactive(obs, parameters=None):
    """Return a bounded, local MINING steering suggestion, never send input.

    Optional measured models are ``motion_models.normal`` and ``.zr`` with
    forward_speed (viewport heights/s), input_gain_per_second (widths/s² per
    normalized stick), drag_per_second, maximum_lateral_speed (widths/s).
    ``motion_calibrated`` is diagnostic only; runtime authorization is separate.
    ``allow_hit_recovery=True`` explicitly permits limited inward steering from
    a current ``player_hit`` appearance while the normal play phase is unknown.
    ``allow_contact_mitigation=True`` permits nonaccelerated contact reduction
    from a current blue body only after strict safe-next-step search fails.
    ``prediction_timing='capture_clock'`` explicitly interprets
    effective_delay_seconds as backend-report-to-capture effective delay.
    observation_age_seconds and command_submission_delay_seconds are separate.
    Optional mining_input_history entries
    are actual {after,state} reports (or normalized {after,axis,zr}) with source
    backend_send_state/commands.jsonl.after. guard_submission_after is accepted
    as an explicitly approximate planning clock, never substituted into fits.
    Missing history is labelled unknown.
    steering_change_weight and route_continuity_weight default to zero. These
    are in-search soft costs, never a post-search input filter. Optional
    mixed_zr_prefix_seconds tries a short ZR prefix then a normal suffix on
    the full visible horizon, using one shared capture-clock prefix only.
    """
    started=perf_counter()
    parameters=parameters or {}
    if obs.game != Game.MINING:
        return _empty("等待可靠的游戏运行画面")
    remaining=_finite(obs.metrics.get("time_remaining",1))
    if obs.metrics.get("game_ended") is True or remaining is None or remaining <= 0:
        return _empty("倒计时结束，等待结算")
    age=_finite(obs.metrics.get("observation_age_seconds",0))
    maximum_age=_finite(parameters.get("maximum_observation_age_seconds",.15))
    if age is None or maximum_age is None or maximum_age <= 0 or age < 0 or age > maximum_age:
        return _empty("画面过期，松开输入")
    try:geometry=ObstacleGeometry(obs,parameters)
    except (ValueError,TypeError) as error:return _empty(f"碰撞几何配置无效：{error}")
    if obs.phase == "unknown" and parameters.get("allow_hit_recovery") is True:
        return _hit_recovery(obs,parameters,geometry)
    if obs.phase != "playing" or _finite(obs.confidence,0) < .8:
        return _empty("等待可靠的游戏运行画面")
    players=obs.of_kind("player")
    if (len(players)!=1 or not _valid_entity(players[0]) or players[0].confidence < .8
            or players[0].samples < 2 or _finite(players[0].vx) is None):
        return _empty("玩家缺失、闪烁或尚无连续轨迹")
    player=players[0]
    corridors=[e for e in obs.of_kind("corridor") if _valid_entity(e) and e.confidence >= .8]
    if len({round(e.y,5) for e in corridors})<2:
        return _empty("道路分层观测不足")
    road=Road(corridors)
    try:
        normal,normal_source=_motion(parameters,obs,"normal")
        boosted,boosted_source=_motion(parameters,obs,"zr")
        margin=_positive(parameters,"wall_margin",.012)
        uncertainty=_positive(parameters,"position_uncertainty",.004)
        latency=_positive(parameters,"latency_seconds",.035)
        depth=_positive(parameters,"lookahead_depth",.32)
        step=_positive(parameters,"step_seconds",.10)
        beam_width=int(parameters.get("beam_width",14))
        maximum_steps=int(parameters.get("maximum_steps",14))
        maximum_axis=min(1.0,_positive(parameters,"maximum_axis",1.0))
        star_reward=_finite(parameters.get("star_reward",0.0))
        steering_change_weight=_finite(parameters.get("steering_change_weight",0.0))
        route_continuity_weight=_finite(parameters.get("route_continuity_weight",0.0))
        center_attraction_weight=_finite(parameters.get("center_attraction_weight",1.8))
        lateral_travel_weight=_finite(parameters.get("lateral_travel_weight",0.0))
        if (steering_change_weight is None or steering_change_weight<0
                or route_continuity_weight is None or route_continuity_weight<0
                or center_attraction_weight is None or center_attraction_weight<0
                or lateral_travel_weight is None or lateral_travel_weight<0):
            raise ValueError("路线/转向连续性软权重必须非负")
        mixed_prefixes=parameters.get("mixed_zr_prefix_seconds",())
        if (not isinstance(mixed_prefixes,(tuple,list))
                or any(_finite(s) is None or not .05<=s<=.5 for s in mixed_prefixes)):
            raise ValueError("mixed_zr_prefix_seconds 必须为 .05 到 .5 秒的列表")
        mixed_prefixes=sorted(set(mixed_prefixes),reverse=True)
        for key in ("allow_full_axis_with_clearance","compare_star_modes","star_mode_upper_bound_pruning","circle_guided_search","straightness_clear_only"):
            if type(parameters.get(key,False)) is not bool:raise ValueError(f"{key} 必须为布尔值")
        _positive(parameters,"full_axis_minimum_clearance",.02)
        star_mode_margin=_positive(parameters,"star_mode_gain_margin_seconds",.25)
        for key,default in (("sharp_turn_axis_fraction",.95),("sharp_turn_authority_fraction",1.05),("sharp_turn_slope",1.05)):
            _positive(parameters,key,default)
        pickup_scale=_positive(parameters,"star_pickup_radius_scale",.75)
        if star_reward is None or star_reward<0 or pickup_scale>1:
            raise ValueError("star_reward 必须非负，star_pickup_radius_scale 不得放大机体")
        timing=parameters.get("prediction_timing","legacy")
        if timing not in {"legacy","capture_clock"}:raise ValueError("prediction_timing 无效")
        capture_clock=(_capture_clock_context(obs,parameters,normal,boosted,age,latency)
                       if timing=="capture_clock" else None)
        allow_full_fast,mixed_prefixes,mixed_guard=_mixed_deadline_context(obs,parameters,capture_clock,mixed_prefixes)
        if not 1<=beam_width<=64 or not 1<=maximum_steps<=30 or not .02<=step<=.25 or depth>0.8:
            raise ValueError("搜索范围超出实时策略限制")
    except (ValueError,TypeError,AttributeError) as error:
        return _empty(f"运动或搜索配置无效：{error}")
    rx=player.w/2+margin+uncertainty
    ry=player.h/2+uncertainty
    settings={"radius_x":rx,"radius_y":ry,"latency":latency,"lookahead_depth":depth,"geometry":geometry,
              "step_seconds":step,"beam_width":beam_width,"maximum_steps":maximum_steps,
              "maximum_axis":maximum_axis,"capture_clock":capture_clock,"star_reward":star_reward,
              "stars":_time_stars(obs,road) if star_reward>0 else (),
              "star_pickup_radii":(player.w*pickup_scale/2,player.h*pickup_scale/2)}
    previous_axis=(capture_clock["history"][-1][1] if capture_clock is not None and capture_clock["history"] else 0.)
    previous_route,route_source=_route_reference(obs,route_continuity_weight)
    settings.update(steering_change_weight=steering_change_weight,route_continuity_weight=route_continuity_weight,
                    previous_axis=previous_axis,previous_route=previous_route,
                    center_attraction_weight=center_attraction_weight,lateral_travel_weight=lateral_travel_weight)
    if road.rows[0][0] > player.y-ry or road.rows[-1][0] < player.y+ry+.03:
        return _empty("道路未覆盖玩家完整尺寸及前方")
    spans=_spans(road,player.y,rx,ry)
    if not spans:
        return _empty("当前道路宽度不足完整玩家包络")
    current_safe=any(a<=player.x<=b for a,b in spans)
    nearest=_closest(spans,player.x)
    raw_hazards=[e for e in obs.entities if e.kind in ("obstacle","rock","wall") and _valid_entity(e) and e.confidence>=.8]
    buff_bound=_finite(obs.metrics.get("buff_remaining_lower_bound_seconds"),0)
    powered=(any(e.confidence>=.8 for e in obs.of_kind("egg_buff"))
             and parameters.get("allow_powered_rocks") is True
             and buff_bound >= depth/min(normal.forward,boosted.forward)+latency)
    hazards=[e for e in raw_hazards if e.kind=="wall" or not powered]
    forward_limit=min(player.y+depth,road.rows[-1][0]-ry)
    visible_hazard_ahead=any(bottom>=player.y-ry and top<=forward_limit+ry
        for e in hazards for _,_,top,bottom in [geometry.bounds(e)])
    visible_star_ahead=any(e.y+e.h/2>=player.y-ry and e.y-e.h/2<=forward_limit+ry for _,e in settings["stars"])
    clear_straightness=not visible_hazard_ahead and not visible_star_ahead
    if parameters.get("straightness_clear_only") is True and not clear_straightness:
        settings.update(center_attraction_weight=1.8,lateral_travel_weight=0.)
    nearest_hazard=min((top-player.y for e in hazards for left,right,top,bottom in [geometry.bounds(e)]
                        if bottom>=player.y-ry and left-player.w/2-.10<player.x<right+player.w/2+.10),default=math.inf)
    wall_distance=min((min(player.x-a,b-player.x) for a,b in spans if a<=player.x<=b),default=-abs(nearest-player.x))
    base={"policy":"mining_reactive","experimental":True,
          "motion_calibrated":parameters.get("motion_calibrated") is True,
          "forward_speed_source":{"normal":normal_source,"zr":boosted_source},
          "wall_distance":wall_distance,"nearest_hazard_depth":None if math.isinf(nearest_hazard) else nearest_hazard,
          "powered_rocks_allowed":powered,"path":[],"risks":[]}
    base["mixed_guard"]=mixed_guard
    base["collision_geometry"]=geometry.diagnostics()
    base["guided_circle_search"]={"enabled":parameters.get("circle_guided_search") is True,"attempted":False}
    base.update(center_attraction_weight=settings["center_attraction_weight"],lateral_travel_weight=settings["lateral_travel_weight"])
    base["straightness_cost"]={"clear_only":parameters.get("straightness_clear_only") is True,
        "clear_of_visible_stars_and_hazards":clear_straightness,"visible_hazard_ahead":visible_hazard_ahead,
        "visible_star_ahead":visible_star_ahead,"configured_center_weight":center_attraction_weight,
        "configured_lateral_travel_weight":lateral_travel_weight,
        "effective_center_weight":settings["center_attraction_weight"],
        "effective_lateral_travel_weight":settings["lateral_travel_weight"],"post_output_smoothing":False}
    base.update(star_reward=star_reward,visible_time_star_candidates=len(settings["stars"]),
                collected_predicted=[],star_reward_total_predicted=0.0,star_collection_confirmed=False,
                star_guidance_is_not_collection=True)
    base.update(steering_change_weight=steering_change_weight,route_continuity_weight=route_continuity_weight,
                previous_axis=previous_axis,previous_axis_source=(capture_clock["history_source"]
                    if capture_clock is not None and capture_clock["history"] else "unknown_assumed_neutral_for_soft_cost"),
                previous_route_source=route_source)
    if capture_clock is not None:
        prefix=_prediction_prefix(player,normal,settings)
        end=prefix[-1]
        base["timing_compensation"]={**capture_clock["diagnostics"],
            "new_action_can_affect_capture_state_at":{"x":end[0],"depth":end[1],"seconds":end[2],"velocity":end[3]},
            "future_command_controls_prefix":False}
        base["forward_dynamics"]={**capture_clock["diagnostics"]["forward_dynamics"],
            "prefix_end_forward_speed":capture_clock.get("prefix_end_forward_speed"),
            "layer_state_shared_across_lateral_nodes":True}
    if not current_safe:
        # Contact was already present in the observation. Do not pretend this
        # starts with a collision-free envelope; offer an explicit recovery.
        delta=nearest-player.x
        control=math.copysign(min(maximum_axis,max(.4,min(1,abs(delta)*10))),delta)
        if not control:
            return _empty("玩家包络已出界且无法确定恢复方向",base)
        allowed,recovery_check=_wall_recovery_obstacle_check(player,hazards,normal,settings,control)
        base["recovery_obstacle_check"]=recovery_check
        if not allowed:
            if parameters.get("allow_contact_mitigation") is True:
                mitigation=_contact_mitigation(obs,player,road,hazards,normal,settings,maximum_axis,
                                               base,"wall_recovery_blocked_by_obstacle")
                if mitigation is not None:return mitigation
            return _empty("向内恢复会加重可见障碍接触，未找到比中立更好的短时动作；中立不保证脱困",
                          {**base,"mode":"recovery_blocked","acceleration_allowed":False,
                           "route_status":"contact_not_collision_free"})
        base.update(mode="wall_recovery",target={"x":nearest,"depth":player.y},
                    risks=["观测时已贴墙/包络越界，当前恢复不保证脱离碰撞"],acceleration_allowed=False)
        return Decision(ControllerState(left_stick=StickVector(round(control*32767),0)),
                        "MINING：已贴墙，松开 ZR 并向可见通道内恢复",min(obs.confidence,player.confidence),base)
    near_wall=wall_distance<parameters.get("release_wall_distance",.025)
    near_hazard=nearest_hazard<parameters.get("release_hazard_depth",.18)
    if not allow_full_fast:
        fast,expanded,fast_status=None,0,"not_attempted_mixed_deadline_guard"
    elif near_wall or near_hazard:
        fast,expanded,fast_status=None,0,"not_attempted_near_risk"
    else:
        fast,expanded,fast_status=_search(road,player,hazards,boosted,settings,True)
    risks=[]
    sharp=False;authority=None;mixed_switch=None
    if fast:
        # Large early steering or large direction changes require more time
        # than an easy bend. These are declared heuristics until measured.
        clearance=(_early_route_clearance(fast,road,player,hazards,boosted,settings,3*step)
                   if parameters.get("allow_full_axis_with_clearance") is True and fast_status=="complete" else None)
        sharp,authority=_turn_authority(fast,boosted,maximum_axis,parameters,clearance=clearance,complete=fast_status=="complete")
    accelerate=bool(fast and fast_status=="complete" and not sharp and not near_wall and not near_hazard)
    base.update(fast_route_status=fast_status,turn_authority=authority,mixed_candidates=[])
    if not accelerate and not near_wall and not near_hazard:
        for seconds in mixed_prefixes:
            candidate,extra,candidate_status=_search(road,player,hazards,boosted,settings,True,seconds,normal)
            expanded+=extra
            candidate_sharp=True;candidate_authority=None
            if candidate is not None:
                clearance=(_early_route_clearance(candidate,road,player,hazards,boosted,settings,min(3*step,seconds))
                           if parameters.get("allow_full_axis_with_clearance") is True and candidate_status=="complete" else None)
                candidate_sharp,candidate_authority=_turn_authority(candidate,boosted,maximum_axis,parameters,seconds,
                    clearance=clearance,complete=candidate_status=="complete")
            base["mixed_candidates"].append({"zr_seconds":seconds,"status":candidate_status,"sharp":candidate_sharp,
                                              "turn_authority":candidate_authority})
            if candidate is not None and candidate_status=="complete" and not candidate_sharp:
                fast=candidate;fast_status=candidate_status;accelerate=True;mixed_switch=seconds
                authority=candidate_authority;sharp=False
                break
    normal_comparison=None;star_mode_selected=False
    base["star_mode_comparison"]={"enabled":parameters.get("compare_star_modes") is True,"attempted":False}
    compare_normal=bool(accelerate and parameters.get("compare_star_modes") is True and settings["stars"])
    if compare_normal and parameters.get("star_mode_upper_bound_pruning") is True:
        normal_prefix=_prediction_prefix(player,normal,settings)[-1]
        prefix_seconds=normal_prefix[2] if capture_clock is not None else 0.
        remaining_depth=max(0.,fast.path[-1][1]-normal_prefix[1])
        dynamic_model=capture_clock.get("forward_model") if capture_clock is not None else None
        normal_seconds=prefix_seconds+(dynamic_model.time_to_distance(capture_clock["prefix_end_forward_speed"],False,remaining_depth)
            if dynamic_model is not None else remaining_depth/normal.forward)
        extra_star_upper=len(settings["stars"])-len(fast.collected_predicted)
        extra_seconds=normal_seconds-fast.path[-1][2]
        gain_upper=extra_star_upper-extra_seconds
        if extra_star_upper<=0 or gain_upper<=star_mode_margin:
            compare_normal=False
            base["star_mode_comparison"].update(selected="fast",skipped="net_gain_upper_bound_insufficient",
                extra_stars_upper_bound=extra_star_upper,extra_travel_seconds_if_complete=extra_seconds,
                net_time_gain_upper_bound_seconds=gain_upper,gain_margin_seconds=star_mode_margin,
                bound_scope=("all_visible_stars_and_same_endpoint_dynamic_model_time_not_actual_outcome" if dynamic_model is not None
                             else "all_visible_stars_and_same_endpoint_constant_model_time_not_actual_outcome"))
    if compare_normal:
        alternative,extra,alternative_status=_search(road,player,hazards,normal,settings)
        expanded+=extra;normal_comparison=(alternative,alternative_status)
        same_depth=bool(alternative is not None and alternative_status=="complete"
                        and fast_status=="complete" and abs(alternative.path[-1][1]-fast.path[-1][1])<=1e-6)
        comparison={"enabled":True,"attempted":True,"normal_status":alternative_status,
            "same_complete_depth":same_depth,"gain_margin_seconds":star_mode_margin,
            "seconds_per_star":1.0,"star_collection_confirmed":False,"selected":"fast"}
        if same_depth:
            extra_stars=len(alternative.collected_predicted)-len(fast.collected_predicted)
            extra_seconds=alternative.path[-1][2]-fast.path[-1][2]
            gain=extra_stars-extra_seconds
            star_mode_selected=extra_stars>0 and gain>star_mode_margin
            comparison.update(fast_stars_predicted=len(fast.collected_predicted),normal_stars_predicted=len(alternative.collected_predicted),
                extra_stars_predicted=extra_stars,extra_travel_seconds=extra_seconds,net_time_gain_seconds=gain,
                selected="normal" if star_mode_selected else "fast")
            if star_mode_selected:
                accelerate=False;mixed_switch=None
                risks.append("普通速度同终点路线的预计额外加时大于额外耗时，松开 ZR 争取加时星")
        base["star_mode_comparison"]=comparison
    if accelerate:
        plan,status,motion=fast,fast_status,boosted
    else:
        if normal_comparison is not None:plan,status=normal_comparison
        else:
            plan,extra,status=_search(road,player,hazards,normal,settings)
            expanded+=extra
        motion=normal
        if parameters.get("circle_guided_search") is True and plan is not None and status=="partial":
            alternative,extra,alternative_status,guided=_guided_normal_alternative(road,player,hazards,normal,settings,plan)
            expanded+=extra;base["guided_circle_search"].update(guided)
            if alternative is not None and alternative_status=="complete":plan,status=alternative,alternative_status
        base["mixed_guard"]["normal_replanned"]=True
        if sharp:risks.append("前方转向需求较大，松开 ZR")
        if near_wall:risks.append("玩家靠近墙包络，松开 ZR")
        if near_hazard:risks.append("近处存在障碍，松开 ZR")
        if plan is None and fast:
            # A safe fast path does not imply the slower-time dynamics are
            # safe. Do not execute its initial control under a different model.
            return _empty("松开加速模型中无可靠下一步，等待重新观测",{**base,"expanded_nodes":expanded})
    if plan is None:
        if parameters.get("allow_contact_mitigation") is True:
            mitigation=_contact_mitigation(obs,player,road,hazards,normal,settings,maximum_axis,base,status)
            if mitigation is not None:return mitigation
        return _empty("前方没有可确认的避障路线，松开输入",{**base,"expanded_nodes":expanded,"search_failure":status})
    if status!="complete":
        risks.append("前视路线只覆盖部分时域，下一帧必须重算")
        accelerate=False
    first=plan.path[0]
    control=first[3]
    control=max(-maximum_axis,min(maximum_axis,control))
    if capture_clock is not None:
        path=[{"x":x,"depth":y,"seconds":t,"steering":None,"accelerate":None,
               "phase":"already_reported_or_unknown_input_prefix"} for x,y,t,_ in _prediction_prefix(player,motion,settings)]
    else:path=[{"x":player.x,"depth":player.y,"seconds":0,"steering":control,"accelerate":accelerate}]
    prefix=_prediction_prefix(player,motion,settings)[-1]
    action_start=prefix[2] if capture_clock is not None else 0.
    path.extend({"x":x,"depth":y,"seconds":t,"steering":u,
                 "accelerate":accelerate and (mixed_switch is None or t<=action_start+mixed_switch+1e-8)}
                for x,y,t,u,_ in plan.path)
    before=(prefix[0],prefix[1],prefix[2] if capture_clock is not None else 0.0)
    predicted=[];seen=set()
    for x,y,t,_,_ in plan.path:
        for key,star in settings["stars"]:
            if key not in plan.collected_predicted or key in seen:continue
            fraction=contact_fraction(before[0],before[1],x,y,star,*settings["star_pickup_radii"])
            if fraction is not None:
                seen.add(key);predicted.append({"key":key,"x":star.x,"depth":star.y,
                    "seconds":before[2]+fraction*(t-before[2]),"confirmed":False})
        before=(x,y,t)
    base.update(mode=("accelerated_then_normal" if mixed_switch is not None else "accelerated") if accelerate else "careful_turn",target={"x":first[0],"depth":first[1]},
                path=path,risks=risks,acceleration_allowed=accelerate,expanded_nodes=expanded,
                planned_seconds=plan.path[-1][2],route_status=status,predicted_lateral_velocity=first[4],
                planning_ms=(perf_counter()-started)*1000,
                model={"forward_speed":motion.forward,"input_gain_per_second":motion.input_gain,
                       "drag_per_second":motion.drag,"maximum_lateral_speed":motion.maximum_speed,
                       "neutral_velocity":motion.neutral_velocity})
    base.update(turn_authority=authority,mixed_zr_prefix_seconds=mixed_switch,
                future_mode_schedule=([{ "mode":"zr","start_seconds":action_start,"end_seconds":action_start+mixed_switch},
                                       {"mode":"normal","start_seconds":action_start+mixed_switch,"end_seconds":plan.path[-1][2]}]
                                      if mixed_switch is not None else []),
                mode_switch_delay_added_seconds=0.,
                mixed_forward_response_assumption=("integrated_zr_acceleration_then_normal_exponential_response"
                    if capture_clock is not None and capture_clock.get("forward_model") is not None
                    else "piecewise_constant_mode_speed_not_measured_transition"))
    base["mixed_guard"]["full_fast_rearm_verified"]=bool(accelerate and mixed_switch is None
        and status=="complete" and not sharp and not near_wall and not near_hazard)
    base.update(collected_predicted=predicted,star_reward_total_predicted=star_reward*len(predicted),
                star_target=predicted[0] if predicted else None)
    buttons=frozenset({Button.ZR}) if accelerate else frozenset()
    reason="可见通道内保持 ZR，提前跟随路线" if accelerate else "松开 ZR，按可见弯道/障碍提前转向并抑制横向惯性"
    if mixed_switch is not None:reason="短暂保持 ZR，完整前视路线预计随后松开加速；下一帧重算"
    if star_mode_selected:reason="松开 ZR，预计额外加时星收益超过同距离增加的行进时间"
    if predicted:reason+="；路线预计接触加时星，尚未确认拾取"
    return Decision(ControllerState(buttons,StickVector(round(control*32767),0)),"MINING："+reason,
                    min(obs.confidence,player.confidence),base)
