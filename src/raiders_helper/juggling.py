"""Fast, explainable two-player interception suggestions for JUGGLING.

The policy does not identify Salmonid species. It consumes tracked collision
boxes and velocities; an optional measured gravity estimate improves airborne
prediction. Gains, deadbands and the search horizon are controller heuristics,
not claims about the game's physics. No hardware I/O occurs here.
"""
from dataclasses import dataclass, replace
import math

from .control.buttons import Button, ControllerState, StickVector
from .models import Decision, Entity, Game, Observation


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _valid_entity(entity):
    return all(_finite(v) for v in (entity.x, entity.y, entity.w, entity.h,
                                    entity.vx, entity.vy, entity.confidence)) and entity.w > 0 and entity.h > 0


def reflected_x(x, left, right):
    """Reflect a point at outer court walls, supporting multiple crossings."""
    if not all(_finite(v) for v in (x, left, right)) or right <= left:
        raise ValueError("球场边界无效")
    span = right-left
    phase = (x-left) % (2*span)
    return left + phase if phase <= span else right - (phase-span)


def time_to_contact(ball, actor, gravity=0.0):
    """First descending contact of ball bottom with actor top, in seconds.

    gravity==0 is a local constant-velocity approximation. A rising trajectory
    without a measured downward acceleration has no predicted catch time.
    """
    target_y = actor.y - actor.h/2 - ball.h/2
    if ball.y >= target_y:
        # Already crossing the receiving body: urgent, not a future floor hit.
        if ball.y-ball.h/2 <= actor.y+actor.h/2 and ball.vy >= 0:
            return 0.0
        return None
    distance = target_y-ball.y
    if abs(gravity) < 1e-9:
        return distance/ball.vy if ball.vy > 1e-6 else None
    discriminant = ball.vy*ball.vy + 2*gravity*distance
    if discriminant < 0:
        return None
    root = math.sqrt(discriminant)
    roots = ((-ball.vy-root)/gravity, (-ball.vy+root)/gravity)
    valid = [t for t in roots if t >= 0 and ball.vy+gravity*t > 0]
    return min(valid) if valid else None


@dataclass(frozen=True)
class _Catch:
    ball: Entity
    side: str
    target_x: float
    seconds: float
    available: float
    delta: float
    urgency: float
    reachable: bool | None
    axis: int
    slide_recommended: bool
    prediction_source: str = 'local_constant_velocity'


def _speed(parameters, side):
    value = parameters.get("lateral_speed_"+side, parameters.get("lateral_speed"))
    return float(value) if _finite(value) and value > 0 else None


def _catch_dict(catch):
    return {
        "track_id": catch.ball.track_id, "side": catch.side,
        "ball_x": catch.ball.x, "ball_y": catch.ball.y,
        "target_x": catch.target_x, "time_to_contact": catch.seconds,
        "available_seconds": catch.available, "urgency": catch.urgency,
        "reachable": catch.reachable, "required_axis": catch.axis,
        "slide_recommended": catch.slide_recommended,
        "prediction_source": catch.prediction_source,
        "delta_at_input_estimate":catch.delta,
    }


def decide_juggling(obs: Observation, parameters: dict | None = None) -> Decision:
    """O(N log N) trajectory preparation and at most 16 two-actor assignments.

    Parameters: split_x, arena_left/right, prediction_horizon_seconds,
    deadband, steering_gain; optional measured lateral_speed[_left/_right].
    Slide additionally needs calibrated=true, slide.enabled/calibrated=true,
    positive slide.speed/minimum_distance/maximum_time_to_contact, a measured
    normal lateral speed and the corresponding metrics.*_slide_ready==1.
    """
    parameters = parameters or {}
    idle=parameters.get('idle_centering',{})
    idle_enabled=isinstance(idle,dict) and idle.get('enabled') is True
    empty = {"policy": "juggling_interception", "targets": {"left": None, "right": None},
             "intercepts": [], "urgency": {"left": 0.0, "right": 0.0},
             "fever_active":obs.metrics.get("fever_active") is True,
             "score_pickup_candidates":len(obs.of_kind("score_pickup")),
             "pickup_priority":"catch_first","star_detours_enabled":False}
    if obs.game != Game.JUGGLING or obs.phase != "playing" or obs.confidence < 0.8:
        return Decision(reason="JUGGLING：等待可靠的运行画面", diagnostics=empty)
    lives = obs.metrics.get("lives_remaining")
    if lives is not None and (not _finite(lives) or lives <= 0):
        return Decision(reason="JUGGLING：生命已耗尽或读数无效", diagnostics=empty)
    actors = {"left": obs.of_kind("player"), "right": obs.of_kind("partner")}
    if any(len(group) != 1 for group in actors.values()):
        return Decision(reason="JUGGLING：两侧角色缺失或不唯一", diagnostics=empty)
    actors = {side: group[0] for side, group in actors.items()}
    if any(not _valid_entity(actor) or actor.confidence < 0.8 for actor in actors.values()):
        return Decision(reason="JUGGLING：角色观测不可靠", diagnostics=empty)

    split = parameters.get("split_x", 0.5)
    left, right = parameters.get("arena_left", 0.0), parameters.get("arena_right", 1.0)
    horizon = parameters.get("prediction_horizon_seconds", 1.5)
    fallback_horizon=parameters.get('fallback_prediction_horizon_seconds',horizon)
    deadband = parameters.get("deadband", 0.01)
    gain = parameters.get("steering_gain", 6.0)
    latency = obs.metrics.get("latency_seconds", parameters.get("latency_seconds", 0.0))
    age = obs.metrics.get("observation_age_seconds", 0.0)
    gravity = obs.metrics.get("ball_gravity_estimate", parameters.get("gravity_estimate", 0.0))
    velocity_lag=parameters.get('velocity_smoothing_lag_seconds',0.)
    values = (split, left, right, horizon, fallback_horizon, deadband, gain, latency, age, gravity,velocity_lag)
    if (not all(_finite(v) for v in values) or not left < split < right
            or horizon <= 0 or fallback_horizon<=0 or not 0<=velocity_lag<=.15
            or deadband < 0 or gain <= 0 or latency < 0 or age < 0 or gravity < 0):
        return Decision(reason="JUGGLING：预测参数或时延读数无效", diagnostics=empty)
    for key in ("lateral_speed", "lateral_speed_left", "lateral_speed_right"):
        if key in parameters and parameters[key] is not None and (not _finite(parameters[key]) or parameters[key] <= 0):
            return Decision(reason="JUGGLING：角色移动速率配置无效", diagnostics=empty)
    if any(not _finite(parameters.get('deadband_'+side,deadband)) or not 0<=parameters.get('deadband_'+side,deadband)<=.04 for side in actors):
        return Decision(reason='JUGGLING：接球稳定范围无效',diagnostics=empty)
    latency += age
    projection=obs.metrics.get('juggling_actor_input_projection',{})
    projection_valid=(parameters.get('input_response_model',{}).get('enabled') is True
        and projection.get('accepted') is True
        and projection.get('source')=='backend_after_history_with_held_last_input'
        and _finite(projection.get('observation_timestamp'))
        and abs(projection['observation_timestamp']-obs.timestamp)<1e-6
        and _finite(projection.get('seconds_from_observation'))
        and .04<=projection['seconds_from_observation']<=.35
        and all(_finite(projection.get('positions',{}).get(side)) for side in actors))
    if projection_valid:latency=projection['seconds_from_observation']
    if not actors["left"].x < split < actors["right"].x:
        return Decision(reason="JUGGLING：左右角色身份与半场不一致", diagnostics=empty)

    all_balls = obs.of_kind("ball")
    balls = [ball for ball in all_balls if _valid_entity(ball) and ball.confidence >= 0.8 and ball.samples >= 2]
    ids = [ball.track_id for ball in balls if ball.track_id >= 0]
    if len(set(ids)) != len(ids) or (len(balls) > 1 and any(ball.track_id < 0 for ball in balls)):
        return Decision(reason="JUGGLING：多球身份不明确，等待跟踪恢复", diagnostics=empty)
    if not balls and not idle_enabled:
        return Decision(reason="JUGGLING：等待连续的鲑鱼轨迹", diagnostics=empty)

    candidates = {"left": [], "right": []}
    intercepts = []
    late_count = 0
    for ball in balls:
        model=obs.metrics.get('ball_flight_models',{}).get(str(ball.track_id),{})
        fitted=None
        if (parameters.get('use_track_flight_models') is True and model.get('accepted') is True
                and all(_finite(model.get(k)) for k in ('timestamp','x','y','vx','vy','gravity'))
                and abs(model['timestamp']-obs.timestamp)<1e-6
                and abs(model['x']-ball.x)*640<=4 and abs(model['y']-ball.y)*480<=4
                and .05<=model['gravity']<=2.5 and abs(model['vx'])<=1.5 and abs(model['vy'])<=3):
            fitted=replace(ball,x=model['x'],y=model['y'],vx=model['vx'],vy=model['vy'])
        for side, actor in actors.items():
            lag=velocity_lag
            window=obs.metrics.get('ball_velocity_windows',{}).get(str(ball.track_id))
            if parameters.get('use_actual_velocity_window') is True and _finite(window) and 0<=window<=.6:
                lag=window/2
            forecast=replace(ball,vy=ball.vy+gravity*lag) if gravity and lag else ball
            used_gravity=gravity;prediction_source='configured_gravity_prior' if gravity else 'local_constant_velocity'
            if fitted is not None:
                fitted_seconds=time_to_contact(fitted,actor,model['gravity'])
                # Longer extrapolation has not been independently verified.
                if fitted_seconds is not None and fitted_seconds<=.65:
                    forecast=fitted;used_gravity=model['gravity'];prediction_source='current_track_quadratic_window'
            seconds = time_to_contact(forecast, actor, used_gravity)
            if seconds is None or seconds > horizon:
                continue
            if prediction_source!='current_track_quadratic_window' and seconds>fallback_horizon:
                continue
            low, high = left+ball.w/2, right-ball.w/2
            if high <= low:
                continue
            predicted = reflected_x(forecast.x+forecast.vx*seconds, low, high)
            # Assign by the predicted receiving half, not the ball's current x.
            if (predicted < split) != (side == "left"):
                continue
            actor_low = left+actor.w/2 if side == "left" else split+actor.w/2
            actor_high = split-actor.w/2 if side == "left" else right-actor.w/2
            if actor_low > actor_high:
                continue
            target = max(actor_low, min(actor_high, predicted))
            actor_estimate=projection['positions'][side] if projection_valid else actor.x+actor.vx*latency
            actor_at_input = max(actor_low, min(actor_high, actor_estimate))
            delta = target-actor_at_input
            contact_margin = max(0.0, (actor.w+ball.w)/2-deadband)
            available = max(0.0, seconds-latency)
            if seconds < latency and abs(delta) > contact_margin:
                late_count += 1
                intercepts.append({"track_id": ball.track_id, "side": side, "target_x": target,
                                   "time_to_contact": seconds, "eligible": False, "reason": "控制到达前已错过接触"})
                continue
            speed = _speed(parameters, side)
            reachable = None if speed is None else abs(delta) <= speed*available+contact_margin
            if abs(delta) <= parameters.get('deadband_'+side,deadband):
                axis = 0
            elif speed is not None:
                axis = round(max(-1.0, min(1.0, delta/(speed*max(available, 0.02))))*32767)
            else:
                axis = round(max(-1.0, min(1.0, delta*gain))*32767)
            catch = _Catch(ball, side, target, seconds, available, delta,
                           1.0/max(seconds, 0.02), reachable, axis, reachable is False,prediction_source)
            candidates[side].append(catch)
            intercepts.append({**_catch_dict(catch), "eligible": True})

    # Only three most urgent options per actor enter the joint assignment.
    # Keeping idle is necessary if both receiving predictions refer to one ball.
    choices = {side: [None]+sorted(options, key=lambda item: (item.seconds, abs(item.delta)))[:3]
               for side, options in candidates.items()}
    best, best_value = (None, None), (-1.0, -math.inf)
    for a in choices["left"]:
        for b in choices["right"]:
            if a is not None and b is not None and a.ball is b.ball:
                continue
            selected = [item for item in (a, b) if item is not None]
            value = (sum(item.urgency for item in selected), -sum(abs(item.delta) for item in selected))
            if value > best_value:
                best, best_value = (a, b), value
    selected = dict(zip(("left", "right"), best))
    diagnostics = {
        **empty, "model": "constant_acceleration" if gravity else "local_constant_velocity",
        "gravity_estimate": gravity, "latency_seconds": latency,
        "targets": {side: _catch_dict(catch) if catch else None for side, catch in selected.items()},
        "urgency": {side: catch.urgency if catch else 0.0 for side, catch in selected.items()},
        "intercepts": intercepts, "ignored_ball_count": len(all_balls)-len(balls), "late_ball_count": late_count,
        "parameter_scope": "控制增益、死区和预测时域是可调启发式；运动速率需测量",
        "track_flight_model_targets":sum(c.prediction_source=='current_track_quadratic_window' for c in selected.values() if c),
        "actor_prefix_source":"actual_submitted_input_model" if projection_valid else "smoothed_observed_velocity",
        "actor_at_input_estimate":projection['positions'] if projection_valid else None,
    }
    if not any(selected.values()) and not idle_enabled:
        return Decision(reason="JUGGLING：当前没有可用的未来接触，保持观察", diagnostics=diagnostics)

    buttons = set()
    slide = parameters.get("slide", {})
    slide_valid = (isinstance(slide, dict) and parameters.get("calibrated") is True
                   and slide.get("calibrated") is True and slide.get("enabled") is True
                   and all(_finite(slide.get(key)) and slide[key] > 0
                           for key in ("speed", "minimum_distance", "maximum_time_to_contact")))
    for side, catch in selected.items():
        speed = _speed(parameters, side)
        if (catch is not None and slide_valid and speed is not None and slide["speed"] > speed
                and obs.metrics.get(side+"_slide_ready") == 1
                and catch.slide_recommended and abs(catch.delta) >= slide["minimum_distance"]
                and 0 < catch.available <= slide["maximum_time_to_contact"]
                and abs(catch.delta) <= slide["speed"]*catch.available+(actors[side].w+catch.ball.w)/2):
            buttons.add(Button.ZL if side == "left" else Button.ZR)
    axes={side:catch.axis if catch else 0 for side,catch in selected.items()}
    idle_positions={}
    if idle_enabled:
        idle_gain=idle.get('gain',6.);idle_deadband=idle.get('deadband',.012);idle_max=idle.get('maximum_axis',.6)
        if (all(_finite(v) for v in (idle_gain,idle_deadband,idle_max))
                and 0<idle_gain<=12 and 0<=idle_deadband<=.04 and .05<=idle_max<=1):
            for side,actor in actors.items():
                if selected[side] is not None:continue
                home=(left+split)/2 if side=='left' else (split+right)/2
                future=projection['positions'][side] if projection_valid else actor.x+actor.vx*latency
                delta=home-future
                axes[side]=0 if abs(delta)<=idle_deadband else round(max(-idle_max,min(idle_max,delta*idle_gain))*32767)
                idle_positions[side]=home
    diagnostics['idle_middle_positions']=idle_positions
    state = ControllerState(frozenset(buttons),StickVector(axes['left'],0),StickVector(axes['right'],0))
    parts = []
    for side, catch in selected.items():
        if catch:
            label = "左侧" if side == "left" else "右侧"
            direction = "对准" if catch.axis == 0 else ("向右" if catch.axis > 0 else "向左")
            parts.append(f"{label}{direction}接 #{catch.ball.track_id}（{catch.seconds:.2f}秒）")
        elif side in idle_positions:
            parts.append(('左侧' if side=='left' else '右侧')+'回中等待')
    confidence = min(obs.confidence, *(actor.confidence for actor in actors.values()),
                     *(catch.ball.confidence for catch in selected.values() if catch))
    reason="JUGGLING："+"；".join(parts)
    if obs.metrics.get("fever_active") is True:reason+="；Fever 保球优先"
    return Decision(state, reason, confidence, diagnostics)
