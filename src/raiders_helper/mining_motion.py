"""Observed MINING motion conditioned on ZR, never a controller or game spec.

``fit_mining_motion`` accepts the live action/observation rows used by
motion_calibration. Horizontal velocity is estimated from consecutive player
positions. World/scroll speed MUST be independently observed: a nearly fixed
player.y, SCORE, or a planner's assumed speed is not such a measurement.

Normal clear motion must be labelled explicitly. Collision/rock/buff events and
their neighbourhood are excluded rather than absorbed into ordinary drag.
The finite forward-speed candidates are compared on held-out rounds/blocks;
even a constant-acceleration winner never establishes unbounded future speed.
No output changes a profile's calibrated flag or applies a gameplay action.
"""
from bisect import bisect_right
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
import math

import numpy as np


_EVENTS = frozenset({"collision", "wall_collision", "rock_collision", "rock_break",
                     "egg_pickup", "buff_active", "egg_buff", "buff_changed"})
_UNITS = "viewport_heights_per_second"


def _map(value):
    return asdict(value) if is_dataclass(value) else (value if isinstance(value, dict) else {})


def _number(value):
    return float(value) if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) else None


def _state(value):
    state = _map(value)
    buttons = state.get("buttons", [])
    if not isinstance(buttons, (list, tuple, set, frozenset)):
        return None
    buttons = {getattr(button, "value", button) for button in buttons}
    if not buttons <= {"ZR"}:
        return None
    left, right = _map(state.get("left_stick", {})), _map(state.get("right_stick", {}))
    values = [_number(left.get("x", 0)), _number(left.get("y", 0)),
              _number(right.get("x", 0)), _number(right.get("y", 0))]
    if any(value is None or abs(value) > 32767 for value in values) or any(values[1:]):
        return None
    return values[0]/32767, "ZR" in buttons


@dataclass(frozen=True)
class _Point:
    t: float
    round_id: int
    sequence: int
    track: int
    x: float
    forward: float | None


def _normalise(rows, config):
    rejects, events, points, commands = Counter(), {}, [], {}
    confidence = config.get("minimum_confidence", .8)
    for row in rows:
        row = _map(row)
        observation = _map(row.get("observation"))
        game = getattr(observation.get("game"), "value", observation.get("game"))
        sent, observed = _number(row.get("timestamp")), _number(observation.get("timestamp"))
        round_id = row.get("round_id")
        if row.get("live") is not True or row.get("executed") is not True:
            rejects["not_live_executed"] += 1; continue
        if game != "mining" or sent is None or observed is None or type(round_id) is not int or round_id < 1:
            rejects["invalid_game_time_or_round"] += 1; continue
        state = _state(row.get("state"))
        if state is None:
            rejects["unsupported_input"] += 1; continue
        commands.setdefault(round_id, []).append((sent, *state))
        metrics = _map(observation.get("metrics"))
        labelled = row.get("events", metrics.get("motion_events", []))
        labelled = set(labelled) if isinstance(labelled, (list, tuple, set, frozenset)) else set()
        labelled.update(key for key in _EVENTS if metrics.get(key) is True)
        raw_entities = observation.get("entities", [])
        if not isinstance(raw_entities, (list, tuple)):
            rejects["invalid_entities"] += 1; continue
        entities = [_map(entity) for entity in raw_entities]
        if any(entity.get("kind") == "egg_buff" for entity in entities):
            labelled.add("egg_buff")
        bad = labelled & _EVENTS
        if bad:
            events.setdefault(round_id, []).append(observed)
            for name in bad: rejects["event:"+name] += 1
            continue
        normal = metrics.get("motion_regime") == "normal_clear"
        if not normal:
            # An empty detector result does not prove absence of collision or buff.
            rejects["normal_regime_not_labelled"] += 1; continue
        score = _number(observation.get("confidence"))
        if observation.get("phase") != "playing" or score is None or score < confidence:
            rejects["unreliable_phase"] += 1; continue
        players = [entity for entity in entities if entity.get("kind") == "player"]
        if len(players) != 1:
            rejects["player_missing_or_ambiguous"] += 1; continue
        player = players[0]
        x, entity_score = _number(player.get("x")), _number(player.get("confidence"))
        track, sequence = player.get("track_id"), observation.get("sequence")
        if (x is None or not 0 <= x <= 1 or entity_score is None or entity_score < confidence
                or type(track) is not int or track < 0 or type(sequence) is not int
                or _number(player.get("samples")) is None
                or player["samples"] < config.get("minimum_track_samples", 3)):
            rejects["unreliable_player_track"] += 1; continue
        forward = _number(metrics.get("forward_speed"))
        speed_score = _number(metrics.get("forward_speed_confidence"))
        source = metrics.get("forward_speed_source")
        if (forward is None or forward < 0 or speed_score is None or speed_score < confidence
                or metrics.get("forward_speed_observed") is not True
                or metrics.get("forward_speed_units") != _UNITS
                or source not in config.get("forward_speed_sources", ["background_flow", "landmark_tracking", "external_measurement"])):
            forward = None
            rejects["forward_speed_not_observed"] += 1
        points.append(_Point(observed, round_id, sequence, track, x, forward))
    guard = config.get("event_guard_seconds", .25)
    clean = [point for point in points if not any(abs(point.t-event) <= guard for event in events.get(point.round_id, []))]
    rejects["event_neighbourhood"] += len(points)-len(clean)
    for values in commands.values(): values.sort()
    return sorted(clean, key=lambda point:(point.round_id, point.t)), commands, rejects


def _input_at(commands, round_id, timestamp, max_age):
    values = commands.get(round_id, [])
    index = bisect_right(values, (timestamp, math.inf, True))-1
    if index < 0 or not 0 <= timestamp-values[index][0] <= max_age:
        return None
    return values[index][1:]


def _continuous(a, b, config):
    return (a.round_id == b.round_id and a.track == b.track
            and 0 < b.t-a.t <= config.get("max_gap_seconds", .05)
            and 0 < b.sequence-a.sequence <= config.get("max_sequence_gap", 1))


def _samples(points, commands, delay, axis, zr, config):
    result = []
    if axis == "horizontal":
        for a,b,c in zip(points, points[1:], points[2:]):
            if not (_continuous(a,b,config) and _continuous(b,c,config)): continue
            dt0, dt1 = b.t-a.t, c.t-b.t
            start, end = (a.t+b.t)/2, (b.t+c.t)/2
            first = _input_at(commands,a.round_id,start-delay,config.get("max_input_age_seconds",.05))
            last = _input_at(commands,a.round_id,end-delay,config.get("max_input_age_seconds",.05))
            if first is None or last is None or first[1] != zr or last[1] != zr: continue
            result.append({"round":a.round_id,"t":start,"end":end,"dt":end-start,
                           "v":(b.x-a.x)/dt0,"next":(c.x-b.x)/dt1,"u":first[0]})
    else:
        for a,b in zip(points,points[1:]):
            if not _continuous(a,b,config) or a.forward is None or b.forward is None: continue
            first = _input_at(commands,a.round_id,a.t-delay,config.get("max_input_age_seconds",.05))
            last = _input_at(commands,a.round_id,b.t-delay,config.get("max_input_age_seconds",.05))
            if first is None or last is None or first[1] != zr or last[1] != zr: continue
            result.append({"round":a.round_id,"t":a.t,"end":b.t,"dt":b.t-a.t,
                           "v":a.forward,"next":b.forward,"u":first[0]})
    return result


def _split(samples, config):
    rounds = sorted({sample["round"] for sample in samples})
    if len(rounds) >= 2:
        return ([i for i,sample in enumerate(samples) if sample["round"] != rounds[-1]],
                [i for i,sample in enumerate(samples) if sample["round"] == rounds[-1]], "whole_last_round")
    if not samples: return [],[],"none"
    cutoff = samples[0]["t"]+.75*(samples[-1]["end"]-samples[0]["t"])
    gap = max(config.get("delay_grid_seconds", [0]))+config.get("max_gap_seconds", .05)*2
    return ([i for i,sample in enumerate(samples) if sample["end"] <= cutoff-gap],
            [i for i,sample in enumerate(samples) if sample["t"] >= cutoff+gap], "isolated_time_blocks")


def _rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual)-np.asarray(predicted))**2)))


def _quality(samples, train, test, config, axis):
    rounds = sorted({sample["round"] for sample in samples})
    return {"sample_coverage":len(samples) >= config.get("min_samples",80),
            "round_coverage":len(rounds) >= config.get("min_rounds",3),
            "observed_span":sum(sample["dt"] for sample in samples) >= config.get("min_span_seconds",5),
            "independent_validation":len(train) >= 20 and len(test) >= 20,
            "motion_variation": bool(samples) and bool(np.ptp([sample["v"] for sample in samples]) >= config.get("min_velocity_span",.03)),
            "input_coverage":axis != "horizontal" or (bool(samples) and bool(np.ptp([sample["u"] for sample in samples]) >= config.get("min_axis_coverage",.25)))}


def _forward_rollout(samples, ids, coefficients):
    errors, previous, predicted = [], None, 0.0
    for index in ids:
        sample=samples[index]
        if previous is None or previous["round"] != sample["round"] or abs(previous["end"]-sample["t"]) > 1e-6:
            predicted=sample["v"]
        predicted += sample["dt"]*(coefficients[0]*predicted+coefficients[1])
        errors.append(predicted-sample["next"])
        previous=sample
    return _rmse(errors, np.zeros(len(errors)))


def _fit(points, commands, zr, axis, config):
    empty={"passed":False,"axis":axis,"zr_held":zr,"reason":"可用连续实机观测不足","candidates":[]}
    candidates=[]
    for delay in config.get("delay_grid_seconds", [0,1/60,2/60,3/60,4/60,.1,.15,.2]):
        samples=_samples(points,commands,delay,axis,zr,config)
        train,test,split=_split(samples,config)
        if len(train)<20 or len(test)<20: continue
        dt=np.asarray([sample["dt"] for sample in samples]);v=np.asarray([sample["v"] for sample in samples])
        y=np.asarray([sample["next"] for sample in samples]);u=np.asarray([sample["u"] for sample in samples])
        quality=_quality(samples,train,test,config,axis)
        designs = [("horizontal_response",np.column_stack((dt*v,dt*u,dt)))] if axis=="horizontal" else [
            ("constant_acceleration",dt[:,None]),("terminal_speed_candidate",np.column_stack((dt*v,dt)))]
        for name,matrix in designs:
            if np.linalg.matrix_rank(matrix[train]) < matrix.shape[1]:continue
            coefficients=np.linalg.lstsq(matrix[train],(y-v)[train],rcond=None)[0]
            predictions=v+matrix@coefficients
            parameters={"velocity_feedback_per_second":float(coefficients[0]),"input_gain_per_second":float(coefficients[1]),
                        "bias_widths_per_second_squared":float(coefficients[2])} if axis=="horizontal" else (
                {"acceleration_heights_per_second_squared":float(coefficients[0])} if name=="constant_acceleration" else
                {"velocity_feedback_per_second":float(coefficients[0]),"drive_heights_per_second_squared":float(coefficients[1])})
            if name=="terminal_speed_candidate":
                if coefficients[0]>=-1e-6 or coefficients[1]<=0:continue
                parameters.update(terminal_speed_heights_per_second=float(-coefficients[1]/coefficients[0]),
                                  time_constant_seconds=float(-1/coefficients[0]))
            holdout=_rmse(y[test],predictions[test])
            rollout=None if axis=="horizontal" else _forward_rollout(samples,test,
                (0,coefficients[0]) if name=="constant_acceleration" else coefficients)
            checks=dict(quality)
            threshold=_number(config.get("max_holdout_rmse"))
            checks["error_threshold_configured"]=threshold is not None and threshold>0
            checks["holdout_error"]=bool(checks["error_threshold_configured"] and holdout<=threshold)
            if axis=="forward":
                threshold=_number(config.get("max_forward_rollout_rmse"))
                checks["rollout_threshold_configured"]=threshold is not None and threshold>0
                checks["rollout_error"]=bool(checks["rollout_threshold_configured"] and rollout<=threshold)
            candidate={"name":name,"delay_seconds":float(delay),"parameters":parameters,
                "train_rmse":_rmse(y[train],predictions[train]),"holdout_rmse":holdout,
                "holdout_rollout_rmse":rollout,"sample_count":len(samples),"train_count":len(train),
                "holdout_count":len(test),"round_ids":sorted({sample["round"] for sample in samples}),
                "split":split,"checks":checks,"passed":bool(all(checks.values())),
                "observed_velocity_range":[float(v.min()),float(v.max())]}
            candidates.append(candidate)
    if not candidates:return empty
    # Delay selection uses TRAINING data only, separately for each finite model.
    best_by_model={}
    for candidate in candidates:
        old=best_by_model.get(candidate["name"])
        if old is None or candidate["train_rmse"]<old["train_rmse"]:
            best_by_model[candidate["name"]]=candidate
    finalists=list(best_by_model.values())
    if axis=="horizontal":
        selected=finalists[0]
        interpretation="按ZR状态分组的局部横向速度响应，不是完整转弯/碰撞模型"
    else:
        constant=best_by_model.get("constant_acceleration")
        terminal=best_by_model.get("terminal_speed_candidate")
        margin=config.get("model_selection_margin",.15)
        if terminal and (constant is None or terminal["holdout_rollout_rmse"]<(1-margin)*constant["holdout_rollout_rmse"]):
            selected=terminal
            interpretation="留出片段更支持速度收敛候选；只在所测速度范围内成立"
        else:
            selected=constant or terminal
            interpretation="所测片段用恒加速度近似即可；不能据此外推速度无限增长"
    return {"passed":selected["passed"],"axis":axis,"zr_held":zr,"selected":selected["name"],
            "selected_model":selected,"candidates":finalists,"interpretation":interpretation,
            "reason":"所声明的局部模型通过样本与留出误差门槛" if selected["passed"] else "观测覆盖或留出误差未达到显式门槛"}


def fit_mining_motion(rows, config=None):
    """Fit normal clear MINING observations without assuming forward speed.

    Required row fields match motion_calibration. Observation.metrics must
    explicitly label ``motion_regime: normal_clear``. For forward fitting add:
    ``forward_speed``, ``forward_speed_observed: true``, confidence >= .8,
    units ``viewport_heights_per_second``, and source ``background_flow``,
    ``landmark_tracking`` or ``external_measurement``.

    ``motion_events`` or boolean collision/rock/buff metrics mark exclusions.
    Error thresholds are deliberately absent by default, so no model can pass
    until the experiment declares its tolerances. ``calibrated`` is always false.
    """
    config=config or {}
    delays=config.get("delay_grid_seconds",[0,1/60,2/60,3/60,4/60,.1,.15,.2])
    if (not isinstance(delays,(list,tuple)) or not delays
            or any(_number(delay) is None or delay<0 or delay>1 for delay in delays)):
        raise ValueError("delay_grid_seconds必须为0到1秒之间的非空候选列表")
    config={**config,"delay_grid_seconds":delays}
    rows=list(rows)
    points,commands,rejects=_normalise(rows,config)
    horizontal={name:_fit(points,commands,zr,"horizontal",config) for name,zr in (("zr_off",False),("zr_on",True))}
    forward={name:_fit(points,commands,zr,"forward",config) for name,zr in (("zr_off",False),("zr_on",True))}
    forward_observed=any(point.forward is not None for point in points)
    unobserved=["boost_dynamics","collision_recovery","rock_break_response","full_curvature_model"]
    if not forward_observed:unobserved.append("world_forward_speed_and_acceleration")
    models={f"{axis}.{regime}":report["selected_model"]["parameters"] for axis,groups in
            (("horizontal",horizontal),("forward",forward)) for regime,report in groups.items() if "selected_model" in report}
    passed=all(report["passed"] for report in list(horizontal.values())+list(forward.values()))
    return {"version":1,"game":"mining","scope":"normal_clear_motion_conditioned_on_ZR",
            "calibrated":False,"passed":passed,"row_count":len(rows),"usable_observations":len(points),
            "exclusions":dict(rejects),"horizontal":horizontal,"forward":forward,
            "forward_speed_observed":forward_observed,"model_parameters_observed":models,
            "unobserved_parameters":unobserved,"infinite_acceleration_established":False,
            "reason":"局部模型达到显式验证门槛，仍需独立应用验证" if passed else
                ("没有可靠世界/滚屏速度观测，不能判断持续ZR后的前进加速度或最高速度" if not forward_observed else
                 "样本、ZR分组覆盖或验证误差不足，尚不能确定普通运动模型"),
            "limitations":["拟合系数是局部估计，不是游戏内部物理常数", "近固定player.y不能替代世界速度",
                           "模型比较只覆盖所测时间与速度区间，不证明无限加速或全局最高分",
                           "验证集用于有限候选模型比较，不是第三组完全独立测试集",
                           "碰撞、碎石、强化与其切换邻近数据须由独立事件观测标注并剔除"]}
