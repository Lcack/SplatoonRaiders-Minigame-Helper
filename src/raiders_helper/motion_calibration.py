"""Explicit live probes and held-out horizontal action-response identification.

Rows must describe *actually submitted* commands, never replay advice:
  {timestamp: send_completed_monotonic_seconds, round_id: 1, live: True,
   executed: True, state: state_dict(actual_sent_state),
   observation: asdict(observation)}
Observation.timestamp is the capture-read-completed timestamp in the same clock
domain. Estimated delay therefore includes the observed action/capture chain;
it is not a measurement of the controller transport alone. Human interventions,
unknown camera buffering and imperfect detections can still confound a fit.
No fit in this module marks a game or hardware profile calibrated.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
import math
import time

import numpy as np

from .control.buttons import Button, ControllerState, StickVector
from .models import Decision, Game
from .trace import state_dict


DEFAULT_PROBE_CONFIG = {
    "enabled": False,
    "pulse_seconds": 0.18,
    "neutral_seconds": 0.30,
    "levels": [0.35, 0.65],
    "min_confidence": 0.85,  # entity score; overall phase threshold is 0.8
    "min_track_samples": 3,
    "position_margin": 0.04,
    "enable_jump": False,
    "enable_jets": False,
    "fit": {
        "delay_grid_seconds": [0, 1/60, 2/60, 3/60, 4/60, 5/60, 0.1, 8/60, 10/60, 0.2],
        "min_samples": 80,
        "min_rounds": 3,
        "min_span_seconds": 5,
        "min_axis_coverage": 0.25,
        "min_input_transitions": 6,
        "max_gap_seconds": 0.05,
        "max_sequence_gap": 1,
        "min_position_span": 0.03,
        "min_response_gain_per_second": 0.01,
        # Engineering starting thresholds, NOT validated game tolerances.
        "max_holdout_rmse": 0.08,
        "max_normalized_holdout_rmse": 0.30,
    },
}


def _number(value):
    return (float(value) if not isinstance(value, bool)
            and isinstance(value, (int, float)) and math.isfinite(value) else None)


def _positive(config, key, default):
    value = _number(config.get(key, default))
    if value is None or value <= 0:
        raise ValueError(f"{key} 必须为有限正数")
    return value


class CalibrationProbe:
    """Short open-loop identification pulses; not a score-playing policy.

    Only `enabled: true` opts in. Missing/ambiguous actors, stale observations,
    clock discontinuities and viewport-edge travel produce neutral advice.
    The caller must separately authorize hardware, enforce final freshness,
    record the actually submitted state, and retry between rounds.
    """
    def __init__(self, game, config=None):
        self.game = Game(game)
        self.config = config or {}
        self.pulse = _positive(self.config, "pulse_seconds", 0.18)
        self.neutral = _positive(self.config, "neutral_seconds", 0.30)
        self.entity_score = _positive(self.config, "min_confidence", 0.85)
        self.margin = _positive(self.config, "position_margin", 0.04)
        if self.margin >= 0.2:
            raise ValueError("position_margin 必须小于0.2")
        bounds = self.config.get("play_bounds", [0.0, 1.0])
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(_number(v) is None for v in bounds) or not 0 <= bounds[0] < bounds[1] <= 1):
            raise ValueError("play_bounds 必须是0到1之间的递增边界")
        self.min_samples = self.config.get("min_track_samples", 3)
        if type(self.min_samples) is not int or self.min_samples < 2:
            raise ValueError("min_track_samples 必须为至少2的整数")
        levels = self.config.get("levels", [0.35, 0.65])
        if not isinstance(levels, (list, tuple)) or not levels or any(
                _number(v) is None or not 0 < v <= 1 for v in levels):
            raise ValueError("levels 必须为0到1之间的非空正数列表")
        self.script = []
        if self.game == Game.LANDING:
            if self.config.get("enable_jets") is True:
                for label, buttons in (("left_jet", {Button.ZL}), ("right_jet", {Button.ZR}),
                                       ("both_jets", {Button.ZL, Button.ZR})):
                    self._step(label, ControllerState(frozenset(buttons)))
        else:
            roles = ("player", "partner") if self.game == Game.JUGGLING else ("player",)
            for role in roles:
                for level in levels:
                    for sign in (1, -1):
                        axis = StickVector(round(sign*level*32767), 0)
                        state = (ControllerState(left_stick=axis) if role == "player"
                                 else ControllerState(right_stick=axis))
                        self._step(f"{role}_{sign*level:+g}", state)
            if self.game == Game.HOPPING and self.config.get("enable_jump") is True:
                self._step("jump_B", ControllerState(frozenset({Button.B})), min(self.pulse, 0.08))
        if not self.script:
            self.script = [("neutral_only", self.neutral, ControllerState.neutral())]
        self.period = sum(step[1] for step in self.script)
        self.reset_round()

    def _step(self, label, state, duration=None):
        self.script.extend(((f"neutral_before_{label}", self.neutral, ControllerState.neutral()),
                            (label, self.pulse if duration is None else duration, state)))

    def reset_round(self, round_id=None):
        self.round_id, self.elapsed, self.last_time, self.last_ready = round_id, 0.0, None, False

    def _out(self, state, reason, probe_id="blocked"):
        return Decision(state, "标定："+reason, diagnostics={"calibration_probe": {
            "probe_id": probe_id, "round_id": self.round_id,
            "elapsed_seconds": self.elapsed, "period_seconds": self.period,
            "intended_state": state_dict(state), "experimental": True,
        }})

    def decide(self, obs, now=None):
        now = time.perf_counter() if now is None else now
        current = _number(now)
        dt = current-self.last_time if current is not None and self.last_time is not None else None
        previous_ready = self.last_ready
        self.last_time, self.last_ready = current, False
        neutral = ControllerState.neutral()
        if self.config.get("enabled") is not True:
            return self._out(neutral, "探针未启用")
        if current is None or (dt is not None and not 0 <= dt <= 0.10):
            return self._out(neutral, "时钟不连续，松开并重新等待连续画面")
        if obs.game != self.game or obs.phase != "playing" or _number(obs.confidence) is None or obs.confidence < 0.8:
            return self._out(neutral, "未确认游戏运行画面")
        age = _number(obs.metrics.get("observation_age_seconds", 0))
        if age is None or age > 0.15 or age < 0:
            return self._out(neutral, "观测已过期或时间无效")
        roles = ("player", "partner") if self.game == Game.JUGGLING else ("player",)
        actors = {}
        for role in roles:
            candidates = obs.of_kind(role)
            if len(candidates) != 1:
                return self._out(neutral, f"{role}缺失或不唯一")
            entity = candidates[0]
            if (_number(entity.confidence) is None or entity.confidence < self.entity_score or entity.samples < self.min_samples
                    or entity.track_id < 0 or any(_number(v) is None for v in (entity.x, entity.y, entity.w, entity.h))):
                return self._out(neutral, f"{role}尚无可靠连续轨迹")
            actors[role] = entity
        self.last_ready = True
        if not previous_ready:
            return self._out(neutral, "连续画面恢复，先松开一帧")
        self.elapsed += dt or 0.0
        position = self.elapsed % self.period
        label, state = "neutral", neutral
        for step_label, duration, step_state in self.script:
            if position < duration:
                label, state = step_label, step_state
                break
            position -= duration
        if state.buttons and self.game == Game.LANDING and (
                obs.metrics.get("pose_known") is not True or obs.metrics.get("pose_confidence", 0) < 0.58):
            return self._out(neutral, "喷射探针等待可靠朝向", label)
        if Button.B in state.buttons:
            balls = obs.of_kind("ball")
            if (len(balls) != 1 or balls[0].samples < self.min_samples
                    or _number(balls[0].confidence) is None or balls[0].confidence < self.entity_score):
                return self._out(neutral, "顶球探针等待可靠乌贼轨迹", label)
        for role, vector in (("player", state.left_stick), ("partner", state.right_stick)):
            if not vector.x:
                continue
            entity = actors[role]
            low, high = (0.0, 0.5) if self.game == Game.JUGGLING and role == "player" else (
                (0.5, 1.0) if self.game == Game.JUGGLING else tuple(self.config.get("play_bounds", [0.0, 1.0])))
            if ((vector.x < 0 and entity.x-entity.w/2 <= low+self.margin)
                    or (vector.x > 0 and entity.x+entity.w/2 >= high-self.margin)):
                return self._out(neutral, "接近边界，取消向外的探针", label)
        return self._out(state, "短脉冲采样，非保球/得分策略", label)


def _mapping(value):
    value = asdict(value) if is_dataclass(value) else value
    return value if isinstance(value, dict) else {}


def _state(value):
    if isinstance(value, ControllerState):
        value = state_dict(value)
    if not isinstance(value, dict):
        return None
    try:
        left = _number(value["left_stick"]["x"])
        right = _number(value["right_stick"]["x"])
        buttons = [Button(v).value for v in value.get("buttons", [])]
        if left is None or right is None or abs(left) > 32767 or abs(right) > 32767:
            return None
        return {"left": left/32767, "right": right/32767, "buttons": buttons}
    except (KeyError, TypeError, ValueError):
        return None


def _input_at(events, timestamp, axis, max_age=None):
    times = events["times"]
    index = bisect_right(times, timestamp)-1
    if index < 0 or (max_age is not None and timestamp-times[index] > max_age):
        return None
    state = events["states"][index]
    # A button action is outside this horizontal-only model. A long command
    # history gap may contain a watchdog release that was not logged as a row;
    # do not assume the last nonneutral command remained physically held.
    return None if state["buttons"] else state[axis]


def _fit_actor(rows, actor, config):
    axis = "left" if actor == "player" else "right"
    rejects = Counter()
    events = defaultdict(lambda: {"times": [], "states": []})
    points = []
    confidence = _positive(config, "min_entity_confidence", 0.85)
    max_gap = _positive(config, "max_gap_seconds", 0.05)
    max_sequence_gap = config.get("max_sequence_gap", 1)
    if type(max_sequence_gap) is not int or max_sequence_gap < 1:
        raise ValueError("max_sequence_gap 必须为正整数")
    delay_grid = config.get("delay_grid_seconds", [0, 1/60, 2/60, 3/60, 4/60, 5/60, 0.1, 8/60, 10/60, 0.2])
    if not isinstance(delay_grid, (list, tuple)) or not delay_grid or any(
            _number(v) is None or v < 0 or v > 1 for v in delay_grid):
        raise ValueError("delay_grid_seconds 必须为0到1秒的非空延迟列表")
    delays = sorted(set(map(float, delay_grid)))
    previous = None
    for row in rows:
        reason = None
        observation = _mapping(row.get("observation", {}))
        command = _state(row.get("state"))
        timestamp = _number(row.get("timestamp"))
        round_id = row.get("round_id")
        if row.get("live") is not True or row.get("executed") is not True:
            reason = "not_live_executed"
        elif timestamp is None or isinstance(round_id, bool) or not isinstance(round_id, (int, str)) or command is None:
            reason = "invalid_row"
        if reason:
            rejects[reason] += 1
            previous = None
            continue
        # Commands remain in history even if the associated observation was
        # unusable: the hardware still received that actual command.
        if events[round_id]["times"] and timestamp <= events[round_id]["times"][-1]:
            rejects["command_clock_not_increasing"] += 1
            previous = None
            continue
        events[round_id]["times"].append(timestamp)
        events[round_id]["states"].append(command)
        capture_time = _number(observation.get("timestamp", timestamp))
        sequence = observation.get("sequence")
        candidates = [_mapping(e) for e in observation.get("entities", []) if _mapping(e).get("kind") == actor]
        if (observation.get("phase") != "playing" or _number(observation.get("confidence")) is None
                or observation["confidence"] < 0.8):
            reason = "not_reliable_playing"
        elif (capture_time is None or capture_time > timestamp or type(sequence) is not int
              or timestamp-capture_time > config.get("max_frame_age_seconds", 0.2)):
            reason = "invalid_observation_time_or_sequence"
        elif len(candidates) != 1:
            reason = "actor_missing_or_ambiguous"
        else:
            entity = candidates[0]
            if (_number(entity.get("confidence")) is None or entity["confidence"] < confidence
                    or entity.get("samples", 0) < 3 or type(entity.get("track_id")) is not int
                    or entity["track_id"] < 0 or _number(entity.get("x")) is None):
                reason = "actor_track_unreliable"
            elif command["buttons"]:
                reason = "buttons_outside_horizontal_model"
        if reason:
            rejects[reason] += 1
            previous = None
            continue
        current = {"round": round_id, "time": capture_time, "x": float(entity["x"]),
                   "sequence": sequence, "track": entity["track_id"]}
        if previous:
            dt = current["time"]-previous["time"]
            if (current["round"] == previous["round"] and current["track"] == previous["track"]
                    and 0 < current["sequence"]-previous["sequence"] <= max_sequence_gap and 0 < dt <= max_gap):
                current["velocity"] = (current["x"]-previous["x"])/dt
                current["velocity_time"] = (current["time"]+previous["time"])/2
                if "velocity" in previous:
                    span = current["velocity_time"]-previous["velocity_time"]
                    points.append({"round": round_id, "time": current["velocity_time"], "dt": span,
                                   "previous_velocity": previous["velocity"], "velocity": current["velocity"],
                                   "x": current["x"]})
            else:
                rejects["gap_round_or_track_change"] += 1
        previous = current
    # Use the identical eligible observations for all delay candidates.
    points = [p for p in points if all(
        _input_at(events[p["round"]], p["time"]-delay, axis, max_gap) is not None for delay in delays)]
    base = {"actor": actor, "passed": False, "sample_count": len(points), "rejections": dict(rejects),
            "velocity_source": "consecutive_position_differences", "scope": "horizontal_only",
            "reasons": []}
    if len(points) < 10:
        base["reasons"] = ["可用连续实机样本不足"]
        return base
    rounds = list(dict.fromkeys(p["round"] for p in points))
    if len(rounds) >= 2:
        test_rounds = set(rounds[-max(1, math.ceil(len(rounds)*0.25)):])
        train_ids = [i for i, p in enumerate(points) if p["round"] not in test_rounds]
        test_ids = [i for i, p in enumerate(points) if p["round"] in test_rounds]
        split = "whole_rounds"
    else:
        border = points[math.floor(len(points)*0.7)]["time"]
        train_ids = [i for i, p in enumerate(points) if p["time"] < border-delays[-1]-max_gap]
        test_ids = [i for i, p in enumerate(points) if p["time"] >= border]
        split = "purged_time_blocks"
    if len(train_ids) < 6 or len(test_ids) < 3:
        base["reasons"] = ["训练或独立验证样本不足"]
        return base
    y = np.array([p["velocity"] for p in points])
    velocity = np.array([p["previous_velocity"] for p in points])
    dts = np.array([p["dt"] for p in points])
    candidates = []
    for delay in delays:
        u = np.array([_input_at(events[p["round"]], p["time"]-delay, axis, max_gap) for p in points])
        # Differential parametrization tolerates nonidentical frame intervals:
        # v_next = v_prev + dt * (a*v_prev + b*u_delayed + c).
        matrix = np.column_stack((dts*velocity, dts*u, dts))
        train = matrix[train_ids]
        if np.linalg.matrix_rank(train) < 3:
            continue
        coefficients = np.linalg.lstsq(train, (y-velocity)[train_ids], rcond=None)[0]
        predictions = velocity+matrix@coefficients
        train_rmse = float(np.sqrt(np.mean((predictions[train_ids]-y[train_ids])**2)))
        candidates.append((train_rmse, delay, coefficients, predictions, u))
    if not candidates:
        base["reasons"] = ["输入激励或速度变化不足，设计矩阵秩不足"]
        return base
    # Delay selection uses training only. Held-out data never choose a delay.
    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
    train_rmse, delay, coefficients, predictions, u = candidates[0]
    holdout_rmse = float(np.sqrt(np.mean((predictions[test_ids]-y[test_ids])**2)))
    holdout_std = float(np.std(y[test_ids]))
    normalized = holdout_rmse/max(holdout_std, 0.01)
    median_dt = float(np.median(dts))
    span = float(np.sum(dts))  # observed consecutive time, excluding unseen gaps
    transitions = sum(points[i]["round"] == points[i-1]["round"] and abs(u[i]-u[i-1]) > 0.10
                      for i in range(1, len(points)))
    position_span = float(max(p["x"] for p in points)-min(p["x"] for p in points))
    std_input = float(np.std(u[train_ids]))
    checks = {
        "sample_count": len(points) >= config.get("min_samples", 80),
        "round_count": len(rounds) >= config.get("min_rounds", 3),
        "observed_span": span >= config.get("min_span_seconds", 5),
        "input_coverage": std_input >= config.get("min_axis_coverage", 0.25),
        "both_directions": float(np.mean(u > 0.1)) >= 0.05 and float(np.mean(u < -0.1)) >= 0.05,
        "neutral_samples": float(np.mean(abs(u) < 0.05)) >= 0.05,
        "input_transitions": transitions >= config.get("min_input_transitions", 6),
        "position_motion": position_span >= config.get("min_position_span", 0.03),
        "nonzero_response": abs(coefficients[1]) >= config.get("min_response_gain_per_second", 0.01),
        "holdout_threshold_configured": _number(config.get("max_holdout_rmse")) is not None
            and config["max_holdout_rmse"] > 0,
        "normalized_threshold_configured": _number(config.get("max_normalized_holdout_rmse")) is not None
            and config["max_normalized_holdout_rmse"] > 0,
    }
    checks["holdout_error"] = checks["holdout_threshold_configured"] and holdout_rmse <= config["max_holdout_rmse"]
    checks["normalized_holdout_error"] = (checks["normalized_threshold_configured"]
                                         and normalized <= config["max_normalized_holdout_rmse"])
    base.update({"passed": bool(all(checks.values())), "checks": checks, "round_ids": rounds,
                 "split": split, "train_count": len(train_ids), "holdout_count": len(test_ids),
                 "span_seconds": float(span), "median_interval_seconds": median_dt,
                 "input_standard_deviation": std_input, "input_transitions": int(transitions),
                 "position_span": position_span, "train_rmse": train_rmse,
                 "holdout_rmse": holdout_rmse, "normalized_holdout_rmse": float(normalized),
                 "holdout_velocity_std": holdout_std, "effective_delay_seconds": float(delay),
                 "continuous_coefficients": {"velocity_feedback_per_second": float(coefficients[0]),
                                             "input_gain_per_second": float(coefficients[1]),
                                             "bias_widths_per_second_squared": float(coefficients[2])},
                 "discrete_at_median_interval": {"alpha": float(1+coefficients[0]*median_dt),
                                                  "beta": float(coefficients[1]*median_dt),
                                                  "bias": float(coefficients[2]*median_dt)},
                 "delay_train_grid": [{"seconds": float(c[1]), "rmse": float(c[0])} for c in candidates],
                 "reasons": [key for key, passed in checks.items() if not passed]})
    return base


def fit_motion(rows, config=None):
    """Fit an observed horizontal response, with explicit held-out thresholds.

    `config` may be the `fit` dictionary or the entire probe configuration.
    `passed` means only the declared sample/holdout checks passed for this
    limited model. It never means vision, full dynamics, score play or the
    hardware profile is calibrated. Landing's coupled model is not fitted.
    """
    config = config or {}
    config = config.get("fit", config)
    rows = list(rows)
    games = {str(_mapping(row.get("observation", {})).get("game", "")) for row in rows
             if row.get("live") is True and row.get("executed") is True}
    # str(Game.HOPPING) is Game.HOPPING; JSON encodes its value. Support both.
    games = {game.split(".")[-1].lower() for game in games if game}
    result = {"version": 1, "scope": "horizontal_action_response_only", "calibrated": False,
              "passed": False, "actors": {}, "row_count": len(rows),
              "limitations": ["不是画面识别准确率或完整游戏动力学", "电脑时钟延迟不能分离串口与采集链路",
                              "未估计MINING世界纵向滚屏、HOPPING球重力或LANDING姿态耦合"]}
    if len(games) != 1 or next(iter(games), "") not in {game.value for game in Game}:
        result["reason"] = "缺少单一游戏的已执行实机数据，拒绝合并或拟合回放"
        return result
    result["game"] = game = next(iter(games))
    if game == "landing":
        result["reason"] = "LANDING需姿态、推力、角加速度和风的耦合模型；当前仅保留原始采样，不能通过横向模型宣称标定完成"
        return result
    roles = ("player", "partner") if game == "juggling" else ("player",)
    result["actors"] = {role: _fit_actor(rows, role, config) for role in roles}
    result["passed"] = all(model["passed"] for model in result["actors"].values())
    result["reason"] = ("所声明横向模型通过留出误差与覆盖检查，仍需独立应用验证"
                        if result["passed"] else "样本、激励或留出误差尚未满足所声明阈值")
    return result
