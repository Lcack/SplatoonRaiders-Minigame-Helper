"""Bounded, explicitly enabled MINING pulse experiments, never score play.

The speed/margin bounds are experimental hypotheses, not calibrated physics.
Every interruption invalidates the whole segment; the next attempt begins a
new baseline with a new identity. The caller records actual transport times.
"""
from dataclasses import dataclass
import math
from numbers import Real

from .control.buttons import Button, ControllerState, StickVector
from .mining_planner import Road, contact_fraction
from .models import Decision, Game


def _number(value):
    return float(value) if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) else None


def _positive(config, key, default, maximum):
    value = _number(config.get(key, default))
    if value is None or not 0 < value <= maximum:
        raise ValueError(f"{key} 必须为0到{maximum}之间的有限正数")
    return value


def _entity_valid(entity):
    return (all(_number(getattr(entity, name)) is not None for name in ("x", "y", "w", "h", "confidence"))
            and 0 <= entity.x <= 1 and 0 <= entity.y <= 1 and entity.w > 0 and entity.h >= 0)


def _local_span(road, y, depth, rx, ry):
    """A full-body interval over only a short local band, never all future rows."""
    lower, upper = y-ry, y+depth+ry
    if not road.rows or lower < road.rows[0][0]-1e-6 or upper > road.rows[-1][0]+1e-6:
        return None
    depths = [lower, upper, *[row_y for row_y, _ in road.rows if lower < row_y < upper]]
    left, right = 0.0, 1.0
    for row_y in depths:
        spans = road.intervals(row_y)
        if len(spans) != 1:
            return None
        a, b = spans[0]
        left, right = max(left, a+rx), min(right, b-rx)
    return (left, right) if left <= right else None


@dataclass(frozen=True)
class _Spec:
    kind: str
    axis: float
    pulse: float
    reverse: float
    coast: float
    target_delta: float | None = None

    def stages(self, baseline):
        stages = [("baseline", baseline, 0.0), ("pulse", self.pulse, self.axis)]
        if self.reverse:
            stages.append(("reverse", self.reverse, -self.axis))
        stages.append(("coast", self.coast, 0.0))
        return stages


class MiningPhysicsProbe:
    """Return experimental inputs; no capture, controller or background thread.

    Public configuration defaults: enabled=False, axis=.4, baseline_seconds=.30,
    pulse_seconds=[.35,.60], coast_seconds=.55, reverse_pulse_seconds=.45,
    reverse_seconds=.25, reverse_coast_seconds=.40, max_start_velocity=.03,
    max_lateral_speed=.4, coast_reserve_seconds=.10, near_depth=.07.

    ``max_lateral_speed`` bounds widths/s per normalized stick; remaining
    displacement reserves multiply this hypothesis by the requested axis.
    Coast reserves only a short reaction window and is rechecked each frame.
    These checks cannot certify an unseen future road or hardware response.
    """
    def __init__(self, config=None):
        self.config = dict(config or {})
        self.enabled = self.config.get("enabled") is True
        self.axis = _positive(self.config, "axis", .4, .65)
        self.baseline = _positive(self.config, "baseline_seconds", .30, 2.0)
        self.coast = _positive(self.config, "coast_seconds", .55, 2.0)
        self.reverse_pulse = _positive(self.config, "reverse_pulse_seconds", .45, 1.0)
        self.reverse = _positive(self.config, "reverse_seconds", .25, 1.0)
        self.reverse_coast = _positive(self.config, "reverse_coast_seconds", .40, 2.0)
        durations = self.config.get("pulse_seconds", [.35, .60])
        if (not isinstance(durations, (list, tuple)) or not durations
                or any(_number(value) is None or not .1 <= value <= 1 for value in durations)):
            raise ValueError("pulse_seconds 必须是0.1到1秒之间的非空时长列表")
        self.max_start_velocity = _positive(self.config, "max_start_velocity", .03, .10)
        self.max_lateral = _positive(self.config, "max_lateral_speed", .4, 1.0)
        self.tail = _positive(self.config, "coast_reserve_seconds", .10, .5)
        self.near_depth = _positive(self.config, "near_depth", .07, .20)
        self.hazard_depth = _positive(self.config, "hazard_depth", .14, .30)
        self.margin = _positive(self.config, "wall_margin", .012, .08)
        self.uncertainty = _positive(self.config, "position_uncertainty", .004, .04)
        self.max_age = _positive(self.config, "max_frame_age_seconds", .18, .20)
        self.max_dt = _positive(self.config, "max_step_seconds", .10, .20)
        self.positioning_axis = _positive(self.config, "positioning_axis", .30, .40)
        self.min_samples = self.config.get("min_track_samples", 3)
        if type(self.min_samples) is not int or self.min_samples < 3:
            raise ValueError("min_track_samples 必须为至少3的整数")
        self.specs = [_Spec("pulse_coast", sign*self.axis, float(duration), 0.0, self.coast)
                      for duration in durations for sign in (1, -1)]
        self.specs += [_Spec("reverse", sign*self.axis, self.reverse_pulse, self.reverse, self.reverse_coast)
                       for sign in (1, -1)]
        if "target_tasks" in self.config:
            tasks=self.config["target_tasks"]
            if not isinstance(tasks,list) or not tasks:raise ValueError("target_tasks需为非空列表")
            self.specs=[]
            for task in tasks:
                axis,delta,hold=(_number(task.get(k)) for k in ("axis","delta_x","pulse_seconds"))
                if (axis is None or delta is None or hold is None or not 0<abs(axis)<=.65
                        or not .005<=abs(delta)<=.10 or axis*delta<=0 or not .1<=hold<=1):
                    raise ValueError("目标任务需有效方向、位移与保持时长")
                self.specs.append(_Spec("pulse_coast",axis,hold,0.,self.coast,delta))
        self.reset_round(1)

    def reset_round(self, round_id, mode="normal"):
        if type(round_id) is not int or round_id < 1 or mode not in {"normal", "zr"}:
            raise ValueError("round_id需为正整数，mode需为normal或zr")
        self.round_id, self.mode = round_id, mode
        self._last_time = self._last_sequence = self._experiment_started = None
        self._active = None
        self._spec_index = self._attempt = self._ready_frames = 0

    def _output(self, now, reason, *, axis=0.0, source="waiting", active=None,
                validity="not_started", events=(), extra=None):
        active = self._active if active is None else active
        spec = active["spec"] if active else self.specs[self._spec_index % len(self.specs)]
        stage = active["stages"][active["index"]][0] if active else source
        elapsed = max(0.0, now-active["started"]) if active and now is not None else 0.0
        stage_elapsed = max(0.0, now-active["stage_started"]) if active and now is not None else 0.0
        identification = {
            "round": self.round_id, "round_id": self.round_id, "mode": self.mode,
            "segment_id": active["id"] if active else None,
            "protocol": spec.kind, "stage": stage, "axis": axis,
            "pulse_seconds": spec.pulse, "reverse_seconds": spec.reverse,
            "coast_seconds": spec.coast, "baseline_seconds": self.baseline,
            "elapsed": elapsed, "experiment_elapsed": max(0.0, now-self._experiment_started)
                if now is not None and self._experiment_started is not None else 0.0,
            "stage_elapsed": stage_elapsed, "stage_started_at": active["stage_started"] if active else None,
            "segment_started_at": active["started"] if active else None,
            "segment_validity": validity, "segment_aborted": validity == "aborted",
            "abort_reason": reason if validity == "aborted" else None,
            "state_input_kind": source, "source": source, "events": list(events),
            "experimental": True, "calibrated": False,
            "requested_delta_x": spec.target_delta,
            "target_reference_x": active.get("target_reference_x") if active else None,
            "target_reference_timestamp": active.get("target_reference_timestamp") if active else None,
            "target_reference_velocity": active.get("target_reference_velocity") if active else None,
            "target_x": active.get("target_x") if active else None,
            "safety_hypothesis": {"max_lateral_speed": self.max_lateral,
                                  "coast_reserve_seconds": self.tail, "near_depth": self.near_depth},
        }
        if extra:
            identification.update(extra)
        buttons = frozenset({Button.ZR}) if source == "probe" and self.mode == "zr" else frozenset()
        state = ControllerState(buttons, StickVector(round(axis*32767), 0))
        return Decision(state, "MINING标定："+reason,
                        diagnostics={"identification": identification})

    def _abort(self, now, reason):
        active = self._active
        self._active = None
        self._ready_frames = 0
        if active:
            self._spec_index += 1
            event = {"type": "aborted", "segment_id": active["id"], "reason": reason,
                     "timestamp": now, "elapsed": max(0.0, now-active["started"]) if now is not None else None}
            return self._output(now, reason, source="aborted", active=active,
                                validity="aborted", events=[event], extra={"stage": "aborted"})
        return self._output(now, reason)

    def _scene(self, obs, now):
        if obs.game != Game.MINING or obs.phase != "playing" or _number(obs.confidence) is None or obs.confidence < .8:
            return None, "未确认MINING运行画面"
        metrics = obs.metrics
        if metrics.get("game_ended") is True or _number(metrics.get("time_remaining", 1)) is None or metrics.get("time_remaining", 1) <= 0:
            return None, "游戏已结束或计时无效"
        if any(metrics.get(key) is True for key in ("collision_suspected", "hit_body_detected", "collision", "wall_collision", "rock_collision", "buff_active", "egg_buff")):
            return None, "碰撞、强化或异常运动，取消本段"
        if obs.of_kind("player_hit") or obs.of_kind("egg_buff") or obs.of_kind("egg_pickup"):
            return None, "碰撞外观或强化目标出现，取消本段"
        age, metric_age = now-_number(obs.timestamp) if _number(obs.timestamp) is not None else None, _number(metrics.get("observation_age_seconds", 0))
        if age is None or metric_age is None or not 0 <= age <= self.max_age or not 0 <= metric_age <= self.max_age:
            return None, "画面过期或采集时间无效"
        hud = _number(metrics.get("hud_anchor_score"))
        if hud is None or hud < .65:
            return None, "HUD未确认"
        players = obs.of_kind("player")
        if len(players) != 1:
            return None, "蓝色玩家缺失或不唯一"
        player = players[0]
        if (not _entity_valid(player) or player.confidence < .85 or player.samples < self.min_samples
                or type(player.track_id) is not int or player.track_id < 0 or _number(player.vx) is None):
            return None, "蓝色玩家尚无可靠连续身份"
        if abs(player.vx) > self.max_lateral:
            return None, "观测横速超过当前试验边界"
        corridors = [e for e in obs.of_kind("corridor") if _entity_valid(e) and e.confidence >= .8]
        road = Road(corridors)
        rx, ry = player.w/2+self.margin+self.uncertainty, player.h/2+self.uncertainty
        body = _local_span(road, player.y, 0.0, player.w/2, player.h/2)
        local = _local_span(road, player.y, self.near_depth, rx, ry)
        if body is None or local is None or not body[0] <= player.x <= body[1]:
            return None, "道路未可靠覆盖完整身体及近前方"
        hazards = []
        for entity in obs.entities:
            if entity.kind not in {"obstacle", "rock", "wall"}:
                continue
            if not _entity_valid(entity):
                return None, "障碍坐标不可靠"
            if entity.confidence >= .8:
                hazards.append(entity)
        return (player, road, local, rx, ry, hazards), None

    def _offsets(self, stages, index, stage_elapsed):
        """Conservative remaining lateral envelope, refreshed every frame."""
        position, low, high, last_axis = 0.0, 0.0, 0.0, 0.0
        for offset, (_, duration, axis) in enumerate(stages[index:]):
            duration = max(0.0, duration-stage_elapsed) if offset == 0 else duration
            if axis:
                position += self.max_lateral*axis*duration
                low, high, last_axis = min(low, position), max(high, position), axis
        # A short reaction reserve, not a claim that drag is already known.
        if last_axis:
            position += self.max_lateral*last_axis*self.tail
            low, high = min(low, position), max(high, position)
        return low, high

    def _clear(self, scene, low, high):
        player, _, local, rx, ry, hazards = scene
        if player.x+low < local[0] or player.x+high > local[1]:
            return False, "测试方向的道路余量不足"
        left, right = player.x+low-rx, player.x+high+rx
        top, bottom = player.y-ry, player.y+self.hazard_depth+ry
        if any(left <= e.x+e.w/2 and e.x-e.w/2 <= right and top <= e.y+e.h/2 and e.y-e.h/2 <= bottom for e in hazards):
            return False, "本体或近前方存在障碍风险"
        return True, None

    def _position(self, scene, now, spec, reason):
        player, road, local, rx, ry, hazards = scene
        stages = spec.stages(self.baseline)
        low, high = self._offsets(stages, 0, 0)
        start_left, start_right = local[0]-low, local[1]-high
        target = (local[0]+local[1])/2
        if start_left <= start_right:
            target = max(start_left, min(start_right, target))
        error = target-player.x
        control = max(-self.positioning_axis, min(self.positioning_axis, 3.2*error-1.5*player.vx))
        if abs(error) < .008 and abs(player.vx) < .02:
            control = 0.0
        destination = player.x+control*self.max_lateral*self.max_dt
        # Positioning may correct a margin breach, but may not leave real floor.
        if (not road.swept_safe(player.x, player.y, destination, player.y+self.near_depth,
                                player.w/2+self.uncertainty, ry)
                or any(contact_fraction(player.x, player.y, destination, player.y+self.near_depth,
                                        e, rx, ry) is not None for e in hazards)):
            control = 0.0
        return self._output(now, reason, axis=control, source="positioning",
                            extra={"positioning_target_x": target, "start_interval": [start_left, start_right],
                                   "reserved_lateral_offsets": [low, high]})

    def decide(self, obs, now):
        now = _number(now)
        if not self.enabled:
            return self._abort(now, "探针未显式启用")
        previous = self._last_time
        self._last_time = now
        if now is None or (previous is not None and not 0 < now-previous <= self.max_dt+1e-9):
            return self._abort(now, "采样间隔中断或时钟不连续")
        if self._experiment_started is None:
            self._experiment_started = now
        sequence = obs.sequence
        if type(sequence) is not int or (self._last_sequence is not None and sequence <= self._last_sequence):
            return self._abort(now, "重复帧或采集序号倒退")
        self._last_sequence = sequence
        scene, reason = self._scene(obs, now)
        if scene is None:
            return self._abort(now, reason)
        player = scene[0]
        if self._active:
            active = self._active
            if player.track_id != active["track"]:
                return self._abort(now, "玩家身份切换，中止整段")
            stage, duration, axis = active["stages"][active["index"]]
            stage_elapsed = now-active["stage_started"]
            low, high = self._offsets(active["stages"], active["index"], stage_elapsed)
            # A reversal does not instantly cancel the measured old velocity.
            drift = player.vx*self.max_dt
            low, high = min(low, drift), max(high, drift)
            clear, reason = self._clear(scene, low, high)
            if not clear:
                return self._abort(now, reason)
            events = []
            if stage_elapsed >= duration-1e-9:
                if stage == "baseline" and abs(player.vx) > self.max_start_velocity:
                    return self._abort(now, "基线结束仍有明显横向漂移")
                events.append({"type": "stage_finished", "segment_id": active["id"], "stage": stage,
                               "timestamp": now, "actual_seconds": stage_elapsed, "nominal_seconds": duration})
                active["index"] += 1
                active["stage_started"] = now
                if active["index"] == len(active["stages"]):
                    self._active = None
                    self._spec_index += 1
                    self._ready_frames = 0
                    events.append({"type": "completed", "segment_id": active["id"], "timestamp": now,
                                   "elapsed": now-active["started"]})
                    active["index"] -= 1
                    return self._output(now, "整段采样完成，释放输入后准备下一段", source="completed", active=active,
                                        validity="completed", events=events, extra={"stage": "completed"})
                stage, _, axis = active["stages"][active["index"]]
                if stage=="pulse" and active["spec"].target_delta is not None:
                    active.update(target_reference_x=player.x,target_reference_timestamp=obs.timestamp,
                                  target_reference_velocity=player.vx,
                                  target_x=player.x+active["spec"].target_delta)
            return self._output(now, "连续分段采样；以实际发送日志确定脉宽", axis=axis,
                                source="probe", validity="running", events=events)

        self._ready_frames += 1
        spec = self.specs[self._spec_index % len(self.specs)]
        low, high = self._offsets(spec.stages(self.baseline), 0, 0)
        clear, reason = self._clear(scene, low, high)
        if self._ready_frames < 2 or abs(player.vx) > self.max_start_velocity or not clear:
            return self._position(scene, now, spec, reason or "等待连续静止轨迹并定位到测试起点")
        self._attempt += 1
        self._active = {"id": f"r{self.round_id:03d}-{self.mode}-s{self._attempt:04d}",
                        "spec": spec, "stages": spec.stages(self.baseline), "index": 0,
                        "started": now, "stage_started": now, "track": player.track_id}
        return self._output(now, "开始新段基线，后续保持本段固定模式", source="probe", validity="running",
                            events=[{"type": "started", "segment_id": self._active["id"], "timestamp": now}])
