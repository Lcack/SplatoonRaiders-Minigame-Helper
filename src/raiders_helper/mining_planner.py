"""Short-horizon model search, NOT a reconstruction of the game's physics.

All motion values are measurements supplied by the caller. Synthetic fixtures
must label them as assumptions. Unknown buff duration never grants future rock
immunity. Coordinates: x across viewport, depth y down the CURRENT viewport.
"""
from dataclasses import dataclass, field, replace
from functools import lru_cache
import math

from .control.buttons import Button, ControllerState, StickVector
from .models import Decision


@dataclass(frozen=True)
class MotionModel:
    forward_speed: float
    accelerated_speed: float
    lateral_speed: float
    latency_seconds: float
    uncertainty_x: float
    uncertainty_y: float
    calibrated: bool = False
    # A measured LOWER bound, not the ~10 s reported in walkthroughs.
    egg_buff_lower_bound_seconds: float = 0.0
    boost_speed: float | None = None

    def __post_init__(self):
        values = (self.forward_speed, self.accelerated_speed, self.lateral_speed,
                  self.latency_seconds, self.uncertainty_x, self.uncertainty_y,
                  self.egg_buff_lower_bound_seconds)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("运动模型参数必须为有限数")
        if min(self.forward_speed, self.accelerated_speed, self.lateral_speed) <= 0:
            raise ValueError("运动速率必须大于零")
        if self.accelerated_speed < self.forward_speed or min(values[3:]) < 0:
            raise ValueError("运动模型参数范围不正确")
        if self.boost_speed is not None and (not math.isfinite(self.boost_speed) or self.boost_speed <= 0):
            raise ValueError("强化速率必须为正有限数")


@dataclass(frozen=True)
class SearchSettings:
    horizon_seconds: float = 1.5
    step_seconds: float = 0.1
    beam_width: int = 24
    wall_margin: float = 0.015
    steering_penalty: float = 0.002
    time_value: float | None = None
    max_expanded_nodes: int = 30000

    def __post_init__(self):
        values = (self.horizon_seconds, self.step_seconds, self.wall_margin, self.steering_penalty)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("规划参数必须为非负有限数")
        if not 0 < self.step_seconds <= self.horizon_seconds <= 5 or not 1 <= self.beam_width <= 512:
            raise ValueError("规划时域或搜索宽度超出范围")
        if self.time_value is not None and (not math.isfinite(self.time_value) or self.time_value < 0):
            raise ValueError("时间价值必须为非负有限数")
        if not 1 <= self.max_expanded_nodes <= 100000:
            raise ValueError("搜索节点预算超出范围")


@dataclass(frozen=True)
class PathPoint:
    seconds: float
    x: float
    depth: float
    steering: float = 0.0
    accelerate: bool = False


@dataclass(frozen=True)
class Plan:
    points: tuple[PathPoint, ...] = ()
    value: float = 0.0
    gained_seconds: float = 0.0
    collected: tuple[str, ...] = ()
    reason: str = "没有可用路线"
    expanded_nodes: int = 0


class Road:
    def __init__(self, corridors):
        rows = {}
        for e in corridors:
            rows.setdefault(round(e.y, 5), []).append((e.x - e.w / 2, e.x + e.w / 2))
        self.rows = [(y, sorted(intervals)) for y, intervals in sorted(rows.items())]

    @lru_cache(maxsize=4096)
    def intervals(self, y):
        if not self.rows or y < self.rows[0][0] - 1e-6 or y > self.rows[-1][0] + 1e-6:
            return ()
        for row_y, spans in self.rows:
            if abs(y - row_y) < 1e-6:
                return spans
        for (y0, a), (y1, b) in zip(self.rows, self.rows[1:]):
            if y0 < y < y1:
                alpha = (y - y0) / (y1 - y0)
                if len(a) == len(b) == 1:
                    return ((a[0][0] * (1-alpha) + b[0][0] * alpha,
                             a[0][1] * (1-alpha) + b[0][1] * alpha),)
                # Splits/merges are ambiguous: keep only overlapping intervals.
                return tuple((max(l0, l1), min(r0, r1)) for l0, r0 in a for l1, r1 in b
                             if max(l0, l1) < min(r0, r1))
        return ()

    def segment_safe(self, x0, y0, x1, y1, radius):
        # Check all piecewise-linear road knots plus endpoints. For a single
        # interval this covers the whole line, rather than just its endpoint.
        if y0 == y1:
            return any(left + radius <= min(x0, x1) and max(x0, x1) <= right - radius
                       for left, right in self.intervals(y0))
        depths = sorted(set([y0, y1] + [y for y, _ in self.rows if y0 < y < y1]))
        def xpos(y):
            return x0 + (x1-x0) * (y-y0)/(y1-y0)
        for a, b in zip(depths, depths[1:]):
            mid = (a+b)/2
            spans = [self.intervals(y) for y in (a, mid, b)]
            if len(spans[0]) == len(spans[1]) == len(spans[2]):
                # A continuous path must stay in the SAME branch, not jump
                # between disjoint intervals via unobserved gaps.
                if not any(all(left + radius <= xpos(y) <= right - radius
                               for y, (left, right) in zip((a, mid, b), branch))
                           for branch in zip(*spans)):
                    return False
            else:
                # Topology change: only a shared constant interval is trusted.
                if not any(max(l0, l1, l2)+radius <= min(xpos(a), xpos(b))
                           and max(xpos(a), xpos(b)) <= min(r0, r1, r2)-radius
                           for l0, r0 in spans[0] for l1, r1 in spans[1] for l2, r2 in spans[2]):
                    return False
        return True

    def swept_safe(self, x0, y0, x1, y1, radius_x, radius_y):
        """Check the full moving rectangle against all piecewise road knots.

        At each depth, the swept body's occupied x is an interval. Both its
        envelope and road boundaries are linear between the knots below.
        Checking three offset center lines alone can miss a thin inward wall.
        """
        lower, upper = y0-radius_y, y1+radius_y
        depths = sorted(set([lower, upper, y0+radius_y, y1-radius_y]
                            + [y for y, _ in self.rows if lower < y < upper]))
        depths = [y for y in depths if lower <= y <= upper]
        def envelope(y):
            if abs(y1-y0) < 1e-12:
                return min(x0, x1)-radius_x, max(x0, x1)+radius_x
            t0 = max(0.0, min(1.0, (y-radius_y-y0)/(y1-y0)))
            t1 = max(0.0, min(1.0, (y+radius_y-y0)/(y1-y0)))
            a, b = x0+(x1-x0)*t0, x0+(x1-x0)*t1
            return min(a, b)-radius_x, max(a, b)+radius_x
        if len(depths) == 1:
            left, right = envelope(depths[0])
            return any(a <= left and right <= b for a, b in self.intervals(depths[0]))
        for a, b in zip(depths, depths[1:]):
            sample_y = (a, (a+b)/2, b)
            spans = [self.intervals(y) for y in sample_y]
            occupied = [envelope(y) for y in sample_y]
            if len(spans[0]) == len(spans[1]) == len(spans[2]):
                if not any(all(left <= low and high <= right
                               for (left, right), (low, high) in zip(branch, occupied))
                           for branch in zip(*spans)):
                    return False
            else:
                if not any(max(l0, l1, l2) <= min(low for low, _ in occupied)
                           and max(high for _, high in occupied) <= min(r0, r1, r2)
                           for l0, r0 in spans[0] for l1, r1 in spans[1] for l2, r2 in spans[2]):
                    return False
        return True


def contact_fraction(x0, y0, x1, y1, entity, radius_x, radius_y):
    """Continuous segment vs inflated rectangle, returns first contact 0..1."""
    low, high = 0.0, 1.0
    for start, end, center, radius in ((x0, x1, entity.x, entity.w/2 + radius_x),
                                        (y0, y1, entity.y, entity.h/2 + radius_y)):
        delta = end - start
        if radius < 0:
            return None
        if abs(delta) < 1e-12:
            if not center-radius <= start <= center+radius:
                return None
        else:
            t0, t1 = sorted(((center-radius-start)/delta, (center+radius-start)/delta))
            low, high = max(low, t0), min(high, t1)
            if low > high:
                return None
    return low


@dataclass(frozen=True)
class _Node:
    path: tuple[PathPoint, ...]
    remaining: float | None
    buff_until: float
    collected: frozenset[int] = field(default_factory=frozenset)
    gained: float = 0.0
    penalty: float = 0.0


def plan_mining(obs, model: MotionModel, settings=SearchSettings()):
    if obs.phase != "playing" or obs.confidence < 0.8:
        return Plan(reason="状态置信度不足")
    players = obs.of_kind("player")
    if len(players) != 1:
        return Plan(reason="玩家缺失或不唯一")
    player = players[0]
    road = Road(obs.of_kind("corridor"))
    total_uncertainty_x = model.uncertainty_x + model.lateral_speed * model.latency_seconds
    radius_x = player.w/2 + total_uncertainty_x
    radius_y = player.h/2 + model.uncertainty_y
    road_radius = radius_x + settings.wall_margin
    @lru_cache(maxsize=32768)
    def road_safe(x0, y0, x1, y1):
        return road.swept_safe(x0, y0, x1, y1, road_radius, radius_y)
    if not road_safe(player.x, player.y, player.x, player.y):
        return Plan(reason="道路观测未覆盖玩家，或玩家已贴墙")
    remaining = obs.metrics.get("time_remaining")
    if remaining is not None and (not math.isfinite(remaining) or remaining <= 0):
        return Plan(reason="时间已耗尽或计时读数无效")
    if (obs.of_kind("egg_buff") or obs.of_kind("egg_pickup")) and model.boost_speed is None:
        return Plan(reason="强化前进速率尚未测量，不能预测含卵的路线")
    buff_until = obs.metrics.get("buff_remaining_lower_bound_seconds", 0.0) if obs.of_kind("egg_buff") else 0.0
    if not math.isfinite(buff_until) or buff_until < 0:
        buff_until = 0.0
    if obs.of_kind("egg_buff"):
        covered = buff_until - model.latency_seconds
        if covered < settings.step_seconds:
            return Plan(reason="强化剩余有效区间未知或不足一个规划步，等待状态更新")
        # A lower bound must never be treated as an exact expiry: only search
        # within the interval where the active movement mode is confirmed.
        settings = replace(settings, horizon_seconds=min(settings.horizon_seconds, covered))
    if obs.of_kind("egg_pickup") and model.egg_buff_lower_bound_seconds < settings.horizon_seconds + model.latency_seconds:
        return Plan(reason="吃卵后的强化有效区间尚不足以覆盖规划时域")
    objects = tuple(e for e in obs.entities if e.kind in ("reward", "time_pickup", "egg_pickup", "obstacle", "wall"))
    # Deduplicate known tracker IDs so two detections cannot double the reward.
    unique, seen = [], set()
    for e in objects:
        identity = (e.kind, e.track_id) if e.track_id >= 0 else (e.kind, e.x, e.y, e.w, e.h)
        if identity not in seen:
            unique.append(e)
            seen.add(identity)
    objects = tuple(unique)

    @lru_cache(maxsize=32768)
    def geometric_contacts(x0, y0, x1, y1):
        contacts = []
        for index, e in enumerate(objects):
            hazard = e.kind in ("wall", "obstacle")
            if not hazard and e.confidence < 0.8:
                continue
            # Subtract uncertainty from the COMBINED actor/object reach, not
            # just the actor half width. A negative combined radius means the
            # pickup cannot be guaranteed, even when nominal centers coincide.
            rx = radius_x if hazard else player.w/2 - total_uncertainty_x
            ry = radius_y if hazard else player.h/2 - model.uncertainty_y
            fraction = contact_fraction(x0, y0, x1, y1, e, rx, ry)
            if fraction is not None:
                contacts.append((fraction, 0 if hazard else 1, index, e))
        return tuple(sorted(contacts, key=lambda item: item[:3]))

    def advance(node, dt, steering, accelerate, speed_override=None):
        p = node.path[-1]
        gained, collected, buff_end = node.gained, set(node.collected), node.buff_until
        time_left = node.remaining
        path = list(node.path)
        x, y, now = p.x, p.depth, p.seconds
        end_time = now + dt
        # Split exactly at contacts and movement-mode changes. In particular,
        # an egg accelerates the REMAINDER of this same step; recompute its
        # swept path instead of checking only the old-speed endpoint.
        while now < end_time - 1e-10:
            speed = model.accelerated_speed if accelerate else model.forward_speed
            powered = now < buff_end
            if powered:
                speed = model.boost_speed
            if speed_override is not None:
                speed = speed_override
            interval = end_time-now
            if powered:
                interval = min(interval, buff_end-now)
            end_x = round(x + steering * model.lateral_speed * interval, 8)
            end_y = round(y + speed * interval, 8)
            contacts = [event for event in geometric_contacts(x, y, end_x, end_y) if event[2] not in collected]
            event = contacts[0] if contacts else None
            fraction = event[0] if event else 1.0
            elapsed = fraction * interval
            contact_x, contact_y = x+(end_x-x)*fraction, y+(end_y-y)*fraction
            if not road_safe(x, y, contact_x, contact_y):
                return None
            if time_left is not None:
                time_left -= elapsed
                if time_left <= 0:
                    return None
            x, y = round(contact_x, 8), round(contact_y, 8)
            now += elapsed
            if elapsed > 1e-10:
                path.append(PathPoint(now, x, y, steering, accelerate))
            if event:
                _, _, index, e = event
                if e.kind == "wall" or (e.kind == "obstacle" and now >= buff_end):
                    return None
                bonus = 2.0 if e.kind == "egg_pickup" else 1.0
                if e.kind == "egg_pickup":
                    buff_end = max(buff_end, now + model.egg_buff_lower_bound_seconds)
                gained += bonus
                if time_left is not None:
                    time_left += bonus
                collected.add(index)
            elif elapsed <= 1e-10:
                return None
        return _Node(tuple(path), time_left, buff_end, frozenset(collected), gained,
                     node.penalty + settings.steering_penalty * abs(steering-p.steering))

    start = _Node((PathPoint(0, player.x, player.y),), remaining, buff_until)
    # Until new input arrives, model neutral steering at observed scroll speed.
    latency = model.latency_seconds
    if latency:
        observed = obs.metrics.get("forward_speed", model.boost_speed if obs.of_kind("egg_buff") else model.accelerated_speed)
        if not math.isfinite(observed) or observed <= 0:
            return Plan(reason="当前前进速率无效")
        end_y = player.y + observed * latency
        if not road_safe(player.x, player.y, player.x, end_y):
            return Plan(reason="控制延迟期间可能撞墙，需重新标定或降低速度")
        # Conservatively use the faster modeled speed if current speed is not
        # known. The same continuous hazard checks apply during the delay.
        start = advance(start, latency, 0, observed > model.forward_speed, speed_override=observed)
        if start is None:
            return Plan(reason="控制延迟期间存在碰撞或时间耗尽风险")
    time_value = settings.time_value if settings.time_value is not None else model.forward_speed
    def value(node):
        return node.path[-1].depth - player.y + time_value * node.gained - node.penalty
    beam, expanded = [start], 0
    best = start
    steering_options = (-1.0, -0.5, 0.0, 0.5, 1.0)
    steps = max(1, math.ceil(settings.horizon_seconds / settings.step_seconds))
    for step in range(steps):
        dt = min(settings.step_seconds, settings.horizon_seconds - step*settings.step_seconds)
        candidates = []
        for node in beam:
            for accelerate in (False, True):
                for steering in steering_options:
                    expanded += 1
                    if expanded > settings.max_expanded_nodes:
                        return Plan(reason="搜索达到节点预算，缩小搜索宽度或时域", expanded_nodes=expanded-1)
                    child = advance(node, dt, steering, accelerate)
                    if child is not None:
                        candidates.append(child)
        if not candidates:
            return Plan(reason="规划时域被障碍、剩余时间或观测边界截断", expanded_nodes=expanded)
        # Keep distinct endpoints/reward histories. Otherwise action sequences
        # reaching the same state consume the whole beam and hide alternatives.
        candidates.sort(key=value, reverse=True)
        unique_states = {}
        for node in candidates:
            end = node.path[-1]
            key = (round(end.x, 3), round(end.depth, 3), node.collected, round(node.buff_until, 2))
            unique_states.setdefault(key, node)
        beam = list(unique_states.values())[:settings.beam_width]
        best = max(beam, key=value)
    if not any(p.seconds > latency + 1e-9 for p in best.path):
        return Plan(reason="可见范围内没有可执行的首步路线", expanded_nodes=expanded)
    return Plan(best.path, value(best), best.gained,
                tuple(objects[i].kind + ":" + str(objects[i].track_id) for i in sorted(best.collected)),
                "短时域规划成功（模型预测）", expanded)


def planning_decision(obs, parameters):
    motion = parameters.get("motion")
    if not motion:
        return Decision(reason="MINING 规划器等待测量的运动模型")
    model = MotionModel(**motion)
    if model.calibrated is not True:
        return Decision(reason="MINING 规划器的运动模型尚未标定")
    settings = SearchSettings(**parameters.get("search", {}))
    plan = plan_mining(obs, model, settings)
    if not plan.points:
        return Decision(reason="MINING：" + plan.reason)
    action = next(p for p in plan.points if p.seconds > model.latency_seconds + 1e-9)
    state = ControllerState(frozenset({Button.ZR}) if action.accelerate else frozenset(),
                            StickVector(round(action.steering * 32767), 0))
    diagnostics = {"planner": "beam_search", "predicted_added_seconds": plan.gained_seconds,
                   "model_value": plan.value, "collected": plan.collected,
                   "expanded_nodes": plan.expanded_nodes,
                   "path": [{"seconds": p.seconds, "x": p.x, "depth": p.depth} for p in plan.points]}
    return Decision(state, f"MINING 前瞻规划：预测加时 {plan.gained_seconds:g} 秒；{len(plan.points)} 个路径点", obs.confidence, diagnostics)
