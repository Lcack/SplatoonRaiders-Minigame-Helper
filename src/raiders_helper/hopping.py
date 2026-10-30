"""Fast HOPPING scene analysis and experimental input suggestions.

This is a local prediction policy, not verified game physics. Gravity, movement
speed, jump timing and buff duration are never filled from walkthrough guesses.
The few default margins/horizons are declared control heuristics. Hardware
calibration and stale-frame gating remain the application's responsibility.
"""
import math

from .control.buttons import Button, ControllerState, StickVector
from .models import Decision, Entity, Game, Observation


def _finite(value, fallback=None):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else fallback


def _positive(value):
    value = _finite(value)
    return value if value is not None and value > 0 else None


def _reflect(x, low, high):
    """Fold horizontal prediction at the configured reflecting side bounds."""
    width = high-low
    if width <= 0:
        return (low+high)/2
    folded = (x-low) % (2*width)
    return low + (folded if folded <= width else 2*width-folded)


def _arrival(ball, head_y, gravity):
    """Time to the robot's head plane on a descending branch, if observable."""
    delta = head_y-ball.y
    if gravity is not None and gravity > 1e-8:
        discriminant = ball.vy*ball.vy + 2*gravity*delta
        if discriminant < 0:
            return None
        roots = ((-ball.vy-math.sqrt(discriminant))/gravity,
                 (-ball.vy+math.sqrt(discriminant))/gravity)
        choices = [t for t in roots if t > 0 and ball.vy+gravity*t > 0]
        return min(choices, default=None)
    if ball.vy <= 1e-8 or delta <= 0:
        return None
    return delta/ball.vy


def _axis(delta, gain, deadzone):
    if abs(delta) <= deadzone:
        return 0
    return round(max(-1.0, min(1.0, delta*gain))*32767)


def _wait(reason, risks=()):
    return Decision(reason="HOPPING："+reason, diagnostics={
        "policy": "hopping_fast", "mode": "unknown", "experimental": True,
        "target": None, "intercepts": [], "risks": list(risks),
    })


def decide_hopping(obs: Observation, parameters: dict | None = None) -> Decision:
    """Keep the squid alive first; exploit eggs/golden enemies only if feasible.

    Parameters accepted by this policy:
      lateral_speed: measured robot speed in viewport widths/s (optional).
      gravity or metrics.ball_gravity_estimate: measured squid acceleration.
      latency_seconds: reaction delay; metrics take precedence.
      allow_jump + jump_window_seconds: explicit opt-in to B suggestions.
      calibrated: status of the policy's motion model (diagnostic only).

    Missing motion estimates restrict opportunities rather than inventing them.
    """
    parameters = parameters or {}
    if (obs.game != Game.HOPPING or obs.phase != "playing"
            or not math.isfinite(obs.confidence) or obs.confidence < 0.8):
        return _wait("画面阶段或整体识别尚不可靠")
    countdown = _finite(obs.metrics.get("time_remaining"))
    if countdown is not None and countdown <= 0:
        return _wait("倒计时结束，等待结算")
    players = [e for e in obs.of_kind("player") if e.confidence >= 0.8]
    balls = [e for e in obs.of_kind("ball") if e.confidence >= 0.8]
    if len(players) != 1:
        return _wait("机器人缺失或身份不唯一")
    if len(balls) != 1:
        return _wait("乌贼缺失或出现多个候选，等待消除歧义")
    player, ball = players[0], balls[0]
    if ball.samples < 2:
        return _wait("等待乌贼的连续轨迹")
    if not all(math.isfinite(v) for e in (player, ball) for v in (e.x, e.y, e.w, e.h, e.vx, e.vy)):
        return _wait("目标位置或速度不是有效数值")

    risks = []
    lateral_speed = _positive(parameters.get("lateral_speed"))
    gravity = _finite(obs.metrics.get("ball_gravity_estimate"), _finite(parameters.get("gravity")))
    if gravity is not None and gravity < 0:
        gravity = None
    latency = max(0.0, _finite(obs.metrics.get("latency_seconds"), _finite(parameters.get("latency_seconds"), 0.0)))
    preview = max(0.05, min(1.0, _finite(parameters.get("preview_seconds"), 0.25)))
    urgent_seconds = max(0.05, _finite(parameters.get("urgent_seconds"), 0.35))
    gain = max(0.1, min(20, _finite(parameters.get("axis_gain"), 6.0)))
    deadzone = max(0.0, min(0.1, _finite(parameters.get("deadzone"), 0.012)))
    catch_margin = max(0.0, min(0.1, _finite(parameters.get("catch_margin"), 0.008)))
    bounds = parameters.get("play_bounds", [0.0, 1.0])
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
            or not all(_finite(v) is not None for v in bounds) or not 0 <= bounds[0] < bounds[1] <= 1):
        return _wait("play_bounds 配置无效")
    low, high = bounds[0]+ball.w/2, bounds[1]-ball.w/2
    robot_low, robot_high = bounds[0]+player.w/2, bounds[1]-player.w/2
    if low >= high or robot_low >= robot_high:
        return _wait("目标尺寸超出可操作区域")
    head_y = player.y-player.h/2-ball.h/2
    catch_seconds = _arrival(ball, head_y, gravity)
    prediction_time = catch_seconds if catch_seconds is not None else latency+preview
    catch_x = _reflect(ball.x+ball.vx*prediction_time, low, high)
    catch_x = max(robot_low, min(robot_high, catch_x))
    catch_tolerance = max(0.0, (player.w+ball.w)/2-catch_margin)
    available = max(0.0, catch_seconds-latency) if catch_seconds is not None else None
    reachable = (None if lateral_speed is None or available is None
                 else abs(catch_x-player.x) <= lateral_speed*available+catch_tolerance)
    missed_plane = ball.vy > 0 and ball.y >= head_y
    urgent = missed_plane or (catch_seconds is not None and catch_seconds <= latency+urgent_seconds)
    if catch_seconds is None:
        risks.append("未观察到可预测的下落接触，暂用短时水平跟随")
    if gravity is None:
        risks.append("使用局部匀速预测，尚无测得的重力模型")
    if lateral_speed is None:
        risks.append("横向速度未测量，限制远距离收益目标")
    if reachable is False:
        risks.append("按当前测得速度可能赶不上乌贼，立即向落点移动")

    reliable = lambda kind: [e for e in obs.of_kind(kind) if e.confidence >= 0.8
                              and all(math.isfinite(v) for v in (e.x, e.y, e.vx, e.vy, e.w, e.h))]
    buff = bool(reliable("egg_buff"))
    buff_candidates = bool(obs.of_kind("egg_buff"))
    buff_bound = max(0.0, _finite(obs.metrics.get("buff_remaining_lower_bound_seconds"), 0.0))
    if buff and buff_bound == 0:
        risks.append("已见强化外观，但剩余有效时间未知，避免远距离主动撞敌")
    elif buff_candidates and not buff:
        risks.append("强化候选不可靠，按普通状态避敌")
    eggs, golden, enemies = reliable("egg_pickup"), reliable("golden_enemy"), reliable("enemy")
    mode = "powered" if buff else "normal_keep_ball"
    objective = None
    target_x = catch_x
    reason = "向乌贼预测落点移动，优先保球"

    def clamp_robot(x):
        return max(robot_low, min(robot_high, x))

    def hits_enemy_on_leg(start_x, end_x, start_time, end_time):
        """Local linear actor/enemy overlap across a horizontal travel leg."""
        duration = max(0.0, end_time-start_time)
        for enemy in enemies+golden:
            x0 = start_x-(enemy.x+enemy.vx*start_time)
            y0 = player.y-(enemy.y+enemy.vy*start_time)
            dx = end_x-start_x-enemy.vx*duration
            dy = -enemy.vy*duration
            lower, upper = 0.0, 1.0
            for start, delta, radius in ((x0, dx, (player.w+enemy.w)/2+catch_margin),
                                         (y0, dy, (player.h+enemy.h)/2)):
                if abs(delta) < 1e-12:
                    if abs(start) > radius:
                        lower, upper = 1, 0
                        break
                else:
                    a, b = sorted(((-radius-start)/delta, (radius-start)/delta))
                    lower, upper = max(lower, a), min(upper, b)
            if lower <= upper:
                return True
        return False

    def detour(e, powered=False):
        """Return (target x, ETA) only if visit AND return fit the catch window."""
        if urgent or reachable is False or catch_seconds is None:
            return None
        # A robot can only collect/contact objects at its body plane. Falling
        # eggs are projected to that plane; airborne enemies are not chased.
        vertical_reach = (player.h+e.h)/2
        enter, leave = latency, catch_seconds-urgent_seconds
        if e.kind == "egg_pickup" and abs(e.vy) > 1e-8:
            a, b = sorted(((player.y-vertical_reach-e.y)/e.vy,
                           (player.y+vertical_reach-e.y)/e.vy))
            enter, leave = max(enter, a), min(leave, b)
        elif abs(e.y-player.y) > vertical_reach:
            return None
        if enter > leave:
            return None
        arrival = max(0.0, enter)
        point_x = clamp_robot(_reflect(e.x+e.vx*arrival, bounds[0]+e.w/2, bounds[1]-e.w/2))
        if lateral_speed is None:
            # No speed model: opportunities already in the catch corridor only.
            if abs(point_x-catch_x) > catch_tolerance or abs(point_x-player.x) > catch_tolerance:
                return None
            return_time = 0.0
        else:
            # Iteratively intersect the horizontal travel requirement with the
            # object's actual body-plane contact window; do not add latency a
            # second time to a world-relative falling-egg arrival time.
            for _ in range(3):
                arrival = max(arrival, latency+abs(point_x-player.x)/lateral_speed)
                point_x = clamp_robot(_reflect(e.x+e.vx*arrival, bounds[0]+e.w/2, bounds[1]-e.w/2))
            if arrival > leave or abs(point_x-player.x) > lateral_speed*max(0, arrival-latency)+1e-8:
                return None
            return_time = abs(point_x-catch_x)/lateral_speed
            if arrival+return_time+urgent_seconds >= catch_seconds:
                return None
        if powered and buff_bound <= arrival+preview:
            return None
        if not buff and (hits_enemy_on_leg(player.x, point_x, latency, arrival)
                         or hits_enemy_on_leg(point_x, catch_x, arrival, arrival+return_time)):
            return None
        return point_x, arrival

    if not urgent and reachable is not False:
        egg_options = [(route[1], e, route) for e in eggs if (route := detour(e)) is not None]
        if egg_options and not buff:
            _, objective, route = min(egg_options, key=lambda candidate: candidate[0])
            target_x, _ = route
            mode, reason = "collect_egg", "先接可达鲑鱼卵，再返回乌贼落点"
        elif buff:
            enemy_options = [(route[1], e, route) for e in enemies
                             if (route := detour(e, powered=True)) is not None]
            if enemy_options:
                _, objective, route = min(enemy_options, key=lambda candidate: candidate[0])
                target_x, _ = route
                reason = "强化有效区间内接触可达地面鲑鱼，预留返程接球时间"
        if objective is None and golden and not buff:
            # Moving the robot into an airborne enemy does not itself stomp it.
            # Hold the catch point; the optional heading window below can
            # suggest a big bounce, without inventing an impact-direction law.
            objective = min(golden, key=lambda e: abs(e.x-catch_x))
            mode, reason = "seek_golden", "先站到乌贼下方，观察金色目标并准备下一次顶球"

    if urgent or reachable is False:
        target_x, objective = catch_x, None
        mode = "urgent_catch" if not buff else "powered_keep_ball"
        reason = ("乌贼已接近或越过头部接触平面，向其下方移动，当前预测不能保证接住"
                  if missed_plane else "乌贼即将落下，放弃收益目标并立即接球")
    elif buff and objective is None:
        mode = "powered_keep_ball"
        reason = "保持强化状态下的接球位置，兼顾空地鲑鱼的接触机会"

    # For ordinary enemies, choose the safer edge of the valid catch interval.
    # Never leave that interval just to avoid a fish and knowingly drop the squid.
    avoidance_horizon = min(prediction_time, max(preview, latency))
    def collision_risk(x):
        count = 0
        for e in enemies+golden:
            ex = _reflect(e.x+e.vx*avoidance_horizon, bounds[0]+e.w/2, bounds[1]-e.w/2)
            ey = e.y+e.vy*avoidance_horizon
            if abs(ey-player.y) <= (e.h+player.h)/2 and abs(ex-x) <= (e.w+player.w)/2+catch_margin:
                count += 1
        return count
    if not buff and objective is None:
        candidates = [target_x, clamp_robot(catch_x-catch_tolerance), clamp_robot(catch_x+catch_tolerance)]
        before = collision_risk(target_x)
        if before:
            target_x = min(candidates, key=lambda x: (collision_risk(x), abs(x-player.x)))
            risks.append("接球区域存在普通鲑鱼，优先保球并在接球范围内避让")
            if collision_risk(target_x) < before:
                reason += "；向接球范围内较安全的一侧调整"

    jump_window = _positive(parameters.get("jump_window_seconds"))
    jump = (parameters.get("allow_jump") is True and jump_window is not None
            and catch_seconds is not None and ball.vy > 0
            and latency < catch_seconds <= latency+jump_window
            and abs(player.x-catch_x) <= catch_tolerance)
    buttons = frozenset({Button.B}) if jump else frozenset()
    if jump:
        reason += "；接触窗口内建议 B 键顶球"
    state = ControllerState(buttons, StickVector(_axis(target_x-player.x, gain, deadzone), 0))
    intercepts = [{"kind": "ball", "track_id": ball.track_id, "x": catch_x,
                   "seconds": catch_seconds, "reachable": reachable,
                   "model": "ballistic" if gravity is not None and gravity > 0 else "local_linear"}]
    diagnostics = {
        "policy": "hopping_fast", "mode": mode,
        "experimental": parameters.get("calibrated") is not True,
        "target": {"x": target_x, "role": "robot"},
        "objective": ({"kind": objective.kind, "track_id": objective.track_id,
                       "x": objective.x, "y": objective.y} if objective else None),
        "intercepts": intercepts, "risks": risks, "buff_confirmed": buff,
        "ball_saved_first": True, "jump_suggested": jump,
        "latency_seconds": latency,
    }
    return Decision(state, "HOPPING："+reason, min(obs.confidence, player.confidence, ball.confidence), diagnostics)
