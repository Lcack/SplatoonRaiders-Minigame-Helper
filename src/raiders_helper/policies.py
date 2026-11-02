"""Experimental baselines, not validated game solvers. See docs/RESEARCH.md."""
from .control.buttons import Button, ControllerState, StickVector
from .models import Decision, Game


def axis(delta):
    return round(max(-1.0, min(1.0, delta * 6)) * 32767) if abs(delta) > 0.015 else 0


def intercept(ball, player):
    # Local constant-velocity extrapolation. Gravity/bounces are not fitted yet.
    dt = max(0.0, min(0.5, (player.y - ball.y) / ball.vy)) if ball.vy > 0.03 else 0.1
    return max(0.03, min(0.97, ball.x + ball.vx * dt)), dt


def decide(obs, parameters=None):
    parameters = parameters or {}
    if obs.phase != "playing" or obs.confidence < 0.8:
        return Decision(reason="画面状态或目标尚未可靠识别")
    if obs.metrics.get("time_remaining", 1) <= 0:
        return Decision(reason="倒计时已结束，释放控制并等待结算画面")
    players = obs.of_kind("player")
    if len(players) != 1:
        return Decision(reason="玩家目标缺失或不唯一")
    player = players[0]
    if obs.game == Game.JUGGLING and parameters.get("algorithm") != "baseline":
        from .juggling import decide_juggling
        return decide_juggling(obs, parameters)
    if obs.game == Game.HOPPING and parameters.get("algorithm") != "baseline":
        from .hopping import decide_hopping
        return decide_hopping(obs, parameters)
    if obs.game == Game.LANDING:
        from .landing import decide_landing
        return decide_landing(obs, parameters)
    if obs.game == Game.MINING:
        if parameters.get("algorithm") == "reactive_supervised":
            from .mining_reactive import decide_mining_reactive
            return decide_mining_reactive(obs, parameters)
        if parameters.get("algorithm") == "lookahead":
            from .mining_planner import planning_decision
            return planning_decision(obs, parameters)
        corridors = obs.of_kind("corridor")
        if not corridors:
            return Decision(reason="MINING：尚未识别道路边界，等待通道标定")
        # Stay within road intervals at every configured lookahead row.
        # Walls remain forbidden even with the egg buff active.
        rows = sorted({round(e.y, 4) for e in corridors})
        margin = player.w / 2 + 0.025
        lanes = [i / 40 for i in range(2, 39)
                 if all(any(round(e.y, 4) == row and e.x - e.w / 2 + margin <= i / 40 <= e.x + e.w / 2 - margin
                            for e in corridors) for row in rows)]
        if not lanes:
            return Decision(reason="MINING：前方通道没有可靠的直行走廊，松开加速等待重识别")
        hazards = obs.of_kind("obstacle") + obs.of_kind("wall")
        items = obs.of_kind("reward")
        powered = bool(obs.of_kind("egg_buff"))
        if powered:
            items += obs.of_kind("obstacle")
        def cost(x):
            risk = sum(5 for e in hazards if (e.kind == "wall" or not powered) and abs(x - e.x) < (e.w + player.w) / 2 + 0.05
                       and player.y - 0.08 < e.y < player.y + 0.4)
            reward = min((abs(x - e.x) for e in items if e.y > player.y), default=0.3)
            return risk + abs(x - player.x) + reward * 0.3
        target = min(lanes, key=cost)
        danger = any((e.kind == "wall" or not powered) and abs(e.x - player.x) < (e.w + player.w) / 2 + 0.08
                     and player.y - 0.08 < e.y < player.y + 0.3 for e in hazards)
        aligned = abs(target - player.x) < 0.08
        buttons = frozenset() if danger or not aligned else frozenset({Button.ZR})
        return Decision(ControllerState(buttons, StickVector(axis(target - player.x), 0)),
                        "原型：墙为硬约束；" + ("强化中允许碎石通过" if powered else "优先避石、转向时松开加速"), obs.confidence)
    if obs.game == Game.HOPPING:
        balls = obs.of_kind("ball")
        if len(balls) != 1 or balls[0].samples < 2:
            return Decision(reason="等待乌贼的连续运动轨迹")
        target, dt = intercept(balls[0], player)
        # Jump timing remains disabled until a user-calibrated threshold is supplied.
        jump_window = parameters.get("jump_window_seconds", 0)
        buttons = frozenset({Button.B}) if 0 < dt < jump_window else frozenset()
        return Decision(ControllerState(buttons, StickVector(axis(target - player.x), 0)),
                        "原型：向乌贼预测落点移动", obs.confidence)
    if obs.game == Game.JUGGLING:
        partners = obs.of_kind("partner")
        balls = [b for b in obs.of_kind("ball") if b.samples >= 2 and b.vy > 0.03]
        if len(partners) != 1 or not balls:
            return Decision(reason="等待双角色与下落目标的轨迹")
        partner = partners[0]
        split = parameters.get("split_x", 0.5)
        # Public footage shows a central divider. Partition target assignments
        # before choosing the soonest intercept; never send both to one side.
        def target_axis(actor, left_side):
            candidates = [(x, dt) for ball in balls for x, dt in [intercept(ball, actor)]
                          if (x < split) == left_side]
            return axis(min(candidates, key=lambda p: p[1])[0] - actor.x) if candidates else 0
        left, right = target_axis(player, True), target_axis(partner, False)
        return Decision(ControllerState(left_stick=StickVector(left, 0), right_stick=StickVector(right, 0)),
                        "原型：优先接住下落目标，分配给两个摇杆", obs.confidence)
    return Decision(reason="LANDING：已接入观测；推力、倾角和制动模型尚未标定")
