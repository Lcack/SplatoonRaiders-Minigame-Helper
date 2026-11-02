"""Observed menu/game/result navigation, with optional repeat-round sampling.

This module never opens a controller. Button defaults are unverified control
hypotheses; hardware use needs a separately calibrated session profile. Every
button pulse is tied to the screen that justified it, with bounded retries.
"""
import math
import time

from .control.buttons import Button, ControllerState
from .models import Decision, Game


DEFAULT_ORDER = tuple(game.value for game in Game)
DEFAULT_BUTTONS = {
    "select": ("A",),
    "start": ("A",),
    "result_choose_quit": ("DPAD_DOWN",),
    "result_confirm_quit": ("A",),
    "result_choose_retry": ("DPAD_UP",),
    "result_confirm_retry": ("A",),
}


def _positive(config, key, default):
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} 必须为有限正数")
    return float(value)


def _buttons(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("菜单动作 buttons 必须为非空按键列表")
    return frozenset(Button(value) for value in values)


class SessionFlow:
    """Pure menu policy for a selected game, optionally sampling repeat rounds.

    Screens are menu:<game>, countdown:<game>, optional title:<game>,
    playing:<game>, result:retry, result:quit, or unknown. `menu_only` ends
    with the desired game highlighted on the selection menu, without A.
    The observed reference goes straight from menu confirmation to countdown;
    countdown always releases input and never sends a second start press.
    routes can override one navigation step, for example:
      {"mining->landing": {"buttons": ["DPAD_DOWN"], "expect": "menu:hopping"}}
    The source and target in this key are observed focus and desired game.
    ``expect`` records the expected next focus for logging; a different known
    menu focus is re-observed and replanned, never assumed to be that focus.
    end_behavior='retry' repeats at results; max_rounds defaults to 3, while
    an explicit 0 removes that round limit. stop_after_round(reason) requests
    stopping on a result screen. Neither a round limit nor completion of this
    flow is evidence that a fitted motion model has passed validation.
    """

    def __init__(self, game, config=None, menu_only=False, end_behavior="quit", max_rounds=None):
        self.game = Game(game).value
        self.config = config or {}
        self.menu_only = bool(menu_only)
        if end_behavior not in ("quit", "retry"):
            raise ValueError("end_behavior 必须为 quit 或 retry")
        self.end_behavior = end_behavior
        self.max_rounds = self.config.get("max_rounds", 3) if max_rounds is None else max_rounds
        if isinstance(self.max_rounds, bool) or not isinstance(self.max_rounds, int) or self.max_rounds < 0:
            raise ValueError("max_rounds 必须为非负整数；0 表示显式不限轮数")
        self.started_rounds = self.completed_rounds = 0
        self.completion_reason = ""
        self._stop_reason = ""
        self._events = []
        self.hold_seconds = _positive(self.config, "hold_seconds", 0.08)
        self.gap_seconds = _positive(self.config, "gap_seconds", 0.25)
        self.settle_seconds = _positive(self.config, "screen_settle_seconds", 0.05)
        self.step_timeout = _positive(self.config, "step_timeout_seconds", 15)
        self.unknown_timeout = _positive(self.config, "unknown_timeout_seconds", 10)
        self.max_play_seconds = _positive(self.config, "max_play_seconds", 1800)
        self.max_attempts = self.config.get("max_attempts", 3)
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("max_attempts 必须为正整数")
        self.order = tuple(self.config.get("menu_order", DEFAULT_ORDER))
        if len(self.order) != 4 or set(self.order) != set(DEFAULT_ORDER):
            raise ValueError("menu_order 必须恰好包含四个小游戏且不重复")
        button_config = self.config.get("buttons", {})
        self.buttons = {key: _buttons(button_config.get(key, value)) for key, value in DEFAULT_BUTTONS.items()}
        self.routes = {}
        for key, route in self.config.get("routes", {}).items():
            try:
                source, target = key.split("->")
                expected = route["expect"]
                if source not in self.order or target not in self.order or source == target:
                    raise ValueError
                if expected not in {f"menu:{name}" for name in self.order if name != source}:
                    raise ValueError
                self.routes[key] = (_buttons(route["buttons"]), expected)
            except (AttributeError, KeyError, TypeError, ValueError):
                raise ValueError(f"无效单步菜单路线: {key}") from None
        self.done = self.failed = False
        self.failure = ""
        self.stage = "await_menu"
        self.screen = "unknown"
        self._last_time = self._stage_since = self._stable_since = None
        self._last_screen = None
        self._unknown_since = None
        self._selected_sent = self._seen_playing = False
        self._round_active = self._retry_sent = False
        self._step = None
        self._step_since = self._hold_until = self._next_press_at = 0.0
        self._attempts = 0
        self._release_pending = False

    def _stage(self, stage, now):
        if self.stage != stage:
            self.stage, self._stage_since = stage, now
            self._step = None
            self._hold_until = 0.0
            self._release_pending = False

    def suspend(self, duration):
        """Exclude a software debugging pause from automation deadlines.

        This does not pause the console game. Resume always requires a new
        stable visual observation and cancels any pre-pause button pulse.
        """
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise ValueError("暂停时长必须为有限非负数")
        for name in ("_last_time", "_stage_since", "_stable_since", "_unknown_since", "_step_since", "_next_press_at"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value+duration)
        self._hold_until = 0.0
        self._release_pending = False
        self._last_screen = None

    def _out(self, reason, buttons=frozenset(), game_decision=None):
        state = game_decision.state if game_decision else ControllerState(buttons)
        diagnostics = dict(game_decision.diagnostics) if game_decision else {}
        diagnostics["session"] = {
            "stage": self.stage, "screen": self.screen, "game": self.game,
            "done": self.done, "failed": self.failed, "failure": self.failure,
            "attempt": self._attempts, "expected": self._step[2] if self._step else None,
            "menu_only": self.menu_only,
            "end_behavior": self.end_behavior,
            "started_rounds": self.started_rounds, "completed_rounds": self.completed_rounds,
            "max_rounds": self.max_rounds, "stop_requested": self._stop_reason,
            "completion_reason": self.completion_reason,
            "events": [dict(event) for event in self._events],
            "experimental": self.config.get("calibrated") is not True,
        }
        return Decision(state, reason, game_decision.confidence if game_decision else 0.0, diagnostics)

    def _fail(self, reason, now):
        self.failed, self.failure = True, reason
        self._stage("failed", now)
        return self._out("会话已停止：" + reason)

    def stop_after_round(self, reason):
        """Ask to stop at a result; the caller owns fit/validation criteria."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("stop_after_round 需要说明停止原因")
        if not self.done and not self.failed:
            self._stop_reason = reason.strip()

    def _finish(self, reason, now):
        self.done, self.completion_reason = True, reason
        self._stage("done", now)
        return self._out(reason)

    def _pulse(self, action, buttons, expected, now):
        step = (self.screen, action, expected)
        if step != self._step:
            self._step, self._step_since = step, now
            self._attempts = 0
            # A new screen must have been released and settled before here.
            self._hold_until = 0.0
        if now-self._step_since >= self.step_timeout:
            return self._fail("菜单步骤等待超时", now)
        if now < self._hold_until:
            return self._out(f"菜单：{action}（短按中）", buttons)
        if self._release_pending:
            # Even after a scheduling stall, expose at least one neutral state
            # before another pulse. A later pulse cannot stand in for release.
            self._release_pending = False
            return self._out("菜单：短按结束，松开输入")
        if now < self._next_press_at:
            return self._out("菜单：已松开，等待画面反馈")
        if self._attempts >= self.max_attempts:
            return self._fail("菜单按键重试次数用尽，未确认预期画面", now)
        self._attempts += 1
        self._hold_until = now+self.hold_seconds
        self._next_press_at = self._hold_until+self.gap_seconds
        self._release_pending = True
        if action == "select":
            self._selected_sent = True
        elif action == "result_confirm_retry":
            self._retry_sent = True
        return self._out(f"菜单：{action}（第 {self._attempts} 次）", buttons)

    def update(self, screen, game_decision=None, now=None):
        self._events = []
        if self.done or self.failed:
            return self._out("会话完成，保持中立" if self.done else "会话已停止："+self.failure)
        now = time.perf_counter() if now is None else now
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            return self._fail("时钟不是有限数值", self._last_time or 0.0)
        if self._last_time is not None and now < self._last_time:
            return self._fail("时钟倒退", now)
        self._last_time = now
        if self._stage_since is None:
            self._stage_since = now
        limit = self.max_play_seconds if self.stage == "playing" else self.step_timeout
        if now-self._stage_since >= limit:
            return self._fail("游戏运行超时" if self.stage == "playing" else "会话阶段等待超时", now)

        self.screen = screen if isinstance(screen, str) else "unknown"
        changed = self.screen != self._last_screen
        if changed:
            self._last_screen, self._stable_since = self.screen, now
            self._hold_until = 0.0  # Never carry a pulse onto another screen.
            self._release_pending = False
        prefix, _, value = self.screen.partition(":")
        valid = ((prefix == "menu" and value in self.order)
                 or (prefix in ("title", "countdown", "playing") and value in self.order)
                 or self.screen in ("result:retry", "result:quit"))
        if not valid:
            if self._unknown_since is None:
                self._unknown_since = now
            if now-self._unknown_since >= self.unknown_timeout:
                return self._fail("未知画面持续超时", now)
            return self._out("会话：未知画面，松开输入等待识别")
        self._unknown_since = None

        if prefix in ("title", "countdown", "playing") and value != self.game:
            return self._fail("识别到其他小游戏，禁止继续按键", now)
        if prefix == "menu":
            if self.stage == "returning":
                return self._finish("已回到四游戏选择页，单次任务完成", now)
            if self._seen_playing:
                return self._fail("未经过结算确认就出现选择页，停止本次任务", now)
            self._stage("selecting", now)
            if self.menu_only and value == self.game:
                return self._finish("目标游戏已高亮，菜单测试完成；未确认开局", now)
        elif prefix in ("title", "countdown"):
            if self.menu_only:
                return self._fail("菜单测试意外离开选择页，停止输入", now)
            if (not self._selected_sent or self._round_active
                    or (self._seen_playing and not self._retry_sent)):
                return self._fail("标题或倒数画面不符合当前菜单进入流程", now)
            self._stage("starting", now)
        elif prefix == "playing":
            if self.menu_only:
                return self._fail("菜单测试已进入游戏，停止输入", now)
            if (not self._selected_sent or self.stage == "returning"
                    or (self._seen_playing and not self._round_active and not self._retry_sent)):
                return self._fail("运行画面不符合当前会话流程", now)
            if not self._round_active:
                self.started_rounds += 1
                self._events.append({"type": "round_started", "round": self.started_rounds,
                                     "timestamp": now, "screen": self.screen})
                self._round_active, self._retry_sent = True, False
            self._seen_playing = True
            self._stage("playing", now)
        else:
            if not self._seen_playing:
                return self._fail("尚未确认进入游戏，不能操作结果页", now)
            if self._round_active:
                self.completed_rounds += 1
                self._round_active = False
                self._events.append({"type": "round_completed", "round": self.completed_rounds,
                                     "timestamp": now, "screen": self.screen})
            elif self.stage == "starting":
                return self._fail("重试倒数后未确认开局就出现结果页", now)
            if self._stop_reason:
                if self._retry_sent:
                    # The retry confirmation may already be queued in the
                    # console even while its old result screen is still seen.
                    # Do not report a successful stop before that round ends.
                    return self._out("已确认重试，等待下一轮结束后按外部请求停手")
                return self._finish("本轮结束，按外部请求停手："+self._stop_reason, now)
            if self.end_behavior == "retry":
                if self.max_rounds and self.completed_rounds >= self.max_rounds:
                    return self._finish(f"已完成 {self.completed_rounds} 轮采样，在结果页停手；尚不代表拟合通过", now)
                self._stage("retrying", now)
            else:
                self._stage("returning", now)

        if changed or now-self._stable_since < self.settle_seconds:
            return self._out("会话：画面变化，已松开并等待稳定")
        if prefix == "menu":
            if value == self.game:
                return self._pulse("select", self.buttons["select"], f"countdown:{self.game}|playing:{self.game}", now)
            route = self.routes.get(f"{value}->{self.game}")
            if route is None:
                offset = 1 if self.order.index(value) < self.order.index(self.game) else -1
                next_game = self.order[self.order.index(value)+offset]
                route = (frozenset({Button.DPAD_DOWN if offset > 0 else Button.DPAD_UP}), f"menu:{next_game}")
            return self._pulse("navigate", route[0], route[1], now)
        if prefix == "title":
            return self._pulse("start", self.buttons["start"], f"playing:{self.game}", now)
        if prefix == "countdown":
            return self._out("会话：等待开局倒数，保持中立")
        if prefix == "playing":
            return self._out(game_decision.reason, game_decision=game_decision) if game_decision else self._out("会话：等待游戏策略建议")
        if self.end_behavior == "retry":
            action = "result_confirm_retry" if value == "retry" else "result_choose_retry"
            expected = f"countdown:{self.game}|playing:{self.game}" if value == "retry" else "result:retry"
        else:
            action = "result_choose_quit" if value == "retry" else "result_confirm_quit"
            expected = "result:quit" if value == "retry" else "menu:any"
        return self._pulse(action, self.buttons[action], expected, now)
