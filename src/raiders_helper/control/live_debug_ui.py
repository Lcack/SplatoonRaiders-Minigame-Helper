"""The familiar controller layout, sharing the live task's control owner.

This window does not construct a serial backend. Press/release bindings update
an in-memory relay; the task loop still owns all actual transport submissions.
Held inputs renew a 150 ms state lease every 50 ms, without neutral gaps.
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

from .buttons import Button, ControllerState, StickVector
from .debug_ui import BUTTON_LABELS, ControllerDebugApp, VirtualStick
from ..trace import state_dict


class LiveInputRelay:
    """Latest full state with a short lease; no serial transport is owned here."""
    def __init__(self, hub_provider, lease_ms=150, renew_seconds=0.05):
        if type(lease_ms) is not int or not 30 <= lease_ms <= 1000:
            raise ValueError("输入租约必须为30–1000毫秒")
        if not 0 < renew_seconds < lease_ms/1000:
            raise ValueError("续租间隔必须短于输入租约")
        self.hub_provider = hub_provider
        self.lease_ms = lease_ms
        self.renew_seconds = renew_seconds
        self.desired = ControllerState.neutral()
        self.message = "先在主窗口开始人工调试任务，再使用虚拟手柄。"
        self._hub = None
        self._owner_receipt = None
        self._last_release = None
        self._last_tick = None
        self._last_send = None
        self._last_sent_state = None
        self._owned = False

    def _current_hub(self):
        hub = self.hub_provider()
        if hub is not self._hub:
            old = self._hub
            if old is not None:
                try:
                    old.submit({"command": "release"})
                except (RuntimeError, PermissionError, ValueError):
                    pass
            self._hub = hub
            self._owner_receipt = self._last_release = None
            self._last_send = self._last_sent_state = None
            self._owned = False
        return hub

    def _stop(self, message):
        self.release(message)

    def request(self, state, now=None):
        """Called on ButtonPress/Release or drag; send changes immediately."""
        if not isinstance(state, ControllerState):
            raise ValueError("需要完整的ControllerState")
        now = time.perf_counter() if now is None else now
        hub = self._current_hub()
        if hub is None:
            self._stop("任务接口未连接，请先在主窗口开始任务。")
            return False
        if state == ControllerState.neutral():
            self.release()
            return True
        self.desired = state
        self._send_state(hub, now, explicit=True)
        return self.desired != ControllerState.neutral()

    def _send_state(self, hub, now, explicit=False):
        if self.desired == ControllerState.neutral():
            return
        try:
            if self._owner_receipt is not None:
                result = hub.command_result(self._owner_receipt)
                if not result or result["status"] in {"queued", "running"}:
                    self.message = "等待主任务暂停并交出输入权限……"
                    return
                self._owner_receipt = None
                if result["status"] != "succeeded":
                    self._stop(result.get("error", "输入权限交接失败"))
                    return
            if not hub.paused:
                if self._owned and not explicit:
                    self._stop("自动流程已恢复，旧手柄输入已取消；重新按下才会接管。")
                    return
                self._owned = False
                self._owner_receipt = hub.submit({"command": "pause"})["id"]
                self.message = "已请求暂停，等待输入权限。"
                return
            self._owned = True
            changed = self.desired != self._last_sent_state
            due = self._last_send is None or now-self._last_send >= self.renew_seconds
            if changed or due:
                hub.submit({"command": "state", **state_dict(self.desired), "hold_ms": self.lease_ms})
                self._last_send, self._last_sent_state = now, self.desired
                self.message = "按下即发送；按住连续保持，松开立即更新或归中。"
        except (RuntimeError, PermissionError, ValueError) as error:
            self._stop(str(error))

    def tick(self, now=None):
        now = time.perf_counter() if now is None else now
        hub = self.hub_provider()
        if hub is not self._hub:
            self.release("任务连接已变化，旧输入已清除。")
            self._current_hub()
            self._last_tick = now
            return
        if self._last_tick is not None and not 0 <= now-self._last_tick <= self.lease_ms/1000:
            self.release("窗口调度中断，已取消待发输入并归中。")
        self._last_tick = now
        if hub is not None:
            self._send_state(hub, now)

    def release(self, message="已请求释放全部按键和摇杆；自动流程保持暂停。"):
        self.desired = ControllerState.neutral()
        self._owner_receipt = None
        self._last_sent_state = None
        self.message = message
        hub = self._hub or self.hub_provider()
        if hub is None:
            return
        try:
            pending = hub.command_result(self._last_release) if self._last_release else None
            if not pending or pending["status"] not in {"queued", "running"}:
                self._last_release = hub.submit({"command": "release"})["id"]
            # A subsequent new press waits for this release to be applied.
            self._owner_receipt = self._last_release
        except (RuntimeError, PermissionError, ValueError):
            pass


class LiveDebugWindow(ControllerDebugApp):
    """Reuse ControllerDebugApp's layout helpers without its backend/sender."""
    def __init__(self, master, hub_provider):
        # Deliberately do not call ControllerDebugApp.__init__: that standalone
        # app creates its own connection controls and serial sender.
        self.root = tk.Toplevel(master)
        self.root.title("Raiders 实机手柄调试")
        self.root.geometry("860x730")
        self.root.minsize(820, 680)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _: self.release_all())
        self.root.bind("<FocusOut>", self._focus_out, add=True)
        self.hub_provider = hub_provider
        self.relay = LiveInputRelay(hub_provider)
        self.pressed_buttons = set()
        self.sticky_mode = False
        self.button_widgets = {}
        self.combo_widgets = {}
        self._closed = False
        self._resetting = False
        self._after = None
        self._last_status_at = 0.0
        self.status_var = tk.StringVar(value="共享主任务连接，不单独打开串口。")
        self.input_var = tk.StringVar(value="按键：无；左右摇杆归中")
        self.actual_var = tk.StringVar(value="尚无本任务实际发送记录")
        self._build_live()
        self._refresh()

    def _build_live(self):
        connection = ttk.LabelFrame(self.root, text="任务连接与输入权限", padding=6)
        connection.pack(fill="x", padx=6, pady=4)
        ttk.Label(connection, textvariable=self.status_var, wraplength=620).pack(side="left", fill="x", expand=True)
        ttk.Button(connection, text="暂停自动", command=self._pause).pack(side="left", padx=4)
        ttk.Button(connection, text="全部归中", command=self.release_all).pack(side="left", padx=4)
        ttk.Label(self.root, text="布局沿用原调试界面。按下即发送，按住连续保持；松手更新输入，失焦或关闭立即归中。",
                  wraplength=820).pack(fill="x", padx=8, pady=3)
        body = ttk.Frame(self.root)
        body.pack(fill="x", padx=6, pady=2)
        left = self._panel(body, "左摇杆", 0, 0)
        right = self._panel(body, "右摇杆", 0, 2)
        dpad = self._panel(body, "十字键", 1, 0)
        face = self._panel(body, "ABXY", 1, 2)
        center = self._panel(body, "肩键 / 系统键", 0, 1, rowspan=2)
        self.left_stick = VirtualStick(left, "左摇杆", lambda value: self._stick_changed("left", value))
        self.right_stick = VirtualStick(right, "右摇杆", lambda value: self._stick_changed("right", value))
        self.left_stick.place(relx=.5, rely=.5, anchor="center")
        self.right_stick.place(relx=.5, rely=.5, anchor="center")
        for stick in (self.left_stick, self.right_stick):
            stick.return_on_release.set(True)
            for child in stick.winfo_children():
                if isinstance(child, ttk.Checkbutton):
                    child.configure(state="disabled")
        self._square_button(left, "L3", 184, 184, 34, button=Button.L_STICK_PRESS)
        self._square_button(right, "R3", 184, 184, 34, button=Button.R_STICK_PRESS)
        self._place_dpad(dpad)
        self._place_face_buttons(face)
        self._place_shoulder_buttons(center)
        # Keep pointer release and focus loss authoritative; no sticky state.
        self.sticky_button.configure(text="松手释放", state="disabled")
        summary = ttk.LabelFrame(self.root, text="输入状态", padding=6)
        summary.pack(fill="both", expand=True, padx=6, pady=5)
        ttk.Label(summary, textvariable=self.input_var, wraplength=800).pack(fill="x")
        ttk.Label(summary, textvariable=self.actual_var, wraplength=800).pack(fill="x", pady=(4, 0))
        self.log = tk.Text(summary, height=3, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True, pady=(4, 0))

    def _pause(self):
        hub = self.hub_provider()
        if hub is None:
            self.log_message("先在主窗口开始任务。")
            return
        try:
            hub.submit({"command": "pause"})
        except (RuntimeError, PermissionError, ValueError) as error:
            self.log_message(str(error))

    def _button_down(self, button):
        self.pressed_buttons.add(button)
        self.log_message(f"{BUTTON_LABELS[button]} 按下")
        self._refresh_button_colors()
        self._push_state()

    def _button_up(self, button):
        self.pressed_buttons.discard(button)
        self._refresh_button_colors()
        self.log_message(f"{BUTTON_LABELS[button]} 松开")
        self._push_state()

    def _combo_down(self, buttons):
        self.pressed_buttons.update(buttons)
        self._refresh_button_colors()
        self.log_message("组合按下：" + "+".join(button.value for button in buttons))
        self._push_state()

    def _combo_up(self, buttons):
        self.pressed_buttons.difference_update(buttons)
        self._refresh_button_colors()
        self.log_message("组合松开：" + "+".join(button.value for button in buttons))
        self._push_state()

    def _stick_changed(self, side, value):
        if not self._resetting:
            self._push_state(side, value)

    def _push_state(self, side=None, value=None):
        left, right = self.left_stick.value, self.right_stick.value
        if side == "left" and value is not None:
            left = value
        if side == "right" and value is not None:
            right = value
        state = ControllerState(frozenset(self.pressed_buttons), left, right)
        self.relay.request(state)
        labels = "+".join(sorted(button.value for button in state.buttons)) or "无"
        self.input_var.set(f"界面按键：{labels}；左杆 ({left.x}, {left.y})；右杆 ({right.x}, {right.y})")

    def release_all(self):
        self.pressed_buttons.clear()
        self._refresh_button_colors()
        self._resetting = True
        try:
            self.left_stick.reset()
            self.right_stick.reset()
        finally:
            self._resetting = False
        self.relay.release()
        self.input_var.set("界面按键：无；左右摇杆归中")
        self.log_message("释放全部按键与摇杆；自动流程保持暂停")

    def _focus_out(self, _event=None):
        if not self._closed:
            self.root.after_idle(self._release_if_unfocused)

    def _release_if_unfocused(self):
        if self._closed:
            return
        focused = self.root.focus_displayof()
        if focused is None or focused.winfo_toplevel() is not self.root:
            self.release_all()

    def _refresh(self):
        if self._closed:
            return
        now = time.perf_counter()
        self.relay.tick(now)
        if now-self._last_status_at >= .10:
            self._last_status_at = now
            hub = self.hub_provider()
            if hub is None:
                self.status_var.set("任务未连接：在主窗口选择人工调试并开始任务。")
            else:
                status = hub.status().get("control", {})
                allowed = status.get("allowed", False)
                paused = status.get("paused", False)
                self.status_var.set(f"共享任务接口 · {'控制已启用' if allowed else '仅观察，未启用控制'} · "
                                    f"{'自动流程已暂停' if paused else '自动流程运行中'}\n{self.relay.message}")
                events = hub.events(limit=1)
                if events:
                    state = events[-1]["state"]
                    keys = "+".join(state["buttons"]) or "无"
                    self.actual_var.set(f"最近实际发送：{keys}；左杆 {state['left_stick']}；右杆 {state['right_stick']}")
        self._after = self.root.after(10, self._refresh)

    def show(self):
        if not self._closed:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_set()

    @property
    def closed(self):
        return self._closed

    def close(self):
        if self._closed:
            return
        self.release_all()
        self._closed = True
        if self._after is not None:
            self.root.after_cancel(self._after)
        self.root.destroy()
