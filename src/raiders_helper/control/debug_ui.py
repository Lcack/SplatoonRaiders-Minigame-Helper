from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable

from raiders_helper.control.backend import ControllerBackend
from raiders_helper.control.buttons import Button, ControllerState, StickVector
from raiders_helper.control.dummy import DummyBackend
from raiders_helper.control.easycon import EasyConBackend


BUTTON_LABELS = {
    Button.Y: "Y",
    Button.X: "X",
    Button.B: "B",
    Button.A: "A",
    Button.L: "L",
    Button.R: "R",
    Button.ZL: "ZL",
    Button.ZR: "ZR",
    Button.MINUS: "−",
    Button.PLUS: "+",
    Button.L_STICK_PRESS: "L摇杆按下",
    Button.R_STICK_PRESS: "R摇杆按下",
    Button.HOME: "Home",
    Button.CAPTURE: "截图",
    Button.DPAD_UP: "↑",
    Button.DPAD_DOWN: "↓",
    Button.DPAD_LEFT: "←",
    Button.DPAD_RIGHT: "→",
}


class StateSender:
    def __init__(
        self,
        backend: ControllerBackend,
        report_error: Callable[[str], None],
    ):
        self.backend = backend
        self.report_error = report_error
        self._queue: queue.Queue[ControllerState | None] = queue.Queue()
        self._last_error = ""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def send(self, state: ControllerState) -> None:
        self._queue.put(state)

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while True:
            state = self._queue.get()
            if state is None:
                return
            if not self.backend.send_state(state):
                error = getattr(self.backend, "last_error", "") or "发送状态失败"
                if error != self._last_error:
                    self._last_error = error
                    self.report_error(error)
            else:
                self._last_error = ""


class VirtualStick(tk.Frame):
    SIZE = 140
    RADIUS = 51

    def __init__(
        self,
        master,
        label: str,
        on_change: Callable[[StickVector], None],
    ):
        super().__init__(master)
        self._on_change = on_change
        self._dragging = False
        self._value = StickVector()
        self.return_on_release = tk.BooleanVar(value=True)
        self.value_label = ttk.Label(self, text=f"{label}: x=0, y=0")
        self.value_label.pack()
        self.canvas = tk.Canvas(
            self,
            width=self.SIZE,
            height=self.SIZE,
            background="#eeeeee",
            highlightthickness=0,
        )
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._move)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        ttk.Checkbutton(
            self,
            text="松手归中",
            variable=self.return_on_release,
        ).pack()
        self._draw()

    @property
    def value(self) -> StickVector:
        return self._value

    def reset(self) -> None:
        self._value = StickVector()
        self._draw()
        self._on_change(self._value)

    def _move(self, event) -> None:
        self._dragging = True
        center = self.SIZE / 2
        dx = event.x - center
        dy = event.y - center
        distance = (dx * dx + dy * dy) ** 0.5
        if distance > self.RADIUS:
            dx = dx / distance * self.RADIUS
            dy = dy / distance * self.RADIUS
        x = round(dx / self.RADIUS * 32767)
        y = round(-dy / self.RADIUS * 32767)
        self._value = StickVector(x=x, y=y)
        self._draw()
        self._on_change(self._value)

    def _release(self, event) -> None:
        del event
        if self._dragging and self.return_on_release.get():
            self.reset()
        self._dragging = False

    def _draw(self) -> None:
        self.canvas.delete("all")
        center = self.SIZE / 2
        self.canvas.create_oval(
            center - self.RADIUS,
            center - self.RADIUS,
            center + self.RADIUS,
            center + self.RADIUS,
            outline="#777777",
            width=2,
        )
        self.canvas.create_line(center, 10, center, self.SIZE - 10, fill="#cccccc")
        self.canvas.create_line(10, center, self.SIZE - 10, center, fill="#cccccc")
        knob_x = center + self._value.x / 32767 * self.RADIUS
        knob_y = center - self._value.y / 32767 * self.RADIUS
        self.canvas.create_oval(
            knob_x - 14,
            knob_y - 14,
            knob_x + 14,
            knob_y + 14,
            fill="#3778c2",
            outline="#173e6b",
            width=2,
        )
        self.value_label.configure(
            text=f"x={self._value.x}, y={self._value.y}"
        )


class ControllerDebugApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Raiders EasyCon 控制器调试")
        self.root.geometry("860x780")
        self.root.minsize(860, 740)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.backend: ControllerBackend | None = None
        self.sender: StateSender | None = None
        self.pressed_buttons: set[Button] = set()
        self.sticky_mode = False
        self.button_widgets: dict[Button, tk.Button] = {}
        self.combo_widgets: dict[tuple[Button, ...], tk.Button] = {}
        self._ui_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._build()
        self.root.after(50, self._drain_ui_queue)

    def _build(self) -> None:
        connection = ttk.LabelFrame(self.root, text="连接")
        connection.pack(fill="x", padx=6, pady=4)

        self.mode_var = tk.StringVar(value="EasyCon")
        mode = ttk.Combobox(
            connection,
            textvariable=self.mode_var,
            values=("EasyCon", "Dummy"),
            state="readonly",
            width=12,
        )
        mode.grid(row=0, column=0, padx=4, pady=4)
        mode.bind("<<ComboboxSelected>>", lambda _: self.refresh_ports())

        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(
            connection,
            textvariable=self.port_var,
            state="readonly",
            width=42,
        )
        self.port_combo.grid(row=0, column=1, padx=4, pady=4)
        ttk.Button(
            connection, text="刷新串口", command=self.refresh_ports
        ).grid(row=0, column=2, padx=4)
        self.connect_button = ttk.Button(
            connection, text="连接", command=self.connect
        )
        self.connect_button.grid(row=0, column=3, padx=4)
        ttk.Button(
            connection, text="断开", command=self.disconnect
        ).grid(row=0, column=4, padx=4)
        self.status_var = tk.StringVar(value="未连接")
        ttk.Label(connection, textvariable=self.status_var).grid(
            row=0, column=5, padx=8
        )

        body = ttk.Frame(self.root)
        body.pack(fill="x", expand=False, padx=6, pady=2)

        left_stick_panel = self._panel(body, "左摇杆", 0, 0)
        right_stick_panel = self._panel(body, "右摇杆", 0, 2)
        dpad_panel = self._panel(body, "十字键", 1, 0)
        face_panel = self._panel(body, "ABXY", 1, 2)
        center_panel = self._panel(body, "肩键 / 系统键", 0, 1, rowspan=2)

        self.left_stick = VirtualStick(
            left_stick_panel,
            "左摇杆",
            lambda value: self._stick_changed("left", value),
        )
        self.left_stick.place(relx=0.5, rely=0.5, anchor="center")
        self.right_stick = VirtualStick(
            right_stick_panel,
            "右摇杆",
            lambda value: self._stick_changed("right", value),
        )
        self.right_stick.place(relx=0.5, rely=0.5, anchor="center")
        self._square_button(
            left_stick_panel,
            "L3",
            184,
            184,
            34,
            button=Button.L_STICK_PRESS,
        )
        self._square_button(
            right_stick_panel,
            "R3",
            184,
            184,
            34,
            button=Button.R_STICK_PRESS,
        )

        self._place_dpad(dpad_panel)
        self._place_face_buttons(face_panel)
        self._place_shoulder_buttons(center_panel)

        log_frame = ttk.LabelFrame(self.root, text="协议日志")
        log_frame.pack(fill="both", expand=True, padx=6, pady=(2, 6))
        self.log = tk.Text(log_frame, height=6, state="disabled")
        self.log.pack(fill="both", expand=True)
        self.refresh_ports()

    def _panel(
        self,
        parent,
        title: str,
        row: int,
        column: int,
        rowspan: int = 1,
    ) -> tk.Frame:
        width = 310 if column == 1 else 230
        height = 466 if rowspan == 2 else 230
        panel = tk.Frame(
            parent,
            width=width,
            height=height,
            borderwidth=1,
            relief="solid",
        )
        panel.grid(
            row=row,
            column=column,
            rowspan=rowspan,
            padx=3,
            pady=3,
            sticky="nsew",
        )
        panel.grid_propagate(False)
        ttk.Label(panel, text=title).place(relx=0.5, y=4, anchor="n")
        return panel

    def _square_button(
        self,
        parent,
        label: str,
        x: int,
        y: int,
        size: int,
        button: Button | None = None,
        combo: tuple[Button, ...] | None = None,
        box_width: int | None = None,
    ) -> tk.Button:
        widget = tk.Button(
            parent,
            text=label,
            width=0,
            height=0,
            font=("Segoe UI", 9, "bold"),
            takefocus=0,
            relief="raised",
            borderwidth=2,
            wraplength=max(20, (box_width or size) - 8),
            justify="center",
        )
        widget.place(x=x, y=y, width=box_width or size, height=size)
        if button is not None:
            widget.bind(
                "<ButtonPress-1>",
                lambda event, value=button: self._button_down(value),
            )
            widget.bind(
                "<ButtonRelease-1>",
                lambda event, value=button: self._button_up(value),
            )
            self.button_widgets[button] = widget
        if combo is not None:
            key = tuple(combo)
            widget.bind(
                "<ButtonPress-1>",
                lambda event, values=key: self._combo_down(values),
            )
            widget.bind(
                "<ButtonRelease-1>",
                lambda event, values=key: self._combo_up(values),
            )
            self.combo_widgets[key] = widget
        return widget

    def _place_dpad(self, panel: tk.Frame) -> None:
        size = 44
        self._square_button(
            panel, "↑", 93, 50, size, button=Button.DPAD_UP
        )
        self._square_button(
            panel, "←", 43, 98, size, button=Button.DPAD_LEFT
        )
        self._square_button(
            panel, "→", 143, 98, size, button=Button.DPAD_RIGHT
        )
        self._square_button(
            panel, "↓", 93, 146, size, button=Button.DPAD_DOWN
        )

    def _place_face_buttons(self, panel: tk.Frame) -> None:
        size = 44
        self._square_button(panel, "X", 93, 50, size, button=Button.X)
        self._square_button(panel, "Y", 43, 98, size, button=Button.Y)
        self._square_button(panel, "A", 143, 98, size, button=Button.A)
        self._square_button(panel, "B", 93, 146, size, button=Button.B)

    def _place_shoulder_buttons(self, panel: tk.Frame) -> None:
        size = 52
        combo_width = 90
        x_positions = (50, 110, 208)
        y_positions = (86, 150, 214, 278, 342)
        self._square_button(
            panel, "ZL", x_positions[0], y_positions[0], size, button=Button.ZL
        )
        self._square_button(
            panel,
            "ZL+ZR",
            x_positions[1],
            y_positions[0],
            size,
            combo=(Button.ZL, Button.ZR),
            box_width=combo_width,
        )
        self._square_button(
            panel, "ZR", x_positions[2], y_positions[0], size, button=Button.ZR
        )
        self._square_button(
            panel, "L", x_positions[0], y_positions[1], size, button=Button.L
        )
        self._square_button(
            panel,
            "L+R",
            x_positions[1],
            y_positions[1],
            size,
            combo=(Button.L, Button.R),
            box_width=combo_width,
        )
        self._square_button(
            panel, "R", x_positions[2], y_positions[1], size, button=Button.R
        )
        self._square_button(
            panel,
            "截图",
            97,
            y_positions[2],
            size,
            button=Button.CAPTURE,
        )
        self._square_button(
            panel,
            "HOME",
            161,
            y_positions[2],
            size,
            button=Button.HOME,
        )
        self._square_button(
            panel,
            "−",
            97,
            y_positions[3],
            size,
            button=Button.MINUS,
        )
        self._square_button(
            panel,
            "+",
            161,
            y_positions[3],
            size,
            button=Button.PLUS,
        )
        ttk.Button(
            panel,
            text="回中",
            command=self.release_all,
        ).place(x=97, y=y_positions[4], width=size, height=size)
        self.sticky_button = tk.Button(
            panel,
            text="粘滞键",
            font=("Segoe UI", 9, "bold"),
            takefocus=0,
            relief="raised",
            borderwidth=1,
            command=self._toggle_sticky_mode,
        )
        self.sticky_button.place(x=161, y=y_positions[4], width=size, height=size)

    def refresh_ports(self) -> None:
        if self.mode_var.get() == "Dummy":
            self.port_combo.configure(values=("Dummy",))
            self.port_var.set("Dummy")
            return
        try:
            ports = EasyConBackend.list_ports()
        except Exception as error:
            self.log_message(f"枚举串口失败: {error}")
            return
        values = tuple(f"{port.device} | {port.description}" for port in ports)
        self.port_combo.configure(values=values)
        if values and not self.port_var.get():
            self.port_var.set(values[0])
        if not values:
            self.port_var.set("")

    def connect(self) -> None:
        self.disconnect()
        if self.mode_var.get() == "Dummy":
            self.backend = DummyBackend()
            connected = self.backend.connect()
            label = "Dummy"
        else:
            port = self.port_var.get().split("|", 1)[0].strip()
            self.backend = EasyConBackend()
            connected = self.backend.connect(port)
            label = port
        if not connected:
            error = getattr(self.backend, "last_error", "") or "未知错误"
            self.backend = None
            self.status_var.set(f"连接失败: {error}")
            self.log_message(f"连接失败: {error}")
            return
        self.status_var.set(f"已连接 {label}")
        self.sender = StateSender(self.backend, self._report_error)
        self.log_message(f"已连接 {label}")

    def disconnect(self) -> None:
        if self.sender is not None:
            self.sender.stop()
            self.sender = None
        if self.backend is not None:
            try:
                self.backend.disconnect()
            finally:
                self.backend = None
        self.pressed_buttons.clear()
        self._refresh_button_colors()
        self.status_var.set("未连接")

    def _button_down(self, button: Button) -> None:
        if button in self.pressed_buttons:
            if not self.sticky_mode:
                return
            self.pressed_buttons.remove(button)
            self.log_message(f"{BUTTON_LABELS[button]} 粘滞释放")
        else:
            self.pressed_buttons.add(button)
            self.log_message(
                f"{BUTTON_LABELS[button]} "
                f"{'粘滞按下' if self.sticky_mode else '按下'}"
            )
        self._refresh_button_colors()
        self._push_state()

    def _button_up(self, button: Button) -> None:
        if self.sticky_mode:
            return
        if button not in self.pressed_buttons:
            return
        self.pressed_buttons.remove(button)
        self._refresh_button_colors()
        self.log_message(f"{BUTTON_LABELS[button]} 释放")
        self._push_state()

    def _combo_down(self, buttons: tuple[Button, ...]) -> None:
        active = all(button in self.pressed_buttons for button in buttons)
        if self.sticky_mode and active:
            self.pressed_buttons.difference_update(buttons)
            state_label = "粘滞释放"
        else:
            self.pressed_buttons.update(buttons)
            state_label = "粘滞按下" if self.sticky_mode else "按下"
        self._refresh_button_colors()
        self.log_message(
            f"组合{state_label}: "
            + " + ".join(button.value for button in buttons)
        )
        self._push_state()

    def _combo_up(self, buttons: tuple[Button, ...]) -> None:
        if self.sticky_mode:
            return
        self.pressed_buttons.difference_update(buttons)
        self._refresh_button_colors()
        self.log_message("组合释放: " + " + ".join(button.value for button in buttons))
        self._push_state()

    def _toggle_sticky_mode(self) -> None:
        self.sticky_mode = not self.sticky_mode
        if self.sticky_mode:
            self.sticky_button.configure(
                background="#3778c2",
                foreground="white",
                relief="sunken",
            )
            self.log_message("粘滞键模式开启")
        else:
            self.sticky_button.configure(
                background="SystemButtonFace",
                foreground="black",
                relief="raised",
            )
            self.log_message("粘滞键模式关闭，执行回中")
            self.release_all()

    def _refresh_button_colors(self) -> None:
        for button, widget in self.button_widgets.items():
            widget.configure(
                background="#3778c2" if button in self.pressed_buttons else "SystemButtonFace",
                foreground="white" if button in self.pressed_buttons else "black",
            )
        for buttons, widget in self.combo_widgets.items():
            active = all(button in self.pressed_buttons for button in buttons)
            widget.configure(
                background="#3778c2" if active else "SystemButtonFace",
                foreground="white" if active else "black",
            )

    def _stick_changed(self, side: str, value: StickVector) -> None:
        self._push_state(side=side, value=value)

    def _push_state(self, side: str | None = None, value: StickVector | None = None) -> None:
        if self.sender is None:
            return
        left = self.left_stick.value
        right = self.right_stick.value
        if side == "left" and value is not None:
            left = value
        if side == "right" and value is not None:
            right = value
        self.sender.send(
            ControllerState(
                buttons=frozenset(self.pressed_buttons),
                left_stick=left,
                right_stick=right,
            )
        )

    def release_all(self) -> None:
        self.pressed_buttons.clear()
        self._refresh_button_colors()
        self.left_stick.reset()
        self.right_stick.reset()
        self._push_state()
        self.log_message("释放全部按键，摇杆归中")

    def _report_error(self, message: str) -> None:
        self._ui_queue.put(lambda: self.log_message(f"错误: {message}"))

    def _drain_ui_queue(self) -> None:
        while True:
            try:
                callback = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            callback()
        self.root.after(50, self._drain_ui_queue)

    def log_message(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def close(self) -> None:
        try:
            self.release_all()
            self.disconnect()
        finally:
            self.root.destroy()


def run_debug_ui() -> int:
    root = tk.Tk()
    ControllerDebugApp(root)
    root.mainloop()
    return 0

