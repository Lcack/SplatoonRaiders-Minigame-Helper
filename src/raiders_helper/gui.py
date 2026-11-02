import json
import hashlib
from collections import deque
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .capture.models import CaptureConfig, FramePacket
from .capture.recorder import VideoRecorder
from .capture.video_source import OpenCvVideoSource
from .capture.worker import CaptureWorker
from .control.easycon import EasyConBackend
from .control.guard import GuardedController
from .demo import demo_packet, demo_profile
from .models import Game
from .runtime import Engine
from .trace import TraceWriter, state_dict
from .vision import create_recognizer, load_profile
from .analysis_worker import AnalysisWorker
from .advice_overlay import advice_geometry
from .active_ai_status import ActiveAIMonitor, snapshot_loaded_terminal_sources
# Freeze this terminal's already imported input/display owner separately from
# the external, per-round AI child. These imports do not open any devices.
from . import session_runtime as _terminal_session_runtime
from . import manual_input_pump as _terminal_manual_pump
from . import debug_api as _terminal_debug_api
from . import experiment_recording as _terminal_experiment_recording
from . import mining_vision as _terminal_preview_vision
from . import arcade_vision as _terminal_preview_arcade


_TERMINAL_SOURCE_HASHES = snapshot_loaded_terminal_sources()
_PREVIEW_VISION_SHA = hashlib.sha256(Path(_terminal_preview_vision.__file__).read_bytes()).hexdigest()
_PREVIEW_ARCADE_SHA = hashlib.sha256(Path(_terminal_preview_arcade.__file__).read_bytes()).hexdigest()


class Dashboard:
    TASK_AUTO_MODE="自主 AI（局间更新）"
    TASK_INTERFACE_MODE="采集预览 / 助手接口"
    TASK_MODE_HELP = {
        "自主 AI（局间更新）": "MINING/JUGGLING 使用当前实机核心自主运行，局间重新加载；运行轮数0表示持续Retry。",
        "采集预览 / 助手接口": "连接采集与控制接口，等待独立AI程序或助手接管。",
        "手柄调试（人工）": "由独立手柄调试窗口操作，不运行自主策略。",
        "仅定位菜单": "从四游戏选择页把光标移动到所选游戏后停手，不确认开局；需要已验证的菜单配置。",
        "Retry 参数标定": "运行短脉冲探针并逐轮 Retry；达到轮数或横向模型验证门槛后停手。0 表示持续采样，Landing 耦合模型尚未完成。",
    }
    TASK_MODE_BACKENDS={TASK_AUTO_MODE:'manual',TASK_INTERFACE_MODE:'manual','手柄调试（人工）':'manual',
        '人工调试':'manual','仅定位菜单':'menu','单局后返回':'single','Retry 参数标定':'calibrate'}

    def __init__(self, root, video_path=None, task_terminal=False):
        self.root = root
        self.task_terminal = task_terminal
        root.title("Raiders 小游戏任务终端" if task_terminal else "Raiders 小游戏助手 · 框架预览")
        if task_terminal:
            width=min(1450,root.winfo_screenwidth()-60)
            height=min(960,root.winfo_screenheight()-90)
            root.geometry(f"{width}x{height}")
            root.minsize(min(1100,width),min(760,height))
        else:
            root.geometry("1180x800")
        self.worker = self.video = self.engine = self.guard = self.trace = None
        self.analysis = None
        self.preselected_video=video_path
        self.recorder = VideoRecorder()
        self.packet = self.last_obs = None
        self.sequence = 0
        self.running = False
        self.paused = False
        self.step_once = False
        self.seek_seconds = tk.StringVar(value="0")
        self.game = tk.StringVar(value="mining")
        self.source = tk.StringVar(value="采集卡" if task_terminal else "合成演示")
        self.device = tk.StringVar(value="0")
        self.backend = tk.StringVar(value="DSHOW")
        self.profile = tk.StringVar()
        self.port = tk.StringVar(value="COM5" if task_terminal else "")
        self.status = tk.StringVar(value="选择游戏和任务模式后点击开始；当前未连接设备。" if task_terminal else "未启动。合成演示用于验证软件链路，画面不是游戏素材。")
        self._task_thread = None
        self._task_stop = threading.Event()
        self._task_lock = threading.Lock()
        self._task_latest = None
        self._task_hub = None
        self._task_finished = None
        self._task_logs = deque(maxlen=500)
        self._task_closing = False
        self._task_widgets = []
        self._task_setup_widgets = []
        self._task_debug_window = None
        self._task_pending_start = None
        self._task_ai_process=None
        self._task_ai_stream=None
        self._task_auto_pending=None
        self._task_recording_topmost=False
        self._task_active_mode = None
        self._task_preview_overlay = True
        self._task_preview_notice = ""
        self._task_display_ai = None
        self._task_ai_log_key = None
        self._task_ai_log_time = 0.0
        project=Path(__file__).resolve().parents[2]
        self._task_ai_monitor=ActiveAIMonitor(project/"artifacts/active_ai.json",repository_root=project,
            terminal_source_hashes=_TERMINAL_SOURCE_HASHES) if task_terminal else None
        if self.task_terminal:
            self._build_task_controls(root)
        else:
            self._build_legacy_controls(root)
        if task_terminal:
            body = ttk.Panedwindow(root, orient="horizontal")
            body.pack(fill="both", expand=True, padx=10, pady=(5,0))
            preview = ttk.LabelFrame(body, text="实时采集画面与识别")
            self.task_preview_panel = preview
            sidebar = ttk.LabelFrame(body, text="运行状态")
            body.add(preview, weight=4)
            body.add(sidebar, weight=1)
            self.canvas = tk.Canvas(preview, bg="#151922", highlightthickness=0, width=900, height=506)
            self.canvas.pack(fill="both", expand=True)
            self.canvas.create_text(450,240,text="先启动采集，核对 NS 画面\n连接控制器后可打开独立手柄调试窗口",fill="#d7e0ee",font=("Microsoft YaHei UI",17),justify="center")
            self.task_summary = tk.StringVar(value="尚未开始\n自主AI：未运行\n实际运行版本：无有效运行心跳\n助手接口：未启动\n录像：未录制")
            ttk.Label(sidebar,textvariable=self.task_summary,justify="left",wraplength=285,padding=10).pack(fill="x")
            self.details = tk.Text(sidebar,width=35,height=10,wrap="word",font=("Microsoft YaHei UI",10))
            self.details.pack(fill="both",expand=True,padx=6,pady=6)
            log_box=ttk.LabelFrame(root,text="任务日志")
            log_box.pack(fill="x",padx=10,pady=6)
            self.task_log=tk.Text(log_box,height=7,wrap="word",font=("Microsoft YaHei UI",9),state="disabled")
            scroll=ttk.Scrollbar(log_box,command=self.task_log.yview)
            self.task_log.configure(yscrollcommand=scroll.set)
            scroll.pack(side="right",fill="y")
            self.task_log.pack(fill="x",expand=True)
        else:
            self.canvas = tk.Canvas(root, bg="#151922", highlightthickness=0)
            self.canvas.pack(fill="both", expand=True, padx=10)
            self.details = tk.Text(root, height=5, wrap="word", font=("Microsoft YaHei UI", 10))
            self.details.pack(fill="x", padx=10, pady=8)
        ttk.Label(root, textvariable=self.status, wraplength=1150, padding=8).pack(fill="x")
        root.bind("<Escape>", lambda _: self.disarm())
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(30, self.tick)

    def _build_task_controls(self, root):
        self.task_mode=tk.StringVar(value=self.TASK_AUTO_MODE)
        self.task_mode_display=tk.StringVar(value=self.task_mode.get())
        self.task_resolution=tk.StringVar(value="1920x1080")
        self.task_fps=tk.StringVar(value="60")
        self.task_rounds=tk.StringVar(value="3")
        self.task_control=tk.BooleanVar(value=True)
        self.task_record=tk.BooleanVar(value=True)
        self.task_mode_hint=tk.StringVar(value=self.TASK_MODE_HELP[self.task_mode.get()])
        self.task_mode.trace_add("write",lambda *_:self.task_mode_hint.set(self.TASK_MODE_HELP.get(self.task_mode.get(),"")))
        self.task_mode.trace_add("write",lambda *_:self.task_mode_display.set(self.task_mode.get())
                                 if not (self._task_display_ai and self._task_display_ai.running) else None)
        self.task_connection_status=tk.StringVar(value="采集：未启动  |  控制器：未连接")
        project=Path(__file__).resolve().parents[2]
        local=project/"profiles/session.local.json"
        self.session_profile=tk.StringVar(value=str(local if local.exists() else project/"profiles/session.json"))
        connection=ttk.LabelFrame(root,text="采集卡与控制器",padding=8)
        connection.pack(fill="x",padx=10,pady=(8,4))
        capture_row=ttk.Frame(connection)
        capture_row.pack(fill="x")
        controller_row=ttk.Frame(connection)
        controller_row.pack(fill="x",pady=(7,0))
        def combo(parent,label,variable,values,width=10):
            ttk.Label(parent,text=label).pack(side="left",padx=(0,4))
            widget=ttk.Combobox(parent,textvariable=variable,values=values,state="readonly",width=width)
            widget.pack(side="left",padx=(0,10))
            self._task_widgets.append(widget)
            return widget
        combo(capture_row,"采集后端",self.backend,["DSHOW","MSMF","ANY"])
        ttk.Label(capture_row,text="设备索引").pack(side="left")
        device=ttk.Spinbox(capture_row,from_=0,to=20,textvariable=self.device,width=4)
        device.pack(side="left",padx=5);self._task_widgets.append(device)
        combo(capture_row,"分辨率",self.task_resolution,["1920x1080","1280x720","960x540"],12)
        combo(capture_row,"FPS",self.task_fps,["60","30"],4)
        ttk.Button(capture_row,text="启动采集",command=self._task_start_capture).pack(side="left",padx=4)
        ttk.Label(capture_row,textvariable=self.task_connection_status).pack(side="left",padx=12)
        ttk.Label(controller_row,text="EasyCon 串口").pack(side="left")
        self.port_box=ttk.Combobox(controller_row,textvariable=self.port,width=28)
        self.port_box.pack(side="left",padx=5);self._task_widgets.append(self.port_box)
        ttk.Button(controller_row,text="刷新串口",command=self.refresh_ports).pack(side="left",padx=4)
        ttk.Button(controller_row,text="连接控制器",command=self._task_connect_controller).pack(side="left",padx=4)
        ttk.Button(controller_row,text="手柄调试",command=self.controller_debug).pack(side="left",padx=4)
        enabled=ttk.Checkbutton(controller_row,text="任务启用控制",variable=self.task_control)
        enabled.pack(side="left",padx=8);self._task_widgets.append(enabled)
        task=ttk.LabelFrame(root,text="任务设置",padding=8)
        task.pack(fill="x",padx=10,pady=4)
        setup_start=len(self._task_widgets)
        task_row=ttk.Frame(task)
        task_row.pack(fill="x")
        combo(task_row,"小游戏",self.game,[g.value for g in Game],12)
        self.task_mode_box=combo(task_row,"任务模式",self.task_mode_display,list(self.TASK_MODE_HELP),22)
        self.task_mode_box.bind("<<ComboboxSelected>>",lambda _:self.task_mode.set(self.task_mode_display.get()))
        ttk.Label(task_row,text="运行轮数（0持续）").pack(side="left")
        rounds=ttk.Spinbox(task_row,from_=0,to=999,textvariable=self.task_rounds,width=5)
        rounds.pack(side="left",padx=5);self._task_widgets.append(rounds)
        record=ttk.Checkbutton(task_row,text="仅游戏内录像",variable=self.task_record)
        record.pack(side="left",padx=6);self._task_widgets.append(record)
        self.task_start=ttk.Button(task_row,text="开始任务",command=self.start)
        self.task_start.pack(side="left",padx=5)
        ttk.Button(task_row,text="暂停 / 继续",command=self._task_pause).pack(side="left",padx=4)
        ttk.Button(task_row,text="停止任务",command=self.stop).pack(side="left",padx=4)
        ttk.Button(task_row,text="归中（Esc）",command=self.disarm).pack(side="left",padx=4)
        paths=ttk.Frame(root,padding=(10,3))
        paths.pack(fill="x")
        ttk.Label(paths,text="菜单/标定配置").pack(side="left")
        entry=ttk.Entry(paths,textvariable=self.session_profile,width=67)
        entry.pack(side="left",padx=5);self._task_widgets.append(entry)
        ttk.Button(paths,text="选择",command=lambda:self._task_choose_path(self.session_profile)).pack(side="left")
        ttk.Label(paths,text="游戏配置（空白使用默认）").pack(side="left",padx=(12,0))
        entry=ttk.Entry(paths,textvariable=self.profile,width=35)
        entry.pack(side="left",padx=5);self._task_widgets.append(entry)
        ttk.Button(paths,text="选择",command=self.choose_profile).pack(side="left")
        self._task_setup_widgets=self._task_widgets[setup_start:]
        control=ttk.Frame(root,padding=(10,3))
        control.pack(fill="x")
        ttk.Button(control,text="截图",command=self.snapshot).pack(side="left",padx=8)
        ttk.Button(control,text="打开记录目录",command=self._task_open_output).pack(side="left",padx=3)

    def _build_legacy_controls(self, root):
        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")
        for label, variable, values in [("小游戏", self.game, [g.value for g in Game]),
                                         ("画面来源", self.source, ["合成演示", "视频回放", "采集卡"]),
                                         ("采集后端", self.backend, ["DSHOW", "MSMF", "ANY"])]:
            ttk.Label(top, text=label).pack(side="left", padx=4)
            ttk.Combobox(top, textvariable=variable, values=values, state="readonly", width=13).pack(side="left")
        ttk.Label(top, text="设备索引").pack(side="left", padx=5)
        ttk.Entry(top, textvariable=self.device, width=4).pack(side="left")
        ttk.Button(top, text="启动观察", command=self.start).pack(side="left", padx=8)
        ttk.Button(top, text="停止 / 释放", command=self.stop).pack(side="left")
        row = ttk.Frame(root, padding=(10, 0, 10, 8))
        row.pack(fill="x")
        ttk.Label(row, text="识别配置").pack(side="left")
        ttk.Entry(row, textvariable=self.profile, width=55).pack(side="left", padx=5)
        ttk.Button(row, text="选择 JSON", command=self.choose_profile).pack(side="left")
        ttk.Button(row, text="截图", command=self.snapshot).pack(side="left", padx=5)
        ttk.Button(row, text="录像开关", command=self.toggle_record).pack(side="left")
        ttk.Button(row, text="手柄调试窗口", command=self.controller_debug).pack(side="left", padx=5)
        row = ttk.Frame(root, padding=(10, 0, 10, 8))
        row.pack(fill="x")
        ttk.Label(row, text="EasyCon 串口").pack(side="left")
        self.port_box = ttk.Combobox(row, textvariable=self.port, width=14)
        self.port_box.pack(side="left", padx=5)
        ttk.Button(row, text="刷新串口", command=self.refresh_ports).pack(side="left")
        ttk.Button(row, text="启用实机控制", command=self.arm).pack(side="left", padx=5)
        ttk.Button(row, text="释放控制（Esc）", command=self.disarm).pack(side="left")
        ttk.Label(row, text="默认只观察；实机策略尚未验证", foreground="#9c5300").pack(side="left", padx=16)
        row = ttk.Frame(root, padding=(10, 0, 10, 8))
        row.pack(fill="x")
        ttk.Button(row, text="回放暂停 / 继续", command=self.pause_video).pack(side="left")
        ttk.Button(row, text="下一帧", command=self.next_frame).pack(side="left", padx=5)
        ttk.Entry(row, textvariable=self.seek_seconds, width=8).pack(side="left")
        ttk.Button(row, text="跳转到秒", command=self.seek_video).pack(side="left", padx=5)

    def choose_profile(self):
        if self.task_terminal:
            self._task_choose_path(self.profile)
            return
        path = filedialog.askopenfilename(filetypes=[("识别配置", "*.json")])
        if path:
            self.profile.set(path)

    def pause_video(self):
        if self.video:
            self.paused = not self.paused

    def next_frame(self):
        if self.video:
            self.paused, self.step_once = True, True

    def seek_video(self):
        if self.video:
            try:
                self.sequence = self.video.seek(round(float(self.seek_seconds.get()) * self.video.fps))
                self.engine = Engine(create_recognizer(self.engine.recognizer.game, self.engine.recognizer.profile))
                self.paused, self.step_once = True, True
            except (ValueError, RuntimeError) as error:
                self.status.set(str(error))

    def refresh_ports(self):
        self.port_box["values"] = [p.device for p in EasyConBackend.list_ports()]
        self.status.set("串口列表已刷新；列表为空时检查 pyserial 和设备连接。")

    def controller_debug(self):
        if self.task_terminal:
            if self._task_debug_window is not None and not self._task_debug_window.closed:
                self._task_debug_window.show()
            else:
                from .control.live_debug_ui import LiveDebugWindow
                self._task_debug_window=LiveDebugWindow(self.root,self._task_current_hub)
            return
        if self.guard:
            self.status.set("请先停止当前实机控制，释放串口。")
            return
        subprocess.Popen([sys.executable, str(Path(__file__).resolve().parents[2] / "run.py"), "controller-debug"])

    def start(self):
        if self.task_terminal:
            self._task_request_session()
            return
        if not self.stop():
            return
        try:
            game = Game(self.game.get())
            self.active_source = self.source.get()
            if self.active_source == "合成演示":
                profile = demo_profile(game)
            else:
                default_profile = Path(__file__).resolve().parents[2] / "profiles" / f"{game.value}.json"
                profile = load_profile(self.profile.get() or default_profile, game)
            self.engine = Engine(create_recognizer(game, profile), live=self.active_source == "采集卡")
            if self.active_source == "视频回放":
                path = str(self.preselected_video) if self.preselected_video else filedialog.askopenfilename(filetypes=[("视频", "*.mp4 *.avi *.mkv *.mov"), ("全部", "*.*")])
                self.preselected_video=None
                if not path:
                    return
                self.video = OpenCvVideoSource(Path(path))
                self.video.open()
            elif self.active_source == "采集卡":
                self.worker = CaptureWorker(CaptureConfig(device_index=int(self.device.get()), backend=self.backend.get()))
                self.worker.start()
            self.session = Path("artifacts") / datetime.now().strftime("run-%Y%m%d-%H%M%S-%f")
            self.trace = TraceWriter(self.session / "trace.jsonl")
            self.sequence = 0
            self.paused = self.step_once = False
            if self.video:
                reference = Path(__file__).resolve().parents[2]/"artifacts/reference_videos/utagaming-l3dMQV1sar8/reference.mp4"
                if self.video.path.resolve() == reference.resolve():
                    seconds = {Game.MINING:0, Game.HOPPING:325, Game.JUGGLING:663, Game.LANDING:963}[game]
                    self.sequence = self.video.seek(round(seconds*self.video.fps))
                    self.seek_seconds.set(str(seconds))
            self.packet = None
            self.running = True
            if self.worker:
                self.analysis = AnalysisWorker(self.engine, self.worker.latest)
                self.analysis.start()
            self.status.set(f"{self.active_source} / {game.value} / 只观察 / 日志：{self.session}")
        except Exception as error:
            self.stop()
            messagebox.showerror("启动失败", str(error))

    def disarm(self):
        if self.task_terminal:
            self._task_command({"command":"release"})
            return
        if self.engine:
            self.engine.disarm()
        self.status.set("控制已释放，当前只观察。")

    def arm(self):
        if not self.running or not self.engine:
            self.status.set("请先启动采集卡观察。")
            return
        try:
            profile = self.engine.recognizer.profile
            if not self.engine.live or not profile.get("calibrated") or profile.get("synthetic") or profile.get("advisory_only"):
                raise ValueError("需实机采集与 calibrated=true 的已校准配置")
            if not self.guard:
                backend = EasyConBackend()
                if not backend.connect(self.port.get()):
                    raise RuntimeError(backend.last_error)
                self.guard = GuardedController(backend)
                self.engine.controller = self.guard
            self.engine.arm()
            self.status.set("实机控制已启用；Esc 立即释放。")
        except Exception as error:
            self.disarm()
            messagebox.showerror("无法启用控制", str(error))

    def stop(self):
        if self.task_terminal:
            return self._task_stop_session()
        self.running = False
        self.disarm()
        if self.analysis:
            try:
                self.analysis.stop()
                self.analysis = None
            except RuntimeError as error:
                self.status.set(str(error))
                return False
        if self.guard:
            self.guard.close()
            self.guard = None
        self.recorder.stop()
        if self.video:
            self.video.close()
            self.video = None
        if self.trace:
            self.trace.close()
            self.trace = None
        if self.worker:
            try:
                self.worker.stop()
                self.worker = None
            except RuntimeError as error:
                self.status.set(str(error))
                return False
        return True

    def snapshot(self):
        if self.packet is None:
            return
        self.session.mkdir(parents=True, exist_ok=True)
        path = self.session / f"frame-{self.packet.sequence:06d}.png"
        ok, data = cv2.imencode(".png", self.packet.frame)
        if ok:
            path.write_bytes(data.tobytes())
            self.status.set(f"原始截图已保存：{path}")

    def toggle_record(self):
        if self.recorder.recording:
            self.status.set(f"录像已保存：{self.recorder.stop()}")
        elif self.packet is not None and self.running:
            fps = self.video.fps if self.video else 30
            try:
                path = self.recorder.start(self.session, self.packet.frame.shape, fps)
                self.status.set(f"正在录像：{path}；逐观察帧写入，时序以 trace 时间戳为准。")
            except Exception as error:
                messagebox.showerror("录像失败", str(error))

    def tick(self):
        if self.task_terminal:
            self._task_tick()
            return
        delay = 33
        try:
            if self.running:
                packet = None
                prepared = None
                if self.worker:
                    if self.worker.error:
                        raise RuntimeError(self.worker.error)
                    if self.analysis and self.analysis.error:
                        raise RuntimeError(self.analysis.error)
                    prepared = self.analysis.latest() if self.analysis else None
                    packet = prepared[0] if prepared else None
                    if packet and packet.sequence == self.sequence:
                        if time.perf_counter() - packet.timestamp > 0.2:
                            self.disarm()
                        packet = None
                elif self.video:
                    if self.paused and not self.step_once:
                        return
                    self.step_once = False
                    frame = self.video.read()
                    delay = max(1, round(1000 / self.video.fps))
                    if frame is None:
                        self.stop()
                        self.status.set("视频回放结束，日志已保存。")
                    else:
                        packet = FramePacket(frame, self.sequence / self.video.fps, self.sequence + 1, self.video.fps)
                else:
                    packet = demo_packet(self.engine.recognizer.game, self.sequence + 1)
                if packet:
                    self.packet, self.sequence = packet, packet.sequence
                    obs, decision = prepared[1:] if prepared else self.engine.process(packet)
                    self.last_obs = obs
                    self.trace.write(obs, decision, self.engine.armed)
                    if self.recorder.recording:
                        self.recorder.write(packet.frame)
                    self.render(packet, obs, decision)
        except Exception as error:
            self.stop()
            self.status.set(f"已停止：{error}")
        finally:
            self.root.after(delay, self.tick)

    def render(self, packet, obs, decision):
        frame = packet.frame.copy()
        fh, fw = frame.shape[:2]
        rx, ry, rw, rh = self.engine.recognizer.profile.get("viewport", [0, 0, 1, 1])
        draw_overlay = not self.task_terminal or self._task_preview_overlay
        for entity in obs.entities if draw_overlay else ():
            x, y = (rx + entity.x * rw) * fw, (ry + entity.y * rh) * fh
            w, h = entity.w * rw * fw, entity.h * rh * fh
            cv2.rectangle(frame, (round(x - w / 2), round(y - h / 2)),
                          (round(x + w / 2), round(y + h / 2)), (220, 160, 40), 1)
            cv2.putText(frame, f"{entity.kind} #{entity.track_id}", (round(x), round(y - h / 2 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 160, 40), 1)
        path = decision.diagnostics.get("path", []) if draw_overlay else []
        for start, end in zip(path, path[1:]):
            cv2.line(frame, (round((rx+start["x"]*rw)*fw), round((ry+start["depth"]*rh)*fh)),
                     (round((rx+end["x"]*rw)*fw), round((ry+end["depth"]*rh)*fh)), (255, 100, 255), 2)
        lines, markers = advice_geometry(obs, decision) if draw_overlay else ([], [])
        def position(point):
            return round((rx+point[0]*rw)*fw), round((ry+point[1]*rh)*fh)
        for start, end in lines:
            cv2.line(frame, position(start), position(end), (230, 150, 230), 2)
        for marker in markers:
            cv2.drawMarker(frame, position(marker), (80, 220, 255), cv2.MARKER_CROSS, 20, 2)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((max(200, self.canvas.winfo_width()), max(150, self.canvas.winfo_height())))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2, image=self.photo)
        if self.task_terminal and hasattr(self,"root") and getattr(self,"session",None):
            # Export our own layout for a window-only demonstration recording.
            # This contains no desktop image and writes only when layout changes.
            layout={"window_title":self.root.title(),"client_size":[self.root.winfo_width(),self.root.winfo_height()],
                "preview_rect":[self.canvas.winfo_rootx()-self.root.winfo_rootx()+(self.canvas.winfo_width()-image.width)//2,
                                self.canvas.winfo_rooty()-self.root.winfo_rooty()+(self.canvas.winfo_height()-image.height)//2,
                                image.width,image.height]}
            if layout!=getattr(self,"_task_recording_layout",None):
                try:
                    (self.session/"terminal_layout.json").write_text(json.dumps(layout,ensure_ascii=False),encoding="utf8")
                    self._task_recording_layout=layout
                except OSError:pass
        text = (f"{obs.game.value.upper()} · {'合成演示（非游戏画面）' if self.active_source == '合成演示' else self.active_source}"
                f" · 状态 {obs.phase} · 观测得分 {obs.confidence:.2f} · 采集帧 {obs.sequence}\n"
                f"策略：{self._task_display_reason(decision.reason) if self.task_terminal else decision.reason}\n"
                f"{'当前接口输入' if self.task_terminal else '建议输入'}：{json.dumps(state_dict(decision.state), ensure_ascii=False)}\n"
                f"执行：{'已启用' if self.engine.armed else '只观察'} · 识别 {decision.diagnostics.get('timing',{}).get('recognition_ms',0):.1f} ms"
                f" · 决策 {decision.diagnostics.get('timing',{}).get('decision_ms',0):.2f} ms"
                f"{(' · '+ '；'.join(obs.notes)) if not self.task_terminal else ''}")
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("end", text)
        if self.task_terminal:
            self.details.configure(state="disabled")

    def close(self):
        if self.task_terminal:
            self._task_closing=True
            if self._task_debug_window is not None and not self._task_debug_window.closed:
                try:
                    self._task_debug_window.close()
                except Exception as error:
                    self._task_log_message(f"手柄调试窗口关闭错误：{error}")
            self._task_stop_session()
            self._task_wait_close()
            return
        self.stop()
        self.root.destroy()

    def _task_choose_path(self, variable):
        if self._task_thread and self._task_thread.is_alive() and self._task_active_mode != "manual":
            self.status.set("请先停止任务再修改配置。")
            return
        path=filedialog.askopenfilename(filetypes=[("配置 JSON","*.json")])
        if path:
            variable.set(path)

    def _task_log_message(self, message):
        with self._task_lock:
            self._task_logs.append(str(message))

    @staticmethod
    def _task_transport_text(text):
        return (str(text).replace("调试：有时限的手动输入", "控制接口：正在发送输入")
                .replace("调试：自动控制已暂停，保持中立", "控制接口：输入已释放，等待下一条指令")
                .replace("人工实机调试：等待接口指令，保持中立", "控制接口：待命，输入保持中立"))

    def _task_display_reason(self, fallback):
        ai=getattr(self,"_task_display_ai",None)
        if ai is not None and ai.running:
            return ai.reason or "自主 AI 正在运行"
        return self._task_transport_text(fallback)

    def _task_visible_logs(self, logs, now):
        ai=getattr(self,"_task_display_ai",None)
        result=[]
        for message in logs:
            # Only suppress the polling transport echo. Errors, lifecycle and
            # connection messages remain visible and original logs stay on disk.
            transport=(message.startswith("[await_menu]") and any(text in message for text in (
                "调试：有时限的手动输入", "调试：自动控制已暂停", "人工实机调试：等待接口指令",
                "调试：控制归中，切换输入所有权")))
            if not (ai is not None and ai.running and transport):
                result.append(self._task_transport_text(message))
        if ai is not None and ai.running:
            key=(ai.loaded_version,ai.round_id,ai.reason)
            last_key=getattr(self,"_task_ai_log_key",None)
            last_time=getattr(self,"_task_ai_log_time",0.)
            if now-last_time>=1 and (key!=last_key or now-last_time>=4):
                result.append(f"[自主 AI · 第 {ai.round_id or '?'} 局 · {ai.loaded_version}] {ai.reason or '正在运行'}")
                self._task_ai_log_key,self._task_ai_log_time=key,now
        elif getattr(self,"_task_ai_log_key",None) is not None:
            result.append("自主 AI 当前已停止或心跳失效，控制接口待命。")
            self._task_ai_log_key=None
        return result

    def _task_current_hub(self):
        with self._task_lock:
            return self._task_hub

    def _task_settings(self):
        return {"game":self.game.get(),"mode":self.task_mode.get(),"resolution":self.task_resolution.get(),
                "fps":self.task_fps.get(),"device":self.device.get(),"rounds":self.task_rounds.get(),
                "profile":self.profile.get(),"session":self.session_profile.get(),"control":self.task_control.get(),
                "port":self.port.get(),"backend":self.backend.get(),"record":self.task_record.get()}

    def _task_start_capture(self):
        if self._task_thread and self._task_thread.is_alive():
            self.status.set("采集正在运行；修改采集参数前请先停止任务。")
            return
        self._task_request_session(mode_override="manual",control_override=False)

    def _task_connect_controller(self):
        if self.running and self.engine and self.engine.armed:
            self.status.set("控制器已连接，可打开「手柄调试」。")
            return
        self.task_control.set(True)
        self._task_request_session(mode_override="manual",control_override=True)

    def _task_request_session(self, mode_override=None, control_override=None):
        settings=self._task_settings()
        if mode_override is not None:
            settings["mode"]={"manual":self.TASK_INTERFACE_MODE,"menu":"仅定位菜单","single":"单局后返回","calibrate":"Retry 参数标定"}[mode_override]
        if control_override is not None:
            settings["control"]=bool(control_override)
        if not self.running and (not self._task_thread or not self._task_thread.is_alive()):
            self._task_start_session(settings=settings)
            return
        token=object()
        self._task_pending_start=token
        self._task_stop_session(cancel_pending=False)
        self.status.set("正在保存当前记录并释放设备，随后启动所选任务……")
        def wait_for_release():
            if self._task_closing or self._task_pending_start is not token:
                return
            if self.running or self._task_thread.is_alive():
                self.root.after(30,wait_for_release)
                return
            self._task_pending_start=None
            self._task_start_session(settings=settings)
        self.root.after(30,wait_for_release)

    def _task_command(self, command):
        with self._task_lock:
            hub=self._task_hub
        if hub is None:
            self.status.set("任务接口尚未启动，请先点击开始任务。")
            return False
        try:
            hub.submit(command)
            return True
        except (ValueError,RuntimeError,PermissionError) as error:
            self.status.set(str(error))
            self._task_log_message(f"界面控制：{error}")
            return False

    def _task_pause(self):
        with self._task_lock:
            hub=self._task_hub
        if hub:
            self._task_command({"command":"resume" if hub.paused else "pause"})
        else:
            self.status.set("请先开始任务。")

    def _task_press(self, button):
        self._task_queue_pulse({"command":"press","buttons":[button],"hold_ms":80})

    def _task_stick(self, axis, value):
        self._task_queue_pulse({"command":"press",axis:{"x":value,"y":0},"hold_ms":120})

    def _task_queue_pulse(self, payload):
        if not self._task_command({"command":"pause"}):
            return
        deadline=time.perf_counter()+1
        def wait_for_owner():
            with self._task_lock:
                hub=self._task_hub
            if hub is None or self._task_stop.is_set():
                return
            if hub.paused:
                self._task_command(payload)
            elif time.perf_counter()<deadline:
                self.root.after(20,wait_for_owner)
            else:
                self.status.set("自动控制尚未交出输入权限，短按未发送。")
        self.root.after(20,wait_for_owner)

    def _task_start_session(self, settings=None):
        if self.running or (self._task_thread and self._task_thread.is_alive()):
            self.status.set("任务仍在运行或收尾，请先停止并等待释放完成。")
            return
        try:
            settings=settings or self._task_settings()
            project=Path(__file__).resolve().parents[2]
            game=Game(settings["game"]).value
            mode=self.TASK_MODE_BACKENDS[settings['mode']]
            autonomous=settings['mode']==self.TASK_AUTO_MODE
            if autonomous and (game not in ('mining','juggling') or not settings['control']):
                raise ValueError('自主AI当前支持MINING/JUGGLING，并需要启用控制器')
            width,height=map(int,settings["resolution"].split("x"))
            fps=float(settings["fps"]);device=int(settings["device"]);rounds=int(settings["rounds"])
            if fps <= 0 or rounds < 0 or device < 0:
                raise ValueError("采集参数或轮数无效")
            default_profile={'mining':'mining.supervised.local.json','juggling':'juggling.live.local.json'}.get(game,f'{game}.json') if autonomous else f'{game}.json'
            profile_path=Path(settings["profile"]) if settings["profile"] else project/"profiles"/default_profile
            profile=load_profile(profile_path,game)
            session_path=Path(settings["session"]).resolve()
            session_config=json.loads(session_path.read_text(encoding="utf-8"))
            from .session_runtime import authorize_session
            authorize_session(settings["control"],mode,session_config,profile,settings["port"])
            output=project/"artifacts"/datetime.now().strftime("live-%Y%m%d-%H%M%S-%f")
            args=SimpleNamespace(game=game,mode=mode,device=device,backend=settings["backend"],width=width,height=height,
                fps=fps,port=settings["port"],control=settings["control"],session=session_path,profile=profile_path,
                rounds=rounds,no_record=not settings["record"],no_debug=False,debug_port=0,output=output,seconds=0)
        except (ValueError,KeyError,OSError) as error:
            self.status.set(f"无法启动：{error}")
            self._task_log_message(f"无法启动：{error}")
            return
        self._task_stop=threading.Event()
        self._task_active_mode=mode
        self.session=output
        self.engine=SimpleNamespace(recognizer=SimpleNamespace(profile=profile,game=Game(game)),armed=False)
        self.active_source="采集卡"
        self.sequence=-1
        self.packet=None
        self._task_last_status=None
        with self._task_lock:
            self._task_latest=None
            self._task_finished=None
            self._task_hub=None
        self._task_widget_states={widget:widget.cget("state") for widget in self._task_widgets}
        for widget in self._task_widgets:
            widget.configure(state="disabled")
        if mode=="manual":
            for widget in self._task_setup_widgets:
                widget.configure(state=self._task_widget_states[widget])
        self.task_start.configure(state="normal" if mode=="manual" else "disabled")
        self.running=True
        self.task_connection_status.set("采集：连接中  |  控制器："+("连接中" if args.control else "未连接"))
        self.status.set("正在启动采集与任务接口……")
        self.task_summary.set(f"游戏：{game.upper()}\n模式：{settings['mode']}\n状态：正在连接\n记录目录：{output.name}")
        self._task_log_message(f"启动 {game.upper()} / {mode} / {'发送控制' if args.control else '只观察'}")
        def update(packet,obs,decision,status):
            with self._task_lock:
                self._task_latest=(packet,obs,decision,status)
        def set_hub(hub):
            with self._task_lock:
                self._task_hub=hub
        def run():
            error=None
            try:
                from .session_runtime import run_session
                code=run_session(args,stop_event=self._task_stop,on_log=self._task_log_message,on_update=update,on_hub=set_hub)
            except Exception as exc:
                code=1;error=str(exc)
                self._task_log_message(f"任务错误：{error}")
                if "连接" in error or "采集" in error:
                    self._task_log_message("若旧的命令行终端仍在运行，请先在旧窗口按 Ctrl+C，释放 COM5 和采集卡。")
            finally:
                with self._task_lock:
                    self._task_finished=(code,error)
        self._task_thread=threading.Thread(target=run,name="raiders-live-session",daemon=True)
        self._task_thread.start()
        if autonomous:
            token=object();self._task_auto_pending=token;deadline=time.monotonic()+15
            def start_when_ready():
                if self._task_auto_pending is not token or self._task_stop.is_set():return
                status=self._task_last_status
                if (status and status.get('control_enabled') and self._task_current_hub()
                        and str(status.get('screen','')).startswith(('menu:','result:'))):
                    self._task_auto_pending=None
                    self._task_launch_autoplay(game,profile_path,rounds)
                elif time.monotonic()<deadline:self.root.after(100,start_when_ready)
                else:self.status.set('自主AI未启动：等待四项小游戏选择页或所选游戏结算页。')
            self.root.after(100,start_when_ready)

    def _task_launch_autoplay(self,game,profile_path,rounds):
        project=Path(__file__).resolve().parents[2]
        if self._task_display_ai and self._task_display_ai.running:
            self.status.set('自主AI已在运行，未启动第二个控制者。');return
        output=self.session/'autoplay'
        python=Path(sys.executable)
        if python.stem.lower()=='pythonw':python=python.with_name('python.exe')
        if game=='juggling':
            command=[str(python),str(project/'tools/juggling_hot_reload.py'),'--rounds',str(rounds),
                '--seconds','600','--profile',str(profile_path),'--parameters',str(project/'profiles/juggling.ai.local.json'),
                '--output',str(output)]
        else:
            command=[str(python),str(project/'tools/mining_hot_reload.py'),'--endpoint',str(project/'artifacts/control_endpoint.json'),
                '--profile',str(profile_path),'--parameters',str(project/'profiles/mining.ai.local.json'),
                '--output',str(output),'--rounds',str(rounds),'--seconds-per-round','600','--round-pause-seconds','2']
        try:
            self._task_ai_stream=(self.session/'autoplay-launcher.log').open('ab',buffering=0)
            self._task_ai_process=subprocess.Popen(command,cwd=project,stdout=self._task_ai_stream,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env={**os.environ,'PYTHONIOENCODING':'utf-8'})
            self._task_log_message(f'{game.upper()} 自主AI已启动 / {rounds if rounds else "持续"}局 / 局间重载')
        except OSError as error:self.status.set(f'自主AI启动失败：{error}')

    def _task_stop_session(self, cancel_pending=True):
        self._task_auto_pending=None
        if self._task_ai_process is not None and self._task_ai_process.poll() is None:
            subprocess.run(['taskkill','/PID',str(self._task_ai_process.pid),'/T','/F'],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self._task_ai_process=None
        if self._task_ai_stream is not None:self._task_ai_stream.close();self._task_ai_stream=None
        if cancel_pending:
            self._task_pending_start=None
        if self._task_thread and self._task_thread.is_alive():
            self._task_stop.set()
            with self._task_lock:
                hub=self._task_hub
            if hub:
                try:
                    hub.submit({"command":"release"})
                except (RuntimeError,ValueError):
                    pass
            self.status.set("正在停止任务并释放控制……")
            return False
        return True

    def _task_update_status(self, status, hub, *, active=True):
        session=getattr(self,"session",None) if active and self.running else None
        ai=self._task_ai_monitor.poll(session)
        self._task_display_ai=ai
        if hasattr(self,"task_mode_display"):
            display=f"{status['game'].upper()} 自主 AI" if ai.running else ('自主 AI（待命）' if self.task_mode.get()==self.TASK_AUTO_MODE else self.task_mode.get())
            self.task_mode_display.set(display)
            editable=not active or status["mode"]=="manual"
            self.task_mode_box.configure(state="disabled" if ai.running or not editable else "readonly")
        preview_source=("src/raiders_helper/mining_vision.py" if status["game"]=="mining"
                        else "src/raiders_helper/arcade_vision.py")
        preview_sha=_PREVIEW_VISION_SHA if status["game"]=="mining" else _PREVIEW_ARCADE_SHA
        loaded_preview=ai.source_hashes.get(preview_source)
        preview_mismatch=ai.running and loaded_preview is not None and loaded_preview != preview_sha
        self._task_preview_overlay = not preview_mismatch
        if str(status.get('screen','')).startswith(('result:','menu:','countdown:')):
            self._task_preview_overlay=False
        self._task_preview_notice = "AI已更新，预览叠加未同步" if preview_mismatch else ""
        self.task_preview_panel.configure(text=self._task_preview_notice or "实时采集画面与识别")
        recording_label = ('游戏录像：录制中' if status.get('recording_active') and active else
                           ('游戏录像：等待开局' if status.get('recording_enabled') and active else '游戏录像：已关闭'))
        terminal_recording=""
        if active and session:
            try:
                report=json.loads((Path(session)/"terminal_recording.json").read_text(encoding="utf8"))
                if report.get("running") and 0<=time.time()-report.get("updated_unix",0)<3:
                    terminal_recording="终端录像：录制中（仅本局）"
                elif report.get("completed"):
                    terminal_recording="终端录像：本局已录制完成"
            except (OSError,ValueError,TypeError):pass
        if terminal_recording:recording_label+="  |  "+terminal_recording
        topmost=terminal_recording.startswith('终端录像：录制中')
        if topmost!=getattr(self,'_task_recording_topmost',False):
            self._task_recording_topmost=topmost;self.root.attributes('-topmost',topmost)
            if topmost:self.root.lift()
        if active:
            self.task_connection_status.set(f"采集：{status['capture_fps_estimate']:.1f} fps  |  控制器：{'已连接' if status['control_enabled'] else '未连接'}  |  {recording_label}  |  {ai.short_text}")
        else:
            self.task_connection_status.set(f"采集：已停止  |  控制器：未连接  |  {ai.short_text}")
        mode=status["mode"]
        lines=[f"游戏：{status['game'].upper()}",f"画面：{status['screen']}",
               f"控制：{'已启用' if status['control_enabled'] and active else '已释放' if not active else '只观察'}"]
        if mode == "manual":
            lines.append("本终端：采集预览与控制接口")
            lines.append("任务模式：自主 AI（运行中）" if ai.running else "任务模式：控制接口待命")
        else:
            paused=bool(hub and hub.paused)
            lines.extend([f"任务模式：{mode}",f"本地任务流程：{'已暂停' if paused else '运行' if active else '已停止'}",
                          f"任务回合：{status['completed_rounds']} / {status['started_rounds']}"])
            if mode == "calibrate":
                lines.append(f"横向拟合：{'达到门槛' if status['fit_passed'] else '尚未通过'}")
        lines.extend([recording_label,ai.summary_text,f"助手接口：{'已连接' if hub and active else '已关闭'}",
                      f"记录：{Path(status['output']).name}"])
        if self._task_preview_notice:
            lines.append(self._task_preview_notice)
        self.task_summary.set("\n".join(lines))

    def _task_tick(self):
        try:
            with self._task_lock:
                latest=self._task_latest
                logs=list(self._task_logs);self._task_logs.clear()
                finished=self._task_finished
                self._task_finished=None
                hub=self._task_hub
            if latest:
                packet,obs,decision,status=latest
                self._task_last_status=status
                previous_overlay = self._task_preview_overlay
                self._task_update_status(status,hub)
                if packet.sequence != self.sequence or previous_overlay != self._task_preview_overlay:
                    self.packet,self.sequence,self.last_obs=packet,packet.sequence,obs
                    self.engine.armed=bool(status["control_enabled"])
                    self.render(packet,obs,decision)
                if self.running and not self._task_stop.is_set():
                    self.status.set(f"{self._task_display_reason(status['reason'])} · 记录目录：{status['output']}")
            logs=self._task_visible_logs(logs,time.monotonic())
            if logs:
                self.task_log.configure(state="normal")
                self.task_log.insert("end","\n".join(logs)+"\n")
                count=int(self.task_log.index("end-1c").split(".")[0])
                if count>1000:
                    self.task_log.delete("1.0",f"{count-800}.0")
                self.task_log.see("end")
                self.task_log.configure(state="disabled")
            if finished is not None:
                code,error=finished
                self.running=False
                self._task_active_mode=None
                with self._task_lock:
                    self._task_latest=None
                if self.engine:
                    self.engine.armed=False
                for widget,state in self._task_widget_states.items():
                    widget.configure(state=state)
                self.task_start.configure(state="normal")
                self.task_connection_status.set("采集：已停止  |  控制器：未连接  |  AI未运行")
                self.status.set(f"任务已停止：{error}" if error else "任务已结束，控制已释放，记录已保存。")
                if self._task_last_status:
                    self._task_update_status(self._task_last_status,None,active=False)
                else:
                    self.task_summary.set("任务已停止\n"+self._task_ai_monitor.poll(None).summary_text+"\n助手接口：已关闭\n控制：已释放")
        except Exception as error:
            self._task_stop_session()
            self.status.set(f"界面更新错误：{error}")
        finally:
            if not self._task_closing:
                self.root.after(33,self.tick)

    def _task_open_output(self):
        if getattr(self,"session",None) and self.session.exists():
            os.startfile(str(self.session))
        else:
            self.status.set("尚未创建记录目录。")

    def _task_wait_close(self):
        if self._task_thread and self._task_thread.is_alive():
            self.root.after(50,self._task_wait_close)
        else:
            self.root.destroy()


def run_gui(game=None, video=None, task_terminal=False, auto_start=False, record=True, profile=None,rounds=3):
    root = tk.Tk()
    app=Dashboard(root,video,task_terminal=task_terminal)
    if task_terminal:
        app.task_record.set(record)
        app.task_rounds.set(str(rounds))
    if game:
        app.game.set(game)
    if profile:
        app.profile.set(str(Path(profile).resolve()))
    if video:
        app.source.set("视频回放")
        root.after(100,app.start)
    elif task_terminal and auto_start:
        # Launchers own the AI process; this flag only prepares their shared
        # capture/controller interface and must not spawn a second AI.
        root.after(700,lambda:app._task_request_session(mode_override='manual'))
    root.mainloop()
    return 0
