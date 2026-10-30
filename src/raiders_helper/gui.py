import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
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


class Dashboard:
    def __init__(self, root, video_path=None):
        self.root = root
        root.title("Raiders 小游戏助手 · 框架预览")
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
        self.source = tk.StringVar(value="合成演示")
        self.device = tk.StringVar(value="0")
        self.backend = tk.StringVar(value="DSHOW")
        self.profile = tk.StringVar()
        self.port = tk.StringVar()
        self.status = tk.StringVar(value="未启动。合成演示用于验证软件链路，画面不是游戏素材。")
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
        self.canvas = tk.Canvas(root, bg="#151922", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10)
        self.details = tk.Text(root, height=5, wrap="word", font=("Microsoft YaHei UI", 10))
        self.details.pack(fill="x", padx=10, pady=8)
        ttk.Label(root, textvariable=self.status, wraplength=1150, padding=8).pack(fill="x")
        root.bind("<Escape>", lambda _: self.disarm())
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(30, self.tick)

    def choose_profile(self):
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
        if self.guard:
            self.status.set("请先停止当前实机控制，释放串口。")
            return
        subprocess.Popen([sys.executable, str(Path(__file__).resolve().parents[2] / "run.py"), "controller-debug"])

    def start(self):
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
        for entity in obs.entities:
            x, y = (rx + entity.x * rw) * fw, (ry + entity.y * rh) * fh
            w, h = entity.w * rw * fw, entity.h * rh * fh
            cv2.rectangle(frame, (round(x - w / 2), round(y - h / 2)),
                          (round(x + w / 2), round(y + h / 2)), (220, 160, 40), 1)
            cv2.putText(frame, f"{entity.kind} #{entity.track_id}", (round(x), round(y - h / 2 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 160, 40), 1)
        path = decision.diagnostics.get("path", [])
        for start, end in zip(path, path[1:]):
            cv2.line(frame, (round((rx+start["x"]*rw)*fw), round((ry+start["depth"]*rh)*fh)),
                     (round((rx+end["x"]*rw)*fw), round((ry+end["depth"]*rh)*fh)), (255, 100, 255), 2)
        lines, markers = advice_geometry(obs, decision)
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
        text = (f"{obs.game.value.upper()} · {'合成演示（非游戏画面）' if self.active_source == '合成演示' else self.active_source}"
                f" · 状态 {obs.phase} · 观测得分 {obs.confidence:.2f} · 帧 {obs.sequence}\n"
                f"策略：{decision.reason}\n建议输入：{json.dumps(state_dict(decision.state), ensure_ascii=False)}\n"
                f"执行：{'已启用' if self.engine.armed else '只观察'} · 识别 {decision.diagnostics.get('timing',{}).get('recognition_ms',0):.1f} ms"
                f" · 决策 {decision.diagnostics.get('timing',{}).get('decision_ms',0):.2f} ms · {'；'.join(obs.notes)}")
        self.details.delete("1.0", "end")
        self.details.insert("end", text)

    def close(self):
        self.stop()
        self.root.destroy()


def run_gui(game=None, video=None):
    root = tk.Tk()
    app=Dashboard(root,video)
    if game:
        app.game.set(game)
    if video:
        app.source.set("视频回放")
        root.after(100,app.start)
    root.mainloop()
    return 0
