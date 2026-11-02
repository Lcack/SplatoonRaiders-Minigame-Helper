"""Terminal live-session runner with independent menu and gameplay gates."""
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from .capture.models import CaptureConfig
from .capture.worker import CaptureWorker
from .control.easycon import EasyConBackend
from .control.guard import GuardedController
from .models import Decision
from .runtime import Engine
from .trace import TraceWriter, state_dict
from .vision import create_recognizer, crop_roi, load_profile


ROOT = Path(__file__).resolve().parents[2]


class _IdleTraceGate:
    """Keep gameplay evidence, but sample unchanged inactive screens at 1 Hz.

    A software control pause does not pause the console. Only visual inactive
    phases end full-rate logging; unknown frames inside a game stay full-rate.
    Timing, template scores and confidence jitter are not state transitions.
    """
    def __init__(self):
        self._in_game = False
        self._last_key = None
        self._last_write = -math.inf

    def should_write(self, screen, observation, decision, status, now):
        prefix = screen.partition(":")[0]
        inactive = {"menu", "result", "result_pending", "game_over", "pause", "paused", "overlay"}
        if (prefix in inactive or observation.phase in inactive
                or observation.metrics.get("game_ended", False)):
            self._in_game = False
        elif prefix in {"countdown", "playing"} or observation.phase in {"countdown", "playing"}:
            self._in_game = True
        key = (screen, status["screen"], status["stage"], status["source"],
               status["started_rounds"], status["completed_rounds"], observation.phase,
               bool(observation.metrics.get("game_ended", False)), decision.state,
               decision.reason, decision.diagnostics.get("session"),
               decision.diagnostics.get("debug", {}).get("paused"))
        write = self._in_game or key != self._last_key or now-self._last_write >= 1.0
        self._last_key = key
        if write:
            self._last_write = now
        return write


class ScreenMatcher:
    """AND/OR visual anchors; ambiguous screens fail closed to unknown."""
    def __init__(self, config, root):
        self.config, self.root = config, Path(root)
        self.templates = {}
        self._score_cache = {}
        self.previous, self.count = "unknown", 0
        self.has_ui_candidates = False
        self.stable_frames = config.get("stable_frames", 3)
        if type(self.stable_frames) is not int or self.stable_frames < 1:
            raise ValueError("stable_frames 必须为正整数")
        for screen in config.get("screens", []):
            if not screen.get("all_of"):
                raise ValueError("每个界面至少需要一个视觉锚点")
            for group in screen["all_of"]:
                for spec in group.get("any_of", [group]):
                    path = self.root / spec["path"]
                    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
                    if image is None or image.std() < 1:
                        raise ValueError(f"无效界面模板：{path}")
                    self.templates[spec["path"]] = image

    def score(self, gray, spec):
        key = (spec["path"], tuple(spec.get("roi", [0, 0, 1, 1])))
        if key in self._score_cache:
            return self._score_cache[key]
        tile = crop_roi(gray, spec.get("roi", [0, 0, 1, 1]))
        template = self.templates[spec["path"]]
        if any(a < b for a, b in zip(tile.shape, template.shape)) or tile.std() < 1:
            return -1.0
        value = float(cv2.matchTemplate(tile, template, cv2.TM_CCOEFF_NORMED).max())
        value = value if math.isfinite(value) else -1.0
        self._score_cache[key] = value
        return value

    def observe(self, frame):
        area = cv2.resize(crop_roi(frame, self.config.get("viewport", [0, 0, 1, 1])),
                          tuple(self.config.get("reference_size", [640, 480])))
        gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        self._score_cache = {}
        matches, scores = [], {}
        for screen in self.config.get("screens", []):
            passed, values = True, []
            for group in screen["all_of"]:
                alternatives = [(self.score(gray, spec), spec.get("threshold", 0.90))
                                for spec in group.get("any_of", [group])]
                passed = passed and any(score >= threshold for score, threshold in alternatives)
                values.append(max(score for score, _ in alternatives))
            scores[screen["name"]] = min(values)
            if passed:
                matches.append(screen["name"])
        screen = matches[0] if len(matches) == 1 else "unknown"
        self.has_ui_candidates = bool(matches)
        self.count = self.count+1 if screen == self.previous else 1
        self.previous = screen
        return (screen if self.count >= self.stable_frames else "unknown"), scores


def authorize_session(control, mode, session, profile, port):
    if not control:
        return
    if not port:
        raise ValueError("发送按键需要 --port 指定 EasyCon 串口")
    if mode == "manual":
        return  # Explicit short manual test; no menu or gameplay AI is run.
    if session.get("calibrated") is not True or not session.get("screens"):
        raise ValueError("菜单模板与按键未校准；先只观察，完成实测后使用独立实机 session 配置")
    if mode == "single" and (profile.get("calibrated") is not True
                             or profile.get("advisory_only") or profile.get("synthetic")):
        raise ValueError("单局自动控制需要已校准的游戏 profile；参考视频配置仅建议，可先用 --mode menu 测菜单")
    if mode == "calibrate" and (session.get("calibration", {}).get("enabled") is not True or profile.get("synthetic")):
        raise ValueError("标定需 calibration.enabled=true 的实机探针配置，不能使用合成演示配置")


def select_game(value=None):
    games = ("mining", "hopping", "juggling", "landing")
    if value is None:
        print("选择小游戏：1 MINING / 2 HOPPING / 3 JUGGLING / 4 LANDING")
        value = input("输入编号或名称：").strip().lower()
    if value in ("1", "2", "3", "4"):
        return games[int(value)-1]
    if value not in games:
        raise ValueError("小游戏必须是 1–4 或 mining/hopping/juggling/landing")
    return value


def run_session(args, *, stop_event=None, on_log=None, on_update=None, on_hub=None):
    """Run live processing on the caller's thread, including a GUI worker.

    Callbacks are synchronous and must remain lightweight. ``on_update`` gets
    the newest packet/observation/decision and a status dictionary; a GUI can
    replace one pending display result rather than queue every capture frame.
    Callback errors propagate after controller and capture cleanup.
    """
    from .session_flow import SessionFlow
    from .motion_calibration import CalibrationProbe, fit_motion
    from .experiment_recording import AsyncExperimentRecorder, RecordingBackend
    from .debug_api import DebugHub
    from .manual_input_pump import ManualInputPump
    def emit(message, *, flush=False):
        if on_log is not None:
            on_log(str(message))
        else:
            print(message, flush=flush)

    def wait(seconds):
        if stop_event is not None:
            stop_event.wait(seconds)
        else:
            time.sleep(seconds)

    game = select_game(args.game)
    path = args.session.resolve()
    session = json.loads(path.read_text(encoding="utf-8"))
    if session.get("version") != 1:
        raise ValueError("不支持的 session 配置版本")
    profile = load_profile(args.profile or ROOT/"profiles"/f"{game}.json", game)
    authorize_session(args.control, args.mode, session, profile, args.port)
    matcher = ScreenMatcher(session, path.parent)
    engine = Engine(create_recognizer(game, profile), live=True)
    flow = SessionFlow(game, {**session.get("flow", {}), "calibrated":session.get("calibrated",False)}, menu_only=args.mode == "menu",
                       end_behavior="retry" if args.mode == "calibrate" else "quit", max_rounds=args.rounds)
    probe = CalibrationProbe(game, session.get("calibration", {})) if args.mode == "calibrate" else None
    output = args.output or ROOT/"artifacts"/datetime.now().strftime("live-%Y%m%d-%H%M%S-%f")
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录已有内容，请选择新目录；不指定 --output 会自动创建独立实验目录")
    output.mkdir(parents=True, exist_ok=True)
    (output/"experiment.json").write_text(json.dumps({"game":game,"mode":args.mode,"live":True,
        "control_enabled":args.control,"capture":{"device":args.device,"backend":args.backend,
            "requested_size":[args.width,args.height],"requested_fps":args.fps},
        "session_config":session,"game_profile":profile,"clock":"time.perf_counter seconds",
        "clock_note":"capture timestamp is read completion; command timestamps bracket backend.send_state",
        "input_transport_owner":"manual_input_pump" if args.mode == "manual" and args.control and not args.no_debug else "analysis_loop",
        "idle_logging":{"unchanged_trace_interval_seconds":1.0,
            "gameplay_and_transport":"full_rate; unknown frames after gameplay remain full_rate"},
        "calibrated":False},ensure_ascii=False,indent=2),encoding="utf-8")
    trace = TraceWriter(output/"trace.jsonl")
    recorder = None if args.no_record else AsyncExperimentRecorder(output, fps=args.fps, initially_active=False)
    worker = CaptureWorker(CaptureConfig(args.device, args.backend, args.width, args.height, args.fps),
                           on_packet=recorder.on_frame if recorder else None)
    hub = None if args.no_debug else DebugHub(output, control_allowed=args.control, preferred_port=args.debug_port,
                                             capture_provider=worker.latest)
    guard, backend, events, samples_file, manual_pump = None, None, None, None, None
    fit_rows, last_fit, was_manual = [], None, False
    fit_resume_after = 0.0
    pause_started = None
    sequence = -1
    timed_out = False
    stopped = False
    idle_trace_gate = _IdleTraceGate()
    last_sent_state, last_sent_at = None, 0.0
    def submit_state(state, force=False):
        nonlocal last_sent_state, last_sent_at
        now = time.perf_counter()
        if guard and (force or state != last_sent_state or (state != Decision().state and now-last_sent_at >= 0.05)):
            guard.submit(state)
            last_sent_state, last_sent_at = state, time.perf_counter()
    try:
        emit(f"{game} / {'发送控制' if args.control else '只观察，不发送按键'} / {args.mode} / 日志 {output}")
        emit("请停在四项小游戏选择页；可停止并释放控制。")
        if stop_event is not None and stop_event.is_set():
            emit("已停止，保存本次数据并释放控制；不代表任务或拟合完成。")
            return 0
        if hub:
            hub.start()
            if on_hub is not None:
                on_hub(hub)
            emit(f"本机调试连接信息：{output/'control_endpoint.json'}（含本次令牌）")
        if args.control:
            def transport_event(event):
                if hub and event.get("success"):
                    from_pump = manual_pump is not None and event.get("thread") == ManualInputPump.THREAD_NAME
                    submitted_sequence = manual_pump.submitting_sequence if from_pump else sequence
                    hub.record_control(event["state"], timestamp=event["after"], sequence=submitted_sequence,
                                       source="manual_pump" if from_pump else "transport")
            backend = RecordingBackend(EasyConBackend(), output, callback=transport_event)
            if not backend.connect(args.port):
                raise RuntimeError(backend.last_error)
            guard = GuardedController(backend)
        events = (output/"session.jsonl").open("w", encoding="utf-8")
        if probe:
            samples_file = (output/"calibration_samples.jsonl").open("w", encoding="utf-8")
        worker.start()
        if args.mode == "manual" and hub and guard:
            manual_pump = ManualInputPump(hub, guard, worker.latest, stop_event=stop_event,
                health_error=lambda: worker.error or backend.recording_error or guard.error)
            manual_pump.start()
        sequence, started, last_print, last_status = -1, time.perf_counter(), 0.0, None
        saved_status = set()
        while not flow.done and not flow.failed:
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            now = time.perf_counter()
            if args.seconds and now-started >= args.seconds:
                timed_out = True
                break
            if worker.error:
                raise RuntimeError(worker.error)
            if recorder and recorder.error:
                raise RuntimeError(f"实验录像失败：{recorder.error}")
            if backend and backend.recording_error:
                raise RuntimeError(backend.recording_error)
            if guard and guard.error:
                raise RuntimeError(guard.error)
            if manual_pump and manual_pump.error:
                raise RuntimeError(manual_pump.error)
            packet = worker.latest()
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            manual = hub.take_manual(now) if hub and manual_pump is None else None
            if packet is None or packet.sequence == sequence or now-packet.timestamp > 0.2:
                # A release request and pulse expiry must still be processed
                # when capture stalls. New nonneutral input needs a fresh view.
                if guard and manual is not None and manual_pump is None:
                    state = manual.state if packet is not None and 0 <= now-packet.timestamp <= 0.2 else Decision().state
                    submit_state(state)
            if packet is None:
                if now-started > 10:
                    raise RuntimeError("10秒内未取得采集帧")
                wait(0.005)
                continue
            age = now-packet.timestamp
            if age > 0.2:
                if guard and manual_pump is None:
                    submit_state(Decision().state)
                if age > 2:
                    raise RuntimeError("采集画面停止更新超过2秒")
                wait(0.005)
                continue
            if packet.sequence == sequence:
                wait(0.001)
                continue
            sequence = packet.sequence
            menu_started = time.perf_counter()
            screen, screen_scores = matcher.observe(packet.frame)
            menu_ms = (time.perf_counter()-menu_started)*1000
            observation, game_decision = engine.process(packet)
            # Never let an unstable/ambiguous known UI fall through to gameplay.
            if screen == "unknown" and matcher.has_ui_candidates:
                effective = "unknown"
            else:
                effective = screen if screen != "unknown" else (
                    f"playing:{game}" if observation.phase == "playing" else "unknown")
            if recorder:
                # Stopping may use a first-frame UI hit; starting still needs
                # the stable screen or the recognizer's confirmed play phase.
                raw_screen=getattr(matcher,"previous","unknown")
                recording_screen=raw_screen if raw_screen.partition(":")[0] in {"menu","result"} else effective
                recorder.observe_phase(recording_screen, observation.phase, timestamp=packet.timestamp,
                    game_ended=bool(observation.metrics.get("game_ended",False)),
                    new_round_confirmed=bool(observation.metrics.get("new_round_confirmed",False)))
            source = "automatic"
            if args.mode == "manual":
                decision = (manual_pump.snapshot() if manual_pump else manual) or Decision(reason="人工实机调试：等待接口指令，保持中立")
                source = "manual"
            elif manual is not None:
                decision, source = manual, "manual"
                was_manual = True
                if pause_started is None:
                    pause_started = now
            else:
                if was_manual:
                    # Exclude manual interventions and their immediate response
                    # from automatic identification samples.
                    was_manual = False
                    if pause_started is not None:
                        flow.suspend(max(0,now-pause_started))
                        pause_started = None
                    fit_resume_after = time.perf_counter()+0.4
                    if probe:
                        probe.reset_round(flow.started_rounds)
                strategy = probe.decide(observation, time.perf_counter()) if probe else game_decision
                decision = flow.update(effective, strategy, time.perf_counter())
                if probe and effective == f"playing:{game}":
                    source = "calibration_probe"
                for event in decision.diagnostics.get("session", {}).get("events", []):
                    if probe and event["type"] == "round_started":
                        probe.reset_round(event["round"])
                    if probe and event["type"] == "round_completed":
                        last_fit = fit_motion(fit_rows, {"game": game, **session.get("calibration", {})})
                        (output/"motion_fit.json").write_text(json.dumps(last_fit, ensure_ascii=False, indent=2), encoding="utf-8")
                        if args.control and last_fit.get("passed"):
                            flow.stop_after_round("横向动作响应达到配置的跨回合验证门槛；完整游戏仍未标定")
                        emit(f"第{event['round']}轮采样结束：模型门槛{'通过' if last_fit.get('passed') else '尚未通过'}", flush=True)
            timing = {**game_decision.diagnostics.get("timing", {}), "menu_recognition_ms": menu_ms,
                      "session_frame_age_ms": (time.perf_counter()-packet.timestamp)*1000}
            decision = replace(decision, diagnostics={**decision.diagnostics, "timing": timing})
            # Check freshness again at the final transport boundary.
            if manual_pump is None and not 0 <= time.perf_counter()-packet.timestamp <= 0.2:
                decision = Decision(reason="菜单/游戏分析完成时画面已过期，释放输入",
                                    diagnostics={"timing": timing})
                source = "stale_release"
            if guard and manual_pump is None:
                submit_state(decision.state, force=True)
            submitted_at = time.perf_counter()
            if probe and guard and source == "calibration_probe" and submitted_at >= fit_resume_after:
                row = {"timestamp": submitted_at, "round_id": flow.started_rounds, "live": True,
                       "executed": True, "game": game, "state": state_dict(decision.state),
                       "observation": asdict(observation)}
                fit_rows.append(row)
                # Retain recent complete rounds in memory; all samples stay on disk.
                if len(fit_rows) > 100000:
                    oldest = max(1, flow.started_rounds-5)
                    fit_rows = [row for row in fit_rows if row["round_id"] >= oldest][-100000:]
                samples_file.write(json.dumps(row, ensure_ascii=False)+"\n")
            current_status = {"output": str(output.resolve()), "game": game, "mode": args.mode,
                              "screen": effective, "stage": flow.stage, "control_enabled": bool(guard),
                              "started_rounds": flow.started_rounds, "completed_rounds": flow.completed_rounds,
                              "reason": decision.reason, "source": source,
                              "capture_size": list(packet.frame.shape[:2][::-1]),
                              "capture_fps_estimate": packet.measured_fps,
                              "recording_enabled": recorder is not None,
                              "recording_active": bool(recorder and recorder.active),
                              "timing": timing, "fit_passed": bool(last_fit and last_fit.get("passed"))}
            current_status["input_transport_owner"] = ManualInputPump.THREAD_NAME if manual_pump else "analysis_loop"
            if hub:
                # Raw capture requests use worker.latest directly. Publishing
                # the analyzed packet here would add another full CV cycle of
                # latency before an external policy can see the frame.
                hub.publish(status=current_status)
            if on_update is not None:
                on_update(packet, observation, decision, current_status)
            raw_screen = getattr(matcher, "previous", "unknown")
            log_screen = raw_screen if raw_screen.partition(":")[0] in {"menu", "result"} else effective
            if idle_trace_gate.should_write(log_screen, observation, decision, current_status, now):
                trace.write(observation, decision, args.control)
                events.write(json.dumps({"sequence": sequence, "screen": effective, "stage": flow.stage,
                                         "scores": screen_scores, "reason": decision.reason, "source": source,
                                         "session": decision.diagnostics.get("session", {})}, ensure_ascii=False)+"\n")
            status = (effective, flow.stage)
            if status not in saved_status and len(saved_status) < 40:
                name = f"screen-{sequence:08d}.png"
                cv2.imencode(".png", packet.frame)[1].tofile(str(output/name))
                saved_status.add(status)
            if status != last_status or now-last_print >= 1:
                emit(f"[{flow.stage}] {effective}：{decision.reason}", flush=True)
                last_print = now
            last_status = status
        if flow.failed:
            raise RuntimeError(f"流程已停止：{flow.failure}；查看 session.jsonl 与截图")
        if stopped:
            emit("已停止，保存本次数据并释放控制；不代表任务或拟合完成。")
        elif timed_out:
            emit("本次实机测试时限结束，已保存数据；不代表任务或拟合完成。")
        elif args.mode == "calibrate":
            emit(f"标定采样结束，共{flow.completed_rounds}轮；验证门槛{'通过' if last_fit and last_fit.get('passed') else '尚未通过'}。")
        else:
            emit("菜单定位测试结束，停在目标选项，未确认开局。" if args.mode == "menu" else "已观察到一局结束并确认回到选择页。")
        return 0
    except KeyboardInterrupt:
        emit("已中断，释放控制。")
        return 130
    finally:
        # Revoke input before any potentially blocked capture shutdown.
        active_error, cleanup_error = sys.exception(), None
        def cleanup(action):
            nonlocal cleanup_error
            try:
                action()
            except Exception as error:
                if cleanup_error is None:
                    cleanup_error = error
        if guard:
            if manual_pump:
                cleanup(manual_pump.close)
            cleanup(guard.close)
        if backend:
            cleanup(backend.close)
        cleanup(trace.close)
        if events:
            cleanup(events.close)
        if samples_file:
            cleanup(samples_file.close)
        def stop_capture():
            try:
                worker.stop()
            except RuntimeError as error:
                emit(f"停止采集：{error}")
        if hub:
            cleanup(hub.close)
        cleanup(stop_capture)
        if recorder:
            cleanup(recorder.close)
        if on_hub is not None:
            cleanup(lambda: on_hub(None))
        if cleanup_error is not None:
            if active_error is None:
                raise cleanup_error
            active_error.add_note(f"实机收尾另有错误：{cleanup_error}")
