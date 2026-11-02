"""Nonblocking capture recording and controller transport audit sidecars.

Frame submission retains the packet's ndarray by reference. The capture owner
must never mutate a published frame; resize, letterboxing, encoding and disk I/O
all happen on the recorder thread. A bounded frame queue limits retained image
memory. Small drop records use a separate metadata queue so full image queues do
not erase evidence of dropped capture sequences.

The MP4 has fixed container FPS. frames.jsonl, rather than its playback duration,
is the source of capture timing. A successful controller event means only that
the wrapped backend returned success, never that the console executed it.
"""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import json
import math
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np

from .control.backend import ControllerBackend


def _state_dict(state):
    return {"buttons": sorted(button.value for button in state.buttons),
            "left_stick": asdict(state.left_stick),
            "right_stick": asdict(state.right_stick)}


class AsyncExperimentRecorder:
    """Record capture packets without encoding or disk waits in on_frame()."""

    def __init__(self, output, fps=60, max_size=(1280, 720), queue_size=16, initially_active=True):
        if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("录像FPS必须为正数")
        if (len(max_size) != 2 or any(type(v) is not int or v < 2 for v in max_size)
                or type(queue_size) is not int or queue_size < 1):
            raise ValueError("录像尺寸和队列长度无效")
        if type(initially_active) is not bool:
            raise ValueError("initially_active 必须为布尔值")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.video_path = self.output / "capture.mp4"
        self.frames_path = self.output / "frames.jsonl"
        self.events_path = self.output / "recording-events.jsonl"
        self.summary_path = self.output / "recording-summary.json"
        self.fps, self.max_size = float(fps), tuple(max_size)
        self._frames = queue.Queue(maxsize=queue_size)
        self._metadata = queue.SimpleQueue()
        self._gate_events = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._accepting = True
        self._active = initially_active
        self._ended_latch = False
        self._error = ""
        self._stats = {"submitted_frames": 0, "enqueued_frames": 0, "written_frames": 0,
                       "queue_full_drops": 0, "encoder_error_drops": 0,
                       "encoding_failure_drops": 0, "rejected_after_close": 0,
                       "ignored_inactive_frames": 0,
                       "recording_starts": int(initially_active), "recording_stops": 0}
        self._writer = None
        self._size = None
        self._video_created = False
        self._first_timestamp = self._last_timestamp = None
        # Waiting in menus must not create an empty movie or a per-frame file.
        self._jsonl = self.frames_path.open("w", encoding="utf-8", buffering=1) if initially_active else None
        self._events_jsonl = None
        if initially_active:
            stamp=time.perf_counter()
            self._gate_events.put({"event":"recording_gate", "active":True,
                                   "reason":"initially_active", "timestamp":stamp,
                                   "changed_at":stamp, "segment":1})
        self._thread = threading.Thread(target=self._run, name="experiment-video", daemon=True)
        self._thread.start()

    @property
    def error(self):
        with self._lock:
            return self._error

    @property
    def active(self):
        with self._lock:
            return self._active and self._accepting

    def set_active(self, active, reason="manual", timestamp=None):
        """Change the recording gate without waiting for encoding or disk I/O.

        Paused packets are counted only in memory. Already accepted frames are
        drained, preserving the last frames of a round. A later round appends
        video indices while sidecar capture times retain the real time gap.
        """
        if type(active) is not bool:
            raise ValueError("active 必须为布尔值")
        changed_at=time.perf_counter()
        timestamp=changed_at if timestamp is None else timestamp
        if type(timestamp) not in (int,float) or not math.isfinite(timestamp):
            raise ValueError("录像门控时间必须为有限数")
        with self._lock:
            if not self._accepting or self._active == active:
                return False
            self._active=active
            self._stats["recording_starts" if active else "recording_stops"] += 1
            self._gate_events.put({"event":"recording_gate", "active":active,
                                   "reason":str(reason), "timestamp":timestamp,
                                   "changed_at":changed_at,
                                   "segment":self._stats["recording_starts"]})
        return True

    def observe_phase(self, screen="unknown", phase="unknown", timestamp=None,
                      game_ended=False, new_round_confirmed=False):
        """Open only on confirmed countdown/play; close on a known UI screen.

        An unknown initial picture never starts recording. Unknown pictures in
        an active round retain it so flashes, collisions and short occlusions
        remain available for diagnosis. A first-frame end signal stops before
        stable result UI is available. Ended rounds stay closed through unknown
        or residual playing labels; a confirmed countdown/new round unlocks it.
        """
        screen=str(screen or "unknown")
        prefix=screen.partition(":")[0]
        if game_ended or prefix in {"menu","result","result_pending","game_over"} or phase in {"result","result_pending","game_over","menu"}:
            with self._lock:
                self._ended_latch=True
            self.set_active(False, reason="game_ended" if game_ended else screen if prefix!="unknown" else str(phase), timestamp=timestamp)
            return self.active
        if prefix in {"pause","paused"}:
            self.set_active(False, reason=screen, timestamp=timestamp)
        elif prefix=="countdown" or phase=="countdown":
            with self._lock:
                self._ended_latch=False
            self.set_active(True, reason=screen, timestamp=timestamp)
        elif phase in {"pause","paused","overlay"}:
            self.set_active(False, reason=str(phase), timestamp=timestamp)
        elif prefix=="playing" or phase=="playing":
            with self._lock:
                may_start=not self._ended_latch or bool(new_round_confirmed)
                if may_start:
                    self._ended_latch=False
            if may_start:
                self.set_active(True, reason=screen if prefix!="unknown" else str(phase), timestamp=timestamp)
        return self.active

    def _set_error(self, message):
        with self._lock:
            if not self._error:
                self._error = message

    @staticmethod
    def _packet_event(packet, submitted_at, status, reason=None, segment=None):
        result = {"status": status, "video_frame_index": None,
                  "capture_sequence": packet.sequence, "capture_timestamp": packet.timestamp,
                  "submitted_at": submitted_at, "measured_fps": packet.measured_fps}
        if reason:
            result["reason"] = reason
        if segment is not None:
            result["recording_segment"] = segment
        return result

    def on_frame(self, packet):
        """Return immediately after enqueue/drop; never copy or encode pixels.

        False indicates queue pressure, a recorder error, or a closed recorder.
        Inactive packets only increment a counter, without per-frame sidecars.
        Accepted/dropped recording packets keep their exact capture mapping.
        """
        submitted_at = time.perf_counter()
        # The short lock makes shutdown atomic with submission. Neither codec
        # nor filesystem operations run while holding it.
        with self._lock:
            if not self._accepting:
                self._stats["rejected_after_close"] += 1
                return False
            self._stats["submitted_frames"] += 1
            if not self._active:
                self._stats["ignored_inactive_frames"] += 1
                return False
            segment=self._stats["recording_starts"]
            if self._error:
                self._stats["encoder_error_drops"] += 1
                self._metadata.put(self._packet_event(packet, submitted_at, "dropped", "encoder_error", segment))
                return False
            try:
                self._frames.put_nowait((packet, submitted_at, segment))
            except queue.Full:
                self._stats["queue_full_drops"] += 1
                self._metadata.put(self._packet_event(packet, submitted_at, "dropped", "queue_full", segment))
                return False
            self._stats["enqueued_frames"] += 1
            return True

    def _create_writer(self, size):
        writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 self.fps, size)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError("无法创建实验录像MP4编码器")
        return writer

    def _prepare_frame(self, frame):
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3
                or frame.shape[2] != 3 or frame.dtype != np.uint8
                or min(frame.shape[:2]) < 2):
            raise ValueError("录像要求非空uint8 BGR帧")
        height, width = frame.shape[:2]
        if self._size is None:
            scale = min(1.0, self.max_size[0]/width, self.max_size[1]/height)
            # MPEG encoders generally require even dimensions.
            self._size = (max(2, int(width*scale)//2*2), max(2, int(height*scale)//2*2))
        target_w, target_h = self._size
        scale = min(target_w/width, target_h/height)
        scaled_w, scaled_h = max(1, round(width*scale)), max(1, round(height*scale))
        resized = (frame if (scaled_w, scaled_h) == (width, height)
                   else cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA))
        if (scaled_w, scaled_h) == self._size:
            return resized
        canvas = np.zeros((target_h, target_w, 3), np.uint8)
        left, top = (target_w-scaled_w)//2, (target_h-scaled_h)//2
        canvas[top:top+scaled_h, left:left+scaled_w] = resized
        return canvas

    def _write_event(self, event):
        try:
            if self._jsonl is None:
                self._jsonl=self.frames_path.open("w", encoding="utf-8", buffering=1)
            self._jsonl.write(json.dumps(event, ensure_ascii=False)+"\n")
        except Exception as exc:
            self._set_error(f"录像侧车写入失败：{type(exc).__name__}")

    def _drain_gate_events(self):
        for _ in range(512):
            try:
                event=self._gate_events.get_nowait()
            except queue.Empty:
                return
            try:
                if self._events_jsonl is None:
                    self._events_jsonl=self.events_path.open("w", encoding="utf-8", buffering=1)
                self._events_jsonl.write(json.dumps(event,ensure_ascii=False)+"\n")
            except Exception as exc:
                self._set_error(f"录像门控日志写入失败：{type(exc).__name__}")

    def _drain_metadata(self):
        # A finite batch prevents a continuous stream of drops starving frames.
        for _ in range(512):
            try:
                event = self._metadata.get_nowait()
            except queue.Empty:
                return
            self._write_event(event)

    def _run(self):
        try:
            while not self._stop.is_set() or not self._frames.empty() or not self._metadata.empty() or not self._gate_events.empty():
                self._drain_gate_events()
                self._drain_metadata()
                try:
                    packet, submitted_at, segment = self._frames.get(timeout=0.02)
                except queue.Empty:
                    continue
                try:
                    if self.error:
                        with self._lock:
                            self._stats["encoder_error_drops"] += 1
                        self._write_event(self._packet_event(packet, submitted_at, "dropped", "encoder_error", segment))
                        continue
                    pixels = self._prepare_frame(packet.frame)
                    if self._writer is None:
                        self._writer = self._create_writer(self._size)
                        self._video_created = True
                    result = self._writer.write(pixels)
                    if result is False:
                        raise RuntimeError("编码器报告写入失败")
                    with self._lock:
                        index = self._stats["written_frames"]
                        self._stats["written_frames"] += 1
                        if self._first_timestamp is None:
                            self._first_timestamp = packet.timestamp
                        self._last_timestamp = packet.timestamp
                    event = self._packet_event(packet, submitted_at, "written", segment=segment)
                    event.update(video_frame_index=index, encoded_at=time.perf_counter(),
                                 raw_size=[packet.frame.shape[1], packet.frame.shape[0]],
                                 encoded_size=list(self._size))
                    self._write_event(event)
                except Exception as exc:
                    self._set_error(f"实验录像编码失败：{type(exc).__name__}")
                    with self._lock:
                        self._stats["encoding_failure_drops"] += 1
                    self._write_event(self._packet_event(packet, submitted_at, "dropped", "encoding_failed", segment))
                finally:
                    self._frames.task_done()
        except Exception as exc:
            self._set_error(f"实验录像线程失败：{type(exc).__name__}")
        finally:
            try:
                if self._writer is not None:
                    self._writer.release()
                if self._jsonl is not None:
                    self._jsonl.close()
                if self._events_jsonl is not None:
                    self._events_jsonl.close()
            except Exception as exc:
                self._set_error(f"实验录像收尾失败：{type(exc).__name__}")

    def _summary(self, complete):
        with self._lock:
            return {**self._stats, "error": self._error, "complete": complete,
                    "recording_active": self._active and self._accepting,
                    "game_ended_latched": self._ended_latch,
                    "container_fps": self.fps, "encoded_size": self._size,
                    "video": self.video_path.name, "video_created": self._video_created,
                    "frames_sidecar": self.frames_path.name,
                    "recording_events": self.events_path.name,
                    "first_capture_timestamp": self._first_timestamp,
                    "last_capture_timestamp": self._last_timestamp,
                    "timing_note": "MP4为固定容器FPS，真实采集时序以frames.jsonl为准；丢帧行无视频帧索引。",
                    "write_note": "written_frames只计编码器write正常返回，未逐帧校验成品文件。"}

    def close(self, timeout=5.0):
        self.set_active(False, reason="close")
        with self._lock:
            self._accepting = False
        self._stop.set()
        self._thread.join(timeout=timeout)
        complete = not self._thread.is_alive()
        if not complete:
            self._set_error("实验录像线程尚未退出，文件可能未完成封装")
        summary = self._summary(complete)
        try:
            self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self._set_error(f"实验录像摘要写入失败：{type(exc).__name__}")
            summary["error"] = self.error
        return summary


class RecordingBackend(ControllerBackend):
    """Wrap below GuardedController so watchdog neutral sends are audited too.

    Command disk writes run on an independent thread. The optional callback is
    synchronous, after the transport finishes and outside the send lock: it must
    only perform bounded, lightweight memory updates, never filesystem I/O.
    Call close()
    after the guard has released/disconnected, to drain the command audit queue.
    No ports, wire payloads or backend error strings are written to the sidecar.
    """

    def __init__(self, backend, output, callback=None):
        self.backend, self.callback = backend, callback
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "commands.jsonl"
        self._file = self.path.open("w", encoding="utf-8", buffering=1)
        self._events = queue.SimpleQueue()
        self._send_lock = threading.RLock()
        self._error_lock = threading.Lock()
        self._recording_error = ""
        self._closed = False
        self._sequence = 0
        self._thread = threading.Thread(target=self._write_events, name="experiment-commands", daemon=True)
        self._thread.start()

    @property
    def connected(self):
        return self.backend.connected

    @property
    def last_error(self):
        return getattr(self.backend, "last_error", "")

    @property
    def recording_error(self):
        with self._error_lock:
            return self._recording_error

    def _set_recording_error(self, message):
        with self._error_lock:
            if not self._recording_error:
                self._recording_error = message

    def connect(self, port=None):
        with self._send_lock:
            return self.backend.connect(port)

    def disconnect(self):
        with self._send_lock:
            return self.backend.disconnect()

    def send_state(self, state):
        event = None
        try:
            with self._send_lock:
                if self._closed:
                    raise RuntimeError("按键实验记录器已关闭")
                self._sequence += 1
                event = {"event": "send_state", "command_sequence": self._sequence,
                         "state": _state_dict(state), "thread": threading.current_thread().name,
                         "scope": "backend_transport_result_not_console_execution"}
                before = time.perf_counter()
                try:
                    result = self.backend.send_state(state)
                except Exception as exc:
                    after = time.perf_counter()
                    event.update(before=before, after=after, success=False, exception_type=type(exc).__name__)
                    self._events.put(event)
                    raise
                after = time.perf_counter()
                event.update(before=before, after=after, success=bool(result))
                self._events.put(event)
                return result
        finally:
            if event is not None and "after" in event and self.callback is not None:
                try:
                    self.callback(deepcopy(event))
                except Exception as exc:
                    self._set_recording_error(f"按键日志回调失败：{type(exc).__name__}")

    def _write_events(self):
        try:
            while True:
                event = self._events.get()
                if event is None:
                    break
                try:
                    self._file.write(json.dumps(event, ensure_ascii=False)+"\n")
                except Exception as exc:
                    self._set_recording_error(f"按键日志写入失败：{type(exc).__name__}")
        finally:
            try:
                self._file.close()
            except Exception as exc:
                self._set_recording_error(f"按键日志收尾失败：{type(exc).__name__}")

    def close(self, timeout=5.0):
        with self._send_lock:
            if not self._closed:
                self._closed = True
                self._events.put(None)
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self._set_recording_error("按键日志线程尚未退出")
