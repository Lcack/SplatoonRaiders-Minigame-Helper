"""One manual-mode input owner, independent of slow vision/display work.

The task runtime must not call take_manual or submit controller states while
this pump owns manual mode. Automatic/calibration sessions keep their old
single-loop owner. Transport logs remain the authority for actual sends.
"""
from dataclasses import replace
import math
import threading
import time

from .control.buttons import ControllerState
from .models import Decision


class ManualInputPump:
    THREAD_NAME = "raiders-manual-input"

    def __init__(self, hub, guard, latest_frame, *, stop_event=None, health_error=None,
                 interval_seconds=.01, max_frame_age_seconds=.20):
        if not .005 <= interval_seconds <= .05 or not .02 <= max_frame_age_seconds <= .20:
            raise ValueError("invalid manual input pump interval/frame age")
        self.hub, self.guard, self.latest_frame = hub, guard, latest_frame
        self.external_stop = stop_event
        self.health_error = health_error
        self.interval, self.max_age = interval_seconds, max_frame_age_seconds
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._snapshot_lock = threading.Lock()
        self._decision = Decision(reason="人工实机调试：等待输入泵启动")
        self._thread = None
        self._error = ""
        self._last_sent = None
        self._last_sent_at = 0.0
        self._last_command_id = None
        # Only this pump's transport callback reads this field; it must not
        # acquire a pump lock while holding the guard/watchdog transport lock.
        self.submitting_sequence = None

    @property
    def error(self):
        return self._error

    def snapshot(self):
        with self._snapshot_lock:
            return self._decision

    def start(self):
        if self._thread is not None:
            raise RuntimeError("manual input pump already started")
        self._thread = threading.Thread(target=self._run, name=self.THREAD_NAME, daemon=True)
        self._thread.start()
        # Deterministic first status without making vision the input owner.
        self._ready.wait(.25)

    def _stopped(self):
        return self._stop.is_set() or bool(self.external_stop and self.external_stop.is_set())

    def _frame_context(self):
        packet = self.latest_frame()
        now = time.perf_counter()
        stamp = getattr(packet, "timestamp", None)
        valid = (not isinstance(stamp, bool) and isinstance(stamp, (int, float))
                 and math.isfinite(stamp) and 0 <= now-stamp <= self.max_age)
        return getattr(packet, "sequence", None), (now-stamp if valid else None), valid

    def _store(self, decision):
        with self._snapshot_lock:
            self._decision = decision

    def _tick(self):
        now = time.perf_counter()
        manual = self.hub.take_manual(now)
        sequence, age, fresh = self._frame_context()
        decision = manual if manual is not None else Decision(reason="人工实机调试：等待接口指令，保持中立")
        if not fresh:
            if manual is not None and manual.state != ControllerState.neutral():
                # A capture interruption cancels the old hold. Recovery needs
                # a new request; it must not resurrect a previously held stick.
                self.hub.submit({"command":"release"})
                self.hub.take_manual(time.perf_counter())
            decision = replace(decision, state=ControllerState.neutral(), reason="手动输入泵：采集未更新，保持中立")
        if self._stopped():
            decision = replace(decision, state=ControllerState.neutral(), reason="手动输入泵：任务停止，释放控制")
        decision = replace(decision, diagnostics={**decision.diagnostics, "manual_input_pump": {
            "owner": self.THREAD_NAME, "interval_seconds": self.interval,
            "capture_sequence": sequence, "capture_age_seconds": age, "capture_fresh": fresh,
        }})
        self.submitting_sequence = sequence
        # New commands and required neutral receipts are submitted immediately.
        # Unchanged nonzero states renew the guard lease, but idle neutral
        # polling does not create 100 transport records per second.
        debug = decision.diagnostics.get("debug", {})
        command_id = debug.get("command_id")
        required = (manual is not None and (
            "transport_required" not in debug or
            (debug["transport_required"] and decision.state == ControllerState.neutral()) or
            command_id != self._last_command_id))
        keepalive = decision.state != ControllerState.neutral() and now-self._last_sent_at >= .05
        if required or keepalive or decision.state != self._last_sent:
            self.guard.submit(decision.state)
            self._last_sent = decision.state
            self._last_sent_at = time.perf_counter()
            self._last_command_id = command_id
        self._store(decision)

    def _run(self):
        try:
            while not self._stopped():
                started = time.perf_counter()
                problem = self.health_error() if self.health_error else ""
                if problem:
                    raise RuntimeError(str(problem))
                self._tick()
                self._ready.set()
                self._stop.wait(max(0.0, self.interval-(time.perf_counter()-started)))
        except Exception as error:
            self._error = f"{type(error).__name__}: {error}"
        finally:
            try:
                self.guard.submit(ControllerState.neutral())
                self._last_sent = ControllerState.neutral()
            except Exception as error:
                if not self._error:
                    self._error = f"{type(error).__name__}: {error}"
            self._store(Decision(reason="手动输入泵：已停止并尝试释放控制",
                                 diagnostics={"manual_input_pump":{"owner":self.THREAD_NAME,"stopped":True}}))
            self._ready.set()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError("手动输入泵尚未退出")
        if self._error:
            raise RuntimeError("手动输入泵失败："+self._error)
