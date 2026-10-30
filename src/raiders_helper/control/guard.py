"""Short command leases: stalled perception must not leave a stick held."""
import threading
import time

from .buttons import ControllerState


class GuardedController:
    def __init__(self, backend, lease_seconds=0.20):
        if not 0.02 <= lease_seconds <= 1.0:
            raise ValueError("invalid command lease")
        self.backend = backend
        self.lease_seconds = lease_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._deadline = 0.0
        self._active = False
        self.error = ""
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def submit(self, state):
        with self._lock:
            if self._stop.is_set() or self.error:
                raise RuntimeError(self.error or "控制器已关闭")
            if not self.backend.send_state(state):
                self.error = getattr(self.backend, "last_error", "") or "控制器发送失败"
                self.backend.send_state(ControllerState.neutral())
                raise RuntimeError(self.error)
            self._active = state != ControllerState.neutral()
            self._deadline = time.perf_counter() + self.lease_seconds

    def release(self):
        with self._lock:
            self._active = False
            if not self.backend.send_state(ControllerState.neutral()):
                self.error = getattr(self.backend, "last_error", "") or "归中失败"

    def _watch(self):
        while not self._stop.wait(0.01):
            with self._lock:
                if self._active and time.perf_counter() >= self._deadline:
                    self._active = False
                    if not self.backend.send_state(ControllerState.neutral()):
                        self.error = "超时归中失败"

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
        try:
            self.release()
        finally:
            self.backend.disconnect()
