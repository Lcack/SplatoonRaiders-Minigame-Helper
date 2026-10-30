import time
import threading
from dataclasses import replace

from .models import Decision
from .policies import decide


class Engine:
    def __init__(self, recognizer, controller=None, live=False, max_age=0.2):
        self.recognizer = recognizer
        self.controller = controller
        self.live = live
        self.max_age = max_age
        self.armed = False
        self.last_sequence = -1
        self._control_lock = threading.RLock()

    def arm(self):
        profile = self.recognizer.profile
        if not self.live or profile.get("calibrated") is not True or profile.get("synthetic") or profile.get("advisory_only"):
            raise ValueError("只有已校准的实机配置可启用控制；演示与回放只观察")
        if self.controller is None:
            raise ValueError("尚未连接控制器")
        with self._control_lock:
            self.armed = True

    def disarm(self):
        with self._control_lock:
            self.armed = False
            if self.controller:
                self.controller.release()

    def process(self, packet, now=None):
        try:
            started = time.perf_counter()
            obs = self.recognizer.observe(packet)
            recognized = time.perf_counter()
            age = ((recognized if now is None else now)-packet.timestamp) if self.live else 0.0
            obs = replace(obs, metrics={**obs.metrics, "observation_age_seconds": max(0, age)})
            decision = decide(obs, self.recognizer.profile.get("policy", {}))
            planned = time.perf_counter()
            timing = {"recognition_ms": (recognized-started)*1000,
                      "decision_ms": (planned-recognized)*1000,
                      "analysis_ms": (planned-started)*1000}
            with self._control_lock:
                # Freshness must be checked at the actual submission boundary,
                # after any lock wait, not just when planning finished.
                current = time.perf_counter() if now is None else now
                if self.live and (packet.sequence <= self.last_sequence or not 0 <= current-packet.timestamp <= self.max_age):
                    decision = Decision(reason="重复帧或画面已过期，释放控制")
                self.last_sequence = packet.sequence
                timing["frame_age_ms"] = max(0,current-packet.timestamp)*1000 if self.live else None
                if self.armed:
                    send_started=time.perf_counter()
                    self.controller.submit(decision.state)
                    timing["controller_send_ms"]=(time.perf_counter()-send_started)*1000
                decision = replace(decision, diagnostics={**decision.diagnostics, "timing": timing})
            return obs, decision
        except Exception:
            self.disarm()
            raise
