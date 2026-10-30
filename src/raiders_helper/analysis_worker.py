"""Analyze the latest capture packet independently of GUI rendering speed."""
import threading
import time


class AnalysisWorker:
    def __init__(self, engine, latest_packet):
        self.engine, self.latest_packet = engine, latest_packet
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._result = None
        self.error = ""
        self.dropped_frames = 0

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def latest(self):
        with self._lock:
            return self._result

    def _run(self):
        sequence = -1
        try:
            while not self._stop.is_set():
                packet = self.latest_packet()
                if packet is None or packet.sequence == sequence:
                    self._stop.wait(0.001)
                    continue
                if sequence >= 0:
                    self.dropped_frames += max(0, packet.sequence-sequence-1)
                sequence = packet.sequence
                if self.engine.live and time.perf_counter()-packet.timestamp > self.engine.max_age:
                    self.engine.disarm()
                    self._stop.wait(0.005)
                    continue
                observation, decision = self.engine.process(packet)
                with self._lock:
                    self._result = (packet, observation, decision)
        except Exception as error:
            self.error = str(error)
            self.engine.disarm()

    def stop(self):
        self._stop.set()
        # Revoke permission before waiting for in-flight vision; even a hung
        # recognizer must not keep the old engine armed after a stop request.
        self.engine.disarm()
        if self._thread:
            self._thread.join(timeout=1)
            if self._thread.is_alive():
                raise RuntimeError("分析线程尚未退出")
            self._thread = None
