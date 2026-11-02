"""Latest-frame worker adapted from AutoTableTurf; source owned by its thread."""
import threading
import time
from collections import deque

from .models import CaptureConfig, FramePacket
from .opencv_source import OpenCvFrameSource


class CaptureWorker:
    def __init__(self, config: CaptureConfig, source_factory=OpenCvFrameSource, on_packet=None):
        self.config = config
        self._factory = source_factory
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._packet = None
        self._error = ""
        self.on_packet = on_packet

    @property
    def error(self):
        with self._lock:
            return self._error

    def latest(self):
        with self._lock:
            return self._packet

    def start(self):
        if self._thread and self._thread.is_alive():
            raise RuntimeError("采集线程尚未停止")
        with self._lock:
            self._packet, self._error = None, ""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
            # A blocked device read must finish on its owner thread. Do not
            # release VideoCapture concurrently or restart over that thread.
            if self._thread.is_alive():
                raise RuntimeError("采集设备读取尚未退出；稍后重试停止")
            self._thread = None

    def _run(self):
        source = self._factory(self.config)
        times = deque(maxlen=120)
        sequence = 0
        try:
            source.open()
            while not self._stop.is_set():
                frame = source.read()
                if frame is None:
                    raise RuntimeError("采集源已结束")
                now = time.perf_counter()
                times.append(now)
                sequence += 1
                fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0
                packet = FramePacket(frame, now, sequence, fps)
                with self._lock:
                    self._packet = packet
                if self.on_packet:
                    self.on_packet(packet)
        except Exception as error:
            with self._lock:
                self._error = str(error)
        finally:
            source.close()
