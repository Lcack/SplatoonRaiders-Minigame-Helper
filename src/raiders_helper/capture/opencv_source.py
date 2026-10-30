from __future__ import annotations

import time
from typing import Any

from raiders_helper.capture.models import CaptureConfig, CaptureSourceInfo


class CaptureError(RuntimeError):
    pass


class OpenCvFrameSource:
    BACKENDS = {
        "DSHOW": 700,
        "MSMF": 1400,
        "ANY": 0,
    }

    def __init__(self, config: CaptureConfig):
        self.config = config
        self._capture: Any | None = None
        self._info: CaptureSourceInfo | None = None

    @property
    def info(self) -> CaptureSourceInfo | None:
        return self._info

    def open(self) -> None:
        try:
            import cv2
        except ImportError as error:
            raise CaptureError("缺少 OpenCV，请安装 vision extra") from error
        backend = self.BACKENDS.get(self.config.backend.upper())
        if backend is None:
            raise CaptureError(f"不支持的采集后端: {self.config.backend}")

        started = time.perf_counter()
        capture = cv2.VideoCapture(self.config.device_index, backend)
        if not capture.isOpened():
            capture.release()
            raise CaptureError(
                f"无法打开采集设备 {self.config.device_index} ({self.config.backend})"
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps > 0:
            capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise CaptureError("采集设备已打开，但无法读取首帧")
        elapsed = time.perf_counter() - started
        if elapsed > 3:
            # The frame is usable, but the caller may still record the delay.
            pass
        self._capture = capture
        self._info = CaptureSourceInfo(
            backend=self.config.backend,
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
        )

    def read(self):
        if self._capture is None:
            raise CaptureError("采集设备未打开")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CaptureError("采集帧读取失败")
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None

