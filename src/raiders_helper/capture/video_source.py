from __future__ import annotations

from pathlib import Path


class OpenCvVideoSource:
    def __init__(self, path: Path):
        self.path = path
        self._capture = None
        self.width = 0
        self.height = 0
        self.fps = 30.0
        self.frame_count = 0

    def open(self) -> None:
        import cv2

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"无法打开视频: {self.path}")
        self._capture = capture
        self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self):
        if self._capture is None:
            raise RuntimeError("video is not open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame

    def seek(self, frame_index: int):
        if self._capture is None:
            raise RuntimeError("video is not open")
        import cv2

        index = max(0, min(self.frame_count - 1, int(frame_index)))
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        return index

    @property
    def position(self) -> int:
        if self._capture is None:
            return 0
        import cv2

        return int(self._capture.get(cv2.CAP_PROP_POS_FRAMES))

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None

