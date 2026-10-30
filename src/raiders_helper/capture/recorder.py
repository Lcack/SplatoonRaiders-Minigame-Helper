from __future__ import annotations

from datetime import datetime
from pathlib import Path


class VideoRecorder:
    def __init__(self):
        self._writer = None
        self.path: Path | None = None

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def start(self, target_dir: Path, frame_shape, fps: float = 30.0) -> Path:
        import cv2

        if self.recording:
            raise RuntimeError("video recorder is already running")
        if len(frame_shape) < 2:
            raise ValueError("invalid frame shape")
        target_dir.mkdir(parents=True, exist_ok=True)
        height, width = frame_shape[:2]
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidates = (
            (target_dir / f"session-{stamp}.mp4", "mp4v"),
            (target_dir / f"session-{stamp}.avi", "MJPG"),
        )
        for path, codec in candidates:
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*codec),
                fps,
                (width, height),
            )
            if writer.isOpened():
                self._writer = writer
                self.path = path
                return path
            writer.release()
        raise RuntimeError("无法创建视频编码器")

    def write(self, frame) -> None:
        if self._writer is None:
            raise RuntimeError("video recorder is not running")
        self._writer.write(frame)

    def stop(self) -> Path | None:
        path = self.path
        if self._writer is not None:
            self._writer.release()
        self._writer = None
        self.path = None
        return path

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

