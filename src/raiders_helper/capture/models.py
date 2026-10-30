from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    device_index: int = 0
    backend: str = "DSHOW"
    width: int = 1920
    height: int = 1080
    fps: float = 60.0


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame: Any
    timestamp: float
    sequence: int
    measured_fps: float


@dataclass(frozen=True, slots=True)
class CaptureSourceInfo:
    backend: str
    width: int
    height: int
    fps: float

