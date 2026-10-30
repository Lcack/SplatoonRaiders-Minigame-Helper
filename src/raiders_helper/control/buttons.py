from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Button(str, Enum):
    Y = "Y"
    X = "X"
    B = "B"
    A = "A"
    L = "L"
    R = "R"
    ZL = "ZL"
    ZR = "ZR"
    MINUS = "MINUS"
    PLUS = "PLUS"
    L_STICK_PRESS = "L_STICK_PRESS"
    R_STICK_PRESS = "R_STICK_PRESS"
    HOME = "HOME"
    CAPTURE = "CAPTURE"
    DPAD_UP = "DPAD_UP"
    DPAD_DOWN = "DPAD_DOWN"
    DPAD_LEFT = "DPAD_LEFT"
    DPAD_RIGHT = "DPAD_RIGHT"


class Stick(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


@dataclass(frozen=True, slots=True)
class StickVector:
    x: int = 0
    y: int = 0

    def __post_init__(self) -> None:
        if not -32767 <= self.x <= 32767 or not -32767 <= self.y <= 32767:
            raise ValueError("stick coordinates must be within -32767..32767")


@dataclass(frozen=True, slots=True)
class ControllerState:
    buttons: frozenset[Button] = field(default_factory=frozenset)
    left_stick: StickVector = field(default_factory=StickVector)
    right_stick: StickVector = field(default_factory=StickVector)

    @classmethod
    def neutral(cls) -> "ControllerState":
        return cls()

    def with_buttons(self, buttons: frozenset[Button]) -> "ControllerState":
        return ControllerState(
            buttons=buttons,
            left_stick=self.left_stick,
            right_stick=self.right_stick,
        )

