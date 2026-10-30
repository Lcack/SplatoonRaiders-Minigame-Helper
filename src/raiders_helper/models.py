from dataclasses import dataclass, field
from enum import Enum

from .control.buttons import ControllerState


class Game(str, Enum):
    MINING = "mining"
    HOPPING = "hopping"
    JUGGLING = "juggling"
    LANDING = "landing"


@dataclass(frozen=True)
class Entity:
    """Coordinates relative to the playfield; y points DOWN; speed in units/s."""
    kind: str
    x: float
    y: float
    w: float
    h: float
    confidence: float
    track_id: int = -1
    vx: float = 0.0
    vy: float = 0.0
    samples: int = 1


@dataclass(frozen=True)
class Observation:
    game: Game
    timestamp: float
    sequence: int
    phase: str = "unknown"
    confidence: float = 0.0
    entities: tuple[Entity, ...] = ()
    notes: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)

    def of_kind(self, kind: str) -> tuple[Entity, ...]:
        return tuple(e for e in self.entities if e.kind == kind)


@dataclass(frozen=True)
class Decision:
    state: ControllerState = field(default_factory=ControllerState.neutral)
    reason: str = "等待画面"
    confidence: float = 0.0
    diagnostics: dict = field(default_factory=dict)
