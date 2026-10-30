from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from raiders_helper.control.buttons import Button, ControllerState


@dataclass(frozen=True, slots=True)
class SerialPortInfo:
    device: str
    description: str


class ControllerBackend(ABC):
    @property
    @abstractmethod
    def connected(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def connect(self, port: str | None = None) -> bool:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def send_state(self, state: ControllerState) -> bool:
        raise NotImplementedError

    def pulse(self, buttons: set[Button], duration_seconds: float = 0.12) -> bool:
        state = ControllerState(buttons=frozenset(buttons))
        if not self.send_state(state):
            return False
        import time

        time.sleep(max(0.0, duration_seconds))
        return self.send_state(ControllerState.neutral())

