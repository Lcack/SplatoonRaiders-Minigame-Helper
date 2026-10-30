from __future__ import annotations

from collections import deque

from raiders_helper.control.backend import ControllerBackend
from raiders_helper.control.buttons import ControllerState


class DummyBackend(ControllerBackend):
    def __init__(self, history_limit: int = 300):
        self._connected = False
        self._state = ControllerState.neutral()
        self._history: deque[ControllerState] = deque(maxlen=history_limit)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def state(self) -> ControllerState:
        return self._state

    @property
    def history(self) -> tuple[ControllerState, ...]:
        return tuple(self._history)

    def connect(self, port: str | None = None) -> bool:
        del port
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._state = ControllerState.neutral()
        self._history.append(self._state)
        self._connected = False

    def send_state(self, state: ControllerState) -> bool:
        self._state = state
        self._history.append(state)
        return self._connected

