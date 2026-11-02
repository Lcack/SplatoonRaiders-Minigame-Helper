from types import SimpleNamespace
import time
import tkinter as tk

import pytest

from raiders_helper import debug_api
from raiders_helper.control import debug_ui
from raiders_helper.control.buttons import Button, ControllerState, StickVector
from raiders_helper.control.live_debug_ui import LiveDebugWindow, LiveInputRelay


@pytest.fixture
def live(tmp_path, monkeypatch):
    clock = SimpleNamespace(value=100.0)
    real_time = time.time
    monkeypatch.setattr(debug_api, "time", SimpleNamespace(perf_counter=lambda: clock.value, time=real_time))
    hub = debug_api.DebugHub(tmp_path, control_allowed=True)
    yield hub, clock
    hub.close()


def owner(hub, clock):
    decision = hub.take_manual(clock.value)
    if decision is not None:
        hub.record_control(decision.state, timestamp=clock.value, source="simulated-transport")
    return decision.state if decision else None


def pause(hub, clock):
    hub.submit({"command": "pause"})
    assert owner(hub, clock) == ControllerState.neutral()


def test_first_press_waits_for_pause_then_sends_without_waiting_for_release(live):
    hub, clock = live
    relay = LiveInputRelay(lambda: hub)
    state = ControllerState(frozenset({Button.A}))
    relay.request(state, clock.value)
    assert hub.paused is False
    assert owner(hub, clock) == ControllerState.neutral()
    clock.value += .01
    relay.tick(clock.value)
    assert owner(hub, clock) == state
    assert relay.desired == state


def test_already_paused_press_is_queued_immediately_on_request(live):
    hub, clock = live
    pause(hub, clock)
    relay = LiveInputRelay(lambda: hub)
    state = ControllerState(frozenset({Button.DPAD_LEFT}))
    assert relay.request(state, clock.value)
    # No relay timer and no button-up event have occurred.
    assert owner(hub, clock) == state


def test_held_input_renews_without_any_neutral_gap(live):
    hub, clock = live
    pause(hub, clock)
    relay = LiveInputRelay(lambda: hub)
    state = ControllerState(frozenset({Button.A}), StickVector(12000, 0))
    relay.request(state, clock.value)
    assert owner(hub, clock) == state
    for _ in range(30):
        clock.value += .02
        relay.tick(clock.value)
        assert owner(hub, clock) == state
    relay.release()
    assert owner(hub, clock) == ControllerState.neutral()
    assert hub.paused


def test_partial_button_release_and_drag_replace_full_state_without_neutral(live):
    hub, clock = live
    pause(hub, clock)
    relay = LiveInputRelay(lambda: hub)
    relay.request(ControllerState(frozenset({Button.A, Button.B}), StickVector(4000, 0)), clock.value)
    owner(hub, clock)
    clock.value += .01
    updated = ControllerState(frozenset({Button.B}), StickVector(-18000, 9000))
    relay.request(updated, clock.value)
    assert owner(hub, clock) == updated
    relay.request(ControllerState.neutral(), clock.value)
    assert owner(hub, clock) == ControllerState.neutral()


def test_release_before_pause_ack_cancels_waiting_press(live):
    hub, clock = live
    relay = LiveInputRelay(lambda: hub)
    relay.request(ControllerState(frozenset({Button.A})), clock.value)
    relay.release()
    assert owner(hub, clock) == ControllerState.neutral()
    clock.value += .02
    relay.tick(clock.value)
    assert owner(hub, clock) == ControllerState.neutral()


def test_stall_expires_lease_and_old_desired_does_not_restart(live):
    hub, clock = live
    pause(hub, clock)
    relay = LiveInputRelay(lambda: hub)
    relay.request(ControllerState(frozenset({Button.A})), clock.value)
    relay.tick(clock.value)
    owner(hub, clock)
    clock.value += .20
    assert owner(hub, clock) == ControllerState.neutral()
    relay.tick(clock.value)
    assert owner(hub, clock) == ControllerState.neutral()
    clock.value += .01
    relay.tick(clock.value)
    assert relay.desired == ControllerState.neutral()
    assert owner(hub, clock) == ControllerState.neutral()


def test_task_hub_change_releases_old_and_does_not_replay_held_state(live, tmp_path):
    old, clock = live
    pause(old, clock)
    selected = [old]
    relay = LiveInputRelay(lambda: selected[0])
    relay.request(ControllerState(frozenset({Button.B})), clock.value)
    owner(old, clock)
    new = debug_api.DebugHub(tmp_path/"new", control_allowed=True)
    try:
        selected[0] = new
        clock.value += .01
        relay.tick(clock.value)
        assert owner(old, clock) == ControllerState.neutral()
        assert owner(new, clock) is None
        assert relay.desired == ControllerState.neutral()
    finally:
        new.close()


def test_readonly_task_cannot_be_used_for_manual_input(live, tmp_path):
    _, clock = live
    hub = debug_api.DebugHub(tmp_path/"readonly", control_allowed=False)
    try:
        pause(hub, clock)
        relay = LiveInputRelay(lambda: hub)
        assert relay.request(ControllerState(frozenset({Button.A})), clock.value) is False
        assert relay.desired == ControllerState.neutral()
        assert owner(hub, clock) == ControllerState.neutral()
    finally:
        hub.close()


@pytest.fixture
def window(live, monkeypatch):
    hub, clock = live
    root = tk.Tk(); root.withdraw()
    def forbidden_backend(*args, **kwargs):
        raise AssertionError("shared live window must not open another serial backend")
    monkeypatch.setattr(debug_ui, "EasyConBackend", forbidden_backend)
    view = LiveDebugWindow(root, lambda: hub)
    view.root.withdraw()
    yield view, hub, clock
    view.close()
    root.destroy()


def test_window_reuses_old_groups_and_sends_on_down_not_click_release(window):
    view, hub, clock = window
    pause(hub, clock)
    assert isinstance(view.left_stick, debug_ui.VirtualStick)
    assert isinstance(view.right_stick, debug_ui.VirtualStick)
    assert set(view.button_widgets) == set(Button)
    assert (Button.ZL, Button.ZR) in view.combo_widgets
    assert view.button_widgets[Button.A].bind("<ButtonPress-1>")
    assert view.button_widgets[Button.A].bind("<ButtonRelease-1>")
    view._button_down(Button.A)
    assert owner(hub, clock).buttons == frozenset({Button.A})
    view._button_up(Button.A)
    assert owner(hub, clock) == ControllerState.neutral()


def test_focus_loss_clears_buttons_sticks_and_sends_release(window, monkeypatch):
    view, hub, clock = window
    pause(hub, clock)
    view._button_down(Button.ZL)
    owner(hub, clock)
    monkeypatch.setattr(view.root, "focus_displayof", lambda: None)
    view._release_if_unfocused()
    assert not view.pressed_buttons
    assert view.left_stick.value == StickVector()
    assert view.right_stick.value == StickVector()
    assert owner(hub, clock) == ControllerState.neutral()


def test_close_requests_release_and_cancels_window_timer(window):
    view, hub, clock = window
    pause(hub, clock)
    view._button_down(Button.A)
    owner(hub, clock)
    view.close()
    assert view.closed
    assert owner(hub, clock) == ControllerState.neutral()
