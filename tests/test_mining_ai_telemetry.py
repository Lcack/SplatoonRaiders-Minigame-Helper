"""Heartbeat filesystem contention must not abort a real-time round."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import mining_ai_telemetry as telemetry


@pytest.fixture
def reporter(tmp_path, monkeypatch):
    clock = SimpleNamespace(wall=100., monotonic=10.)
    monkeypatch.setattr(telemetry.time, "time", lambda: clock.wall)
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: clock.monotonic)
    monkeypatch.setattr(telemetry.time, "sleep", lambda *_: pytest.fail("heartbeat must not sleep"))
    value = telemetry.ActiveAIReporter({"loaded_version": "version-a", "capture_session": "live-one"},
                                       path=tmp_path / "active_ai.json")
    return value, clock


def test_denied_replace_keeps_old_file_then_retries_latest_state_without_sleep(reporter, monkeypatch):
    value, clock = reporter
    assert value.publish(force=True, sequence=1)
    old = value.path.read_bytes()
    replace = Path.replace
    calls = []

    def denied_once(path, target):
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError(5, "Access is denied")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", denied_once)
    clock.wall += .5; clock.monotonic += .5
    assert value.publish(sequence=2) is False
    assert value.path.read_bytes() == old
    assert value.write_failures == 1 and value.last_write_error
    assert value.last_write == 100.
    # No busy retry at control-frame frequency; updates are still retained.
    clock.wall += .1; clock.monotonic += .1
    assert value.publish(sequence=3) is False
    assert len(calls) == 1
    clock.wall += .5; clock.monotonic += .5
    assert value.publish(sequence=4)
    data = json.loads(value.path.read_text(encoding="utf8"))
    assert data["sequence"] == 4 and data["updated_unix"] == clock.wall
    assert data["loaded_version"] == "version-a" and data["capture_session"] == "live-one"
    assert value.last_write_error is None


def test_denied_first_heartbeat_does_not_abort_and_close_publishes_stopped(reporter, monkeypatch):
    value, _ = reporter
    replace = Path.replace
    calls = []

    def denied_once(path, target):
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError(5, "Access is denied")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", denied_once)
    with value:
        assert not value.path.exists()
    data = json.loads(value.path.read_text(encoding="utf8"))
    assert data["running"] is False and data["mode"] == "stopped"
    assert data["loaded_version"] == "version-a"


def test_close_retries_transient_replace_failure_without_sleep(reporter, monkeypatch):
    value, _ = reporter
    assert value.publish(force=True)
    replace = Path.replace
    calls = []

    def denied_twice(path, target):
        calls.append(path)
        if len(calls) <= 2:
            raise PermissionError(5, "Access is denied")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", denied_twice)
    assert value.close()
    assert len(calls) == 3
    assert json.loads(value.path.read_text(encoding="utf8"))["running"] is False


def test_persistent_denial_is_bounded_and_old_heartbeat_not_refreshed(reporter, monkeypatch):
    value, clock = reporter
    assert value.publish(force=True)
    old = value.path.read_bytes()
    calls = []

    def denied(path, target):
        calls.append(path)
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "replace", denied)
    clock.wall += 4; clock.monotonic += 4
    assert value.close() is False
    assert len(calls) == 3 and value.path.read_bytes() == old
    assert value.state["running"] is False
    assert json.loads(old)["updated_unix"] == 100.


def test_permission_error_opening_temp_file_is_also_best_effort(reporter, monkeypatch):
    value, _ = reporter
    def denied(*args, **kwargs):
        raise PermissionError(5, "Access is denied")
    monkeypatch.setattr(Path, "write_text", denied)
    assert value.publish(force=True) is False
    assert value.write_failures == 1


@pytest.mark.parametrize("error", [OSError(28, "No space"), RuntimeError("bug")])
def test_non_permission_errors_are_not_hidden(reporter, monkeypatch, error):
    value, _ = reporter
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(type(error)):
        value.publish(force=True)
    assert value.write_failures == 0


@pytest.mark.parametrize("invalid", [object(), float("nan")])
def test_invalid_metadata_remains_a_visible_error(reporter, invalid):
    value, _ = reporter
    with pytest.raises((TypeError, ValueError)):
        value.publish(force=True, invalid=invalid)


def test_wall_clock_change_does_not_prevent_recovery_retry(reporter):
    value, clock = reporter
    assert value.publish(force=True)
    clock.wall -= 60; clock.monotonic += .5
    assert value.publish(sequence=2)
    assert json.loads(value.path.read_text(encoding="utf8"))["updated_unix"] == 40.
