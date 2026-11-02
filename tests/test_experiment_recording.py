import json
import threading
import time

import cv2
import numpy as np
import pytest

from raiders_helper.capture.models import FramePacket
from raiders_helper.control.buttons import Button, ControllerState, StickVector
from raiders_helper.control.dummy import DummyBackend
from raiders_helper.control.guard import GuardedController
from raiders_helper.experiment_recording import AsyncExperimentRecorder, RecordingBackend


def packet(sequence, shape=(48, 64, 3)):
    return FramePacket(np.full(shape, sequence, np.uint8), 10+sequence/60, sequence, 60)


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_full_frame_queue_never_waits_for_encoder_and_drop_is_mapped(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    written = []
    class Writer:
        def write(self, frame):
            entered.set()
            assert release.wait(2)
            written.append(frame.copy())
        def release(self):
            pass
    monkeypatch.setattr(AsyncExperimentRecorder, "_create_writer", lambda self, size: Writer())
    recorder = AsyncExperimentRecorder(tmp_path, queue_size=1)
    try:
        assert recorder.on_frame(packet(1))
        assert entered.wait(1)
        assert recorder.on_frame(packet(2))
        # Encode remains blocked until AFTER this second thread completes.
        result = []
        submitted = threading.Event()
        def submit():
            result.append(recorder.on_frame(packet(3)))
            submitted.set()
        thread = threading.Thread(target=submit)
        thread.start()
        assert submitted.wait(0.5)
        assert result == [False]
    finally:
        release.set()
        summary = recorder.close()
    assert summary["written_frames"] == 2 and summary["queue_full_drops"] == 1
    events = rows(recorder.frames_path)
    encoded = sorted((e["video_frame_index"], e["capture_sequence"]) for e in events if e["status"] == "written")
    assert encoded == [(0, 1), (1, 2)]
    drop = next(e for e in events if e["status"] == "dropped")
    assert drop["capture_sequence"] == 3 and drop["video_frame_index"] is None
    assert drop["capture_timestamp"] == pytest.approx(10+3/60)
    assert drop["reason"] == "queue_full"
    assert summary["complete"] is True and summary["error"] == ""


def test_encoder_failure_is_reported_without_breaking_capture_submit(tmp_path, monkeypatch):
    attempted = threading.Event()
    def fail(self, size):
        attempted.set()
        raise RuntimeError("codec unavailable")
    monkeypatch.setattr(AsyncExperimentRecorder, "_create_writer", fail)
    recorder = AsyncExperimentRecorder(tmp_path)
    assert recorder.on_frame(packet(1))
    assert attempted.wait(1)
    deadline = time.perf_counter()+1
    while not recorder.error:
        assert time.perf_counter() < deadline
        time.sleep(0.001)
    assert not recorder.on_frame(packet(2))
    summary = recorder.close()
    events = rows(recorder.frames_path)
    assert summary["written_frames"] == 0 and summary["encoding_failure_drops"] == 1
    assert {e["reason"] for e in events} == {"encoding_failed", "encoder_error"}
    assert {e["capture_sequence"] for e in events} == {1, 2}
    assert "RuntimeError" in summary["error"]


def test_mp4_written_frames_have_matching_capture_sidecar_indices(tmp_path):
    recorder = AsyncExperimentRecorder(tmp_path, queue_size=16)
    for sequence in range(1, 7):
        assert recorder.on_frame(packet(sequence))
    summary = recorder.close()
    assert summary["error"] == "" and summary["complete"]
    capture = cv2.VideoCapture(str(recorder.video_path))
    try:
        assert capture.isOpened()
        decoded = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            assert image.shape == (48, 64, 3)
            decoded += 1
    finally:
        capture.release()
    assert decoded == summary["written_frames"] == 6
    events = rows(recorder.frames_path)
    assert [e["video_frame_index"] for e in events] == list(range(decoded))
    assert [e["capture_sequence"] for e in events] == list(range(1, 7))


def test_async_recorder_keeps_input_pixels_unchanged_and_letterboxes_changes(tmp_path, monkeypatch):
    outputs = []
    class Writer:
        def write(self, frame):
            outputs.append(frame.copy())
        def release(self):
            pass
    monkeypatch.setattr(AsyncExperimentRecorder, "_create_writer", lambda self, size: Writer())
    first, second = packet(1, (101, 201, 3)), packet(2, (100, 100, 3))
    before = first.frame.copy(), second.frame.copy()
    recorder = AsyncExperimentRecorder(tmp_path, max_size=(100, 80))
    recorder.on_frame(first)
    recorder.on_frame(second)
    summary = recorder.close()
    assert summary["encoded_size"] == (100, 50)
    assert [im.shape for im in outputs] == [(50, 100, 3), (50, 100, 3)]
    assert np.array_equal(before[0], first.frame) and np.array_equal(before[1], second.frame)
    assert np.count_nonzero(outputs[1][:, :25]) == 0
    assert summary["container_fps"] == 60


def test_recording_backend_records_transport_failures_and_all_state_axes(tmp_path):
    class Backend(DummyBackend):
        last_error = "serial wire contents must not enter log"
        def send_state(self, state):
            super().send_state(state)
            return Button.A not in state.buttons
    backend = Backend()
    callback_events = []
    wrapped = RecordingBackend(backend, tmp_path, callback_events.append)
    assert wrapped.connect("not-recorded-port")
    assert wrapped.connected and wrapped.last_error == backend.last_error
    state = ControllerState(frozenset({Button.A}), StickVector(123, -456), StickVector(-789, 321))
    assert wrapped.send_state(state) is False
    assert wrapped.send_state(ControllerState.neutral()) is True
    wrapped.disconnect()
    wrapped.close()
    events = rows(wrapped.path)
    assert [e["success"] for e in events] == [False, True]
    assert events[0]["state"] == {"buttons": ["A"], "left_stick": {"x": 123, "y": -456},
                                   "right_stick": {"x": -789, "y": 321}}
    assert events[1]["state"]["buttons"] == []
    assert all(e["before"] <= e["after"] and e["thread"] == threading.current_thread().name for e in events)
    assert callback_events == events
    assert "wire contents" not in wrapped.path.read_text(encoding="utf-8")
    assert "not-recorded-port" not in wrapped.path.read_text(encoding="utf-8")


def test_backend_exception_is_audited_by_type_and_propagated(tmp_path):
    class Backend(DummyBackend):
        def send_state(self, state):
            raise OSError("secret serial payload")
    wrapped = RecordingBackend(Backend(), tmp_path)
    with pytest.raises(OSError):
        wrapped.send_state(ControllerState.neutral())
    wrapped.close()
    event = rows(wrapped.path)[0]
    assert event["success"] is False and event["exception_type"] == "OSError"
    assert "secret serial payload" not in wrapped.path.read_text(encoding="utf-8")


def test_guard_watchdog_neutral_is_recorded_below_the_guard(tmp_path):
    backend = DummyBackend()
    wrapped = RecordingBackend(backend, tmp_path)
    wrapped.connect()
    guard = GuardedController(wrapped, lease_seconds=0.03)
    try:
        guard.submit(ControllerState(frozenset({Button.B})))
        deadline = time.perf_counter()+1
        while backend.state != ControllerState.neutral():
            assert time.perf_counter() < deadline
            time.sleep(0.005)
    finally:
        guard.close()
        wrapped.close()
    events = rows(wrapped.path)
    assert events[0]["state"]["buttons"] == ["B"]
    assert any(e["state"]["buttons"] == [] and e["thread"] != threading.current_thread().name for e in events)
    assert not wrapped.connected


def test_slow_audit_disk_write_does_not_block_controller_send(tmp_path):
    entered, release = threading.Event(), threading.Event()
    wrapped = RecordingBackend(DummyBackend(), tmp_path)
    original = wrapped._file
    class SlowFile:
        def write(self, text):
            entered.set()
            assert release.wait(1)
            return original.write(text)
        def close(self):
            original.close()
    wrapped._file = SlowFile()
    wrapped.connect()
    try:
        assert wrapped.send_state(ControllerState.neutral())
        assert entered.wait(1)
        assert wrapped.send_state(ControllerState(frozenset({Button.A})))
    finally:
        release.set()
        wrapped.close()
    assert len(rows(wrapped.path)) == 2


def test_controller_audit_lines_survive_concurrent_callers(tmp_path):
    wrapped = RecordingBackend(DummyBackend(), tmp_path)
    wrapped.connect()
    def send(index):
        for i in range(12):
            assert wrapped.send_state(ControllerState(left_stick=StickVector(index*100+i, 0)))
    threads = [threading.Thread(target=send, args=(i,), name=f"producer-{i}") for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive()
    wrapped.close()
    events = rows(wrapped.path)
    assert len(events) == 48
    assert [e["command_sequence"] for e in events] == list(range(1, 49))
    assert {e["thread"] for e in events} == {f"producer-{i}" for i in range(4)}
    assert len({e["state"]["left_stick"]["x"] for e in events}) == 48
