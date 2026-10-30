import json
import time
from dataclasses import replace

import cv2
import numpy as np
import pytest

from raiders_helper.capture.models import CaptureConfig, FramePacket
from raiders_helper.capture.recorder import VideoRecorder
from raiders_helper.capture.video_source import OpenCvVideoSource
from raiders_helper.capture.worker import CaptureWorker
from raiders_helper.cli import main
from raiders_helper.control.buttons import Button, ControllerState, StickVector
from raiders_helper.control.dummy import DummyBackend
from raiders_helper.control.guard import GuardedController
from raiders_helper.demo import demo_packet, demo_profile
from raiders_helper.models import Entity, Game, Observation
from raiders_helper.policies import decide
from raiders_helper.runtime import Engine
from raiders_helper.vision import Recognizer, Tracker, crop_roi, load_profile


@pytest.mark.parametrize("game", list(Game))
def test_synthetic_pipeline_and_trace(tmp_path, game):
    path = tmp_path / f"{game.value}.jsonl"
    assert main(["demo", "--game", game.value, "--frames", "20", "--output", str(path)]) == 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 20
    assert all(not row["executed"] for row in rows)
    assert rows[-1]["observation"]["phase"] == "playing"
    assert rows[-1]["observation"]["confidence"] >= 0.8
    assert any(e["samples"] >= 2 for e in rows[-1]["observation"]["entities"])


def test_blank_frame_cannot_produce_input():
    recognizer = Recognizer(Game.MINING, demo_profile(Game.MINING))
    for i in range(5):
        obs = recognizer.observe(FramePacket(np.zeros((480, 640, 3), np.uint8), i / 30, i, 30))
        assert obs.phase == "unknown"
        assert decide(obs).state == ControllerState.neutral()


def test_tracking_velocity_and_scene_cut_reset():
    tracker = Tracker()
    entity = Entity("ball", 0.3, 0.3, 0.1, 0.1, 0.9)
    first = tracker.update([entity], 1)[0]
    second = tracker.update([replace(entity, x=0.4)], 1.1)[0]
    assert second.track_id == first.track_id
    assert second.vx == pytest.approx(1)
    after_cut = tracker.update([entity], 3)[0]
    assert after_cut.samples == 1 and after_cut.vx == 0


def test_tracker_does_not_assign_same_identity_twice():
    tracker = Tracker()
    e = Entity("ball", 0.3, 0.3, 0.1, 0.1, 0.9)
    tracker.update([e], 0)
    tracked = tracker.update([e, replace(e, x=0.31)], 0.1)
    assert len({e.track_id for e in tracked}) == 2


def test_template_detection_at_normalized_roi(tmp_path):
    rng = np.random.default_rng(4)
    template = rng.integers(0, 255, (12, 15, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", template)
    assert ok
    (tmp_path / "sprite.png").write_bytes(encoded.tobytes())
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[30:42, 40:55] = template
    profile = {"_root": str(tmp_path), "reference_size": [100, 100],
               "detectors": [{"kind": "player", "method": "template", "path": "sprite.png",
                              "roi": [0.2, 0.2, 0.6, 0.6], "threshold": 0.99}]}
    obs = Recognizer(Game.MINING, profile).observe(FramePacket(frame, 0, 1, 30))
    assert len(obs.entities) == 1
    assert obs.entities[0].x == pytest.approx(0.475)
    assert obs.entities[0].y == pytest.approx(0.36)
    assert obs.phase == "unknown"  # objects alone must not imply gameplay


def mining_obs(*extras):
    player = Entity("player", 0.5, 0.3, 0.05, 0.05, 0.95)
    road = Entity("corridor", 0.5, 0.4, 0.8, 0.01, 0.95)
    return Observation(Game.MINING, 1, 1, "playing", 0.95, (player, road, *extras))


def test_mining_egg_allows_rocks_but_not_walls():
    rock = Entity("obstacle", 0.5, 0.4, 0.1, 0.1, 0.95)
    buff = Entity("egg_buff", 0.1, 0.1, 0.05, 0.05, 0.95)
    assert Button.ZR not in decide(mining_obs(rock)).state.buttons
    assert Button.ZR in decide(mining_obs(rock, buff)).state.buttons
    assert Button.ZR not in decide(mining_obs(replace(rock, kind="wall"), buff)).state.buttons


def test_mining_never_targets_outside_detected_corridor():
    obs = mining_obs(Entity("egg_buff", 0.1, 0.1, 0.05, 0.05, 0.95))
    road = Entity("corridor", 0.25, 0.4, 0.25, 0.01, 0.95)
    obs = replace(obs, entities=(obs.entities[0], road, obs.entities[-1]))
    decision = decide(obs)
    assert decision.state.left_stick.x < 0
    assert Button.ZR not in decision.state.buttons
    missing_road = replace(obs, entities=(obs.entities[0],))
    assert decide(missing_road).state == ControllerState.neutral()


def test_zero_countdown_stops_even_when_road_clear():
    assert decide(replace(mining_obs(), metrics={"time_remaining": 0})).state == ControllerState.neutral()
    assert decide(replace(mining_obs(), metrics={"time_remaining": 40})).state != ControllerState.neutral()


def test_numeric_readout_rejects_unreadable_slots(tmp_path):
    rng = np.random.default_rng(8)
    digits = {}
    frame = np.zeros((50, 100, 3), np.uint8)
    for digit in (0, 4):
        tile = rng.integers(0, 255, (10, 10), dtype=np.uint8)
        ok, data = cv2.imencode(".png", tile)
        assert ok
        (tmp_path / f"{digit}.png").write_bytes(data.tobytes())
        digits[str(digit)] = f"{digit}.png"
        for x in ([10] if digit == 4 else [30, 50]):
            frame[0:10, x:x+10] = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    reader = {"name": "time_remaining", "decimals": 1, "templates": digits,
              "slots": [[0.1, 0, 0.1, 0.2], [0.3, 0, 0.1, 0.2], [0.5, 0, 0.1, 0.2]]}
    recognizer = Recognizer(Game.MINING, {"_root": str(tmp_path), "reference_size": [100, 50], "readouts": [reader]})
    assert recognizer.read_metrics(frame) == {"time_remaining": 40}
    frame[0:10, 30:40] = 0
    assert recognizer.read_metrics(frame) == {}


def test_juggling_respects_left_right_partition():
    obs = Observation(Game.JUGGLING, 1, 1, "playing", 0.95, (
        Entity("player", 0.4, 0.8, 0.1, 0.1, 0.95),
        Entity("partner", 0.8, 0.8, 0.1, 0.1, 0.95),
        Entity("ball", 0.6, 0.7, 0.05, 0.05, 0.95, vy=0.5, samples=3),
    ))
    state = decide(obs).state
    assert state.left_stick.x == 0
    assert state.right_stick.x < 0


def test_watchdog_releases_without_further_frames():
    backend = DummyBackend()
    backend.connect()
    guard = GuardedController(backend, lease_seconds=0.03)
    try:
        guard.submit(ControllerState(left_stick=StickVector(32767, 0)))
        deadline = time.perf_counter() + 1
        while backend.state != ControllerState.neutral() and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert backend.state == ControllerState.neutral()
    finally:
        guard.close()
    assert not backend.connected


def test_replay_and_synthetic_profiles_cannot_arm():
    engine = Engine(Recognizer(Game.MINING, demo_profile(Game.MINING)), live=True)
    with pytest.raises(ValueError):
        engine.arm()
    engine = Engine(Recognizer(Game.MINING, {"calibrated": True}), live=False)
    with pytest.raises(ValueError):
        engine.arm()


def test_live_stale_and_duplicate_frames_release():
    profile = demo_profile(Game.MINING)
    profile.update(calibrated=True, synthetic=False, stable_frames=1)
    backend = DummyBackend()
    backend.connect()
    guard = GuardedController(backend)
    engine = Engine(Recognizer(Game.MINING, profile), guard, live=True)
    try:
        engine.arm()
        packet = demo_packet(Game.MINING, 1, timestamp=1)
        _, decision = engine.process(packet, now=1.01)
        assert decision.state != ControllerState.neutral()
        _, decision = engine.process(packet, now=1.02)
        assert decision.state == ControllerState.neutral()
        _, decision = engine.process(replace(packet, sequence=2), now=2)
        assert decision.state == ControllerState.neutral()
        assert backend.state == ControllerState.neutral()
    finally:
        guard.close()


def test_capture_lifecycle_and_restart():
    sources = []
    class Source:
        def __init__(self, config):
            self.closed = False
            sources.append(self)
        def open(self): pass
        def read(self):
            time.sleep(0.005)
            return np.zeros((48, 64, 3), np.uint8)
        def close(self): self.closed = True
    worker = CaptureWorker(CaptureConfig(), source_factory=Source)
    for _ in range(2):
        worker.start()
        deadline = time.perf_counter() + 1
        while worker.latest() is None and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert worker.latest() is not None
        worker.stop()
    assert all(s.closed for s in sources)


def test_record_replay_extract_roundtrip(tmp_path):
    recorder = VideoRecorder()
    path = recorder.start(tmp_path, (480, 640, 3), fps=10)
    try:
        for i in range(5):
            recorder.write(demo_packet(Game.HOPPING, i).frame)
    finally:
        recorder.stop()
    output = tmp_path / "video.jsonl"
    assert main(["replay", str(path), "--game", "hopping", "--output", str(output)]) == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 5
    target = tmp_path / "frames"
    assert main(["extract", str(path), "--end", "0.5", "--step", "0.1", "--output", str(target)]) == 0
    assert len(json.loads((target / "index.json").read_text(encoding="utf-8"))) == 5


def test_bad_profile_and_roi(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text('{"version":1,"game":"hopping"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile(path, Game.MINING)
    with pytest.raises(ValueError):
        crop_roi(np.zeros((10, 10, 3)), [0.5, 0, 1, 1])
