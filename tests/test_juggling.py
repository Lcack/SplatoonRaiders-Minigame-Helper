from dataclasses import replace
import math
from time import perf_counter

import pytest

from raiders_helper.control.buttons import Button, ControllerState
from raiders_helper.juggling import decide_juggling, reflected_x, time_to_contact
from raiders_helper.models import Entity, Game, Observation


def body(kind, x, y, *, track_id=0, vx=0, vy=0, w=0.05, h=0.05, samples=3, confidence=0.95):
    return Entity(kind, x, y, w, h, confidence, track_id, vx, vy, samples)


def scene(*balls, metrics=None, left=None, right=None):
    return Observation(Game.JUGGLING, 1, 1, "playing", 0.95,
                       (left or body("player", 0.25, 0.85, track_id=100),
                        right or body("partner", 0.75, 0.85, track_id=101), *balls), metrics=metrics or {})


def test_each_half_follows_its_most_urgent_ball():
    obs = scene(body("ball", 0.4, 0.7, track_id=1, vy=0.5),
                body("ball", 0.1, 0.4, track_id=2, vy=0.5),
                body("ball", 0.65, 0.7, track_id=3, vy=0.5),
                body("ball", 0.9, 0.4, track_id=4, vy=0.5))
    decision = decide_juggling(obs, {})
    assert decision.state.left_stick.x > 0
    assert decision.state.right_stick.x < 0
    assert decision.diagnostics["targets"]["left"]["track_id"] == 1
    assert decision.diagnostics["targets"]["right"]["track_id"] == 3


def test_assigns_crossing_ball_by_future_half():
    obs = scene(body("ball", 0.45, 0.6, track_id=1, vx=0.6, vy=0.5))
    decision = decide_juggling(obs, {})
    assert decision.diagnostics["targets"]["left"] is None
    assert decision.diagnostics["targets"]["right"]["target_x"] > 0.5


def test_predicts_outer_wall_reflection_before_assigning():
    obs = scene(body("ball", 0.9, 0.6, track_id=1, vx=1.5, vy=0.5))
    decision = decide_juggling(obs, {})
    target = decision.diagnostics["targets"]["left"]
    assert target is not None
    assert 0 <= target["target_x"] < 0.5
    assert reflected_x(2.2, 0, 1) == pytest.approx(0.2)


def test_optional_measured_gravity_predicts_rising_ball():
    ball = body("ball", 0.4, 0.4, track_id=1, vy=-0.3)
    obs = scene(ball, metrics={"ball_gravity_estimate": 2})
    decision = decide_juggling(obs, {})
    assert decision.diagnostics["model"] == "constant_acceleration"
    target = decision.diagnostics["targets"]["left"]
    assert target is not None and target["time_to_contact"] > 0
    assert time_to_contact(ball, obs.of_kind("player")[0], 0) is None


def test_receiving_body_top_is_used_instead_of_center():
    actor = body("player", 0.25, 0.85, h=0.1)
    ball = body("ball", 0.4, 0.7, track_id=1, vy=0.5, h=0.1)
    assert time_to_contact(ball, actor) == pytest.approx(0.1)


def test_control_latency_does_not_chase_already_missed_contact():
    obs = scene(body("ball", 0.1, 0.78, track_id=1, vy=0.5), metrics={"latency_seconds": 0.2})
    decision = decide_juggling(obs, {})
    assert decision.state == ControllerState.neutral()
    assert decision.diagnostics["late_ball_count"] == 1


def test_latency_and_actor_velocity_change_required_axis():
    actor = body("player", 0.25, 0.85, vx=0.2)
    ball = body("ball", 0.35, 0.5, track_id=1, vy=0.5)
    a = decide_juggling(scene(ball, left=actor), {"lateral_speed": 0.5})
    b = decide_juggling(scene(ball, left=actor, metrics={"latency_seconds": 0.1}), {"lateral_speed": 0.5})
    assert a.state.left_stick.x != b.state.left_stick.x


@pytest.mark.parametrize("balls", [(), (body("ball", 0.4, 0.7, track_id=1, vy=0.5, samples=1),),
                                   (body("ball", 0.4, 0.7, track_id=1, vy=0.5, confidence=0.4),)])
def test_missing_or_unreliable_tracks_are_neutral(balls):
    assert decide_juggling(scene(*balls), {}).state == ControllerState.neutral()


def test_duplicate_track_identity_waits_for_recovery():
    obs = scene(body("ball", 0.4, 0.7, track_id=1, vy=0.5), body("ball", 0.8, 0.7, track_id=1, vy=0.5))
    assert decide_juggling(obs, {}).state == ControllerState.neutral()


def test_multiple_untracked_balls_are_not_assumed_to_have_stable_identity():
    obs = scene(body("ball", 0.4, 0.7, track_id=-1, vy=0.5), body("ball", 0.8, 0.7, track_id=-1, vy=0.5))
    assert decide_juggling(obs, {}).state == ControllerState.neutral()


def test_role_mismatch_and_low_scene_confidence_are_neutral():
    ball = body("ball", 0.4, 0.7, track_id=1, vy=0.5)
    assert decide_juggling(replace(scene(ball), confidence=0.3), {}).state == ControllerState.neutral()
    assert decide_juggling(scene(ball, left=body("player", 0.75, 0.85)), {}).state == ControllerState.neutral()


def test_no_lives_and_invalid_measured_actor_speed_are_neutral():
    ball = body("ball", 0.4, 0.7, track_id=1, vy=0.5)
    assert decide_juggling(scene(ball, metrics={"lives_remaining": 0}), {}).state == ControllerState.neutral()
    assert decide_juggling(scene(ball), {"lateral_speed": -1}).state == ControllerState.neutral()


def test_slide_requires_explicit_calibration_readiness_and_reachability():
    # Synthetic movement values, not game measurements.
    ball = body("ball", 0.45, 0.7, track_id=1, vy=0.5)
    obs = scene(ball, metrics={"left_slide_ready": 1})
    parameters = {"lateral_speed": 0.1, "calibrated": True,
                  "slide": {"enabled": True, "calibrated": True, "speed": 2.0,
                            "minimum_distance": 0.1, "maximum_time_to_contact": 0.4}}
    assert Button.ZL in decide_juggling(obs, parameters).state.buttons
    assert not decide_juggling(obs, {**parameters, "calibrated": False}).state.buttons
    assert not decide_juggling(scene(ball), parameters).state.buttons
    assert not decide_juggling(obs, {**parameters, "slide": {**parameters["slide"], "speed": 0.01}}).state.buttons


@pytest.mark.parametrize("metrics", [{"latency_seconds": -1}, {"ball_gravity_estimate": math.nan},
                                     {"observation_age_seconds": math.inf}])
def test_invalid_prediction_inputs_do_not_create_actions(metrics):
    obs = scene(body("ball", 0.4, 0.7, track_id=1, vy=0.5), metrics=metrics)
    assert decide_juggling(obs, {}).state == ControllerState.neutral()


def test_decision_has_concrete_intercepts_and_urgency_without_slide_defaults():
    obs = scene(body("ball", 0.4, 0.7, track_id=1, vy=0.5))
    decision = decide_juggling(obs, {})
    assert decision.diagnostics["intercepts"]
    assert decision.diagnostics["urgency"]["left"] > 0
    assert not decision.state.buttons
    assert "接 #1" in decision.reason


def test_many_tracks_use_bounded_assignment_and_do_not_duplicate_targets():
    balls = [body("ball", 0.1+i*0.04, 0.4+i*0.005, track_id=i, vy=0.5) for i in range(20)]
    obs = scene(*balls)
    started = perf_counter()
    result = decide_juggling(obs, {})
    elapsed = perf_counter()-started
    targets = [t["track_id"] for t in result.diagnostics["targets"].values() if t]
    assert len(targets) == len(set(targets))
    # A deliberately loose hang/regression threshold, not an FPS guarantee.
    assert elapsed < 0.5
