"""Real-window timestamps and actual submitted states; no hardware access."""
from dataclasses import replace
import math
from pathlib import Path
import sys

import pytest

from raiders_helper.mining_forward import ForwardDynamics
from raiders_helper.mining_forward_seed import ForwardSeedEstimator
from raiders_helper.models import Entity, Game, Observation

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from mining_supervisor import SubmittedInputHistory


PARAMETERS = {"effective_delay_seconds": .125, "dispatch_delay_seconds": .03,
              "forward_dynamics": {"enabled": True, "zr_acceleration": .115745857,
                                   "normal_speed": .25, "normal_tau_seconds": .74}}


def observation(timestamp=10., *, measured=True, reference=None, changes=None, phase="playing", rows=None):
    metrics = {"mining_input_history_source": "guard_submission_after",
               "mining_input_history": rows if rows is not None else [{"after": 9., "axis": 0., "zr": True}],
               "mining_input_history_observed_until": timestamp + .08,
               "mining_input_history_gap_detected": False}
    if measured:
        metrics.update(current_forward_speed=.30, current_forward_speed_observed=True,
                       current_forward_speed_source="tracked_landmark_consensus",
                       current_forward_speed_units="viewport_heights_per_second",
                       current_forward_speed_confidence=.85,
                       current_forward_speed_reference_timestamp=timestamp - .05 if reference is None else reference,
                       current_forward_speed_window_seconds=.10,
                       current_forward_speed_method="paired_positions",
                       current_forward_speed_reference_kind="pair_midpoint_velocity_approximation")
    metrics.update(changes or {})
    return Observation(Game.MINING, timestamp, int(timestamp * 1000), phase, .95, metrics=metrics)


def test_real_window_reference_is_propagated_to_capture_without_relabelling_observation():
    estimator = ForwardSeedEstimator(PARAMETERS)
    obs = observation(changes={"planning_delay_seconds": .05})
    estimator.attach(obs)
    seed = obs.metrics["mining_forward_seed"]
    assert seed["speed"] == pytest.approx(.30 + .115745857 * .05)
    assert seed["timestamp"] == 10.
    assert seed["reference_timestamp"] == 9.95
    assert seed["age_seconds"] == pytest.approx(.05)
    assert seed["source"] == "observed_window_propagated"
    assert seed["confidence_kind"] == "model_estimate_not_observation"
    assert obs.metrics["current_forward_speed"] == .30
    assert obs.metrics["current_forward_speed_observed"] is True
    assert obs.metrics["planning_delay_seconds"] == .05


def test_delayed_actual_mode_boundary_is_integrated_piecewise_not_current_advice():
    estimator = ForwardSeedEstimator(PARAMETERS)
    # ZR report at 9.85 acts at 9.975; the last 25 ms of this window use ZR.
    rows = [{"after": 9., "axis": .4, "zr": False}, {"after": 9.85, "axis": -.7, "zr": True},
            {"after": 9.95, "axis": 0., "zr": False}]
    obs = observation(rows=rows)
    estimator.attach(obs)
    model = ForwardDynamics(.115745857, .25, .74)
    _, middle = model.advance(.30, False, .025)
    _, expected = model.advance(middle, True, .025)
    assert obs.metrics["mining_forward_seed"]["speed"] == pytest.approx(expected)


def test_stored_backend_state_and_normal_deceleration_are_supported():
    state = {"buttons": [], "left_stick": {"x": -18000, "y": 0}, "right_stick": {"x": 0, "y": 0}}
    obs = observation(rows=[{"after": 9., "state": state}])
    ForwardSeedEstimator(PARAMETERS).attach(obs)
    assert obs.metrics["mining_forward_seed"]["speed"] == pytest.approx(.25 + .05 * math.exp(-.05 / .74))


def test_missing_observation_only_carries_recent_seed_and_never_refreshes_real_age():
    estimator = ForwardSeedEstimator(PARAMETERS)
    initial = observation()
    estimator.attach(initial)
    followup = observation(10.4, measured=False)
    estimator.attach(followup)
    seed = followup.metrics["mining_forward_seed"]
    assert seed["source"] == "recent_estimate"
    assert seed["reference_timestamp"] == 9.95
    assert seed["age_seconds"] == pytest.approx(.45)
    assert seed["speed"] == pytest.approx(.30 + .115745857 * .45)
    # Repeated delivery of an old real window must not extend its lifetime.
    limit = observation(11.45, reference=9.95)
    estimator.attach(limit)
    assert limit.metrics["mining_forward_seed"]["valid"]
    expired = observation(11.451, reference=9.95)
    estimator.attach(expired)
    assert expired.metrics["mining_forward_seed"]["reason"] == "last_real_measurement_expired"


def test_new_real_measurement_reanchors_speed_but_not_to_window_end():
    estimator = ForwardSeedEstimator(PARAMETERS)
    estimator.attach(observation())
    obs = observation(10.2, changes={"current_forward_speed": .4})
    estimator.attach(obs)
    assert obs.metrics["mining_forward_seed"]["speed"] == pytest.approx(.4 + .115745857 * .05)
    assert obs.metrics["mining_forward_seed"]["source"] == "observed_window_propagated"


@pytest.mark.parametrize("source,method,confidence", [
    ("dual_landmark_consensus", "paired_positions", .82),
    ("temporal_single_landmark", "temporal_positions", .80),
    ("bilateral_wall_translation", "wall_profile_pair", .80),
])
def test_sparse_measurement_keeps_its_real_source_and_window_on_extrapolation(source, method, confidence):
    estimator = ForwardSeedEstimator(PARAMETERS)
    obs = observation(changes={"current_forward_speed_source": source,
        "current_forward_speed_method": method, "current_forward_speed_confidence": confidence})
    estimator.attach(obs)
    seed = obs.metrics["mining_forward_seed"]
    assert seed["valid"] and seed["measurement_source"] == source
    assert seed["measurement_method"] == method
    assert seed["reference_timestamp"] == 9.95
    following = observation(10.2, measured=False);estimator.attach(following)
    assert following.metrics["mining_forward_seed"]["measurement_source"] == source
    assert following.metrics["mining_forward_seed"]["source"] == "recent_estimate"


@pytest.mark.parametrize("source,method,confidence", [
    ("dual_landmark_consensus", "temporal_positions", .85),
    ("temporal_single_landmark", "paired_positions", .85),
    ("dual_landmark_consensus", "paired_positions", .81),
    ("temporal_single_landmark", "temporal_positions", .79),
    ("bilateral_wall_translation", "paired_positions", .85),
    ("bilateral_wall_translation", "wall_profile_pair", .79),
])
def test_sparse_source_does_not_bypass_method_or_confidence_contract(source, method, confidence):
    obs = observation(changes={"current_forward_speed_source": source,
        "current_forward_speed_method": method, "current_forward_speed_confidence": confidence})
    ForwardSeedEstimator(PARAMETERS).attach(obs)
    assert not obs.metrics["mining_forward_seed"]["valid"]


def test_real_three_landmark_zero_revokes_existing_speed_instead_of_carrying_it():
    estimator = ForwardSeedEstimator(PARAMETERS)
    estimator.attach(observation())
    stopped = observation(10.2, changes={"current_forward_speed": 0.,
        "current_forward_speed_track_count": 3, "current_forward_speed_track_ids": [1, 2, 3]})
    estimator.attach(stopped)
    assert stopped.metrics["mining_forward_seed"]["reason"] == "observed_stop_response_not_fitted"
    no_new_measurement = observation(10.4, measured=False)
    estimator.attach(no_new_measurement)
    assert not no_new_measurement.metrics["mining_forward_seed"]["valid"]


@pytest.mark.parametrize("change", [
    {"current_forward_speed_observed": False}, {"current_forward_speed_confidence": .79},
    {"current_forward_speed_confidence": float("nan")}, {"current_forward_speed_units": "pixels/s"},
    {"current_forward_speed_source": "mode_guess"}, {"current_forward_speed_reference_timestamp": 10.01},
    {"current_forward_speed_window_seconds": 0.}, {"current_forward_speed_method": "unknown"},
    {"current_forward_speed": float("nan")}, {"current_forward_speed": -1.},
])
def test_untrusted_initial_measurement_does_not_create_a_seed(change):
    obs = observation(changes=change)
    ForwardSeedEstimator(PARAMETERS).attach(obs)
    assert not obs.metrics["mining_forward_seed"]["valid"]
    assert obs.metrics["mining_forward_seed"]["reason"] == "no_trustworthy_landmark_seed"


@pytest.mark.parametrize("change", [
    {"collision_suspected": True}, {"hit_body_detected": True}, {"buff_active": True},
    {"motion_mode": "powered"}, {"player_appearance": "powered"}, {"game_ended": True},
])
def test_unsafe_frame_revokes_seed_and_blocks_overlapping_measurement_windows(change):
    estimator = ForwardSeedEstimator(PARAMETERS)
    estimator.attach(observation())
    unsafe = observation(10.1, changes=change)
    estimator.attach(unsafe)
    assert not unsafe.metrics["mining_forward_seed"]["valid"]
    overlap = observation(10.15)
    estimator.attach(overlap)
    assert not overlap.metrics["mining_forward_seed"]["valid"]
    clear = observation(10.3)
    estimator.attach(clear)
    assert clear.metrics["mining_forward_seed"]["valid"]


@pytest.mark.parametrize("phase", ["unknown", "result", "countdown", "menu"])
def test_non_playing_never_carries_seed_across_round_or_visibility_loss(phase):
    estimator = ForwardSeedEstimator(PARAMETERS)
    estimator.attach(observation())
    obs = observation(10.1, phase=phase)
    estimator.attach(obs)
    assert not obs.metrics["mining_forward_seed"]["valid"]
    subsequent = observation(10.3, measured=False)
    estimator.attach(subsequent)
    assert not subsequent.metrics["mining_forward_seed"]["valid"]


def test_powered_entity_revokes_even_without_metric():
    obs = replace(observation(), entities=(Entity("egg_buff", .5, .5, .1, .1, .5),))
    ForwardSeedEstimator(PARAMETERS).attach(obs)
    assert not obs.metrics["mining_forward_seed"]["valid"]


@pytest.mark.parametrize("changes,rows", [
    ({"mining_input_history_source": "advice"}, None),
    ({"mining_input_history_observed_until": 9.8}, None),
    ({"mining_input_history_gap_detected": True}, None),
    ({}, []), ({}, [{"after": 9.9, "axis": 0., "zr": True}]),
    ({}, [{"after": 9., "axis": 0., "zr": True}, {"after": 8., "axis": 0., "zr": False}]),
    ({}, [{"after": 9., "axis": 0., "zr": True}, {"after": 11., "axis": 0., "zr": False}]),
    ({}, [{"after": 9., "state": {"buttons": ["A"]}}]),
    ({}, [{"after": 9., "state": {"buttons": ["ZR"], "left_stick": {"y": 1}}}]),
])
def test_incomplete_or_unmodelled_actual_history_invalidates(changes, rows):
    obs = observation(changes=changes, rows=rows)
    ForwardSeedEstimator(PARAMETERS).attach(obs)
    assert not obs.metrics["mining_forward_seed"]["valid"]


def test_historical_menu_key_outside_window_does_not_poison_playing_interval():
    obs = observation(rows=[{"after": 8., "state": {"buttons": ["A"]}},
                            {"after": 9., "axis": 0., "zr": True}])
    ForwardSeedEstimator(PARAMETERS).attach(obs)
    assert obs.metrics["mining_forward_seed"]["valid"]


def test_input_history_gap_drops_existing_seed_even_with_current_anchor():
    estimator = ForwardSeedEstimator(PARAMETERS)
    estimator.attach(observation())
    gap = observation(10.2, changes={"mining_input_history_gap_detected": True})
    estimator.attach(gap)
    assert gap.metrics["mining_forward_seed"]["reason"] == "input_history_gap"
    next_obs = observation(10.4, measured=False)
    estimator.attach(next_obs)
    assert not next_obs.metrics["mining_forward_seed"]["valid"]


@pytest.mark.parametrize("timestamp", [10., 9.9])
def test_repeated_or_reversed_capture_time_cannot_double_integrate(timestamp):
    estimator = ForwardSeedEstimator(PARAMETERS)
    estimator.attach(observation())
    obs = observation(timestamp)
    estimator.attach(obs)
    assert obs.metrics["mining_forward_seed"]["reason"] == "non_monotonic_capture_time"


def test_disabled_feature_does_not_add_state_or_require_coefficients():
    obs = observation()
    ForwardSeedEstimator({}).attach(obs)
    assert "mining_forward_seed" not in obs.metrics


@pytest.mark.parametrize("change", [{"maximum_seed_age_seconds": 2.}, {"maximum_seed_age_seconds": 0},
                                    {"zr_acceleration": 0}, {"normal_tau_seconds": float("nan")},
                                    {"enabled": "true"}])
def test_invalid_config_fails_before_round_start(change):
    config = {**PARAMETERS["forward_dynamics"], **change}
    with pytest.raises(ValueError):
        ForwardSeedEstimator({**PARAMETERS, "forward_dynamics": config})


def test_missing_effective_delay_is_not_replaced_by_dispatch_or_planning_delay():
    with pytest.raises(ValueError):
        ForwardSeedEstimator({"forward_dynamics": PARAMETERS["forward_dynamics"],
                              "planning_delay_seconds": .05, "dispatch_delay_seconds": .03})


def test_submitted_history_announces_real_event_gap_and_can_resume_after_it():
    class Client:
        events = []
        def request(self, path):
            return {"events": self.events}
    client = Client()
    history = SubmittedInputHistory(client)
    state = {"buttons": [], "left_stick": {"x": 0, "y": 0}}
    client.events = [{"id": 1, "timestamp": 9., "state": state}]
    first = observation(); history.attach(first)
    assert first.metrics["mining_input_history_gap_detected"] is False
    client.events = [{"id": 3, "timestamp": 9.9, "state": state}]
    gap = observation(10.1); history.attach(gap)
    assert gap.metrics["mining_input_history_gap_detected"] is True
    assert len(gap.metrics["mining_input_history"]) == 1
    client.events = []
    next_obs = observation(10.2); history.attach(next_obs)
    assert next_obs.metrics["mining_input_history_gap_detected"] is False
