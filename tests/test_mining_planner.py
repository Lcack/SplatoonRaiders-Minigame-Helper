from dataclasses import replace

import pytest

from raiders_helper.control.buttons import ControllerState
from raiders_helper.mining_planner import MotionModel, Road, SearchSettings, contact_fraction, plan_mining, planning_decision
from raiders_helper.models import Entity, Game, Observation
from raiders_helper.vision import fill_recognized_obstacle_gaps


def entity(kind, x, y, w=0.04, h=0.04, track_id=-1):
    return Entity(kind, x, y, w, h, 0.95, track_id)


def world(*objects, road=None, metrics=None):
    road = road if road is not None else [entity("corridor", 0.5, y, 0.95, 0.001) for y in (0, 0.25, 0.5, 0.75, 1)]
    return Observation(Game.MINING, 1, 1, "playing", 0.95,
                       (entity("player", 0.5, 0.15), *road, *objects), metrics=metrics or {})


def motion(**changes):
    # ALL values here are synthetic assumptions; no game measurements implied.
    data = dict(forward_speed=0.2, accelerated_speed=0.3, lateral_speed=0.6,
                latency_seconds=0, uncertainty_x=0.005, uncertainty_y=0.005,
                boost_speed=0.4, calibrated=True)
    data.update(changes)
    return MotionModel(**data)


SETTINGS = SearchSettings(horizon_seconds=1, step_seconds=0.1, beam_width=64)


def test_curved_road_without_common_x_has_a_route():
    road = [entity("corridor", x, y, 0.24, 0.001) for y, x in
            [(0, 0.5), (0.15, 0.5), (0.3, 0.7), (0.45, 0.5), (0.65, 0.3), (1, 0.3)]]
    assert max(e.x-e.w/2 for e in road) > min(e.x+e.w/2 for e in road)
    plan = plan_mining(world(road=road), motion(lateral_speed=1), SETTINGS)
    assert plan.points, plan.reason
    assert max(p.x for p in plan.points) > 0.57


def test_road_rejects_diagonal_cut_and_disjoint_branch_jump():
    road = Road([entity("corridor", 0.5, 0, 1), entity("corridor", 0.8, 0.5, 0.2), entity("corridor", 0.5, 1, 1)])
    assert not road.segment_safe(0.5, 0, 0.5, 1, 0.02)
    split = Road([entity("corridor", x, y, 0.35) for y in (0, 1) for x in (0.2, 0.8)])
    assert not split.segment_safe(0.2, 0.2, 0.8, 0.8, 0.02)


def test_full_body_does_not_skip_thin_road_notch_between_offsets():
    road = Road([entity("corridor", 0.5, 0, 1), entity("corridor", 0.5, 0.22, 1),
                 entity("corridor", 0.8, 0.2225, 0.08), entity("corridor", 0.5, 0.225, 1),
                 entity("corridor", 0.5, 1, 1)])
    assert not road.swept_safe(0.5, 0.2, 0.5, 0.2, 0.01, 0.025)
    assert not road.swept_safe(0.5, 0.2, 0.5, 0.22, 0.01, 0.025)


def test_thin_obstacle_between_endpoints_is_not_skipped():
    rock = entity("obstacle", 0.5, 0.3, 0.02, 0.005)
    assert contact_fraction(0.5, 0.2, 0.5, 0.4, rock, 0.02, 0.02) is not None


def test_group_can_outweigh_nearest_single_pickup():
    obs = world(entity("reward", 0.35, 0.24, track_id=1),
                *(entity("reward", 0.7, y, track_id=i+2) for i, y in enumerate((0.28, 0.34, 0.4))))
    plan = plan_mining(obs, motion(), SETTINGS)
    assert plan.gained_seconds >= 2, plan
    assert plan.points[1].steering > 0


def test_pickup_cannot_be_counted_twice_across_steps_or_duplicate_id():
    star = entity("reward", 0.5, 0.25, h=0.12, track_id=7)
    plan = plan_mining(world(star, star), motion(), SETTINGS)
    assert plan.gained_seconds == 1
    assert len(plan.collected) == 1


def test_known_buff_breaks_rocks_and_walls_stay_forbidden():
    buff = entity("egg_buff", 0.1, 0.1)
    rock = entity("obstacle", 0.5, 0.32, w=0.95, h=0.01)
    obs = world(buff, rock, metrics={"buff_remaining_lower_bound_seconds": 2})
    assert plan_mining(obs, motion(), SETTINGS).gained_seconds == 1
    blocked = replace(obs, entities=tuple(replace(e, kind="wall") if e is rock else e for e in obs.entities))
    assert not plan_mining(blocked, motion(), SETTINGS).points
    assert not plan_mining(world(rock), motion(), SETTINGS).points


def test_unknown_buff_duration_does_not_grant_future_immunity():
    obs = world(entity("egg_buff", 0.1, 0.1), entity("obstacle", 0.5, 0.32, w=0.95))
    assert not plan_mining(obs, motion(), SETTINGS).points


def test_egg_needs_separate_boost_model_and_only_uses_measured_interval():
    obs = world(entity("egg_pickup", 0.5, 0.25, track_id=9))
    assert not plan_mining(obs, motion(boost_speed=None), SETTINGS).points
    assert not plan_mining(obs, motion(), SETTINGS).points
    plan = plan_mining(obs, motion(egg_buff_lower_bound_seconds=2), SETTINGS)
    assert plan.gained_seconds == 2


def test_egg_acceleration_inside_one_step_rechecks_swept_path():
    obs = world(entity("egg_pickup", 0.5, 0.17, h=0.002),
                entity("wall", 0.5, 0.20, w=1, h=0.005))
    # Eating the egg partway through this step makes even neutral-forward
    # motion reach the wall before the step ends. The old-speed endpoint did not.
    model = motion(forward_speed=0.12, accelerated_speed=0.2, lateral_speed=0.001,
                   uncertainty_x=0, uncertainty_y=0, boost_speed=1,
                   egg_buff_lower_bound_seconds=1)
    plan = plan_mining(obs, model, SearchSettings(horizon_seconds=0.1, step_seconds=0.1))
    assert not plan.points


def test_large_position_uncertainty_cannot_manufacture_pickup_reward():
    obs = world(entity("reward", 0.5, 0.25, w=0.01, h=0.01))
    plan = plan_mining(obs, motion(uncertainty_x=0.05), SETTINGS)
    assert plan.points
    assert plan.gained_seconds == 0


def test_timer_expiry_before_pickup_is_not_rescued_retroactively():
    obs = world(entity("reward", 0.5, 0.5), metrics={"time_remaining": 0.1})
    assert not plan_mining(obs, motion(), SETTINGS).points


def test_latency_envelope_can_reject_route_that_is_safe_without_latency():
    road = [entity("corridor", 0.5, y, 0.2) for y in (0, 1)]
    obs = world(road=road)
    assert plan_mining(obs, motion(), SETTINGS).points
    assert not plan_mining(obs, motion(latency_seconds=0.2), SETTINGS).points


def test_uncalibrated_profile_never_outputs_new_planner_input():
    model = motion(calibrated=False)
    assert planning_decision(world(), {"motion": model.__dict__}).state == ControllerState.neutral()
    assert planning_decision(world(), {}).state == ControllerState.neutral()


def test_exhausted_node_budget_returns_no_unfinished_action():
    plan = plan_mining(world(), motion(), replace(SETTINGS, max_expanded_nodes=1))
    assert not plan.points
    assert plan.expanded_nodes == 1


def test_road_gap_can_be_filled_by_rock_but_never_by_wall_or_unknown():
    intervals = [entity("corridor", 0.25, 0.3, 0.3), entity("corridor", 0.75, 0.3, 0.3)]
    assert len(fill_recognized_obstacle_gaps(intervals)) == 2
    rock = entity("obstacle", 0.5, 0.3, 0.22, 0.1)
    restored = fill_recognized_obstacle_gaps([*intervals, rock])
    roads = [e for e in restored if e.kind == "corridor"]
    assert len(roads) == 1
    assert roads[0].w == pytest.approx(0.8)
    with_wall = fill_recognized_obstacle_gaps([*intervals, rock, entity("wall", 0.5, 0.3)])
    assert len([e for e in with_wall if e.kind == "corridor"]) == 2


@pytest.mark.parametrize("changes", [{"forward_speed": 0}, {"latency_seconds": -1}, {"uncertainty_x": float("nan")}, {"boost_speed": -1}])
def test_invalid_dynamics_are_rejected(changes):
    with pytest.raises(ValueError):
        motion(**changes)
