from dataclasses import replace
from time import perf_counter

from raiders_helper.control.buttons import Button, ControllerState
from raiders_helper.hopping import decide_hopping
from raiders_helper.models import Entity, Game, Observation


def e(kind, x, y, *, vx=0, vy=0, w=0.06, h=0.06, confidence=0.95, samples=3, track_id=0):
    return Entity(kind, x, y, w, h, confidence, track_id, vx, vy, samples)


def scene(*objects, ball=None, metrics=None):
    ball = ball or e("ball", 0.5, 0.3, vy=0.3, track_id=2)
    return Observation(Game.HOPPING, 1, 1, "playing", 0.95,
                       (e("player", 0.5, 0.85, w=0.12, track_id=1), ball, *objects),
                       metrics=metrics or {})


def test_missing_or_ambiguous_ball_never_generates_input():
    obs = scene()
    assert decide_hopping(replace(obs, entities=(obs.entities[0],))).state == ControllerState.neutral()
    assert decide_hopping(scene(e("ball", 0.7, 0.4, vy=0.2))).state == ControllerState.neutral()
    assert decide_hopping(scene(ball=e("ball", 0.5, 0.5, samples=1))).state == ControllerState.neutral()


def test_low_confidence_or_wrong_phase_is_unknown():
    obs = scene()
    assert decide_hopping(replace(obs, confidence=0.2)).diagnostics["mode"] == "unknown"
    assert decide_hopping(replace(obs, phase="result")).state == ControllerState.neutral()
    assert decide_hopping(scene(ball=e("ball", 0.5, 0.4, confidence=0.4))).state == ControllerState.neutral()


def test_local_descent_predicts_catch_and_horizontal_movement():
    obs = scene(ball=e("ball", 0.6, 0.5, vx=0.1, vy=0.4))
    result = decide_hopping(obs)
    assert result.state.left_stick.x > 0
    intercept = result.diagnostics["intercepts"][0]
    assert 0.6 < intercept["x"] < 0.8
    assert intercept["seconds"] > 0
    assert result.diagnostics["experimental"]


def test_imminent_ball_overrides_egg_and_golden_objectives():
    obs = scene(e("egg_pickup", 0.8, 0.85), e("golden_enemy", 0.2, 0.3),
                ball=e("ball", 0.45, 0.75, vy=0.5))
    result = decide_hopping(obs, {"lateral_speed": 1})
    assert result.diagnostics["mode"] == "urgent_catch"
    assert result.diagnostics["objective"] is None
    assert result.state.left_stick.x < 0


def test_ball_below_catch_plane_does_not_chase_golden_target():
    obs = scene(e("golden_enemy", 0.2, 0.3), ball=e("ball", 0.7, 0.83, vy=0.5))
    result = decide_hopping(obs)
    assert result.diagnostics["mode"] == "urgent_catch"
    assert result.diagnostics["objective"] is None


def test_safe_egg_detour_reserves_time_to_return_to_ball():
    obs = scene(e("egg_pickup", 0.65, 0.85, track_id=4),
                ball=e("ball", 0.5, 0.2, vy=0.2))
    result = decide_hopping(obs, {"lateral_speed": 0.7})
    assert result.diagnostics["mode"] == "collect_egg"
    assert result.diagnostics["objective"]["kind"] == "egg_pickup"
    assert result.state.left_stick.x > 0
    assert not result.diagnostics["buff_confirmed"]


def test_ordinary_enemy_blocks_an_otherwise_reachable_egg_detour():
    obs = scene(e("egg_pickup", 0.7, 0.85, track_id=4), e("enemy", 0.62, 0.85),
                ball=e("ball", 0.5, 0.2, vy=0.2))
    result = decide_hopping(obs, {"lateral_speed": 0.7})
    assert result.diagnostics["objective"] is None


def test_falling_egg_that_leaves_robot_plane_before_arrival_is_rejected():
    obs = scene(e("egg_pickup", 0.85, 0.85, vy=1, h=0.02),
                ball=e("ball", 0.5, 0.2, vy=0.2))
    result = decide_hopping(obs, {"lateral_speed": 0.5})
    assert result.diagnostics["objective"] is None


def test_distant_egg_is_not_pursued_without_speed_model():
    result = decide_hopping(scene(e("egg_pickup", 0.85, 0.85)))
    assert result.diagnostics["objective"] is None
    assert result.diagnostics["mode"] == "normal_keep_ball"


def test_golden_target_means_prepare_heading_not_robot_collision():
    result = decide_hopping(scene(e("golden_enemy", 0.8, 0.3, track_id=5)))
    assert result.diagnostics["mode"] == "seek_golden"
    assert result.diagnostics["objective"]["track_id"] == 5
    assert result.diagnostics["target"]["x"] == 0.5
    assert Button.B not in result.state.buttons


def test_powered_contact_needs_confirmed_buff_and_remaining_window():
    enemy = e("enemy", 0.65, 0.85, track_id=8)
    buff = e("egg_buff", 0.05, 0.05, track_id=9)
    obs = scene(enemy, buff, ball=e("ball", 0.5, 0.2, vy=0.2),
                metrics={"buff_remaining_lower_bound_seconds": 2})
    result = decide_hopping(obs, {"lateral_speed": 0.7})
    assert result.diagnostics["mode"] == "powered"
    assert result.diagnostics["objective"]["kind"] == "enemy"
    assert result.state.left_stick.x > 0
    uncertain = decide_hopping(replace(obs, metrics={}), {"lateral_speed": 0.7})
    assert uncertain.diagnostics["mode"] == "powered_keep_ball"
    assert uncertain.diagnostics["objective"] is None
    weak = replace(obs, entities=tuple(replace(item, confidence=0.4) if item.kind == "egg_buff" else item for item in obs.entities))
    weak_result = decide_hopping(weak, {"lateral_speed": 0.7})
    assert not weak_result.diagnostics["buff_confirmed"]
    assert weak_result.diagnostics["objective"] is None


def test_normal_enemy_is_avoided_within_ball_catch_interval():
    result = decide_hopping(scene(e("enemy", 0.5, 0.85, w=0.02)), {"catch_margin": 0})
    assert result.diagnostics["target"]["x"] != 0.5
    assert any("普通鲑鱼" in risk for risk in result.diagnostics["risks"])


def test_jump_requires_opt_in_and_predicted_heading_window():
    obs = scene(ball=e("ball", 0.5, 0.75, vy=0.5))
    assert Button.B not in decide_hopping(obs, {"jump_window_seconds": 0.2}).state.buttons
    result = decide_hopping(obs, {"allow_jump": True, "jump_window_seconds": 0.2})
    assert Button.B in result.state.buttons
    far = scene(ball=e("ball", 0.8, 0.75, vy=0.5))
    assert Button.B not in decide_hopping(far, {"allow_jump": True, "jump_window_seconds": 0.2}).state.buttons


def test_latency_changes_reachability_and_jump_deadline():
    obs = scene(ball=e("ball", 0.65, 0.7, vy=0.3))
    quick = decide_hopping(obs, {"lateral_speed": 0.5})
    late = decide_hopping(obs, {"lateral_speed": 0.5, "latency_seconds": 0.4})
    assert quick.diagnostics["intercepts"][0]["reachable"]
    assert not late.diagnostics["intercepts"][0]["reachable"]


def test_side_reflection_keeps_prediction_in_play_bounds():
    result = decide_hopping(scene(ball=e("ball", 0.8, 0.3, vx=0.8, vy=0.3)),
                            {"play_bounds": [0.1, 0.9]})
    assert 0.16 <= result.diagnostics["target"]["x"] <= 0.84


def test_measured_gravity_can_predict_ascending_ball_return():
    result = decide_hopping(scene(ball=e("ball", 0.5, 0.3, vy=-0.3)), {"gravity": 1})
    assert result.diagnostics["intercepts"][0]["seconds"] > 0
    assert result.diagnostics["intercepts"][0]["model"] == "ballistic"


def test_fast_policy_has_no_search_or_external_io():
    # A generous guard catches accidental heavy search/I/O; this is not a claim
    # about capture-card end-to-end latency or a strict per-frame real-time SLA.
    obs = scene(*(e("enemy", i/20, 0.85, vx=0.1) for i in range(20)))
    started = perf_counter()
    for _ in range(200):
        decide_hopping(obs, {"lateral_speed": 0.8})
    assert perf_counter()-started < 1
