"""Geometry and motion here are declared synthetic hypotheses, not NS data."""
from dataclasses import replace

import pytest

from raiders_helper.control.buttons import Button,ControllerState
from raiders_helper.mining_planner import Road,contact_fraction
from raiders_helper.mining_reactive import decide_mining_reactive,_sweep_constraints,_compiled_safe
from raiders_helper.models import Entity,Game,Observation


def entity(kind,x,y,w=.04,h=.04,**changes):
    return Entity(kind,x,y,w,h,.95,track_id=1,samples=5,**changes)


def world(*objects,road=None,x=.5,vx=0,metrics=None):
    road=road if road is not None else [entity("corridor",.5,y,.94,.001) for y in (0,.15,.2,.35,.5,.7,1)]
    return Observation(Game.MINING,1,1,"playing",.95,
        (entity("player",x,.2,.04,.04,vx=vx),*road,*objects),metrics=metrics or {})


def test_centered_clear_straight_keeps_zr():
    decision=decide_mining_reactive(world())
    assert decision.state.buttons==frozenset({Button.ZR})
    assert decision.state.left_stick.x==0
    assert decision.diagnostics["motion_calibrated"] is False


def clearance_motion_parameters():
    return {"maximum_axis":.65,"lookahead_depth":.68,"maximum_steps":30,
        "motion_models":{"normal":{"forward_speed":.25,"input_gain_per_second":2.65,"drag_per_second":8.38,"maximum_lateral_speed":.4},
                         "zr":{"forward_speed":.5,"input_gain_per_second":3.177,"drag_per_second":22.14,"maximum_lateral_speed":.25}}}


def test_full_axis_on_complete_clear_route_can_keep_boost_only_when_enabled():
    parameters=clearance_motion_parameters();obs=world(x=.2)
    assert Button.ZR not in decide_mining_reactive(obs,parameters).state.buttons
    decision=decide_mining_reactive(obs,{**parameters,"allow_full_axis_with_clearance":True})
    assert Button.ZR in decision.state.buttons
    assert decision.diagnostics["route_status"]=="complete"
    authority=decision.diagnostics["turn_authority"]
    assert authority["axis_gate_waived"] is True
    assert authority["early_envelope_clearance_widths"]>.02


@pytest.mark.parametrize("change",[{"full_axis_minimum_clearance":.15},{"sharp_turn_authority_fraction":.9}])
def test_full_axis_opt_in_does_not_bypass_clearance_or_authority_limits(change):
    decision=decide_mining_reactive(world(x=.2),{**clearance_motion_parameters(),
        "allow_full_axis_with_clearance":True,**change})
    assert Button.ZR not in decision.state.buttons


def test_full_axis_opt_in_does_not_bypass_visible_obstacle():
    rock=entity("rock",.5,.27,1,.02)
    decision=decide_mining_reactive(world(rock),{**clearance_motion_parameters(),
        "allow_full_axis_with_clearance":True,"compare_star_modes":True,"star_reward":1e6})
    assert Button.ZR not in decision.state.buttons


def test_no_visible_stars_does_not_add_normal_mode_comparison():
    decision=decide_mining_reactive(world(),{"compare_star_modes":True})
    assert decision.diagnostics["star_mode_comparison"]=={"enabled":True,"attempted":False}


def test_partial_normal_path_cannot_win_same_distance_star_comparison():
    decision=decide_mining_reactive(world(entity("time_star",.5,.65,.02,.02)),{
        "compare_star_modes":True,"star_reward":.04,"lookahead_depth":.68,"maximum_steps":14,
        "motion_models":{"normal":{"forward_speed":.1},"zr":{"forward_speed":.5}}})
    comparison=decision.diagnostics["star_mode_comparison"]
    assert comparison["attempted"] is True
    assert comparison["normal_status"]=="partial"
    assert comparison["same_complete_depth"] is False
    assert comparison["selected"]=="fast"
    assert Button.ZR in decision.state.buttons


@pytest.mark.parametrize("changes",[{"compare_star_modes":"true"},{"allow_full_axis_with_clearance":1},
    {"full_axis_minimum_clearance":0},{"star_mode_gain_margin_seconds":float("nan")}])
def test_mode_comparison_and_clearance_parameters_are_validated(changes):
    assert decide_mining_reactive(world(),changes).state==ControllerState.neutral()


def test_recorded_star_group_compares_extra_seconds_at_same_complete_depth():
    import json
    from pathlib import Path
    directory=Path(__file__).parents[1]/"artifacts/mining-A-campaign/batch-02/round-0001"
    if not (directory/"gameplay/trace.jsonl").exists():pytest.skip("local real-NS observation is not bundled")
    rows=[json.loads(s) for s in (directory/"gameplay/trace.jsonl").read_text(encoding="utf-8").splitlines()]
    data=next(r["observation"] for r in rows if r["observation"]["sequence"]==4313)
    obs=Observation(Game.MINING,data["timestamp"],data["sequence"],data["phase"],data["confidence"],
                    tuple(Entity(**e) for e in data["entities"]),metrics=data["metrics"])
    parameters=json.loads((directory/"parameters.json").read_text())
    decision=decide_mining_reactive(obs,{**parameters,"compare_star_modes":True})
    comparison=decision.diagnostics["star_mode_comparison"]
    assert comparison["same_complete_depth"] is True and comparison["selected"]=="normal"
    assert comparison["extra_stars_predicted"]==3
    assert comparison["net_time_gain_seconds"]==pytest.approx(comparison["extra_stars_predicted"]-comparison["extra_travel_seconds"])
    assert comparison["net_time_gain_seconds"]>.25 and Button.ZR not in decision.state.buttons
    guarded=decide_mining_reactive(obs,{**parameters,"compare_star_modes":True,"star_mode_gain_margin_seconds":100})
    assert guarded.diagnostics["star_mode_comparison"]["selected"]=="fast"
    assert Button.ZR in guarded.state.buttons


def test_curved_layers_without_common_x_get_advance_steering():
    road=[entity("corridor",x,y,.30,.001) for y,x in
        [(0,.4),(.15,.4),(.2,.4),(.3,.50),(.4,.64),(.55,.72),(.8,.72),(1,.72)]]
    assert max(e.x-e.w/2 for e in road)>min(e.x+e.w/2 for e in road)
    decision=decide_mining_reactive(world(road=road,x=.4),{
        "motion_models":{"normal":{"forward_speed":.24,"input_gain_per_second":8,"drag_per_second":4,
                                     "maximum_lateral_speed":1.2},
                         "zr":{"forward_speed":.4,"input_gain_per_second":8,"drag_per_second":4,
                                 "maximum_lateral_speed":1.2}},"maximum_steps":20})
    assert decision.diagnostics["path"],decision.reason
    assert decision.state.left_stick.x>0
    assert max(p["x"] for p in decision.diagnostics["path"])>.56
    road_model=Road(road)
    for a,b in zip(decision.diagnostics["path"],decision.diagnostics["path"][1:]):
        assert road_model.swept_safe(a["x"],a["depth"],b["x"],b["depth"],.036,.024)


def test_near_wall_releases_zr_and_moves_inward():
    decision=decide_mining_reactive(world(x=.075))
    assert Button.ZR not in decision.state.buttons
    assert decision.state.left_stick.x>0


def test_current_envelope_touching_wall_has_explicit_recovery_not_fake_safe_path():
    decision=decide_mining_reactive(world(x=.055))
    assert decision.diagnostics["mode"]=="wall_recovery"
    assert decision.state.left_stick.x>0
    assert not decision.state.buttons
    assert decision.diagnostics["risks"]


def test_recovery_does_not_steer_outward_because_inward_velocity_is_large():
    decision=decide_mining_reactive(world(x=.055,vx=.8))
    assert decision.state.left_stick.x>0


def test_recovery_also_respects_explicit_maximum_axis():
    decision=decide_mining_reactive(world(x=.055),{"maximum_axis":.25})
    assert 0 < decision.state.left_stick.x <= round(.25*32767)


@pytest.mark.parametrize("rock_x",[.105,.09])
def test_wall_recovery_cannot_create_or_increase_rock_contact(rock_x):
    # The robot is touching the left wall, but inward is not automatically safe.
    obs=world(entity("rock",rock_x,.2,.02,.10),x=.055)
    decision=decide_mining_reactive(obs,{"allow_contact_mitigation":True})
    assert decision.state==ControllerState.neutral()
    assert decision.diagnostics["mode"]=="recovery_blocked"
    assert decision.diagnostics["recovery_obstacle_check"]["blocked_by"] in {
        "new_obstacle_contact","increases_existing_rock_overlap"}
    assert "不保证脱困" in decision.reason
    assert decision.diagnostics["path"]==[]


def test_wall_recovery_keeps_inward_action_when_visible_rock_is_clear():
    decision=decide_mining_reactive(world(entity("rock",.4,.5,.05,.05),x=.055))
    assert decision.diagnostics["mode"]=="wall_recovery"
    assert decision.state.left_stick.x>0 and not decision.state.buttons
    assert decision.diagnostics["recovery_obstacle_check"]["checked_hazards"]==1


def test_horizontal_velocity_is_braked_before_overshooting_straight_center():
    decision=decide_mining_reactive(world(vx=.6))
    assert decision.state.left_stick.x<0
    assert Button.ZR not in decision.state.buttons


def test_near_rock_gets_a_collision_free_detour_and_no_zr():
    rock=entity("obstacle",.5,.39,.12,.025)
    decision=decide_mining_reactive(world(rock))
    assert decision.diagnostics["path"],decision.reason
    assert decision.state.left_stick.x!=0
    assert Button.ZR not in decision.state.buttons
    for a,b in zip(decision.diagnostics["path"],decision.diagnostics["path"][1:]):
        assert contact_fraction(a["x"],a["depth"],b["x"],b["depth"],rock,.036,.024) is None


def test_thin_immediate_obstacle_is_not_skipped_between_steps():
    decision=decide_mining_reactive(world(entity("obstacle",.5,.245,.6,.001)))
    assert not decision.diagnostics["path"]
    assert decision.state==ControllerState.neutral()


def test_full_width_wall_has_no_route_even_with_buff():
    obs=world(entity("egg_buff",.5,.2),entity("wall",.5,.29,1,.01),metrics={"buff_remaining_lower_bound_seconds":20})
    decision=decide_mining_reactive(obs,{"allow_powered_rocks":True})
    assert decision.state==ControllerState.neutral()
    assert decision.diagnostics["route_status"]=="partial"
    assert max(p["depth"] for p in decision.diagnostics["path"])<.29-.005-.024


def test_buff_appearance_without_lower_bound_does_not_enable_rock_immunity():
    obs=world(entity("egg_buff",.5,.2),entity("obstacle",.5,.29,1,.01))
    decision=decide_mining_reactive(obs,{"allow_powered_rocks":True})
    assert not decision.diagnostics["powered_rocks_allowed"]
    assert decision.state==ControllerState.neutral()
    valid=replace(obs,metrics={"buff_remaining_lower_bound_seconds":20})
    allowed=decide_mining_reactive(valid,{"allow_powered_rocks":True})
    assert allowed.diagnostics["powered_rocks_allowed"]
    assert allowed.diagnostics["path"]


def test_road_missing_player_body_top_cannot_be_assumed_valid():
    road=[entity("corridor",.5,y,.9,.001) for y in (.2,.4,.8)]
    decision=decide_mining_reactive(world(road=road))
    assert decision.state==ControllerState.neutral()
    assert "完整尺寸" in decision.reason


@pytest.mark.parametrize("change",["phase","missing_player","single_sample","stale"])
def test_unknown_or_untracked_player_never_moves(change):
    obs=world()
    if change=="phase":obs=replace(obs,phase="unknown")
    if change=="missing_player":obs=replace(obs,entities=obs.entities[1:])
    if change=="single_sample":obs=replace(obs,entities=(replace(obs.entities[0],samples=1),*obs.entities[1:]))
    if change=="stale":obs=replace(obs,metrics={"observation_age_seconds":.2})
    assert decide_mining_reactive(obs).state==ControllerState.neutral()


def test_velocity_response_can_differ_between_normal_and_zr_models():
    decision=decide_mining_reactive(world(),{"motion_models":{"zr":{
        "forward_speed":.4,"input_gain_per_second":7.2,"drag_per_second":6.0,"maximum_lateral_speed":.8}}})
    assert decision.diagnostics["model"]["input_gain_per_second"]==7.2
    assert decision.diagnostics["model"]["drag_per_second"]==6


def test_observed_forward_speed_requires_units_confidence_and_explicit_flag():
    invalid=world(metrics={"forward_speed":3})
    assert decide_mining_reactive(invalid).diagnostics["model"]["forward_speed"]==.5
    valid=world(metrics={"forward_speed":.35,"forward_speed_confidence":.9,"forward_speed_observed":True,
                         "forward_speed_units":"viewport_heights_per_second","forward_speed_source":"landmark_tracking"})
    assert decide_mining_reactive(valid).diagnostics["model"]["forward_speed"]==.35


def test_step_budget_cannot_claim_full_horizon_or_keep_zr():
    decision=decide_mining_reactive(world(),{"maximum_steps":2})
    assert decision.diagnostics["route_status"]=="partial"
    assert not decision.state.buttons


@pytest.mark.parametrize("config",[{"beam_width":0},{"step_seconds":.001},{"maximum_axis":float('nan')},{"motion_models":{"zr":{"input_gain_per_second":float('nan')}}}])
def test_invalid_model_config_fails_closed(config):
    assert decide_mining_reactive(world(),config).state==ControllerState.neutral()


def test_precompiled_body_envelope_matches_continuous_road_test():
    import random
    rng=random.Random(720)
    for _ in range(150):
        road=Road([entity("corridor",rng.uniform(.35,.65),y,rng.uniform(.25,.65),.001)
                   for y in (0,.16,.21,.23,.27,.31,.4,.6)])
        a,b=rng.uniform(.1,.9),rng.uniform(.1,.9)
        constraints=_sweep_constraints(road,.18,.32,.03,.035)
        assert constraints is not None
        assert _compiled_safe(constraints,a,b)==road.swept_safe(a,.18,b,.32,.03,.035)


def hit_world(x=.055,road=None,metrics=None,*objects):
    obs=world(*objects,x=x,road=road)
    data={"collision_suspected":True,"hit_body_detected":True,"hit_body_candidates":1,
          "hit_body_identity_age":.1,"hud_anchor_score":.9}
    data.update(metrics or {})
    return replace(obs,phase="unknown",confidence=0,
                   entities=(replace(obs.entities[0],kind="player_hit",samples=1),*obs.entities[1:]),
                   metrics=data)


@pytest.mark.parametrize("x,sign",[(.055,1),(.945,-1)])
def test_visible_hit_recovery_is_inward_limited_and_never_accelerates(x,sign):
    obs=hit_world(x=x)
    decision=decide_mining_reactive(obs,{"allow_hit_recovery":True})
    assert 0 < sign*decision.state.left_stick.x <= round(.45*32767)
    assert not decision.state.buttons
    assert decision.diagnostics["mode"]=="collision_recovery"
    assert decision.diagnostics["recovery_source"]=="observed_player_hit"
    assert decision.diagnostics["path"]==[]
    assert decision.diagnostics["requires_current_hit"] is True
    assert decision.diagnostics["motion_calibrated"] is False
    assert decision.diagnostics["risks"]
    assert obs.phase=="unknown" and not obs.of_kind("player")


@pytest.mark.parametrize("parameters",[{}, {"allow_hit_recovery":False}, {"allow_hit_recovery":1}])
def test_hit_recovery_requires_explicit_boolean_opt_in(parameters):
    assert decide_mining_reactive(hit_world(),parameters).state==ControllerState.neutral()


@pytest.mark.parametrize("parameters,expected",[
    ({"maximum_axis":.2},.2),({"hit_recovery_axis":.3},.3),({"hit_recovery_axis":1},.6)])
def test_hit_recovery_respects_user_axis_cap_and_its_own_hard_cap(parameters,expected):
    decision=decide_mining_reactive(hit_world(),{"allow_hit_recovery":True,**parameters})
    assert decision.state.left_stick.x==round(expected*32767)


@pytest.mark.parametrize("metrics",[
    {"collision_suspected":False},{"hit_body_detected":False},
    {"hit_body_candidates":0},{"hit_body_candidates":2},
    {"ambiguous_player":True},{"ambiguous_hit_body":True},
    {"hit_body_identity_age":0},{"hit_body_identity_age":.751},
    {"hit_body_identity_age":float("nan")},{"hud_anchor_score":.64},
    {"game_ended":True},{"time_remaining":0},{"time_remaining":float("nan")},
    {"observation_age_seconds":.151},{"observation_age_seconds":-.01},
    {"observation_age_seconds":float("nan")},
])
def test_unreliable_stale_or_ended_hit_never_moves(metrics):
    obs=hit_world(metrics=metrics)
    assert decide_mining_reactive(obs,{"allow_hit_recovery":True}).state==ControllerState.neutral()


@pytest.mark.parametrize("phase",["menu","countdown","result","result_pending","paused"])
def test_hit_appearance_in_a_known_nonplay_phase_never_moves(phase):
    obs=replace(hit_world(),phase=phase)
    assert decide_mining_reactive(obs,{"allow_hit_recovery":True}).state==ControllerState.neutral()


@pytest.mark.parametrize("change",["missing","duplicate","unconfident","invalid","blue_also_visible"])
def test_hit_recovery_needs_one_real_valid_hit_entity(change):
    obs=hit_world()
    hit=obs.entities[0]
    if change=="missing":obs=replace(obs,entities=obs.entities[1:])
    if change=="duplicate":obs=replace(obs,entities=(*obs.entities,replace(hit,track_id=2)))
    if change=="unconfident":obs=replace(obs,entities=(replace(hit,confidence=.79),*obs.entities[1:]))
    if change=="invalid":obs=replace(obs,entities=(replace(hit,x=float("nan")),*obs.entities[1:]))
    if change=="blue_also_visible":obs=replace(obs,entities=(*obs.entities,replace(hit,kind="player")))
    assert decide_mining_reactive(obs,{"allow_hit_recovery":True}).state==ControllerState.neutral()


def test_hit_disappearing_next_frame_immediately_stops_recovery():
    obs=hit_world()
    assert decide_mining_reactive(obs,{"allow_hit_recovery":True}).state.left_stick.x>0
    missing=replace(obs,sequence=obs.sequence+1,entities=obs.entities[1:])
    assert decide_mining_reactive(missing,{"allow_hit_recovery":True}).state==ControllerState.neutral()


def test_interior_hit_has_no_assumed_side_wall_direction():
    assert decide_mining_reactive(hit_world(x=.5),{"allow_hit_recovery":True}).state==ControllerState.neutral()


@pytest.mark.parametrize("road",[
    [],
    [entity("corridor",.5,.2,.9,.001)],
    [entity("corridor",.5,y,.9,.001) for y in (.2,.4,.8)],
    [entity("corridor",.5,y,.04,.001) for y in (0,.2,.4,.8)],
    [entity("corridor",x,y,.25,.001) for y in (0,.2,.4,.8) for x in (.2,.8)],
])
def test_hit_recovery_refuses_missing_incomplete_too_narrow_or_branching_road(road):
    obs=hit_world(road=road)
    assert decide_mining_reactive(obs,{"allow_hit_recovery":True}).state==ControllerState.neutral()


def test_hit_recovery_does_not_steer_through_visible_obstacle():
    obs=hit_world(.055,None,None,entity("obstacle",.11,.2,.04,.04))
    assert decide_mining_reactive(obs,{"allow_hit_recovery":True}).state==ControllerState.neutral()


def test_real_left_wall_hit_interface_requests_rightward_release_of_zr():
    """The two images seed appearance identity, not a measured time series."""
    import cv2
    import numpy as np
    from pathlib import Path
    from raiders_helper.capture.models import FramePacket
    from raiders_helper.mining_vision import MiningLiveRecognizer
    directory=Path(__file__).resolve().parents[1]/"artifacts/mining-supervision/round-03"
    paths=[directory/f"frame-{seq:08d}.jpg" for seq in (13698,13822)]
    if not all(path.exists() for path in paths):pytest.skip("local NS collision evidence not bundled")
    frames=[cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1) for path in paths]
    recognizer=MiningLiveRecognizer(profile={"stable_frames":1,"phase_anchors":False})
    recognizer.observe(FramePacket(frames[0],1.0,1,60))
    obs=recognizer.observe(FramePacket(frames[1],1.5,2,60))
    assert obs.phase=="unknown" and obs.of_kind("player_hit") and not obs.of_kind("player")
    decision=decide_mining_reactive(obs,{"allow_hit_recovery":True})
    assert decision.state.left_stick.x>0 and not decision.state.buttons
    assert decision.diagnostics["mode"]=="collision_recovery"


@pytest.mark.parametrize("parameters",[
    {"maximum_axis":float("nan")},{"hit_recovery_axis":0},
    {"wall_margin":-.1},{"position_uncertainty":"bad"},
    {"maximum_observation_age_seconds":float("nan")},
])
def test_hit_recovery_invalid_limits_fail_closed(parameters):
    assert decide_mining_reactive(hit_world(),{"allow_hit_recovery":True,**parameters}).state==ControllerState.neutral()


def contact_world(*extra,metrics=None):
    # Synthetic current rectangle overlap, not a measured game hitbox.
    return world(entity("rock",.55,.22,.1,.1),*extra,metrics=metrics)


def test_current_rock_overlap_can_only_mitigate_after_explicit_opt_in():
    obs=contact_world()
    for parameters in ({},{"allow_contact_mitigation":False},{"allow_contact_mitigation":1}):
        assert decide_mining_reactive(obs,parameters).state==ControllerState.neutral()
    decision=decide_mining_reactive(obs,{"allow_contact_mitigation":True})
    assert decision.state.left_stick.x<0 and not decision.state.buttons
    assert decision.diagnostics["mode"]=="contact_mitigation"
    assert decision.diagnostics["route_status"]=="contact_not_collision_free"
    assert decision.diagnostics["relative_to_neutral"]["rock_overlap_reduction"]>0
    assert decision.diagnostics["existing_rock_contacts"]==1
    assert decision.diagnostics["path"]==[] and decision.diagnostics["risks"]


def test_latency_projected_side_wall_contact_mitigates_inward_without_zr():
    obs=world(x=.92,vx=.3)
    parameters={"latency_seconds":.1}
    assert decide_mining_reactive(obs,parameters).state==ControllerState.neutral()
    decision=decide_mining_reactive(obs,{**parameters,"allow_contact_mitigation":True})
    assert decision.state.left_stick.x<0 and not decision.state.buttons
    assert decision.diagnostics["mode"]=="contact_mitigation"
    assert decision.diagnostics["latency_wall_contact"] is True
    assert decision.diagnostics["relative_to_neutral"]["wall_overlap_reduction"]>0


def test_mitigation_clamps_controls_before_model_evaluation():
    decision=decide_mining_reactive(contact_world(),{"allow_contact_mitigation":True,"maximum_axis":.25})
    assert decision.state.left_stick.x==-round(.25*32767)
    assert decision.diagnostics["predicted_end"]["x"]>.49


def test_mitigation_chooses_smaller_input_when_full_input_would_hit_another_rock():
    second=entity("rock",.43,.2,.04,.1)
    decision=decide_mining_reactive(contact_world(second),{"allow_contact_mitigation":True})
    assert decision.diagnostics["mode"]=="contact_mitigation"
    assert decision.state.left_stick.x==-round(.5*32767)
    assert decision.diagnostics["predicted_end"]["x"]-.036>=second.x+second.w/2


def test_mitigation_rejects_all_directions_if_reducing_one_rock_would_touch_another():
    second=entity("rock",.443,.2,.04,.1)
    decision=decide_mining_reactive(contact_world(second),{"allow_contact_mitigation":True})
    assert decision.state==ControllerState.neutral()
    assert decision.diagnostics.get("mode")!="contact_mitigation"


def test_mitigation_cannot_trade_increased_existing_contact_for_aggregate_gain():
    second=entity("rock",.45,.22,.1,.1)
    assert decide_mining_reactive(contact_world(second),{"allow_contact_mitigation":True}).state==ControllerState.neutral()


@pytest.mark.parametrize("kind",["wall","obstacle"])
def test_full_width_contact_with_no_side_exit_never_becomes_mitigation_success(kind):
    obs=world(entity(kind,.5,.2,1,.1))
    assert decide_mining_reactive(obs,{"allow_contact_mitigation":True}).state==ControllerState.neutral()


def test_future_full_width_wall_without_current_contact_still_stays_neutral():
    obs=world(entity("wall",.5,.25,1,.02))
    decision=decide_mining_reactive(obs,{"allow_contact_mitigation":True})
    assert decision.state==ControllerState.neutral()
    assert decision.diagnostics.get("mode")!="contact_mitigation"


def test_future_road_corner_without_current_or_latency_contact_does_not_open_mitigation():
    road=[entity("corridor",x,y,.30,.001) for y,x in
          [(0,.5),(.15,.5),(.2,.5),(.225,.5),(.25,.2),(.4,.2),(.8,.2)]]
    decision=decide_mining_reactive(world(road=road),{"allow_contact_mitigation":True})
    assert decision.state==ControllerState.neutral()
    assert decision.diagnostics.get("mode")!="contact_mitigation"


def test_mitigation_refuses_ambiguous_branching_road():
    road=[entity("corridor",x,y,w,.001) for y in (0,.15,.2,.35,.6,.9)
          for x,w in ((.5,.4),(.9,.1))]
    decision=decide_mining_reactive(contact_world(),{"allow_contact_mitigation":True})
    assert decision.diagnostics["mode"]=="contact_mitigation"
    ambiguous=world(entity("rock",.55,.22,.1,.1),road=road)
    assert decide_mining_reactive(ambiguous,{"allow_contact_mitigation":True}).state==ControllerState.neutral()


@pytest.mark.parametrize("change",["stale","ended","time_zero","unknown","result","single_sample","duplicate","ambiguous","incomplete"])
def test_contact_mitigation_preserves_freshness_identity_and_stop_gates(change):
    obs=contact_world()
    if change=="stale":obs=replace(obs,metrics={"observation_age_seconds":.2})
    if change=="ended":obs=replace(obs,metrics={"game_ended":True})
    if change=="time_zero":obs=replace(obs,metrics={"time_remaining":0})
    if change in ("unknown","result"):obs=replace(obs,phase=change)
    if change=="single_sample":obs=replace(obs,entities=(replace(obs.entities[0],samples=1),*obs.entities[1:]))
    if change=="duplicate":obs=replace(obs,entities=(*obs.entities,replace(obs.entities[0],track_id=2)))
    if change=="ambiguous":obs=replace(obs,metrics={"ambiguous_player":True})
    if change=="incomplete":obs=replace(obs,entities=tuple(e for e in obs.entities if e.kind!="corridor" or e.y>=.2))
    assert decide_mining_reactive(obs,{"allow_contact_mitigation":True}).state==ControllerState.neutral()


def test_contact_mitigation_does_not_replace_existing_wall_recovery_or_safe_planning():
    parameters={"allow_contact_mitigation":True}
    assert decide_mining_reactive(world(x=.055),parameters).diagnostics["mode"]=="wall_recovery"
    assert decide_mining_reactive(world(),parameters).diagnostics["mode"]=="accelerated"


@pytest.mark.parametrize("sequence,expected_sign",[(3175,-1),(3456,-1),(2226,0)])
def test_recorded_r5_contact_mitigation_boundaries(sequence,expected_sign):
    """Replay PC observations only; this does not validate an actual action."""
    import json
    from pathlib import Path
    directory=Path(__file__).resolve().parents[1]/"artifacts/mining-supervision/round-05"
    if not (directory/"trace.jsonl").exists():pytest.skip("local NS R5 evidence not bundled")
    rows=[json.loads(line) for line in (directory/"trace.jsonl").read_text(encoding="utf-8").splitlines()]
    data=next(row["observation"] for row in rows if row["observation"]["sequence"]==sequence)
    obs=Observation(Game.MINING,data["timestamp"],data["sequence"],data["phase"],data["confidence"],
                    tuple(Entity(**e) for e in data["entities"]),metrics=data["metrics"])
    parameters=json.loads((directory/"summary.json").read_text(encoding="utf-8"))["parameters"]
    parameters["allow_contact_mitigation"]=True
    decision=decide_mining_reactive(obs,parameters)
    assert not decision.state.buttons and not decision.diagnostics["path"]
    if expected_sign:
        assert expected_sign*decision.state.left_stick.x>0
        assert decision.diagnostics["mode"]=="contact_mitigation"
    else:
        assert decision.state==ControllerState.neutral()
        assert decision.diagnostics.get("mode")!="contact_mitigation"


def test_capture_clock_prefix_advances_old_world_and_counts_effective_delay_once():
    obs=world(vx=.1,metrics={"observation_age_seconds":.11,"forward_speed":.7,
        "forward_speed_confidence":.9,"forward_speed_observed":True,
        "forward_speed_units":"viewport_heights_per_second","forward_speed_source":"landmark_tracking"})
    decision=decide_mining_reactive(obs,{"prediction_timing":"capture_clock","effective_delay_seconds":.12,
                                        "lookahead_depth":.5,"maximum_steps":20})
    timing=decision.diagnostics["timing_compensation"]
    end=timing["new_action_can_affect_capture_state_at"]
    assert end["seconds"]==pytest.approx(.23)
    assert end["x"]==pytest.approx(.523)
    assert end["depth"]==pytest.approx(.361)
    assert timing["extra_capture_buffer_seconds_added"]==0
    assert timing["current_scroll_observed"] is True
    assert timing["future_command_controls_prefix"] is False
    prefix=[p for p in decision.diagnostics["path"] if p.get("phase")]
    assert prefix and all(p["steering"] is None for p in prefix)
    assert prefix[-1]["seconds"]==pytest.approx(.23)
    assert decision.diagnostics["path"][len(prefix)]["seconds"]>.23


def test_capture_clock_replays_pending_report_instead_of_applying_new_action_early():
    import math
    metrics={"observation_age_seconds":.1,"mining_input_history_source":"backend_send_state",
        "mining_input_history":[{"after":.5,"axis":0.,"zr":False},{"after":1.05,"axis":.4,"zr":False}]}
    parameters={"prediction_timing":"capture_clock","effective_delay_seconds":.12,
        "current_forward_speed_hypothesis":.25,"require_verified_input_history":True,
        "motion_models":{"normal":{"input_gain_per_second":2.,"drag_per_second":4.,"maximum_lateral_speed":1.}}}
    decision=decide_mining_reactive(world(metrics=metrics),parameters)
    timing=decision.diagnostics["timing_compensation"]
    end=timing["new_action_can_affect_capture_state_at"]
    # Submitted at capture+.05, response at +.17, so exactly .05s of old
    # rightward input occurs before the NEW action's +.22s response point.
    assert end["x"]==pytest.approx(.5+.2*.05-.2*(1-math.exp(-.2))/4)
    assert end["velocity"]==pytest.approx(.2*(1-math.exp(-.2)))
    assert end["depth"]==pytest.approx(.255)
    assert timing["input_history_source"]=="actual_backend_reports_shifted_on_capture_clock"


def test_submission_wait_is_separate_from_frame_age_and_fitted_report_delay():
    decision=decide_mining_reactive(world(vx=.1,metrics={"observation_age_seconds":.11}),{
        "prediction_timing":"capture_clock","effective_delay_seconds":.12,
        "command_submission_delay_seconds":.09,"current_forward_speed_hypothesis":.7})
    timing=decision.diagnostics["timing_compensation"]
    assert timing["prefix_seconds"]==pytest.approx(.32)
    assert timing["new_action_can_affect_capture_state_at"]["x"]==pytest.approx(.532)
    assert timing["new_action_can_affect_capture_state_at"]["depth"]==pytest.approx(.424)
    assert timing["effective_delay_seconds"]==.12
    assert timing["command_submission_delay_source"]=="configured_decision_to_backend_report_hypothesis"
    assert timing["extra_capture_buffer_seconds_added"]==0


def test_guard_submission_clock_is_explicitly_approximate_and_for_planning_only():
    obs=world(metrics={"observation_age_seconds":.05,"mining_input_history_source":"guard_submission_after",
        "mining_input_history":[{"after":.5,"axis":0.,"zr":False}]})
    decision=decide_mining_reactive(obs,{"prediction_timing":"capture_clock","require_verified_input_history":True})
    timing=decision.diagnostics["timing_compensation"]
    assert timing["input_history_source"]=="guard_submission_reports_approx_backend_after"
    assert timing["input_history_for_planning_only"] is True


def test_capture_clock_accepts_full_actual_controller_reports():
    metrics={"observation_age_seconds":.05,"mining_input_history_source":"commands.jsonl.after",
        "mining_input_history":[{"after":.5,"state":{"buttons":["ZR"],"left_stick":{"x":0,"y":0}}}]}
    decision=decide_mining_reactive(world(metrics=metrics),{"prediction_timing":"capture_clock",
        "effective_delay_seconds":.125,"require_verified_input_history":True})
    assert decision.diagnostics["timing_compensation"]["prefix_seconds"]==pytest.approx(.175)
    assert decision.diagnostics["timing_compensation"]["input_history_source"]=="actual_backend_reports_shifted_on_capture_clock"


def mode_prefix_world(history,**extra):
    return world(metrics={"observation_age_seconds":.1,"mining_input_history_source":"backend_send_state",
        "mining_input_history":history,**extra})


def mode_prefix_parameters():
    return {"prediction_timing":"capture_clock","effective_delay_seconds":.125,
        "dispatch_delay_seconds":.03,"prefix_forward_from_input_history":True,
        "current_forward_speed_hypothesis":.5,
        "motion_models":{"normal":{"forward_speed":.25},"zr":{"forward_speed":.5}}}


def test_normal_prefix_uses_submitted_mode_speed_instead_of_fixed_fast_speed():
    decision=decide_mining_reactive(mode_prefix_world([{"after":.5,"axis":0.,"zr":False}]),mode_prefix_parameters())
    timing=decision.diagnostics["timing_compensation"]
    assert timing["prefix_forward_from_input_history"] is True
    assert timing["current_scroll_observed"] is False
    assert timing["current_scroll_speed"] is None
    assert timing["new_action_can_affect_capture_state_at"]["depth"]==pytest.approx(.2+.255*.25)


def test_mode_change_in_prefix_integrates_both_delayed_submitted_modes_once():
    obs=mode_prefix_world([{"after":.5,"axis":0.,"zr":False},{"after":.95,"axis":0.,"zr":True}])
    timing=decide_mining_reactive(obs,mode_prefix_parameters()).diagnostics["timing_compensation"]
    # ZR reported 50ms before the picture acts at capture +75ms, not at capture.
    assert timing["prefix_seconds"]==pytest.approx(.255)
    assert timing["new_action_can_affect_capture_state_at"]["depth"]==pytest.approx(.2+.075*.25+.180*.5)
    assert "not_measured_transition" in timing["current_scroll_source"]


def test_reliable_zero_scroll_overrides_submitted_mode_hypothesis():
    obs=mode_prefix_world([{"after":.5,"axis":0.,"zr":True}],forward_speed=0.,
        forward_speed_observed=True,forward_speed_confidence=.9,forward_speed_units="viewport_heights_per_second")
    timing=decide_mining_reactive(obs,mode_prefix_parameters()).diagnostics["timing_compensation"]
    assert timing["prefix_forward_from_input_history"] is False
    assert timing["current_scroll_observed"] is True
    assert timing["new_action_can_affect_capture_state_at"]["depth"]==pytest.approx(.2)


def test_incomplete_mode_history_keeps_explicit_fixed_speed_fallback():
    obs=mode_prefix_world([{"after":1.02,"axis":0.,"zr":False}])
    timing=decide_mining_reactive(obs,mode_prefix_parameters()).diagnostics["timing_compensation"]
    assert timing["prefix_forward_from_input_history"] is False
    assert timing["new_action_can_affect_capture_state_at"]["depth"]==pytest.approx(.2+.255*.5)


def test_non_boolean_prefix_mode_option_fails_closed():
    decision=decide_mining_reactive(world(),{**mode_prefix_parameters(),"prefix_forward_from_input_history":"true"})
    assert decision.state==ControllerState.neutral()
    assert "必须为布尔值" in decision.reason


@pytest.mark.parametrize("speed",[0.,.36])
def test_prefix_only_landmark_speed_does_not_overwrite_future_mode_models(speed):
    obs=mode_prefix_world([{"after":.5,"axis":0.,"zr":True}],
        current_forward_speed=speed,current_forward_speed_observed=True,current_forward_speed_confidence=.85,
        current_forward_speed_units="viewport_heights_per_second",current_forward_speed_source="tracked_landmark_consensus")
    parameters=mode_prefix_parameters()
    decision=decide_mining_reactive(obs,parameters)
    timing=decision.diagnostics["timing_compensation"]
    assert timing["new_action_can_affect_capture_state_at"]["depth"]==pytest.approx(.2+.255*speed)
    assert timing["current_scroll_observation_scope"]=="prefix_only"
    assert timing["prefix_forward_from_input_history"] is False
    from raiders_helper.mining_reactive import _motion
    assert _motion(parameters,obs,"normal")[0].forward==.25
    assert _motion(parameters,obs,"zr")[0].forward==.5


def test_missing_prefix_consensus_immediately_returns_to_submitted_mode():
    metrics=dict(current_forward_speed=.01,current_forward_speed_observed=False,current_forward_speed_confidence=.85,
                 current_forward_speed_units="viewport_heights_per_second")
    obs=mode_prefix_world([{"after":.5,"axis":0.,"zr":False}],**metrics)
    timing=decide_mining_reactive(obs,mode_prefix_parameters()).diagnostics["timing_compensation"]
    assert timing["prefix_forward_from_input_history"] is True
    assert timing["new_action_can_affect_capture_state_at"]["depth"]==pytest.approx(.2+.255*.25)


def test_inevitable_old_input_rock_contact_can_reduce_overlap_without_fake_safe_path():
    obs=world(entity("rock",.55,.31,.10,.08),metrics={"observation_age_seconds":.1})
    decision=decide_mining_reactive(obs,{"prediction_timing":"capture_clock","effective_delay_seconds":.12,
        "current_forward_speed_hypothesis":.5,"allow_contact_mitigation":True})
    assert decision.diagnostics["mode"]=="contact_mitigation"
    assert decision.diagnostics["current_rock_contacts"]==0
    assert decision.diagnostics["uncontrollable_prefix_rock_contacts"]==1
    assert decision.state.left_stick.x<0 and not decision.state.buttons
    assert decision.diagnostics["path"]==[]
    assert decision.diagnostics["relative_to_neutral"]["rock_overlap_reduction"]>0


def test_missing_history_and_unobserved_scroll_remain_explicit_hypotheses():
    decision=decide_mining_reactive(world(metrics={"observation_age_seconds":.1}),{
        "prediction_timing":"capture_clock","effective_delay_seconds":.12,"current_forward_speed_hypothesis":.75})
    timing=decision.diagnostics["timing_compensation"]
    assert timing["current_scroll_speed"]==.75 and timing["current_scroll_observed"] is False
    assert timing["current_scroll_source"]=="configured_current_speed_hypothesis_not_observed"
    assert timing["input_history_source"]=="unknown_input_constant_observed_vx"
    strict=decide_mining_reactive(world(),{"prediction_timing":"capture_clock","require_verified_input_history":True})
    assert strict.state==ControllerState.neutral() and "实际发送历史" in strict.reason


def test_current_observed_zero_scroll_does_not_become_player_y_or_assumed_motion():
    metrics={"observation_age_seconds":.1,"forward_speed":0.,"forward_speed_confidence":.9,
             "forward_speed_observed":True,"forward_speed_units":"viewport_heights_per_second",
             "forward_speed_source":"landmark_tracking"}
    decision=decide_mining_reactive(world(metrics=metrics),{"prediction_timing":"capture_clock","effective_delay_seconds":.12})
    timing=decision.diagnostics["timing_compensation"]
    assert timing["current_scroll_observed"] is True
    assert timing["new_action_can_affect_capture_state_at"]["depth"]==pytest.approx(.2)
    assert "not_acceleration_model" in timing["future_scroll_assumption"]


def test_obstacle_crossed_during_old_image_and_pending_delay_is_not_skipped():
    obstacle=entity("obstacle",.5,.27,.9,.005)
    decision=decide_mining_reactive(world(obstacle,metrics={"observation_age_seconds":.1}),{
        "prediction_timing":"capture_clock","effective_delay_seconds":.12,"current_forward_speed_hypothesis":.5})
    assert decision.state==ControllerState.neutral()
    assert decision.diagnostics["search_failure"]=="pre_action_prefix_hits_observed_obstacle"
    assert not decision.diagnostics["path"]


@pytest.mark.parametrize("change",["future","backward_clock","bad_axis","bad_mode"])
def test_invalid_actual_history_never_produces_a_fake_safe_plan(change):
    history=[{"after":.5,"axis":0.,"zr":False},{"after":1.05,"axis":.4,"zr":False}]
    if change=="future":history[-1]["after"]=1.3
    if change=="backward_clock":history[-1]["after"]=.4
    if change=="bad_axis":history[-1]["axis"]=float("nan")
    if change=="bad_mode":history[-1]["zr"]="False"
    obs=world(metrics={"observation_age_seconds":.1,"mining_input_history_source":"backend_send_state","mining_input_history":history})
    assert decide_mining_reactive(obs,{"prediction_timing":"capture_clock"}).state==ControllerState.neutral()


@pytest.mark.parametrize("parameters",[
    {"effective_delay_seconds":float("nan")},{"effective_delay_seconds":-.01},
    {"effective_delay_seconds":.501},{"current_forward_speed_hypothesis":float("nan")},
    {"motion_models":{"normal":{"neutral_velocity":float("nan")}}},
])
def test_invalid_capture_clock_parameters_fail_closed(parameters):
    assert decide_mining_reactive(world(),{"prediction_timing":"capture_clock",**parameters}).state==ControllerState.neutral()


def test_search_limits_the_actual_simulated_axis_before_integrating():
    decision=decide_mining_reactive(world(vx=.3),{"maximum_axis":.25})
    assert decision.diagnostics["path"]
    assert all(abs(p["steering"])<=.25 for p in decision.diagnostics["path"])


@pytest.mark.parametrize("sign",[-1,1])
def test_constant_acceleration_speed_cap_integrates_time_before_and_after_hit(sign):
    from raiders_helper.mining_reactive import _Motion,_advance
    distance,velocity=_advance(_Motion(.25,4.,0.,.2),0.,sign,.2)
    # a=4 reaches v=.2 at .05s: .005 distance under acceleration plus
    # .2*.15=.030 after the cap. An endpoint trapezoid would give .020.
    assert distance==pytest.approx(sign*.035)
    assert velocity==pytest.approx(sign*.2)


def test_positive_star_reward_actively_steers_to_a_safe_reachable_time_star():
    star=replace(entity("time_star",.63,.50,.03,.03),track_id=42)
    baseline=decide_mining_reactive(world(star))
    decision=decide_mining_reactive(world(star),{"star_reward":.10})
    assert baseline.state.left_stick.x==0
    assert decision.state.left_stick.x>0
    assert len(decision.diagnostics["collected_predicted"])==1
    assert decision.diagnostics["star_target"]["key"]=="t:42"
    assert decision.diagnostics["star_target"]["confirmed"] is False
    assert decision.diagnostics["star_collection_confirmed"] is False
    assert "尚未确认拾取" in decision.reason


def test_one_star_reward_is_not_repeated_across_steps_or_duplicate_detections():
    star=replace(entity("time_star",.5,.4,.03,.12),track_id=42)
    decision=decide_mining_reactive(world(star,star),{"star_reward":.1})
    assert decision.diagnostics["visible_time_star_candidates"]==1
    assert len(decision.diagnostics["collected_predicted"])==1
    assert decision.diagnostics["star_reward_total_predicted"]==pytest.approx(.1)


def test_distinct_reachable_stars_are_counted_once_each_without_modifying_time():
    a=replace(entity("time_star",.5,.35,.025,.025),track_id=41)
    b=replace(entity("time_star",.5,.47,.025,.025),track_id=42)
    obs=world(a,b,metrics={"time_remaining":6.})
    decision=decide_mining_reactive(obs,{"star_reward":.1})
    assert len(decision.diagnostics["collected_predicted"])==2
    assert decision.diagnostics["star_reward_total_predicted"]==pytest.approx(.2)
    assert obs.metrics["time_remaining"]==6.
    assert decision.diagnostics["star_collection_confirmed"] is False


def test_star_inside_boulder_does_not_relax_hard_collision_constraints():
    star=replace(entity("time_star",.63,.48,.025,.025),track_id=42)
    rock=entity("rock",.63,.48,.15,.15)
    decision=decide_mining_reactive(world(star,rock),{"star_reward":1e6})
    assert not decision.diagnostics["collected_predicted"]
    for a,b in zip(decision.diagnostics["path"],decision.diagnostics["path"][1:]):
        assert contact_fraction(a["x"],a["depth"],b["x"],b["depth"],rock,.036,.024) is None


def test_star_outside_true_road_is_not_a_reason_to_drive_through_a_wall():
    road=[entity("corridor",.5,y,.4,.001) for y in (0,.15,.2,.4,.7,1)]
    star=replace(entity("time_star",.82,.48,.03,.03),track_id=42)
    decision=decide_mining_reactive(world(star,road=road),{"star_reward":1e6})
    assert decision.diagnostics["visible_time_star_candidates"]==0
    assert not decision.diagnostics["collected_predicted"]
    assert all(.336<=p["x"]<=.664 for p in decision.diagnostics["path"])


def test_star_behind_full_width_wall_does_not_open_a_route():
    star=replace(entity("time_star",.5,.5,.03,.03),track_id=42)
    decision=decide_mining_reactive(world(star,entity("wall",.5,.38,1,.02)),{"star_reward":1e6})
    assert decision.state==ControllerState.neutral()
    assert not decision.diagnostics["collected_predicted"]


def test_star_already_touched_by_uncontrollable_prefix_is_not_new_action_reward():
    star=replace(entity("time_star",.5,.2,.03,.03),track_id=42)
    decision=decide_mining_reactive(world(star),{"star_reward":.1})
    assert not decision.diagnostics["collected_predicted"]
    assert decision.diagnostics["star_reward_total_predicted"]==0


@pytest.mark.parametrize("parameters",[{"star_reward":-1},{"star_reward":float("nan")},{"star_pickup_radius_scale":1.01}])
def test_star_reward_never_widens_the_observed_body_or_accepts_invalid_cost(parameters):
    assert decide_mining_reactive(world(),parameters).state==ControllerState.neutral()


def test_sharp_turn_uses_fraction_of_reduced_available_stick():
    from raiders_helper.mining_reactive import _Motion,_Node,_turn_authority
    plan=_Node(.5,0,.3,0,((.5,.25,.1,.65,0),(.51,.3,.2,.65,.1)))
    sharp,audit=_turn_authority(plan,_Motion(.5,3.17,22.14,.25),.65,{})
    assert sharp
    assert audit["early_axis_fraction"]==1
    assert audit["available_steady_slope_hypothesis"]<.2


def test_soft_steering_cost_uses_actual_axis_and_allows_gentle_intermediate_input():
    metrics={"mining_input_history_source":"guard_submission_after",
        "mining_input_history":[{"after":.5,"axis":.3,"zr":True}]}
    params={"prediction_timing":"capture_clock","effective_delay_seconds":.02,"maximum_axis":.65}
    obs=world(x=.48,metrics=metrics)
    baseline=decide_mining_reactive(obs,params)
    smoother=decide_mining_reactive(obs,{**params,"steering_change_weight":.003})
    assert baseline.state.left_stick.x==0
    assert smoother.state.left_stick.x/32767==pytest.approx(.15,abs=1/32767)
    assert smoother.diagnostics["previous_axis"]==.3
    assert smoother.diagnostics["previous_axis_source"]=="guard_submission_reports_approx_backend_after"


def test_soft_cost_does_not_hard_limit_urgent_opposite_braking():
    metrics={"mining_input_history_source":"guard_submission_after",
        "mining_input_history":[{"after":.5,"axis":.65,"zr":False}]}
    decision=decide_mining_reactive(world(x=.8,vx=.6,metrics=metrics),{
        "prediction_timing":"capture_clock","effective_delay_seconds":0,
        "maximum_axis":.65,"steering_change_weight":.003})
    assert decision.state.left_stick.x/32767==pytest.approx(-.65,abs=1/32767)
    assert decision.diagnostics["route_status"]=="complete"
    road=Road(world().of_kind("corridor"))
    for a,b in zip(decision.diagnostics["path"],decision.diagnostics["path"][1:]):
        assert road.swept_safe(a["x"],a["depth"],b["x"],b["depth"],.036,.024)


def test_old_route_is_reprojected_with_declared_scroll_not_same_index():
    from raiders_helper.mining_reactive import _route_reference,_route_x
    metrics={"mining_previous_route":{"timestamp":.8,"forward_speed":.5,
        "scroll_source":"configured_scroll_hypothesis","path":[{"x":.3,"depth":.4},{"x":.7,"depth":.6}]}}
    points,source=_route_reference(world(metrics=metrics),1)
    assert _route_x(points,.4)==pytest.approx(.5)
    assert source=="reprojected_configured_scroll_hypothesis"
    assert _route_reference(world(metrics=metrics),0)==((),"disabled")


def test_mixed_boost_finishes_full_horizon_when_pure_boost_cannot_and_has_one_prefix():
    road=[entity("corridor",x,y,.30,.001) for y,x in
        [(0,.4),(.15,.4),(.2,.4),(.4,.4),(.6,.6),(.8,.7),(1,.7)]]
    params={"maximum_axis":.65,"lookahead_depth":.68,"maximum_steps":30,"beam_width":18,
        "steering_change_weight":.003,"mixed_zr_prefix_seconds":[.1,.2,.3],
        "prediction_timing":"capture_clock","effective_delay_seconds":.125,
        "dispatch_delay_seconds":.03,"current_forward_speed_hypothesis":0,
        "motion_models":{"normal":{"forward_speed":.25,"input_gain_per_second":2.65,
            "drag_per_second":8.38,"maximum_lateral_speed":.4},
            "zr":{"forward_speed":.5,"input_gain_per_second":3.17,
            "drag_per_second":22.14,"maximum_lateral_speed":.25}}}
    metrics={"observation_age_seconds":.1,"mining_input_history_source":"guard_submission_after",
        "mining_input_history":[{"after":.5,"axis":0.,"zr":False}]}
    decision=decide_mining_reactive(world(road=road,x=.4,metrics=metrics),params)
    d=decision.diagnostics
    assert d["mode"]=="accelerated_then_normal"
    assert d["fast_route_status"]=="partial"
    assert d["route_status"]=="complete"
    assert Button.ZR in decision.state.buttons
    schedule=d["future_mode_schedule"]
    assert schedule[0]["start_seconds"]==pytest.approx(.255)
    assert schedule[0]["end_seconds"]==schedule[1]["start_seconds"]==pytest.approx(.555)
    assert d["mode_switch_delay_added_seconds"]==0
    assert d["path"][-1]["depth"]==pytest.approx(.88)
    assert any(p["accelerate"] is False for p in d["path"])
    for a,b in zip(d["path"],d["path"][1:]):
        assert Road(road).swept_safe(a["x"],a["depth"],b["x"],b["depth"],.036,.024)
    # The suffix starts from the prefix's actual predicted velocity, without
    # resetting it or paying capture/report delay for a second time.
    from raiders_helper.mining_reactive import _advance,_motion
    obs=world(road=road,x=.4,metrics=metrics)
    normal,_=_motion(params,obs,"normal");boosted,_=_motion(params,obs,"zr")
    x=.4;velocity=0.;previous_time=.255
    for point in d["path"]:
        if point["steering"] is None:continue
        distance,velocity=_advance(boosted if point["accelerate"] else normal,velocity,
                                   point["steering"],point["seconds"]-previous_time)
        x+=distance;previous_time=point["seconds"]
        assert x==pytest.approx(point["x"],abs=1e-12)


def test_mixed_prefix_never_claims_safe_fast_route_through_full_width_wall():
    decision=decide_mining_reactive(world(entity("wall",.5,.55,1,.02),
        entity("time_star",.5,.65,.03,.03)),{"lookahead_depth":.6,"maximum_steps":30,
            "star_reward":1e6,"steering_change_weight":.003,"mixed_zr_prefix_seconds":[.1,.2,.3]})
    assert Button.ZR not in decision.state.buttons
    assert decision.diagnostics["mixed_zr_prefix_seconds"] is None
    assert decision.diagnostics["collected_predicted"]==[]
    assert all(p["depth"]<.55-.01-.024 for p in decision.diagnostics["path"])


@pytest.mark.parametrize("parameters",[{"steering_change_weight":-1},{"route_continuity_weight":float("nan")},
    {"mixed_zr_prefix_seconds":[0]},{"mixed_zr_prefix_seconds":[float("nan")]},
    {"sharp_turn_axis_fraction":0}])
def test_invalid_smoothing_and_mode_schedule_fail_closed(parameters):
    assert decide_mining_reactive(world(),parameters).state==ControllerState.neutral()
