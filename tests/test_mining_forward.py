"""Synthetic integration identities; no claim that coefficients are game laws."""
from dataclasses import replace
import math

import pytest

from raiders_helper.mining_forward import ForwardDynamics,seed_context
from raiders_helper.mining_forward_seed import ForwardSeedEstimator
from raiders_helper.mining_reactive import decide_mining_reactive,_turn_authority,_Motion,_Node
from raiders_helper.models import Entity,Observation,Game


MODEL=ForwardDynamics(.1,.25,.8)


@pytest.mark.parametrize("boost",[False,True])
@pytest.mark.parametrize("speed",[0.,.25,.4])
def test_vertical_integration_composes_and_inverts(boost,speed):
    d1,v1=MODEL.advance(speed,boost,.17)
    d2,v2=MODEL.advance(v1,boost,.39)
    whole,vend=MODEL.advance(speed,boost,.56)
    assert d1+d2==pytest.approx(whole,abs=1e-12)
    assert v2==pytest.approx(vend,abs=1e-12)
    assert MODEL.time_to_distance(speed,boost,whole)==pytest.approx(.56,abs=1e-10)


def test_normal_release_keeps_forward_inertia_instead_of_instant_reset():
    distance,speed=MODEL.advance(.4,False,.1)
    assert .25<speed<.4
    assert distance>.25*.1
    assert distance==pytest.approx(.025+.15*.8*(1-math.exp(-.1/.8)))


def test_mixed_vertical_schedule_has_continuous_velocity_and_same_endpoint():
    distance=.5;switch=.2
    duration=MODEL.arrival_time(.3,distance,True,switch)
    first,speed=MODEL.advance(.3,True,switch)
    second,_=MODEL.advance(speed,False,duration-switch)
    assert first+second==pytest.approx(distance)


def setup_world(*objects,speed=.3,boost=False):
    metrics={"observation_age_seconds":.05,"mining_input_history_source":"backend_send_state",
        "mining_input_history":[{"after":.5,"axis":0.,"zr":boost}],
        "mining_forward_seed":{"valid":True,"speed":speed,"timestamp":1.,"age_seconds":.1,
            "reference_timestamp":.9,"source":"observed_window_propagated","units":"viewport_heights_per_second"}}
    entities=(Entity("player",.5,.2,.04,.04,.95,1,samples=5),
              *[Entity("corridor",.5,y,.94,.001,.95) for y in (0,.15,.2,.4,.6,.8,1)],*objects)
    obs=Observation(Game.MINING,1.,1,"playing",.95,entities,metrics=metrics)
    params={"prediction_timing":"capture_clock","effective_delay_seconds":.125,"dispatch_delay_seconds":.03,
        "lookahead_depth":.4,"maximum_steps":30,"step_seconds":.1,
        "forward_dynamics":{"enabled":True,"zr_acceleration":.1,"normal_speed":.25,"normal_tau_seconds":.8}}
    return obs,params


def test_prefix_and_future_share_continuous_forward_speed_but_preserve_lateral_model():
    obs,params=setup_world()
    decision=decide_mining_reactive(obs,params)
    prefix_seconds=.05+.03+.125
    distance,speed=MODEL.advance(.3,False,prefix_seconds)
    timing=decision.diagnostics["timing_compensation"]
    assert timing["new_action_can_affect_capture_state_at"]["depth"]==pytest.approx(.2+distance)
    assert decision.diagnostics["forward_dynamics"]["prefix_end_forward_speed"]==pytest.approx(speed)
    assert decision.diagnostics["forward_dynamics"]["active"] is True
    path=decision.diagnostics["path"]
    first=next(p for p in path if p.get("phase") is None)
    ds,v=MODEL.advance(speed,True,first["seconds"]-prefix_seconds)
    assert first["depth"]==pytest.approx(.2+distance+ds)
    assert decision.diagnostics["model"]["input_gain_per_second"]==5.
    assert decision.diagnostics["route_status"]=="complete"
    expected=prefix_seconds+MODEL.time_to_distance(speed,True,.4-distance)
    assert path[-1]["seconds"]==pytest.approx(expected)
    assert path[-1]["depth"]==pytest.approx(.6)


def test_prefix_mode_change_is_shifted_by_effective_delay_once():
    obs,params=setup_world()
    obs.metrics["mining_input_history"].append({"after":.95,"axis":0.,"zr":True})
    decision=decide_mining_reactive(obs,params)
    first,v=MODEL.advance(.3,False,.075)
    second,v=MODEL.advance(v,True,.205-.075)
    timing=decision.diagnostics["timing_compensation"]
    assert timing["new_action_can_affect_capture_state_at"]["depth"]==pytest.approx(.2+first+second)
    assert decision.diagnostics["forward_dynamics"]["prefix_end_forward_speed"]==pytest.approx(v)


@pytest.mark.parametrize("change",["missing","old","wrong_clock","wrong_units","zero","history_missing"])
def test_unusable_seed_has_explicit_legacy_fallback(change):
    obs,params=setup_world();seed=obs.metrics["mining_forward_seed"]
    if change=="missing":seed.update(valid=False,reason="no_current_consensus")
    if change=="old":seed["age_seconds"]=1.51
    if change=="wrong_clock":seed["timestamp"]=.9
    if change=="wrong_units":seed["units"]="pixels"
    if change=="zero":seed["speed"]=0.
    if change=="history_missing":obs.metrics["mining_input_history"]=[]
    decision=decide_mining_reactive(obs,params)
    diagnostic=decision.diagnostics["forward_dynamics"]
    assert diagnostic["active"] is False and diagnostic["fallback_reason"]
    legacy=decide_mining_reactive(obs,{**params,"forward_dynamics":{"enabled":False}})
    assert decision.state==legacy.state
    if change=="missing":assert diagnostic["seed_unavailable_reason"]=="no_current_consensus"


@pytest.mark.parametrize("change",["hit","powered","buff_entity","nonplaying"])
def test_dynamic_model_cannot_consume_seed_during_unfitted_conditions(change):
    obs,params=setup_world()
    if change=="hit":obs.metrics["collision_suspected"]=True
    if change=="powered":obs.metrics["player_appearance"]="powered"
    if change=="buff_entity":obs=replace(obs,entities=(*obs.entities,Entity("egg_buff",.5,.2,.04,.04,.95)))
    if change=="nonplaying":obs=replace(obs,phase="unknown")
    model,speed,diagnostic=seed_context(obs,params)
    assert model is None and speed is None and not diagnostic["active"]


def test_dynamic_star_bound_uses_integrated_normal_arrival_time():
    obs,params=setup_world(Entity("time_star",.5,.55,.02,.02,.95,3),speed=.4)
    params.update(compare_star_modes=True,star_mode_upper_bound_pruning=True,star_reward=.04)
    decision=decide_mining_reactive(obs,params)
    comparison=decision.diagnostics["star_mode_comparison"]
    assert comparison["skipped"]=="net_gain_upper_bound_insufficient"
    assert "dynamic_model_time" in comparison["bound_scope"]
    prefix=decision.diagnostics["timing_compensation"]["new_action_can_affect_capture_state_at"]
    v=decision.diagnostics["forward_dynamics"]["prefix_end_forward_speed"]
    expected=prefix["seconds"]+MODEL.time_to_distance(v,False,.6-prefix["depth"])-decision.diagnostics["planned_seconds"]
    assert comparison["extra_travel_seconds_if_complete"]==pytest.approx(expected)


def test_turn_authority_uses_lateral_rate_when_forward_speed_is_building():
    # Lateral .08/s is below available .1/s, although dy/dt=.2 differs
    # from the legacy .5 mode constant. A fixed-slope ratio would reject it.
    path=((.5,.2,.1,.5,.08),(.508,.22,.2,.5,.08),(.516,.24,.3,.5,.08))
    plan=_Node(.516,.08,.24,0.,path)
    sharp,diagnostics=_turn_authority(plan,_Motion(.5,1.,10.,.3),1.,{})
    assert sharp is False
    assert diagnostics["early_slope_authority_fraction"]==pytest.approx(.8)


def test_dynamic_coefficients_must_be_explicit_and_seed_age_cannot_be_extended():
    obs,params=setup_world()
    with pytest.raises(ValueError):seed_context(obs,{"forward_dynamics":{"enabled":True}})
    params["forward_dynamics"]["maximum_seed_age_seconds"]=2.
    with pytest.raises(ValueError):seed_context(obs,params)


def test_real_window_seed_and_policy_prefix_integrate_the_same_clock_once():
    obs,params=setup_world(boost=True)
    obs.metrics.pop("mining_forward_seed")
    obs.metrics.update(mining_input_history_observed_until=1.05,
        current_forward_speed=.3,current_forward_speed_observed=True,
        current_forward_speed_source="tracked_landmark_consensus",
        current_forward_speed_units="viewport_heights_per_second",
        current_forward_speed_confidence=.85,current_forward_speed_reference_timestamp=.95,
        current_forward_speed_window_seconds=.1,current_forward_speed_method="paired_positions")
    ForwardSeedEstimator(params).attach(obs)
    decision=decide_mining_reactive(obs,params)
    dynamic=decision.diagnostics["forward_dynamics"]
    # Measured speed belongs at t=.95. Capture t=1.0 and prefix end t=1.205
    # have one continuous ZR history, with no duplicate effective delay.
    assert dynamic["seed_speed"]==pytest.approx(.305)
    assert dynamic["prefix_end_forward_speed"]==pytest.approx(.3255)
    assert dynamic["seed_source"]=="observed_window_propagated"
    prefix=decision.diagnostics["timing_compensation"]["new_action_can_affect_capture_state_at"]
    assert prefix["depth"]==pytest.approx(.2+.305*.205+.5*.1*.205**2)
