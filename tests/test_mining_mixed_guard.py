"""A promised mixed release constrains replanning, not just output buttons."""
from dataclasses import replace
import math

import pytest

from raiders_helper.control.buttons import Button
from raiders_helper.models import Entity,Game,Observation
from raiders_helper import mining_reactive as core
from raiders_helper.mining_mixed_guard import MixedDeadlineGuard


def world(phase="eligible",**guard):
    metrics={"observation_age_seconds":.05,"planning_delay_seconds":.04,
        "mining_input_history_source":"backend_send_state",
        "mining_input_history":[{"after":.5,"axis":0.,"zr":True}],
        "mining_forward_seed":{"valid":True,"speed":.3,"timestamp":1.,"age_seconds":.1,
            "reference_timestamp":.9,"source":"observed_window_propagated","units":"viewport_heights_per_second"},
        "mining_mixed_guard":{"enabled":True,"phase":phase,**guard}}
    entities=(Entity("player",.5,.2,.04,.04,.95,1,samples=5),
        *[Entity("corridor",.5,y,.94,.001,.95) for y in (0,.15,.2,.4,.6,.8,1)])
    obs=Observation(Game.MINING,1.,1,"playing",.95,entities,metrics=metrics)
    params={"prediction_timing":"capture_clock","effective_delay_seconds":.125,"dispatch_delay_seconds":.03,
        "lookahead_depth":.4,"maximum_steps":30,"step_seconds":.1,"mixed_zr_prefix_seconds":[.2],
        "mixed_deadline_guard":True,
        "motion_models":{"normal":{"input_gain_per_second":2.},"zr":{"input_gain_per_second":5.}},
        "forward_dynamics":{"enabled":True,"zr_acceleration":.1,"normal_speed":.25,"normal_tau_seconds":.8}}
    return obs,params


def record_searches(monkeypatch):
    original=core._search;calls=[]
    def wrapper(road,player,hazards,motion,settings,boost=False,switch_after=None,suffix=None):
        calls.append((boost,switch_after,motion.input_gain))
        return original(road,player,hazards,motion,settings,boost,switch_after,suffix)
    monkeypatch.setattr(core,"_search",wrapper)
    return calls


def test_scheduled_release_splits_a_short_step_and_cannot_be_cancelled_by_pure_fast():
    obs,params=world("scheduled",release_at=1.12)
    decision=core.decide_mining_reactive(obs,params)
    guard=decision.diagnostics["mixed_guard"]
    assert guard["remaining_zr_seconds"]==pytest.approx(.03)
    assert decision.diagnostics["mode"]=="accelerated_then_normal"
    assert decision.diagnostics["mixed_zr_prefix_seconds"]==pytest.approx(.03)
    schedule=decision.diagnostics["future_mode_schedule"]
    assert obs.timestamp+schedule[0]["end_seconds"]==pytest.approx(1.12+.03+.125)
    assert schedule[1]["start_seconds"]==schedule[0]["end_seconds"]
    assert not guard["full_fast_rearm_verified"]
    assert decision.diagnostics["mixed_forward_response_assumption"]=="integrated_zr_acceleration_then_normal_exponential_response"


def test_replanning_keeps_the_first_release_clock_instead_of_restarting_point_two():
    obs,params=world("scheduled",release_at=1.20)
    first=core.decide_mining_reactive(obs,params)
    later=replace(obs,timestamp=1.05,sequence=2,metrics={**obs.metrics,
        "mining_forward_seed":{**obs.metrics["mining_forward_seed"],"timestamp":1.05}})
    second=core.decide_mining_reactive(later,params)
    assert first.diagnostics["mixed_zr_prefix_seconds"]==pytest.approx(.11)
    assert second.diagnostics["mixed_zr_prefix_seconds"]==pytest.approx(.06)
    for observation,decision in ((obs,first),(later,second)):
        end=decision.diagnostics["future_mode_schedule"][0]["end_seconds"]
        assert observation.timestamp+end==pytest.approx(1.20+.03+.125)


@pytest.mark.parametrize("release_at",[1.09,1.,.9])
def test_expired_scheduled_plan_recomputes_normal_dynamics_not_fast_axis_without_zr(monkeypatch,release_at):
    obs,params=world("scheduled",release_at=release_at)
    calls=record_searches(monkeypatch)
    decision=core.decide_mining_reactive(obs,params)
    assert calls and all(not boost and switch is None and gain==2. for boost,switch,gain in calls)
    assert Button.ZR not in decision.state.buttons
    assert decision.diagnostics["model"]["input_gain_per_second"]==2.
    assert decision.diagnostics["mixed_guard"]["normal_replanned"] is True
    assert decision.diagnostics["mixed_guard"]["force_normal"] is True


@pytest.mark.parametrize("metric",[None,{}, {"enabled":False,"phase":"eligible"},
    {"enabled":True,"phase":"invalid"},{"enabled":True,"phase":"scheduled"},
    {"enabled":True,"phase":"scheduled","release_at":math.inf}])
def test_enabled_guard_without_valid_supervisor_state_falls_back_to_normal_search(monkeypatch,metric):
    obs,params=world();obs.metrics["mining_mixed_guard"]=metric
    calls=record_searches(monkeypatch)
    decision=core.decide_mining_reactive(obs,params)
    assert calls and all(not boost for boost,_,_ in calls)
    assert Button.ZR not in decision.state.buttons
    assert decision.diagnostics["mixed_guard"]["normal_replanned"]


def test_braking_requires_normal_replan_until_supervisor_confirms_release(monkeypatch):
    obs,params=world("braking",allow_full_fast_rearm=False)
    calls=record_searches(monkeypatch)
    decision=core.decide_mining_reactive(obs,params)
    assert calls==[(False,None,2.)]
    assert decision.diagnostics["mixed_guard"]["normal_replanned"]
    assert not decision.diagnostics["mixed_guard"]["full_fast_rearm_verified"]
    assert Button.ZR not in decision.state.buttons


def test_release_confirmed_braking_can_rearm_only_with_selected_complete_pure_fast():
    obs,params=world("braking",allow_full_fast_rearm=True)
    decision=core.decide_mining_reactive(obs,params)
    assert decision.diagnostics["mode"]=="accelerated"
    assert decision.diagnostics["route_status"]=="complete"
    assert decision.diagnostics["mixed_guard"]["full_fast_rearm_verified"]
    assert not decision.diagnostics["mixed_guard"]["normal_replanned"]


def test_partial_fast_cannot_rearm_by_starting_another_mixed_deadline(monkeypatch):
    obs,params=world("braking",allow_full_fast_rearm=True)
    original=core._search;calls=[]
    def partial_fast(road,player,hazards,motion,settings,boost=False,switch_after=None,suffix=None):
        calls.append((boost,switch_after))
        if boost:return None,0,"partial"
        return original(road,player,hazards,motion,settings,boost,switch_after,suffix)
    monkeypatch.setattr(core,"_search",partial_fast)
    decision=core.decide_mining_reactive(obs,params)
    assert calls==[(True,None),(False,None)]
    assert Button.ZR not in decision.state.buttons
    assert not decision.diagnostics["mixed_guard"]["full_fast_rearm_verified"]
    assert decision.diagnostics["mixed_guard"]["normal_replanned"]


def test_disabled_guard_preserves_legacy_control_with_no_supervisor_metric():
    obs,params=world("scheduled",release_at=.9)
    params["mixed_deadline_guard"]=False
    baseline=core.decide_mining_reactive(obs,params)
    obs.metrics.pop("mining_mixed_guard")
    again=core.decide_mining_reactive(obs,params)
    assert baseline.state==again.state
    assert baseline.diagnostics["mode"]==again.diagnostics["mode"]=="accelerated"


def test_real_core_and_execution_guard_require_normal_replan_and_capture_clock_release_before_rearming():
    # A bending road needs a real mixed search; later wide-road snapshots let
    # the test exercise rearming without constructing policy diagnostic flags.
    obs,params=world()
    params.update(maximum_axis=.65,lookahead_depth=.68,beam_width=18,
        steering_change_weight=.003,mixed_zr_prefix_seconds=[.1,.2,.3],
        current_forward_speed_hypothesis=0,forward_dynamics={"enabled":False},
        motion_models={"normal":{"forward_speed":.25,"input_gain_per_second":2.65,
            "drag_per_second":8.38,"maximum_lateral_speed":.4},
            "zr":{"forward_speed":.5,"input_gain_per_second":3.17,
            "drag_per_second":22.14,"maximum_lateral_speed":.25}})
    road=tuple(Entity("corridor",x,y,.30,.001,.95) for y,x in
        [(0,.4),(.15,.4),(.2,.4),(.4,.4),(.6,.6),(.8,.7),(1,.7)])
    first_metrics={**obs.metrics,"observation_age_seconds":.1,"planning_delay_seconds":0.,
        "mining_input_history":[{"after":.5,"axis":0.,"zr":False}],
        "mining_input_history_observed_until":1.1}
    first=replace(obs,entities=(Entity("player",.4,.2,.04,.04,.95,1,samples=5),*road),metrics=first_metrics)
    guard=MixedDeadlineGuard(params);guard.attach(first,now=1.1)
    mixed=core.decide_mining_reactive(first,params)
    assert mixed.diagnostics["mode"]=="accelerated_then_normal"
    state,deadline,_=guard.apply(first,mixed,now=1.11)
    assert Button.ZR in state.buttons and deadline==pytest.approx(1.41)

    def snapshot(timestamp,sequence,released=False):
        history=[{"after":.5,"axis":0.,"zr":False},{"after":1.15,"axis":0.,"zr":True}]
        if released:history.append({"after":1.53,"axis":0.,"zr":False})
        return replace(obs,timestamp=timestamp,sequence=sequence,metrics={**first_metrics,
            "mining_input_history":history,"mining_input_history_observed_until":timestamp+.1})

    braking=snapshot(1.42,2);guard.attach(braking,now=1.52)
    normal=core.decide_mining_reactive(braking,params)
    assert normal.diagnostics["mixed_guard"]["normal_replanned"]
    state,deadline,_=guard.apply(braking,normal,now=1.53)
    assert Button.ZR not in state.buttons and deadline is None

    # A successful normal report exists, but its after+L is beyond capture.
    pending=snapshot(1.60,3,released=True);guard.attach(pending,now=1.70)
    assert not pending.metrics["mining_mixed_guard"]["allow_full_fast_rearm"]
    waiting=core.decide_mining_reactive(pending,params)
    assert Button.ZR not in waiting.state.buttons
    guard.apply(pending,waiting,now=1.71)

    confirmed=snapshot(1.72,4,released=True);guard.attach(confirmed,now=1.82)
    assert confirmed.metrics["mining_mixed_guard"]["allow_full_fast_rearm"]
    fast=core.decide_mining_reactive(confirmed,params)
    assert fast.diagnostics["mixed_guard"]["full_fast_rearm_verified"]
    state,deadline,execution=guard.apply(confirmed,fast,now=1.83)
    assert Button.ZR in state.buttons and deadline is None
    assert execution["phase_after_execution"]=="eligible"
