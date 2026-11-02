"""Prospective budgets and separate A/P/D/L; no live inputs."""
from dataclasses import replace
import math
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from mining_supervisor import PlanningDelayBudget
import raiders_helper.mining_reactive as policy
from raiders_helper.models import Entity,Game,Observation
from raiders_helper.control.buttons import Button,ControllerState


def observation(metrics=None):
    return Observation(Game.MINING,1,1,'playing',.95,
        (Entity('player',.5,.2,.04,.04,.95,samples=5),
         *(Entity('corridor',.5,y,.94,.001,.95) for y in (0,.15,.2,.35,.5,.7,1))),metrics=metrics or {})


def test_budget_uses_only_previous_timings_and_reports_not_a_measurement():
    budget=PlanningDelayBudget(enabled=True)
    first=observation();budget.attach(first)
    assert first.metrics['planning_delay_seconds']==.05
    for elapsed in [.01,.02,.03,.04,.05,.06,.07,.08]:budget.record(elapsed,'playing')
    second=observation();budget.attach(second)
    assert second.metrics['planning_delay_seconds']==pytest.approx(.0765)
    budget.record(.10,'playing')
    assert second.metrics['planning_delay_seconds']==pytest.approx(.0765)
    third=observation();budget.attach(third)
    assert third.metrics['planning_delay_seconds']==.08
    assert third.metrics['planning_delay_is_budget_not_current_measurement'] is True


def test_non_playing_calls_do_not_collapse_budget_and_window_is_bounded():
    budget=PlanningDelayBudget(enabled=True,window=8)
    for _ in range(20):budget.record(0.,'countdown')
    assert len(budget.samples)==0
    for _ in range(20):budget.record(.001,'playing')
    obs=observation();budget.attach(obs)
    assert len(budget.samples)==8 and obs.metrics['planning_delay_seconds']==.03


def test_disabled_budget_does_not_override_existing_metadata():
    obs=observation({'planning_delay_seconds':.02})
    budget=PlanningDelayBudget()
    budget.attach(obs);budget.record(.10,'playing')
    assert obs.metrics=={'planning_delay_seconds':.02} and not budget.samples


@pytest.mark.parametrize('change',[{'initial':float('nan')},{'minimum':-.1},{'maximum':.2},
    {'window':2},{'quantile':2},{'enabled':'true'},{'minimum':.1}])
def test_invalid_budget_is_rejected_before_round_start(change):
    with pytest.raises(ValueError):PlanningDelayBudget(**change)


def test_planning_budget_adds_prefix_time_without_changing_dispatch_or_effective_delay():
    params={'prediction_timing':'capture_clock','effective_delay_seconds':.125,'dispatch_delay_seconds':.03,
            'current_forward_speed_hypothesis':.25,'lookahead_depth':.55,'maximum_steps':30}
    old=policy.decide_mining_reactive(observation({'observation_age_seconds':.08}),params)
    new=policy.decide_mining_reactive(observation({'observation_age_seconds':.08,
        'planning_delay_seconds':.05,'planning_delay_source':'test_previous_samples'}),params)
    a,b=old.diagnostics['timing_compensation'],new.diagnostics['timing_compensation']
    assert a['prefix_seconds']==pytest.approx(.235)
    assert b['prefix_seconds']==pytest.approx(.285)
    assert b['dispatch_delay_seconds']==a['dispatch_delay_seconds']==.03
    assert b['effective_delay_seconds']==a['effective_delay_seconds']==.125
    assert b['planning_delay_seconds']==.05 and b['planning_delay_source']=='test_previous_samples'
    assert b['new_action_can_affect_capture_state_at']['depth']-a['new_action_can_affect_capture_state_at']['depth']==pytest.approx(.0125)


@pytest.mark.parametrize('bad',[-.1,.151,float('nan'),True])
def test_invalid_planning_delay_fails_closed(bad):
    result=policy.decide_mining_reactive(observation({'planning_delay_seconds':bad}),{'prediction_timing':'capture_clock'})
    assert result.state==ControllerState.neutral() and 'planning_delay_seconds' in result.reason


def test_planning_budget_does_not_admit_input_reports_from_the_future():
    metrics={'observation_age_seconds':.08,'planning_delay_seconds':.05,
        'mining_input_history_source':'commands.jsonl.after','mining_input_history':[{'after':1.10,'axis':0.,'zr':False}]}
    result=policy.decide_mining_reactive(observation(metrics),{'prediction_timing':'capture_clock'})
    assert result.state==ControllerState.neutral() and '历史时钟' in result.reason


def test_unprofitable_upper_bound_skips_normal_search_without_changing_control(monkeypatch):
    obs=observation();obs=replace(obs,entities=(*obs.entities,Entity('time_star',.5,.65,.02,.02,.95,track_id=2)))
    params={'compare_star_modes':True,'star_reward':.04,'lookahead_depth':.68,'maximum_steps':30}
    baseline=policy.decide_mining_reactive(obs,params)
    original=policy._search;calls=[]
    def spy(*args,**kwargs):
        calls.append(args[3].forward)
        return original(*args,**kwargs)
    monkeypatch.setattr(policy,'_search',spy)
    result=policy.decide_mining_reactive(obs,{**params,'star_mode_upper_bound_pruning':True})
    comparison=result.diagnostics['star_mode_comparison']
    assert comparison['attempted'] is False and comparison['skipped']=='net_gain_upper_bound_insufficient'
    assert result.state==baseline.state and len(calls)==1


def test_upper_bound_reaches_same_endpoint_after_exact_capture_prefix(monkeypatch):
    obs=observation({'observation_age_seconds':.08,'planning_delay_seconds':.05})
    obs=replace(obs,entities=(*obs.entities,Entity('time_star',.5,.65,.02,.02,.95,track_id=2)))
    params={'compare_star_modes':True,'star_reward':.04,'lookahead_depth':.68,'maximum_steps':30,
        'prediction_timing':'capture_clock','effective_delay_seconds':.125,'dispatch_delay_seconds':.03,
        'current_forward_speed_hypothesis':.25,'star_mode_upper_bound_pruning':True,
        'motion_models':{'normal':{'forward_speed':.25},'zr':{'forward_speed':.5}}}
    result=policy.decide_mining_reactive(obs,params)
    d=result.diagnostics;c=d['star_mode_comparison'];p=d['timing_compensation']
    normal_time=p['prefix_seconds']+(d['path'][-1]['depth']-p['new_action_can_affect_capture_state_at']['depth'])/.25
    assert c['extra_travel_seconds_if_complete']==pytest.approx(normal_time-d['planned_seconds'])
    assert Button.ZR in result.state.buttons


def test_longer_compute_prefix_pickup_is_not_new_action_reward():
    obs=observation({'observation_age_seconds':.08,'planning_delay_seconds':.05})
    obs=replace(obs,entities=(*obs.entities,Entity('time_star',.5,.27,.02,.02,.95,track_id=2)))
    result=policy.decide_mining_reactive(obs,{'star_reward':.04,'prediction_timing':'capture_clock',
        'effective_delay_seconds':.125,'dispatch_delay_seconds':.03,'current_forward_speed_hypothesis':.25})
    assert result.diagnostics['visible_time_star_candidates']==1
    assert result.diagnostics['collected_predicted']==[]


def test_profitable_real_star_group_is_not_pruned():
    import json
    from pathlib import Path
    base=Path.cwd()/'artifacts/mining-A-campaign/batch-02/round-0001'
    rows=[json.loads(line) for line in (base/'gameplay/trace.jsonl').read_text(encoding='utf8').splitlines()]
    raw=next(r['observation'] for r in rows if r['observation']['sequence']==4313)
    obs=Observation(Game(raw['game']),raw['timestamp'],raw['sequence'],raw['phase'],raw['confidence'],
                    tuple(Entity(**e) for e in raw['entities']),metrics=raw['metrics'])
    params=json.loads((base/'parameters.json').read_text(encoding='utf8'));params['compare_star_modes']=True
    old=policy.decide_mining_reactive(obs,params)
    new=policy.decide_mining_reactive(obs,{**params,'star_mode_upper_bound_pruning':True})
    assert old.diagnostics['star_mode_comparison']['selected']=='normal'
    assert new.diagnostics['star_mode_comparison']['attempted'] is True
    assert new.diagnostics['star_mode_comparison']['selected']=='normal'
    assert old.state==new.state
