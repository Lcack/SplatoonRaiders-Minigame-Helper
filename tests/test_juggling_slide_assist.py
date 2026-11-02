import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from juggling_slide_assist import distance_at,integrate_with_slides,SlideAssist
from raiders_helper.control.buttons import Button,ControllerState
from raiders_helper.control.buttons import StickVector
from dataclasses import replace
from raiders_helper.models import Entity,Game,Observation,Decision

MODEL={'distance':.2966,'duration_seconds':.3782}


def report(t,x=0,buttons=()):return {'after':t,'state':{'buttons':list(buttons),'left_stick':{'x':x,'y':0},'right_stick':{'x':0,'y':0}}}


def test_release_does_not_remove_measured_slide_coast():
    items=[report(9),report(10,13107,['ZL']),report(10.085)]
    x=integrate_with_slides(items,'left',10.15,10.4,.666,MODEL)
    assert x==pytest.approx(MODEL['distance']-distance_at(.15,MODEL))
    assert x>0


def test_cannot_model_unvalidated_fast_retrigger():
    items=[report(9),report(10,13107,['ZL']),report(10.08),report(10.2,13107,['ZL'])]
    assert integrate_with_slides(items,'left',10.15,10.4,.666,MODEL) is None


def test_actual_fourth_direction_stays_inside_declared_model_error():
    path=Path(__file__).resolve().parents[1]/'artifacts/juggling-live/slide-motion-02.json'
    if not path.exists():pytest.skip('local actual pulse evidence not bundled')
    case=json.loads(path.read_text(encoding='utf8'))[3];points=case['points']
    x0=case['observed_start_x_approx'];errors=[abs((x0-distance_at(p['t_after_submit']-.106708,MODEL))-p['x'])*640 for p in points]
    assert max(errors)<16


def test_no_autonomous_slide_without_opt_in():
    obs=Observation(Game.JUGGLING,10,1)
    d=Decision()
    assert SlideAssist({}).apply(obs,d,10)==(d,None,None)


def rescue_scene():
    obs=Observation(Game.JUGGLING,10,1,'playing',.9,(
        Entity('player',.1,.9,.1,.1,.9),Entity('partner',.75,.9,.05,.05,.9),
        Entity('ball',.32,.6,.06,.06,.9,track_id=3,vy=.8,samples=10)),metrics={
            'juggling_actor_input_projection':{'accepted':True,'positions':{'left':.1,'right':.75},'held_input_extension_seconds':.02}})
    target={'track_id':3,'target_x':.32,'available_seconds':.18,'time_to_contact':.35}
    decision=Decision(ControllerState(left_stick=StickVector(32767,0)),diagnostics={
        'targets':{'left':target,'right':None},'intercepts':[],'latency_seconds':.17})
    return obs,decision


def test_rescue_has_finite_pulse_then_locks_role_until_coast_ends():
    obs,d=rescue_scene();assist=SlideAssist({'enabled':True,'model':MODEL})
    started,deadline,released=assist.apply(obs,d,10)
    assert Button.ZL in started.state.buttons
    assert deadline==pytest.approx(10.085)
    assert released.left_stick.x==0 and Button.ZL not in released.buttons
    locked,deadline,_=assist.apply(replace(obs,timestamp=10.1),d,10.1)
    assert locked.state.left_stick.x==0 and Button.ZL not in locked.state.buttons
    assert deadline is None


def test_another_early_ball_rejects_slide_that_would_leave_it_behind():
    obs,d=rescue_scene();d=replace(d,diagnostics={**d.diagnostics,'intercepts':[
        {'track_id':4,'eligible':True,'side':'left','target_x':.1,'time_to_contact':.4}]})
    result,_,_=SlideAssist({'enabled':True,'model':MODEL}).apply(obs,d,10)
    assert not result.state.buttons


def test_unknown_frame_cancels_button_and_cannot_restart_same_pulse():
    obs,d=rescue_scene();assist=SlideAssist({'enabled':True,'model':MODEL});assist.apply(obs,d,10)
    result,_,_=assist.apply(replace(obs,phase='unknown',confidence=0),Decision(),10.03)
    assert not result.state.buttons
    result,_,_=assist.apply(obs,d,10.06)
    assert not result.state.buttons


def test_fixed_delay_model_has_actual_fourth_direction_error_below_eight_pixels():
    root=Path(__file__).resolve().parents[1]/'artifacts/juggling-live'
    if not (root/'slide-motion-02.json').exists():pytest.skip('actual local pulse evidence not bundled')
    case=json.loads((root/'slide-motion-02.json').read_text(encoding='utf8'))[3]
    model=json.loads((root/'slide-impulse-fixed-delay-02.json').read_text(encoding='utf8'))
    x0=case['observed_start_x_approx']
    errors=[abs(x0-distance_at(p['t_after_submit']-.12,model)-p['x'])*640 for p in case['points']]
    assert max(errors)<8
