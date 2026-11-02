import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from juggling_control_model import SubmittedInputObserver,project_actors,IdleCenterGate
from dataclasses import replace
from raiders_helper.juggling import decide_juggling
from raiders_helper.models import Entity,Game,Observation


def scene():
    return Observation(Game.JUGGLING,10,1,'playing',.9,(
        Entity('player',.25,.9,.1,.1,.9),Entity('partner',.75,.9,.05,.05,.9)))


def item(t,left=0,right=0):
    return {'after':t,'state':{'buttons':[],'left_stick':{'x':left,'y':0},'right_stick':{'x':right,'y':0}}}


def test_reversal_uses_pending_actual_inputs_not_smoothed_actor_velocity():
    m=project_actors(scene(),[item(9.8,32767,-32767),item(9.98,-32767,32767)],10.04,10.04,{})
    assert m['accepted']
    assert m['positions']['left']==pytest.approx(.25+.666*(.1-.075))
    assert m['positions']['right']==pytest.approx(.75-.66*(.1-.075))
    assert m['held_input_extension_seconds']==pytest.approx(.015)
    assert m['seconds_from_observation']==pytest.approx(.175)


def test_stale_incomplete_or_slide_history_falls_back():
    assert not project_actors(scene(),[item(9.8)],9,10.04,{})['accepted']
    assert not project_actors(scene(),[item(9.9)],10.04,10.04,{})['accepted']
    slide=item(9.8);slide['state']['buttons']=['ZL']
    assert not project_actors(scene(),[slide],10.04,10.04,{})['accepted']


def test_cursor_gap_discards_unprovable_older_inputs():
    class Client:
        def request(self,path):return {'events':[{'id':5,'timestamp':10,'state':item(10)['state']}]}
    o=SubmittedInputObserver(Client(),{});o.cursor=2;o.items.append(item(9))
    o.collect()
    assert len(o.items)==1 and o.items[0]['after']==10


def test_middle_return_waits_for_continuous_idle_and_catch_interrupts_it():
    gate=IdleCenterGate(.3);p={'idle_centering':{'enabled':True}}
    base=scene();entities=(replace(base.entities[0],x=.1),base.entities[1]);base=replace(base,entities=entities)
    first=gate.apply(base,decide_juggling(base,p));assert first.state.left_stick.x==0
    later=replace(base,timestamp=10.2);assert gate.apply(later,decide_juggling(later,p)).state.left_stick.x==0
    later=replace(base,timestamp=10.35);assert gate.apply(later,decide_juggling(later,p)).state.left_stick.x>0
    ball=Entity('ball',.08,.65,.05,.05,.9,track_id=2,vy=.5,samples=3)
    catch=replace(base,timestamp=10.4,entities=(*entities,ball))
    assert gate.apply(catch,decide_juggling(catch,p)).state.left_stick.x<0
    later=replace(base,timestamp=10.45);assert gate.apply(later,decide_juggling(later,p)).state.left_stick.x==0


def test_current_actual_slide_prefix_includes_motion_after_button_release():
    from juggling_slide_assist import distance_at
    model={'distance':.296409,'duration_seconds':.353148}
    obs=replace(scene(),timestamp=10.2,entities=(replace(scene().entities[0],x=.1+distance_at(.08,model)),scene().entities[1]))
    start=item(10,13107);start['state']['buttons']=['ZL']
    inputs=[item(9),start,item(10.085)]
    predicted=project_actors(obs,inputs,10.24,10.24,{'slide_model':model})
    assert predicted['accepted']
    assert predicted['positions']['left']==pytest.approx(.1+distance_at(.255,model))
    assert not project_actors(obs,inputs,10.24,10.24,{})['accepted']
