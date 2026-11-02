import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from juggling_slide_probe import SlideProbe
from raiders_helper.control.buttons import Button,ControllerState
from raiders_helper.models import Entity,Game,Observation


def scene(phase='playing',x=.12):return Observation(Game.JUGGLING,1,1,phase,.9,(
    Entity('player',x,.9,.1,.1,.9),Entity('partner',.75,.9,.05,.05,.9)),metrics={'observation_age_seconds':.04})


def test_pulse_only_after_observed_middle_and_settle_then_releases():
    p=SlideProbe({})
    for t in (0,.04,.08):assert p.decide(scene(),t).state==ControllerState.neutral()
    assert not p.decide(scene(),.50).state.buttons
    assert Button.ZL in p.decide(scene(),.54).state.buttons
    assert p.decide(scene(),.66).state==ControllerState.neutral()


def test_unknown_frame_and_edge_cancel_pulse():
    p=SlideProbe({});p.stage='pulse';p.since=1
    assert p.decide(scene('unknown'),1.02).state==ControllerState.neutral()
    p.stage='pulse';p.since=2
    assert p.decide(scene(x=.44),2.02).state==ControllerState.neutral()
