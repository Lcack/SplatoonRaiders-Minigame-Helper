from dataclasses import replace
import math

import pytest

from raiders_helper.juggling_prediction import BallTrajectoryObserver, fit_window
from raiders_helper.models import Entity


def points():
    return [(10+t, .25+.2*t, .4-.3*t+.5*.9*t*t, .07, .06)
            for t in [i*.04 for i in range(12)]]


def test_measured_parabola_and_temporal_holdout():
    model=fit_window(points())
    assert model['accepted']
    assert model['gravity']==pytest.approx(.9)
    assert model['vx']==pytest.approx(.2)
    assert model['vy']==pytest.approx(-.3+.9*.44)
    assert model['last_two_prediction_error_pixels']==pytest.approx([0,0],abs=1e-8)


@pytest.mark.parametrize('alteration', ['gap','size','nonfinite','abrupt_motion'])
def test_rejects_gaps_shape_changes_invalid_or_changed_motion(alteration):
    history=points()
    if alteration=='gap': history[-1]=(history[-1][0]+.2,*history[-1][1:])
    if alteration=='size': history[-1]=(*history[-1][:3],.2,.06)
    if alteration=='nonfinite': history[-1]=(history[-1][0],math.nan,*history[-1][2:])
    if alteration=='abrupt_motion': history[-1]=(history[-1][0],history[-1][1]+.05,*history[-1][2:])
    assert not fit_window(history)['accepted']


def test_only_current_observed_ball_published_and_reset_on_phase_gap():
    observer=BallTrajectoryObserver()
    for t,x,y,w,h in points():
        e=Entity('ball',x,y,w,h,.9,track_id=3)
        models=observer.observe([e],t,'playing')
    assert models['3']['accepted']
    assert observer.observe([],10.48,'playing')=={}
    observer.observe([e],11,'unknown')
    assert not observer.observe([e],11.04,'playing')['3']['accepted']


def test_actual_downward_to_upward_impulse_resets_history():
    observer=BallTrajectoryObserver()
    for i in range(12):
        e=Entity('ball',.25,.3+.015*i,.07,.06,.9,track_id=3)
        observer.observe([e],10+.04*i,'playing')
    models=observer.observe([replace(e,y=e.y-.04)],10.48,'playing')
    assert not models['3']['accepted']
    assert len(observer.histories[3])==1
