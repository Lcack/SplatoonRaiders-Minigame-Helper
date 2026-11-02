from dataclasses import replace

import cv2
import numpy as np

from raiders_helper.juggling_fever import FeverObserver
from raiders_helper.juggling import decide_juggling
from raiders_helper.models import Entity,Game,Observation


def image(fill=0.):
    hsv=np.zeros((480,640,3),np.uint8)
    hsv[10:24,10:10+round(122*fill)]=(95,100,180)
    return hsv


def diamond(hsv,x=200,y=200):
    cv2.fillConvexPoly(hsv,np.array([[x,y-8],[x+8,y],[x,y+8],[x-8,y]],np.int32),(95,230,240))


def test_gauge_full_starts_fever_and_empty_ends_it_without_treating_partial_charge_as_active():
    f=FeverObserver()
    assert not f.observe(image(.5),(),1.,'playing')[1]['fever_active']
    assert f.observe(image(1.),(),1.1,'playing')[1]['fever_active']
    assert f.observe(image(.5),(),1.5,'playing')[1]['fever_active']
    assert not f.observe(image(0.),(),2.,'playing')[1]['fever_active']
    assert not f.observe(image(1.),(),2.1,'overlay')[1]['fever_active']


def test_observed_score_star_has_its_own_identity_and_does_not_become_a_ball():
    f=FeverObserver();a=image(1.);diamond(a)
    stars,m=f.observe(a,(),1.,'playing')
    assert len(stars)==1 and stars[0].kind=='score_pickup' and stars[0].track_id>=100000
    assert m['score_pickup_is_time_bonus'] is False
    assert m['score_pickup_collection_confirmed'] is False


def test_blue_actor_and_fish_regions_hud_and_dark_crowd_do_not_become_score_stars():
    a=image(1.);diamond(a)
    actor=Entity('player',200/640,200/480,.1,.1,.9)
    assert not FeverObserver.pickups(a,(actor,))
    assert not FeverObserver.pickups(a,(replace(actor,kind='ball'),))
    a=image(1.);diamond(a,y=20);diamond(a,y=465)
    cv2.circle(a,(350,250),10,(60,200,70),-1)
    assert not FeverObserver.pickups(a,())


def test_fever_stars_do_not_override_an_urgent_catch():
    actors=(Entity('player',.25,.85,.08,.08,.95,100,samples=5),
            Entity('partner',.75,.85,.05,.05,.95,101,samples=5))
    ball=Entity('ball',.38,.68,.05,.05,.95,1,0.,.6,5)
    plain=Observation(Game.JUGGLING,1.,1,'playing',.95,(*actors,ball))
    with_stars=replace(plain,entities=(*plain.entities,Entity('score_pickup',.1,.75,.03,.03,.9,100001,0.,.5,5)),
                       metrics={'fever_active':True})
    a,b=decide_juggling(plain),decide_juggling(with_stars)
    assert a.state==b.state
    assert b.diagnostics['pickup_priority']=='catch_first'
    assert b.diagnostics['star_detours_enabled'] is False
