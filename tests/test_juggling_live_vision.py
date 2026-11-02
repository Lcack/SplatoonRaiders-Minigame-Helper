from pathlib import Path

import cv2
import numpy as np
import pytest

from raiders_helper.arcade_vision import FastArcadeRecognizer
from raiders_helper.capture.models import FramePacket
from raiders_helper.vision import load_profile

ROOT=Path(__file__).resolve().parents[1]


def actual(sequence):
    path=ROOT/f'artifacts/juggling-live/confirm-01/round-0002/frame-{sequence:08d}.jpg'
    if not path.exists():pytest.skip('local actual NS evidence not bundled')
    return cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)


def recognizer(enabled=True):
    profile=load_profile(ROOT/'profiles/juggling.json','juggling')
    profile.update(stable_frames=1,juggling_small_fish_supplement=enabled,read_live_lives=True)
    return FastArcadeRecognizer('juggling',profile)


def test_real_low_red_fish_near_receiving_plane_is_detected_before_it_is_lost():
    frame=actual(96218)
    old=recognizer(False).observe(FramePacket(frame,1.,1,60))
    new=recognizer().observe(FramePacket(frame,1.,1,60))
    assert not any(e.y>.8 and e.x>.6 for e in old.of_kind('ball'))
    assert any(e.y>.8 and .64<e.x<.7 for e in new.of_kind('ball'))
    assert any(e['red_pixels']<65 for e in new.metrics['small_fish_supplement'])


def test_real_plus30_text_is_not_a_supplementary_fish():
    obs=recognizer().observe(FramePacket(actual(96192),1.,1,60))
    assert not any(.19<e.x<.3 and .70<e.y<.79 for e in obs.of_kind('ball'))
    assert not any(e['box']==[129,353,56,15] for e in obs.metrics['small_fish_supplement'])


def test_real_falling_red_heart_fragments_are_not_fish():
    obs=recognizer().observe(FramePacket(actual(96244),1.,1,60))
    assert not any(.67<e.x<.73 and .76<e.y<.84 for e in obs.of_kind('ball'))


def test_real_initial_five_hearts_then_four_are_measured_not_assumed():
    root=ROOT/'artifacts/juggling-live/probe-01/round-0001'
    paths=[root/'frame-00016774.jpg',root/'frame-00017157.jpg']
    if not all(p.exists() for p in paths):pytest.skip('local life HUD evidence not bundled')
    r=recognizer();sequence=0
    for p,expected in zip(paths,(5,4)):
        f=cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),1)
        for i in range(3):
            sequence+=1;obs=r.observe(FramePacket(f,sequence*.05,sequence,60))
        assert obs.metrics['lives_remaining']==expected
        assert obs.metrics['lives_first_observed']==5


def test_life_reader_rejects_weak_hud_and_noncontiguous_red_noise():
    r=recognizer();hsv=np.zeros((480,640,3),np.uint8)
    hsv[8:27,250:270]=(0,255,255);hsv[8:27,290:310]=(0,255,255)
    for i in range(4):assert not r.juggling_lives(hsv,.9)['lives_observed']
    hsv[8:27,270:290]=(0,255,255)
    for i in range(4):assert not r.juggling_lives(hsv,.2)['lives_observed']


def test_actual_partner_beside_middle_wall_is_not_lost():
    path=ROOT/'artifacts/juggling-live/partner-blind-zone.jpg'
    if not path.exists():pytest.skip('local actual NS evidence not bundled')
    frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    old=recognizer();assert not old.observe(FramePacket(frame,1,1,60)).of_kind('partner')
    new=recognizer();new.profile['juggling_partner_full_half']=True
    obs=new.observe(FramePacket(frame,1,1,60))
    assert len(obs.of_kind('partner'))==1
    assert .51<obs.of_kind('partner')[0].x<.55
    assert obs.phase=='playing'
