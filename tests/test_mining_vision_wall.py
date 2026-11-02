"""Vision wiring: original floor measurement, priority and reset semantics."""
from pathlib import Path
import cv2,numpy as np,pytest
from raiders_helper.capture.models import FramePacket
from raiders_helper.models import Entity
from raiders_helper.mining_vision import MiningLiveRecognizer

ROOT=Path(__file__).resolve().parents[1]

def scene():
    frame=np.zeros((480,640,3),np.uint8)
    cv2.rectangle(frame,(100,35),(539,479),(30,80,140),-1)
    anchor=cv2.imdecode(np.frombuffer((ROOT/'assets/arcade/score-anchor.png').read_bytes(),np.uint8),1)
    frame[2:2+anchor.shape[0],390:390+anchor.shape[1]]=anchor
    cv2.rectangle(frame,(303,76),(336,108),(30,220,40),-1)
    cv2.rectangle(frame,(310,87),(329,97),(210,45,20),-1)
    return frame

class RecordingEstimator:
    def __init__(self):self.calls=[]
    def observe(self,floor,timestamp,palette,phase='playing',blocked=False,measure=True):
        self.calls.append(dict(floor=floor.copy(),timestamp=timestamp,palette=palette,
                               phase=phase,blocked=blocked,measure=measure))
        if blocked or phase!='playing':return {'wall_scroll':{'accepted':False,'reason':'blocked'}}
        return {'wall_scroll':{'accepted':True},'current_forward_speed_observed':True,
                'current_forward_speed':.42,'current_forward_speed_confidence':.8,
                'current_forward_speed_source':'bilateral_wall_translation',
                'current_forward_speed_method':'wall_profile_pair',
                'current_forward_speed_reference_timestamp':timestamp-.05,
                'current_forward_speed_window_seconds':.1}

def recognizer(**values):
    return MiningLiveRecognizer(profile={'viewport':[0,0,1,1],'phase_anchors':False,'stable_frames':1,**values})

def test_wall_estimator_is_opt_in_and_receives_raw_not_filled_floor():
    disabled=recognizer();assert disabled._wall_scroll is None
    r=recognizer(estimate_wall_scroll=True);fake=RecordingEstimator();r._wall_scroll=fake
    o=r.observe(FramePacket(scene(),1.,1,60));call,=fake.calls
    expected=r.road_mask(cv2.cvtColor(r.last_canonical,cv2.COLOR_BGR2HSV))
    assert np.array_equal(call['floor'],expected)
    assert np.any(call['floor']!=r.last_road_mask)
    assert call['measure'] and not call['blocked'] and call['phase']=='playing'
    assert o.metrics['current_forward_speed_source']=='bilateral_wall_translation'
    assert o.metrics['current_forward_speed_timestamp']==1.
    assert o.metrics['current_forward_speed_reference_timestamp']==.95

def test_current_landmarks_keep_priority_while_wall_history_still_updates(monkeypatch):
    r=recognizer(estimate_wall_scroll=True,estimate_landmark_scroll=True)
    fake=RecordingEstimator();r._wall_scroll=fake
    monkeypatch.setattr(r,'_landmark_scroll',lambda *a,**k:{'current_forward_speed_observed':True,
        'current_forward_speed_confidence':.85,'current_forward_speed':.30,
        'current_forward_speed_source':'tracked_landmark_consensus'})
    o=r.observe(FramePacket(scene(),1.,1,60))
    assert not fake.calls[0]['measure']
    assert o.metrics['current_forward_speed']==.30
    assert o.metrics['current_forward_speed_source']=='tracked_landmark_consensus'
    assert 'wall_scroll' in o.metrics

@pytest.mark.parametrize('blocked_kind',['hit','powered'])
def test_collision_and_powered_frames_explicitly_reset_wall_measurement(monkeypatch,blocked_kind):
    r=recognizer(estimate_wall_scroll=True,floor_backend='adaptive_inner')
    fake=RecordingEstimator();r._wall_scroll=fake
    if blocked_kind=='hit':
        r.last_player_time=.9
        monkeypatch.setattr(r,'_player',lambda *a:(None,0,False))
        monkeypatch.setattr(r,'_hit_player',lambda *a:(Entity('player_hit',.5,.192,.05,.07,.8),1,False))
    else:
        monkeypatch.setattr(r,'_powered_player',lambda *a:(Entity('player',.5,.192,.05,.07,.8),1,False))
    o=r.observe(FramePacket(scene(),1.,1,60))
    assert fake.calls[0]['blocked']
    assert not o.metrics['current_forward_speed_observed']
