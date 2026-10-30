from dataclasses import replace
import threading
import time

import numpy as np
import pytest

from raiders_helper.analysis_worker import AnalysisWorker
from raiders_helper.capture.models import FramePacket
from raiders_helper.control.buttons import ControllerState, StickVector
from raiders_helper.control.dummy import DummyBackend
from raiders_helper.control.guard import GuardedController
from raiders_helper.fast_tracking import PredictiveTracker, assignment
from raiders_helper.models import Decision, Entity, Game, Observation
from raiders_helper.runtime import Engine
from raiders_helper.vision import create_recognizer, load_profile


def ball(x, y=0.3):
    return Entity("ball",x,y,0.04,0.04,0.9)


def test_assignment_finds_global_solution_instead_of_greedy_pairing():
    assert sorted(assignment([[1,2,9],[1.1,9,9]])) == [(0,1),(1,0)]


def test_predictive_tracker_keeps_id_through_crossing():
    tracker=PredictiveTracker()
    first=tracker.update([ball(0.4),ball(0.6)],0)
    tracker.update([ball(0.45),ball(0.55)],0.05)
    crossed=tracker.update([ball(0.51),ball(0.49)],0.11)
    assert crossed[0].track_id == first[0].track_id
    assert crossed[1].track_id == first[1].track_id


def test_missing_tracks_are_not_published_as_real_detections():
    tracker=PredictiveTracker()
    first=tracker.update([ball(0.5)],0)[0]
    assert tracker.update([],0.02) == ()
    returned=tracker.update([ball(0.51)],0.04)[0]
    assert returned.track_id == first.track_id
    assert tracker.update([],0.5) == ()
    assert tracker.update([ball(0.5)],0.52)[0].track_id != first.track_id


def test_bounce_reverses_velocity_without_long_history_lag():
    tracker=PredictiveTracker()
    tracker.update([ball(0.5,0.4)],0)
    assert tracker.update([ball(0.5,0.45)],0.05)[0].vy > 0
    assert tracker.update([ball(0.5,0.43)],0.1)[0].vy < 0


def test_horizontal_bounce_does_not_keep_opposite_history_velocity():
    tracker=PredictiveTracker()
    for i,x in enumerate((0.91,0.94,0.97)):
        tracker.update([ball(x)],i*0.02)
    assert tracker.update([ball(0.94)],0.06)[0].vx < 0


def test_submission_age_includes_time_waiting_for_control_lock():
    class Recognizer:
        profile={"calibrated":True}
        def observe(self,packet):
            return Observation(Game.MINING,packet.timestamp,packet.sequence)
    class Controller:
        def __init__(self):self.states=[]
        def submit(self,state):self.states.append(state)
        def release(self):pass
    controller=Controller();engine=Engine(Recognizer(),controller,live=True,max_age=0.03)
    engine.arm();packet=FramePacket(None,time.perf_counter(),1,60)
    result=[]
    with engine._control_lock:
        thread=threading.Thread(target=lambda:result.append(engine.process(packet)))
        thread.start();time.sleep(0.05)
    thread.join(timeout=1)
    assert controller.states == [ControllerState.neutral()]
    assert result[0][1].diagnostics['timing']['frame_age_ms'] >= 30


@pytest.mark.parametrize("game",["hopping","juggling","landing"])
def test_reference_recognizer_blank_frame_cannot_trigger_advice(game):
    profile=load_profile(f"profiles/{game}.json",game)
    recognizer=create_recognizer(game,profile)
    for i in range(3):
        obs=recognizer.observe(FramePacket(np.zeros((1080,1920,3),np.uint8),i/60,i+1,60))
        assert obs.phase == "unknown" and obs.confidence == 0


def test_advisory_profile_cannot_be_armed_even_with_other_flags_set():
    class Recognizer:
        profile={"calibrated":True,"advisory_only":True}
    engine=Engine(Recognizer(),object(),live=True)
    with pytest.raises(ValueError):
        engine.arm()


def test_disarm_during_inflight_recognition_prevents_late_input():
    entered,finish=threading.Event(),threading.Event()
    class Recognizer:
        profile={"calibrated":True}
        def observe(self,packet):
            entered.set(); assert finish.wait(1)
            return Observation(Game.MINING,packet.timestamp,packet.sequence)
    backend=DummyBackend();backend.connect();guard=GuardedController(backend)
    engine=Engine(Recognizer(),guard,live=True)
    engine.arm()
    packet=FramePacket(None,time.perf_counter(),1,60)
    thread=threading.Thread(target=lambda:engine.process(packet))
    thread.start();assert entered.wait(1)
    engine.disarm();before=len(backend.history);finish.set();thread.join(timeout=1)
    try:
        assert not thread.is_alive()
        assert len(backend.history) == before
        assert backend.state == ControllerState.neutral()
    finally:
        guard.close()


def test_latest_frame_worker_drops_backlog_instead_of_replaying_old_frames():
    entered,finish=threading.Event(),threading.Event()
    current=[FramePacket(None,1,1,60)]
    class EngineStub:
        live=False
        def __init__(self):self.sequences=[]
        def process(self,packet):
            self.sequences.append(packet.sequence)
            if packet.sequence==1:
                entered.set(); assert finish.wait(1)
            return None,None
        def disarm(self):pass
    engine=EngineStub();worker=AnalysisWorker(engine,lambda:current[0]);worker.start()
    assert entered.wait(1)
    current[0]=FramePacket(None,2,10,60);finish.set()
    deadline=time.perf_counter()+1
    while worker.latest() is None or worker.latest()[0].sequence!=10:
        assert time.perf_counter()<deadline
        time.sleep(0.005)
    worker.stop()
    assert engine.sequences == [1,10]
    assert worker.dropped_frames == 8


def _synthetic_ready_recognizer(stable_frames=None):
    """A synthetic HUD anchor and unambiguous sprites isolate phase gating."""
    import cv2
    from raiders_helper.arcade_vision import FastArcadeRecognizer
    profile={"viewport":[0,0,1,1]}
    if stable_frames is not None:
        profile["stable_frames"]=stable_frames
    recognizer=FastArcadeRecognizer(Game.HOPPING,profile)
    anchor=np.random.default_rng(11).integers(0,256,(8,24),dtype=np.uint8)
    recognizer.score_anchor=anchor
    frame=np.zeros((480,640,3),np.uint8)
    frame[5:13,400:424]=cv2.cvtColor(anchor,cv2.COLOR_GRAY2BGR)
    cv2.rectangle(frame,(100,385),(160,439),(0,255,0),-1)
    cv2.rectangle(frame,(111,398),(150,425),(255,0,0),-1)
    cv2.rectangle(frame,(290,100),(315,140),(0,255,0),-1)
    cv2.rectangle(frame,(298,108),(306,118),(255,0,0),-1)
    return recognizer,frame


def test_fast_recognizer_honors_configured_stable_frame_count_and_resets():
    recognizer,frame=_synthetic_ready_recognizer(3)
    phases=[recognizer.observe(FramePacket(frame,i/60,i+1,60)).phase for i in range(3)]
    assert phases == ["unknown","unknown","playing"]
    assert recognizer.observe(FramePacket(np.zeros_like(frame),3/60,4,60)).phase == "unknown"
    phases=[recognizer.observe(FramePacket(frame,i/60,i+1,60)).phase for i in range(4,7)]
    assert phases == ["unknown","unknown","playing"]


def test_fast_recognizer_retains_two_frame_default_when_config_is_absent():
    recognizer,frame=_synthetic_ready_recognizer()
    assert recognizer.observe(FramePacket(frame,0,1,60)).phase == "unknown"
    assert recognizer.observe(FramePacket(frame,1/60,2,60)).phase == "playing"


@pytest.mark.parametrize("value",[0,-1,2.5,True,"3"])
def test_fast_recognizer_rejects_invalid_stable_frame_counts(value):
    from raiders_helper.arcade_vision import FastArcadeRecognizer
    with pytest.raises(ValueError,match="stable_frames"):
        FastArcadeRecognizer(Game.HOPPING,{"stable_frames":value})


def test_object_template_candidate_outside_playfield_is_rejected_without_crashing():
    from raiders_helper.arcade_vision import FastArcadeRecognizer
    recognizer=FastArcadeRecognizer(Game.HOPPING,{})
    recognizer.object_templates["golden_enemy"]=(np.zeros((20,20),np.uint8),)
    area=np.zeros((480,640,3),np.uint8)
    assert recognizer.match_object(area,"golden_enemy",(620,100,10,10),24) is None
