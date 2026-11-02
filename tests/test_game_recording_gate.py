"""Recording follows confirmed game scenes, without menu-frame disk growth."""
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from raiders_helper.capture.models import FramePacket
from raiders_helper.experiment_recording import AsyncExperimentRecorder


def packet(sequence, timestamp=None):
    return FramePacket(np.full((48,64,3),sequence,np.uint8),
                       10+sequence/60 if timestamp is None else timestamp,sequence,60)


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_menu_and_initial_unknown_never_open_video_or_per_frame_files(tmp_path):
    recorder=AsyncExperimentRecorder(tmp_path,initially_active=False)
    for index,screen in enumerate(["unknown","menu:mining","menu:hopping","result:retry"]):
        assert recorder.observe_phase(screen,"unknown",timestamp=index) is False
        assert recorder.on_frame(packet(index+1)) is False
    time.sleep(.03)
    assert not recorder.video_path.exists() and not recorder.frames_path.exists()
    assert not recorder.events_path.exists()
    summary=recorder.close()
    assert summary["ignored_inactive_frames"]==4
    assert summary["recording_starts"]==0 and summary["written_frames"]==0
    assert summary["recording_active"] is False and not summary["video_created"]


def test_countdown_play_unknown_result_and_retry_preserve_mapping_without_menu_frames(tmp_path):
    recorder=AsyncExperimentRecorder(tmp_path,initially_active=False)
    assert not recorder.on_frame(packet(1,10))
    assert recorder.observe_phase("countdown:mining",timestamp=11)
    assert recorder.on_frame(packet(2,11.01))
    assert recorder.observe_phase("playing:mining",timestamp=11.1)
    assert recorder.on_frame(packet(3,11.11))
    # A flashing player/temporary unknown is useful diagnostic evidence.
    assert recorder.observe_phase("unknown","unknown",timestamp=11.2)
    assert recorder.on_frame(packet(4,11.21))
    # A known result wins even when the gameplay HUD detector still says play.
    assert recorder.observe_phase("result:retry","playing",timestamp=12) is False
    assert not recorder.on_frame(packet(5,12.01))
    assert recorder.observe_phase("menu:mining",timestamp=20) is False
    assert not recorder.on_frame(packet(6,20.01))
    assert recorder.observe_phase("countdown:mining",timestamp=30)
    assert recorder.on_frame(packet(7,30.01))
    assert recorder.observe_phase("result:quit",timestamp=31) is False
    assert not recorder.on_frame(packet(8,31.01))
    summary=recorder.close()
    assert summary["ignored_inactive_frames"]==4
    assert summary["written_frames"]==4 and summary["recording_starts"]==2 and summary["recording_stops"]==2
    events=lines(recorder.frames_path)
    assert [row["capture_sequence"] for row in events]==[2,3,4,7]
    assert [row["video_frame_index"] for row in events]==[0,1,2,3]
    assert [row["recording_segment"] for row in events]==[1,1,1,2]
    assert events[3]["capture_timestamp"]-events[2]["capture_timestamp"]>18
    gates=lines(recorder.events_path)
    assert [row["active"] for row in gates]==[True,False,True,False]
    assert [row["reason"] for row in gates]==["countdown:mining","result:retry","countdown:mining","result:quit"]
    capture=cv2.VideoCapture(str(recorder.video_path))
    try:
        decoded=0
        while capture.read()[0]:decoded+=1
    finally:capture.release()
    assert decoded==4


def test_gate_transition_does_not_wait_for_a_blocked_encoder(tmp_path,monkeypatch):
    entered,release=threading.Event(),threading.Event()
    class Writer:
        def write(self,frame):
            entered.set();assert release.wait(1)
        def release(self):pass
    monkeypatch.setattr(AsyncExperimentRecorder,"_create_writer",lambda self,size:Writer())
    recorder=AsyncExperimentRecorder(tmp_path,initially_active=False)
    try:
        recorder.set_active(True,"test-start")
        assert recorder.on_frame(packet(1)) and entered.wait(1)
        done=threading.Event()
        thread=threading.Thread(target=lambda:(recorder.set_active(False,"test-stop"),done.set()))
        thread.start()
        assert done.wait(.2)
        assert not recorder.on_frame(packet(2))
        thread.join()
    finally:
        release.set();summary=recorder.close()
    assert summary["written_frames"]==1 and summary["ignored_inactive_frames"]==1


def test_confirmed_scene_overrides_gameplay_phase_and_explicit_pause_stops(tmp_path):
    recorder=AsyncExperimentRecorder(tmp_path,initially_active=False)
    assert recorder.observe_phase("countdown:landing","overlay")
    assert not recorder.observe_phase("result:retry","playing")
    assert not recorder.observe_phase("menu:mining","playing")
    assert not recorder.observe_phase("unknown","playing")
    assert recorder.observe_phase("countdown:mining","unknown")
    assert not recorder.observe_phase("unknown","paused")
    assert not recorder.observe_phase("unknown","unknown")
    assert recorder.observe_phase("playing:mining","unknown")
    assert not recorder.observe_phase("unknown","overlay")
    recorder.close()


def test_first_frame_game_end_stops_and_residual_playing_cannot_reopen(tmp_path):
    recorder=AsyncExperimentRecorder(tmp_path,initially_active=False)
    assert recorder.observe_phase("playing:mining",timestamp=1)
    assert recorder.on_frame(packet(1,1.01))
    # TIME=0 may precede the visible result box. The still-present HUD/player
    # must not reopen recording in this same observation or subsequent flicker.
    assert not recorder.observe_phase("unknown","playing",timestamp=2,game_ended=True)
    assert not recorder.on_frame(packet(2,2.01))
    assert not recorder.observe_phase("unknown","unknown",timestamp=2.1)
    assert not recorder.observe_phase("playing:mining","playing",timestamp=2.2)
    assert not recorder.on_frame(packet(3,2.21))
    assert recorder.observe_phase("countdown:mining","unknown",timestamp=3)
    assert recorder.on_frame(packet(4,3.01))
    recorder.close()
    events=lines(recorder.frames_path)
    assert [row["capture_sequence"] for row in events]==[1,4]
    gates=lines(recorder.events_path)
    assert [(row["active"],row["reason"]) for row in gates[:3]]==[
        (True,"playing:mining"),(False,"game_ended"),(True,"countdown:mining")]


def test_explicit_new_round_can_reopen_after_end_but_unknown_cannot(tmp_path):
    recorder=AsyncExperimentRecorder(tmp_path,initially_active=False)
    assert recorder.observe_phase("playing:mining",timestamp=1)
    assert not recorder.observe_phase("result:retry","playing",timestamp=2)
    assert not recorder.observe_phase("unknown","unknown",timestamp=3,new_round_confirmed=True)
    assert not recorder.observe_phase("playing:mining","playing",timestamp=3.1)
    assert recorder.observe_phase("playing:mining","playing",timestamp=4,new_round_confirmed=True)
    recorder.close()
