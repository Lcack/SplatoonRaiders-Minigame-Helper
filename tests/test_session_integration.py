"""Run the real terminal orchestration with a simulated capture/transport."""
import json
from types import SimpleNamespace
import time

import numpy as np

from raiders_helper import session_runtime as runtime
from raiders_helper.capture.models import FramePacket
from raiders_helper.control.buttons import Button, ControllerState
from raiders_helper.control.dummy import DummyBackend
from raiders_helper.models import Decision, Entity, Game, Observation


def test_calibration_retries_real_flow_and_records_sent_probe_not_score_policy(tmp_path,monkeypatch):
    session=tmp_path/"session.json"
    session.write_text(json.dumps({"version":1,"calibrated":True,"screens":[1],
        "flow":{"screen_settle_seconds":.000001,"hold_seconds":.00001,"gap_seconds":.000001,"max_attempts":50},
        "calibration":{"enabled":True}}),encoding="utf-8")
    screens=(["menu:hopping"]*10+["countdown:hopping"]*5+["playing:hopping"]*15+
             ["result:retry"]*10+["countdown:hopping"]*5+["playing:hopping"]*15+["result:retry"]*10)
    class Worker:
        error=""
        def __init__(self,*args,**kwargs):self.index=0
        def start(self):pass
        def latest(self):
            frame=np.zeros((100,200,3),np.uint8);frame[0,0,0]=self.index
            self.index+=1
            assert self.index <= len(screens),"flow did not terminate"
            return FramePacket(frame,time.perf_counter(),self.index,60)
        def stop(self):pass
    class Matcher:
        has_ui_candidates=False
        def __init__(self,*args):pass
        def observe(self,frame):return screens[int(frame[0,0,0])],{}
    class Engine:
        def __init__(self,*args,**kwargs):pass
        def process(self,packet):
            phase="playing" if screens[packet.sequence-1]=="playing:hopping" else "unknown"
            obs=Observation(Game.HOPPING,packet.timestamp,packet.sequence,phase,.82,
                (Entity("player",.5,.8,.05,.05,.86,track_id=1,samples=5),),metrics={"observation_age_seconds":0})
            # A score-playing action must NEVER leak into the calibration path.
            return obs,Decision(ControllerState(frozenset({Button.B})),"ordinary AI",diagnostics={"timing":{}})
    backend=DummyBackend()
    monkeypatch.setattr(runtime,"CaptureWorker",Worker)
    monkeypatch.setattr(runtime,"ScreenMatcher",Matcher)
    monkeypatch.setattr(runtime,"Engine",Engine)
    monkeypatch.setattr(runtime,"EasyConBackend",lambda:backend)
    monkeypatch.setattr(runtime,"load_profile",lambda *args:{"calibrated":False,"advisory_only":True})
    monkeypatch.setattr(runtime,"create_recognizer",lambda *args:object())
    output=tmp_path/"run"
    args=SimpleNamespace(game="hopping",session=session,profile=None,control=True,mode="calibrate",port="dummy",
        rounds=2,output=output,no_record=True,no_debug=True,debug_port=0,device=0,backend="DSHOW",
        width=1920,height=1080,fps=60,seconds=10)
    assert runtime.run_session(args)==0
    commands=[json.loads(line) for line in (output/"commands.jsonl").read_text().splitlines()]
    assert commands and all("B" not in row["state"]["buttons"] for row in commands)
    assert any("A" in row["state"]["buttons"] for row in commands)
    samples=[json.loads(line) for line in (output/"calibration_samples.jsonl").read_text().splitlines()]
    assert {row["round_id"] for row in samples}=={1,2}
    assert all(row["live"] and row["executed"] for row in samples)
    report=json.loads((output/"motion_fit.json").read_text(encoding="utf-8"))
    assert report["passed"] is False and report["calibrated"] is False
