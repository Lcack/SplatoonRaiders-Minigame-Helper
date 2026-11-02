"""GUI callbacks use the real session orchestration, with no real devices."""
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from raiders_helper import debug_api, experiment_recording
from raiders_helper import session_runtime as runtime
from raiders_helper.capture.models import FramePacket
from raiders_helper.control.buttons import Button, ControllerState
from raiders_helper.control.dummy import DummyBackend
from raiders_helper.models import Decision, Game, Observation


def setup_session(tmp_path, monkeypatch, *, latest, control=False):
    session=tmp_path/"session.json"
    session.write_text(json.dumps({"version":1,"screens":[]}),encoding="utf-8")
    instances={}

    class Worker:
        error=""
        def __init__(self,*args,**kwargs):
            self.started=self.stopped=False
            instances["worker"]=self
        def start(self):self.started=True
        def latest(self):return latest()
        def stop(self):self.stopped=True

    class Matcher:
        has_ui_candidates=False
        def __init__(self,*args):pass
        def observe(self,frame):return "unknown",{}

    class Engine:
        def __init__(self,*args,**kwargs):pass
        def process(self,packet):
            observation=Observation(Game.HOPPING,packet.timestamp,packet.sequence)
            return observation,Decision(diagnostics={"timing":{"analysis_ms":1.25}})

    class Recorder:
        error=""
        def __init__(self,*args,**kwargs):
            self.closed=False
            self.active=kwargs.get("initially_active",True)
            self.observations=[]
            self.phase_flags=[]
            instances["recorder"]=self
        def on_frame(self,packet):pass
        def observe_phase(self,screen,phase,timestamp=None,**kwargs):
            self.observations.append((screen,phase,timestamp))
            self.phase_flags.append(kwargs)
            return self.active
        def close(self):self.closed=True

    class Hub:
        def __init__(self,*args,**kwargs):
            self.started=self.closed=False
            self.status=None
            self.controls=[]
            instances["hub"]=self
        def start(self):self.started=True
        def close(self):self.closed=True
        def take_manual(self,now):
            return Decision(ControllerState(frozenset({Button.A})),"手动短按") if control else None
        def publish(self,status,**kwargs):self.status=status
        def record_control(self,state,**kwargs):self.controls.append(state)

    backend=DummyBackend()
    monkeypatch.setattr(runtime,"CaptureWorker",Worker)
    monkeypatch.setattr(runtime,"ScreenMatcher",Matcher)
    monkeypatch.setattr(runtime,"Engine",Engine)
    monkeypatch.setattr(runtime,"EasyConBackend",lambda:backend)
    monkeypatch.setattr(runtime,"load_profile",lambda *args:{"advisory_only":True})
    monkeypatch.setattr(runtime,"create_recognizer",lambda *args:object())
    monkeypatch.setattr(experiment_recording,"AsyncExperimentRecorder",Recorder)
    monkeypatch.setattr(debug_api,"DebugHub",Hub)
    args=SimpleNamespace(game="hopping",session=session,profile=None,control=control,mode="manual",port="dummy",
        rounds=3,output=tmp_path/"run",no_record=False,no_debug=False,debug_port=0,device=0,backend="DSHOW",
        width=1920,height=1080,fps=60,seconds=0)
    return args,instances,backend


@pytest.mark.parametrize("stale",[False,True])
def test_stop_without_fresh_capture_closes_resources_and_revokes_hub(tmp_path,monkeypatch,capsys,stale):
    stop=threading.Event()
    def latest():
        stop.set()
        return (FramePacket(np.zeros((48,64,3),np.uint8),time.perf_counter()-3,1,60)
                if stale else None)
    args,instances,_=setup_session(tmp_path,monkeypatch,latest=latest)
    logs,hubs=[],[]
    assert runtime.run_session(args,stop_event=stop,on_log=logs.append,on_hub=hubs.append)==0
    assert hubs==[instances["hub"],None]
    assert instances["hub"].started and instances["hub"].closed
    assert instances["worker"].stopped and instances["recorder"].closed
    assert any("已停止" in line for line in logs)
    assert all("已观察到一局结束" not in line for line in logs)
    assert capsys.readouterr().out==""


def test_update_callback_has_display_context_and_stop_releases_input(tmp_path,monkeypatch):
    stop=threading.Event()
    packet=FramePacket(np.zeros((100,200,3),np.uint8),time.perf_counter(),17,60)
    # Refresh timestamp when polled; a simulated frame must not age during setup.
    def latest():return FramePacket(packet.frame,time.perf_counter(),17,60)
    args,instances,backend=setup_session(tmp_path,monkeypatch,latest=latest,control=True)
    results=[]
    def update(*context):
        results.append(context)
        stop.set()
    assert runtime.run_session(args,stop_event=stop,on_log=lambda _:None,on_update=update)==0
    assert len(results)==1
    received,observation,decision,status=results[0]
    assert received.sequence==17 and observation.sequence==17
    assert decision.state.buttons==frozenset({Button.A})
    assert status["output"]==str(args.output.resolve())
    assert status["game"]=="hopping" and status["mode"]=="manual"
    assert status["screen"]=="unknown" and status["stage"]=="await_menu"
    assert status["control_enabled"] and status["source"]=="manual"
    assert status["capture_size"]==[200,100] and status["capture_fps_estimate"]==60
    assert status["started_rounds"]==status["completed_rounds"]==0
    assert status["recording_enabled"] is True and status["recording_active"] is False
    assert instances["recorder"].observations==[("unknown","unknown",received.timestamp)]
    assert status["timing"]["analysis_ms"]==1.25 and not status["fit_passed"]
    assert instances["hub"].status==status
    assert not backend.connected and backend.state==ControllerState.neutral()
    events=[json.loads(line) for line in (args.output/"commands.jsonl").read_text().splitlines()]
    assert any(event["state"]["buttons"]==["A"] for event in events)
    assert events[-1]["state"]["buttons"]==[]


def test_update_callback_failure_propagates_after_controller_and_capture_cleanup(tmp_path,monkeypatch):
    def latest():return FramePacket(np.zeros((48,64,3),np.uint8),time.perf_counter(),1,60)
    args,instances,backend=setup_session(tmp_path,monkeypatch,latest=latest,control=True)
    hubs=[]
    def broken_update(*args):raise ValueError("display callback failed")
    with pytest.raises(ValueError,match="display callback failed"):
        runtime.run_session(args,on_log=lambda _:None,on_update=broken_update,on_hub=hubs.append)
    assert instances["worker"].stopped and instances["recorder"].closed
    assert instances["hub"].closed and hubs[-1] is None
    assert not backend.connected and backend.state==ControllerState.neutral()


def test_hub_handoff_failure_is_revoked_and_closes_unstarted_resources(tmp_path,monkeypatch):
    args,instances,backend=setup_session(tmp_path,monkeypatch,latest=lambda:None,control=True)
    hubs=[]
    def handoff(hub):
        hubs.append(hub)
        if hub is not None:raise ValueError("handoff failed")
    with pytest.raises(ValueError,match="handoff failed"):
        runtime.run_session(args,on_log=lambda _:None,on_hub=handoff)
    assert hubs==[instances["hub"],None]
    assert instances["hub"].closed and instances["recorder"].closed
    assert instances["worker"].stopped and not instances["worker"].started
    assert not backend.connected


def test_log_callback_failure_still_closes_recorder_and_retracts_hub(tmp_path,monkeypatch):
    args,instances,_=setup_session(tmp_path,monkeypatch,latest=lambda:None)
    hubs=[]
    def log(message):raise ValueError("log callback failed")
    with pytest.raises(ValueError,match="log callback failed"):
        runtime.run_session(args,on_log=log,on_hub=hubs.append)
    assert hubs==[None] and instances["hub"].closed
    assert instances["recorder"].closed and instances["worker"].stopped


def test_cli_logging_remains_available_without_log_callback(tmp_path,monkeypatch,capsys):
    stop=threading.Event()
    def latest():stop.set()
    args,_,_=setup_session(tmp_path,monkeypatch,latest=latest)
    assert runtime.run_session(args,stop_event=stop)==0
    text=capsys.readouterr().out
    assert "hopping" in text and "已停止" in text


def test_raw_result_stops_recorder_before_stable_ui_and_does_not_publish_play(tmp_path,monkeypatch):
    stop=threading.Event()
    def latest():return FramePacket(np.zeros((48,64,3),np.uint8),time.perf_counter(),1,60)
    args,instances,_=setup_session(tmp_path,monkeypatch,latest=latest)
    class Matcher:
        has_ui_candidates=True
        previous="result:retry"
        def __init__(self,*args):pass
        def observe(self,frame):return "unknown",{}
    class Engine:
        def __init__(self,*args,**kwargs):pass
        def process(self,packet):
            return Observation(Game.HOPPING,packet.timestamp,packet.sequence,"playing",.9),Decision(diagnostics={"timing":{}})
    monkeypatch.setattr(runtime,"ScreenMatcher",Matcher)
    monkeypatch.setattr(runtime,"Engine",Engine)
    updates=[]
    def update(*context):updates.append(context);stop.set()
    assert runtime.run_session(args,stop_event=stop,on_log=lambda _:None,on_update=update)==0
    assert instances["recorder"].observations[0][0:2]==("result:retry","playing")
    assert updates[0][3]["screen"]=="unknown"


def test_raw_countdown_does_not_start_recorder_until_ui_is_stable(tmp_path,monkeypatch):
    stop=threading.Event()
    def latest():return FramePacket(np.zeros((48,64,3),np.uint8),time.perf_counter(),1,60)
    args,instances,_=setup_session(tmp_path,monkeypatch,latest=latest)
    class Matcher:
        has_ui_candidates=True
        previous="countdown:hopping"
        def __init__(self,*args):pass
        def observe(self,frame):return "unknown",{}
    monkeypatch.setattr(runtime,"ScreenMatcher",Matcher)
    assert runtime.run_session(args,stop_event=stop,on_log=lambda _:None,
                               on_update=lambda *context:stop.set())==0
    assert instances["recorder"].observations[0][0:2]==("unknown","unknown")


def test_game_ended_metric_reaches_gate_even_when_phase_still_playing(tmp_path,monkeypatch):
    stop=threading.Event()
    def latest():return FramePacket(np.zeros((48,64,3),np.uint8),time.perf_counter(),1,60)
    args,instances,_=setup_session(tmp_path,monkeypatch,latest=latest)
    class Engine:
        def __init__(self,*args,**kwargs):pass
        def process(self,packet):
            return Observation(Game.HOPPING,packet.timestamp,packet.sequence,"playing",.9,
                               metrics={"game_ended":True}),Decision(diagnostics={"timing":{}})
    monkeypatch.setattr(runtime,"Engine",Engine)
    assert runtime.run_session(args,stop_event=stop,on_log=lambda _:None,
                               on_update=lambda *context:stop.set())==0
    assert instances["recorder"].phase_flags[0]["game_ended"] is True


@pytest.mark.parametrize("playing",[False,True])
def test_idle_log_sampling_keeps_display_updates_and_actual_transport(tmp_path,monkeypatch,playing):
    stop=threading.Event()
    counter=0
    def latest():
        nonlocal counter
        counter+=1
        return FramePacket(np.zeros((48,64,3),np.uint8),time.perf_counter(),counter,60)
    args,instances,_=setup_session(tmp_path,monkeypatch,latest=latest,control=True)
    class Engine:
        def __init__(self,*args,**kwargs):pass
        def process(self,packet):
            return Observation(Game.HOPPING,packet.timestamp,packet.sequence,
                               "playing" if playing else "unknown"),Decision()
    monkeypatch.setattr(runtime,"Engine",Engine)
    updates=[]
    def update(*context):
        updates.append(context)
        if len(updates)==20:stop.set()
    assert runtime.run_session(args,stop_event=stop,on_log=lambda _:None,on_update=update)==0
    traces=[json.loads(line) for line in (args.output/"trace.jsonl").read_text(encoding="utf-8").splitlines()]
    sessions=[json.loads(line) for line in (args.output/"session.jsonl").read_text(encoding="utf-8").splitlines()]
    commands=[json.loads(line) for line in (args.output/"commands.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(updates)==20
    assert len(traces)==len(sessions)==(20 if playing else 1)
    # Input pumping is independent of the twenty display/analysis updates.
    # Preserve the stronger audit property: every real send reaches the hub
    # callback and disk log, including the final neutral releases.
    assert any(row["state"]["buttons"]==["A"] for row in commands)
    assert [row["state"] for row in commands]==instances["hub"].controls
    assert commands[-1]["state"]["buttons"]==[]
