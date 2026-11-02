"""Real hub/guard with simulated capture and transport, never real devices."""
import http.client
import json
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import numpy as np
import pytest

from raiders_helper import session_runtime as runtime
from raiders_helper.capture.models import FramePacket
from raiders_helper.control.buttons import Button, ControllerState
from raiders_helper.control.dummy import DummyBackend
from raiders_helper.control.guard import GuardedController
from raiders_helper.debug_api import DebugHub
from raiders_helper.experiment_recording import RecordingBackend
from raiders_helper.manual_input_pump import ManualInputPump
from raiders_helper.models import Decision, Game, Observation


def wait_for(condition, timeout=.75):
    deadline=time.perf_counter()+timeout
    while not condition():
        assert time.perf_counter()<deadline, "condition timed out"
        time.sleep(.002)


class TimelineBackend(DummyBackend):
    def __init__(self):
        super().__init__()
        self.timeline=[]

    def send_state(self,state):
        result=super().send_state(state)
        self.timeline.append((time.perf_counter(),state))
        return result


def request(hub,method,path,payload=None):
    endpoint=hub.endpoint
    address=urlsplit(endpoint["url"])
    connection=http.client.HTTPConnection(address.hostname,address.port,timeout=1)
    try:
        connection.request(method,path,body=None if payload is None else json.dumps(payload).encode(),
            headers={endpoint["token_header"]:endpoint["token"],"Content-Type":"application/json"})
        response=connection.getresponse()
        value=json.loads(response.read())
        assert response.status in {200,202}
        return value
    finally:
        connection.close()


@pytest.fixture
def pump_context(tmp_path):
    hub=DebugHub(tmp_path/"hub",control_allowed=True)
    backend=TimelineBackend()
    wrapped=RecordingBackend(backend,tmp_path/"transport",callback=lambda event:
        hub.record_control(event["state"],timestamp=event["after"]) if event.get("success") else None)
    assert wrapped.connect("dummy")
    guard=GuardedController(wrapped)
    fresh=[True]
    def frame():
        return SimpleNamespace(timestamp=time.perf_counter()-(0 if fresh[0] else .4),sequence=31)
    stop=threading.Event()
    pump=ManualInputPump(hub,guard,frame,stop_event=stop)
    pump.start()
    try:
        yield SimpleNamespace(hub=hub,backend=backend,guard=guard,pump=pump,fresh=fresh,stop=stop)
    finally:
        pump.close();guard.close();wrapped.close();hub.close()


def pause(context):
    receipt=context.hub.submit({"command":"pause"})
    wait_for(lambda:context.hub.command_result(receipt["id"])["status"]=="succeeded")


def test_short_press_expiry_receipt_follows_actual_neutral_transport(pump_context):
    ctx=pump_context;pause(ctx)
    receipt=ctx.hub.submit({"command":"press","buttons":["A"],"hold_ms":60})
    wait_for(lambda:Button.A in ctx.backend.state.buttons)
    assert ctx.hub.command_result(receipt["id"])["status"]=="running"
    wait_for(lambda:ctx.hub.command_result(receipt["id"])["status"]=="succeeded")
    assert ctx.backend.state==ControllerState.neutral()
    assert any(Button.A in state.buttons for _,state in ctx.backend.timeline)


def test_idle_neutral_is_not_repeated_but_explicit_neutral_state_has_real_send_receipt(pump_context):
    ctx=pump_context;pause(ctx)
    wait_for(lambda:ctx.pump.snapshot().diagnostics.get("debug",{}).get("transport_required") is False)
    before=len(ctx.backend.timeline)
    time.sleep(.08)
    assert len(ctx.backend.timeline)==before
    receipt=ctx.hub.submit({"command":"state","left_stick":{"x":0,"y":0},"hold_ms":80})
    wait_for(lambda:ctx.hub.command_result(receipt["id"])["status"]=="succeeded")
    assert len(ctx.backend.timeline)==before+1
    time.sleep(.04)
    assert len(ctx.backend.timeline)==before+1


def test_stale_capture_releases_and_does_not_resurrect_old_request(pump_context):
    ctx=pump_context;pause(ctx)
    receipt=ctx.hub.submit({"command":"press","buttons":["A"],"hold_ms":500})
    wait_for(lambda:Button.A in ctx.backend.state.buttons)
    ctx.fresh[0]=False
    wait_for(lambda:ctx.backend.state==ControllerState.neutral(),timeout=.20)
    wait_for(lambda:ctx.hub.command_result(receipt["id"])["status"]=="failed")
    released=time.perf_counter()
    ctx.fresh[0]=True
    time.sleep(.04)
    assert all(state==ControllerState.neutral() for stamp,state in ctx.backend.timeline if stamp>=released)
    new=ctx.hub.submit({"command":"press","buttons":["B"],"hold_ms":40})
    wait_for(lambda:ctx.hub.command_result(new["id"])["status"]=="succeeded")
    assert any(Button.B in state.buttons for stamp,state in ctx.backend.timeline if stamp>=released)


def test_stop_event_releases_without_waiting_for_analysis_or_capture_timeout(pump_context):
    ctx=pump_context;pause(ctx)
    ctx.hub.submit({"command":"state","buttons":["ZR"],"hold_ms":500})
    wait_for(lambda:Button.ZR in ctx.backend.state.buttons)
    ctx.stop.set()
    wait_for(lambda:ctx.backend.state==ControllerState.neutral(),timeout=.20)
    assert ctx.pump.snapshot().diagnostics["manual_input_pump"]["stopped"]


def test_renewed_states_have_no_intervening_neutral_and_resume_releases(pump_context):
    ctx=pump_context;pause(ctx)
    first=None
    for _ in range(4):
        receipt=ctx.hub.submit({"command":"state","buttons":["ZR"],"hold_ms":120})
        wait_for(lambda:Button.ZR in ctx.backend.state.buttons)
        first=first or next(stamp for stamp,state in ctx.backend.timeline if Button.ZR in state.buttons)
        time.sleep(.03)
    end=time.perf_counter()
    assert all(Button.ZR in state.buttons for stamp,state in ctx.backend.timeline if first<=stamp<=end)
    resume=ctx.hub.submit({"command":"resume"})
    wait_for(lambda:ctx.hub.command_result(resume["id"])["status"]=="succeeded")
    wait_for(lambda:ctx.backend.state==ControllerState.neutral())
    assert not ctx.hub.paused


def test_slow_recognizer_does_not_delay_http_press_or_reapply_expired_input(tmp_path,monkeypatch):
    entered,unblock,stop=threading.Event(),threading.Event(),threading.Event()
    hubs=[];updates=[];errors=[]
    frame=np.zeros((48,64,3),np.uint8)
    class Worker:
        error=""
        def __init__(self,*args,**kwargs):pass
        def start(self):pass
        def latest(self):return FramePacket(frame,time.perf_counter(),11,60)
        def stop(self):pass
    class Matcher:
        has_ui_candidates=False
        def __init__(self,*args):pass
        def observe(self,frame):return "unknown",{}
    class SlowEngine:
        def __init__(self,*args,**kwargs):pass
        def process(self,packet):
            entered.set()
            assert unblock.wait(2),"test did not release simulated slow vision"
            return Observation(Game.MINING,packet.timestamp,packet.sequence,"playing",.9),Decision(
                ControllerState(frozenset({Button.B})),"score advice must never own manual input")
    backend=TimelineBackend()
    monkeypatch.setattr(runtime,"CaptureWorker",Worker)
    monkeypatch.setattr(runtime,"ScreenMatcher",Matcher)
    monkeypatch.setattr(runtime,"Engine",SlowEngine)
    monkeypatch.setattr(runtime,"EasyConBackend",lambda:backend)
    monkeypatch.setattr(runtime,"load_profile",lambda *args:{"advisory_only":True})
    monkeypatch.setattr(runtime,"create_recognizer",lambda *args:object())
    session=tmp_path/"session.json";session.write_text('{"version":1,"screens":[]}',encoding="utf8")
    args=SimpleNamespace(game="mining",session=session,profile=None,control=True,mode="manual",port="dummy",
        rounds=3,output=tmp_path/"run",no_record=True,no_debug=False,debug_port=0,device=0,backend="DSHOW",
        width=1920,height=1080,fps=60,seconds=0)
    def update(*context):updates.append(context);stop.set()
    def run():
        try:runtime.run_session(args,stop_event=stop,on_log=lambda _:None,on_hub=hubs.append,on_update=update)
        except BaseException as error:errors.append(error)
    thread=threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(1)
        hub=hubs[0]
        # Raw capture remains available while menu/game analysis is blocked.
        body,meta=hub.frame()
        assert meta["sequence"]==11 and meta["source"]=="capture_worker_latest"
        assert meta["age_seconds"]<.1 and not unblock.is_set()
        assert hub.status()["frames"]["capture"]["sequence"]==11
        receipt=request(hub,"POST","/commands",{"command":"pause"})
        wait_for(lambda:request(hub,"GET","/commands/"+receipt["id"])["status"]=="succeeded")
        issued=time.perf_counter()
        receipt=request(hub,"POST","/commands",{"command":"press","buttons":["A"],"hold_ms":80})
        wait_for(lambda:any(Button.A in state.buttons for _,state in backend.timeline),timeout=.20)
        sent=next(stamp for stamp,state in backend.timeline if Button.A in state.buttons)
        assert sent-issued<.20 and not unblock.is_set()
        wait_for(lambda:request(hub,"GET","/commands/"+receipt["id"])["status"]=="succeeded")
        assert backend.state==ControllerState.neutral()
        neutral=time.perf_counter()
        unblock.set()
        thread.join(timeout=2)
        assert not thread.is_alive() and not errors
        assert updates and updates[0][2].state==ControllerState.neutral()
        assert updates[0][3]["input_transport_owner"]==ManualInputPump.THREAD_NAME
        assert all(state==ControllerState.neutral() for stamp,state in backend.timeline if stamp>=neutral)
        commands=[json.loads(line) for line in (args.output/"commands.jsonl").read_text(encoding="utf8").splitlines()]
        assert all(row["thread"]==ManualInputPump.THREAD_NAME for row in commands if row["state"]["buttons"]==["A"])
        assert not any(row["state"]["buttons"]==["B"] for row in commands)
    finally:
        unblock.set();stop.set();thread.join(timeout=3)
