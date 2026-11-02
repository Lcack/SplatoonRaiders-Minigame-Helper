import http.client
import json
import time
from urllib.parse import urlsplit

import cv2
import numpy as np
import pytest

from raiders_helper.control.buttons import Button, ControllerState, StickVector
from raiders_helper.debug_api import DebugHub
from raiders_helper.trace import state_dict


def submit_at(hub, payload, stamp, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr("raiders_helper.debug_api.time.perf_counter", lambda:stamp)
        return hub.submit(payload)


def request(hub, method, path, payload=None, *, authenticated=True, headers=None):
    endpoint=hub.endpoint
    address=urlsplit(endpoint["url"])
    supplied={"Content-Type":"application/json",**(headers or {})}
    if authenticated:
        supplied[endpoint["token_header"]]=endpoint["token"]
    connection=http.client.HTTPConnection(address.hostname,address.port,timeout=2)
    try:
        body=None if payload is None else json.dumps(payload).encode()
        connection.request(method,path,body=body,headers=supplied)
        response=connection.getresponse()
        data=response.read()
        return response.status,data,dict(response.getheaders())
    finally:
        connection.close()


@pytest.fixture
def hub(tmp_path):
    instance=DebugHub(tmp_path/"run",control_allowed=True)
    instance.start()
    try:
        yield instance
    finally:
        instance.close()


def pause(hub):
    receipt=hub.submit({"command":"pause"})
    decision=hub.take_manual()
    assert decision.state==ControllerState.neutral()
    assert hub.command_result(receipt["id"])["status"]=="succeeded"


def test_http_authentication_origin_host_and_discovery(hub):
    endpoint=hub.endpoint
    local=json.loads((hub.session_dir/"control_endpoint.json").read_text(encoding="utf-8"))
    assert local["token"]==endpoint["token"]
    assert urlsplit(endpoint["url"]).hostname=="127.0.0.1"
    assert request(hub,"GET","/status",authenticated=False)[0]==401
    assert request(hub,"GET","/status",headers={"Origin":"http://localhost"})[0]==403
    assert request(hub,"GET","/status",headers={"Host":"foreign.invalid"})[0]==403
    status,body,_=request(hub,"GET","/status")
    assert status==200
    assert endpoint["token"].encode() not in body


def test_legacy_client_token_header_and_commands_are_supported(hub):
    endpoint=hub.endpoint
    code,body,_=request(hub,"POST","/commands",{"command":"pause"},authenticated=False,
                        headers={"X-Tableturf-Token":endpoint["token"]})
    assert code==202
    receipt=json.loads(body)
    assert hub.paused is False
    assert receipt["status"]=="queued"
    hub.take_manual()
    code,body,_=request(hub,"GET","/commands/"+receipt["id"])
    assert code==200 and json.loads(body)["status"]=="succeeded"


def test_http_threads_only_queue_controls_and_owner_loop_applies_them(hub):
    assert request(hub,"POST","/control",{"buttons":["A"]})[0]==409
    assert request(hub,"POST","/pause",{})[0]==202
    assert hub.paused is False
    assert hub.take_manual().state==ControllerState.neutral()
    assert hub.paused
    code,body,_=request(hub,"POST","/control",{"buttons":["A"],"hold_ms":100})
    assert code==202
    receipt=json.loads(body)
    assert hub.command_result(receipt["id"])["status"]=="queued"
    decision=hub.take_manual()
    assert decision.state.buttons==frozenset({Button.A})
    assert hub.command_result(receipt["id"])["status"]=="running"


def test_pulse_deadline_does_not_renew_and_success_requires_actual_release(hub):
    pause(hub)
    receipt=hub.submit({"command":"press","left_stick":{"x":12345,"y":0},"hold_ms":100})
    now=time.perf_counter()
    first=hub.take_manual(now)
    assert first.state.left_stick.x==12345
    hub.record_control(state_dict(first.state),timestamp=now,source="transport")
    assert hub.take_manual(now+.08).state==first.state
    assert hub.take_manual(now+.101).state==ControllerState.neutral()
    assert hub.command_result(receipt["id"])["status"]=="running"
    hub.record_control(ControllerState.neutral(),timestamp=now+.101,source="watchdog")
    assert hub.command_result(receipt["id"])["status"]=="succeeded"
    assert hub.take_manual(now+.2).state==ControllerState.neutral()
    assert hub.paused


def test_held_state_renewals_are_continuous_and_polling_does_not_extend_the_lease(hub,monkeypatch):
    pause(hub)
    receipts=[]
    for stamp in (10.0,10.05,10.10,10.15,10.20):
        receipt=submit_at(hub,{"command":"state","buttons":["A"],"hold_ms":150},stamp,monkeypatch)
        receipts.append(receipt)
        decision=hub.take_manual(stamp)
        assert decision.state.buttons==frozenset({Button.A})
        hub.record_control(decision.state,timestamp=stamp)
        assert hub.take_manual(stamp+.04).state==decision.state
    for old,new in zip(receipts,receipts[1:]):
        completed=hub.command_result(old["id"])
        assert completed["status"]=="succeeded" and completed["superseded_by"]==new["id"]
    assert hub.take_manual(10.349).state.buttons==frozenset({Button.A})
    assert hub.take_manual(10.351).state==ControllerState.neutral()
    assert hub.command_result(receipts[-1]["id"])["status"]=="running"
    hub.record_control(ControllerState.neutral(),timestamp=10.351)
    assert hub.command_result(receipts[-1]["id"])["status"]=="succeeded"
    assert hub.take_manual(10.5).state==ControllerState.neutral()


def test_state_changes_buttons_and_both_axes_without_an_intermediate_release(hub,monkeypatch):
    pause(hub)
    first=submit_at(hub,{"command":"state","buttons":["A"],"left_stick":{"x":1000}},20,monkeypatch)
    initial=hub.take_manual(20)
    hub.record_control(initial.state,timestamp=20)
    updated=submit_at(hub,{"command":"state","buttons":["A","B"],
        "left_stick":{"x":-2000,"y":3000},"right_stick":{"x":4000,"y":-5000}},20.03,monkeypatch)
    changed=hub.take_manual(20.03)
    assert changed.state==ControllerState(frozenset({Button.A,Button.B}),StickVector(-2000,3000),StickVector(4000,-5000))
    assert changed.state!=ControllerState.neutral()
    assert hub.command_result(first["id"])["superseded_by"]==updated["id"]


def test_unconsumed_state_updates_coalesce_to_the_latest_complete_value(hub,monkeypatch):
    pause(hub)
    first=submit_at(hub,{"command":"state","buttons":["A"]},30,monkeypatch)
    second=submit_at(hub,{"command":"state","buttons":["B"]},30.01,monkeypatch)
    latest=submit_at(hub,{"command":"state","right_stick":{"x":6000}},30.02,monkeypatch)
    assert hub.status()["control"]["pending_commands"]==1
    assert hub.command_result(first["id"])["status"]=="failed"
    assert hub.command_result(second["id"])["status"]=="failed"
    decision=hub.take_manual(30.02)
    assert not decision.state.buttons and decision.state.right_stick.x==6000
    assert decision.diagnostics["debug"]["command_id"]==latest["id"]


def test_state_lease_is_bounded_by_submission_time_not_delayed_owner_pump(hub,monkeypatch):
    pause(hub)
    receipt=submit_at(hub,{"command":"state","buttons":["A"],"hold_ms":100},40,monkeypatch)
    decision=hub.take_manual(40.06)
    hub.record_control(decision.state,timestamp=40.06)
    assert decision.state.buttons==frozenset({Button.A})
    assert hub.take_manual(40.101).state==ControllerState.neutral()
    hub.record_control(ControllerState.neutral(),timestamp=40.101)
    assert hub.command_result(receipt["id"])["status"]=="succeeded"
    stale=submit_at(hub,{"command":"state","buttons":["B"],"hold_ms":50},41,monkeypatch)
    assert hub.take_manual(41.06).state==ControllerState.neutral()
    assert hub.command_result(stale["id"])["status"]=="failed"


def test_late_state_update_cannot_skip_an_expired_lease_release(hub,monkeypatch):
    pause(hub)
    first=submit_at(hub,{"command":"state","buttons":["A"],"hold_ms":100},50,monkeypatch)
    hub.record_control(hub.take_manual(50).state,timestamp=50)
    late=submit_at(hub,{"command":"state","buttons":["B"]},50.11,monkeypatch)
    assert hub.take_manual(50.11).state==ControllerState.neutral()
    assert hub.command_result(late["id"])["status"]=="queued"
    # Owner must acknowledge an actual neutral transport submission first.
    assert hub.take_manual(50.115).state==ControllerState.neutral()
    hub.record_control(ControllerState.neutral(),timestamp=50.115)
    assert hub.command_result(first["id"])["status"]=="succeeded"
    assert hub.take_manual(50.12).state.buttons==frozenset({Button.B})


def test_neutral_state_preempts_pending_controls_and_waits_for_actual_neutral(hub,monkeypatch):
    pause(hub)
    active=submit_at(hub,{"command":"state","buttons":["A"]},60,monkeypatch)
    hub.record_control(hub.take_manual(60).state,timestamp=60)
    resume=submit_at(hub,{"command":"resume"},60.01,monkeypatch)
    queued=submit_at(hub,{"command":"state","buttons":["B"]},60.02,monkeypatch)
    released=submit_at(hub,{"command":"state"},60.03,monkeypatch)
    assert hub.take_manual(60.03).state==ControllerState.neutral() and hub.paused
    for receipt in (active,resume,queued):
        assert hub.command_result(receipt["id"])["status"]=="failed"
    assert hub.command_result(released["id"])["status"]=="running"
    hub.record_control(ControllerState.neutral(),timestamp=60.03)
    assert hub.command_result(released["id"])["status"]=="succeeded"
    assert hub.take_manual(60.04).state==ControllerState.neutral()


def test_release_preempts_held_state_and_queued_renewal(hub,monkeypatch):
    pause(hub)
    active=submit_at(hub,{"command":"state","buttons":["A"]},70,monkeypatch)
    hub.record_control(hub.take_manual(70).state,timestamp=70)
    renewal=submit_at(hub,{"command":"state","buttons":["A"]},70.05,monkeypatch)
    release=submit_at(hub,{"command":"release"},70.06,monkeypatch)
    assert hub.take_manual(70.06).state==ControllerState.neutral()
    assert hub.paused
    assert hub.command_result(active["id"])["status"]=="failed"
    assert hub.command_result(renewal["id"])["status"]=="failed"
    assert hub.command_result(release["id"])["status"]=="succeeded"


def test_state_can_replace_queued_or_active_pulse_but_not_fake_its_completion(hub,monkeypatch):
    pause(hub)
    queued=submit_at(hub,{"command":"press","buttons":["B"]},80,monkeypatch)
    state=submit_at(hub,{"command":"state","buttons":["A"]},80.01,monkeypatch)
    assert hub.command_result(queued["id"])["status"]=="failed"
    assert hub.take_manual(80.01).state.buttons==frozenset({Button.A})
    # No actual submit was recorded: replacing cannot make it successful.
    submit_at(hub,{"command":"state","buttons":["B"]},80.02,monkeypatch)
    assert hub.take_manual(80.02).state.buttons==frozenset({Button.B})
    assert hub.command_result(state["id"])["status"]=="failed"
    with pytest.raises(RuntimeError,match="already pending"):
        hub.submit({"command":"press","buttons":["A"]})


def test_http_state_updates_obey_runtime_permission_and_pause_gates(hub,tmp_path):
    assert request(hub,"POST","/commands",{"command":"state","buttons":["A"]})[0]==409
    pause(hub)
    assert request(hub,"POST","/commands",{"command":"state","buttons":["A"],"hold_ms":150})[0]==202
    assert hub.take_manual().state.buttons==frozenset({Button.A})
    hub.publish(control_allowed=False)
    assert hub.take_manual().state==ControllerState.neutral()
    assert request(hub,"POST","/commands",{"command":"state","buttons":["A"]})[0]==403
    assert "state" in hub.status()["control"]["command_names"]


@pytest.mark.parametrize("payload",[
    {"command":"state","buttons":["INVALID"]},
    {"command":"state","hold_ms":0},
    {"command":"state","hold_ms":True},
    {"command":"state","hold_ms":1001},
    {"command":"state","left_stick":{"x":1.5}},
])
def test_invalid_held_state_is_rejected_before_queueing(hub,payload):
    pause(hub)
    assert request(hub,"POST","/commands",payload)[0]==400
    assert hub.status()["control"]["pending_commands"]==0


def test_undelivered_pulse_is_not_reported_as_success(hub):
    pause(hub)
    receipt=hub.submit({"command":"press","buttons":["B"],"hold_ms":30})
    now=time.perf_counter()
    hub.take_manual(now)
    hub.take_manual(now+.04)
    hub.record_control(ControllerState.neutral(),timestamp=now+.04)
    assert hub.command_result(receipt["id"])["status"]=="failed"


def test_release_cancels_active_pulse_and_queued_resume(hub):
    pause(hub)
    pulse=hub.submit({"command":"press","buttons":["B"]})
    hub.take_manual()
    resume=hub.submit({"command":"resume"})
    release=hub.submit({"command":"release"})
    assert hub.take_manual().state==ControllerState.neutral()
    assert hub.paused
    assert hub.command_result(pulse["id"])["status"]=="failed"
    assert hub.command_result(resume["id"])["status"]=="failed"
    assert hub.command_result(release["id"])["status"]=="succeeded"
    assert hub.take_manual().state==ControllerState.neutral()


def test_resume_exposes_neutral_tick_before_automation(hub):
    pause(hub)
    hub.submit({"command":"resume"})
    assert hub.take_manual().state==ControllerState.neutral()
    assert hub.paused is False
    assert hub.take_manual() is None


def test_runtime_permission_gate_cannot_be_enabled_through_http(tmp_path):
    hub=DebugHub(tmp_path,control_allowed=False)
    hub.start()
    try:
        pause(hub)
        assert request(hub,"POST","/control",{"buttons":["A"]})[0]==403
        assert request(hub,"POST","/control",{"buttons":["A"],"control_allowed":True})[0]==400
    finally:
        hub.close()


def test_runtime_revocation_interrupts_a_manual_pulse(hub):
    pause(hub)
    receipt=hub.submit({"command":"press","buttons":["A"]})
    hub.take_manual()
    hub.publish(control_allowed=False)
    assert hub.take_manual().state==ControllerState.neutral()
    assert hub.command_result(receipt["id"])["status"]=="failed"


def test_stale_queue_is_not_replayed_and_manual_requests_do_not_backlog(hub):
    pause(hub)
    receipt=hub.submit({"command":"press","buttons":["A"]})
    with pytest.raises(RuntimeError,match="already pending"):
        hub.submit({"command":"press","buttons":["B"]})
    assert hub.take_manual(time.perf_counter()+1.1).state==ControllerState.neutral()
    assert hub.command_result(receipt["id"])["status"]=="failed"


def test_frame_snapshot_is_lazy_png_immutable_and_has_sequence(hub):
    frame=np.zeros((20,30,3),np.uint8)
    frame[3:12,5:15]=(20,160,240)
    stamp=time.perf_counter()-.1
    hub.publish({"stage":"playing"},frame=frame,sequence=72,captured_at=stamp)
    frame[:]=255
    code,data,headers=request(hub,"GET","/frame.png")
    assert code==200 and headers["Content-Type"]=="image/png"
    decoded=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    assert tuple(decoded[0,0])==(0,0,0)
    assert headers["X-Frame-Sequence"]=="72"
    assert float(headers["X-Frame-Age-Seconds"])>=.1
    assert float(headers["X-Frame-Captured-At"]) == pytest.approx(stamp, abs=1e-8)
    assert request(hub,"GET","/frame?kind=capture")[1]==data
    assert request(hub,"GET","/frame?kind=analysis")[0]==503
    assert request(hub,"GET","/frame?path=../secret")[0]==400


def test_reduced_frame_keeps_source_identity_and_default_full_size_cache(hub,monkeypatch):
    frame=np.full((900,1600,3),(20,160,240),np.uint8)
    stamp=time.perf_counter()-.1
    hub.publish(frame=frame,sequence=81,captured_at=stamp)
    shapes=[]
    encode=cv2.imencode
    def observe_encode(extension,image,*args,**kwargs):
        shapes.append(image.shape)
        return encode(extension,image,*args,**kwargs)
    monkeypatch.setattr(cv2,"imencode",observe_encode)

    code,data,headers=request(hub,"GET","/frame?kind=capture&max_width=960")
    assert code==200
    assert cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR).shape==(540,960,3)
    assert headers["X-Frame-Sequence"]=="81"
    assert float(headers["X-Frame-Captured-At"])==pytest.approx(stamp,abs=1e-8)
    assert shapes==[(540,960,3)]  # A reduced request need not encode the full image.

    code,original,_=request(hub,"GET","/frame.png")
    assert code==200
    assert cv2.imdecode(np.frombuffer(original,np.uint8),cv2.IMREAD_COLOR).shape==frame.shape
    assert request(hub,"GET","/frame")[1]==original
    assert shapes==[(540,960,3),(900,1600,3)]

    hub.publish(frame=np.full_like(frame,(90,80,70)),sequence=82,captured_at=stamp+.01)
    code,new,headers=request(hub,"GET","/frame?max_width=960")
    assert code==200 and headers["X-Frame-Sequence"]=="82"
    decoded=cv2.imdecode(np.frombuffer(new,np.uint8),cv2.IMREAD_COLOR)
    assert tuple(decoded[0,0])==(90,80,70)
    assert float(headers["X-Frame-Captured-At"])==pytest.approx(stamp+.01,abs=1e-8)


def test_reduced_encoded_snapshot_preserves_aspect_and_never_upscales(hub):
    frame=np.full((601,1000,3),(90,60,30),np.uint8)
    original=cv2.imencode(".png",frame)[1].tobytes()
    hub.publish(frame_png=original,sequence=91,kind="analysis")
    code,data,headers=request(hub,"GET","/frame?kind=analysis&max_width=640")
    assert code==200 and headers["X-Frame-Sequence"]=="91"
    assert cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR).shape==(385,640,3)
    assert request(hub,"GET","/frame?kind=analysis&max_width=1920")[1]==original
    assert request(hub,"GET","/frame?kind=analysis")[1]==original


@pytest.mark.parametrize("max_width",[None,640])
def test_frame_publication_during_encoding_does_not_replace_new_generation(hub,monkeypatch,max_width):
    old_stamp=time.perf_counter()-.1
    hub.publish(frame=np.full((720,1280,3),25,np.uint8),sequence=101,captured_at=old_stamp)
    encode=cv2.imencode
    def publish_during_encode(extension,image,*args,**kwargs):
        hub.publish(frame=np.full((720,1280,3),200,np.uint8),sequence=102,captured_at=old_stamp+.01)
        return encode(extension,image,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(cv2,"imencode",publish_during_encode)
        old,meta=hub.frame(max_width=max_width)
    assert meta["sequence"]==101 and meta["captured_at"]==old_stamp
    assert int(cv2.imdecode(np.frombuffer(old,np.uint8),cv2.IMREAD_COLOR)[0,0,0])==25
    new,meta=hub.frame()
    assert meta["sequence"]==102 and meta["captured_at"]==old_stamp+.01
    decoded=cv2.imdecode(np.frombuffer(new,np.uint8),cv2.IMREAD_COLOR)
    assert decoded.shape==(720,1280,3) and int(decoded[0,0,0])==200


@pytest.mark.parametrize("query",[
    "max_width=", "max_width=319", "max_width=1921", "max_width=0",
    "max_width=-960", "max_width=960.0", "max_width=nan", "max_width=true",
    "max_width=%20960", "max_width=%EF%BC%99%EF%BC%96%EF%BC%90",
    "max_width=640&max_width=960", "kind=capture&kind=analysis&max_width=960",
    "max_width=960&path=../secret",
])
def test_invalid_frame_reduction_queries_are_rejected(hub,query):
    assert request(hub,"GET","/frame?"+query)[0]==400


@pytest.mark.parametrize("max_width",[True,False,960.0,"960",319,1921])
def test_direct_frame_reduction_rejects_invalid_width(hub,max_width):
    with pytest.raises(ValueError,match="max_width"):
        hub.frame(max_width=max_width)


def test_reduced_frames_require_the_same_authentication_and_local_origin(hub):
    route="/frame?max_width=960"
    assert request(hub,"GET",route,authenticated=False)[0]==401
    assert request(hub,"GET",route,headers={"Origin":"http://localhost"})[0]==403
    assert request(hub,"GET",route,headers={"Host":"foreign.invalid"})[0]==403


def test_events_are_bounded_actual_submissions_not_suggestions(tmp_path):
    hub=DebugHub(tmp_path,max_events=3)
    for i in range(5):
        hub.record_control(ControllerState(left_stick=StickVector(i,0)),timestamp=i,sequence=i,source="auto")
    assert [e["id"] for e in hub.events()]==[3,4,5]
    assert [e["id"] for e in hub.events(limit=1,after=3)]==[5]
    assert hub.events()[0]["state"]["left_stick"]["x"]==2


@pytest.mark.parametrize("payload",[
    {"command":[]}, {"command":"press","buttons":["INVALID"]},
    {"command":"press","buttons":["A","A"]},
    {"command":"press","buttons":["A"],"hold_ms":True},
    {"command":"press","buttons":["A"],"hold_ms":1001},
    {"command":"press","left_stick":{"x":32768}},
    {"command":"press","left_stick":{"x":True}},
    {"command":"press","left_stick":{"x":0,"z":1}},
    {"command":"press"}, {"command":"resume","buttons":["A"]},
])
def test_invalid_commands_are_rejected_without_queueing(hub,payload):
    assert request(hub,"POST","/commands",payload)[0]==400
    assert hub.status()["control"]["pending_commands"]==0


def test_close_revokes_discovery_without_clobbering_a_newer_process(tmp_path):
    first=DebugHub(tmp_path/"first")
    second=DebugHub(tmp_path/"second")
    first.start();second.start()
    try:
        assert first.endpoint["token"]!=second.endpoint["token"]
        first.close()
        shared=json.loads((tmp_path/"control_endpoint.json").read_text(encoding="utf-8"))
        assert shared["token"]==second.endpoint["token"] and shared["running"]
        assert not json.loads((tmp_path/"first/control_endpoint.json").read_text(encoding="utf-8"))["running"]
    finally:
        first.close();second.close()


def _client():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/"tools/live_client.py"
    spec=importlib.util.spec_from_file_location("raiders_live_client_test",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_client_reads_status_frame_and_queued_pause_without_printing_token(hub,tmp_path,capsys):
    client=_client()
    hub.publish({"stage":"await_menu"},frame=np.zeros((10,20,3),np.uint8),sequence=4)
    args=["--session",str(hub.session_dir)]
    assert client(args+["status"])==0
    target=tmp_path/"frame.png"
    assert client(args+["frame","--output",str(target)])==0
    assert target.read_bytes().startswith(b"\x89PNG")
    assert client(args+["--no-wait","pause"])==0
    assert hub.paused is False
    assert hub.endpoint["token"] not in capsys.readouterr().out


def test_client_does_not_send_capability_to_remote_discovery(tmp_path,capsys):
    client=_client()
    endpoint=tmp_path/"control_endpoint.json"
    endpoint.write_text(json.dumps({"running":True,"url":"http://example.invalid:80","token":"private"}),encoding="utf-8")
    assert client(["--endpoint",str(endpoint),"status"])==1
    output=capsys.readouterr().out
    assert "private" not in output and "本机HTTP" in output
