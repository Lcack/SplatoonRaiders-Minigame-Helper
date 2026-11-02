"""Clocked release is independent of camera updates; no hardware is opened."""
import importlib.util
from pathlib import Path
import threading
import time

from raiders_helper.control.buttons import Button,ControllerState,StickVector

spec=importlib.util.spec_from_file_location("mining_supervisor_test",Path(__file__).parents[1]/"tools/mining_supervisor.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Client:
    def __init__(self):self.commands=[];self.released=threading.Event()
    def command(self,payload):
        self.commands.append((time.perf_counter(),payload))
        if payload.get("command")=="state" and payload["left_stick"]["x"]==0:self.released.set()


def test_deadline_releases_axis_without_waiting_for_another_frame_and_preserves_mode():
    client=Client();lease=module.LatestLease(client)
    state=ControllerState(frozenset({Button.ZR}),StickVector(12000,0))
    deadline=time.perf_counter()+.10
    lease.update(state,release_at=deadline,release_state=ControllerState(frozenset({Button.ZR})))
    lease.thread.start()
    try:
        assert client.released.wait(.22)
        release_time,payload=next((t,p) for t,p in client.commands if p.get("command")=="state" and p["left_stick"]["x"]==0)
        assert release_time>=deadline
        assert payload["buttons"]==["ZR"]
        assert any(p.get("left_stick",{}).get("x")==12000 for _,p in client.commands)
    finally:lease.close()
    assert client.commands[-1][1]["command"]=="release"


def test_expired_deadline_never_resurrects_old_hold_on_a_late_update():
    client=Client();lease=module.LatestLease(client)
    lease.update(ControllerState(left_stick=StickVector(12000,0)),release_at=time.perf_counter()-.01)
    lease.thread.start()
    try:time.sleep(.035)
    finally:lease.close()
    assert not any(p.get("left_stick",{}).get("x",0) for _,p in client.commands)
