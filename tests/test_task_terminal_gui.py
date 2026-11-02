"""Exercise the reused Tk window with an in-process simulated session."""
import builtins
import threading
import time
import tkinter as tk

import numpy as np
import pytest

from raiders_helper import session_runtime
from raiders_helper.capture.models import FramePacket
from raiders_helper.control.buttons import Button, ControllerState
from raiders_helper.debug_api import DebugHub
from raiders_helper.gui import Dashboard
from raiders_helper.models import Decision, Entity, Game, Observation


def pump(root, condition, timeout=3):
    deadline=time.perf_counter()+timeout
    while not condition():
        assert time.perf_counter()<deadline,"GUI did not reach expected state"
        root.update()
        time.sleep(.005)
    root.update()


@pytest.fixture
def window():
    root=tk.Tk();root.withdraw()
    app=Dashboard(root,task_terminal=True)
    app.task_mode.set(app.TASK_INTERFACE_MODE)
    yield root,app
    app._task_stop.set()
    if app._task_thread:
        app._task_thread.join(timeout=2)
    root.destroy()


def test_task_window_runs_selected_game_without_console_and_shares_debug_ownership(window,tmp_path,monkeypatch):
    root,app=window
    received=[];states=[]
    def no_console(*args,**kwargs):raise AssertionError("GUI must never call input()")
    monkeypatch.setattr(builtins,"input",no_console)
    def fake_session(args,*,stop_event,on_log,on_update,on_hub):
        received.append(args)
        hub=DebugHub(tmp_path,control_allowed=True)
        on_hub(hub);on_log("GUI 后台日志已接入")
        try:
            i=0
            while not stop_event.is_set():
                i+=1;now=time.perf_counter()
                manual=hub.take_manual(now)
                decision=manual or Decision(reason="正在观察")
                states.append(decision.state)
                hub.record_control(decision.state,timestamp=now,sequence=i,source="test")
                packet=FramePacket(np.zeros((180,320,3),np.uint8),now,i,60)
                obs=Observation(Game(args.game),now,i,"playing",.9,(Entity("player",.5,.7,.1,.1,.9),))
                on_update(packet,obs,decision,{"output":str(args.output),"game":args.game,"mode":args.mode,
                    "screen":"playing:hopping","stage":"await_menu","control_enabled":True,
                    "started_rounds":0,"completed_rounds":0,"capture_fps_estimate":60,"fit_passed":False,
                    "reason":decision.reason})
                stop_event.wait(.01)
            return 0
        finally:
            hub.close();on_hub(None)
    monkeypatch.setattr(session_runtime,"run_session",fake_session)
    app.game.set("hopping")
    app._task_start_session()
    pump(root,lambda:app.packet is not None)
    assert received[0].game=="hopping" and received[0].mode=="manual"
    assert received[0].no_debug is False and received[0].no_record is False
    assert "GUI 后台日志已接入" in app.task_log.get("1.0","end")
    assert app.engine.armed
    app._task_press("A")
    pump(root,lambda:any(Button.A in state.buttons for state in states))
    pump(root,lambda:states[-1]==ControllerState.neutral())
    assert app._task_hub.paused
    app.stop()
    pump(root,lambda:not app.running)
    assert not app._task_thread.is_alive()
    assert app._task_hub is None and not app.engine.armed
    assert str(app.task_start.cget("state"))=="normal"
    assert app._task_latest is None


def test_unverified_calibration_is_reported_in_window_before_connecting(window,monkeypatch):
    root,app=window
    called=[]
    monkeypatch.setattr(session_runtime,"run_session",lambda *args,**kwargs:called.append(True))
    app.task_mode.set("Retry 参数标定")
    app._task_start_session()
    root.update()
    assert not called and not app.running
    assert "无法启动" in app.status.get()


def all_widgets(widget):
    for child in widget.winfo_children():
        yield child
        yield from all_widgets(child)


def test_connection_rows_have_real_actions_and_virtual_keys_are_not_on_main_page(window):
    root,app=window
    buttons={str(widget.cget("text")):widget for widget in all_widgets(root)
             if widget.winfo_class()=="TButton"}
    assert {"启动采集","连接控制器","刷新串口","手柄调试"} <= buttons.keys()
    assert buttons["启动采集"].master != buttons["连接控制器"].master
    assert buttons["连接控制器"].master == buttons["手柄调试"].master
    assert not {"A","B","ZL","ZR","↑","↓","左杆←","右杆→"} & buttons.keys()
    app.task_mode.set("Retry 参数标定")
    assert "探针" in app.task_mode_hint.get() and "Landing" in app.task_mode_hint.get()
    app.task_mode.set("仅定位菜单")
    assert "不确认开局" in app.task_mode_hint.get()


def test_capture_and_connect_buttons_request_distinct_manual_sessions(window,monkeypatch):
    root,app=window
    requested=[]
    monkeypatch.setattr(app,"_task_start_session",lambda settings=None:requested.append(settings))
    app.task_mode.set("Retry 参数标定")
    app._task_start_capture()
    app._task_connect_controller()
    assert [item["mode"] for item in requested]==[app.TASK_INTERFACE_MODE,app.TASK_INTERFACE_MODE]
    assert [item["control"] for item in requested]==[False,True]
    assert app.task_mode.get()=="Retry 参数标定"


def test_switch_waits_for_release_and_uses_the_requested_settings(window,monkeypatch):
    root,app=window
    released=threading.Event();calls=[]
    class OldTask:
        def is_alive(self):return not released.is_set()
        def join(self,timeout=None):pass
    app._task_thread=OldTask();app.running=True
    def finish_old():
        released.set();app.running=False
    def start_new(settings=None):calls.append(settings)
    monkeypatch.setattr(app,"_task_start_session",start_new)
    app.task_mode.set("Retry 参数标定");app.game.set("juggling")
    app._task_request_session()
    assert not calls and app._task_stop.is_set()
    app.game.set("landing")
    root.after(50,finish_old)
    pump(root,lambda:bool(calls))
    assert calls[0]["game"]=="juggling" and calls[0]["mode"]=="Retry 参数标定"


def test_stop_cancels_a_pending_task_switch(window,monkeypatch):
    root,app=window
    class OldTask:
        alive=True
        def is_alive(self):return self.alive
        def join(self,timeout=None):pass
    old=OldTask();app._task_thread=old;app.running=True;calls=[]
    monkeypatch.setattr(app,"_task_start_session",lambda settings=None:calls.append(settings))
    app._task_request_session()
    app.stop()
    old.alive=False;app.running=False
    root.after(80,lambda:None)
    until=time.perf_counter()+.12
    while time.perf_counter()<until:
        root.update();time.sleep(.005)
    assert not calls and app._task_pending_start is None


def test_controller_debug_is_single_separate_window_and_closes_before_task(window,monkeypatch):
    root,app=window
    made=[];order=[]
    class FakeDebug:
        def __init__(self,master,hub_provider):
            self.root=object();self.closed=False;self.shown=0;self.provider=hub_provider
            made.append(self)
            assert master is root
        def show(self):self.shown+=1
        def close(self):self.closed=True;order.append("debug")
    monkeypatch.setattr("raiders_helper.control.live_debug_ui.LiveDebugWindow",FakeDebug)
    app.controller_debug();app.controller_debug()
    assert len(made)==1 and made[0].shown==1
    assert made[0].provider() is app._task_hub
    monkeypatch.setattr(app,"_task_stop_session",lambda:order.append("task"))
    monkeypatch.setattr(app,"_task_wait_close",lambda:order.append("wait"))
    app.close()
    assert order==["debug","task","wait"]


def test_autonomous_juggling_starts_its_driver_after_interface_is_ready(window,tmp_path,monkeypatch):
    root,app=window;calls=[]
    def fake_session(args,*,stop_event,on_log,on_update,on_hub):
        hub=DebugHub(tmp_path,control_allowed=True);on_hub(hub)
        try:
            while not stop_event.is_set():
                now=time.perf_counter();packet=FramePacket(np.zeros((180,320,3),np.uint8),now,1,60)
                obs=Observation(Game.JUGGLING,now,1,'unknown')
                on_update(packet,obs,Decision(),{'output':str(args.output),'game':'juggling','mode':'manual',
                    'screen':'result:retry','stage':'await_menu','control_enabled':True,'started_rounds':0,
                    'completed_rounds':0,'capture_fps_estimate':60,'fit_passed':False,'reason':'接口待命'})
                stop_event.wait(.02)
        finally:hub.close();on_hub(None)
        return 0
    monkeypatch.setattr(session_runtime,'run_session',fake_session)
    monkeypatch.setattr(app,'_task_launch_autoplay',lambda *args:calls.append(args))
    app.game.set('juggling');app.task_mode.set(app.TASK_AUTO_MODE);app._task_start_session()
    pump(root,lambda:bool(calls))
    assert calls[0][0]=='juggling' and calls[0][1].name=='juggling.live.local.json'
    assert calls[0][2]==3
    assert '人工' not in app.task_mode_display.get()


def test_prototype_game_does_not_claim_to_support_autonomous_live_ai(window):
    root,app=window;app.game.set('hopping');app.task_mode.set(app.TASK_AUTO_MODE)
    app._task_start_session()
    assert not app.running and 'MINING/JUGGLING' in app.status.get()
