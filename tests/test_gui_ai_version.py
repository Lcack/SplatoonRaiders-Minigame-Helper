"""UI version/preview integration with mock widgets, no Tk or devices."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from raiders_helper import gui
from raiders_helper.active_ai_status import read_active_ai
from raiders_helper.capture.models import FramePacket
from raiders_helper.models import Decision, Entity, Game, Observation


class Widget:
    def __init__(self):self.value="";self.options={}
    def set(self,value):self.value=value
    def get(self):return self.value
    def configure(self,**kw):self.options.update(kw)
    def winfo_width(self):return 240
    def winfo_height(self):return 180
    def delete(self,*args):self.value=""
    def insert(self,index,value):self.value+=value
    def create_image(self,*args,**kwargs):pass


@pytest.fixture
def terminal(tmp_path,monkeypatch):
    core=tmp_path/"src/raiders_helper/mining_vision.py";core.parent.mkdir(parents=True)
    core.write_text("old=True",encoding="utf8")
    old=hashlib.sha256(core.read_bytes()).hexdigest()
    monkeypatch.setattr(gui,"_PREVIEW_VISION_SHA",old)
    session=tmp_path/"capture-session"
    data={"running":True,"updated_unix":100.,"capture_session":str(session),"pid":42,
          "loaded_version":"abcdef123456","source_hashes":{"src/raiders_helper/mining_vision.py":old},
          "profile_sha":"1"*64,"parameters_sha":"2"*64,"round":2,"reason":"保持 ZR，沿直线前进"}
    path=tmp_path/"active_ai.json"
    def write():path.write_text(json.dumps(data),encoding="utf8")
    write()
    app=gui.Dashboard.__new__(gui.Dashboard)
    app.task_terminal=True;app.running=True;app.session=session
    app._task_preview_overlay=True;app._task_preview_notice=""
    app._task_recording_topmost=False
    app.task_summary=Widget();app.task_connection_status=Widget();app.task_preview_panel=Widget()
    app.task_mode=Widget();app.task_mode.set("人工调试")
    app.task_mode_display=Widget();app.task_mode_box=Widget()
    app.canvas=Widget();app.details=Widget();app.active_source="采集卡"
    app.engine=SimpleNamespace(recognizer=SimpleNamespace(profile={"viewport":[0,0,1,1]}),armed=True)
    app._task_ai_monitor=SimpleNamespace(poll=lambda current:read_active_ai(path,current,repository_root=tmp_path,now=101.))
    status={"game":"mining","mode":"manual","screen":"playing:mining","stage":"await_menu",
            "control_enabled":True,"capture_fps_estimate":60,"recording_enabled":False,
            "recording_active":False,"completed_rounds":0,"started_rounds":0,"fit_passed":False,"output":str(session)}
    return SimpleNamespace(app=app,data=data,write=write,core=core,old=old,status=status)


def test_running_version_is_visible_and_manual_terminal_does_not_claim_to_decide(terminal):
    ctx=terminal;ctx.app._task_update_status(ctx.status,None)
    assert "实际运行版本：abcdef123456" in ctx.app.task_summary.get()
    assert "abcdef123456" in ctx.app.task_connection_status.get()
    assert "本终端：采集预览与控制接口" in ctx.app.task_summary.get()
    assert "自动流程" not in ctx.app.task_summary.get()
    assert ctx.app._task_preview_overlay


def test_disk_core_edit_marks_next_round_update_but_keeps_current_loaded_hash(terminal):
    ctx=terminal;ctx.core.write_text("next=True",encoding="utf8")
    ctx.app._task_update_status(ctx.status,None)
    assert "当前局旧版，下局自动更新" in ctx.app.task_summary.get()
    assert "实际运行版本：abcdef123456" in ctx.app.task_summary.get()
    assert ctx.app._task_preview_overlay  # Current child and GUI are both old.


def test_new_child_version_hides_stale_preview_entities_path_and_advice(terminal,monkeypatch):
    ctx=terminal;ctx.core.write_text("next=True",encoding="utf8")
    ctx.data["source_hashes"]["src/raiders_helper/mining_vision.py"]=hashlib.sha256(ctx.core.read_bytes()).hexdigest()
    ctx.data["loaded_version"]="123456abcdef";ctx.write()
    ctx.app._task_update_status(ctx.status,None)
    assert not ctx.app._task_preview_overlay
    assert "AI已更新，预览叠加未同步" in ctx.app.task_preview_panel.options["text"]
    assert "实际运行版本：123456abcdef" in ctx.app.task_summary.get()
    def forbidden(*args,**kwargs):raise AssertionError("stale preview geometry must not be drawn")
    for name in ("rectangle","line","putText","drawMarker"):
        monkeypatch.setattr(gui.cv2,name,forbidden)
    monkeypatch.setattr(gui,"advice_geometry",forbidden)
    monkeypatch.setattr(gui.ImageTk,"PhotoImage",lambda image:image.copy())
    pixels=np.full((180,240,3),(10,25,35),np.uint8)
    packet=FramePacket(pixels,1,4,60)
    observation=Observation(Game.MINING,1,4,"playing",.9,(Entity("corridor",.5,.5,.5,.05,.9),),
                            notes=("MINING实际颜色/形状原型",))
    ctx.app.render(packet,observation,Decision(diagnostics={"path":[{"x":.4,"depth":.2},{"x":.5,"depth":.3}]}))
    assert tuple(np.asarray(ctx.app.photo)[0,0])==(35,25,10)
    assert "实际颜色/形状原型" not in ctx.app.details.get()


@pytest.mark.parametrize("change",["stale","session"])
def test_invalid_heartbeat_does_not_leave_a_running_version_on_screen(terminal,change):
    ctx=terminal
    if change=="stale":ctx.data["updated_unix"]=90.
    else:ctx.data["capture_session"]="another-session"
    ctx.write();ctx.app._task_update_status(ctx.status,None)
    assert "abcdef123456" not in ctx.app.task_summary.get()
    assert "abcdef123456" not in ctx.app.task_connection_status.get()


def test_stopped_capture_clears_external_version_even_if_its_heartbeat_is_recent(terminal):
    ctx=terminal;ctx.app.running=False
    ctx.app._task_update_status(ctx.status,None,active=False)
    assert "abcdef123456" not in ctx.app.task_summary.get()
    assert "采集：已停止" in ctx.app.task_connection_status.get()


def test_external_ai_mode_and_logs_follow_heartbeat_without_changing_control_configuration(terminal):
    app=terminal.app;app._task_update_status(terminal.status,None)
    assert app.task_mode_display.get()=="MINING 自主 AI"
    assert app.task_mode.get()=="人工调试"  # Internal transport ownership is unchanged.
    assert app.task_mode_box.options["state"]=="disabled"
    logs=app._task_visible_logs([
        "[await_menu] playing:mining：调试：有时限的手动输入",
        "[await_menu] playing:mining：调试：自动控制已暂停，保持中立",
        "任务错误：模拟串口故障"],10.)
    assert "任务错误：模拟串口故障" in logs
    assert any("保持 ZR，沿直线前进" in line for line in logs)
    assert not any("手动输入" in line or "自动控制已暂停" in line for line in logs)
    assert app._task_display_reason("调试：自动控制已暂停，保持中立")=="保持 ZR，沿直线前进"
    terminal.data["updated_unix"]=90.;terminal.write();app._task_update_status(terminal.status,None)
    assert app.task_mode_display.get()=="人工调试"
    assert app.task_mode_box.options["state"]=="readonly"
    assert "输入已释放" in app._task_display_reason("调试：自动控制已暂停，保持中立")
