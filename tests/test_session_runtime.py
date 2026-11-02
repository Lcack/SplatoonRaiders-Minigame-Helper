import cv2
import numpy as np
import pytest

from raiders_helper.control.buttons import Button, ControllerState
from raiders_helper.models import Decision, Game, Observation
from raiders_helper.session_runtime import _IdleTraceGate, ScreenMatcher, authorize_session, select_game


def anchor(tmp_path, name, seed):
    data=np.random.default_rng(seed).integers(0,256,(10,20),dtype=np.uint8)
    (tmp_path/name).write_bytes(cv2.imencode(".png",data)[1].tobytes())
    return data


def test_screen_requires_cursor_and_label_and_resets_on_missing_anchor(tmp_path):
    cursor=anchor(tmp_path,"cursor.png",1);label=anchor(tmp_path,"label.png",2)
    frame=np.zeros((100,200,3),np.uint8)
    frame[10:20,10:30]=cv2.cvtColor(cursor,cv2.COLOR_GRAY2BGR)
    frame[10:20,60:80]=cv2.cvtColor(label,cv2.COLOR_GRAY2BGR)
    config={"reference_size":[200,100],"stable_frames":2,"screens":[{"name":"menu:mining","all_of":[
        {"path":"cursor.png","roi":[0,0,0.25,0.4],"threshold":.95},
        {"path":"label.png","roi":[.25,0,.25,.4],"threshold":.95}]}]}
    matcher=ScreenMatcher(config,tmp_path)
    assert matcher.observe(frame)[0]=="unknown" and matcher.has_ui_candidates
    assert matcher.observe(frame)[0]=="menu:mining"
    frame[10:20,60:80]=0
    assert matcher.observe(frame)[0]=="unknown" and not matcher.has_ui_candidates


def test_two_matching_screen_candidates_are_ambiguous(tmp_path):
    pattern=anchor(tmp_path,"x.png",5)
    frame=cv2.cvtColor(pattern,cv2.COLOR_GRAY2BGR)
    matcher=ScreenMatcher({"stable_frames":1,"reference_size":[20,10],"screens":[
        {"name":name,"all_of":[{"path":"x.png","threshold":.99}]} for name in ("menu:mining","menu:hopping")
    ]},tmp_path)
    assert matcher.observe(frame)[0]=="unknown"
    assert matcher.has_ui_candidates


def test_advisory_game_profile_cannot_use_single_control_but_can_test_menu():
    session={"calibrated":True,"screens":[1]}
    profile={"calibrated":True,"advisory_only":True}
    with pytest.raises(ValueError,match="游戏 profile"):
        authorize_session(True,"single",session,profile,"COM1")
    authorize_session(True,"menu",session,profile,"COM1")


def test_uncalibrated_menu_cannot_send_inputs():
    with pytest.raises(ValueError,match="菜单模板"):
        authorize_session(True,"menu",{"calibrated":False},{},"COM1")
    authorize_session(False,"single",{}, {}, None)


def test_verified_single_still_requires_controller_port():
    with pytest.raises(ValueError,match="--port"):
        authorize_session(True,"single",{"calibrated":True,"screens":[1]},
                          {"calibrated":True},None)


@pytest.mark.parametrize("value,expected",[("1","mining"),("4","landing"),("juggling","juggling")])
def test_terminal_game_choices(value,expected):
    assert select_game(value)==expected


def idle_context(phase="unknown", screen="menu:mining", state=None, **diagnostics):
    observation=Observation(Game.MINING,10,1,phase)
    decision=Decision(state or ControllerState.neutral(),"same decision",diagnostics=diagnostics)
    status={"screen":screen,"stage":"await_menu","source":"manual",
            "started_rounds":0,"completed_rounds":0}
    return observation,decision,status


def test_unchanged_idle_logs_at_one_hz_despite_timing_and_score_jitter():
    gate=_IdleTraceGate()
    obs,decision,status=idle_context(timing={"analysis_ms":10})
    assert gate.should_write("menu:mining",obs,decision,status,0)
    obs,decision,status=idle_context(timing={"analysis_ms":15})
    assert not gate.should_write("menu:mining",obs,decision,status,.01)
    assert not gate.should_write("menu:mining",obs,decision,status,.999)
    assert gate.should_write("menu:mining",obs,decision,status,1)


def test_idle_input_changes_and_session_events_are_logged_immediately():
    gate=_IdleTraceGate()
    obs,decision,status=idle_context()
    assert gate.should_write("menu:mining",obs,decision,status,0)
    _,pressed,_=idle_context(state=ControllerState(frozenset({Button.A})))
    assert gate.should_write("menu:mining",obs,pressed,status,.01)
    assert not gate.should_write("menu:mining",obs,pressed,status,.02)
    assert gate.should_write("menu:mining",obs,decision,status,.03)
    _,event,_=idle_context(session={"events":[{"type":"round_completed","round":1}]})
    assert gate.should_write("result:retry",obs,event,status,.04)


def test_gameplay_and_unknown_after_play_remain_full_rate_when_control_is_paused():
    gate=_IdleTraceGate()
    for i,phase in enumerate(("countdown","playing","unknown","unknown","playing")):
        obs,decision,status=idle_context(phase=phase,screen="unknown",debug={"paused":True})
        assert gate.should_write("unknown",obs,decision,status,i*.01)
    obs,decision,status=idle_context(phase="playing",screen="unknown",debug={"paused":True})
    # A first raw result hit wins over residual gameplay HUD recognition.
    assert gate.should_write("result:retry",obs,decision,status,.05)
    assert not gate.should_write("result:retry",obs,decision,status,.06)


@pytest.mark.parametrize("phase",["menu","result","result_pending","game_over","paused","overlay"])
def test_explicit_inactive_phase_ends_full_rate_logs(phase):
    gate=_IdleTraceGate()
    obs,decision,status=idle_context(phase="playing",screen="unknown")
    assert gate.should_write("unknown",obs,decision,status,0)
    obs,decision,status=idle_context(phase=phase,screen="unknown")
    assert gate.should_write("unknown",obs,decision,status,.01)
    assert not gate.should_write("unknown",obs,decision,status,.02)
