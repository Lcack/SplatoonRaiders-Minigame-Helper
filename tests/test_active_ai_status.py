"""Version displays follow loaded heartbeats, never the newest disk hash."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from raiders_helper.active_ai_status import ActiveAIMonitor, read_active_ai


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def ai(tmp_path):
    core=tmp_path/"src/raiders_helper/mining_vision.py"
    core.parent.mkdir(parents=True);core.write_text("old_core=True",encoding="utf8")
    terminal=tmp_path/"src/raiders_helper/session_runtime.py"
    terminal.write_text("old_terminal=True",encoding="utf8")
    session=tmp_path/"artifacts/live-one"
    data={"running":True,"updated_unix":100.,"pid":1234,"capture_session":str(session),
          "run_dir":str(tmp_path/"run"),"loaded_version":"abcdef123456",
          "source_hashes":{"src/raiders_helper/mining_vision.py":sha(core),
                           "src/raiders_helper/session_runtime.py":sha(terminal)},
          "profile_sha":"1"*64,"parameters_sha":"2"*64,"round":3,"mode":"play","reason":"playing"}
    path=tmp_path/"active_ai.json"
    def write():path.write_text(json.dumps(data),encoding="utf8")
    write()
    return SimpleNamespace(root=tmp_path,core=core,terminal=terminal,session=session,path=path,data=data,write=write,
        read=lambda **kw:read_active_ai(path,session,repository_root=tmp_path,now=101.,**kw))


def test_fresh_matching_heartbeat_displays_reported_loaded_version(ai):
    value=ai.read()
    assert value.running and value.loaded_version=="abcdef123456"
    assert value.pid==1234 and value.round_id==3
    assert "实际运行版本：abcdef123456" in value.summary_text
    assert "abcdef123456" in value.short_text
    assert value.source_hashes==ai.data["source_hashes"]


@pytest.mark.parametrize("age",[3.,10.,-1.])
def test_stale_or_future_heartbeat_never_displays_old_version(ai,age):
    value=read_active_ai(ai.path,ai.session,repository_root=ai.root,now=100.+age)
    assert not value.running and value.loaded_version is None
    assert "abcdef123456" not in value.summary_text and "过期" in value.short_text


def test_different_capture_session_cannot_claim_this_terminal_is_running_ai(ai):
    value=read_active_ai(ai.path,ai.root/"artifacts/other",repository_root=ai.root,now=101.)
    assert value.state=="session_mismatch" and not value.running
    assert value.loaded_version is None


def test_stopped_report_and_unconnected_terminal_do_not_display_old_hash(ai):
    ai.data["running"]=False;ai.write()
    assert not ai.read().running and ai.read().loaded_version is None
    value=read_active_ai(ai.path,None,repository_root=ai.root,now=101.)
    assert not value.running and value.loaded_version is None


def test_core_disk_edit_keeps_old_running_hash_until_next_child_reports_new_version(ai):
    old=ai.data["source_hashes"]["src/raiders_helper/mining_vision.py"]
    ai.core.write_text("new_core=True",encoding="utf8")
    value=ai.read()
    assert value.loaded_version=="abcdef123456" and value.state=="source_changed"
    assert "当前局旧版，下局自动更新" in value.summary_text
    assert sha(ai.core)[:12] not in value.summary_text
    assert value.source_hashes["src/raiders_helper/mining_vision.py"]==old
    ai.data.update(loaded_version="123456abcdef",updated_unix=100.5)
    ai.data["source_hashes"]["src/raiders_helper/mining_vision.py"]=sha(ai.core);ai.write()
    fresh=ai.read()
    assert fresh.state=="running" and fresh.loaded_version=="123456abcdef"
    assert "当前局旧版" not in fresh.summary_text


def test_new_child_bundle_cannot_clear_the_old_terminal_owner_restart_warning(ai):
    owner={"src/raiders_helper/session_runtime.py":sha(ai.terminal)}
    ai.terminal.write_text("new_terminal=True",encoding="utf8")
    # A fresh worker can verify current files without restarting the GUI owner.
    ai.data["source_hashes"]["src/raiders_helper/session_runtime.py"]=sha(ai.terminal);ai.write()
    value=ai.read(terminal_source_hashes=owner)
    assert value.running and value.restart_required_sources==tuple(owner)
    assert "需重启终端" in value.summary_text
    assert "下局自动更新" not in value.summary_text
    ai.data["running"]=False;ai.write()
    assert "需重启终端" in ai.read(terminal_source_hashes=owner).summary_text


def test_monitor_caches_disk_reads_but_expires_running_claim_without_waiting_next_poll(ai):
    monitor=ActiveAIMonitor(ai.path,repository_root=ai.root)
    assert monitor.poll(ai.session,now=102.9,monotonic=0).running
    value=monitor.poll(ai.session,now=103.01,monotonic=.11)
    assert not value.running and value.loaded_version is None


def test_monitor_rechecks_source_once_per_second_and_invalidates_session_immediately(ai):
    monitor=ActiveAIMonitor(ai.path,repository_root=ai.root)
    assert monitor.poll(ai.session,now=100.1,monotonic=0).state=="running"
    ai.core.write_text("updated=True",encoding="utf8")
    assert monitor.poll(ai.session,now=100.5,monotonic=.4).state=="running"
    assert monitor.poll(ai.session,now=101.2,monotonic=1.1).state=="source_changed"
    assert not monitor.poll(ai.root/"new-session",now=101.3,monotonic=1.2).running


@pytest.mark.parametrize("bad",["{broken", "[]", "null"])
def test_bad_json_does_not_crash_the_capture_ui(ai,bad):
    ai.path.write_text(bad,encoding="utf8")
    assert not ai.read().running


@pytest.mark.parametrize("relative",["../outside.py","E:/outside.py"])
def test_unsafe_source_paths_are_not_read_or_shown_as_current(ai,relative):
    ai.data["source_hashes"]={relative:"0"*64};ai.write()
    value=ai.read()
    assert not value.running and value.loaded_version is None


def test_missing_heartbeat_is_not_running(ai):
    ai.path.unlink()
    assert not ai.read().running
