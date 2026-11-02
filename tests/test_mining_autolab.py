"""Mock scheduling/traceability checks; these tests never connect hardware."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


@pytest.fixture
def lab(tmp_path, monkeypatch):
    source=Path(__file__).resolve().parents[1]/"tools/mining_autolab.py"
    spec=importlib.util.spec_from_file_location("mining_autolab_test",source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config=tmp_path/"config.json"
    profile=tmp_path/"profile.json"
    config.write_text(json.dumps({"enabled":True,"revision":1}),encoding="utf8")
    profile.write_text(json.dumps({"game":"mining","revision":1}),encoding="utf8")
    state=SimpleNamespace(module=module,config=config,profile=profile,output=tmp_path/"output",
                          clients=[],calls=[],release_calls=[],secret="TEST_CLIENT_TOKEN_NOT_FOR_LOGS")
    class FakeClient:
        def __init__(self,endpoint,frame_width):
            assert frame_width==960
            self.headers={"X-Raiders-Token":state.secret}
            state.clients.append(self)
        def command(self,payload,wait=False):
            state.release_calls.append((payload,wait))
            return {"status":"succeeded"}
    def collect(client,prof,conf,output,round_id,mode,max_seconds):
        assert client is state.clients[0]
        output.mkdir(parents=True,exist_ok=False)
        (output/"trace.jsonl").write_text("",encoding="utf8")
        call={"round_id":round_id,"mode":mode,"output":output,"config":dict(conf),"profile":dict(prof)}
        state.calls.append(call)
        summary={"round_id":round_id,"mode":mode,"result":"result:retry",
                 "actual_commands":str(tmp_path/"session/commands.jsonl"),
                 "source_session":str(tmp_path/"session"),"seconds":40.,"frames":500,"playing_frames":470,
                 "events":[{"type":"completed"},{"type":"aborted"}],
                 "arbitrary_private_headers":client.headers}
        (output/"summary.json").write_text(json.dumps({k:v for k,v in summary.items() if k!="arbitrary_private_headers"}),encoding="utf8")
        return summary
    state.collect=collect
    monkeypatch.setattr(module,"LiveClient",FakeClient)
    monkeypatch.setattr(module,"collect_round",collect)
    monkeypatch.setattr(module,"MiningLiveRecognizer",lambda *args:None)
    monkeypatch.setattr(module.time,"sleep",lambda seconds:None)
    state.args=lambda *extra:["--endpoint",str(tmp_path/"endpoint.json"),"--config",str(config),
                            "--profile",str(profile),"--output",str(state.output),*extra]
    return state


def test_batches_split_without_resetting_global_rounds_or_modes(lab):
    assert lab.module.main(lab.args("--rounds","7","--batch-size","3"))==0
    assert [call["round_id"] for call in lab.calls]==list(range(1,8))
    assert [call["mode"] for call in lab.calls]==["normal","zr","normal","zr","normal","zr","normal"]
    batches=sorted(lab.output.glob("batch-*"))
    assert [len(list(path.glob("round-*/trace.jsonl"))) for path in batches]==[3,3,1]
    assert lab.calls[3]["output"]==lab.output/"batch-0002/round-01"
    assert [len(read(path/"summary.json")["rounds"]) for path in batches]==[3,3,1]
    assert all(read(path/"summary.json")["status"]=="collected_not_fitted" for path in batches)
    current=read(lab.output/"current.json")
    assert current==read(lab.output/"summary.json")
    assert current["completed_rounds"]==7 and current["completed_batches"]==2
    assert current["status"]=="collected_not_fitted" and current["current_round"] is None
    assert current["release_succeeded"] and current["automatic_parameter_changes"] is False
    assert lab.release_calls==[({"command":"release"},True)]
    events=[json.loads(line) for line in (lab.output/"experiment-index.jsonl").read_text(encoding="utf8").splitlines()]
    assert len(events)==14
    assert [row["event"] for row in events]==["round_started","round_completed"]*7
    assert all("headers" not in row for row in events)
    assert lab.secret not in "\n".join(path.read_text(encoding="utf8") for path in lab.output.rglob("*.json*"))


def test_default_is_six_rounds_in_one_analysis_compatible_batch(lab):
    assert lab.module.main(lab.args())==0
    assert len(lab.calls)==6
    batches=list(lab.output.glob("batch-*"))
    assert len(batches)==1 and len(list(batches[0].glob("round-*/trace.jsonl")))==6
    assert read(lab.output/"summary.json")["completed_batches"]==1


def test_inputs_are_reloaded_and_exact_bytes_are_snapshotted_before_each_round(lab,monkeypatch):
    original_config=lab.config.read_bytes()
    original_profile=lab.profile.read_bytes()
    versions=[]
    def collect(*args):
        output=args[3]
        inputs=output.parent/"inputs"/output.name
        versions.append(read(inputs/"manifest.json"))
        result=lab.collect(*args)
        if len(lab.calls)==1:
            lab.config.write_text('{"enabled": true, "revision": 2}\n',encoding="utf8")
            lab.profile.write_text('{"game": "mining", "revision": 3}\n',encoding="utf8")
        return result
    monkeypatch.setattr(lab.module,"collect_round",collect)
    assert lab.module.main(lab.args("--rounds","2"))==0
    assert [call["config"]["revision"] for call in lab.calls]==[1,2]
    assert [call["profile"]["revision"] for call in lab.calls]==[1,3]
    for name,content in [("config",original_config),("profile",original_profile)]:
        first=versions[0]["inputs"][name]
        assert Path(first["snapshot"]).read_bytes()==content
        assert first["sha256"]==hashlib.sha256(content).hexdigest()
        second=versions[1]["inputs"][name]
        assert second["sha256"]!=first["sha256"]
        assert second["sha256"]==hashlib.sha256(Path(second["snapshot"]).read_bytes()).hexdigest()


def test_startup_code_manifest_is_frozen_and_code_changes_require_restart(lab,monkeypatch):
    source=lab.config.parent/"already-loaded.py"
    source.write_text("old_source = True\n",encoding="utf8")
    old_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    frozen={"files":[{"path":str(source),"sha256":old_hash}],
            "restart_required_for_code_changes":True,"scope":"synthetic startup manifest fixture"}
    monkeypatch.setattr(lab.module,"_STARTUP_SOURCE_VERSIONS",frozen)
    def forbid_rehash():
        raise AssertionError("must not pretend a later source edit was imported")
    monkeypatch.setattr(lab.module,"_startup_source_versions",forbid_rehash)
    def collect(*args):
        result=lab.collect(*args)
        source.write_text("new_source = True\n",encoding="utf8")
        return result
    monkeypatch.setattr(lab.module,"collect_round",collect)
    assert lab.module.main(lab.args("--rounds","2"))==0
    saved=read(lab.output/"loaded-sources.json")
    assert saved==frozen and saved["files"][0]["sha256"]!=hashlib.sha256(source.read_bytes()).hexdigest()
    assert read(lab.output/"current.json")["code_changes_require_restart"] is True
    for index in (1,2):
        manifest=read(lab.output/f"batch-0001/inputs/round-{index:02d}/manifest.json")
        assert manifest["code_changes_require_restart"] is True
        assert Path(manifest["loaded_source_manifest"])==lab.output/"loaded-sources.json"


def test_disabling_config_between_rounds_stops_before_another_start(lab,monkeypatch):
    def collect(*args):
        result=lab.collect(*args)
        lab.config.write_text('{"enabled": false}',encoding="utf8")
        return result
    monkeypatch.setattr(lab.module,"collect_round",collect)
    assert lab.module.main(lab.args("--rounds","6"))==1
    assert len(lab.calls)==1 and len(lab.release_calls)==1
    current=read(lab.output/"current.json")
    assert current["completed_rounds"]==1 and current["failure"]["stage"]=="validate_inputs"
    assert current["current_round"]["round_id"]==2
    assert read(lab.output/"batch-0001/inputs/round-02/config.json")=={"enabled":False}


def test_unlimited_mode_stops_on_exception_releases_and_does_not_log_payload(lab,monkeypatch,capsys):
    def collect(*args):
        if args[4]==4:
            raise RuntimeError(lab.secret)
        return lab.collect(*args)
    monkeypatch.setattr(lab.module,"collect_round",collect)
    assert lab.module.main(lab.args("--rounds","0","--batch-size","3"))==1
    assert len(lab.calls)==3 and len(lab.release_calls)==1
    state=read(lab.output/"current.json")
    assert state["status"]=="failed" and state["completed_rounds"]==3
    assert state["failure"]=={"stage":"collect_round","error_type":"RuntimeError"}
    assert read(lab.output/"batch-0001/summary.json")["status"]=="collected_not_fitted"
    assert read(lab.output/"batch-0002/summary.json")["status"]=="failed"
    assert lab.secret not in capsys.readouterr().err
    assert lab.secret not in "\n".join(path.read_text(encoding="utf8") for path in lab.output.rglob("*.json*"))


@pytest.mark.parametrize("result",[None,"unknown","playing:mining"])
def test_unconfirmed_result_never_advances_to_another_retry(lab,monkeypatch,result):
    def collect(*args):
        summary=lab.collect(*args);summary["result"]=result
        return summary
    monkeypatch.setattr(lab.module,"collect_round",collect)
    assert lab.module.main(lab.args("--rounds","3"))==1
    assert len(lab.calls)==1 and len(lab.release_calls)==1
    state=read(lab.output/"summary.json")
    assert state["completed_rounds"]==0 and state["failure"]["stage"]=="confirm_result"


def test_ctrl_c_stops_continuous_mode_and_releases(lab,monkeypatch):
    def collect(*args):
        if args[4]==2:raise KeyboardInterrupt
        return lab.collect(*args)
    monkeypatch.setattr(lab.module,"collect_round",collect)
    assert lab.module.main(lab.args("--rounds","0"))==130
    state=read(lab.output/"current.json")
    assert state["status"]=="interrupted" and state["completed_rounds"]==1
    assert state["release_succeeded"] and len(lab.release_calls)==1


def test_stop_request_after_round_preserves_data_and_never_starts_next_round(lab,monkeypatch):
    def collect(*args):
        result=lab.collect(*args)
        (lab.output/"stop.request").touch()
        return result
    monkeypatch.setattr(lab.module,"collect_round",collect)
    assert lab.module.main(lab.args("--rounds","0"))==0
    state=read(lab.output/"current.json")
    assert state["status"]=="stopped_by_request" and state["completed_rounds"]==1
    assert state["stop_observed_boundary"]=="after_round"
    assert len(lab.calls)==1 and lab.release_calls==[({"command":"release"},True)]
    assert read(lab.calls[0]["output"]/"summary.json")["result"]=="result:retry"
    assert read(lab.output/"batch-0001/summary.json")["rounds"][0]["segments_completed"]==1


def test_stop_request_during_between_round_wait_is_seen_before_next_start(lab,monkeypatch):
    monkeypatch.setattr(lab.module.time,"sleep",lambda seconds:(lab.output/"stop.request").touch())
    assert lab.module.main(lab.args("--rounds","0"))==0
    state=read(lab.output/"current.json")
    assert state["status"]=="stopped_by_request" and state["completed_rounds"]==1
    assert state["stop_observed_boundary"]=="before_round"
    assert len(lab.calls)==1 and len(lab.release_calls)==1


def test_failure_to_release_is_reported_even_after_successful_collection(lab,monkeypatch):
    def fail_release(self,payload,wait=False):
        lab.release_calls.append((payload,wait))
        raise RuntimeError(lab.secret)
    # The fake client type is only instantiated inside run_autolab.
    monkeypatch.setattr(lab.module.LiveClient,"command",fail_release)
    assert lab.module.main(lab.args("--rounds","1"))==1
    state=read(lab.output/"current.json")
    assert state["completed_rounds"]==1 and state["release_succeeded"] is False
    assert state["failure"]=={"stage":"final_release","error_type":"RuntimeError"}


def test_bad_inputs_are_preserved_but_never_create_a_controller(lab):
    lab.config.write_text("{broken JSON",encoding="utf8")
    assert lab.module.main(lab.args("--rounds","1"))==1
    state=read(lab.output/"current.json")
    assert state["failure"]["stage"]=="snapshot_inputs"
    assert state["release_attempted"] is False and not lab.clients and not lab.calls
    snapshot=lab.output/"batch-0001/inputs/round-01/config.json"
    assert snapshot.read_bytes()==lab.config.read_bytes()


def test_existing_output_is_not_overwritten_even_when_empty(lab):
    lab.output.mkdir()
    with pytest.raises(SystemExit) as error:lab.module.main(lab.args())
    assert error.value.code==2 and not list(lab.output.iterdir()) and not lab.clients


@pytest.mark.parametrize("extra",[
    ["--rounds","-1"],["--batch-size","0"],["--seconds-per-round","0"],
    ["--seconds-per-round","nan"],["--seconds-per-round","inf"],
])
def test_invalid_cli_values_do_not_create_output_or_client(lab,extra):
    with pytest.raises(SystemExit) as error:lab.module.main(lab.args(*extra))
    assert error.value.code==2 and not lab.output.exists() and not lab.clients
