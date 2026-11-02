"""Version boundaries and fresh imports; these tests never control an NS."""
import argparse
import importlib
import json
import os
from pathlib import Path
import py_compile
import shutil
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
hot = importlib.import_module("mining_hot_reload")
worker = importlib.import_module("mining_round_worker")


def inputs(directory):
    profile, parameters = directory / "profile-live.json", directory / "parameters-live.json"
    profile.write_text(json.dumps({"game": "mining", "phase_anchors": False, "stable_frames": 1}), encoding="utf-8")
    parameters.write_text('{"maximum_axis": 0.65}', encoding="utf-8")
    return profile, parameters


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / "source"
    (root / "src/raiders_helper").mkdir(parents=True)
    (root / "tools").mkdir()
    for relative in hot.TOOL_SOURCES:
        (root / relative).write_text("# test source\n", encoding="utf-8")
    (root / "src/raiders_helper/mining_motion.py").write_text("VERSION = 'v1'\n", encoding="utf-8")
    return root


def args_for(directory, rounds=2):
    profile, parameters = inputs(directory)
    return argparse.Namespace(profile=profile, parameters=parameters, endpoint=directory / "unused-endpoint.json",
        output=directory / "output", rounds=rounds, seconds_per_round=600)


class FakeClient:
    def __init__(self):
        self.commands = []
    def request(self, route):
        assert route == "/status"
        return {"output": "capture-one"}
    def command(self, payload, wait=False):
        self.commands.append(payload)
        return {"status": "succeeded"}


def complete_fake_worker(path, manifest):
    directory = Path(manifest["round_directory"])
    hot.atomic_json(directory / "loaded-version.json", {"version_id": manifest["version_id"], "pid": manifest["round"]})
    hot.atomic_json(directory / "worker-result.json", {"version_id": manifest["version_id"], "status": "completed",
        "report": {"result": "result:retry"}})
    return 0


def test_each_round_snapshots_latest_inputs_and_source_without_ai_import_or_restart(tmp_path, source_tree):
    args = args_for(tmp_path)
    client = FakeClient(); versions = []
    def launcher(path, manifest):
        _, value = hot.read_object(manifest["inputs"]["parameters"]["snapshot"])
        versions.append((manifest["source_version"], value["maximum_axis"]))
        code = complete_fake_worker(path, manifest)
        if len(versions) == 1:
            args.parameters.write_text('{"maximum_axis": 0.5}', encoding="utf-8")
            (source_tree / "src/raiders_helper/mining_motion.py").write_text("VERSION = 'v2'\n", encoding="utf-8")
        return code
    summary = hot.run_coordinator(args, source_root=source_tree, client_factory=lambda _: client, launcher=launcher)
    assert summary["completed_rounds"] == 2
    assert summary["status"] == "completed"
    assert versions[0][0] != versions[1][0]
    assert [v[1] for v in versions] == [.65, .5]
    assert client.commands == [{"command": "release"}] * 3
    assert summary["rounds"][0]["config_version"] != summary["rounds"][1]["config_version"]


@pytest.mark.parametrize("fault", ["parameters_json", "profile_shape", "source_syntax"])
def test_bad_snapshot_stops_before_launch_and_releases(tmp_path, source_tree, fault):
    args = args_for(tmp_path);client = FakeClient(); launched = []
    if fault == "parameters_json":args.parameters.write_text('{"maximum_axis":', encoding="utf-8")
    if fault == "profile_shape":args.profile.write_text('[]', encoding="utf-8")
    if fault == "source_syntax":(source_tree / "src/raiders_helper/mining_motion.py").write_text('def broken(:', encoding="utf-8")
    with pytest.raises((ValueError, SyntaxError)):
        hot.run_coordinator(args, source_root=source_tree, client_factory=lambda _: client,
                            launcher=lambda *a: launched.append(a))
    assert launched == []
    assert client.commands[-1] == {"command": "release"}
    assert json.loads((args.output / "summary.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_existing_output_is_never_overwritten_or_connected(tmp_path, source_tree):
    args = args_for(tmp_path);args.output.mkdir()
    sentinel = args.output / "untouched.txt";sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        hot.run_coordinator(args, source_root=source_tree, client_factory=lambda _: pytest.fail("must not connect"))
    assert sentinel.read_text() == "keep"


def test_continuous_mode_stops_only_at_next_round_boundary(tmp_path, source_tree):
    args = args_for(tmp_path, rounds=0);client = FakeClient(); launches = []
    def launcher(path, manifest):
        launches.append(path)
        (args.output / "stop.request").write_text("stop after this round")
        return complete_fake_worker(path, manifest)
    summary = hot.run_coordinator(args, source_root=source_tree, client_factory=lambda _: client, launcher=launcher)
    assert len(launches) == summary["completed_rounds"] == 1
    assert summary["status"] == "stopped_at_round_boundary"
    assert client.commands[-1] == {"command": "release"}


def test_worker_failure_does_not_retry_or_launch_another_round(tmp_path, source_tree):
    args = args_for(tmp_path);client = FakeClient();launches = []
    def launcher(path, manifest):
        launches.append(path)
        return 7
    with pytest.raises(RuntimeError, match="exit=7"):
        hot.run_coordinator(args, source_root=source_tree, client_factory=lambda _: client, launcher=launcher)
    assert len(launches) == 1
    assert client.commands[-1] == {"command": "release"}


def make_manifest(tmp_path, source_tree):
    profile, parameters = inputs(tmp_path)
    return hot.snapshot_round(source_tree, tmp_path / "round", profile, parameters,
                              tmp_path / "endpoint-does-not-exist.json", 600, 1, "capture-one")


def test_source_edit_between_ready_and_A_blocks_A(tmp_path, source_tree):
    manifest = make_manifest(tmp_path, source_tree);client = FakeClient()
    guarded = worker.StartVerifiedClient(client, manifest)
    (source_tree / "src/raiders_helper/mining_motion.py").write_text("VERSION = 'v2'\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="源码"):
        guarded.command({"command": "press", "buttons": ["A"]}, wait=True)
    assert client.commands == []
    assert not (tmp_path / "round/start-verified.json").exists()


def test_capture_session_change_blocks_A(tmp_path, source_tree):
    manifest = make_manifest(tmp_path, source_tree);client = FakeClient()
    client.request = lambda _: {"output": "different-session"}
    with pytest.raises(RuntimeError, match="采集会话"):
        worker.StartVerifiedClient(client, manifest).command({"command": "press", "buttons": ["A"]})
    assert client.commands == []


def test_editing_parameters_during_startup_blocks_A(tmp_path, source_tree):
    manifest = make_manifest(tmp_path, source_tree);client = FakeClient()
    Path(manifest["inputs"]["parameters"]["source"]).write_text('{"maximum_axis": 0.5}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="parameters"):
        worker.StartVerifiedClient(client, manifest).command({"command": "press", "buttons": ["A"]})
    assert client.commands == []


def test_mismatched_source_fails_before_imports_or_live_client(tmp_path, source_tree, monkeypatch):
    manifest = make_manifest(tmp_path, source_tree)
    (source_tree / "src/raiders_helper/mining_motion.py").write_text("VERSION = 'v2'\n", encoding="utf-8")
    monkeypatch.setattr(worker, "check_cache_isolation", lambda _: None)
    def unexpected_import(*args):pytest.fail("AI must not import after source mismatch")
    monkeypatch.setattr(worker.importlib, "import_module", unexpected_import)
    with pytest.raises(RuntimeError, match="源码"):
        worker.run_worker(tmp_path / "round/manifest.json", verify_only=True)
    assert not (tmp_path / "round/loaded-version.json").exists()
    assert json.loads((tmp_path / "round/worker-result.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_worker_command_requires_empty_unique_cache_and_B(tmp_path, source_tree):
    manifest = make_manifest(tmp_path, source_tree)
    command = hot.worker_command(tmp_path / "round/manifest.json", manifest, verify_only=True)
    assert command[:3] == [sys.executable, "-B", "-X"]
    assert command[3] == f"pycache_prefix={manifest['cache_prefix']}"
    assert command[-1] == "--verify-only"
    (Path(manifest["cache_prefix"]) / "stale.pyc").write_bytes(b"cache")
    with pytest.raises(RuntimeError, match="为空"):
        hot.worker_command(tmp_path / "round/manifest.json", manifest)


def test_live_worker_ready_precedes_A_and_failure_releases_and_stops_heartbeat(tmp_path, source_tree, monkeypatch):
    manifest = make_manifest(tmp_path, source_tree);client = FakeClient(); seen = []
    class Reporter:
        def __init__(self, metadata, **kwargs):seen.append(("metadata", metadata))
        def publish(self, **kwargs):seen.append(("publish", kwargs))
        def close(self):seen.append(("closed", None))
    def fake_round(guarded, recognizer, policy, parameters, output, seconds, *, telemetry):
        ready = json.loads((tmp_path / "round/loaded-version.json").read_text(encoding="utf-8"))
        assert ready["ready"] is True
        assert ready["version_id"] == manifest["version_id"]
        assert parameters == {"maximum_axis": .65}
        guarded.command({"command": "press", "buttons": ["A"]}, wait=True)
        assert (tmp_path / "round/start-verified.json").exists()
        raise RuntimeError("synthetic policy failure")
    modules = {name: SimpleNamespace(__file__=str(source_tree / "src" / (name.replace(".", "/") + ".py")))
               for name in worker.CORE_MODULES}
    modules["raiders_helper.mining_vision"].MiningLiveRecognizer = lambda *args: object()
    modules["raiders_helper.mining_reactive"].decide_mining_reactive = object()
    modules["mining_supervisor"] = SimpleNamespace(LiveClient=lambda *a, **k: client, run_round=fake_round)
    modules["mining_pulse_supervisor"] = object()
    modules["mining_ai_telemetry"] = SimpleNamespace(ActiveAIReporter=Reporter)
    monkeypatch.setattr(worker, "check_cache_isolation", lambda _: None)
    monkeypatch.setattr(worker.importlib, "import_module", lambda name: modules[name])
    with pytest.raises(RuntimeError, match="synthetic policy failure"):
        worker.run_worker(tmp_path / "round/manifest.json")
    assert client.commands[-1] == {"command": "release"}
    assert seen[-1][0] == "closed"
    assert seen[0][1]["parameters_path"] == manifest["inputs"]["parameters"]["source"]
    assert json.loads((tmp_path / "round/worker-result.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_real_fresh_process_ignores_same_timestamp_same_size_old_pyc(tmp_path):
    # Real vision/physics imports run from an isolated source copy. An absent
    # endpoint proves verify-only neither opens LiveClient nor presses A.
    root = tmp_path / "real-source"
    shutil.copytree(ROOT / "src/raiders_helper", root / "src/raiders_helper", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "assets", root / "assets")
    (root / "tools").mkdir()
    for relative in hot.TOOL_SOURCES:shutil.copy2(ROOT / relative, root / relative)
    profile, parameters = inputs(tmp_path)
    motion = root / "src/raiders_helper/mining_motion.py"
    original = motion.read_bytes()
    motion.write_bytes(original + b"\n__version__ = 'v1'\n")
    stat = motion.stat()
    py_compile.compile(str(motion), doraise=True)
    loaded = []
    for number, marker in ((1, "v1"), (2, "v2")):
        if number == 2:
            motion.write_bytes(original + b"\n__version__ = 'v2'\n")
            os.utime(motion, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            assert motion.stat().st_size == stat.st_size
            assert motion.stat().st_mtime_ns == stat.st_mtime_ns
        directory = tmp_path / f"round-{number}"
        manifest = hot.snapshot_round(root, directory, profile, parameters, tmp_path / "absent-endpoint.json",
                                      600, number, "not-a-live-session")
        assert hot.launch_worker(directory / "manifest.json", manifest, verify_only=True) == 0, (directory / "worker.log").read_text(encoding="utf-8")
        value = json.loads((directory / "loaded-version.json").read_text(encoding="utf-8"))
        loaded.append(value)
        assert value["module_versions"]["raiders_helper.mining_motion"]["declared_version"] == marker
        assert value["verify_only"] is True
        assert value["bytecode_writes_disabled"] is True
        assert not (directory / "start-verified.json").exists()
        assert not (directory / "gameplay").exists()
    assert loaded[0]["pid"] != loaded[1]["pid"]
    assert loaded[0]["version_id"] != loaded[1]["version_id"]
    assert loaded[0]["inputs"]["parameters"]["sha256"] == loaded[1]["inputs"]["parameters"]["sha256"]
    assert not (root / "artifacts/active_ai.json").exists()
