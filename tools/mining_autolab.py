"""Run versioned MINING physics collection batches without automatic tuning.

Each round delegates menu/result confirmation and bounded inputs to
mining_pulse_supervisor.collect_round. A failure stops the run; this scheduler
never presses Retry itself. Config/profile changes take effect next round.
No endpoint contents, client headers or exception payloads enter these logs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT/"src", ROOT/"tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from mining_pulse_supervisor import collect_round
from mining_supervisor import LiveClient
from raiders_helper.mining_identification import MiningPhysicsProbe
from raiders_helper.mining_vision import MiningLiveRecognizer
# Preload the recognizer's local screen dependency before freezing versions.
# Importing these classes does not start capture or connect a controller.
from raiders_helper.session_runtime import ScreenMatcher as _ScreenMatcher


def _stamp():
    return {"utc": datetime.now(timezone.utc).isoformat(), "monotonic": time.perf_counter()}


def _startup_source_versions():
    paths = {Path(__file__).resolve()}
    for name, module in tuple(sys.modules.items()):
        if name.startswith("raiders_helper") or name in {"mining_pulse_supervisor", "mining_supervisor"}:
            filename = getattr(module, "__file__", None)
            if filename and Path(filename).suffix == ".py":
                paths.add(Path(filename).resolve())
    files = []
    for path in sorted(paths):
        if path.is_relative_to(ROOT):
            files.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {"recorded_at": _stamp(), "files": files, "restart_required_for_code_changes": True,
            "scope": "local Python modules already imported in this scheduler process; startup disk hashes",
            "reload_semantics": "Only config/profile are reread between rounds. Python edits require a new process; this list is never refreshed mid-run.",
            "separate_process": "The capture/controller GUI has its own already-loaded code; this manifest does not claim to version that process."}


# Frozen once at import, not after each round or after a subsequent source edit.
_STARTUP_SOURCE_VERSIONS = _startup_source_versions()


def _write_json(path, value):
    """Replace only this run's own status file, so readers see complete JSON."""
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf8")
    temporary.replace(path)


def _snapshot_inputs(config_path, profile_path, folder, source_manifest):
    # Read both versions once; the exact parsed bytes are what collect_round gets.
    raw = {"config": config_path.read_bytes(), "profile": profile_path.read_bytes()}
    sources = {"config": config_path, "profile": profile_path}
    folder.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name, content in raw.items():
        snapshot = folder/(name+".json")
        snapshot.write_bytes(content)
        versions[name] = {"source": str(sources[name]), "snapshot": str(snapshot.resolve()),
                          "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    manifest = {"version": 1, "read_at": _stamp(), "inputs": versions,
                "loaded_source_manifest": str(source_manifest), "code_changes_require_restart": True,
                "scope": "exact per-round input files; no automatic parameter changes"}
    _write_json(folder/"manifest.json", manifest)
    config, profile = (json.loads(raw[name].decode("utf-8-sig")) for name in ("config", "profile"))
    if not isinstance(config, dict) or not isinstance(profile, dict):
        raise ValueError("config/profile 必须是JSON对象")
    return config, profile, {"manifest": str((folder/"manifest.json").resolve()),
                             "loaded_source_manifest": str(source_manifest), **versions}


def _validate_inputs(config, profile):
    # Validate before collect_round can confirm A and start the console game.
    if config.get("enabled") is not True:
        raise ValueError("每局探针都需明确enabled=true")
    if profile.get("game", "mining") != "mining" or profile.get("synthetic") is True:
        raise ValueError("只接受MINING实机识别配置")
    MiningPhysicsProbe(config)
    MiningLiveRecognizer("mining", profile)


def _round_record(summary, identity, inputs):
    # Whitelist summary fields; never copy an arbitrary service/client object.
    record = dict(identity)
    for key in ("source_session", "actual_commands", "seconds", "frames", "playing_frames", "result"):
        if key in summary:
            record[key] = summary[key]
    events = summary.get("events", [])
    record["segments_completed"] = sum(isinstance(event, dict) and event.get("type") == "completed" for event in events)
    record["segments_aborted"] = sum(isinstance(event, dict) and event.get("type") == "aborted" for event in events)
    record["inputs"] = inputs
    record["calibrated"] = False
    return record


def _stop_requested(path):
    try:
        path.stat()
    except FileNotFoundError:
        return False
    return True


def run_autolab(args):
    """Return an exit code; write bounded summaries plus an append-only index."""
    output = args.output.resolve()
    # Refuse any existing directory, including an empty one; there is no resume.
    output.mkdir(parents=True, exist_ok=False)
    config_path, profile_path = args.config.resolve(), args.profile.resolve()
    state = {"version": 1, "task": "mining_physics_collection", "status": "starting",
             "pid": os.getpid(), "loaded_source_manifest": str(output/"loaded-sources.json"),
             "code_changes_require_restart": True,
             "reload_semantics": "config/profile per round; Python source changes require process restart",
             "requested_rounds": args.rounds, "batch_size": args.batch_size,
             "seconds_per_round": args.seconds_per_round, "completed_rounds": 0,
             "completed_batches": 0, "config_path": str(config_path), "profile_path": str(profile_path),
             "automatic_parameter_changes": False, "calibrated": False,
             "stop_request_path": str(output/"stop.request"),
             "current_round": None, "last_completed_round": None, "release_attempted": False}
    client, batch, batch_path = None, None, None
    phase, exit_code, failure = "initial_status", 0, None

    def publish():
        state["updated_at"] = _stamp()
        _write_json(output/"current.json", state)
        # This aggregate remains bounded even with --rounds 0.
        _write_json(output/"summary.json", state)

    def index(event):
        with (output/"experiment-index.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps({"at": _stamp(), **event}, ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()

    def stop_at_boundary(boundary):
        if not _stop_requested(output/"stop.request"):
            return False
        state.update(status="stopped_by_request", current_round=None,
                     stop_observed_at=_stamp(), stop_observed_boundary=boundary)
        if batch is not None:
            batch["status"] = "stopped_by_request"
        index({"event": "stop_requested", "boundary": boundary,
               "completed_rounds": state["completed_rounds"]})
        return True

    try:
        phase = "startup_source_manifest"
        _write_json(output/"loaded-sources.json", _STARTUP_SOURCE_VERSIONS)
        publish()
        round_id = 0
        while args.rounds == 0 or round_id < args.rounds:
            phase = "check_stop_before_round"
            if stop_at_boundary("before_round"):
                break
            round_id += 1
            batch_id = (round_id-1)//args.batch_size+1
            local_id = (round_id-1)%args.batch_size+1
            mode = "normal" if round_id%2 else "zr"
            next_batch_path = output/f"batch-{batch_id:04d}"
            round_path = next_batch_path/f"round-{local_id:02d}"
            identity = {"round_id": round_id, "batch_id": batch_id, "batch_round": local_id,
                        "mode": mode, "directory": str(round_path.resolve())}
            state.update(status="preparing", current_round=identity)
            if local_id == 1:
                # Failure while opening the next batch must not relabel the
                # previous, fully collected batch as failed.
                batch, batch_path = None, None
                phase = "create_batch"
                batch_path = next_batch_path
                batch_path.mkdir()
                batch = {"version": 1, "batch_id": batch_id, "status": "collecting",
                         "rounds": [], "calibrated": False, "automatic_parameter_changes": False}
                _write_json(batch_path/"summary.json", batch)
            phase = "snapshot_inputs"
            publish()
            config, profile, inputs = _snapshot_inputs(config_path, profile_path,
                batch_path/"inputs"/f"round-{local_id:02d}", output/"loaded-sources.json")
            phase = "validate_inputs"
            _validate_inputs(config, profile)
            state["current_round"] = {**identity, "inputs": inputs}
            state["status"] = "collecting"
            phase = "client_initialization"
            if client is None:
                client = LiveClient(args.endpoint, frame_width=960)
            phase = "round_started_index"
            index({"event": "round_started", **identity, "inputs": inputs})
            publish()
            print(f"自动采集 batch-{batch_id:04d} / R{round_id:03d} {mode} "
                  f"config={inputs['config']['sha256'][:12]} profile={inputs['profile']['sha256'][:12]}", flush=True)
            phase = "collect_round"
            summary = collect_round(client, profile, config, round_path, round_id, mode, args.seconds_per_round)
            phase = "confirm_result"
            if not isinstance(summary, dict) or summary.get("result") not in {"result:retry", "result:quit"}:
                raise RuntimeError("未确认明确结算，停止，不继续Retry")
            record = _round_record(summary, identity, inputs)
            phase = "round_completed_index"
            index({"event": "round_completed", **record})
            batch["rounds"].append(record)
            batch["status"] = "collected_not_fitted" if local_id == args.batch_size else "collecting"
            _write_json(batch_path/"summary.json", batch)
            state["completed_rounds"] = round_id
            state["completed_batches"] = batch_id if local_id == args.batch_size else batch_id-1
            state.update(status="between_rounds", last_completed_round=record, current_round=None)
            publish()
            phase = "check_stop_after_round"
            if stop_at_boundary("after_round"):
                break
            if args.rounds == 0 or round_id < args.rounds:
                phase = "between_rounds"
                time.sleep(.35)
        if state["status"] != "stopped_by_request":
            state["status"] = "collected_not_fitted"
        if batch is not None and state["status"] != "stopped_by_request":
            batch["status"] = "collected_not_fitted"
    except KeyboardInterrupt:
        exit_code = 130
        state["status"] = "interrupted"
        failure = {"stage": phase, "error_type": "KeyboardInterrupt"}
    except Exception as error:
        exit_code = 1
        state["status"] = "failed"
        # Exception messages can contain an HTTP request/header representation.
        # Report type and stage; the collector's own evidence remains on disk.
        failure = {"stage": phase, "error_type": type(error).__name__}
    finally:
        if client is not None:
            state["release_attempted"] = True
            try:
                client.command({"command": "release"}, wait=True)
                state["release_succeeded"] = True
            except BaseException as error:
                state["release_succeeded"] = False
                state["release_error_type"] = type(error).__name__
                if exit_code == 0:
                    exit_code = 1
                    state["status"] = "failed"
                    failure = {"stage": "final_release", "error_type": type(error).__name__}
        if failure:
            state["failure"] = failure
            if batch is not None:
                batch.update(status=state["status"], failure=failure)
            index({"event": state["status"], "round": state.get("current_round"), **failure})
        if batch is not None:
            _write_json(batch_path/"summary.json", batch)
        publish()
    if exit_code:
        print(f"自动实验已停止：{state['status']} / {failure['stage']} / {failure['error_type']}；"
              f"已完成{state['completed_rounds']}局，详见 {output/'current.json'}。", file=sys.stderr)
    elif state["status"] == "stopped_by_request":
        print(f"已在局界按stop.request停止，保留{state['completed_rounds']}局完整数据，未开始下一局。", flush=True)
    else:
        print(f"已连续采集{state['completed_rounds']}局，尚未拟合或自动调参。", flush=True)
    return exit_code


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf8")
    parser = argparse.ArgumentParser(description="MINING物理实验自动连续调度，逐局重读配置，不自动调参")
    parser.add_argument("--endpoint", type=Path, default=ROOT/"artifacts/control_endpoint.json")
    parser.add_argument("--profile", type=Path, default=ROOT/"artifacts/mining-supervision/profile.json")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6, help="采集局数；0表示持续，异常或Ctrl+C时停止")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--seconds-per-round", type=float, default=90)
    args = parser.parse_args(argv)
    if args.rounds < 0 or args.batch_size < 1 or not math.isfinite(args.seconds_per_round) or args.seconds_per_round <= 0:
        parser.error("rounds需非负、batch-size需为正、每局时限需为有限正数")
    if args.output.exists():
        parser.error("输出路径已存在，拒绝覆盖；请选择新目录")
    return run_autolab(args)


if __name__ == "__main__":
    raise SystemExit(main())
