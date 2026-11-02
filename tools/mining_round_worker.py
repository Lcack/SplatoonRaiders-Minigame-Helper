"""One fresh-import MINING round; --verify-only never opens the live client."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

from mining_hot_reload import atomic_json, read_object, utc_now, verify_manifest


CORE_MODULES = ("raiders_helper.mining_vision", "raiders_helper.mining_reactive",
    "raiders_helper.mining_planner", "raiders_helper.mining_motion",
    "raiders_helper.motion_calibration", "raiders_helper.mining_identification")


def check_cache_isolation(manifest):
    expected = Path(manifest["cache_prefix"]).resolve()
    if (not sys.dont_write_bytecode or sys.pycache_prefix is None
            or Path(sys.pycache_prefix).resolve() != expected
            or not expected.is_dir() or any(expected.iterdir())):
        raise RuntimeError("worker必须用-B及本局独立空pycache_prefix启动，拒绝读取旧缓存")


class StartVerifiedClient:
    """Delegate to existing LiveClient, rechecking immediately before A."""
    def __init__(self, client, manifest):
        self._client = client
        self._manifest = manifest
        self._start_checked = False

    def __getattr__(self, name):
        return getattr(self._client, name)

    def command(self, payload, wait=False):
        if payload.get("command") == "press" and "A" in payload.get("buttons", []):
            if not self._start_checked:
                verify_manifest(self._manifest)
                status = self._client.request("/status")
                if status.get("output") != self._manifest["capture_session"]:
                    raise RuntimeError("采集会话在局前改变，拒绝按A")
                atomic_json(Path(self._manifest["round_directory"]) / "start-verified.json", {
                    "version_id": self._manifest["version_id"], "verified_at": utc_now(),
                    "capture_session": status["output"], "check": "source_inputs_and_session_immediately_before_A"})
                self._start_checked = True
        return self._client.command(payload, wait=wait)


def run_worker(manifest_path, *, verify_only=False):
    _, manifest = read_object(manifest_path)
    directory = Path(manifest["round_directory"])
    result = {"version_id": manifest.get("version_id"), "status": "loading", "pid": os.getpid()}
    client = None
    publisher = None
    try:
        check_cache_isolation(manifest)
        verify_manifest(manifest, check_syntax=True)
        root = Path(manifest["source_root"])
        sys.path.insert(0, str(root / "src"))
        sys.path.insert(0, str(root / "tools"))
        modules = {name: importlib.import_module(name) for name in CORE_MODULES}
        supervisor = importlib.import_module("mining_supervisor")
        telemetry = importlib.import_module("mining_ai_telemetry")
        # run_round lazily imports this evidence helper. Load it now so a
        # mid-round source edit cannot supply another helper version later.
        importlib.import_module("mining_pulse_supervisor")
        _, profile = read_object(manifest["inputs"]["profile"]["snapshot"])
        _, parameters = read_object(manifest["inputs"]["parameters"]["snapshot"])
        recognizer = modules["raiders_helper.mining_vision"].MiningLiveRecognizer("mining", profile)
        policy = modules["raiders_helper.mining_reactive"].decide_mining_reactive
        # Import/constructor failure or concurrent source modification stops
        # here, before constructing a live client or pressing any button.
        verify_manifest(manifest)
        loaded = {"schema_version": manifest["schema_version"], "ready": True, "loaded_at": utc_now(),
            "pid": os.getpid(), "verify_only": verify_only, "source_root": manifest["source_root"],
            "source_hashes": manifest["source_hashes"], "source_version": manifest["source_version"],
            "config_version": manifest["config_version"], "version_id": manifest["version_id"],
            "inputs": manifest["inputs"], "round": manifest["round"], "capture_session": manifest["capture_session"],
            "python_executable": sys.executable, "python_version": sys.version, "cache_prefix": sys.pycache_prefix,
            "bytecode_writes_disabled": sys.dont_write_bytecode,
            "module_versions": {name: {"file": module.__file__, "declared_version": getattr(module, "__version__", None)}
                                for name, module in modules.items()},
            "verification_scope": "pre_and_post_import_hashes_and_compiled_tree; first_A_rechecks_sources_inputs_session",
            "hardware_owner_reloaded": False,
            "compiled_source_hashes_are_not_all_imported_modules": True}
        atomic_json(directory / "loaded-version.json", loaded)
        print(json.dumps({"ready": True, "version_id": loaded["version_id"], "pid": loaded["pid"],
                          "verify_only": verify_only}, ensure_ascii=False), flush=True)
        if verify_only:
            result["status"] = "verified_only"
            return result
        raw_client = supervisor.LiveClient(Path(manifest["endpoint"]), frame_width=960)
        client = StartVerifiedClient(raw_client, manifest)
        current = client.request("/status")
        if current.get("output") != manifest["capture_session"]:
            raise RuntimeError("当前任务终端与快照采集会话不一致")
        publisher = telemetry.ActiveAIReporter({"capture_session": manifest["capture_session"],
            "run_dir": manifest["round_directory"], "loaded_version": manifest["version_id"],
            "source_hashes": manifest["source_hashes"], "source_version": manifest["source_version"],
            "config_version": manifest["config_version"], "round": manifest["round"],
            "profile_sha": manifest["inputs"]["profile"]["sha256"],
            "parameters_sha": manifest["inputs"]["parameters"]["sha256"],
            "profile_path": manifest["inputs"]["profile"]["source"],
            "parameters_path": manifest["inputs"]["parameters"]["source"],
            "mode": "ready", "reason": "本局新进程已导入并核查版本，等待确认开始"},
            path=root / "artifacts/active_ai.json")
        publisher.publish(force=True)
        report = supervisor.run_round(client, recognizer, policy, parameters, Path(manifest["gameplay_output"]),
                                      manifest["seconds_per_round"], telemetry=publisher)
        if not str(report.get("result", "")).startswith("result:"):
            raise RuntimeError("未确认本局结算，停止且不盲按Retry")
        result.update(status="completed", report=report)
        return result
    except BaseException as error:
        result.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if client is not None:
            try:
                client.command({"command": "release"}, wait=True)
            except Exception as error:
                result["final_release_error"] = str(error)
        if publisher is not None:
            try:
                publisher.close()
            except Exception as error:
                result["telemetry_close_error"] = str(error)
        result["finished_at"] = utc_now()
        atomic_json(directory / "worker-result.json", result)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true", help="仅核查版本并导入/构造AI，不连接实机")
    args = parser.parse_args()
    try:
        run_worker(args.manifest, verify_only=args.verify_only)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"本局worker未完成：{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
