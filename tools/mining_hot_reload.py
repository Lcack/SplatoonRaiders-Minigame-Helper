"""Reload MINING Python AI and JSON inputs between real rounds, not mid-round.

The coordinator imports only the standard library. The existing GUI remains
the owner of capture and controller hardware; children use its HTTP interface.
Each round gets a new Python interpreter and an empty bytecode-cache prefix.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
TOOL_SOURCES = ("tools/mining_hot_reload.py", "tools/mining_round_worker.py",
                "tools/mining_supervisor.py", "tools/mining_pulse_supervisor.py", "tools/mining_ai_telemetry.py")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_object(path):
    raw = Path(path).read_bytes()
    def invalid_constant(value):
        raise ValueError(f"JSON不接受非有限数：{value}")
    value = json.loads(raw, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError(f"JSON顶层必须为对象：{path}")
    return raw, value


def source_files(root):
    root = Path(root)
    paths = list((root / "src/raiders_helper").rglob("*.py"))
    paths.extend(root / name for name in TOOL_SOURCES)
    return sorted(set(paths))


def source_hashes(root, *, check_syntax=False):
    root = Path(root).resolve()
    result = {}
    for path in source_files(root):
        raw = path.read_bytes()
        if check_syntax:
            compile(raw, str(path), "exec", dont_inherit=True)
        result[path.relative_to(root).as_posix()] = sha256(raw)
    return result


def version_digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def verify_manifest(manifest, *, check_syntax=False):
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("不支持的round manifest版本")
    root = Path(manifest["source_root"]).resolve()
    actual = source_hashes(root, check_syntax=check_syntax)
    expected = manifest["source_hashes"]
    if actual != expected:
        changes = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
        raise RuntimeError("源码在局间快照后改变，拒绝开始：" + ", ".join(changes))
    for name in ("profile", "parameters"):
        item = manifest["inputs"][name]
        raw, _ = read_object(item["snapshot"])
        if sha256(raw) != manifest["inputs"][name]["sha256"]:
            raise RuntimeError(f"{name}快照校验失败，拒绝开始")
        live_raw, _ = read_object(item["source"])
        if sha256(live_raw) != item["sha256"]:
            raise RuntimeError(f"{name}在局间快照后改变，拒绝开始；下一次启动读取新配置")
    if version_digest(expected) != manifest["source_version"]:
        raise RuntimeError("source_version与源码散列不一致")
    inputs = {name: manifest["inputs"][name]["sha256"] for name in ("profile", "parameters")}
    if version_digest(inputs) != manifest["config_version"]:
        raise RuntimeError("config_version与输入散列不一致")
    if version_digest({"source": expected, "inputs": inputs}) != manifest["version_id"]:
        raise RuntimeError("version_id与本局快照不一致")
    return actual


def snapshot_round(root, directory, profile, parameters, endpoint, seconds, number, capture_session):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    before = source_hashes(root, check_syntax=True)
    inputs, objects = {}, {}
    for name, source in (("profile", profile), ("parameters", parameters)):
        raw, value = read_object(source)
        inputs[name] = {"source": str(Path(source).resolve()), "snapshot": str(directory / f"{name}.json"),
                        "sha256": sha256(raw)}
        objects[name] = raw
    if before != source_hashes(root):
        raise RuntimeError("读取局间快照期间源码发生变化，拒绝启动")
    directory.mkdir(parents=True, exist_ok=False)
    cache = directory / "empty-bytecode-cache"
    cache.mkdir()
    for name, raw in objects.items():
        (directory / f"{name}.json").write_bytes(raw)
    input_hashes = {name: value["sha256"] for name, value in inputs.items()}
    manifest = {"schema_version": SCHEMA_VERSION, "created_at": utc_now(), "round": number,
        "source_root": str(root), "source_hashes": before, "source_version": version_digest(before),
        "config_version": version_digest(input_hashes), "version_id": version_digest({"source": before, "inputs": input_hashes}),
        "inputs": inputs, "endpoint": str(Path(endpoint).resolve()), "seconds_per_round": seconds,
        "round_directory": str(directory), "gameplay_output": str(directory / "gameplay"),
        "cache_prefix": str(cache), "capture_session": str(capture_session),
        "scope": "Python source tree and two JSON inputs; no hardware restart or source rollback"}
    atomic_json(directory / "manifest.json", manifest)
    return manifest


class EndpointClient:
    """Persistent coordinator client; no AI, capture, or hardware imports."""
    def __init__(self, endpoint):
        _, data = read_object(endpoint)
        if not data.get("running") or not data.get("url", "").startswith("http://127.0.0.1:"):
            raise ValueError("需要正在运行的本机任务终端接口")
        self.url = data["url"].rstrip("/")
        self.headers = {data.get("token_header", "X-Raiders-Token"): data["token"], "Content-Type": "application/json"}
        self.opener = build_opener(ProxyHandler({}))

    def request(self, route, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        with self.opener.open(Request(self.url + route, data=data, headers=self.headers), timeout=3) as response:
            return json.loads(response.read())

    def command(self, payload, wait=False):
        result = self.request("/commands", payload)
        deadline = time.perf_counter() + 3
        while wait and result["status"] in ("queued", "running") and time.perf_counter() < deadline:
            time.sleep(.015)
            result = self.request("/commands/" + result["id"])
        if wait and result["status"] != "succeeded":
            raise RuntimeError(f"任务终端指令失败：{result}")
        return result


def worker_command(manifest_path, manifest, *, verify_only=False):
    cache = Path(manifest["cache_prefix"])
    if not cache.is_dir() or any(cache.iterdir()):
        raise RuntimeError("每局bytecode cache目录必须存在且为空")
    command = [sys.executable, "-B", "-X", f"pycache_prefix={cache}",
               str(Path(manifest["source_root"]) / "tools/mining_round_worker.py"), "--manifest", str(manifest_path)]
    if verify_only:
        command.append("--verify-only")
    return command


def launch_worker(manifest_path, manifest, *, verify_only=False):
    command = worker_command(manifest_path, manifest, verify_only=verify_only)
    log_path = Path(manifest["round_directory"]) / "worker.log"
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with log_path.open("xb") as log:
        return subprocess.run(command, cwd=manifest["source_root"], stdout=log, stderr=subprocess.STDOUT,
                              creationflags=creationflags, check=False,
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                              timeout=60 if verify_only else manifest["seconds_per_round"] + 60).returncode


def run_coordinator(args, *, client_factory=EndpointClient, launcher=launch_worker, source_root=ROOT):
    if args.rounds < 0 or not math.isfinite(args.seconds_per_round) or args.seconds_per_round <= 0:
        raise ValueError("rounds必须≥0，seconds-per-round必须为有限正数")
    round_pause=getattr(args,"round_pause_seconds",0.)
    if not math.isfinite(round_pause) or not 0<=round_pause<=30:
        raise ValueError("round-pause-seconds必须在0到30秒之间")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"requested_rounds": args.rounds, "completed_rounds": 0, "rounds": [], "status": "starting",
               "created_at": utc_now(), "reload_boundary": "between_rounds_only", "hardware_restart": False}
    client = None
    try:
        client = client_factory(args.endpoint)
        while args.rounds == 0 or summary["completed_rounds"] < args.rounds:
            if (output / "stop.request").exists():
                summary["status"] = "stopped_at_round_boundary"
                break
            client.command({"command": "release"}, wait=True)
            status = client.request("/status")
            session = status.get("output")
            if not isinstance(session, str) or not session:
                raise RuntimeError("无法确认当前采集会话，拒绝启动")
            number = summary["completed_rounds"] + 1
            directory = output / f"round-{number:04d}"
            manifest = snapshot_round(source_root, directory, args.profile, args.parameters, args.endpoint,
                                      args.seconds_per_round, number, session)
            verify_manifest(manifest, check_syntax=True)
            summary["status"] = "worker_running"
            summary["current_round"] = number
            atomic_json(output / "summary.json", summary)
            print(f"第{number}局：新进程加载AI {manifest['version_id'][:12]}，采集与控制器保持连接", flush=True)
            exit_code = launcher(directory / "manifest.json", manifest)
            if exit_code != 0:
                raise RuntimeError(f"第{number}局worker失败（exit={exit_code}），已停止；日志 {directory / 'worker.log'}")
            _, loaded = read_object(directory / "loaded-version.json")
            _, result = read_object(directory / "worker-result.json")
            if loaded.get("version_id") != manifest["version_id"] or result.get("version_id") != manifest["version_id"]:
                raise RuntimeError("worker返回的加载版本与本局快照不一致")
            report = result.get("report", {})
            if result.get("status") != "completed" or not str(report.get("result", "")).startswith("result:"):
                raise RuntimeError("未确认本局结束，停止且不盲按Retry")
            summary["rounds"].append({"round": number, "directory": str(directory), "version_id": manifest["version_id"],
                "source_version": manifest["source_version"], "config_version": manifest["config_version"], "pid": loaded.get("pid"),
                "report": report})
            summary["completed_rounds"] += 1
            atomic_json(output / "summary.json", summary)
            if round_pause and (args.rounds==0 or summary["completed_rounds"]<args.rounds):
                # Controls have been released by the completed worker. This
                # optional pause leaves the result visible for a demo recording.
                time.sleep(round_pause)
        else:
            summary["status"] = "completed"
        return summary
    except BaseException as error:
        summary["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if client is not None:
            try:
                client.command({"command": "release"}, wait=True)
            except Exception as error:
                summary["final_release_error"] = str(error)
        summary["finished_at"] = utc_now()
        atomic_json(output / "summary.json", summary)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", type=Path, default=ROOT / "artifacts/control_endpoint.json")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=1, help="0表示持续运行，到局界检查stop.request")
    parser.add_argument("--seconds-per-round", type=float, default=600)
    parser.add_argument("--round-pause-seconds",type=float,default=0,help="结算后等待秒数，默认0；用于展示最终分数")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出目录已存在，拒绝覆盖")
    try:
        run_coordinator(args)
    except KeyboardInterrupt:
        print("已中断，尝试释放控制。", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"热更新监督已停止：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
