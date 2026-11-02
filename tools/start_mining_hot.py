"""One-click MINING GUI + between-round AI reload; never sends game inputs."""
from contextlib import contextmanager
from datetime import datetime
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from mining_hot_reload import EndpointClient, atomic_json, read_object

ROOT = Path(__file__).resolve().parents[1]
START_SCREENS = {"menu:mining", "result:retry", "result:quit"}


def pid_running(pid):
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.OpenProcess.restype = wintypes.HANDLE
        api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = api.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # Access denied is uncertain/alive.
        try:
            code = wintypes.DWORD()
            return not api.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            api.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def active_ai_running(root, *, now=time.time, alive=pid_running):
    try:
        _, value = read_object(Path(root) / "artifacts/active_ai.json")
    except (OSError, ValueError):
        return False
    stamp = value.get("updated_unix")
    return (value.get("running") is True and type(stamp) in (float, int) and math.isfinite(stamp)
            and 0 <= now() - stamp <= 3 and alive(value.get("pid")))


@contextmanager
def starter_lock(path, *, alive=pid_running):
    path = Path(path)
    for attempt in range(2):
        try:
            stream = path.open("x", encoding="utf-8")
            break
        except FileExistsError:
            try:
                _, value = read_object(path)
            except (OSError, ValueError):
                raise RuntimeError("启动锁状态未确认，未启动第二个控制进程") from None
            if alive(value.get("pid")) or alive(value.get("coordinator_pid")):
                raise RuntimeError("MINING热更新任务已在启动或运行，未重复打开终端")
            if attempt:
                raise RuntimeError("启动锁被其他进程占用")
            path.unlink()
    with stream:
        stream.write(json.dumps({"pid": os.getpid()}))
    try:
        yield path
    finally:
        try:
            _, value = read_object(path)
            if value.get("pid") == os.getpid():
                path.unlink()
        except (OSError, ValueError):
            pass


def probe_session(endpoint, *, client_factory=EndpointClient, alive=pid_running):
    if not Path(endpoint).exists():
        return None
    _, info = read_object(endpoint)
    if info.get("running") is False:
        return None
    if info.get("running") is not True:
        raise RuntimeError("本机接口状态未确认，未重复打开控制器")
    try:
        client = client_factory(endpoint)
        return client.request("/status")
    except Exception:
        if type(info.get("pid")) is int and not alive(info["pid"]):
            return None
        raise RuntimeError("现有任务终端接口未响应，未重复打开串口；关闭现有终端后重新启动") from None


def validate_start_screen(status):
    if status.get("game") != "mining" or status.get("mode") != "manual":
        raise RuntimeError("现有会话不是MINING人工监督模式，未启动AI")
    if status.get("control_enabled") is not True:
        raise RuntimeError("MINING终端尚未连接控制器，先在终端完成连接")
    if status.get("screen") not in START_SCREENS:
        raise RuntimeError("请停在MINING选项或RETRY/QUIT结算页；当前画面不适合开始，未发送按键")
    if not isinstance(status.get("output"), str) or not status["output"]:
        raise RuntimeError("无法确认采集会话，未启动AI")


def launch(root=ROOT, *, endpoint=None, timeout=45, popen=subprocess.Popen,
           client_factory=EndpointClient, alive=pid_running, now=time.time,
           monotonic=time.monotonic, sleep=time.sleep):
    root = Path(root).resolve()
    endpoint = Path(endpoint or root / "artifacts/control_endpoint.json").resolve()
    directory = root / "artifacts" / ("hot-mining-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {"launcher_pid": os.getpid(), "directory": str(directory), "status": "starting"}
    with (directory / "launcher.log").open("ab", buffering=0) as log:
        def note(message):
            log.write((datetime.now().isoformat(timespec="seconds") + " " + message + "\n").encode("utf-8"))
        try:
            if active_ai_running(root, now=now, alive=alive):
                raise RuntimeError("已有有效AI控制心跳，未启动第二个控制者")
            with starter_lock(root / "artifacts/mining-hot-start.lock", alive=alive) as lock:
                if active_ai_running(root, now=now, alive=alive):
                    raise RuntimeError("已有AI正在运行，未重复启动")
                profile = root / "profiles/mining.supervised.local.json"
                parameters = root / "profiles/mining.ai.local.json"
                read_object(profile); read_object(parameters)
                python = Path(sys.executable)
                if python.stem.lower() == "pythonw" and python.with_name("python.exe").exists():
                    python = python.with_name("python.exe")
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                options = {"cwd": str(root), "stdout": log, "stderr": subprocess.STDOUT,
                           "creationflags": flags, "env": {**os.environ, "PYTHONIOENCODING": "utf-8"}}
                status = probe_session(endpoint, client_factory=client_factory, alive=alive)
                gui = None
                if status is None:
                    note("没有有效运行会话，打开MINING任务终端；录像关闭")
                    gui = popen([str(python), str(root / "run.py"), "task-terminal", "--game", "mining",
                                 "--profile", str(profile), "--auto-start", "--no-record"], **options)
                    deadline = monotonic() + timeout
                    while monotonic() < deadline:
                        if gui.poll() is not None:
                            raise RuntimeError("任务终端启动后已退出，详情见launcher.log")
                        sleep(.25)
                        status = probe_session(endpoint, client_factory=client_factory, alive=alive)
                        if status and all(key in status for key in ("game", "mode", "control_enabled", "screen", "output")):
                            if status.get("game") != "mining" or status.get("mode") != "manual":
                                validate_start_screen(status)
                            if status.get("screen") not in {"unknown", None, ""}:
                                validate_start_screen(status)
                                break
                    else:
                        raise RuntimeError("等待任务终端接口超时，未启动AI；可在GUI检查采集卡与控制器")
                else:
                    note("复用现有任务终端，未重新打开采集卡或串口")
                validate_start_screen(status)
                if active_ai_running(root, now=now, alive=alive):
                    raise RuntimeError("等待期间已有AI接管，未启动第二个控制者")
                # The coordinator refuses an existing output directory, so
                # keep launcher evidence above its initially absent run/.
                command = [str(python), str(root / "tools/mining_hot_reload.py"), "--endpoint", str(endpoint),
                           "--profile", str(profile), "--parameters", str(parameters),
                           "--output", str(directory / "run"), "--rounds", "0"]
                coordinator = popen(command, **options)
                atomic_json(lock, {"pid": os.getpid(), "coordinator_pid": coordinator.pid})
                metadata.update(status="running", gui_started=gui is not None, coordinator_pid=coordinator.pid,
                                capture_session=status["output"], coordinator_output=str(directory / "run"))
                atomic_json(directory / "startup.json", metadata)
                note(f"热更新AI已启动，PID={coordinator.pid}；局间停止文件：run/stop.request")
                code = coordinator.wait()
                metadata.update(status="finished", exit_code=code)
                note(f"热更新AI已结束，exit={code}；GUI保持由用户管理")
                return metadata
        except BaseException as error:
            metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
            note(metadata["error"])
            raise RuntimeError(f"{error}\n日志：{directory / 'launcher.log'}") from None
        finally:
            atomic_json(directory / "startup.json", metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", type=Path, default=ROOT / "artifacts/control_endpoint.json")
    args = parser.parse_args()
    try:
        result = launch(endpoint=args.endpoint)
        return result.get("exit_code", 0)
    except Exception as error:
        if os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, str(error), "MINING热更新任务未启动", 0x10)
        elif sys.stderr is not None:
            print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
