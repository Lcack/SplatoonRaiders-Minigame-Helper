"""One-click entry tests mock every process and HTTP call; no NS interaction."""
import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
starter = importlib.import_module("start_mining_hot")


@pytest.fixture
def root(tmp_path):
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "profiles").mkdir()
    for filename in ("mining.supervised.local.json", "mining.ai.local.json"):
        (tmp_path / "profiles" / filename).write_text("{}", encoding="utf-8")
    return tmp_path


def endpoint(root):
    path = root / "artifacts/control_endpoint.json"
    path.write_text(json.dumps({"running": True, "pid": 42, "token": "never-log-this-token"}), encoding="utf-8")
    return path


def status(**changes):
    return {"game": "mining", "mode": "manual", "screen": "result:retry", "control_enabled": True,
            "output": "capture-one", **changes}


class Client:
    def __init__(self, state):self.state = state
    def request(self, route):
        assert route == "/status"
        return self.state


class Process:
    def __init__(self, pid):self.pid = pid
    def poll(self):return None
    def wait(self):return 0


def test_reuses_existing_gui_and_only_launches_hidden_coordinator(root):
    endpoint(root);calls = []
    def popen(command, **options):
        calls.append((command, options))
        output = Path(command[command.index("--output") + 1])
        assert output.name == "run" and not output.exists()
        assert output.parent.joinpath("launcher.log").exists()
        assert command[-2:] == ["--rounds", "0"]
        return Process(1001)
    result = starter.launch(root, popen=popen, client_factory=lambda _: Client(status()), alive=lambda _: True)
    assert len(calls) == 1
    assert "mining_hot_reload.py" in calls[0][0][1]
    assert result["gui_started"] is False
    assert not (root / "artifacts/mining-hot-start.lock").exists()
    if sys.platform == "win32":assert calls[0][1]["creationflags"] == starter.subprocess.CREATE_NO_WINDOW
    log = Path(result["directory"]).joinpath("launcher.log").read_text(encoding="utf-8")
    assert "never-log-this-token" not in log


def test_no_session_opens_gui_then_waits_for_endpoint_before_coordinator(root):
    calls = []
    def popen(command, **options):
        calls.append(command)
        if "task-terminal" in command:
            endpoint(root)
            assert "--auto-start" in command and "--no-record" in command
            assert command[command.index("--game") + 1] == "mining"
        return Process(1000 + len(calls))
    result = starter.launch(root, popen=popen, client_factory=lambda _: Client(status()),
                            alive=lambda _: True, sleep=lambda _: None)
    assert result["gui_started"] is True
    assert len(calls) == 2
    assert "task-terminal" in calls[0]
    assert "mining_hot_reload.py" in calls[1][1]


@pytest.mark.parametrize("changes", [{"game": "hopping"}, {"mode": "single"},
    {"screen": "playing:mining"}, {"screen": "unknown"}, {"control_enabled": False}])
def test_wrong_game_or_unsafe_screen_does_not_spawn_or_send_inputs(root, changes):
    endpoint(root)
    with pytest.raises(RuntimeError):
        starter.launch(root, popen=lambda *a, **k: pytest.fail("must not launch"),
                       client_factory=lambda _: Client(status(**changes)), alive=lambda _: True)
    logs = list((root / "artifacts").glob("hot-mining-*/launcher.log"))
    assert len(logs) == 1
    assert "never-log-this-token" not in logs[0].read_text(encoding="utf-8")


def test_active_heartbeat_blocks_even_before_api_connection(root):
    (root / "artifacts/active_ai.json").write_text(json.dumps({"running": True, "updated_unix": 100,
        "pid": 333}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="心跳"):
        starter.launch(root, now=lambda: 101, alive=lambda _: True,
            popen=lambda *a, **k: pytest.fail("duplicate process"),
            client_factory=lambda _: pytest.fail("must refuse before connecting"))


def test_live_start_lock_covers_first_heartbeat_gap(root):
    (root / "artifacts/mining-hot-start.lock").write_text('{"pid": 333}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="启动或运行"):
        starter.launch(root, alive=lambda _: True, popen=lambda *a, **k: pytest.fail("duplicate process"))


def test_unreachable_live_endpoint_does_not_open_second_serial_owner(root):
    endpoint(root)
    class Unreachable:
        def request(self, route):raise OSError("connection timed out")
    with pytest.raises(RuntimeError, match="未重复打开串口"):
        starter.launch(root, client_factory=lambda _: Unreachable(), alive=lambda _: True,
                       popen=lambda *a, **k: pytest.fail("must not open another GUI"))
