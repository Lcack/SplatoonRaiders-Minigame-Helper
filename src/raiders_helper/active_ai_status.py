"""Read external-AI heartbeats without treating disk edits as loaded code.

This module never starts/reloads code or controls a game. A valid heartbeat is
a report from the external process, not proof that a particular decision was
executed. Transport logs remain authoritative for actual controller writes.
"""
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def _session_key(value, root):
    if value is None:
        return None
    try:
        path = Path(value)
        if not path.is_absolute():
            path = root/path
        return os.path.normcase(str(path.resolve())).casefold()
    except (OSError, TypeError, ValueError):
        return None


def _sha(value):
    return value.lower() if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) else None


def _round_reloadable_source(path):
    path = path.replace("\\", "/").lower()
    return ((path.startswith(("src/raiders_helper/mining_","src/raiders_helper/juggling")) and path.endswith(".py"))
            or path in {"src/raiders_helper/arcade_vision.py","src/raiders_helper/fast_tracking.py",
                        "tools/mining_supervisor.py","tools/juggling_supervisor.py","tools/read_mining_score_offline.py"}
            or path.startswith("assets/juggling_live_score/"))


def snapshot_loaded_terminal_sources(repository_root=ROOT):
    """Freeze startup files of this GUI's imported hardware/display owner.

    AI-core modules are deliberately excluded: their controlling instance is
    the fresh external child, rather than this GUI's advisory display engine.
    """
    root = Path(repository_root).resolve()
    terminal_modules = {"gui", "active_ai_status", "session_runtime", "manual_input_pump", "debug_api",
                        "runtime", "analysis_worker", "advice_overlay", "experiment_recording"}
    result = {}
    for name, module in tuple(sys.modules.items()):
        if not (name.startswith(("raiders_helper.control", "raiders_helper.capture")) or
                name in {"raiders_helper."+part for part in terminal_modules}):
            continue
        filename = getattr(module, "__file__", None)
        if not filename or Path(filename).suffix != ".py":
            continue
        path = Path(filename).resolve()
        if path.is_relative_to(root):
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


@dataclass(frozen=True)
class ActiveAIStatus:
    state: str
    message: str
    running: bool = False
    loaded_version: str | None = None
    updated_unix: float | None = None
    pid: int | None = None
    round_id: int | str | None = None
    mode: str = ""
    reason: str = ""
    profile_sha: str | None = None
    parameters_sha: str | None = None
    changed_sources: tuple[str, ...] = ()
    hot_changed_sources: tuple[str, ...] = ()
    restart_required_sources: tuple[str, ...] = ()
    unverifiable_sources: tuple[str, ...] = ()
    source_hashes: dict = field(default_factory=dict)

    @property
    def short_text(self):
        if self.running:
            prefix = "AI旧版" if self.changed_sources else "AI运行版"
            return f"{prefix} {self.loaded_version}"
        return {"stale":"AI状态过期", "session_mismatch":"AI未运行（其他会话）",
                "invalid":"AI状态未确认"}.get(self.state, "AI未运行")

    @property
    def summary_text(self):
        lines = [f"自主AI：{self.message}", f"实际运行版本：{self.loaded_version or '无有效运行心跳'}"]
        if self.running:
            if self.hot_changed_sources:
                lines.append("当前局旧版，下局自动更新")
            if self.unverifiable_sources:
                lines.append("源码版本未能核对，以已加载版本为准")
            if self.pid is not None:
                lines.append(f"AI进程：{self.pid} · 回合：{self.round_id if self.round_id is not None else '未提供'}")
            profile = self.profile_sha[:8] if self.profile_sha else "未提供"
            parameters = self.parameters_sha[:8] if self.parameters_sha else "未提供"
            lines.append(f"已加载配置/参数：{profile} / {parameters}")
        if self.restart_required_sources:
            lines.append("终端/控制源码已改，需重启终端生效")
        lines.append("AI核心/配置参数在局间更新；终端与控制层需重启")
        return "\n".join(lines)


def _read_active_ai_report(path, capture_session, *, repository_root=ROOT, now=None, max_age_seconds=3.0):
    root = Path(repository_root).resolve()
    current_session = _session_key(capture_session, root)
    if current_session is None:
        return ActiveAIStatus("not_running", "未运行（未连接采集）")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return ActiveAIStatus("not_running", "未运行（无有效心跳）")
    except (OSError, ValueError):
        return ActiveAIStatus("invalid", "状态未确认（心跳不可读）")
    if not isinstance(data, dict):
        return ActiveAIStatus("invalid", "状态未确认（心跳格式无效）")
    if _session_key(data.get("capture_session"), root) != current_session:
        return ActiveAIStatus("session_mismatch", "未运行（心跳属于其他采集会话）")
    stamp = data.get("updated_unix")
    clock = time.time() if now is None else now
    if (isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp)
            or not 0 <= clock-stamp < max_age_seconds):
        return ActiveAIStatus("stale", "状态过期（无3秒内有效心跳）")
    if data.get("running") is not True:
        return ActiveAIStatus("not_running", "未运行（外部程序已停止）", updated_unix=stamp)
    version = data.get("loaded_version")
    hashes = data.get("source_hashes")
    if (not isinstance(version, str) or not re.fullmatch(r"[0-9a-fA-F]{12,64}", version)
            or not isinstance(hashes, dict) or not hashes):
        return ActiveAIStatus("invalid", "状态未确认（运行版本信息不完整）")
    changed, unverifiable = [], []
    for relative, expected in hashes.items():
        if not isinstance(relative, str) or _sha(expected) is None:
            return ActiveAIStatus("invalid", "状态未确认（源码版本清单无效）")
        candidate = Path(relative)
        if candidate.is_absolute():
            return ActiveAIStatus("invalid", "状态未确认（源码路径超出项目）")
        try:
            candidate = (root/candidate).resolve()
            if not candidate.is_relative_to(root):
                return ActiveAIStatus("invalid", "状态未确认（源码路径超出项目）")
            actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError:
            unverifiable.append(relative)
            continue
        if actual != expected.lower() and _round_reloadable_source(relative):
            changed.append(relative)
    pid = data.get("pid")
    pid = pid if type(pid) is int and pid > 0 else None
    round_id = data.get("round")
    round_id = round_id if type(round_id) in (int, str) else None
    mode = data.get("mode")
    mode = mode.replace("\n", " ")[:60] if isinstance(mode, str) else ""
    reason = data.get("reason")
    reason = reason.replace("\n", " ")[:500] if isinstance(reason, str) else ""
    hot = tuple(path for path in changed if _round_reloadable_source(path))
    return ActiveAIStatus(state="source_changed" if changed else "source_unverified" if unverifiable else "running",
        message="运行中（外部自主程序）", running=True, loaded_version=version[:12].lower(),
        updated_unix=stamp, pid=pid, round_id=round_id, mode=mode, reason=reason, profile_sha=_sha(data.get("profile_sha")),
        parameters_sha=_sha(data.get("parameters_sha")), changed_sources=tuple(changed),
        hot_changed_sources=hot, unverifiable_sources=tuple(unverifiable),
        source_hashes={path.replace("\\", "/").casefold():sha.lower() for path,sha in hashes.items()})


def read_active_ai(path, capture_session, *, repository_root=ROOT, now=None, max_age_seconds=3.0,
                   terminal_source_hashes=None):
    """Read reported child versions; separately check this GUI's startup files."""
    root = Path(repository_root).resolve()
    value = _read_active_ai_report(path, capture_session, repository_root=root, now=now,
                                   max_age_seconds=max_age_seconds)
    changed, unverifiable = [], []
    for relative, expected in (terminal_source_hashes or {}).items():
        try:
            candidate = (root/relative).resolve()
            if not candidate.is_relative_to(root) or _sha(expected) is None:
                unverifiable.append(str(relative))
                continue
            actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except (OSError, TypeError, ValueError):
            unverifiable.append(str(relative))
            continue
        if actual != expected.lower():
            changed.append(relative)
    return replace(value, restart_required_sources=tuple(changed),
                   unverifiable_sources=tuple(sorted(set(value.unverifiable_sources) | set(unverifiable))))


class ActiveAIMonitor:
    """Poll disk at most once/second; expire cached heartbeats immediately."""
    def __init__(self, path, *, repository_root=ROOT, interval_seconds=1.0, terminal_source_hashes=None):
        self.path, self.root = Path(path), Path(repository_root).resolve()
        self.interval = interval_seconds
        self.terminal_source_hashes = dict(terminal_source_hashes or {})
        self._next_check = 0.0
        self._session = None
        self._value = ActiveAIStatus("not_running", "未运行（无有效心跳）")

    def poll(self, capture_session, *, now=None, monotonic=None):
        clock = time.time() if now is None else now
        tick = time.perf_counter() if monotonic is None else monotonic
        key = _session_key(capture_session, self.root)
        if tick >= self._next_check or key != self._session:
            self._value = read_active_ai(self.path, capture_session, repository_root=self.root, now=clock,
                                         terminal_source_hashes=self.terminal_source_hashes)
            self._session, self._next_check = key, tick+self.interval
        if (self._value.running and self._value.updated_unix is not None
                and not 0 <= clock-self._value.updated_unix < 3):
            return replace(self._value, state="stale", message="状态过期（无3秒内有效心跳）", running=False,
                           loaded_version=None, changed_sources=(), hot_changed_sources=())
        return self._value
