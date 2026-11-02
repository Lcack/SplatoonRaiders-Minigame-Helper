"""Loopback diagnostics and main-loop-owned manual control.

The discovery/HTTP shapes follow AutoTableTurf's local control server. HTTP
threads never touch a controller. ``take_manual`` must run on the same loop
that owns automatic control; its non-None Decision replaces automatic input.
"""
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from .control.buttons import Button, ControllerState, StickVector
from .models import Decision
from .trace import state_dict


COMMANDS = frozenset({"pause", "resume", "release", "press", "state"})
INPUT_COMMANDS = frozenset({"press", "state"})


def _freeze(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _clock(value):
    value = time.perf_counter() if value is None else value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("time must be a finite monotonic timestamp")
    return float(value)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = False
    request_queue_size = 8


class DebugHub:
    def __init__(self, session_dir, *, control_allowed=False, preferred_port=0,
                 discovery_path=None, max_events=256, capture_provider=None):
        if type(preferred_port) is not int or not 0 <= preferred_port <= 65535:
            raise ValueError("preferred_port must be 0..65535")
        if type(max_events) is not int or not 1 <= max_events <= 2048:
            raise ValueError("max_events must be 1..2048")
        if capture_provider is not None and not callable(capture_provider):
            raise ValueError("capture_provider must be callable or None")
        self._capture_provider = capture_provider
        self.session_dir = Path(session_dir).resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.discovery_path = Path(discovery_path).resolve() if discovery_path else self.session_dir.parent/"control_endpoint.json"
        self.preferred_port = preferred_port
        self._token = secrets.token_urlsafe(32)
        self._lock = threading.RLock()
        self._control_allowed = bool(control_allowed)
        self._paused = False
        self._closed = False
        self._status = {}
        self._frames = {}
        self._generation = 0
        self._queue = deque()
        self._receipts = OrderedDict()
        self._events = deque(maxlen=max_events)
        self._next_event = 1
        self._active = None
        self._release_receipt = None
        self._server = self._thread = self._endpoint = None

    @property
    def paused(self):
        with self._lock:
            return self._paused

    @property
    def endpoint(self):
        """Includes the capability token for local discovery; never log it."""
        with self._lock:
            return _freeze(self._endpoint) if self._endpoint else None

    def publish(self, status=None, *, frame=None, frame_png=None, sequence=None,
                captured_at=None, kind="capture", control_allowed=None):
        """Publish raw or encoded frames; raw frames are copied, encoded lazily.

        Callers can publish at a display/debug cadence rather than every control
        tick. ``captured_at`` uses perf_counter, the same domain as FramePacket.
        """
        if kind not in {"capture", "analysis"}:
            raise ValueError("kind must be capture or analysis")
        if frame is not None and frame_png is not None:
            raise ValueError("supply frame or frame_png, not both")
        if status is not None and not isinstance(status, dict):
            raise ValueError("status must be a JSON object")
        frozen = _freeze(status) if status is not None else None
        raw = None
        if frame is not None:
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
                raise ValueError("frame must be a nonempty uint8 BGR image")
            raw = frame.copy()
        if frame_png is not None and (not isinstance(frame_png, bytes) or not frame_png or len(frame_png)>24*1024*1024):
            raise ValueError("frame_png must be 1..24MiB bytes")
        stamp = _clock(captured_at)
        with self._lock:
            if frozen is not None:
                self._status = frozen
            if control_allowed is not None:
                self._control_allowed = bool(control_allowed)
                if not self._control_allowed:
                    self._paused = True
                    self._cancel_controls("runtime control permission revoked")
            if raw is not None or frame_png is not None:
                self._generation += 1
                self._frames[kind] = {"raw": raw, "encoded": frame_png, "sequence": sequence,
                                      "captured_at": stamp, "generation": self._generation}

    def _capture_packet(self):
        """Read the capture owner's immutable latest packet, outside hub locks.

        The provider must be a nonblocking snapshot getter, never a device
        read. None means no frame; an older analysis frame is not a fallback.
        """
        with self._lock:
            if self._closed:
                raise FileNotFoundError("debug interface closed")
            provider = self._capture_provider
        if provider is None:
            return None, False
        packet = provider()
        with self._lock:
            if self._closed or self._capture_provider is not provider:
                raise FileNotFoundError("capture provider closed")
        if packet is not None:
            _clock(packet.timestamp)
            if type(packet.sequence) is not int or packet.sequence < 0:
                raise ValueError("capture sequence must be a nonnegative integer")
        return packet, True

    def frame(self, kind="capture", max_width=None):
        """Return one source snapshot, optionally reduced before PNG encoding.

        Reduced images are request-local; only the original PNG is cached.
        Source sequence and capture time always describe the same snapshot.
        """
        if max_width is not None and (type(max_width) is not int or not 320 <= max_width <= 1920):
            raise ValueError("max_width must be an integer in 320..1920")
        packet, provided = self._capture_packet() if kind == "capture" else (None, False)
        if provided:
            if packet is None:
                raise FileNotFoundError("capture frame not available")
            frame = packet.frame
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
                    or frame.shape[2] != 3 or not frame.size):
                raise ValueError("capture provider frame must be nonempty uint8 BGR")
            # One request owns one pixel snapshot and its matching metadata.
            item = {"raw": frame.copy(), "encoded": None, "sequence": packet.sequence,
                    "captured_at": packet.timestamp, "generation": None}
        else:
            with self._lock:
                if self._closed:
                    raise FileNotFoundError("debug interface closed")
                item = self._frames.get(kind)
                if item is None:
                    raise FileNotFoundError("frame not available")
                item = dict(item)
        data = item["encoded"]
        raw = item["raw"]
        reduced = False
        if max_width is not None:
            if raw is None:
                raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if raw is None:
                    raise OSError("frame decoding failed")
            height, width = raw.shape[:2]
            if width > max_width:
                raw = cv2.resize(raw, (max_width, max(1, round(height*max_width/width))),
                                 interpolation=cv2.INTER_AREA)
                data, reduced = None, True
        if data is None:
            ok, encoded = cv2.imencode(".png", raw)
            if not ok:
                raise OSError("frame encoding failed")
            data = encoded.tobytes()
            if not reduced and not provided:
                with self._lock:
                    current = self._frames.get(kind)
                    if current and current["generation"] == item["generation"]:
                        current["encoded"] = data
        with self._lock:
            if self._closed:
                raise FileNotFoundError("debug interface closed")
        return data, {"sequence": item["sequence"], "captured_at": item["captured_at"],
                      "age_seconds": max(0.0, time.perf_counter()-item["captured_at"]),
                      "source": "capture_worker_latest" if provided else "memory"}

    def status(self):
        try:
            packet, provided = self._capture_packet()
        except FileNotFoundError:
            packet, provided = None, False
        now = time.perf_counter()
        with self._lock:
            value = _freeze(self._status)
            value["control"] = {"running": not self._closed, "paused": self._paused,
                                "allowed": self._control_allowed,
                                "pending_commands": len(self._queue),
                                "command_names": sorted(COMMANDS),
                                "manual_active": self._active is not None}
            value["frames"] = {name: {"available": True, "sequence": item["sequence"],
                                      "age_seconds": max(0.0, now-item["captured_at"])}
                               for name, item in self._frames.items()}
            if self._closed:
                value["frames"] = {}
            elif provided:
                value["frames"]["capture"] = ({"available": False, "source": "capture_worker_latest"}
                    if packet is None else {"available": True, "sequence": packet.sequence,
                        "age_seconds": max(0.0, now-packet.timestamp), "source": "capture_worker_latest"})
            return value

    def record_control(self, state, *, timestamp=None, sequence=None, source="auto", command_id=None):
        """Record only after an actual successful controller submit/release.

        This confirms PC transport submission, never console receipt or game
        response. An expired manual pulse succeeds after its neutral submission.
        """
        if isinstance(state, dict):
            state = ControllerState(frozenset(Button(value) for value in state.get("buttons", [])),
                                    StickVector(**state.get("left_stick", {})),
                                    StickVector(**state.get("right_stick", {})))
        if not isinstance(state, ControllerState):
            raise ValueError("record_control expects ControllerState or its JSON dict")
        stamp = _clock(timestamp)
        with self._lock:
            event = {"id": self._next_event, "timestamp": stamp, "sequence": sequence,
                     "source": str(source), "state": state_dict(state), "command_id": command_id}
            self._next_event += 1
            self._events.append(event)
            if self._active and state == self._active["state"]:
                if command_id is None or command_id == self._active["id"]:
                    self._active["applied"] = True
            if self._release_receipt and state == ControllerState.neutral():
                identifier, applied = self._release_receipt
                self._finish(identifier, error=None if applied else "manual pulse was not submitted")
                self._release_receipt = None

    def events(self, limit=64, after=0):
        if type(limit) is not int or not 1 <= limit <= 256 or type(after) is not int or after < 0:
            raise ValueError("limit must be 1..256 and after must be nonnegative")
        with self._lock:
            return _freeze([event for event in self._events if event["id"]>after][-limit:])

    def _finish(self, identifier, error=None):
        receipt = self._receipts.get(identifier)
        if receipt is not None:
            receipt.update(status="failed" if error else "succeeded", finished_unix=time.time())
            if error:
                receipt["error"] = error

    def _cancel_controls(self, reason):
        remaining = deque()
        for command in self._queue:
            if command["command"] in INPUT_COMMANDS:
                self._finish(command["id"], reason)
            else:
                remaining.append(command)
        self._queue = remaining
        if self._active:
            self._finish(self._active["id"], reason)
            self._active = None
        if self._release_receipt:
            self._finish(self._release_receipt[0], reason)
            self._release_receipt = None

    @staticmethod
    def _validate(payload):
        if not isinstance(payload, dict):
            raise ValueError("command must be a JSON object")
        name = payload.get("command")
        if not isinstance(name, str) or name not in COMMANDS:
            raise ValueError("command must be pause, resume, release, press or state")
        allowed = {"command", "buttons", "left_stick", "right_stick", "hold_ms"} if name in INPUT_COMMANDS else {"command"}
        if set(payload)-allowed:
            raise ValueError("unexpected command fields")
        result = {"command": name}
        if name not in INPUT_COMMANDS:
            return result
        buttons = payload.get("buttons", [])
        if not isinstance(buttons, list) or len(buttons)>8 or any(not isinstance(b,str) for b in buttons) or len(set(buttons))!=len(buttons):
            raise ValueError("buttons must be a list of distinct Button names")
        try:
            keys = frozenset(Button(b) for b in buttons)
        except ValueError:
            raise ValueError("unknown controller button") from None
        sticks = []
        for key in ("left_stick", "right_stick"):
            vector = payload.get(key, {})
            if not isinstance(vector, dict) or set(vector)-{"x", "y"}:
                raise ValueError("sticks must contain integer x/y values")
            x, y = vector.get("x", 0), vector.get("y", 0)
            if type(x) is not int or type(y) is not int:
                raise ValueError("stick x/y must be integers")
            sticks.append(StickVector(x, y))
        state = ControllerState(keys, *sticks)
        if name == "press" and state == ControllerState.neutral():
            raise ValueError("use release for a neutral state")
        hold = payload.get("hold_ms", 150 if name == "state" else 120)
        if type(hold) is not int or not 30 <= hold <= 1000:
            raise ValueError("hold_ms must be an integer in 30..1000")
        return {**result, "buttons": buttons, "left_stick": {"x": sticks[0].x,"y": sticks[0].y},
                "right_stick": {"x": sticks[1].x,"y": sticks[1].y}, "hold_ms": hold, "state": state}

    def submit(self, payload):
        command = self._validate(payload)
        now = time.perf_counter()
        with self._lock:
            if self._closed:
                raise RuntimeError("debug interface is closed")
            name = command["command"]
            if name in INPUT_COMMANDS:
                if not self._control_allowed:
                    raise PermissionError("runtime has not enabled hardware control")
                if not self._paused:
                    raise RuntimeError("pause automatic control before manual input")
                if name == "press" and (self._active or self._release_receipt
                                         or any(c["command"] in INPUT_COMMANDS for c in self._queue)):
                    raise RuntimeError("one manual pulse is already pending")
            if name == "state" and command["state"] != ControllerState.neutral():
                # A pointer/axis may change faster than the capture/owner loop.
                # Keep the latest requested complete state, never a FIFO of old
                # mouse positions or held keys. A queued neutral/urgent release
                # remains ahead of it and can still cancel the new request.
                remaining = deque()
                for pending in self._queue:
                    if pending["command"] in INPUT_COMMANDS and pending["state"] != ControllerState.neutral():
                        self._finish(pending["id"], "manual state superseded before submission")
                    else:
                        remaining.append(pending)
                self._queue = remaining
            if len(self._queue)>=32:
                raise RuntimeError("command queue is full")
            identifier = secrets.token_hex(12)
            command.update(id=identifier, submitted_at=now)
            receipt = {k:v for k,v in command.items() if k not in {"state", "submitted_at"}}
            receipt.update(status="queued", submitted_unix=time.time())
            self._receipts[identifier] = receipt
            if name in {"pause", "release"} or (name == "state" and command["state"] == ControllerState.neutral()):
                self._queue.appendleft(command)
            else:
                self._queue.append(command)
            while len(self._receipts)>256:
                first = next(iter(self._receipts))
                if self._receipts[first]["status"] in {"queued", "running"}:
                    break
                self._receipts.pop(first)
            return _freeze(receipt)

    def command_result(self, identifier):
        with self._lock:
            value = self._receipts.get(identifier)
            return _freeze(value) if value is not None else None

    def take_manual(self, now=None):
        """Owner-loop pump. None permits automation; a Decision overrides it.

        Pause/release/resume always expose a neutral owner-loop tick. ``press``
        is one finite pulse. ``state`` atomically changes/renews a bounded
        complete manual state without neutral gaps while its lease is current.
        A state lease ends at submitted_at + hold_ms; polling does not renew it.
        """
        now = _clock(now)
        with self._lock:
            transition = False
            while self._queue:
                command = self._queue.popleft()
                identifier, name = command["id"], command["command"]
                self._receipts[identifier]["status"] = "running"
                neutral_state = name == "state" and command["state"] == ControllerState.neutral()
                if name in {"pause", "release", "resume"} or neutral_state:
                    self._cancel_controls("manual input cancelled by "+("neutral state" if neutral_state else name))
                    if name in {"pause", "release"} or neutral_state:
                        # An already queued resume must not undo an urgent stop.
                        for pending in self._queue:
                            self._finish(pending["id"], "queued command cancelled by "+name)
                        self._queue.clear()
                    self._paused = name != "resume"
                    if neutral_state:
                        # A neutral state update is acknowledged by the actual
                        # transport callback, not merely by queue consumption.
                        self._release_receipt = (identifier, True)
                    else:
                        self._finish(identifier)
                    transition = True
                    continue
                if transition:
                    self._queue.appendleft(command)
                    self._receipts[identifier]["status"] = "queued"
                    break
                until = command["submitted_at"]+command["hold_ms"]/1000 if name == "state" else now+command["hold_ms"]/1000
                if (not self._paused or not self._control_allowed or now-command["submitted_at"]>1.0
                        or (name == "state" and now >= until)):
                    self._finish(identifier, "manual request is stale or control ownership changed")
                    continue
                if self._release_receipt:
                    # A previous lease expired and its neutral state has not
                    # yet reached the transport. Preserve that release tick.
                    self._queue.appendleft(command)
                    self._receipts[identifier]["status"] = "queued"
                    break
                if (self._active and now >= self._active["until"]
                        and command["submitted_at"] > self._active["until"]):
                    # A late update cannot silently resurrect an expired hold.
                    active, self._active = self._active, None
                    self._release_receipt = (active["id"], active["applied"])
                    self._queue.appendleft(command)
                    self._receipts[identifier]["status"] = "queued"
                    break
                if self._active:
                    active = self._active
                    if active["command"] == "state" and active["applied"]:
                        self._finish(active["id"])
                        self._receipts[active["id"]].update(completion="superseded", superseded_by=identifier)
                    else:
                        self._finish(active["id"], "manual input replaced by a state update")
                self._active = {**command, "until": until, "applied": False}
                break
            if transition or self._closed:
                return Decision(reason="调试：控制归中，切换输入所有权", diagnostics={"debug":{
                    "source":"manual","paused":self._paused,"transport_required":True}})
            if self._active and now >= self._active["until"]:
                active, self._active = self._active, None
                self._release_receipt = (active["id"], active["applied"])
            if self._active:
                return Decision(self._active["state"], "调试：有时限的手动输入", diagnostics={
                    "debug":{"source":"manual","paused":True,"command_id":self._active["id"],
                             "until":self._active["until"],"transport_required":True}})
            if self._paused:
                return Decision(reason="调试：自动控制已暂停，保持中立", diagnostics={"debug":{
                    "source":"manual","paused":True,"transport_required":self._release_receipt is not None}})
            return None

    def _write_discovery(self):
        for path in {self.session_dir/"control_endpoint.json", self.discovery_path}:
            if self._closed and path.is_file():
                try:
                    if json.loads(path.read_text(encoding="utf-8")).get("token") != self._token:
                        continue
                except (OSError, ValueError):
                    continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name+"."+secrets.token_hex(6)+".tmp")
            temporary.write_text(json.dumps(self._endpoint, ensure_ascii=False, indent=2),encoding="utf-8")
            try:
                temporary.chmod(0o600)
            except OSError:
                pass
            temporary.replace(path)

    def start(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("closed debug hub cannot restart")
            if self._server:
                return self.endpoint
            owner = self

            class Handler(BaseHTTPRequestHandler):
                server_version, sys_version = "RaidersDebug/1", ""

                def setup(self):
                    super().setup()
                    self.connection.settimeout(3)

                def log_message(self, *_):
                    pass

                def send(self, code, value, content_type="application/json; charset=utf-8", headers=None):
                    data = value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False,allow_nan=False).encode("utf-8")
                    self.send_response(code)
                    for key, val in {"Content-Type":content_type,"Content-Length":str(len(data)),
                                     "Cache-Control":"no-store, max-age=0","X-Content-Type-Options":"nosniff",
                                     "Connection":"close", **(headers or {})}.items():
                        self.send_header(key,str(val))
                    self.end_headers()
                    self.close_connection = True
                    try:
                        self.wfile.write(data)
                    except (BrokenPipeError,ConnectionResetError):
                        pass

                def auth(self):
                    try:
                        host = urlsplit("//"+self.headers.get("Host","")).hostname
                    except ValueError:
                        host = None
                    if host not in {"localhost","127.0.0.1"} or self.headers.get("Origin") is not None:
                        self.send(403,{"error":"local non-browser clients only"})
                        return False
                    supplied = self.headers.get("X-Raiders-Token",self.headers.get("X-Tableturf-Token",""))
                    if not secrets.compare_digest(supplied.encode("utf-8"),owner._token.encode("utf-8")):
                        self.send(401,{"error":"missing or invalid access token"})
                        return False
                    return True

                def do_GET(self):
                    if not self.auth():
                        return
                    route = urlsplit(self.path)
                    try:
                        if route.path == "/status" and not route.query:
                            self.send(200,owner.status())
                        elif route.path in {"/frame.png","/frame"}:
                            query = parse_qs(route.query,keep_blank_values=True)
                            if set(query)-{"kind","max_width"} or any(len(values)!=1 for values in query.values()):
                                raise ValueError("frame accepts one kind=capture|analysis and optional max_width=320..1920")
                            kind = query.get("kind",["capture"])[0]
                            if kind not in {"capture","analysis"}:
                                raise ValueError("unknown frame kind")
                            max_width = query.get("max_width",[None])[0]
                            if max_width is not None:
                                if not max_width.isascii() or not max_width.isdecimal():
                                    raise ValueError("max_width must be an integer in 320..1920")
                                max_width = int(max_width)
                            data, meta = owner.frame(kind,max_width=max_width)
                            self.send(200,data,"image/png",{"X-Frame-Sequence":meta["sequence"] if meta["sequence"] is not None else "",
                                      "X-Frame-Age-Seconds":f"{meta['age_seconds']:.4f}",
                                      "X-Frame-Captured-At":f"{meta['captured_at']:.9f}","X-Frame-Source":meta["source"]})
                        elif route.path == "/events":
                            query = parse_qs(route.query,keep_blank_values=True)
                            if set(query)-{"limit","after"} or any(len(v)!=1 for v in query.values()):
                                raise ValueError("events accepts one limit and after")
                            self.send(200,{"events":owner.events(int(query.get("limit",[64])[0]),int(query.get("after",[0])[0]))})
                        elif route.path.startswith("/commands/") and not route.query:
                            value = owner.command_result(route.path[len("/commands/"):])
                            self.send(200,value) if value else self.send(404,{"error":"unknown command"})
                        else:
                            self.send(404,{"error":"unknown endpoint"})
                    except ValueError as error:
                        self.send(400,{"error":str(error)})
                    except OSError:
                        self.send(503,{"error":"frame not available yet"})

                def do_POST(self):
                    if not self.auth():
                        return
                    direct = {"/pause":"pause","/resume":"resume","/release":"release","/control":"press"}
                    if self.path not in direct and self.path != "/commands":
                        self.send(404,{"error":"unknown endpoint"})
                        return
                    try:
                        if self.headers.get("Transfer-Encoding") is not None:
                            raise ValueError("chunked command bodies are not supported")
                        length = int(self.headers.get("Content-Length","0"))
                        if not 0 <= length <= 8192:
                            self.send(413,{"error":"command body must be at most 8192 bytes"})
                            return
                        if length and self.headers.get_content_type() != "application/json":
                            self.send(415,{"error":"commands require application/json"})
                            return
                        payload = json.loads(self.rfile.read(length)) if length else {}
                        if self.path in direct:
                            if not isinstance(payload,dict) or "command" in payload:
                                raise ValueError("direct endpoints do not accept command field")
                            payload = {"command":direct[self.path],**payload}
                        self.send(202,owner.submit(payload))
                    except PermissionError as error:
                        self.send(403,{"error":str(error)})
                    except (ValueError,UnicodeDecodeError) as error:
                        self.send(400,{"error":str(error)})
                    except RuntimeError as error:
                        self.send(409,{"error":str(error)})

            self._server = _Server(("127.0.0.1",self.preferred_port),Handler)
            self._endpoint = {"schema_version":1,"service":"raiders-debug","pid":os.getpid(),
                              "url":f"http://127.0.0.1:{self._server.server_port}","token":self._token,
                              "token_header":"X-Raiders-Token","running":True,
                              "session_dir":str(self.session_dir),"started_unix":time.time()}
            self._thread = threading.Thread(target=self._server.serve_forever,kwargs={"poll_interval":.05},daemon=True)
            try:
                self._write_discovery()
                self._thread.start()
            except Exception:
                self._server.server_close()
                self._server = None
                self._endpoint = None
                raise
            return self.endpoint

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed, self._paused = True, True
            self._capture_provider = None
            self._frames.clear()
            self._cancel_controls("debug interface closed")
            for receipt in self._receipts.values():
                if receipt["status"] in {"queued","running"}:
                    self._finish(receipt["id"],"debug interface closed")
            self._queue.clear()
            if self._endpoint:
                self._endpoint.update(running=False,stopped_unix=time.time())
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=1)
            self._write_discovery()
