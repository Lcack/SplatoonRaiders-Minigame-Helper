"""Actual loaded AI version heartbeat; no input or hot-reload side effects."""
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


class ActiveAIReporter:
    def __init__(self,metadata,path=None,interval=.5):
        self.path=Path(path or ROOT/"artifacts/active_ai.json")
        self.interval=interval;self.last_write=0.
        self.last_attempt_monotonic=float("-inf")
        self.write_failures=0;self.last_write_error=None
        self.state={"schema_version":1,"pid":os.getpid(),"running":True,**metadata}

    def publish(self,*,force=False,**updates):
        self.state.update(updates)
        now=time.time()
        monotonic=time.monotonic()
        if not force and monotonic-self.last_attempt_monotonic<self.interval:return False
        self.state["updated_unix"]=now
        # Serialisation errors are programming/data errors, not transient I/O.
        # Keep them visible instead of hiding them behind the heartbeat guard.
        payload=json.dumps(self.state,ensure_ascii=False,allow_nan=False)
        self.last_attempt_monotonic=monotonic
        temporary=self.path.with_name(self.path.name+f".{os.getpid()}.tmp")
        try:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temporary.write_text(payload,encoding="utf8")
            temporary.replace(self.path)
        except PermissionError as error:
            # Windows readers/antivirus may briefly deny replace. A version
            # display must not abort real-time control. Do not sleep or retry
            # in the control loop; the next interval publishes the latest
            # state. The existing file remains intact and expires normally.
            self.write_failures+=1;self.last_write_error=str(error)
            return False
        self.last_write=now;self.last_write_error=None
        return True

    def close(self):
        # Input has already been released by the worker. Make only a bounded
        # best effort, without sleeps; a persistent lock expires the old
        # heartbeat in the reader instead of blocking process shutdown.
        for _ in range(3):
            if self.publish(force=True,running=False,mode="stopped",reason="AI进程已停止，控制已释放"):
                return True
        return False

    def __enter__(self):
        self.publish(force=True);return self

    def __exit__(self,*_):self.close()
