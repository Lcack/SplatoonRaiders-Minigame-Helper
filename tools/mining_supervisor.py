"""Supervised real-NS MINING rounds through the existing task interface.

No replay source is accepted. The task GUI owns capture, transport and their
authoritative logs. This tool records separate observations/advice for review.
"""
from dataclasses import asdict
from collections import deque
import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time
from urllib.request import ProxyHandler, Request, build_opener

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from raiders_helper.capture.models import FramePacket
from raiders_helper.control.buttons import ControllerState
from raiders_helper.trace import state_dict
from raiders_helper.mining_forward_seed import ForwardSeedEstimator
from raiders_helper.mining_mixed_guard import MixedDeadlineGuard


class LiveClient:
    def __init__(self,endpoint,frame_width=None):
        data=json.loads(Path(endpoint).read_text(encoding="utf-8"))
        if not data.get("running") or not data["url"].startswith("http://127.0.0.1:"):
            raise ValueError("需要正在运行的本机实机接口")
        self.url=data["url"].rstrip("/")
        self.headers={data.get("token_header","X-Raiders-Token"):data["token"],"Content-Type":"application/json"}
        self.local=threading.local()
        self.frame_width=frame_width

    def request(self,path,payload=None):
        if not hasattr(self.local,"opener"):
            self.local.opener=build_opener(ProxyHandler({}))
        request=Request(self.url+path,data=None if payload is None else json.dumps(payload).encode(),headers=self.headers)
        with self.local.opener.open(request,timeout=3) as response:
            body=response.read()
            return (body,dict(response.headers)) if path.startswith("/frame") else json.loads(body)

    def command(self,payload,wait=False):
        result=self.request("/commands",payload)
        if wait:
            deadline=time.perf_counter()+3
            while result["status"] in ("queued","running") and time.perf_counter()<deadline:
                time.sleep(.015)
                result=self.request("/commands/"+result["id"])
            if result["status"]!="succeeded":
                raise RuntimeError(str(result))
        return result

    def frame(self):
        route="/frame?kind=capture"
        if self.frame_width is not None:route+=f"&max_width={self.frame_width}"
        body,headers=self.request(route)
        self.last_frame_png=body
        self.last_frame_headers=headers
        frame=cv2.imdecode(np.frombuffer(body,np.uint8),cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("采集图像解码失败")
        now=time.perf_counter()
        age=float(headers.get("X-Frame-Age-Seconds",0))
        timestamp=float(headers.get("X-Frame-Captured-At",now-age))
        return FramePacket(frame,timestamp,int(headers["X-Frame-Sequence"]),60)


class LatestLease:
    def __init__(self,client,*,clock=None):
        self.client=client
        self.clock=clock or time.perf_counter
        self.lock=threading.Lock();self.stop=threading.Event()
        self.state=ControllerState.neutral();self.updated=0;self.error=None
        self.release_at=None;self.release_state=ControllerState.neutral()
        self.last=None;self.last_sent=0
        self.thread=threading.Thread(target=self.run,daemon=True,name="supervisor-lease")

    def update(self,state,*,release_at=None,release_state=None):
        with self.lock:
            self.state,self.updated=state,self.clock()
            self.release_at=release_at
            self.release_state=release_state or ControllerState.neutral()

    def tick(self):
        """One worker dispatch pass; injected clock supports deadline races."""
        with self.lock:
            state,updated=self.state,self.updated
            release_at,release_state=self.release_at,self.release_state
        now=self.clock()
        if now-updated>.25:
            state=ControllerState.neutral()
        elif release_at is not None and now>=release_at:
            state=release_state
        if state==ControllerState.neutral():
            if self.last!=state:self.client.command({"command":"release"})
        elif state!=self.last or now-self.last_sent>=.045:
            self.client.command({"command":"state",**state_dict(state),"hold_ms":300})
            self.last_sent=now
        self.last=state
        return state

    def run(self):
        try:
            while not self.stop.is_set():
                self.tick()
                self.stop.wait(.005)
        except Exception as error:
            self.error=str(error);self.stop.set()

    def close(self):
        self.stop.set();self.thread.join(timeout=3)
        try:self.client.command({"command":"release"})
        except Exception:pass


class PlanningDelayBudget:
    """Prospective compute-time budget from previous playing calls only."""
    def __init__(self, *, enabled=False, initial=.05, minimum=.03, maximum=.08,
                 window=32, quantile=.95, warmup=8):
        if type(enabled) is not bool:
            raise ValueError("planning_delay_budget_enabled 必须为布尔值")
        if (any(type(v) not in (int,float) or not math.isfinite(v) for v in (initial,minimum,maximum,quantile))
                or not 0<=minimum<=initial<=maximum<=.15 or not .5<=quantile<=1
                or type(window) is not int or not 8<=window<=128
                or type(warmup) is not int or not 1<=warmup<=window):
            raise ValueError("planning delay budget 配置无效")
        self.enabled=enabled;self.initial=initial;self.minimum=minimum;self.maximum=maximum
        self.quantile=quantile;self.warmup=warmup;self.samples=deque(maxlen=window)

    @classmethod
    def from_parameters(cls,parameters):
        return cls(enabled=parameters.get("planning_delay_budget_enabled",False),
            initial=parameters.get("planning_delay_initial_seconds",.05),
            minimum=parameters.get("planning_delay_minimum_seconds",.03),
            maximum=parameters.get("planning_delay_maximum_seconds",.08),
            window=parameters.get("planning_delay_budget_window",32),
            quantile=parameters.get("planning_delay_budget_quantile",.95))

    def attach(self,observation):
        if not self.enabled:return
        estimate=self.initial;source="configured_initial_compute_budget"
        if len(self.samples)>=self.warmup:
            ordered=sorted(self.samples);position=(len(ordered)-1)*self.quantile
            lower=int(position);upper=min(lower+1,len(ordered)-1)
            estimate=ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)
            estimate=max(self.minimum,min(self.maximum,estimate))
            source="rolling_previous_playing_policy_duration_quantile"
        observation.metrics.update(planning_delay_seconds=estimate,planning_delay_source=source,
            planning_delay_samples=len(self.samples),planning_delay_quantile=self.quantile,
            planning_delay_is_budget_not_current_measurement=True)

    def record(self,seconds,phase):
        if not self.enabled or phase!="playing":return
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:
            raise ValueError("measured policy duration 必须有限且非负")
        self.samples.append(seconds)


class SubmittedInputHistory:
    """Planning history from the GUI's post-submit events, not fit timestamps.

    These are recorded just after successful guard submission. Calibration
    still uses authoritative backend before/after in commands.jsonl.
    """
    def __init__(self,client):
        self.client=client;self.cursor=0;self.items=deque(maxlen=256)

    def attach(self,observation):
        events=self.client.request(f"/events?limit=256&after={self.cursor}")["events"]
        gap_detected=False
        for event in events:
            if self.cursor and event["id"]!=self.cursor+1:
                self.items.clear();gap_detected=True
            self.cursor=event["id"]
            self.items.append({"after":event["timestamp"],"state":event["state"]})
        now=time.perf_counter()
        cutoff=observation.timestamp-.6
        while len(self.items)>1 and self.items[1]["after"]<cutoff:self.items.popleft()
        observation.metrics.update(observation_age_seconds=max(0.,now-observation.timestamp),
            mining_input_history_source="guard_submission_after",
            mining_input_history=list(self.items),mining_input_history_observed_until=now,
            mining_input_history_gap_detected=gap_detected)


def run_round(client,recognizer,policy,parameters,output,seconds=65,*,telemetry=None):
    from mining_pulse_supervisor import ContactEvidence
    planning_budget=PlanningDelayBudget.from_parameters(parameters)
    forward_seed=ForwardSeedEstimator(parameters)
    mixed_guard=MixedDeadlineGuard(parameters)
    output.mkdir(parents=True,exist_ok=False)
    status=client.request("/status")
    if status.get("game")!="mining" or not status.get("control_enabled"):
        raise ValueError("需要已连接控制器的MINING人工监督会话")
    screen=status.get("screen")
    if screen not in ("result:retry","result:quit") and screen!="menu:mining":
        raise ValueError(f"本轮应从明确的结算页开始，当前{screen}")
    client.command({"command":"pause"},wait=True)
    if screen=="result:quit":
        client.command({"command":"press","buttons":["DPAD_UP"],"hold_ms":80},wait=True)
        time.sleep(.25)
        if client.request("/status").get("screen")!="result:retry":
            raise RuntimeError("未确认RETRY光标")
    client.command({"command":"press","buttons":["A"],"hold_ms":80},wait=True)
    lease=LatestLease(client);lease.thread.start()
    started=time.perf_counter();last_sequence=-1;seen_playing=False;result=None;records=[];last_snapshot=0
    inputs=SubmittedInputHistory(client)
    evidence=ContactEvidence(max_windows=4)
    problem_frames=[];problem_counts={};last_problem={}
    try:
        with (output/"trace.jsonl").open("w",encoding="utf-8") as trace:
            while time.perf_counter()-started<seconds:
                if lease.error:raise RuntimeError(lease.error)
                fetch_started=time.perf_counter()
                packet=client.frame()
                fetched=time.perf_counter()
                if packet.sequence==last_sequence:
                    time.sleep(.005);continue
                last_sequence=packet.sequence
                obs=recognizer.observe(packet)
                recognized=time.perf_counter()
                inputs.attach(obs)
                forward_seed.attach(obs)
                planning_budget.attach(obs)
                mixed_guard.attach(obs,now=time.perf_counter())
                history_attached=time.perf_counter()
                decision=policy(obs,parameters)
                now=time.perf_counter()
                planning_budget.record(now-history_attached,obs.phase)
                fresh=0<=now-packet.timestamp<=.2
                state,release_at,execution_guard=mixed_guard.apply(obs,decision,now=now,fresh=fresh)
                lease.update(state,release_at=release_at,release_state=ControllerState.neutral())
                seen_playing=seen_playing or obs.phase=="playing"
                # Keep a few exact failure frames in bounded memory. Encode
                # after releasing input, so evidence capture adds no PNG I/O
                # to the live control loop and never records idle menus.
                mode=decision.diagnostics.get("mode")
                if telemetry is not None:
                    telemetry.publish(reason=decision.reason,mode=mode or obs.phase,sequence=packet.sequence)
                if seen_playing:
                    evidence.add(packet,client.last_frame_png,obs,{"stage":"aborted" if
                        mode in {"wall_recovery","collision_recovery","contact_mitigation"} else "observing"})
                problem=(mode if mode in {"wall_recovery","collision_recovery","contact_mitigation"}
                         else "no_plan" if obs.phase=="playing" and mode is None
                         else "missing_body" if seen_playing and obs.phase=="unknown" else None)
                if (problem and len(problem_frames)<8 and problem_counts.get(problem,0)<2
                        and now-last_problem.get(problem,0)>3):
                    problem_frames.append((packet.sequence,problem,packet.frame.copy()))
                    problem_counts[problem]=problem_counts.get(problem,0)+1;last_problem[problem]=now
                row={"live":True,"executed":False,"scope":"advice_join_commands_jsonl_for_actual_transport",
                     "client_timestamp":now,"observation":asdict(obs),"execution_guard":execution_guard,"decision":{"state":state_dict(state),
                     "reason":decision.reason,"diagnostics":decision.diagnostics},"frame_age_ms":(now-packet.timestamp)*1000,
                     "command_suppressed_stale":not fresh,"policy_state":state_dict(decision.state),
                     "timing_ms":{"frame_request":(fetched-fetch_started)*1000,
                         "age_when_fetched":(fetched-packet.timestamp)*1000,
                         "recognition":(recognized-fetched)*1000,
                         "input_history":(history_attached-recognized)*1000,
                         "policy":(now-history_attached)*1000}}
                trace.write(json.dumps(row,ensure_ascii=False)+"\n");records.append(row)
                if now-last_snapshot>1:
                    cv2.imencode(".jpg",packet.frame,[cv2.IMWRITE_JPEG_QUALITY,85])[1].tofile(str(output/f"frame-{packet.sequence:08d}.jpg"))
                    current=client.request("/status")
                    print(f"{now-started:.1f}s {obs.phase} {decision.reason}",flush=True)
                    last_snapshot=now
                    if seen_playing and current.get("screen","").startswith("result:"):
                        result=current["screen"];break
                    if not current.get("control",{}).get("paused"):
                        raise RuntimeError("输入所有权已交还其他操作，停止监督")
            body,_=client.request("/frame?kind=capture")
            (output/"last-frame.png").write_bytes(body)
    finally:
        lease.close()
        if telemetry is not None:
            telemetry.publish(force=True,running=False,mode="round_finished",reason="本局已结束，控制已释放")
        evidence.save(output)
    for sequence,problem,frame in problem_frames:
        cv2.imencode(".png",frame)[1].tofile(str(output/f"problem-{sequence:08d}-{problem}.png"))
    report={"session":status["output"],"wall_seconds":time.perf_counter()-started,
            "frames":len(records),"playing_frames":sum(r["observation"]["phase"]=="playing" for r in records),
            "result":result,"parameters":parameters,"problem_frames":len(problem_frames),
            "scope":"supervised_real_NS_round_not_validation_accuracy"}
    if result is not None:
        from raiders_helper.mining_result_reader import inspect_result
        result_frame=cv2.imdecode(np.frombuffer((output/"last-frame.png").read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        report["result_reading"]=asdict(inspect_result(result_frame))
    (output/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


def main():
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description="经GUI接口监督MINING真实回合，结束归中，保留证据")
    parser.add_argument("--endpoint",type=Path,default=ROOT/"artifacts/control_endpoint.json")
    parser.add_argument("--profile",type=Path)
    parser.add_argument("--parameters",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seconds",type=float,default=65)
    parser.add_argument("--frame-width",type=int,choices=range(320,1921),metavar="320..1920",
                        help="接口取图最大宽度；默认原始分辨率，960可降低PNG开销")
    args=parser.parse_args()
    from raiders_helper.mining_vision import MiningLiveRecognizer
    from raiders_helper.mining_reactive import decide_mining_reactive
    profile=json.loads(args.profile.read_text(encoding="utf-8")) if args.profile else {}
    parameters=json.loads(args.parameters.read_text(encoding="utf-8")) if args.parameters else {}
    recognizer=MiningLiveRecognizer("mining",profile)
    print(json.dumps(run_round(LiveClient(args.endpoint,args.frame_width),recognizer,decide_mining_reactive,parameters,args.output,args.seconds),ensure_ascii=False))


if __name__=="__main__":main()
