"""Collect consecutive real MINING pulse trials via the task GUI's sole input owner.

Planned pulse lengths are labels, never measured execution times. Analysis must
integrate authoritative commands.jsonl send-completed transitions. No video is
enabled here; a bounded PNG ring preserves a few candidate-contact windows.
"""
from collections import deque
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from mining_supervisor import LiveClient,LatestLease
from raiders_helper.control.buttons import ControllerState
from raiders_helper.trace import state_dict
from raiders_helper.mining_vision import MiningLiveRecognizer


class ContactEvidence:
    """Keep at most two merged candidate windows/round, using existing PNG bytes."""
    def __init__(self,max_windows=2):
        self.ring=deque();self.windows=[];self.active=None;self.missing_since=None
        self.max_windows=max_windows;self.triggers=[]

    def add(self,packet,png,obs,ident):
        t=packet.timestamp
        item=(t,packet.sequence,png)
        self.ring.append(item)
        while self.ring and t-self.ring[0][0]>2:self.ring.popleft()
        if self.active is not None:
            self.active["frames"][packet.sequence]=item
        trigger=None
        if obs.metrics.get("hit_body_detected"):trigger="observed_purple_body_candidate"
        elif ident.get("stage")=="aborted":trigger="trial_aborted_candidate"
        if trigger and obs.phase not in {"menu","countdown","result","result_pending"}:
            self.triggers.append({"timestamp":t,"sequence":packet.sequence,"reason":trigger})
            if self.active is None and len(self.windows)<self.max_windows:
                self.active={"start":t-2,"until":t+.7,"triggers":[],
                             "frames":{s:(a,s,b) for a,s,b in self.ring}}
                self.windows.append(self.active)
            if self.active is not None:
                self.active["triggers"].append({"timestamp":t,"sequence":packet.sequence,"reason":trigger})
                self.active["until"]=min(self.active["start"]+3,t+.7)
        if self.active is not None and t>=self.active["until"]:self.active=None

    def save(self,output):
        (output/"contact-events.json").write_text(json.dumps({"candidate_events":self.triggers,
            "saved_windows":len(self.windows),"max_windows":self.max_windows,
            "label":"candidates_not_confirmed_collision; image budget bounded; event timestamps retained"},ensure_ascii=False,indent=2),encoding="utf8")
        for index,window in enumerate(self.windows,1):
            folder=output/"contact-windows"/f"window-{index:02d}";folder.mkdir(parents=True)
            frames=sorted(window["frames"].values())
            for timestamp,sequence,png in frames:(folder/f"{sequence:08d}.png").write_bytes(png)
            (folder/"manifest.json").write_text(json.dumps({
                "label":"candidate_only_not_confirmed_collision",
                "triggers":window["triggers"],"frames":[{"timestamp":t,"sequence":s,"file":f"{s:08d}.png"} for t,s,_ in frames],
                "scope":"only_HTTP_sampled_frames; source_sequence_gaps_are_not_reconstructed"
            },ensure_ascii=False,indent=2),encoding="utf8")


def start_round(client):
    status=client.request("/status")
    if status.get("game")!="mining" or not status.get("control_enabled"):
        raise RuntimeError("需要已连接控制器的MINING任务GUI")
    screen=status.get("screen")
    if screen not in {"result:retry","result:quit","menu:mining"}:
        raise RuntimeError(f"本局需从明确的MINING菜单或结算页开始，当前 {screen}")
    client.command({"command":"pause"},wait=True)
    if screen=="result:quit":
        client.command({"command":"press","buttons":["DPAD_UP"],"hold_ms":80},wait=True)
        time.sleep(.25)
        if client.request("/status").get("screen")!="result:retry":raise RuntimeError("未确认RETRY光标")
    client.command({"command":"press","buttons":["A"],"hold_ms":80},wait=True)
    return status


def collect_round(client,profile,config,output,round_id,mode,max_seconds):
    from raiders_helper.mining_identification import MiningPhysicsProbe
    output.mkdir(parents=True,exist_ok=False)
    initial=start_round(client)
    recognizer=MiningLiveRecognizer("mining",profile)
    config={**config,**config.get("mode_overrides",{}).get(mode,{})}
    probe=MiningPhysicsProbe(config);probe.reset_round(round_id,mode=mode)
    lease=LatestLease(client);lease.thread.start()
    started=time.perf_counter();last_status=0;last_seq=-1;seen_playing=False
    evidence=ContactEvidence();events=[];frames=0;playing=0;result=None
    try:
        with (output/"trace.jsonl").open("w",encoding="utf8") as stream:
            while time.perf_counter()-started<max_seconds:
                if lease.error:raise RuntimeError(lease.error)
                packet=client.frame()
                if packet.sequence==last_seq:time.sleep(.003);continue
                last_seq=packet.sequence
                obs=recognizer.observe(packet)
                now=time.perf_counter();age=now-packet.timestamp
                # The probe checks visual freshness too, using the actual
                # capture timestamp, rather than tracker initialization speed.
                obs.metrics["observation_age_seconds"]=age
                decision=probe.decide(obs,now=now)
                ident=decision.diagnostics.get("identification",{})
                state=decision.state if 0<=age<=.2 and not obs.metrics.get("game_ended") else ControllerState.neutral()
                deadline=None
                if (ident.get("state_input_kind")=="probe" and ident.get("stage")=="pulse"
                        and ident.get("protocol")=="pulse_coast"):
                    deadline=ident["stage_started_at"]+ident["pulse_seconds"]
                # End simple holds on a monotonic deadline in the input
                # worker. The next screenshot is not the release timer.
                lease.update(state,release_at=deadline,
                             release_state=ControllerState(buttons=state.buttons))
                seen_playing=seen_playing or obs.phase=="playing"
                frames+=1;playing+=obs.phase=="playing"
                for event in ident.get("events",[]):events.append({"timestamp":now,"sequence":packet.sequence,**event})
                if seen_playing:evidence.add(packet,client.last_frame_png,obs,ident)
                stream.write(json.dumps({"live":True,"executed":False,
                    "scope":"probe_advice; join actual session commands.jsonl for physical submission",
                    "round_id":round_id,"mode":mode,"client_timestamp":now,"frame_age_ms":age*1000,
                    "observation":asdict(obs),"identification":ident,
                    "decision":{"state":state_dict(state),"reason":decision.reason,"diagnostics":decision.diagnostics}},ensure_ascii=False)+"\n")
                if now-last_status>=1:
                    status=client.request("/status");last_status=now
                    print(f"R{round_id:02d} {mode:6s} {now-started:5.1f}s {obs.phase:14s} {decision.reason}",flush=True)
                    if seen_playing and status.get("screen","").startswith("result:"):
                        result=status["screen"];break
                    if not status.get("control",{}).get("paused"):raise RuntimeError("输入所有权已改变，停止采样")
    finally:
        lease.close()
        evidence.save(output)
    summary={"round_id":round_id,"mode":mode,"source_session":initial["output"],
        "actual_commands":str(Path(initial["output"])/"commands.jsonl"),
        "seconds":time.perf_counter()-started,"frames":frames,"playing_frames":playing,
        "result":result,"events":events,"config":config,"motion_calibrated":False,
        "frame_source":"live_capture_960wide","recording":"no continuous video; bounded candidate-contact PNG windows"}
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf8")
    if result is None:raise RuntimeError("未确认结算，停止批次，不盲按Retry")
    return summary


def main():
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf8")
    parser=argparse.ArgumentParser(description="连续多局MINING分段脉冲采样；批量完成后统一拟合")
    parser.add_argument("--endpoint",type=Path,default=ROOT/"artifacts/control_endpoint.json")
    parser.add_argument("--profile",type=Path,default=ROOT/"artifacts/mining-supervision/profile.json")
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--rounds",type=int,default=6)
    parser.add_argument("--seconds-per-round",type=float,default=90)
    args=parser.parse_args()
    if args.rounds<1:parser.error("rounds需至少1")
    if args.output.exists():parser.error("输出目录必须不存在")
    config=json.loads(args.config.read_text(encoding="utf8"))
    if config.get("enabled") is not True:parser.error("探针需明确enabled=true")
    profile=json.loads(args.profile.read_text(encoding="utf8"))
    args.output.mkdir(parents=True)
    client=LiveClient(args.endpoint,frame_width=960);summaries=[]
    try:
        for i in range(args.rounds):
            mode="normal" if i%2==0 else "zr"
            summary=collect_round(client,profile,config,args.output/f"round-{i+1:02d}",i+1,mode,args.seconds_per_round)
            summaries.append(summary)
            (args.output/"summary.json").write_text(json.dumps({"rounds":summaries,"status":"collecting" if i+1<args.rounds else "collected_not_fitted","calibrated":False},ensure_ascii=False,indent=2),encoding="utf8")
            time.sleep(.35)
    finally:
        try:client.command({"command":"release"},wait=True)
        except Exception:pass


if __name__=="__main__":main()
