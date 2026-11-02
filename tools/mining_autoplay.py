"""Run a frozen MINING controller for a review batch; no parameter learning."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from mining_supervisor import LiveClient,run_round
from raiders_helper.mining_vision import MiningLiveRecognizer
from raiders_helper.mining_reactive import decide_mining_reactive


def main():
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint",type=Path,default=ROOT/"artifacts/control_endpoint.json")
    parser.add_argument("--profile",type=Path,required=True)
    parser.add_argument("--parameters",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--rounds",type=int,default=6)
    parser.add_argument("--seconds-per-round",type=float,default=600)
    args=parser.parse_args()
    if args.rounds<1 or args.output.exists():parser.error("rounds需为正，输出目录必须不存在")
    args.output.mkdir(parents=True)
    frozen={};raw_inputs={}
    for name,path in [("profile",args.profile),("parameters",args.parameters)]:
        raw=path.read_bytes();raw_inputs[name]=json.loads(raw.decode("utf8"))
        (args.output/f"{name}.json").write_bytes(raw)
        frozen[name]={"source":str(path.resolve()),"sha256":hashlib.sha256(raw).hexdigest()}
    frozen["source_hashes"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/"src/raiders_helper/mining_vision.py",ROOT/"src/raiders_helper/mining_reactive.py",ROOT/"tools/mining_supervisor.py"]}
    (args.output/"frozen-inputs.json").write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding="utf8")
    client=LiveClient(args.endpoint,frame_width=960);reports=[]
    try:
        for i in range(args.rounds):
            if (args.output/"stop.request").exists():break
            print(f"自主避障 {i+1}/{args.rounds}（本批模型与策略冻结）",flush=True)
            report=run_round(client,MiningLiveRecognizer("mining",raw_inputs["profile"]),
                decide_mining_reactive,raw_inputs["parameters"],args.output/f"round-{i+1:02d}",args.seconds_per_round)
            reports.append(report)
            (args.output/"summary.json").write_text(json.dumps({"requested_rounds":args.rounds,
                "completed_rounds":len(reports),"reports":reports,"frozen":True,
                "scope":"live gameplay review; not certified collision-free"},ensure_ascii=False,indent=2),encoding="utf8")
            if not report.get("result","").startswith("result:"):raise RuntimeError("未确认结果，停止且不盲按Retry")
            time.sleep(.35)
    finally:
        try:client.command({"command":"release"},wait=True)
        except Exception:pass


if __name__=="__main__":main()
