"""Read-only snapshot replay, not a closed-loop score or safety validation."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from raiders_helper.mining_reactive import decide_mining_reactive
from raiders_helper.models import Entity,Game,Observation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stride",type=int,default=7)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for directory in sorted(args.batch.glob("round-*")):
        summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
        parameters={**summary["parameters"],"steering_change_weight":.003,"mixed_zr_prefix_seconds":[.1,.2,.3]}
        trace=[json.loads(line) for line in (directory/"trace.jsonl").read_text(encoding="utf-8").splitlines()]
        rows=[row for row in trace if row["observation"]["phase"]=="playing"]
        for row in rows[::args.stride]:
            raw=row["observation"]
            obs=Observation(Game(raw["game"]),raw["timestamp"],raw["sequence"],raw["phase"],raw["confidence"],
                tuple(Entity(**entity) for entity in raw["entities"]),tuple(raw.get("notes",())),raw.get("metrics",{}))
            advice=decide_mining_reactive(obs,parameters)
            old=row["decision"]["diagnostics"];new=advice.diagnostics
            records.append({"round":directory.name,"sequence":obs.sequence,"old_mode":old.get("mode"),
                "new_mode":new.get("mode"),"old_axis":row["decision"]["state"]["left_stick"]["x"]/32767,
                "new_axis":advice.state.left_stick.x/32767,"old_route_status":old.get("route_status"),
                "new_route_status":new.get("route_status"),"fast_route_status":new.get("fast_route_status"),
                "mixed_zr_prefix_seconds":new.get("mixed_zr_prefix_seconds"),
                "new_planning_ms":new.get("planning_ms"),"expanded_nodes":new.get("expanded_nodes"),
                "initial_axis_fraction":(new.get("turn_authority") or {}).get("early_axis_fraction")})
    changes=Counter(f"{r['old_mode']} -> {r['new_mode']}" for r in records)
    milliseconds=sorted(r["new_planning_ms"] for r in records if r["new_planning_ms"] is not None)
    report={"scope":"counterfactual_snapshots_on_frozen_observations_not_closed_loop_or_score_validation",
        "samples":len(records),"stride":args.stride,"mode_changes":dict(changes),
        "planning_ms_quantiles":{str(q):milliseconds[round(q*(len(milliseconds)-1))] for q in (.5,.9,1)} if milliseconds else {},
        "source_sha256":hashlib.sha256((ROOT/"src/raiders_helper/mining_reactive.py").read_bytes()).hexdigest(),
        "limitations":["Geometry is the recorded frozen vision output, not reprocessed new occlusion geometry.",
            "Future states and actual histories are from the original policy; this replay cannot predict new score.",
            "Rows are a fixed-stride sample, not actual controller event counts or durations."]}
    (args.output/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    with (args.output/"snapshots.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
