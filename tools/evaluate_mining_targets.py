"""Score frozen-model target trials; never fit parameters or open hardware."""
from collections import defaultdict
import argparse
import json
from pathlib import Path

from analyze_mining_pulses import position_predict


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s.strip()]


def evaluate(batch,models):
    result=[];rejected=[];command_cache={}
    for folder in sorted(batch.glob("round-*")):
        if not (folder/"summary.json").exists():continue
        summary=json.loads((folder/"summary.json").read_text(encoding="utf8"))
        path=summary["actual_commands"]
        if path not in command_cache:command_cache[path]=sorted((r for r in read_rows(Path(path)) if r.get("success")),key=lambda r:r["after"])
        commands=command_cache[path];groups=defaultdict(list)
        for row in read_rows(folder/"trace.jsonl"):
            ident=row.get("identification",{})
            if ident.get("segment_id"):groups[ident["segment_id"]].append(row)
        for sid,rows in groups.items():
            rows=[r for r in rows if r["identification"].get("requested_delta_x") is not None]
            if not rows:continue
            key=folder.name+":"+sid
            completed=[r for r in rows if r["identification"].get("segment_validity")=="completed"]
            if not completed or any(r["identification"].get("segment_aborted") for r in rows):
                rejected.append({"id":key,"reason":"not_completed_or_aborted"});continue
            first=next((r for r in rows if r["identification"].get("stage")=="pulse"),None)
            last=completed[-1]
            if first is None:continue
            d=first["identification"];mode=d["mode"];model=models[mode]["selected_by_training_only"]
            player=next((p for p in last["observation"]["entities"] if p["kind"]=="player"),None)
            if player is None:rejected.append({"id":key,"reason":"no_observed_endpoint"});continue
            reference_time=d["target_reference_timestamp"];reference_x=d["target_reference_x"]
            endpoint_time=last["observation"]["timestamp"];issue=d["stage_started_at"]
            first_on=next((c for c in commands if c["after"]>=issue-.002 and abs(c["state"]["left_stick"]["x"])>1638),None)
            first_off=next((c for c in commands if first_on and c["after"]>first_on["after"] and abs(c["state"]["left_stick"]["x"])<=1638),None)
            prior=[c for c in commands if c["after"]<=reference_time-.35]
            if not prior:rejected.append({"id":key,"reason":"missing_input_history"});continue
            history=[prior[-1]]+[c for c in commands if reference_time-.35<c["after"]<=endpoint_time]
            changes=[];previous=None
            for c in history:
                axis=c["state"]["left_stick"]["x"]/32767
                if axis!=previous:changes.append({"time":c["after"]-reference_time,"axis":axis});previous=axis
            state={"x":reference_x,"v":d["target_reference_velocity"],
                   "previous_axis":history[0]["state"]["left_stick"]["x"]/32767}
            prediction=position_predict(endpoint_time-reference_time,state,changes,model)
            result.append({"id":key,"mode":mode,"requested_delta_x":d["requested_delta_x"],
                "planned_hold_seconds":d["pulse_seconds"],
                "actual_hold_seconds":first_off["after"]-first_on["after"] if first_on and first_off else None,
                "observed_delta_x":player["x"]-reference_x,
                "target_error_widths":player["x"]-d["target_x"],
                "target_error_canonical_pixels":(player["x"]-d["target_x"])*640,
                "actual_input_prediction_error_widths":player["x"]-prediction["x"],
                "initial_velocity_from_tracker":state["v"],"reference_frame_to_issue_seconds":issue-reference_time,
                "endpoint_seconds_after_planned_pulse_start":endpoint_time-issue,
                "scope":"completed_probe_gates_not_independent_collision_ground_truth"})
    return {"calibrated":False,"fitted_on_targets":False,"trials":result,"rejected":rejected,
            "limitation":"tracker initial velocity and visual probe gates are imperfect; targets do not validate collision physics or forward acceleration"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch",type=Path,required=True);p.add_argument("--models",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();models=json.loads(a.models.read_text(encoding="utf8"))["models"]
    report=evaluate(a.batch,models);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({"completed_target_trials":len(report["trials"]),"rejected":len(report["rejected"]),"output":str(a.output.resolve())}))


if __name__=="__main__":main()
