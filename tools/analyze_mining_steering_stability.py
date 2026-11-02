"""Read-only steering switch diagnostics; a reversal is not proof of jitter.

No import of controllers, strategy code, GUI, or live interfaces. Actual input
uses successful commands.after. Advice attribution is explicitly approximate.
"""
from bisect import bisect_right
from collections import Counter
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def analyse(directory,output):
    directory,output=Path(directory),Path(output);output.mkdir(parents=True,exist_ok=True)
    trace=[json.loads(line) for line in (directory/"trace.jsonl").read_text(encoding="utf-8").splitlines()]
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    commands=[json.loads(line) for line in (Path(summary["session"])/"commands.jsonl").read_text(encoding="utf-8").splitlines()]
    commands=sorted((c for c in commands if c.get("success")),key=lambda c:c["after"])
    after_countdown=max((i for i,r in enumerate(trace) if r["observation"]["phase"]=="countdown"),default=-1)
    playing=[r for r in trace[after_countdown+1:] if r["observation"]["phase"]=="playing"]
    start,end=playing[0]["observation"]["timestamp"],playing[-1]["observation"]["timestamp"]
    times=[r["client_timestamp"] for r in trace]
    records=[];last_axis=None;last_sign=None;direction_start=None;neutral_start=None;reversals=[]
    total_variation=0.;counts=Counter()
    for command in commands:
        t=command["after"]
        if not start<=t<=end:continue
        axis=command["state"]["left_stick"]["x"]/32767
        if axis==last_axis:continue
        if last_axis is not None:total_variation+=abs(axis-last_axis)
        last_axis=axis
        k=bisect_right(times,t)-1;row=trace[k] if k>=0 else None
        d=row["decision"].get("diagnostics",{}) if row else {};obs=row["observation"] if row else {}
        player=next((p for p in obs.get("entities",[]) if p["kind"]=="player"),{})
        star=d.get("star_target") or {}
        item={"seconds":t-start,"sequence":obs.get("sequence"),"actual_axis":axis,"zr":"ZR" in command["state"]["buttons"],
            "latest_advice_age_seconds":t-times[k] if k>=0 else None,
            "latest_advice_state_matches":bool(row and row["decision"]["state"]==command["state"]),
            "mode":d.get("mode"),"phase":obs.get("phase"),"player_x":player.get("x"),"player_vx":player.get("vx"),
            "star_target_key":star.get("key"),"star_target_x":star.get("x"),"nearest_hazard_depth":d.get("nearest_hazard_depth"),
            "wall_distance":d.get("wall_distance")}
        records.append(item)
        sign=1 if axis>=.25 else -1 if axis<=-.25 else 0
        if not sign:
            if neutral_start is None:neutral_start=t
            continue
        if neutral_start is not None and t-neutral_start>.15:
            last_sign=None;direction_start=None
        neutral_start=None
        if last_sign is not None and sign!=last_sign:
            duration=t-direction_start
            event={**item,"previous_direction_seconds":duration,"rapid_under_025s":duration<=.25,
                "velocity_braking_candidate":bool(player and axis*player.get("vx",0)<0 and abs(player.get("vx",0))>.02),
                "near_geometry_risk":bool((d.get("nearest_hazard_depth") is not None and d["nearest_hazard_depth"]<.18)
                                          or (d.get("wall_distance") is not None and d["wall_distance"]<.05)),
                "scope":"observed_sign_reversal_not_proven_unnecessary_jitter"}
            reversals.append(event);counts[str(d.get("mode"))]+=1
            direction_start=t
        if last_sign is None:direction_start=t
        last_sign=sign
    parameters=summary["parameters"];axis_cap=parameters.get("maximum_axis",1.);fast=parameters.get("motion_models",{}).get("zr",{})
    speed=fast.get("forward_speed",.5);lateral_cap=fast.get("maximum_lateral_speed",.85)
    accelerated=[r for r in trace if r["decision"].get("diagnostics",{}).get("mode")=="accelerated"]
    saturated=sum(abs(r["decision"]["state"]["left_stick"]["x"]/32767)>=.90*axis_cap for r in accelerated)
    report={"scope":"read_only_steering_switch_diagnostics_not_collision_causality", "calibrated":False,
        "directory":str(directory.resolve()),"playing_time_window_seconds":end-start,"actual_axis_transitions":len(records),
        "actual_axis_total_variation":total_variation,"actual_direction_reversals":len(reversals),
        "rapid_direction_reversals_previous_direction_le025s":sum(e["rapid_under_025s"] for e in reversals),
        "reversal_latest_advice_modes":dict(counts),"reversals":reversals,
        "sharp_gate_audit":{"maximum_axis":axis_cap,"absolute_gate_threshold":.75,
            "absolute_gate_can_trigger":axis_cap>.75,"configured_fast_speed":speed,
            "hard_lateral_cap_over_forward_speed":lateral_cap/speed,
            "curve_gate_threshold":parameters.get("sharp_turn_slope",1.05),
            "curve_gate_can_trigger_under_declared_speed_cap":lateral_cap/speed>parameters.get("sharp_turn_slope",1.05),
            "accelerated_advice_frames":len(accelerated),"accelerated_frames_near_axis_cap":saturated},
        "limitations":["Rapid reversals can be necessary velocity braking or obstacle avoidance; they are not automatically unnecessary jitter.",
            "Latest advice attribution can differ from the report applied by the asynchronous input pump; per-event state match is exposed.",
            "The sign-duration statistic measures the previous direction run, not repeated 30Hz report packet spacing. Neutral gaps >.15s reset runs.",
            "This tool does not replay alternate strategies, change the six-round frozen policy, or infer collision causality from decisions."]}
    (output/"steering_stability_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    with (output/"actual_axis_changes.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]) if records else ["seconds"]);writer.writeheader();writer.writerows(records)
    plot(records,output)
    return report


def plot(records,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname="C:/Windows/Fonts/msyh.ttc");fig,axes=plt.subplots(3,1,figsize=(13,8),sharex=True)
    t=[r["seconds"] for r in records]
    axes[0].step(t,[r["actual_axis"] for r in records],where="post",lw=.8)
    axes[0].set_ylabel("实际摇杆横轴",fontproperties=font)
    axes[1].plot(t,[r["player_x"] if r["player_x"] is not None else np.nan for r in records],".-",ms=2,lw=.5)
    axes[1].plot(t,[r["star_target_x"] if r["star_target_x"] is not None else np.nan for r in records],"x",ms=3)
    axes[1].set_ylabel("角色与预测星目标横坐标",fontproperties=font)
    axes[2].plot(t,[r["nearest_hazard_depth"] if r["nearest_hazard_depth"] is not None else np.nan for r in records],".",ms=3)
    axes[2].set_ylabel("已识别障碍最近纵向距离",fontproperties=font);axes[2].set_xlabel("本轮运行相对时间/秒",fontproperties=font)
    fig.suptitle("冻结策略的实际转向切换：反转不等于无必要抖动",fontproperties=font)
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(output/"steering_switches.png",dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();r=analyse(a.directory,a.output)
    print(json.dumps({k:r[k] for k in ("actual_axis_transitions","actual_direction_reversals","rapid_direction_reversals_previous_direction_le025s","sharp_gate_audit")}))


if __name__=="__main__":main()
