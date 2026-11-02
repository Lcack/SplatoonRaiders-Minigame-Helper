"""Read-only, source-indexed contact appearance and visible-clearance study.

Never interprets sprite boxes, filled masks, ridge colour or a Hough circle as
a game's physical hitbox. Numerical brackets are conditional image geometry,
not calibrated collision radii. The user-supplied cases label raw PNG evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from raiders_helper.capture.models import FramePacket
from raiders_helper.mining_vision import MiningLiveRecognizer


def image(path):
    result=cv2.imdecode(np.frombuffer(Path(path).read_bytes(),np.uint8),1)
    if result is None:raise ValueError(str(path))
    return result


def visible_circles(frame,body):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    smooth=cv2.GaussianBlur(gray,(0,0),1.5)
    raw=cv2.HoughCircles(smooth,cv2.HOUGH_GRADIENT,dp=1.1,minDist=30,param1=65,param2=26,minRadius=17,maxRadius=36)
    if raw is None:return []
    value=cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,2],(0,0),1.2)
    result=[];angles=np.arange(72)*2*np.pi/72
    for x,y,r in raw[0]:
        x,y,r=map(float,(x,y,r))
        if not 45<y<450:continue
        if body and abs(x-body["x"])<body["w"]/2+3 and abs(y-body["y"])<body["h"]/2+3:continue
        radii=np.arange(max(8,r-9),r+8,.25);means=[];fractions=[]
        for radius in radii:
            xx=x+radius*np.cos(angles);yy=y+radius*np.sin(angles)
            keep=np.ones(len(xx),bool)
            if body:keep&=~((abs(xx-body["x"])<body["w"]/2+3)&(abs(yy-body["y"])<body["h"]/2+3))
            samples=cv2.remap(value,xx.astype(np.float32)[None,:],yy.astype(np.float32)[None,:],cv2.INTER_LINEAR).ravel()
            means.append(float(np.mean(samples[keep])) if np.any(keep) else 0.)
            fractions.append(float(np.mean(keep)))
        peak=int(np.argmax(means));floor=float(np.median(means[-8:]));contrast=means[peak]-floor
        if contrast<12 or fractions[peak]<.5:continue
        threshold=(means[peak]+floor)/2;outer=None
        for k in range(peak+1,len(means)):
            if means[k]<=threshold:
                alpha=(means[k-1]-threshold)/max(1e-8,means[k-1]-means[k])
                outer=float(radii[k-1]+alpha*(radii[k]-radii[k-1]));break
        if outer is None:continue
        result.append({"x":x,"y":y,"hough_radius":r,"visible_outer_half_intensity_radius":outer,
            "rim_brightness_contrast":contrast,"rim_visible_fraction":fractions[peak],
            "scope":"visible_radial_brightness_contour_not_collision_boundary"})
    return result


def wall_estimates(obs,body):
    """Fit only ahead of the visible body; never use its occluded wall row."""
    rows=np.asarray(obs.metrics.get("road_rows",[]),float)
    if not len(rows):return {}
    y=body["y"];forward=rows[(rows[:,0]*480>=y+body["h"]/2+5)&(rows[:,0]*480<=y+body["h"]/2+55)]
    if len(forward)<5:return {}
    result={}
    for side,column in (("left",1),("right",2)):
        yy=forward[:,0]*480;xx=forward[:,column]*640
        slope,intercept=np.polyfit(yy-y,xx,1);rms=float(np.sqrt(np.mean((xx-slope*(yy-y)-intercept)**2)))
        extrapolation=float(yy.min()-y);span=max(1.,float(np.ptp(yy)))
        distance=(body["x"]-intercept if side=="left" else intercept-body["x"])/math.sqrt(1+slope*slope)
        # Declared image-fit sensitivity envelope, NOT a confidence interval.
        uncertainty=1.5+2*rms*(1+extrapolation/span)
        result[side]={"signed_center_to_extrapolated_visible_inner_edge_px":float(distance),
            "image_fit_sensitivity_interval_px":[float(distance-uncertainty),float(distance+uncertainty)],
            "line_x_at_body_y":float(intercept),"dx_dy":float(slope),"fit_rms_px":rms,
            "extrapolation_distance_px":extrapolation,"fitted_forward_y_range_px":[float(yy.min()),float(yy.max())],
            "body_occlusion_extrapolated":True,"physical_wall_collision_boundary_known":False}
    return result


def register_far_landmark(before,after,circle):
    x,y=circle["x"],circle["y"];radius=circle["visible_outer_half_intensity_radius"]
    size=int(math.ceil(radius+5));cx,cy=round(x),round(y)
    if cx-size<0 or cx+size>=640 or cy-size<0 or cy+size>=480:return None
    patch=before[cy-size:cy+size+1,cx-size:cx+size+1]
    x0,x1=max(0,cx-size-8),min(640,cx+size+9)
    y0,y1=max(35,cy-size-30),min(480,cy+size+31)
    search=after[y0:y1,x0:x1]
    if search.shape[0]<patch.shape[0] or search.shape[1]<patch.shape[1]:return None
    scores=cv2.matchTemplate(search,patch,cv2.TM_CCOEFF_NORMED)
    _,score,_,loc=cv2.minMaxLoc(scores)
    return {"dx_px":float(x0+loc[0]-(cx-size)),"dy_px":float(y0+loc[1]-(cy-size)),
            "match_score_not_accuracy":float(score),"assumed_registration_uncertainty_px":1.5}


def study_case(case,output,index):
    window=Path(case["window"]);manifest=json.loads((window/"manifest.json").read_text(encoding="utf-8"))
    round_dir=window.parents[1]
    original={r["observation"]["sequence"]:r["observation"] for r in map(json.loads,(round_dir/"trace.jsonl").read_text(encoding="utf-8").splitlines())}
    frames=manifest["frames"]
    reds=[f for f in frames if original.get(f["sequence"],{}).get("metrics",{}).get("hit_body_detected")]
    if not reds:return {"window":str(window.resolve()),"status":"no_observed_hit_body_in_saved_window"},[]
    first_red=reds[0];red_time=first_red["timestamp"]
    recognizer=MiningLiveRecognizer(profile={"floor_backend":"adaptive_inner","phase_anchors":False,"stable_frames":1,
        "scan_y":[v/480 for v in range(40,201,2)]+[.5,.8,.95]})
    records=[];pictures={}
    for meta in frames:
        if meta["timestamp"]>red_time+.12:break
        source=window/meta["file"];raw=image(source)
        obs=recognizer.observe(FramePacket(raw,meta["timestamp"],meta["sequence"],60))
        actor=next((e for e in obs.entities if e.kind in ("player","player_hit")),None)
        if actor is None:continue
        body={"x":actor.x*640,"y":actor.y*480,"w":actor.w*640,"h":actor.h*480,"appearance":actor.kind}
        canonical=recognizer.last_canonical.copy();circles=visible_circles(canonical,body)
        record={"sequence":meta["sequence"],"timestamp":meta["timestamp"],"seconds_to_first_red_appearance":red_time-meta["timestamp"],
            "source_png":str(source.resolve()),"source_size":list(raw.shape[1::-1]),"body_visual_anchor":body,
            "experimental_inner_wall":wall_estimates(obs,body),"visible_circle_candidates":circles,
            "floor_palette":obs.metrics.get("floor_palette"),"hit_body_observed_in_original_trace":original.get(meta["sequence"],{}).get("metrics",{}).get("hit_body_detected",False)}
        records.append(record);pictures[meta["sequence"]]=canonical
    blue=[r for r in records if r["timestamp"]<red_time and r["body_visual_anchor"]["appearance"]=="player"]
    red=next((r for r in records if r["sequence"]==first_red["sequence"]),None)
    if not blue or red is None:return {"window":str(window.resolve()),"status":"body_anchor_not_reacquired_for_pair"},records
    last_blue=blue[-1];safe=[r for r in blue if .15<=r["seconds_to_first_red_appearance"]<=.65]
    summary={"window":str(window.resolve()),"manual_event_kind":case["event_kind"],"manual_label_scope":case.get("label_scope","raw appearance candidate"),
        "status":"appearance_pair_measured_not_physical_hitbox","last_blue_sequence":last_blue["sequence"],"first_red_sequence":red["sequence"],
        "blue_to_red_appearance_gap_seconds":red_time-last_blue["timestamp"],"physical_collision_time_proven":False,
        "physical_robot_hitbox":None,"physical_rock_hitbox":None,"physical_wall_hitbox":None}
    selected=[last_blue,red]
    if case["event_kind"]=="rock":
        body=last_blue["body_visual_anchor"]
        near=[c for c in last_blue["visible_circle_candidates"] if math.hypot(c["x"]-body["x"],c["y"]-body["y"])<95]
        if near:
            target=min(near,key=lambda c:math.hypot(c["x"]-body["x"],c["y"]-body["y"])-c["visible_outer_half_intensity_radius"])
            estimates=[]
            for record in safe+[last_blue]:
                circles=[c for c in record["visible_circle_candidates"] if abs(c["x"]-target["x"])<10 and abs(c["visible_outer_half_intensity_radius"]-target["visible_outer_half_intensity_radius"])<5]
                if not circles:continue
                c=min(circles,key=lambda a:abs(a["x"]-target["x"]))
                b=record["body_visual_anchor"];separation=math.hypot(c["x"]-b["x"],c["y"]-b["y"])
                estimates.append({"sequence":record["sequence"],"seconds_before_red":record["seconds_to_first_red_appearance"],
                    "center_separation_px":separation,"visible_rim_radius_px":c["visible_outer_half_intensity_radius"],
                    "center_to_visible_rim_clearance_px":separation-c["visible_outer_half_intensity_radius"],
                    "label":"pre_red_blue_clear_candidate_not_proven_noncontact" if record is not last_blue else "last_blue_may_already_be_contacted"})
            summary["rock_geometry_before_appearance"]=estimates;summary["last_intact_visible_rim"]=target
            helpers=[c for c in last_blue["visible_circle_candidates"] if math.hypot(c["x"]-body["x"],c["y"]-body["y"])>120]
            if helpers:
                helper=max(helpers,key=lambda c:c["rim_brightness_contrast"])
                reg=register_far_landmark(pictures[last_blue["sequence"]],pictures[red["sequence"]],helper)
                summary["independent_far_rock_registration"]=reg
                if reg and reg["match_score_not_accuracy"]>.85:
                    red_body=red["body_visual_anchor"]
                    separation=math.hypot(target["x"]+reg["dx_px"]-red_body["x"],target["y"]+reg["dy_px"]-red_body["y"])
                    safe_est=[e for e in estimates if e["seconds_before_red"]>=.15]
                    summary["conditional_circular_combined_radius_image_bracket_px"]=[separation-6.,min(e["center_separation_px"]+4.5 for e in safe_est)] if safe_est else None
                    summary["registered_red_center_separation_px"]=separation
                    summary["conditional_bracket_assumptions"]="static circular rock; shared map translation; visual anchor is collision centre; contact exactly at red; pre-red guards truly clear. These assumptions are NOT established."
            if estimates:
                near_safe=min((e for e in estimates if e["seconds_before_red"]>=.15),key=lambda e:e["center_separation_px"],default=None)
                if near_safe:selected.insert(0,next(r for r in records if r["sequence"]==near_safe["sequence"]))
    else:
        side=case.get("wall_side","left")
        summary["wall_side"]=side
        summary["blue_extrapolated_visible_wall"]=last_blue["experimental_inner_wall"].get(side)
        summary["red_extrapolated_visible_wall"]=red["experimental_inner_wall"].get(side)
        distances=[(r,r["experimental_inner_wall"].get(side)) for r in safe]
        distances=[(r,d) for r,d in distances if d]
        if distances:
            near_safe,measurement=min(distances,key=lambda a:a[1]["signed_center_to_extrapolated_visible_inner_edge_px"])
            summary["minimum_pre_red_blue_apparent_clearance"]=measurement
            summary["minimum_pre_red_blue_sequence"]=near_safe["sequence"];selected.insert(0,near_safe)
        summary["wall_boundary_limitation"]="adaptive_inner floor/ridge boundary is experimental and extrapolated across body occlusion; it is not calibrated to the game's collision wall. Signed negative distances cannot be interpreted as negative physical robot radius."
    plot_case(selected,pictures,summary,output/f"case-{index:02d}.png")
    return summary,records


def plot_case(records,pictures,summary,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    fig,axes=plt.subplots(1,len(records),figsize=(5*len(records),4),squeeze=False)
    for ax,record in zip(axes.flat,records):
        ax.imshow(cv2.cvtColor(pictures[record["sequence"]],cv2.COLOR_BGR2RGB));b=record["body_visual_anchor"]
        ax.plot(b["x"],b["y"],"rx",ms=9)
        for circle in record["visible_circle_candidates"]:
            ax.add_patch(plt.Circle((circle["x"],circle["y"]),circle["visible_outer_half_intensity_radius"],fill=False,color="cyan",lw=1))
        for side,data in record["experimental_inner_wall"].items():
            yy=np.linspace(b["y"]-10,b["y"]+70,80);xx=data["line_x_at_body_y"]+data["dx_dy"]*(yy-b["y"])
            ax.plot(xx,yy,"g--",lw=1)
        ax.set_xlim(max(0,b["x"]-130),min(640,b["x"]+150));ax.set_ylim(min(480,b["y"]+150),max(35,b["y"]-45))
        label="蓝体候选" if b["appearance"]=="player" else "紫红外观"
        ax.set_title(f"{record['sequence']} {label}；距首红 {record['seconds_to_first_red_appearance']:.3f}s",fontproperties=font)
    fig.suptitle("图像间距研究：红叉为视觉中心，青线为可见rim，绿虚线为外推内沿；均不是物理碰撞箱",fontproperties=font)
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description="只读分析已人工分类的接触原帧；拒绝把sprite/rim当物理碰撞箱")
    parser.add_argument("--cases",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2)
    cases=json.loads(args.cases.read_text(encoding="utf-8"));summaries=[];records=[]
    for index,case in enumerate(cases,1):
        summary,rows=study_case(case,args.output,index);summaries.append(summary);records.extend(rows)
    report={"scope":"source_indexed_visual_contact_envelope_not_physical_hitbox_calibration","calibrated":False,
        "vision_source_sha256":hashlib.sha256((ROOT/"src/raiders_helper/mining_vision.py").read_bytes()).hexdigest(),
        "cases":summaries,"bound_direction":"For a genuine circular collider with a known boundary: contact d_c implies r>=d_c, noncontact d_s implies r<d_s. Neither colour boundary nor blue appearance alone establishes those premises.",
        "limitations":["All measurements use canonical 640x480 playfield pixels; 960-wide HTTP images already have about one source pixel per canonical horizontal pixel.",
            "Body dimensions are only occlusion masks, never collider radii. Circle radius is a visible half-brightness contour, never the rock collider.",
            "Wall line sensitivity is a declared image-fit envelope, not a statistical confidence interval. Curved/scalloped walls and extrapolation can dominate pixel noise.",
            "Same-frame geometry does not add input/capture latency; blue-to-red appearance timing can lag contact, and its latency is unmeasured.",
            "Physical robot/rock/wall shapes and extreme safe passing distance remain unidentifiable without independent contact labels and additional approach angles."]}
    (args.output/"contact_envelope_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (args.output/"frame_measurements.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),encoding="utf-8")
    print(json.dumps({"cases":len(summaries),"frames":len(records),"output":str(args.output.resolve()),"calibrated":False}))


if __name__=="__main__":main()
