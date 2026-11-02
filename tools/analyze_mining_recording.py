"""Read-only MINING trajectory/background-motion diagnostics for live recordings.

The video container clock is NOT used as the experimental timebase: every
decoded index is joined to frames.jsonl, then to actual commands.jsonl sends.
Background optical flow is reported as a visual proxy, not validated forward
speed. This tool never opens a controller, never labels collisions from a
missing sprite, and never promotes the output to a calibrated motion model.
"""
from bisect import bisect_right
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def canonical(frame):
    h, w = frame.shape[:2]
    return cv2.resize(frame[round(h*55/1080):round(h*1015/1080),
                            round(w/6):round(w*5/6)], (640, 480))


def player_candidate(frame, previous=None):
    """Distinctive blue body seed, optionally expanded to the green outline."""
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    blue=cv2.inRange(hsv,(95,95,60),(138,255,255))
    blue[:42]=0;blue[175:]=0;blue[:,:12]=0;blue[:,628:]=0
    n,_,stats,centers=cv2.connectedComponentsWithStats(blue)
    candidates=[i for i in range(1,n) if stats[i,4]>=25 and 8<=stats[i,2]<=65 and 7<=stats[i,3]<=65]
    if not candidates:
        return None
    def cost(i):
        x,y=centers[i]
        distance=np.linalg.norm(centers[i]-previous) if previous is not None else abs(y-92)
        return stats[i,4]/(1+distance*.05)
    i=max(candidates,key=cost)
    if previous is not None and np.linalg.norm(centers[i]-previous)>95:
        return None
    seed=centers[i]
    body=cv2.inRange(hsv,(40,70,60),(140,255,255))
    body=cv2.morphologyEx(body,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,_,boxes,_=cv2.connectedComponentsWithStats(body)
    connected=[box for box in boxes[1:] if box[0]<=seed[0]<=box[0]+box[2]
               and box[1]<=seed[1]<=box[1]+box[3] and 15<=box[2]<=95 and 15<=box[3]<=85]
    box=max(connected,key=lambda b:b[4])[:4].tolist() if connected else stats[i,:4].tolist()
    return {"x_px":float(seed[0]),"y_px":float(seed[1]),"box":box,
            "blue_pixels":int(stats[i,4]),"scope":"colour_shape_candidate_not_ground_truth"}


def brown_mask(frame):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,(5,65,25),(45,255,255))
    mask[:155]=0;mask[440:]=0;mask[:,:20]=0;mask[:,620:]=0
    return mask


def flow_proxy(previous,current,dt):
    if previous is None or not 0<dt<=.12:
        return {}
    old=cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY)
    gray=cv2.cvtColor(current,cv2.COLOR_BGR2GRAY)
    points=cv2.goodFeaturesToTrack(old,maxCorners=160,qualityLevel=.03,minDistance=9,mask=brown_mask(previous))
    if points is None or len(points)<8:
        return {"tracks":0,"reliable_forward_speed":False}
    moved,ok,error=cv2.calcOpticalFlowPyrLK(old,gray,points,None,winSize=(21,21),maxLevel=3)
    if moved is None or ok is None:
        return {"tracks":0,"reliable_forward_speed":False}
    back,back_ok,_=cv2.calcOpticalFlowPyrLK(gray,old,moved,None,winSize=(21,21),maxLevel=3)
    if back is None or back_ok is None:
        return {"tracks":0,"reliable_forward_speed":False}
    good=(ok.ravel()>0)&(back_ok.ravel()>0)&(np.linalg.norm(back-points,axis=2).ravel()<1.5)
    a=points.reshape(-1,2)[good];b=moved.reshape(-1,2)[good]
    if len(a)<8:
        return {"tracks":len(a),"reliable_forward_speed":False}
    delta=b-a
    median=np.median(delta,axis=0)
    residual=np.linalg.norm(delta-median,axis=1)
    inliers=residual<=max(.7,float(np.median(residual))*2.5)
    if np.count_nonzero(inliers)>=6:
        median=np.median(delta[inliers],axis=0)
    return {"tracks":len(a),"inliers":int(np.count_nonzero(inliers)),
            "median_dx_px":float(median[0]),"median_dy_px":float(median[1]),
            "median_residual_px":float(np.median(residual)),
            "screen_flow_y_heights_per_second":float(median[1]/(480*dt)),
            "forward_proxy_heights_per_second":float(-median[1]/(480*dt)),
            "reliable_forward_speed":False,
            "limitation":"texture/camera displacement proxy; no independent landmark validation"}


def load_sidecars(session):
    frames=[]
    with (session/"frames.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row=json.loads(line)
            if row.get("status")=="written" and row.get("video_frame_index") is not None:
                frames.append(row)
    commands=[]
    with (session/"commands.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row=json.loads(line)
            if row.get("success"):
                commands.append(row)
    return frames,commands


def analyse(session,output,start=126,end=171,stride=2):
    session,output=Path(session),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(2)
    frames,commands=load_sidecars(session)
    if not frames:
        raise ValueError("没有已写入录像的帧侧车")
    origin=frames[0]["capture_timestamp"]
    selected=[r for r in frames if start<=r["capture_timestamp"]-origin<end][::stride]
    if not selected:
        raise ValueError("所选时间段没有可解码的帧")
    command_times=[r["after"] for r in commands]
    transitions=[];previous_state=None
    for command in commands:
        state=command["state"]
        if state!=previous_state:
            transitions.append({"seconds":command["after"]-origin,"state":state})
            previous_state=state
    cap=cv2.VideoCapture(str(session/"capture.mp4"))
    rows=[];last_image=None;last_time=None;last_player=None;missing=0
    first_index=selected[0]["video_frame_index"]
    cap.set(cv2.CAP_PROP_POS_FRAMES,first_index)
    next_index=first_index
    try:
        for meta in selected:
            index=meta["video_frame_index"]
            while next_index<index:
                if not cap.grab():
                    raise RuntimeError("录像跳帧解码失败")
                next_index+=1
            ok,frame=cap.read();next_index+=1
            if not ok:
                raise RuntimeError(f"录像第{index}帧解码失败")
            image=canonical(frame);timestamp=meta["capture_timestamp"]
            dt=timestamp-last_time if last_time is not None else 0
            actor=player_candidate(image,last_player)
            if actor:
                last_player=np.array((actor["x_px"],actor["y_px"]))
                missing=0
            else:
                missing+=1
                # A blinking robot is not permission to acquire a distant
                # blue/white star cluster as a new player identity. Keep the
                # last observed spatial gate; publish no ghost position.
            command_index=bisect_right(command_times,timestamp)-1
            actual=commands[command_index]["state"] if command_index>=0 else None
            row={"video_frame_index":index,"capture_sequence":meta["capture_sequence"],
                 "timestamp":timestamp,"seconds":timestamp-origin,"player":actor,
                 "actual_last_sent":actual,"flow":flow_proxy(last_image,image,dt),
                 "event_label":"unlabelled_not_normal_clear"}
            rows.append(row)
            last_image,last_time=image,timestamp
    finally:
        cap.release()
    visible=[row for row in rows if row["player"]]
    velocities=[]
    for a,b in zip(rows,rows[1:]):
        dt=b["timestamp"]-a["timestamp"]
        if a["player"] and b["player"] and 0<dt<=.08:
            velocities.append((b["seconds"],(b["player"]["x_px"]-a["player"]["x_px"])/dt))
    active_transitions=[r for r in transitions if start<=r["seconds"]<end]
    zr_reports=sum("ZR" in r["state"]["buttons"] for r in commands
                   if origin+start<=r["after"]<origin+end)
    reason=["只有一个运行回合且没有完整正常/碰撞事件标签；不拟合或声明已校准参数。",
            "角色漏检可能来自闪烁/遮挡/颜色条件，不能仅凭漏检断言撞墙。",
            "背景流包含纹理和摄像机变化，尚未用独立地标验证，不能直接当世界前进速度。"]
    if zr_reports==0:reason.append("该运行片段没有ZR报告，无法比较ZR与非ZR转向或增速。")
    report={"scope":"read_only_live_video_action_diagnostics_not_calibration",
            "session":str(session.resolve()),"timebase":"frames.jsonl capture_timestamp joined to commands.jsonl after",
            "start_seconds":start,"end_seconds":end,"stride":stride,"samples":len(rows),
            "visible_candidate_samples":len(visible),"visible_candidate_fraction":len(visible)/len(rows),
            "zr_reports_in_window":zr_reports,"command_transitions":active_transitions,
            "player_x_range_px":[min(r["player"]["x_px"] for r in visible),max(r["player"]["x_px"] for r in visible)] if visible else None,
            "observed_local_velocity_px_per_second_quantiles":np.percentile([v for _,v in velocities],[5,50,95]).tolist() if velocities else None,
            "parameters":{"steering_acceleration":None,"steering_delay":None,"forward_terminal_speed":None,
                          "forward_constant_acceleration":None,"zr_response_difference":None},
            "calibrated":False,"limitations":reason}
    (output/"analysis.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),encoding="utf-8")
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    with (output/"trajectory.csv").open("w",encoding="utf-8",newline="") as file:
        writer=csv.writer(file);writer.writerow(["seconds","capture_sequence","player_x_px","player_y_px","left_x","zr","background_y_heights_per_second","flow_tracks"])
        for row in rows:
            state=row["actual_last_sent"] or {};p=row["player"] or {};flow=row["flow"]
            writer.writerow([row["seconds"],row["capture_sequence"],p.get("x_px"),p.get("y_px"),
                             state.get("left_stick",{}).get("x"),"ZR" in state.get("buttons",[]),
                             flow.get("screen_flow_y_heights_per_second"),flow.get("tracks")])
    plot(rows,output)
    return report


def plot(rows,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    t=[r["seconds"] for r in rows]
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    axes[0].plot(t,[r["player"]["x_px"] if r["player"] else np.nan for r in rows],".",markersize=2)
    axes[0].set_ylabel("角色横坐标候选/像素",fontproperties=font)
    axes[1].step(t,[(r["actual_last_sent"] or {}).get("left_stick",{}).get("x",0)/32767 for r in rows],where="post")
    axes[1].set_ylabel("实际发送左摇杆x",fontproperties=font)
    axes[2].plot(t,[r["flow"].get("screen_flow_y_heights_per_second",np.nan) for r in rows],linewidth=.7)
    axes[2].set_ylabel("背景纵向纹理流/画面高每秒",fontproperties=font)
    axes[2].set_xlabel("相对采集开始时间/秒",fontproperties=font)
    axes[0].set_title("MINING 实机单局诊断：候选轨迹与背景流，非已校准动力学",fontproperties=font)
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(output/"trajectory.png",dpi=140);plt.close(fig)


def steady_axis_response(points, commands, assumed_delay=.125):
    """Observed vx/axis during actual fixed-axis plateaus, not proven steady state.

    The fixed .125s delay and burn-in are exposed assumptions. Increasing the
    burn-in tests sensitivity; it does not independently establish settling.
    """
    transitions=[];previous=None
    for command in commands:
        state=command["state"]
        key=(state.get("left_stick",{}).get("x",0),"ZR" in state.get("buttons",[]))
        if key!=previous:
            transitions.append({"t":command["after"],"u":key[0]/32767,"zr":key[1]})
            previous=key
    plateaus=[dict(a,end=b["t"],duration=b["t"]-a["t"],plateau_id=i)
              for i,(a,b) in enumerate(zip(transitions,transitions[1:])) if abs(a["u"])>=.2]
    origin=points[0]["t"];results=[];raw=[]
    for burn in (.2,.35,.5,.7):
        groups={False:[],True:[]}
        for plateau in plateaus:
            selected=[p for p in points if p["eligible"] and plateau["t"]+assumed_delay+burn<=p["t"]<=plateau["end"]+assumed_delay]
            for a,b in zip(selected,selected[1:]):
                dt=b["t"]-a["t"]
                if b["index"]!=a["index"]+1 or b["track"]!=a["track"] or not 0<dt<=.2:continue
                velocity=(b["x"]-a["x"])/dt
                item={"zr":plateau["zr"],"burn_in_seconds":burn,"assumed_delay_seconds":assumed_delay,
                      "plateau_id":plateau["plateau_id"],"plateau_start_seconds":plateau["t"]-origin,
                      "plateau_end_seconds":plateau["end"]-origin,"start_seconds":a["t"]-origin,
                      "end_seconds":b["t"]-origin,"sequence_start":a["sequence"],"sequence_end":b["sequence"],
                      "axis":plateau["u"],"vx":velocity,"vx_per_axis":velocity/plateau["u"],"dt":dt}
                groups[plateau["zr"]].append(item);raw.append(item)
        for zr,group in groups.items():
            results.append({"zr":zr,"burn_in_seconds":burn,"samples":len(group),
                "plateaus":len({g["plateau_id"] for g in group}),"observed_interval_seconds":sum(g["dt"] for g in group),
                "ratio_quantiles_10_25_50_75_90":np.percentile([g["vx_per_axis"] for g in group],[10,25,50,75,90]).tolist() if group else None,
                "wrong_sign_samples":sum(g["vx_per_axis"]<0 for g in group)})
    by_plateau=[]
    for pid in sorted({r["plateau_id"] for r in raw if r["burn_in_seconds"]==.35}):
        group=[r for r in raw if r["plateau_id"]==pid and r["burn_in_seconds"]==.35]
        by_plateau.append({"plateau_id":pid,"zr":group[0]["zr"],"axis":group[0]["axis"],
            "actual_start_seconds":group[0]["plateau_start_seconds"],"actual_end_seconds":group[0]["plateau_end_seconds"],
            "samples":len(group),"sequence_start":min(g["sequence_start"] for g in group),
            "sequence_end":max(g["sequence_end"] for g in group),
            "mean_vx":sum(g["vx"]*g["dt"] for g in group)/sum(g["dt"] for g in group),
            "median_vx_per_axis":float(np.median([g["vx_per_axis"] for g in group]))})
    return {"scope":"fixed_actual_axis_observed_near_steady_candidate_not_calibration",
            "assumed_delay_seconds":assumed_delay,"summaries":results,"plateaus_after_035s_burn":by_plateau,
            "limitations":["保持axis后按假定.125s画面延迟再丢弃过渡，延迟和稳定时间尚未独立测定。",
                "样本为真实可见同身份清障相邻位置，非像素/事件标注真值；多个样本共享同一次输入平台。",
                "无输入强度随机化，不能证明线性或排除前进速度、道路曲率和相机因素。"]},raw


def plot_fixed_axis_examples(points, steady, output):
    """Show actual equal-input examples without inventing settled-model truth."""
    examples=[]
    for zr in (False,True):
        choices=[p for p in steady["plateaus_after_035s_burn"] if p["zr"]==zr and .45<=p["axis"]<=.55 and p["samples"]>=3]
        if choices:examples.append(max(choices,key=lambda p:p["samples"]))
    if len(examples)!=2:return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    origin=points[0]["t"];delay=steady["assumed_delay_seconds"]
    for ax,example in zip(axes,examples):
        start,end=example["actual_start_seconds"],example["actual_end_seconds"]
        q=[p for p in points if p["x"] is not None and start-.15<=p["t"]-origin<=end+.20]
        if not q:continue
        base=q[0]["x"]
        ax.plot([p["t"]-origin-start for p in q],[p["x"]-base for p in q],"o-",ms=3,lw=.7,label="可见角色位置")
        clear=[p for p in q if p["eligible"]]
        ax.scatter([p["t"]-origin-start for p in clear],[p["x"]-base for p in clear],s=24,facecolors="none",edgecolors="green",label="通过清障筛选")
        ax.axvline(delay+.35,color="orange",ls="--",label="假定延迟+过渡丢弃")
        ax.axvline(end-start,color="grey",ls=":",label="实际输入平台结束")
        title="ZR按住" if example["zr"] else "ZR松开"
        ax.set_title(f"{title}，实际轴 {example['axis']:.3f}；平均横速 {example['mean_vx']:.3f}",fontproperties=font)
        ax.set_xlabel("相对实际发送平台开始/秒",fontproperties=font)
        ax.grid(alpha=.25);ax.legend(prop=font,fontsize=8)
    axes[0].set_ylabel("相对起始角色横坐标/画面宽度",fontproperties=font)
    fig.suptitle("相同摇杆输入的实机轨迹实例：近稳态候选，未标定",fontproperties=font)
    fig.tight_layout();fig.savefig(output/"fixed_axis_examples.png",dpi=150);plt.close(fig)


def analyse_trace(round_directory, output, minimum_sequence=None, maximum_sequence=None):
    """Exploratory closed-loop lateral response, never a calibrated profile.

    Only actual successful transport reports are treated as input. Five
    consecutive visible detections, unchanged identity, clear-wall/hazard
    diagnostics and no synthetic-source tag are required around a sample.
    The geometry diagnostics are themselves unvalidated, so passing this
    filter does not establish a collision-free interval.
    """
    from collections import Counter
    directory, output = Path(round_directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trace = [json.loads(line) for line in (directory/"trace.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    session = Path(summary["session"])
    commands = [json.loads(line) for line in (session/"commands.jsonl").read_text(encoding="utf-8").splitlines()]
    commands = sorted((c for c in commands if c.get("success")), key=lambda c: c["after"])
    command_times = [c["after"] for c in commands]
    counts = Counter(); points=[]
    for index, row in enumerate(trace):
        observation = row["observation"]
        diagnostics = row["decision"].get("diagnostics", {})
        player = next((e for e in observation["entities"] if e["kind"] == "player"), None)
        counts["trace"] += 1
        eligible = False
        within = ((minimum_sequence is None or observation["sequence"] >= minimum_sequence)
                  and (maximum_sequence is None or observation["sequence"] <= maximum_sequence))
        if observation["phase"] == "playing" and within:
            counts["playing"] += 1
            if player and not player.get("source") and observation["metrics"].get("player_detected") is True and not observation["metrics"].get("ambiguous_player"):
                counts["visible_real_source"] += 1
                if player.get("samples", 0) >= 3:
                    counts["track_samples_ge3"] += 1
                    if diagnostics.get("wall_distance", -1) > .05:
                        counts["wall_distance_gt_05"] += 1
                        hazard = diagnostics.get("nearest_hazard_depth")
                        if hazard is None or hazard > .15:
                            counts["hazard_clear"] += 1
                            eligible = True
        points.append({"index":index, "t":observation["timestamp"], "sequence":observation["sequence"],
                       "x":player["x"] if player else None, "vx":player["vx"] if player else None,
                       "track":player["track_id"] if player else None,
                       "eligible":eligible, "wall_distance":diagnostics.get("wall_distance"),
                       "nearest_hazard_depth":diagnostics.get("nearest_hazard_depth")})

    def command_interval(start, end):
        first = bisect_right(command_times, start)-1
        if first < 0 or start-command_times[first] > .12:
            return None
        knots = [start]+[t for t in command_times[first+1:] if start < t < end]+[end]
        total=0.; modes=set(); largest_gap=0.
        for left,right in zip(knots,knots[1:]):
            k=bisect_right(command_times,left)-1
            state=commands[k]["state"]
            modes.add("ZR" in state.get("buttons", []))
            total += (right-left)*state.get("left_stick", {}).get("x",0)/32767
            largest_gap=max(largest_gap,right-command_times[k])
        if len(modes)!=1 or largest_gap>.12:
            return None
        return total/(end-start), modes.pop()

    # x-derived velocities are interval averages; tracker vx is a five-sample
    # historical average, with a direction-change reset. Fit both and expose
    # disagreement instead of reporting the tracker's extra lag as game delay.
    samples=[]
    for index in range(2,len(points)-2):
        window=points[index-2:index+3]
        if not all(p["eligible"] for p in window) or len({p["track"] for p in window})!=1:
            continue
        if not all(0 < b["t"]-a["t"] <= .20 for a,b in zip(window,window[1:])):
            continue
        a,b,c=window[1:4]
        samples.append({"index":index,"track":b["track"],"t":b["t"],
                        "start":(a["t"]+b["t"])/2,"end":(b["t"]+c["t"])/2,
                        "v":(b["x"]-a["x"])/(b["t"]-a["t"]),
                        "next":(c["x"]-b["x"])/(c["t"]-b["t"]),
                        "tracker_start":b["t"],"tracker_end":c["t"],
                        "tracker_v":b["vx"],"tracker_next":c["vx"]})
    counts["five_visible_clear_same_track_centers"] = len(samples)
    origin=points[0]["t"]
    candidates=[]; aligned=[]
    for velocity in ("position_intervals", "tracker_smoothed_vx"):
        for delay in np.arange(0,.301,.025):
            groups={False:[],True:[]}
            for sample in samples:
                start,end=(sample["start"],sample["end"]) if velocity=="position_intervals" else (sample["tracker_start"],sample["tracker_end"])
                command=command_interval(start-delay,end-delay)
                if command is None:continue
                u,zr=command
                v,nxt=(sample["v"],sample["next"]) if velocity=="position_intervals" else (sample["tracker_v"],sample["tracker_next"])
                item={"seconds":sample["t"]-origin,"index":sample["index"],"track":sample["track"],"dt":end-start,
                      "v":v,"next":nxt,"u":u,"zr":zr,"delay":float(delay),"velocity_source":velocity}
                groups[zr].append(item)
                aligned.append(item)
            for zr,group in groups.items():
                if len(group)<12:continue
                dt=np.array([g["dt"] for g in group]);v=np.array([g["v"] for g in group]);nxt=np.array([g["next"] for g in group]);u=np.array([g["u"] for g in group])
                matrix=np.column_stack((dt*(v+nxt)/2,dt*u,dt))
                coefficients=np.linalg.lstsq(matrix,nxt-v,rcond=None)[0]
                residual=nxt-(v+matrix@coefficients)
                baseline=nxt-v
                candidates.append({"velocity_source":velocity,"zr":zr,"delay_seconds":round(float(delay),3),
                    "samples":len(group),"observed_interval_seconds":float(dt.sum()),
                    "input_span":[float(u.min()),float(u.max())],"observed_velocity_quantiles":np.percentile(np.r_[v,nxt],[5,50,95]).tolist(),
                    "gain":float(coefficients[1]),"drag":float(-coefficients[0]),"bias":float(coefficients[2]),
                    "one_step_rmse":float(np.sqrt(np.mean(residual**2))),
                    "zero_acceleration_rmse":float(np.sqrt(np.mean(baseline**2))),
                    "matrix_condition":float(np.linalg.cond(matrix)),
                    "physically_compatible_signs":bool(coefficients[0]<0 and coefficients[1]>0),
                    "time_blocks":len({int((g["seconds"])/5) for g in group}),
                    "scope":"unconstrained_all_data_exploration_no_holdout_no_calibration"})
    common_support=[]
    for velocity in ("position_intervals", "tracker_smoothed_vx"):
        for zr in (False,True):
            same=[a for a in aligned if a["velocity_source"]==velocity and a["zr"]==zr]
            by_delay={delay:[a for a in same if a["delay"]==delay] for delay in {a["delay"] for a in same}}
            ids=set.intersection(*({a["index"] for a in group} for group in by_delay.values())) if by_delay else set()
            for delay,group in sorted(by_delay.items()):
                group=[a for a in group if a["index"] in ids]
                if len(group)<12:continue
                dt=np.array([a["dt"] for a in group]);v=np.array([a["v"] for a in group]);nxt=np.array([a["next"] for a in group]);u=np.array([a["u"] for a in group])
                matrix=np.column_stack((dt*(v+nxt)/2,dt*u,dt))
                co=np.linalg.lstsq(matrix,nxt-v,rcond=None)[0]
                common_support.append({"velocity_source":velocity,"zr":zr,"delay_seconds":round(float(delay),3),"samples":len(group),
                    "gain":float(co[1]),"drag":float(-co[0]),"bias":float(co[2]),
                    "one_step_rmse":float(np.sqrt(np.mean((nxt-v-matrix@co)**2))),
                    "scope":"identical_observations_for_every_delay_no_independent_validation"})
    periods=[]
    for point in points:
        if not point["eligible"]:continue
        if periods and point["index"]==periods[-1]["last_index"]+1 and point["track"]==periods[-1]["track"] and point["t"]-origin-periods[-1]["end_seconds"]<=.2:
            periods[-1].update(last_index=point["index"],end_seconds=point["t"]-origin)
            periods[-1]["samples"]+=1
        else:
            periods.append({"track":point["track"],"first_index":point["index"],"last_index":point["index"],
                            "start_seconds":point["t"]-origin,"end_seconds":point["t"]-origin,"samples":1})
    best=[]
    for velocity in ("position_intervals", "tracker_smoothed_vx"):
        for zr in (False,True):
            choices=[c for c in candidates if c["velocity_source"]==velocity and c["zr"]==zr]
            if choices:best.append(min(choices,key=lambda c:c["one_step_rmse"]))
    steady,steady_samples=steady_axis_response(points,commands)
    plot_fixed_axis_examples(points,steady,output)
    report={"scope":"exploratory_closed_loop_response_not_calibration", "calibrated":False,
            "trace":str((directory/"trace.jsonl").resolve()),"session":str(session),"counts":dict(counts),
            "manual_sequence_window":{"minimum":minimum_sequence,"maximum":maximum_sequence},
            "units":{"gain":"viewport_widths_per_second_squared_per_normalized_stick", "drag":"per_second", "bias":"viewport_widths_per_second_squared"},
            "equation":"delta_v = dt * (-drag * mean(v,next_v) + gain * mean_actual_delayed_axis + bias)",
            "best_all_data_candidates_not_recommendations":best,"all_delay_candidates":candidates,
            "all_delay_common_support":common_support,"eligible_periods_relative_first_trace":periods,
            "fixed_actual_axis_response":steady,
            "limitations":["单局闭环相关性探索，无独立留出验证，不能发布已标定参数。",
              "wall/hazard过滤依赖未经验证的检测，仍可能包含真碰撞、误墙或边界推挤。",
              "每样本周围五次真实可见同身份、wall_distance>.05、hazard为空或>.15；不能排除未识别障碍。",
              "tracker vx是最多五帧回看平均并在反向时重置；其最优时延含视觉滤波延迟。",
              "位置差分对约10Hz观测、像素量化、HTTP头时间估计敏感。",
              "实际发送after只证明后端写入，不证明主机执行的确切时间。",
              "不同delay排除跨ZR切换样本，样本组成略变，最小残差不是可辨识时延。"]}
    (output/"response_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (output/"filtered_observations.jsonl").write_text("".join(json.dumps(p)+"\n" for p in points),encoding="utf-8")
    with (output/"aligned_response_samples.csv").open("w",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(aligned[0]) if aligned else ["seconds"])
        writer.writeheader();writer.writerows(aligned)
    with (output/"steady_axis_samples.csv").open("w",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(steady_samples[0]) if steady_samples else ["seconds"])
        writer.writeheader();writer.writerows(steady_samples)
    return report


def summarize_round(directory, observed_score=None):
    """Full-trace mode counts and time-integrated actual reports, not impacts."""
    from collections import Counter
    directory=Path(directory)
    trace=[json.loads(line) for line in (directory/"trace.jsonl").read_text(encoding="utf-8").splitlines()]
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    playing=[row for row in trace if row["observation"]["phase"]=="playing"]
    start,end=playing[0]["observation"]["timestamp"],playing[-1]["observation"]["timestamp"]
    active=[r for r in trace if start<=r["observation"]["timestamp"]<=end]
    phases=Counter(r["observation"]["phase"] for r in active)
    modes=Counter(r["decision"].get("diagnostics",{}).get("mode","NONE") for r in playing)
    durations=Counter();runs=[]
    for row,nxt in zip(active,active[1:]):
        phase=row["observation"]["phase"];mode=row["decision"].get("diagnostics",{}).get("mode","NONE")
        durations[phase+":"+mode]+=nxt["observation"]["timestamp"]-row["observation"]["timestamp"]
    for index,row in enumerate(active):
        phase=row["observation"]["phase"];d=row["decision"].get("diagnostics",{})
        key="unknown" if phase=="unknown" else "wall_recovery" if d.get("mode")=="wall_recovery" else None
        if not key:continue
        t=row["observation"]["timestamp"]
        if runs and runs[-1]["kind"]==key and runs[-1]["last_index"]==index-1:
            runs[-1].update(last_index=index,last_sequence=row["observation"]["sequence"],end_seconds=t-start)
            runs[-1]["samples"]+=1
        else:runs.append({"kind":key,"first_index":index,"last_index":index,"first_sequence":row["observation"]["sequence"],
                          "last_sequence":row["observation"]["sequence"],"start_seconds":t-start,"end_seconds":t-start,"samples":1})
    commands=[json.loads(line) for line in (Path(summary["session"])/"commands.jsonl").read_text(encoding="utf-8").splitlines()]
    commands=sorted((c for c in commands if c.get("success")),key=lambda c:c["after"])
    times=[c["after"] for c in commands];knots=[start]+[t for t in times if start<t<end]+[end]
    actual=Counter();max_gap=0.;changes=0;previous=None
    for left,right in zip(knots,knots[1:]):
        index=bisect_right(times,left)-1
        if index<0:continue
        state=commands[index]["state"];x=state.get("left_stick",{}).get("x",0);buttons=state.get("buttons",[])
        actual["neutral" if x==0 and not buttons else "active"]+=right-left
        actual["ZR" if "ZR" in buttons else "not_ZR"]+=right-left
        actual["axis" if x else "no_axis"]+=right-left
        max_gap=max(max_gap,right-times[index])
        if previous is not None and previous!=state:changes+=1
        previous=state
    screenshots=[]
    for path in sorted(directory.glob("frame-*.jpg")):
        sequence=int(path.stem.split("-")[-1]);row=min(trace,key=lambda r:abs(r["observation"]["sequence"]-sequence));o=row["observation"]
        screenshots.append({"path":str(path.resolve()),"screenshot_sequence":sequence,"observation_sequence":o["sequence"],
            "seconds":o["timestamp"]-start,"phase":o["phase"],"mode":row["decision"].get("diagnostics",{}).get("mode"),
            "player_detected":o["metrics"].get("player_detected"),"wall_distance":row["decision"].get("diagnostics",{}).get("wall_distance")})
    return {"directory":str(directory.resolve()),"scope":"descriptive_full_trace_and_transport_not_collision_accuracy",
        "manually_observed_score":observed_score,"score_source":str((directory/"last-frame.png").resolve()) if observed_score is not None else None,
        "active_window_seconds":end-start,"playing_mode_samples":dict(modes),"active_window_phase_samples":dict(phases),
        "advice_mode_seconds_sample_hold":dict(durations),"actual_successful_report_seconds":dict(actual),
        "actual_state_transitions":changes,"maximum_actual_send_gap_seconds":max_gap,
        "wake_repair_metric_nonempty_samples":sum(bool(r["observation"]["metrics"].get("wake_repaired_rows")) for r in active),
        "runs":runs,"screenshots":screenshots,"collision_count":None,"calibrated":False,
        "limitations":["模式样本来自完整trace；截图只用于局部人工核查，不能当controller采样总数。",
            "wall_recovery是几何判定，并不等于真实撞墙。unknown包括闪烁、红色外观、特效和普通漏检。",
            "时间窗由首末playing观测定义，末段可能包含已到零秒但结果尚未识别的停滞。",
            "发送持续时间不等于已验证主机执行时间；计分由调用者人工读取结算图提供。"]}


def analyse_boulder_forward(directory, output, verified_tracks, minimum_sequence=None, maximum_sequence=None, window_size=5):
    """Scroll speed from source-image-verified static rocks, never player.y."""
    from collections import defaultdict
    directory,output=Path(directory),Path(output);output.mkdir(parents=True,exist_ok=True)
    trace=[json.loads(line) for line in (directory/"trace.jsonl").read_text(encoding="utf-8").splitlines()]
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    commands=[json.loads(line) for line in (Path(summary["session"])/"commands.jsonl").read_text(encoding="utf-8").splitlines()]
    commands=sorted((c for c in commands if c.get("success")),key=lambda c:c["after"])
    modes=[];previous=None
    for command in commands:
        zr="ZR" in command["state"].get("buttons",[])
        if zr!=previous:modes.append((command["after"],zr));previous=zr
    mode_times=[m[0] for m in modes];history=defaultdict(list);selected=[];all_rocks=[];origin=trace[0]["observation"]["timestamp"]
    for index,row in enumerate(trace):
        observation=row["observation"];t=observation["timestamp"];sequence=observation["sequence"]
        within=(minimum_sequence is None or sequence>=minimum_sequence) and (maximum_sequence is None or sequence<=maximum_sequence)
        if not within or observation["phase"] not in ("playing","unknown"):continue
        for rock in observation["entities"]:
            if rock["kind"] not in ("rock","obstacle") or rock["track_id"] not in verified_tracks:continue
            all_rocks.append({"sequence":sequence,"seconds":t-origin,"track":rock["track_id"],"x":rock["x"],"y":rock["y"],"w":rock["w"],"h":rock["h"],"reported_scroll_speed":-rock["vy"],"samples":rock["samples"]})
            items=history[rock["track_id"]];items.append((index,t,rock));items[:]=items[-window_size:]
            if len(items)<window_size or rock["samples"]<window_size or not .25<=rock["y"]<=.80 or rock["w"]<.055 or rock["h"]<.08:continue
            if any(b[0]!=a[0]+1 or not 0<b[1]-a[1]<=.2 for a,b in zip(items,items[1:])):continue
            if (np.ptp([a[2]["x"] for a in items])>.01 or np.ptp([a[2]["w"] for a in items])>.01
                    or np.ptp([a[2]["h"] for a in items])>.012):continue
            first=bisect_right(mode_times,items[0][1]-.125)-1;last=bisect_right(mode_times,t-.125)-1
            if min(first,last)<0 or first!=last:continue
            clear=True
            for old_index,_,_ in items:
                old=trace[old_index];d=old["decision"].get("diagnostics",{})
                hazard=d.get("nearest_hazard_depth")
                if old["observation"]["phase"]!="playing" or d.get("wall_distance",-1)<=.05 or (hazard is not None and hazard<=.15):clear=False
            selected.append({"sequence":sequence,"seconds":t-origin,"track":rock["track_id"],"zr":modes[last][1],
                "x":rock["x"],"y":rock["y"],"w":rock["w"],"h":rock["h"],"samples":rock["samples"],
                "history_first_sequence":trace[items[0][0]]["observation"]["sequence"],"history_seconds":t-items[0][1],
                "scroll_speed":-(rock["y"]-items[0][2]["y"])/(t-items[0][1]),"reported_minus_vy":-rock["vy"],
                "history_window_size":window_size,"history_any_center_above_08":any(a[2]["y"]>.8 for a in items),
                "history_all_boxes_inside_dark_detector_vertical_roi":all(a[2]["y"]+a[2]["h"]/2<455/480 for a in items),
                "current_phase":observation["phase"],
                "mode_age_at_history_start":items[0][1]-.125-modes[last][0],"whole_window_clear_diagnostics":clear})
    summaries=[]
    for zr in (False,True):
        for clear in (False,True):
            group=[s for s in selected if s["zr"]==zr and s["mode_age_at_history_start"]>=.35 and (not clear or s["whole_window_clear_diagnostics"])]
            summaries.append({"zr":zr,"require_whole_window_clear_diagnostics":clear,"overlapping_window_count":len(group),
                "track_ids":sorted({s["track"] for s in group}),
                "observed_scroll_quantiles_10_25_50_75_90":np.percentile([s["scroll_speed"] for s in group],[10,25,50,75,90]).tolist() if group else None,
                "sequence_ids":[s["sequence"] for s in group]})
    pairs=[]
    for sequence in {s["sequence"] for s in all_rocks}:
        rocks=[s for s in all_rocks if s["sequence"]==sequence and s["samples"]>=3 and .25<=s["y"]<=.80]
        if len(rocks)>=2:pairs.append({"sequence":sequence,"track_ids":[s["track"] for s in rocks],
            "reported_minus_vy":[s["reported_scroll_speed"] for s in rocks],
            "range":max(s["reported_scroll_speed"] for s in rocks)-min(s["reported_scroll_speed"] for s in rocks)})
    end_time=trace[-1]["observation"]["timestamp"]
    zr_segments=[{"start_seconds":max(a[0],origin)-origin,"end_seconds":min(b[0],end_time)-origin,
                  "seconds":min(b[0],end_time)-max(a[0],origin)} for a,b in zip(modes,modes[1:])
                 if a[1] and max(a[0],origin)<min(b[0],end_time)]
    report={"scope":"verified_boulder_landmark_scroll_not_calibrated_world_velocity", "calibrated":False,
        "directory":str(directory.resolve()),"manually_source_image_verified_track_ids":sorted(verified_tracks),
        "manual_sequence_window":{"minimum":minimum_sequence,"maximum":maximum_sequence},
        "units":"viewport_heights_per_second","estimator":"-(rock.y_last-rock.y_first)/observation_time_span; compare -rock.vy; never player.y",
        "timebase":"observation.timestamp (HTTP capture-header estimate), actual commands.after",
        "assumed_input_observation_delay_seconds":.125,"detection_window_size":window_size,"selected_detection_windows":len(selected),"summaries":summaries,
        "actual_continuous_ZR_segments":zr_segments,
        "simultaneous_rock_reported_velocity_agreement":pairs,
        "selected_windows":selected,
        "limitations":["手工核查来源图证明所选track有真实石头，不等于逐帧真值或永久身份保证。",
            f"{window_size}帧窗口互相重叠，计数不能当独立试验次数。尺寸/横坐标稳定门槛只排除部分错配。",
            "石头假定静止在地图中；这是画面滚屏测量，未经独立世界距离校准。",
            "normal/ZR分组包含撞停、重新加速和转弯；分位数不能直接当终端速度。",
            "整窗清障过滤依赖尚未验证的道路/石头检测，严格样本不足时不能确认无碰撞模型。",
            "实际发送after仅证明后端发送，使用.125s对齐假设；模式切换过渡尚未拟合。",
            "trace的观测时间由HTTP采集头推估，短窗速度有时间抖动误差；不能用单个两帧峰值发布精确常数。"]}
    (output/"boulder_forward_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    with (output/"boulder_forward_samples.csv").open("w",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(selected[0]) if selected else ["sequence"]);writer.writeheader();writer.writerows(selected)
    return report


def main():
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description="按采集/发送真实时间对齐MINING录屏，导出候选轨迹和背景流诊断，不连接控制器")
    parser.add_argument("session",type=Path)
    parser.add_argument("--output",type=Path,default=Path("artifacts/mining_recording_analysis"))
    parser.add_argument("--start",type=float,default=126)
    parser.add_argument("--end",type=float,default=171)
    parser.add_argument("--stride",type=int,default=2)
    parser.add_argument("--trace",action="store_true",help="session改为含summary/trace的监督轮次目录，仅探索横向响应")
    parser.add_argument("--compare",type=Path,help="对比两个监督轮次；统计完整trace和实际commands，不拟合或统计真碰撞次数")
    parser.add_argument("--observed-score",type=int,help="人工从第一轮结算图确认的分数，非OCR")
    parser.add_argument("--comparison-score",type=int,help="人工从对比轮结算图确认的分数，非OCR")
    parser.add_argument("--minimum-sequence",type=int,help="离线trace拟合的人工核查起始采集序列")
    parser.add_argument("--maximum-sequence",type=int,help="离线trace拟合的人工核查结束采集序列，剔除结果动画")
    parser.add_argument("--forward",action="store_true",help="仅以已人工核查真石头的轨迹测滚屏速度，不使用player.y")
    parser.add_argument("--verified-tracks",default="",help="已人工核查真石头的track id，逗号分隔")
    parser.add_argument("--forward-window",type=int,default=5,choices=(2,3,5),help="前向测量窗口；2/3帧是放宽的短窗，时间抖动影响更大")
    args=parser.parse_args()
    if args.start<0 or args.end<=args.start or args.stride<1:parser.error("时间区间或stride无效")
    if args.forward:
        if not args.verified_tracks:parser.error("--forward必须明确传入已人工核查的--verified-tracks")
        report=analyse_boulder_forward(args.session,args.output,{int(t) for t in args.verified_tracks.split(',')},args.minimum_sequence,args.maximum_sequence,args.forward_window)
    elif args.compare:
        report={"rounds":[summarize_round(args.session,args.observed_score),summarize_round(args.compare,args.comparison_score)]}
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/"round_comparison.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    else:report=analyse_trace(args.session,args.output,args.minimum_sequence,args.maximum_sequence) if args.trace else analyse(args.session,args.output,args.start,args.end,args.stride)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
