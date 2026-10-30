"""Build a local, single-run LANDING atlas from source-indexed screenshots."""
from pathlib import Path
import argparse
import json
import sys
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from raiders_helper.landing import LandingAtlas,build_atlas,canonical_frame,estimate_pose,decide_landing
from raiders_helper.models import Entity,Observation,Game


def check_samples():
    """Same-run held-out times; stage labels were manually read from frames."""
    root=Path(__file__).resolve().parents[1]
    atlas=LandingAtlas(root/"artifacts/landing_atlas/index.json")
    frame_ids=[57780,58380,58980,59580,60180,60780,61380,61980,62580,63180,63780,64380,
               64980,65580,66180,66780,67380,67980,68580,69180,69780,70380,70980,71580,72180,72780]
    stages=[1,1,2,2,3,3,4,4,5,5,6,6,7,7,7,7,8,8,8,9,9,9,9,10,10,10]
    source_ids={entry["frame"] for entry in json.loads((root/"maps/landing/index.json").read_text(encoding="utf-8"))["layouts"]}
    rows=[]
    for frame_id,stage in zip(frame_ids,stages):
        source=root/f"artifacts/landing_map_candidates/frame-{frame_id:08d}.png"
        frame=canonical_frame(cv2.imdecode(np.frombuffer(source.read_bytes(),np.uint8),cv2.IMREAD_COLOR))
        started=perf_counter()
        match=atlas.match(frame)
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        blue=cv2.inRange(hsv,np.array((105,100,55)),np.array((132,255,255)))
        blue[:40]=0
        blue[440:]=0
        count,_,stats,centers=cv2.connectedComponentsWithStats(blue)
        candidates=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=15]
        pose,decision={},None
        if candidates:
            i=max(candidates,key=lambda i:stats[i,cv2.CC_STAT_AREA])
            x,y=centers[i]
            player=Entity("player",x/640,y/480,0.05,0.07,0.95,samples=3)
            pose=estimate_pose(frame,player,atlas.path.parent/"robot_upright.png")
            metrics={**pose,"map_index":match["map_index"],"map_match_score":match["map_match_score"]}
            obs=Observation(Game.LANDING,frame_id/60,frame_id,"playing",0.95,(player,),metrics=metrics)
            decision=decide_landing(obs,{"atlas_registry":str(atlas.path)})
        clear_overlay=frame_id==72780
        rows.append({"frame":frame_id,"seconds":frame_id/60,"stage_label":stage,
                     "atlas_source_frame":frame_id in source_ids,"clear_overlay":clear_overlay,
                     "map_index":match["map_index"],"map_match_score":match["map_match_score"],
                     "expected_match":-1 if clear_overlay else stage-1,
                     "map_result_correct":match["map_index"]==(-1 if clear_overlay else stage-1),
                     "pose_known":pose.get("pose_known",False),"pose_score":pose.get("pose_confidence",0),
                     "suggested_buttons":sorted(b.value for b in decision.state.buttons) if decision else [],
                     "reason":decision.reason if decision else "未识别到机器人",
                     "elapsed_ms":(perf_counter()-started)*1000})
    heldout=[row for row in rows if not row["atlas_source_frame"]]
    report={"scope":"same_reference_run_sparse_frame_check",
            "limitations":"10帧构建图库、16帧为同局其他时刻（含1帧CLEAR拒识）；不能代表跨视频泛化、姿态准确率或实机控制成功。姿态无角度真值，按钮无反事实得分。",
            "heldout_count":len(heldout),"heldout_map_checks_passed":sum(row["map_result_correct"] for row in heldout),
            "rows":rows}
    target=root/"artifacts/landing_atlas/video_check.json"
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"同局独立时刻地图检查：{report['heldout_map_checks_passed']}/{len(heldout)}；报告 {target}")
    return report


if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description="构建 LANDING 单局地图库与稀疏视频检查")
    parser.add_argument("--check-samples",action="store_true")
    args=parser.parse_args()
    manifest=build_atlas()
    print(f"已构建 {len(manifest['layouts'])} 个录播布局，非完整地图库；策略仍需实机校准。")
    if args.check_samples:
        check_samples()
