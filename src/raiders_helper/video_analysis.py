"""Offline per-frame advice and timings. Never connects to a controller."""
from dataclasses import asdict
import json
from pathlib import Path
import time
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .capture.models import FramePacket
from .advice_overlay import advice_geometry
from .capture.video_source import OpenCvVideoSource
from .runtime import Engine
from .trace import TraceWriter, state_dict
from .vision import create_recognizer, crop_roi, load_profile


@lru_cache(maxsize=1)
def _fonts():
    try:
        return (ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",16), ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",13))
    except OSError:
        return (ImageFont.load_default(), ImageFont.load_default())


def render_analysis(frame, observation, decision, profile):
    area = cv2.resize(crop_roi(frame, profile.get("viewport", [0, 0, 1, 1])), (640, 480))
    im = Image.fromarray(cv2.cvtColor(area, cv2.COLOR_BGR2RGB))
    canvas = Image.new("RGB", (980, 480), "#19212a")
    canvas.paste(im, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font,small=_fonts()
    lines, markers = advice_geometry(observation, decision)
    for a, b in lines:
        draw.line((a[0]*640,a[1]*480,b[0]*640,b[1]*480),fill="#e69ae5",width=2)
    for x,y in markers:
        x,y=x*640,y*480
        draw.line((x-8,y,x+8,y),fill="#ffe071",width=3)
        draw.line((x,y-8,x,y+8),fill="#ffe071",width=3)
    labels = {"player":"机器人", "partner":"右侧角色", "ball":"接球目标", "enemy":"普通敌人",
              "golden_enemy":"金色目标候选", "egg_pickup":"卵候选", "egg_buff":"强化候选"}
    for entity in observation.entities:
        x, y, w, h = entity.x*640, entity.y*480, entity.w*640, entity.h*480
        draw.rectangle((x-w/2,y-h/2,x+w/2,y+h/2), outline="#53cde2", width=2)
        draw.text((x-w/2,max(30,y-h/2-16)),f"{labels.get(entity.kind, entity.kind)} #{entity.track_id}",font=small,fill="#79e1ed")
    timing = decision.diagnostics.get("timing", {})
    draw.text((654, 12), f"{observation.game.value.upper()} 视频局面分析", font=font, fill="white")
    lines = ["仅建议 · 实机参数尚未校准", f"源时间：{observation.timestamp:.3f} 秒", f"画面状态：{observation.phase}",
             f"识别 {timing.get('recognition_ms',0):.2f} 毫秒", f"决策 {timing.get('decision_ms',0):.2f} 毫秒", "", "建议："]
    text = decision.reason
    lines.extend(text[i:i+18] for i in range(0,len(text),18))
    state = state_dict(decision.state)
    lines += ["", "按键：" + (" + ".join(state["buttons"]) or "松开"),
              f"左摇杆：{state['left_stick']['x']}", f"右摇杆：{state['right_stick']['x']}"]
    names={"mode":"策略模式", "map_id":"布局编号", "pose_known":"姿态已知"}
    for key in ("mode", "map_id", "pose_known"):
        if key in decision.diagnostics:
            value=str(decision.diagnostics[key])
            lines.append(f"{names[key]}：{value[:26]}")
    for index,line in enumerate(lines[:21]):
        draw.text((654,48+index*20),line,font=small,fill="#f0c877" if index==0 else "#e1e9ef")
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def analyze_video(video, game, output, profile=None, start=0, end=None, stride=1, annotated=False):
    if start < 0 or (end is not None and end <= start) or stride < 1:
        raise ValueError("要求 start>=0、end>start、stride>=1")
    root = Path(__file__).resolve().parents[2]
    profile = load_profile(profile or root/"profiles"/f"{game}.json", game)
    engine = Engine(create_recognizer(game, profile), live=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = OpenCvVideoSource(Path(video))
    trace = TraceWriter(output/"trace.jsonl")
    writer = None
    records, samples = [], []
    started = time.perf_counter()
    try:
        source.open()
        first = source.seek(round(start*source.fps))
        final = min(source.frame_count, round(end*source.fps)) if end is not None else source.frame_count
        frame_index = first
        while frame_index < final:
            decode_start = time.perf_counter()
            frame = source.read()
            decode_ms = (time.perf_counter()-decode_start)*1000
            if frame is None:
                break
            if (frame_index-first) % stride == 0:
                packet = FramePacket(frame,frame_index/source.fps,frame_index+1,source.fps)
                obs, decision = engine.process(packet)
                trace.write(obs,decision)
                timing = decision.diagnostics["timing"]
                records.append({**timing,"decode_ms":decode_ms,"phase_ready":obs.phase=="playing",
                                "nonneutral":bool(decision.state.buttons or decision.state.left_stick.x or decision.state.right_stick.x),
                                "entities":len(obs.entities)})
                if annotated or len(samples)<6:
                    rendered = render_analysis(frame,obs,decision,profile)
                    if len(samples)<6 and len(records) in (3,10,30,60,120,180):
                        name=f"sample-{frame_index:08d}.png"
                        cv2.imencode(".png",rendered)[1].tofile(str(output/name))
                        samples.append({"path":name,"seconds":packet.timestamp,"reason":decision.reason})
                    if annotated:
                        if writer is None:
                            writer=cv2.VideoWriter(str(output/"annotated.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),source.fps/stride,(980,480))
                            if not writer.isOpened():
                                raise RuntimeError("无法创建标注视频")
                        writer.write(rendered)
            frame_index += 1
    finally:
        source.close()
        trace.close()
        if writer:
            writer.release()
    summary={"game":game,"source":str(Path(video).resolve()),"start":start,"end":end,"stride":stride,
             "frames_analyzed":len(records),"wall_seconds":time.perf_counter()-started,
             "phase_ready_frames":sum(r["phase_ready"] for r in records),
             "advice_frames":sum(r["nonneutral"] for r in records),"samples":samples,
             "scope":"reference_video_advice_not_accuracy_or_real_game_score","timing":{}}
    for key in ("recognition_ms","decision_ms","analysis_ms","decode_ms"):
        values=[r[key] for r in records]
        summary["timing"][key]={"median":float(np.median(values)),"p95":float(np.percentile(values,95)),"max":float(max(values))} if values else {}
    summary["timing_note"]="分析计时不含解码、图像绘制、编码、HDMI、采集卡和串口；解码另列。ready/advice帧数不是识别正确率。"
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary
