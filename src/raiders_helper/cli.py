import argparse
import json
from pathlib import Path
import sys

import cv2

from .capture.models import FramePacket
from .capture.video_source import OpenCvVideoSource
from .demo import demo_packet, demo_profile
from .models import Game
from .runtime import Engine
from .trace import TraceWriter
from .vision import create_recognizer, load_profile


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Raiders 四小游戏采集、识别与控制框架")
    sub = parser.add_subparsers(dest="command", required=True)
    gui=sub.add_parser("gui", help="中文调试界面")
    gui.add_argument("--game",choices=[g.value for g in Game])
    gui.add_argument("--video",type=Path)
    sub.add_parser("controller-debug", help="移植的 EasyCon / Dummy 手柄面板")
    sub.add_parser("ports", help="列出 EasyCon 串口")
    evaluation = sub.add_parser("mining-evaluate", help="MINING 核心规划器的合成模型评估，不连接硬件")
    evaluation.add_argument("--output", type=Path, default=Path("artifacts/mining_evaluation.json"))
    analysis = sub.add_parser("analyze-video", help="三项小游戏逐帧分析、建议与分项耗时（仅离线）")
    analysis.add_argument("video", type=Path)
    analysis.add_argument("--game", choices=["hopping", "juggling", "landing"], required=True)
    analysis.add_argument("--profile", type=Path)
    analysis.add_argument("--start", type=float, default=0)
    analysis.add_argument("--end", type=float)
    analysis.add_argument("--stride", type=int, default=1)
    analysis.add_argument("--output", type=Path, default=Path("artifacts/video_analysis"))
    analysis.add_argument("--annotated", action="store_true", help="另存中文建议标注视频，绘制/编码不计入分析耗时")
    demo = sub.add_parser("demo", help="运行合成画面测试，绝不连接实机")
    demo.add_argument("--game", choices=[g.value for g in Game], default="mining")
    demo.add_argument("--frames", type=int, default=120)
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo.jsonl"))
    replay = sub.add_parser("replay", help="本地视频离线识别并导出 JSONL")
    replay.add_argument("video", type=Path)
    replay.add_argument("--game", choices=[g.value for g in Game], required=True)
    replay.add_argument("--profile", type=Path)
    replay.add_argument("--output", type=Path, default=Path("artifacts/replay.jsonl"))
    replay.add_argument("--frames", type=int, default=0, help="0 表示到视频末尾")
    extract = sub.add_parser("extract", help="从本地实况定间隔抽帧，保留时间索引")
    extract.add_argument("video", type=Path)
    extract.add_argument("--start", type=float, default=0)
    extract.add_argument("--end", type=float, required=True)
    extract.add_argument("--step", type=float, default=1)
    extract.add_argument("--output", type=Path, default=Path("artifacts/frames"))
    args = parser.parse_args(argv)
    try:
        if args.command == "gui":
            from .gui import run_gui
            return run_gui(args.game,args.video)
        if args.command == "analyze-video":
            from .video_analysis import analyze_video
            report = analyze_video(args.video, args.game, args.output, args.profile,
                                   args.start, args.end, args.stride, args.annotated)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.command == "controller-debug":
            from .control.debug_ui import run_debug_ui
            return run_debug_ui()
        if args.command == "ports":
            from .control.easycon import EasyConBackend
            import importlib.util
            if importlib.util.find_spec("serial") is None:
                print("缺少 pyserial；安装：python -m pip install pyserial", file=sys.stderr)
                return 1
            print(json.dumps([{"device": p.device, "description": p.description} for p in EasyConBackend.list_ports()], ensure_ascii=False))
            return 0
        if args.command == "mining-evaluate":
            from .mining_evaluation import run_evaluation
            report = run_evaluation(args.output)
            print(report["notice"])
            for case in report["cases"]:
                print(f"{case['name']}：{'通过' if case['checks_passed'] else '失败'}；"
                      f"预测加时 {case['predicted_added_seconds']:g} 秒；规划 {case['elapsed_ms']:.1f} ms")
            print(f"{report['case_count']} 个合成场景；报告：{args.output}")
            return 0 if report["checks_passed"] else 1
        if args.command == "extract":
            if not 0 <= args.start < args.end or args.step <= 0:
                parser.error("要求 0 <= start < end 且 step > 0")
            video = OpenCvVideoSource(args.video)
            try:
                video.open()
                args.output.mkdir(parents=True, exist_ok=True)
                records, index = [], 0
                while (seconds := args.start + index * args.step) < args.end:
                    frame_index = round(seconds * video.fps)
                    if frame_index >= video.frame_count:
                        break
                    video.seek(frame_index)
                    frame = video.read()
                    if frame is None:
                        break
                    name = f"frame-{frame_index:08d}.png"
                    ok, data = cv2.imencode(".png", frame)
                    if not ok:
                        raise RuntimeError("截图编码失败")
                    (args.output / name).write_bytes(data.tobytes())
                    records.append({"path": name, "frame": frame_index, "seconds": frame_index / video.fps,
                                    "source": str(args.video.resolve())})
                    index += 1
                (args.output / "index.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"已抽取 {len(records)} 帧：{args.output}")
            finally:
                video.close()
            return 0
        if args.frames < 0 or (args.command == "demo" and args.frames == 0):
            parser.error("frames 必须为正数；仅 replay 允许 0")
        profile = demo_profile(args.game) if args.command == "demo" else (load_profile(args.profile, args.game) if args.profile else None)
        engine = Engine(create_recognizer(args.game, profile))
        trace = TraceWriter(args.output)
        video = None
        count = 0
        try:
            if args.command == "replay":
                video = OpenCvVideoSource(args.video)
                video.open()
            while not args.frames or count < args.frames:
                if video:
                    frame = video.read()
                    if frame is None:
                        break
                    packet = FramePacket(frame, count / video.fps, count + 1, video.fps)
                else:
                    packet = demo_packet(args.game, count + 1)
                obs, decision = engine.process(packet)
                trace.write(obs, decision)
                count += 1
        finally:
            trace.close()
            if video:
                video.close()
        print(f"处理 {count} 帧，仅观察，日志：{args.output}")
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
