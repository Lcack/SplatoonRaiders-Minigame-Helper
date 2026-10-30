"""Reproduce six sequential real-video analysis checks; never sends inputs."""
import argparse
import json
import platform
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from raiders_helper.video_analysis import analyze_video


CASES = (
    ("hopping_egg", "hopping", 382, 388, True),
    ("juggling", "juggling", 720, 726, True),
    ("landing", "landing", 983, 989, True),
    ("hopping_normal", "hopping", 590, 596, False),
    ("juggling_dense", "juggling", 870, 876, False),
    ("landing_wind", "landing", 1123, 1129, False),
)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="六段参考实况逐帧分析与分项计时，非正确率或实机验证")
    parser.add_argument("--video", type=Path, default=ROOT / "artifacts/reference_videos/utagaming-l3dMQV1sar8/reference.mp4")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reference_analysis")
    parser.add_argument("--no-annotated", action="store_true")
    args = parser.parse_args()
    reports = []
    for name, game, start, end, annotated in CASES:
        report = analyze_video(args.video, game, args.output / name, start=start, end=end,
                               annotated=annotated and not args.no_annotated)
        reports.append({"case": name, **report})
        print(f"{name}: {report['frames_analyzed']} 帧，分析中位 {report['timing']['analysis_ms']['median']:.2f} ms，"
              f"P95 {report['timing']['analysis_ms']['p95']:.2f} ms", flush=True)
    result = {"scope": "six_short_clips_one_reference_video_not_accuracy_or_control_validation",
              "python": sys.version, "platform": platform.platform(), "opencv": cv2.__version__,
              "opencv_threads": cv2.getNumThreads(), "sequential_runs": True, "cases": reports}
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "report.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
