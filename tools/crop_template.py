"""Crop a template in exactly the same canonical coordinates as the detector."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from raiders_helper.vision import crop_roi

parser = argparse.ArgumentParser(description="从原始截图裁剪识别模板")
parser.add_argument("image", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--viewport", nargs=4, type=float, default=[0, 0, 1, 1])
parser.add_argument("--size", nargs=2, type=int, default=[640, 480])
parser.add_argument("--roi", nargs=4, type=float, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    frame = cv2.imdecode(np.frombuffer(args.image.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        parser.error("无法解码图片")
    canonical = cv2.resize(crop_roi(frame, args.viewport), tuple(args.size))
    template = crop_roi(canonical, args.roi)
    ok, encoded = cv2.imencode(".png", template)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded.tobytes())
    print(f"已保存 {template.shape[1]}×{template.shape[0]} 模板：{args.output}")
