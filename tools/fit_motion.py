"""Fit actual live action-response rows; never opens a controller."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from raiders_helper.motion_calibration import fit_motion


def main():
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description="实机动作响应标定拟合；拒绝回放/未执行数据")
    parser.add_argument("rows",type=Path)
    parser.add_argument("--config",type=Path,help="fit配置或含calibration/fit的session JSON")
    parser.add_argument("--output",type=Path,default=Path("artifacts/motion_fit.json"))
    args=parser.parse_args()
    rows=[json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    config=json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    config=config.get("calibration",config)
    report=fit_motion(rows,config)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(report["reason"])
    print(f"报告：{args.output}；完整实机配置 calibrated 仍为 false。")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
