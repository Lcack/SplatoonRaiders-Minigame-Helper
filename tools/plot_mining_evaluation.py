"""Render the synthetic planner report; all labels explicitly say model scope."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(report_path, output):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    chosen = ["s_curve", "grouped_pickups", "powered_rock", "blocked_wall"]
    cases = {case["case_id"]: case for case in report["cases"]}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, case_id in zip(axes.flat, chosen):
        case = cases[case_id]
        entities = case["assumptions"]["observation"]["entities"]
        road = sorted((e for e in entities if e["kind"] == "corridor"), key=lambda e: e["y"])
        ax.fill_betweenx([e["y"] for e in road], [e["x"]-e["w"]/2 for e in road],
                         [e["x"]+e["w"]/2 for e in road], color="#d6eaf8", label="道路包络")
        for e in entities:
            kind = e["kind"]
            if kind in ("wall", "obstacle"):
                color = "#343a40" if kind == "wall" else "#b5651d"
                ax.add_patch(plt.Rectangle((e["x"]-e["w"]/2, e["y"]-e["h"]/2), e["w"], e["h"], color=color))
            elif kind in ("time_pickup", "reward"):
                ax.scatter(e["x"], e["y"], marker="*", s=110, color="#d98d00", zorder=4)
            elif kind == "player":
                ax.scatter(e["x"], e["y"], marker="o", s=55, color="#1e8449", zorder=5)
        path = case["path"]
        if path:
            ax.plot([p["x"] for p in path], [p["depth"] for p in path], "o-", color="#9b2c85", markersize=3,
                    linewidth=2, label="模型规划路线")
        else:
            ax.text(0.5, 0.55, "无完整可行路线\n释放策略输入", ha="center", color="#b03a2e", fontsize=13)
        ax.set_title(f"{case['name']}\n预计加时 {case['predicted_added_seconds']:g} 秒")
        ax.set(xlim=(0, 1), ylim=(0.95, 0), xlabel="横向位置（游戏区比例）", ylabel="前方深度（当前画面比例）")
        ax.grid(alpha=0.15)
    axes[0, 0].legend(loc="lower right", fontsize=9)
    fig.suptitle("MINING 核心规划器：合成模型验证\n全部场景与运动参数为假设，不代表实机得分或最高分", fontsize=15)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制 MINING 合成规划报告")
    parser.add_argument("--report", type=Path, default=Path("artifacts/mining_evaluation.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/mining_evaluation.png"))
    args = parser.parse_args()
    render(args.report, args.output)
    print(args.output)
