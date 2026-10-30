"""Reproducible synthetic checks of the MINING planning model.

These cases are declared geometric and motion assumptions, not reconstructed
game physics or recorded scores. They never connect to a controller. Runtime
measurements vary between executions; scenario inputs and search are fixed.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from time import perf_counter

from .mining_planner import MotionModel, SearchSettings, plan_mining
from .models import Entity, Game, Observation


SCOPE = "synthetic_model_planning_checks"
ASSUMPTION_NOTICE = (
    "全部道路、目标和运动参数都是显式声明的合成假设；报告仅检查模型内路线和奖励，"
    "不表示实机得分、真实游戏物理参数或规则内最高分。"
)


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    name: str
    observation: Observation
    model: MotionModel
    settings: SearchSettings
    expected_feasible: bool
    minimum_added_seconds: float = 0.0
    maximum_added_seconds: float | None = None
    minimum_lateral_span: float = 0.0
    notes: str = ""


def _entity(kind, x, y, w=0.025, h=0.025, track_id=-1):
    return Entity(kind, x, y, w, h, 1.0, track_id=track_id, samples=3)


def _road(rows):
    return tuple(_entity("corridor", x, y, width, 0.001, 1000 + i)
                 for i, (y, x, width) in enumerate(rows))


def _observation(road, *objects, player_x=0.5, powered=False):
    entities = (_entity("player", player_x, 0.2, 0.03, 0.03, 0), *road, *objects)
    metrics = {"time_remaining": 40.0}
    if powered:
        entities += (_entity("egg_buff", 0.05, 0.05, track_id=900),)
        metrics["buff_remaining_lower_bound_seconds"] = 2.0
    return Observation(Game.MINING, 0.0, 1, "playing", 1.0, entities,
                       ("合成假设场景；非实机观测",), metrics)


def synthetic_cases():
    """Return independent immutable scene descriptions with explicit assumptions."""
    model = MotionModel(
        forward_speed=0.12,
        accelerated_speed=0.2,
        lateral_speed=0.5,
        latency_seconds=0.04,
        uncertainty_x=0.003,
        uncertainty_y=0.003,
        calibrated=False,
        egg_buff_lower_bound_seconds=0.0,
        boost_speed=0.4,
    )
    settings = SearchSettings(
        horizon_seconds=1.6,
        step_seconds=0.1,
        beam_width=24,
        wall_margin=0.015,
        steering_penalty=0.002,
        time_value=0.3,
    )
    wide = _road(((0.0, 0.5, 0.9), (1.0, 0.5, 0.9)))
    # At y=.2 and y=.32 the intervals do not overlap, so a single x cannot
    # describe this path. A sequence of laterally reachable points can.
    bend = _road(((0.0, 0.3, 0.28), (0.2, 0.3, 0.28),
                  (0.32, 0.6, 0.28), (0.44, 0.3, 0.28), (1.0, 0.3, 0.28)))
    group = tuple(_entity("time_pickup", 0.25, y, 0.018, 0.012, 20 + i)
                  for i, y in enumerate((0.32, 0.33, 0.34)))
    single = _entity("time_pickup", 0.57, 0.32, 0.018, 0.012, 23)
    rock = _entity("obstacle", 0.5, 0.35, 0.16, 0.035, 30)
    barrier = _entity("wall", 0.5, 0.32, 1.0, 0.035, 40)
    return (
        EvaluationCase("straight", "直路", _observation(wide), model, settings, True,
                       maximum_added_seconds=0.0, notes="无目标直路，检验前进路线。"),
        EvaluationCase("s_curve", "无全局横向交集的 S 弯", _observation(bend, player_x=0.3),
                       model, settings, True, maximum_added_seconds=0.0,
                       minimum_lateral_span=0.1, notes="需连续转向；不能要求所有道路层共享同一 x。"),
        EvaluationCase("grouped_pickups", "成组加时与单颗近星", _observation(wide, *group, single),
                       model, settings, True, minimum_added_seconds=3.0,
                       notes="三颗左侧加时物与一颗更近的右侧加时物竞争路线。"),
        EvaluationCase("single_pickup_baseline", "单颗加时基准", _observation(wide, single),
                       model, settings, True, minimum_added_seconds=1.0, maximum_added_seconds=1.0,
                       notes="使用同一运动与搜索参数，作为成组奖励场景的对照。"),
        EvaluationCase("powered_rock", "已知有效强化内的石头", _observation(wide, rock, powered=True),
                       model, settings, True, minimum_added_seconds=1.0, maximum_added_seconds=1.0,
                       notes="强化剩余下界假设为 2 秒，强化速度假设为每秒 0.4 游戏区高度。"),
        EvaluationCase("unpowered_rock", "无强化时避石", _observation(wide, rock), model, settings,
                       True, maximum_added_seconds=0.0, minimum_lateral_span=0.09,
                       notes="与强化场景使用相同石头；路线必须绕行且无碎石加时。"),
        EvaluationCase("blocked_wall", "前方全宽墙堵住路线", _observation(wide, barrier),
                       model, settings, False, maximum_added_seconds=0.0,
                       notes="全宽墙位于玩家前方；虽可短暂前进，但完整规划时域无解，应报告不可行。"),
    )


def evaluate_case(case):
    started = perf_counter()
    plan = plan_mining(case.observation, case.model, case.settings)
    elapsed_ms = (perf_counter() - started) * 1000
    feasible = len(plan.points) > 0
    predicted_seconds = plan.points[-1].seconds if feasible else 0.0
    requested_seconds = case.model.latency_seconds + case.settings.horizon_seconds
    horizon_complete = feasible and predicted_seconds + 1e-9 >= requested_seconds
    lateral_span = (max(p.x for p in plan.points) - min(p.x for p in plan.points)) if feasible else 0.0
    checks = {
        "expected_feasibility": feasible == case.expected_feasible,
        "minimum_time_gain": plan.gained_seconds + 1e-9 >= case.minimum_added_seconds,
        "finite_model_value": math.isfinite(plan.value),
        "minimum_lateral_span": lateral_span + 1e-9 >= case.minimum_lateral_span,
        "expected_full_horizon": horizon_complete if case.expected_feasible else not feasible,
    }
    if case.maximum_added_seconds is not None:
        checks["maximum_time_gain"] = plan.gained_seconds <= case.maximum_added_seconds + 1e-9
    return {
        "case_id": case.case_id,
        "name": case.name,
        "scope": SCOPE,
        "notes": case.notes,
        "assumptions": {
            "motion": asdict(case.model),
            "search": asdict(case.settings),
            "observation": asdict(case.observation),
        },
        "path": [asdict(point) for point in plan.points],
        "predicted_added_seconds": plan.gained_seconds,
        "model_value": plan.value,
        "elapsed_ms": elapsed_ms,
        "feasible": feasible,
        "predicted_seconds": predicted_seconds,
        "requested_seconds": requested_seconds,
        "horizon_complete": horizon_complete,
        "collected": list(plan.collected),
        "expanded_nodes": plan.expanded_nodes,
        "reason": plan.reason,
        "checks": checks,
        "checks_passed": all(checks.values()),
    }


def run_evaluation(output: Path):
    """Write a JSON model-check report and return its data, with no hardware I/O."""
    cases = [evaluate_case(case) for case in synthetic_cases()]
    report = {
        "schema_version": 1,
        "scope": SCOPE,
        "notice": ASSUMPTION_NOTICE,
        "value_definition": (
            "预测前进深度 + time_value × 已收集的唯一目标加时 − 转向变化惩罚；"
            "model_value 的单位和权重仅属于此合成模型，不是真实 SCORE。"
        ),
        "reproducibility": "输入和搜索顺序固定；elapsed_ms 是本机计时，重复执行会有差异。",
        "case_count": len(cases),
        "checks_passed": all(case["checks_passed"] for case in cases),
        "cases": cases,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    output.write_text(serialized, encoding="utf-8")
    # Match the JSON-native report on disk (dataclass tuples become lists).
    return json.loads(serialized)
