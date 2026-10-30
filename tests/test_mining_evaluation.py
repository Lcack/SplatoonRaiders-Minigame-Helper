import json

import pytest

from raiders_helper.mining_evaluation import SCOPE, run_evaluation, synthetic_cases


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    output = tmp_path_factory.mktemp("mining_evaluation") / "report.json"
    result = run_evaluation(output)
    assert json.loads(output.read_text(encoding="utf-8")) == result
    return result


def test_model_report_declares_assumptions_and_never_reports_game_score(report):
    assert report["scope"] == SCOPE
    assert "不表示实机得分" in report["notice"]
    assert report["case_count"] >= 6
    assert report["checks_passed"]
    for case in report["cases"]:
        assert case["assumptions"]["motion"]["calibrated"] is False
        assert case["assumptions"]["motion"]["boost_speed"] == 0.4
        assert case["elapsed_ms"] >= 0
        assert isinstance(case["path"], list)
        assert "predicted_added_seconds" in case and "model_value" in case
        assert "score" not in case and "real_score" not in case
        assert case["horizon_complete"] == case["feasible"]
        assert case["checks_passed"], case["case_id"]


def test_grouped_pickups_outperform_single_pickup_control(report):
    cases = {case["case_id"]: case for case in report["cases"]}
    assert cases["grouped_pickups"]["predicted_added_seconds"] >= 3
    assert cases["single_pickup_baseline"]["predicted_added_seconds"] == 1
    assert cases["grouped_pickups"]["model_value"] > cases["single_pickup_baseline"]["model_value"]


def test_s_curve_has_no_shared_x_but_has_a_turning_plan(report):
    case = next(case for case in synthetic_cases() if case.case_id == "s_curve")
    corridors = case.observation.of_kind("corridor")
    left = max(row.x - row.w / 2 for row in corridors)
    right = min(row.x + row.w / 2 for row in corridors)
    assert left > right
    result = next(row for row in report["cases"] if row["case_id"] == "s_curve")
    assert result["feasible"]
    assert max(p["x"] for p in result["path"]) - min(p["x"] for p in result["path"]) >= 0.1


def test_power_changes_rock_reward_but_wall_is_infeasible(report):
    cases = {case["case_id"]: case for case in report["cases"]}
    assert cases["powered_rock"]["feasible"]
    assert cases["powered_rock"]["predicted_added_seconds"] == 1
    assert cases["unpowered_rock"]["feasible"]
    assert cases["unpowered_rock"]["predicted_added_seconds"] == 0
    assert cases["blocked_wall"]["feasible"] is False
    assert cases["blocked_wall"]["path"] == []
