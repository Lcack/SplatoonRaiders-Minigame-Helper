import json
import math

import pytest

from raiders_helper.mining_motion import fit_mining_motion


CONFIG={"delay_grid_seconds":[0],"min_samples":80,"min_rounds":3,"min_span_seconds":5,
        "max_holdout_rmse":1e-7,"max_forward_rollout_rmse":1e-7,"event_guard_seconds":.10}


def observed_rows(rounds=3, forward=True, constant_forward=False, change_last=False):
    """Explicit synthetic dynamics, not measured NS parameters."""
    rows=[];dt=.02
    for round_id in range(1,rounds+1):
        x,v,speed=.5,0,.1+.02*round_id
        for i in range(500):
            zr=100<=i<250 or i>=350
            u=(0,.65,-.65,.35,-.35)[(i//25)%5]
            timestamp=round_id*20+i*dt
            metrics={"motion_regime":"normal_clear"}
            if forward:
                metrics.update(forward_speed=speed,forward_speed_observed=True,
                    forward_speed_confidence=.99,forward_speed_units="viewport_heights_per_second",
                    forward_speed_source="background_flow")
            rows.append({"timestamp":timestamp,"round_id":round_id,"live":True,"executed":True,
                "state":{"buttons":["ZR"] if zr else [],"left_stick":{"x":round(u*32767),"y":0},
                         "right_stick":{"x":0,"y":0}},
                "observation":{"game":"mining","timestamp":timestamp,"sequence":i+1,"phase":"playing",
                    "confidence":.95,"metrics":metrics,"entities":[{"kind":"player","x":x,"y":.19,
                        "confidence":.99,"track_id":round_id,"samples":5,"w":.04,"h":.04}]}})
            actual_u=round(u*32767)/32767
            x+=dt*v
            gain=.55 if zr else .9
            drag=6 if zr else 3
            if change_last and round_id==rounds:gain*=2.5
            v+=dt*(-drag*v+gain*actual_u+.02)
            if constant_forward:
                speed+=dt*(.04 if zr else .02)
            else:
                speed+=dt*((.8 if zr else .1)-(1.2 if zr else .5)*speed)
    return rows


def test_separates_zr_steering_response_and_identifies_terminal_candidate_on_independent_round():
    result=fit_mining_motion(observed_rows(),CONFIG)
    assert result["passed"] and result["calibrated"] is False
    off=result["horizontal"]["zr_off"]["selected_model"]
    on=result["horizontal"]["zr_on"]["selected_model"]
    assert off["split"]=="whole_last_round" and on["split"]=="whole_last_round"
    assert off["parameters"]["velocity_feedback_per_second"]==pytest.approx(-3,abs=1e-8)
    assert on["parameters"]["velocity_feedback_per_second"]==pytest.approx(-6,abs=1e-8)
    assert off["parameters"]["input_gain_per_second"]==pytest.approx(.9,abs=1e-8)
    assert on["parameters"]["input_gain_per_second"]==pytest.approx(.55,abs=1e-8)
    for regime,terminal in (("zr_off",.2),("zr_on",2/3)):
        model=result["forward"][regime]
        assert model["selected"]=="terminal_speed_candidate"
        assert model["selected_model"]["parameters"]["terminal_speed_heights_per_second"]==pytest.approx(terminal)
    assert result["infinite_acceleration_established"] is False


def test_horizontal_player_motion_cannot_establish_world_speed():
    result=fit_mining_motion(observed_rows(forward=False),CONFIG)
    assert result["horizontal"]["zr_on"]["passed"]
    assert result["forward_speed_observed"] is False and result["passed"] is False
    assert "world_forward_speed_and_acceleration" in result["unobserved_parameters"]
    assert result["forward"]["zr_on"]["candidates"]==[]


def test_constant_acceleration_on_test_interval_does_not_claim_unbounded_speed():
    result=fit_mining_motion(observed_rows(constant_forward=True),CONFIG)
    assert result["forward"]["zr_on"]["selected"]=="constant_acceleration"
    assert result["forward"]["zr_on"]["selected_model"]["parameters"]["acceleration_heights_per_second_squared"]==pytest.approx(.04)
    assert result["infinite_acceleration_established"] is False


def test_holdout_round_with_changed_dynamics_rejects_near_perfect_training_fit():
    result=fit_mining_motion(observed_rows(change_last=True),CONFIG)
    model=result["horizontal"]["zr_on"]["selected_model"]
    assert model["train_rmse"]<1e-10
    assert model["holdout_rmse"]>.001
    assert result["horizontal"]["zr_on"]["passed"] is False


@pytest.mark.parametrize("field,value",[("live",False),("executed",False)])
def test_replay_or_advice_rows_are_rejected(field,value):
    rows=observed_rows()
    for row in rows:row[field]=value
    result=fit_mining_motion(rows,CONFIG)
    assert result["usable_observations"]==0 and not result["passed"]
    assert result["exclusions"]["not_live_executed"]==len(rows)


def test_planner_speed_assumption_or_score_proxy_is_not_an_observation():
    rows=observed_rows()
    for row in rows:
        metric=row["observation"]["metrics"]
        metric["forward_speed_observed"]=False
        metric["forward_speed_source"]="score"
    result=fit_mining_motion(rows,CONFIG)
    assert not result["forward_speed_observed"]


def test_collision_buff_and_neighbour_frames_are_not_used_as_ordinary_drag():
    rows=observed_rows()
    for index,event in ((50,"wall_collision"),(170,"rock_collision"),(400,"buff_active")):
        rows[index]["observation"]["metrics"][event]=True
    result=fit_mining_motion(rows,CONFIG)
    assert result["exclusions"]["event:wall_collision"]==1
    assert result["exclusions"]["event:rock_collision"]==1
    assert result["exclusions"]["event:buff_active"]==1
    assert result["exclusions"]["event_neighbourhood"]>=15
    assert result["usable_observations"]<len(rows)-3


def test_unknown_event_labels_and_missing_thresholds_never_auto_calibrate():
    rows=observed_rows()
    result=fit_mining_motion(rows,{"delay_grid_seconds":[0]})
    assert not result["passed"] and not result["calibrated"]
    for row in rows:row["observation"]["metrics"].pop("motion_regime")
    result=fit_mining_motion(rows,CONFIG)
    assert result["usable_observations"]==0
    assert result["exclusions"]["normal_regime_not_labelled"]==len(rows)


def test_track_changes_or_capture_gaps_cannot_create_false_acceleration():
    rows=observed_rows()
    for i,row in enumerate(rows):
        row["observation"]["sequence"]=i*10
        row["observation"]["entities"][0]["track_id"]=i
    result=fit_mining_motion(rows,CONFIG)
    assert not result["passed"]
    assert result["horizontal"]["zr_on"]["candidates"]==[]


def test_empty_or_bad_delay_grid_is_rejected():
    assert fit_mining_motion([],CONFIG)["passed"] is False
    with pytest.raises(ValueError):fit_mining_motion([],{"delay_grid_seconds":[math.nan]})


def test_fit_report_is_json_serializable_and_bad_entity_payload_is_rejected():
    rows=observed_rows()
    rows[0]["observation"]["entities"]=None
    result=fit_mining_motion(rows,CONFIG)
    json.dumps(result,allow_nan=False)
    assert result["exclusions"]["invalid_entities"]==1
