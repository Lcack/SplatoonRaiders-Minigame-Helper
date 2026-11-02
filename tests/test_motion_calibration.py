from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest

from raiders_helper.control.buttons import Button, ControllerState, StickVector
from raiders_helper.models import Entity, Game, Observation
from raiders_helper.motion_calibration import CalibrationProbe, DEFAULT_PROBE_CONFIG, fit_motion
from raiders_helper.trace import state_dict


def scene(game="hopping", x=.5, **kwargs):
    actors=(Entity("player",x,.8,.06,.08,.9,1,samples=5),)
    if game == "juggling":
        actors=(replace(actors[0],x=.25),Entity("partner",.75,.85,.04,.06,.9,2,samples=5))
    return Observation(Game(game),1,1,"playing",.82,actors,metrics=kwargs)


def test_probe_requires_separate_opt_in_and_reliable_continuous_actor():
    assert CalibrationProbe("hopping").decide(scene(),0).state == ControllerState.neutral()
    probe=CalibrationProbe("hopping",{"enabled":True})
    assert probe.decide(scene(),0).state == ControllerState.neutral()
    assert probe.decide(scene(),.05).state == ControllerState.neutral()
    unreliable=replace(scene(),entities=(replace(scene().entities[0],samples=1),))
    assert probe.decide(unreliable,.08).state == ControllerState.neutral()
    assert probe.decide(scene(),.1).state == ControllerState.neutral()
    assert probe.decide(replace(scene(),phase="unknown"),.12).state == ControllerState.neutral()


def sample_probe(probe, obs, stop=1.5, dt=.01):
    return [(i*dt,probe.decide(obs,i*dt)) for i in range(round(stop/dt))]


def test_probe_has_neutral_between_opposite_short_pulses():
    decisions=sample_probe(CalibrationProbe("hopping",{"enabled":True}),scene(),1.4)
    signs=[(t,math.copysign(1,d.state.left_stick.x)) for t,d in decisions if d.state.left_stick.x]
    assert {sign for _,sign in signs} == {-1,1}
    first_negative=min(t for t,sign in signs if sign<0)
    last_positive=max(t for t,sign in signs if sign>0 and t<first_negative)
    assert first_negative-last_positive >= .29
    assert not any(d.state.buttons for _,d in decisions)


def test_wall_margin_cancels_outward_probe_and_missing_frame_pauses():
    probe=CalibrationProbe("hopping",{"enabled":True})
    decisions=sample_probe(probe,scene(x=.96),.47)
    assert all(d.state == ControllerState.neutral() for _,d in decisions)
    assert any("边界" in d.reason for _,d in decisions)
    elapsed=probe.elapsed
    assert probe.decide(scene(),5).state == ControllerState.neutral()
    assert probe.elapsed == elapsed


def test_juggling_roles_are_excited_separately_and_require_both_tracks():
    probe=CalibrationProbe("juggling",{"enabled":True})
    decisions=sample_probe(probe,scene("juggling"),4.0)
    assert any(d.state.left_stick.x for _,d in decisions)
    assert any(d.state.right_stick.x for _,d in decisions)
    assert not any(d.state.left_stick.x and d.state.right_stick.x for _,d in decisions)
    missing=replace(scene("juggling"),entities=scene("juggling").entities[:1])
    assert probe.decide(missing,4.01).state == ControllerState.neutral()


def test_landing_jets_need_explicit_opt_in_and_known_pose():
    obs=scene("landing",pose_known=True,pose_confidence=.9)
    assert all(d.state == ControllerState.neutral() for _,d in
               sample_probe(CalibrationProbe("landing",{"enabled":True}),obs))
    probe=CalibrationProbe("landing",{"enabled":True,"enable_jets":True})
    assert any(d.state.buttons for _,d in sample_probe(probe,obs))
    probe.reset_round(2)
    assert all(d.state == ControllerState.neutral() for _,d in sample_probe(probe,scene("landing")))


def test_probe_default_config_is_json_serializable_and_does_not_claim_calibration():
    data=json.loads(json.dumps(DEFAULT_PROBE_CONFIG))
    assert data["enabled"] is False and data["enable_jump"] is False and data["enable_jets"] is False


def synthetic_rows(game="hopping", rounds=4, frames=400, delay_frames=3, holdout_changed=False,
                   right_excited=True):
    """Known causal discrete system. x_k integrates v_k; u is actual sent axis."""
    rows=[]
    pattern=np.repeat([0,.65,0,-.65,0,.35,0,-.35],10)
    pattern=np.rint(pattern*32767)/32767
    for round_id in range(1,rounds+1):
        base=(round_id-1)*20
        x,v=.25 if game=="juggling" else .5,0.0
        rx,rv=.75,0.0
        for k in range(frames):
            u=float(pattern[k%len(pattern)])
            ru=float(pattern[(k+20)%len(pattern)]) if right_excited else 0
            previous=k-1-delay_frames
            delayed=float(pattern[previous%len(pattern)]) if previous>=0 else 0
            delayed_right=float(pattern[(previous+20)%len(pattern)]) if previous>=0 and right_excited else 0
            alpha,beta=(.55,.015) if holdout_changed and round_id==rounds else (.8,.08)
            v=alpha*v+beta*delayed
            rv=.7*rv+.08*delayed_right
            x+=v*.02;rx+=rv*.02
            entities=[Entity("player",x,.8,.05,.06,.95,1,samples=5)]
            if game=="juggling":entities.append(Entity("partner",rx,.85,.04,.06,.95,2,samples=5))
            obs=Observation(Game(game),base+k*.02,k+1,"playing",.82,tuple(entities))
            state=ControllerState(left_stick=StickVector(round(u*32767),0),
                                  right_stick=StickVector(round(ru*32767),0))
            rows.append({"timestamp":base+k*.02+.003,"round_id":round_id,"live":True,"executed":True,
                         "state":state_dict(state),"observation":asdict(obs)})
    return rows


FIT={"delay_grid_seconds":[0,.02,.04,.06,.08,.1],"max_holdout_rmse":.002,
     "max_normalized_holdout_rmse":.05}


def test_known_delayed_system_recovers_response_and_uses_whole_round_holdout():
    report=fit_motion(synthetic_rows(),FIT)
    model=report["actors"]["player"]
    assert report["passed"] and report["calibrated"] is False
    assert model["effective_delay_seconds"] == pytest.approx(.06)
    assert model["discrete_at_median_interval"]["alpha"] == pytest.approx(.8,abs=1e-7)
    assert model["discrete_at_median_interval"]["beta"] == pytest.approx(.08,abs=1e-7)
    assert model["split"] == "whole_rounds" and model["holdout_count"]>0
    assert model["holdout_rmse"] < 1e-9


def test_changed_unseen_round_rejects_despite_perfect_training_error():
    report=fit_motion(synthetic_rows(holdout_changed=True),FIT)
    model=report["actors"]["player"]
    assert model["train_rmse"] < 1e-9
    assert model["holdout_rmse"] > FIT["max_holdout_rmse"]
    assert not model["passed"] and not report["passed"]


def test_single_round_can_be_reported_with_purged_time_block_validation():
    report=fit_motion(synthetic_rows(rounds=1),{**FIT,"min_rounds":1})
    assert report["actors"]["player"]["split"] == "purged_time_blocks"
    assert report["passed"]


def test_missing_error_thresholds_never_auto_pass_even_perfect_system():
    report=fit_motion(synthetic_rows(),{"delay_grid_seconds":[0,.02,.04,.06,.08,.1]})
    assert report["actors"]["player"]["holdout_rmse"] < 1e-9
    assert report["passed"] is False
    assert report["actors"]["player"]["checks"]["holdout_threshold_configured"] is False


@pytest.mark.parametrize("key",["live","executed"])
def test_replay_or_unexecuted_advice_is_rejected(key):
    rows=synthetic_rows()
    for row in rows:row[key]=False
    report=fit_motion(rows,FIT)
    assert not report["passed"] and report["actors"] == {}


@pytest.mark.parametrize("corrupt",["sequence","track","occlusion"])
def test_skipped_frames_changed_ids_and_occlusion_cannot_form_velocity_pairs(corrupt):
    rows=synthetic_rows()
    for i,row in enumerate(rows):
        if corrupt=="sequence":row["observation"]["sequence"]*=3
        elif corrupt=="track":row["observation"]["entities"][0]["track_id"]=i
        else:row["observation"]["entities"]=[]
    report=fit_motion(rows,FIT)
    assert not report["passed"] and report["actors"]["player"]["sample_count"]==0


def test_constant_input_without_excitation_is_not_identifiable():
    rows=synthetic_rows()
    for row in rows:row["state"]["left_stick"]["x"]=0
    report=fit_motion(rows,FIT)
    assert not report["passed"]
    assert "秩不足" in report["actors"]["player"]["reasons"][0]


def test_juggling_requires_both_actor_models_to_pass():
    both=fit_motion(synthetic_rows("juggling"),FIT)
    assert both["passed"] and set(both["actors"])=={"player","partner"}
    one=fit_motion(synthetic_rows("juggling",right_excited=False),FIT)
    assert one["actors"]["player"]["passed"]
    assert not one["actors"]["partner"]["passed"] and not one["passed"]


def test_landing_cannot_be_declared_complete_by_horizontal_model():
    report=fit_motion(synthetic_rows("landing"),FIT)
    assert not report["passed"] and report["actors"] == {}
    assert report["calibrated"] is False and "耦合" in report["reason"]


def test_stale_observation_timestamps_and_button_rows_are_rejected():
    rows=synthetic_rows()
    for i,row in enumerate(rows):
        if i%2:row["timestamp"]+=.5
        else:row["state"]["buttons"]=["B"]
    assert not fit_motion(rows,FIT)["passed"]


def test_pixel_scale_position_jitter_does_not_masquerade_as_validated_dynamics():
    rows=synthetic_rows()
    rng=np.random.default_rng(51)
    for row in rows:
        # About 0.64 canonical pixels: differencing can materially amplify it.
        row["observation"]["entities"][0]["x"]+=float(rng.normal(0,.001))
    report=fit_motion(rows,FIT)
    assert report["actors"]["player"]["holdout_rmse"] > FIT["max_holdout_rmse"]
    assert not report["passed"]
