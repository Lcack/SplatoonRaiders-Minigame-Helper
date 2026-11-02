"""Synthetic protocol/boundary checks, not evidence of calibrated NS physics."""
from dataclasses import replace

import pytest

from raiders_helper.control.buttons import Button, ControllerState
from raiders_helper.mining_identification import MiningPhysicsProbe
from raiders_helper.models import Entity, Game, Observation


def entity(kind, x, y, w=.04, h=.04, **changes):
    return Entity(kind, x, y, w, h, .9, track_id=7, samples=5, **changes)


def scene(now, sequence, *, x=.5, vx=0.0, road=None, objects=(), metrics=None, age=0.0):
    road = road if road is not None else [entity("corridor", .5, y, .94, .001)
                                        for y in (0, .15, .2, .25, .3, .4, .7, 1)]
    return Observation(Game.MINING, now-age, sequence, "playing", .9,
        (entity("player", x, .2, vx=vx), *road, *objects),
        metrics={"hud_anchor_score": .9, "observation_age_seconds": age, **(metrics or {})})


def diagnostics(decision):
    return decision.diagnostics["identification"]


def run(probe, seconds=10.0, step=.02, start=1.0):
    return [(start+i*step, probe.decide(scene(start+i*step, i+1), start+i*step))
            for i in range(round(seconds/step))]


def until_pulse(probe):
    for i in range(40):
        now=1+i*.02
        decision=probe.decide(scene(now, i+1), now)
        if diagnostics(decision)["stage"] == "pulse":
            return now, i+1, decision
    raise AssertionError("expected a bounded pulse on the wide synthetic road")


def test_default_is_disabled_and_never_sends_zr_or_stick():
    probe=MiningPhysicsProbe()
    for _, decision in run(probe, seconds=.5):
        assert decision.state == ControllerState.neutral()
        assert diagnostics(decision)["segment_id"] is None
        assert diagnostics(decision)["calibrated"] is False


def test_target_task_freezes_reference_at_pulse_start_and_preserves_requested_hold():
    probe=MiningPhysicsProbe({"enabled":True,"target_tasks":[
        {"axis":.4,"delta_x":.03,"pulse_seconds":.243}]})
    now,sequence,decision=until_pulse(probe)
    diag=diagnostics(decision)
    assert diag["pulse_seconds"]==.243
    assert diag["target_reference_x"]==.5
    assert diag["target_x"]==pytest.approx(.53)
    later=probe.decide(scene(now+.02,sequence+1,x=.501),now+.02)
    assert diagnostics(later)["target_x"]==pytest.approx(.53)


def test_target_task_rejects_axis_toward_opposite_requested_displacement():
    with pytest.raises(ValueError,match="目标任务"):
        MiningPhysicsProbe({"target_tasks":[{"axis":-.4,"delta_x":.03,"pulse_seconds":.3}]})


@pytest.mark.parametrize("mode", ["normal", "zr"])
def test_protocol_has_stable_segments_and_alternating_duration_direction_groups(mode):
    probe=MiningPhysicsProbe({"enabled": True})
    probe.reset_round(3, mode)
    rows=run(probe, seconds=10.0)
    completed=[]
    grouped={}
    for now, decision in rows:
        info=diagnostics(decision)
        if info["source"] == "probe":
            assert decision.state.buttons == (frozenset({Button.ZR}) if mode == "zr" else frozenset())
            assert info["mode"] == mode and info["round"] == 3
            assert info["segment_validity"] == "running"
            grouped.setdefault(info["segment_id"], []).append(info)
        else:
            assert not decision.state.buttons
        for event in info["events"]:
            if event["type"] == "stage_finished":
                # The caller's sampling grid determines the actual transition.
                assert event["nominal_seconds"] <= event["actual_seconds"]+1e-9
                assert event["actual_seconds"] < event["nominal_seconds"]+.021
            if event["type"] == "completed":
                completed.append((info, event))
                assert decision.state == ControllerState.neutral()
    assert len(completed) >= 6
    assert len({event["segment_id"] for _, event in completed}) == len(completed)
    pulses=[]
    for info, event in completed[:6]:
        stages=grouped[event["segment_id"]]
        assert stages[0]["stage"] == "baseline" and stages[0]["axis"] == 0
        assert stages[-1]["stage"] == "coast" and stages[-1]["axis"] == 0
        pulse=next(row for row in stages if row["stage"] == "pulse")
        pulses.append((pulse["axis"], pulse["pulse_seconds"]))
        reverse=[row for row in stages if row["stage"] == "reverse"]
        assert bool(reverse) == (info["protocol"] == "reverse")
        if reverse:
            assert all(row["axis"] == -pulse["axis"] for row in reverse)
    assert pulses == [(.4,.35),(-.4,.35),(.4,.60),(-.4,.60),(.4,.45),(-.4,.45)]


@pytest.mark.parametrize("corrupt", [
    "unknown", "missing", "ambiguous", "hit", "collision", "ended", "stale",
    "hud", "track", "samples", "duplicate", "clock_gap", "road", "rock", "speed",
])
def test_interruption_aborts_the_entire_segment_and_next_attempt_has_new_baseline(corrupt):
    probe=MiningPhysicsProbe({"enabled": True})
    probe.reset_round(1, "zr")
    now, sequence, prior=until_pulse(probe)
    old_id=diagnostics(prior)["segment_id"]
    now += .02
    obs=scene(now, sequence+1)
    player=obs.entities[0]
    if corrupt == "unknown": obs=replace(obs, phase="unknown")
    elif corrupt == "missing": obs=replace(obs, entities=obs.entities[1:])
    elif corrupt == "ambiguous": obs=replace(obs, entities=(*obs.entities, player))
    elif corrupt == "hit": obs=replace(obs, entities=(*obs.entities, replace(player, kind="player_hit")))
    elif corrupt == "collision": obs=replace(obs, metrics={**obs.metrics, "collision_suspected": True})
    elif corrupt == "ended": obs=replace(obs, metrics={**obs.metrics, "game_ended": True})
    elif corrupt == "stale": obs=replace(obs, timestamp=now-.3)
    elif corrupt == "hud": obs=replace(obs, metrics={**obs.metrics, "hud_anchor_score": .4})
    elif corrupt == "track": obs=replace(obs, entities=(replace(player, track_id=8), *obs.entities[1:]))
    elif corrupt == "samples": obs=replace(obs, entities=(replace(player, samples=1), *obs.entities[1:]))
    elif corrupt == "duplicate": obs=replace(obs, sequence=sequence)
    elif corrupt == "clock_gap": now += .11; obs=scene(now, sequence+1)
    elif corrupt == "road": obs=replace(obs, entities=(player,))
    elif corrupt == "rock": obs=replace(obs, entities=(*obs.entities, entity("obstacle", .5, .25)))
    elif corrupt == "speed": obs=replace(obs, entities=(replace(player, vx=.5), *obs.entities[1:]))
    decision=probe.decide(obs, now)
    info=diagnostics(decision)
    assert decision.state == ControllerState.neutral()
    assert info["segment_id"] == old_id and info["segment_validity"] == "aborted"
    assert info["segment_aborted"] and info["abort_reason"]
    assert [event["type"] for event in info["events"]] == ["aborted"]
    fresh=[]
    for i in range(1, 4):
        stamp=now+i*.02
        fresh.append(probe.decide(scene(stamp, sequence+1+i), stamp))
    started=[diagnostics(row) for row in fresh if diagnostics(row)["source"] == "probe"]
    assert started and started[0]["stage"] == "baseline"
    assert started[0]["segment_id"] != old_id
    assert not any(diagnostics(row)["stage"] == "pulse" for row in fresh)


def test_remote_future_bends_do_not_prevent_local_clear_baseline():
    road=[entity("corridor", x, y, .45, .001) for x,y in
          [(.5,0),(.5,.15),(.5,.2),(.5,.3),(.85,.5),(.85,.7),(.15,.95),( .15,1)]]
    assert max(e.x-e.w/2 for e in road) > min(e.x+e.w/2 for e in road)
    probe=MiningPhysicsProbe({"enabled": True})
    probe.decide(scene(1,1,road=road),1)
    decision=probe.decide(scene(1.02,2,road=road),1.02)
    assert diagnostics(decision)["stage"] == "baseline"


def test_direction_space_is_reserved_before_the_first_pulse():
    road=[entity("corridor", .5, y, .20, .001) for y in (0,.15,.2,.3,.5,1)]
    probe=MiningPhysicsProbe({"enabled": True})
    probe.reset_round(1, "zr")
    probe.decide(scene(1,1,road=road),1)
    decision=probe.decide(scene(1.02,2,road=road),1.02)
    info=diagnostics(decision)
    assert info["source"] == "positioning" and info["segment_id"] is None
    assert not decision.state.buttons
    assert decision.state.left_stick.x < 0  # Reserve room to the right before +.4.
    assert info["reserved_lateral_offsets"] == pytest.approx([0,.4*.4*(.35+.1)])


def test_only_near_hazards_block_start_and_body_height_must_be_covered():
    for rock_y, expected in [(.30,"positioning"),(.8,"probe")]:
        probe=MiningPhysicsProbe({"enabled": True})
        rock=entity("rock", .5, rock_y)
        probe.decide(scene(1,1,objects=[rock]),1)
        decision=probe.decide(scene(1.02,2,objects=[rock]),1.02)
        assert diagnostics(decision)["source"] == expected
    probe=MiningPhysicsProbe({"enabled": True})
    short=[entity("corridor", .5, y, .9, .001) for y in (.2,.3,.5,1)]
    assert probe.decide(scene(1,1,road=short),1).state == ControllerState.neutral()


def test_ten_hz_steps_and_observed_point_one_four_second_age_are_accepted():
    probe=MiningPhysicsProbe({"enabled": True})
    decisions=[]
    for i in range(20):
        now=1+i*.1
        decisions.append(probe.decide(scene(now,i*6+1,age=.14),now))
    assert any(diagnostics(row)["stage"] == "pulse" for row in decisions)
    assert any(diagnostics(row)["segment_validity"] == "completed" for row in decisions)
    assert not any(diagnostics(row)["segment_aborted"] for row in decisions)


def test_waits_for_small_velocity_and_checks_it_again_after_baseline():
    probe=MiningPhysicsProbe({"enabled": True})
    probe.decide(scene(1,1,vx=.04),1)
    decision=probe.decide(scene(1.02,2,vx=.04),1.02)
    assert diagnostics(decision)["source"] == "positioning"
    assert decision.state.left_stick.x < 0
    decision=probe.decide(scene(1.04,3,vx=.02),1.04)
    assert diagnostics(decision)["stage"] == "baseline"
    for i in range(1,16):
        now=1.04+i*.02
        decision=probe.decide(scene(now,3+i,vx=.04),now)
    assert diagnostics(decision)["segment_aborted"]
    assert "横向漂移" in diagnostics(decision)["abort_reason"]


def test_reset_round_changes_mode_identity_and_discards_the_old_partial_segment():
    probe=MiningPhysicsProbe({"enabled": True})
    now,sequence,old=until_pulse(probe)
    probe.reset_round(2,"zr")
    probe.decide(scene(now+.02,1),now+.02)
    new=probe.decide(scene(now+.04,2),now+.04)
    assert diagnostics(new)["stage"] == "baseline"
    assert diagnostics(new)["segment_id"] != diagnostics(old)["segment_id"]
    assert new.state.buttons == frozenset({Button.ZR})


@pytest.mark.parametrize("config", [
    {"axis": True}, {"axis": .9}, {"pulse_seconds": []}, {"pulse_seconds": [.01]},
    {"pulse_seconds": [float("nan")]}, {"max_step_seconds": 0}, {"min_track_samples": 1},
])
def test_invalid_experiment_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        MiningPhysicsProbe(config)
