"""Clock-controlled execution tests; no controller or real sleeps needed."""
from pathlib import Path
import sys

import pytest

from raiders_helper.control.buttons import Button, ControllerState, StickVector
from raiders_helper.mining_mixed_guard import MixedDeadlineGuard
from raiders_helper.models import Decision, Game, Observation

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from mining_supervisor import LatestLease

PARAMETERS = {"mixed_deadline_guard": True, "effective_delay_seconds": .125}


def observation(capture=9.92, sequence=1, *, phase="playing", reports=None, gap=False):
    return Observation(Game.MINING, capture, sequence, phase, .9, metrics={
        "mining_input_history_source": "guard_submission_after",
        "mining_input_history_observed_until": capture + .10,
        "mining_input_history_gap_detected": gap,
        "mining_input_history": reports if reports is not None else [report(9., True)],
    })


def report(after, zr):
    return {"after": after, "state": {"buttons": ["ZR"] if zr else [],
            "left_stick": {"x": 0, "y": 0}, "right_stick": {"x": 0, "y": 0}}}


def decision(mode="accelerated_then_normal", seconds=.2, *, normal=False, verified=False, axis=-.65):
    boosting = mode.startswith("accelerated")
    state = ControllerState(frozenset({Button.ZR}) if boosting else frozenset(), StickVector(round(axis * 32767), 0))
    return Decision(state, diagnostics={"mode": mode, "mixed_zr_prefix_seconds": seconds if mode == "accelerated_then_normal" else None,
        "route_status": "complete", "mixed_guard": {"enabled": True,
        "normal_replanned": normal, "full_fast_rearm_verified": verified}})


def start(guard):
    obs = observation();guard.attach(obs, now=10.)
    state, deadline, info = guard.apply(obs, decision(), now=10.)
    assert Button.ZR in state.buttons and deadline == pytest.approx(10.2)
    return deadline


def brake(guard):
    start(guard)
    obs = observation(10.25, 2)
    guard.attach(obs, now=10.32)
    normal = decision("careful_turn", normal=True, axis=.10)
    state, deadline, info = guard.apply(obs, normal, now=10.33)
    assert state == normal.state and deadline is None
    return obs


def test_repeated_mixed_never_extends_first_deadline():
    guard = MixedDeadlineGuard(PARAMETERS);start(guard)
    for i, now in enumerate([10.04, 10.09, 10.15], 2):
        obs = observation(now - .08, i);guard.attach(obs, now=now)
        state, deadline, info = guard.apply(obs, decision(), now=now)
        assert deadline == pytest.approx(10.2)
        assert info["first_release_at"] == pytest.approx(10.2)


def test_more_urgent_mixed_can_only_shorten_release():
    guard = MixedDeadlineGuard(PARAMETERS);start(guard)
    obs = observation(10.02, 2);guard.attach(obs, now=10.1)
    _, deadline, info = guard.apply(obs, decision(seconds=.025), now=10.1)
    assert deadline == pytest.approx(10.125)
    assert info["first_release_at"] == pytest.approx(10.2)


def test_deadline_crossed_while_policy_computes_vetoes_entire_old_zr_axis():
    guard = MixedDeadlineGuard(PARAMETERS);start(guard)
    obs = observation(10.10, 2);guard.attach(obs, now=10.18)
    assert obs.metrics["mining_mixed_guard"]["phase"] == "scheduled"
    old_zr_decision = decision(axis=-.8)
    state, deadline, info = guard.apply(obs, old_zr_decision, now=10.21)
    assert state == ControllerState.neutral() and deadline is None
    assert info["actual_control_veto_reason"] == "deadline_crossed_during_policy_compute"
    assert old_zr_decision.state.left_stick.x != 0 and Button.ZR in old_zr_decision.state.buttons


def test_normal_axis_must_come_from_a_new_normal_decision():
    guard = MixedDeadlineGuard(PARAMETERS);brake(guard)
    assert guard.phase == "braking" and guard.normal_decision_seen
    obs = observation(10.4, 3);guard.attach(obs, now=10.48)
    zr = decision(axis=-.65)
    state, _, _ = guard.apply(obs, zr, now=10.49)
    assert state == ControllerState.neutral()
    normal = decision("careful_turn", normal=True, axis=.25)
    state, _, _ = guard.apply(obs, normal, now=10.49)
    assert state == normal.state and state.left_stick.x > 0


def test_release_receipt_must_reach_capture_clock_before_rearm():
    guard = MixedDeadlineGuard(PARAMETERS);brake(guard)
    reports = [report(9., True), report(10.3, False)]
    early = observation(10.4, 3, reports=reports);guard.attach(early, now=10.48)
    assert not early.metrics["mining_mixed_guard"]["allow_full_fast_rearm"]
    state, _, _ = guard.apply(early, decision("accelerated", verified=True), now=10.49)
    assert state == ControllerState.neutral()
    visible = observation(10.45, 4, reports=reports);guard.attach(visible, now=10.53)
    assert visible.metrics["mining_mixed_guard"]["allow_full_fast_rearm"]
    state, deadline, info = guard.apply(visible, decision("accelerated", verified=True), now=10.54)
    assert Button.ZR in state.buttons and deadline is None
    assert info["phase_after_execution"] == "eligible"


def test_unobserved_inflight_zr_after_visible_neutral_prevents_rearm():
    guard = MixedDeadlineGuard(PARAMETERS);brake(guard)
    obs = observation(10.45, 3, reports=[report(10.3, False), report(10.4, True)])
    guard.attach(obs, now=10.53)
    assert not obs.metrics["mining_mixed_guard"]["allow_full_fast_rearm"]


def test_normal_refresh_receipts_need_not_all_be_visible_to_confirm_held_normal():
    guard = MixedDeadlineGuard(PARAMETERS);brake(guard)
    obs = observation(10.45, 3, reports=[report(10.3, False), report(10.4, False), report(10.5, False)])
    guard.attach(obs, now=10.53)
    assert obs.metrics["mining_mixed_guard"]["allow_full_fast_rearm"]
    assert obs.metrics["mining_mixed_guard"]["release_backend_receipt_after"] == 10.3


def test_already_held_neutral_does_not_deadlock_waiting_for_duplicate_release_receipt():
    guard = MixedDeadlineGuard(PARAMETERS);brake(guard)
    obs = observation(10.45, 3, reports=[report(10., False)])
    guard.attach(obs, now=10.53)
    assert obs.metrics["mining_mixed_guard"]["allow_full_fast_rearm"]


@pytest.mark.parametrize("change", ["no_normal", "no_receipt", "history_gap", "advice_source", "missing_history", "newer_zr"])
def test_rearm_requires_both_fresh_normal_replan_and_actual_release_evidence(change):
    guard = MixedDeadlineGuard(PARAMETERS);brake(guard)
    obs = observation(10.45, 3, reports=[report(10.3, False)])
    if change == "no_normal":guard.normal_decision_seen = False
    if change == "no_receipt":obs.metrics["mining_input_history"] = [report(10.3, True)]
    if change == "history_gap":obs.metrics["mining_input_history_gap_detected"] = True
    if change == "advice_source":obs.metrics["mining_input_history_source"] = "policy_advice"
    if change == "missing_history":obs.metrics["mining_input_history"] = []
    if change == "newer_zr":obs.metrics["mining_input_history"] += [report(10.31, True)]
    guard.attach(obs, now=10.53)
    state, _, _ = guard.apply(obs, decision("accelerated", verified=True), now=10.54)
    assert state == ControllerState.neutral() and guard.phase == "braking"


@pytest.mark.parametrize("mode,verified,status", [("accelerated_then_normal", True, "complete"),
    ("accelerated", False, "complete"), ("accelerated", True, "partial")])
def test_rearm_never_accepts_mixed_or_partial_or_unverified_fast(mode, verified, status):
    guard = MixedDeadlineGuard(PARAMETERS);brake(guard)
    obs = observation(10.45, 3, reports=[report(10.3, False)]);guard.attach(obs, now=10.53)
    action = decision(mode, verified=verified);action.diagnostics["route_status"] = status
    state, _, _ = guard.apply(obs, action, now=10.54)
    assert state == ControllerState.neutral() and guard.phase == "braking"


def test_pure_fast_during_scheduled_cannot_cancel_fixed_release():
    guard = MixedDeadlineGuard(PARAMETERS);start(guard)
    obs = observation(10.02, 2);guard.attach(obs, now=10.1)
    state, _, info = guard.apply(obs, decision("accelerated", verified=True), now=10.11)
    assert state == ControllerState.neutral()
    assert info["actual_control_veto_reason"] == "scheduled_release_cannot_be_cancelled_by_inflight_fast"


def test_unknown_flash_does_not_clear_latch_for_another_mixed():
    guard = MixedDeadlineGuard(PARAMETERS);start(guard)
    missing = observation(10.02, 2, phase="unknown");guard.attach(missing, now=10.1)
    guard.apply(missing, decision(), now=10.11)
    reacquired = observation(10.2, 3);guard.attach(reacquired, now=10.28)
    assert reacquired.metrics["mining_mixed_guard"]["phase"] == "braking"
    state, _, _ = guard.apply(reacquired, decision(), now=10.29)
    assert state == ControllerState.neutral()


def test_stale_mixed_cannot_arm_deadline_or_claim_normal_replan():
    guard = MixedDeadlineGuard(PARAMETERS)
    obs = observation();guard.attach(obs, now=10.)
    state, deadline, _ = guard.apply(obs, decision(), now=10.3, fresh=False)
    assert state == ControllerState.neutral() and deadline is None and guard.phase == "eligible"
    start(guard)
    obs = observation(10.22, 2);guard.attach(obs, now=10.4)
    guard.apply(obs, decision("careful_turn", normal=True), now=10.5, fresh=False)
    assert not guard.normal_decision_seen


def test_disabled_guard_preserves_old_behavior():
    guard = MixedDeadlineGuard({});obs = observation();guard.attach(obs, now=10.)
    action = decision();state, deadline, _ = guard.apply(obs, action, now=10.)
    assert state == action.state and deadline is None and "mining_mixed_guard" not in obs.metrics


class FakeClock:
    def __init__(self):self.now = 10.
    def __call__(self):return self.now


class FakeClient:
    def __init__(self, clock):self.clock = clock;self.commands = []
    def command(self, payload):self.commands.append((self.clock(), payload))


def test_lease_releases_at_deadline_without_new_capture_or_policy_callback():
    clock = FakeClock();client = FakeClient(clock);lease = LatestLease(client, clock=clock)
    lease.update(decision().state, release_at=10.2)
    lease.tick();clock.now = 10.199;lease.tick();clock.now = 10.2;lease.tick()
    assert client.commands[-1] == (10.2, {"command": "release"})
    assert all("ZR" in payload["buttons"] for _, payload in client.commands[:-1])


def test_lease_late_update_with_old_deadline_cannot_resurrect_zr():
    clock = FakeClock();client = FakeClient(clock);lease = LatestLease(client, clock=clock)
    lease.update(decision().state, release_at=10.2);lease.tick()
    clock.now = 10.2;lease.tick()
    clock.now = 10.21;lease.update(decision(axis=.8).state, release_at=10.2);lease.tick()
    assert client.commands[-1][1]["command"] == "release"
    assert not any(t >= 10.2 and "ZR" in p.get("buttons", []) for t, p in client.commands)


def test_end_to_end_fakeclock_policy_crossing_deadline_never_issues_old_zr_again():
    clock = FakeClock();client = FakeClient(clock);lease = LatestLease(client, clock=clock)
    guard = MixedDeadlineGuard(PARAMETERS)
    obs = observation();guard.attach(obs, now=clock())
    state, deadline, _ = guard.apply(obs, decision(), now=clock())
    lease.update(state, release_at=deadline);lease.tick()
    clock.now = 10.18;obs = observation(10.10, 2);guard.attach(obs, now=clock())
    clock.now = 10.2;lease.tick()  # Independent worker expires during policy work.
    clock.now = 10.22;state, deadline, info = guard.apply(obs, decision(axis=.9), now=clock())
    lease.update(state, release_at=deadline);lease.tick()
    assert info["actual_control_veto_reason"] == "deadline_crossed_during_policy_compute"
    assert not any(t >= 10.2 and "ZR" in p.get("buttons", []) for t, p in client.commands)
    assert lease.state == ControllerState.neutral()
