"""Host-side latch for the first promised release in a mixed MINING plan.

Deadlines request controller submission; they are not console execution
timestamps. Only successful backend history shifted by L can confirm that
release has reached the capture clock.
"""
import math

from .control.buttons import Button, ControllerState


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class MixedDeadlineGuard:
    def __init__(self, parameters):
        self.enabled = parameters.get("mixed_deadline_guard", False)
        if type(self.enabled) is not bool:
            raise ValueError("mixed_deadline_guard 必须为布尔值")
        self.phase = "eligible"
        self.release_at = None
        self.first_release_at = None
        self.release_requested_at = None
        self.normal_decision_seen = False
        self.last_constraint_sequence = None
        self.allow_rearm = False
        self.release_receipt_after = None
        self.delay = parameters.get("effective_delay_seconds")
        if self.enabled and (not _finite(self.delay) or not 0 <= self.delay <= .5):
            raise ValueError("mixed deadline需要显式有效延迟，范围0到.5秒")

    def _braking(self, now):
        if self.phase != "braking":
            self.release_requested_at = min(now, self.release_at) if self.release_at is not None else now
            self.normal_decision_seen = False
        self.phase = "braking"
        self.allow_rearm = False

    def _expire(self, now):
        if self.phase == "scheduled" and now >= self.release_at:
            self._braking(now)

    def _confirmed_release(self, observation):
        metrics = observation.metrics
        rows = metrics.get("mining_input_history")
        until = metrics.get("mining_input_history_observed_until")
        if (metrics.get("mining_input_history_source") not in {
                "guard_submission_after", "backend_send_state", "commands.jsonl.after"}
                or metrics.get("mining_input_history_gap_detected") is True
                or not isinstance(rows, (list, tuple)) or not rows or not _finite(until)
                or until < observation.timestamp - self.delay
                or observation.timestamp < self.release_requested_at + self.delay):
            return None
        current = None
        previous = -math.inf
        last_unsafe_after = -math.inf
        for row in rows:
            if not isinstance(row, dict) or not _finite(row.get("after")):
                return None
            after = row["after"]
            if after < previous or after > until + 1e-6:
                return None
            previous = after
            state = row.get("state")
            if isinstance(state, dict):
                buttons = state.get("buttons")
                if not isinstance(buttons, (list, tuple)):
                    return None
                unsafe = bool(buttons)
            else:
                if type(row.get("zr")) is not bool:
                    return None
                unsafe = row["zr"]
            if unsafe:last_unsafe_after = after
            if after + self.delay <= observation.timestamp + 1e-9:
                current = row
        if current is None:
            return None
        # A newer submitted ZR still in the capture delay is already known;
        # it must not be hidden by an older visible neutral receipt.
        if last_unsafe_after >= current["after"]:
            return None
        state = current.get("state")
        if isinstance(state, dict):
            buttons = state.get("buttons")
            if not isinstance(buttons, (list, tuple)) or buttons:
                return None
        elif current.get("zr") is not False:
            return None
        # This is the actual latest submitted mode on the capture clock.
        # An already-held normal state also proves release; requiring a new
        # neutral event would deadlock when the backend is already neutral.
        return current["after"]

    def attach(self, observation, *, now):
        if not self.enabled:
            return
        if not _finite(now):
            raise ValueError("mixed deadline时钟必须有限")
        self._expire(now)
        if observation.phase != "playing" and self.phase == "scheduled":
            self._braking(now)
        self.last_constraint_sequence = observation.sequence
        self.release_receipt_after = (self._confirmed_release(observation)
                                      if self.phase == "braking" and observation.phase == "playing" else None)
        self.allow_rearm = bool(self.phase == "braking" and self.normal_decision_seen
                               and self.release_receipt_after is not None)
        observation.metrics["mining_mixed_guard"] = {
            "enabled": True, "phase": self.phase, "release_at": self.release_at,
            "first_release_at": self.first_release_at,
            "release_clock": "host_lease_command_request_not_backend_receipt",
            "allow_full_fast_rearm": self.allow_rearm,
            "normal_decision_seen": self.normal_decision_seen,
            "release_requested_at": self.release_requested_at,
            "release_backend_receipt_after": self.release_receipt_after,
            "release_history_declared_source": observation.metrics.get("mining_input_history_source"),
            "release_confirmed_on_capture_clock": self.release_receipt_after is not None,
            "effective_delay_seconds": self.delay,
            "actual_console_receipt_confirmed": False,
        }

    def apply(self, observation, decision, *, now, fresh=True):
        """Return executable state, independent lease deadline and trace details.

        A decision that crossed the deadline while computing is wholly neutral;
        its ZR-mode lateral axis is never reused under normal physics.
        """
        state = decision.state if fresh else ControllerState.neutral()
        if not self.enabled:
            return state, None, {"enabled": False}
        if not _finite(now):
            raise ValueError("mixed deadline时钟必须有限")
        before = self.phase
        self._expire(now)
        veto = None
        policy = decision.diagnostics
        flags = policy.get("mixed_guard", {})
        if not isinstance(flags, dict):
            flags = {}
        boosting = Button.ZR in state.buttons
        normal_replanned = bool(fresh and observation.phase == "playing"
                                and flags.get("normal_replanned") is True and not boosting)
        if observation.sequence != self.last_constraint_sequence:
            self._braking(now)
            state = ControllerState.neutral();veto = "constraint_observation_mismatch"
        elif not fresh:
            state = ControllerState.neutral();veto = "stale_observation"
        elif observation.phase != "playing":
            if self.phase == "scheduled":self._braking(now)
            state = ControllerState.neutral();veto = "non_playing"
        elif self.phase == "braking":
            if boosting:
                safe_rearm = (before == "braking" and self.allow_rearm
                              and flags.get("full_fast_rearm_verified") is True
                              and policy.get("mode") == "accelerated"
                              and policy.get("route_status") == "complete"
                              and policy.get("mixed_zr_prefix_seconds") is None)
                if safe_rearm:
                    self.phase = "eligible";self.release_at = None;self.first_release_at = None;self.release_requested_at = None
                    self.normal_decision_seen = False;self.allow_rearm = False
                else:
                    state = ControllerState.neutral()
                    veto = ("deadline_crossed_during_policy_compute" if before == "scheduled"
                            else "braking_requires_normal_replan_and_verified_full_fast")
            elif normal_replanned:
                self.normal_decision_seen = True
        elif self.phase == "scheduled":
            if not boosting:
                self._braking(now)
                self.normal_decision_seen = normal_replanned
            elif policy.get("mode") != "accelerated_then_normal":
                # A pure-fast suggestion cannot silently cancel a promised
                # release; its model did not include that pending switch.
                self._braking(now);state = ControllerState.neutral()
                veto = "scheduled_release_cannot_be_cancelled_by_inflight_fast"
            else:
                seconds = policy.get("mixed_zr_prefix_seconds")
                if not _finite(seconds) or not 0 < seconds <= .5:
                    self._braking(now);state = ControllerState.neutral();veto = "invalid_mixed_duration"
                else:
                    # A more urgent plan may release earlier, never later.
                    self.release_at = min(self.release_at, now + seconds)
        elif boosting and policy.get("mode") == "accelerated_then_normal":
            seconds = policy.get("mixed_zr_prefix_seconds")
            if not _finite(seconds) or not 0 < seconds <= .5:
                self._braking(now);state = ControllerState.neutral();veto = "invalid_mixed_duration"
            else:
                self.release_at = now + seconds
                self.first_release_at = self.release_at
                self.phase = "scheduled"
                self.normal_decision_seen = False
        deadline = self.release_at if self.phase == "scheduled" else None
        return state, deadline, {"enabled": True, "phase_before_execution": before,
            "phase_after_execution": self.phase, "release_at": self.release_at,
            "first_release_at": self.first_release_at,
            "lease_release_at": deadline, "release_requested_at": self.release_requested_at,
            "normal_decision_seen": self.normal_decision_seen,
            "actual_control_veto_reason": veto, "policy_axis_reused_after_zr_removal": False}
