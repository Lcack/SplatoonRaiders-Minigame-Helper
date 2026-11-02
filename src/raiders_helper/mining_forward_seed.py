"""Short-lived capture-time forward state from real landmark windows.

This estimates motion between observations. It never changes a vision
measurement into an instantaneous observation, and never starts from a guess.
"""
import math

from .mining_forward import ForwardDynamics


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class ForwardSeedEstimator:
    def __init__(self, parameters):
        config = parameters.get("forward_dynamics", {})
        if not isinstance(config, dict) or type(config.get("enabled", False)) is not bool:
            raise ValueError("forward_dynamics.enabled 必须为布尔值")
        self.enabled = config.get("enabled", False)
        self.model = None
        self.seed = None
        self.blocked_until = -math.inf
        self.last_timestamp = None
        if not self.enabled:
            return
        keys = ("zr_acceleration", "normal_speed", "normal_tau_seconds")
        if any(key not in config for key in keys):
            raise ValueError("启用纵向seed需要显式动力学系数")
        self.model = ForwardDynamics(**{key: config[key] for key in keys})
        self.maximum_age = config.get("maximum_seed_age_seconds", 1.5)
        self.delay = parameters.get("effective_delay_seconds")
        if not _finite(self.maximum_age) or not 0 < self.maximum_age <= 1.5:
            raise ValueError("纵向seed有效期必须大于0且不超过1.5秒")
        if not _finite(self.delay) or not 0 <= self.delay <= .5:
            raise ValueError("纵向seed需要显式有效延迟，范围0到.5秒")

    def _invalidate(self, observation, reason, *, barrier=False):
        self.seed = None
        if barrier and _finite(observation.timestamp):
            self.blocked_until = max(self.blocked_until, observation.timestamp)
        observation.metrics["mining_forward_seed"] = {
            "valid": False, "timestamp": observation.timestamp, "reason": reason,
            "source": "unavailable", "fallback": "legacy_mode_speed",
            "confidence_kind": "model_estimate_not_observation",
        }

    def _history(self, observation, start):
        metrics = observation.metrics
        source = metrics.get("mining_input_history_source")
        if source not in {"guard_submission_after", "commands.jsonl.after", "backend_send_state"}:
            return None
        until = metrics.get("mining_input_history_observed_until")
        rows = metrics.get("mining_input_history")
        if (not _finite(until) or until < observation.timestamp - self.delay
                or not isinstance(rows, (list, tuple)) or not rows
                or metrics.get("mining_input_history_gap_detected") is True):
            return None
        timestamps = []
        for row in rows:
            if not isinstance(row, dict) or not _finite(row.get("after")):
                return None
            timestamp = row["after"] + self.delay
            if row["after"] > until + 1e-6 or timestamps and timestamp < timestamps[-1]:
                return None
            timestamps.append(timestamp)
        active = [i for i, timestamp in enumerate(timestamps) if timestamp <= start + 1e-9]
        if not active:
            return None
        pieces = []
        # Only states affecting this interval need to be modelled. Earlier
        # menu confirmation is not a playing-mode input.
        for i in range(active[-1], len(rows)):
            if timestamps[i] > observation.timestamp:
                break
            row = rows[i]
            state = row.get("state")
            if isinstance(state, dict):
                buttons = state.get("buttons")
                left = state.get("left_stick", {})
                right = state.get("right_stick", {})
                if (not isinstance(buttons, (list, tuple)) or any(b not in {"ZR"} for b in buttons)
                        or not isinstance(left, dict) or not isinstance(right, dict)
                        or not _finite(left.get("x", 0)) or abs(left.get("x", 0)) > 32767
                        or left.get("y", 0) != 0 or any(right.get(k, 0) != 0 for k in ("x", "y"))):
                    return None
                boosted = "ZR" in buttons
            else:
                boosted = row.get("zr")
                if type(boosted) is not bool or not _finite(row.get("axis")) or abs(row["axis"]) > 1:
                    return None
            pieces.append((max(start, timestamps[i]), boosted))
        return pieces

    def attach(self, observation):
        if not self.enabled:
            return
        metrics = observation.metrics
        timestamp = observation.timestamp
        if not _finite(timestamp) or self.last_timestamp is not None and timestamp <= self.last_timestamp:
            self._invalidate(observation, "non_monotonic_capture_time", barrier=True)
            return
        self.last_timestamp = timestamp
        unsafe = (observation.phase != "playing" or metrics.get("game_ended")
                  or metrics.get("collision_suspected") or metrics.get("hit_body_detected")
                  or metrics.get("buff_active") or metrics.get("motion_mode") == "powered"
                  or metrics.get("player_appearance") == "powered"
                  or bool(observation.of_kind("egg_buff")))
        if unsafe:
            self._invalidate(observation, "non_playing_collision_or_powered", barrier=True)
            return
        if metrics.get("mining_input_history_gap_detected") is True:
            self._invalidate(observation, "input_history_gap", barrier=True)
            return
        reference = metrics.get("current_forward_speed_reference_timestamp")
        window = metrics.get("current_forward_speed_window_seconds")
        speed = metrics.get("current_forward_speed")
        measurement_source = metrics.get("current_forward_speed_source")
        method = metrics.get("current_forward_speed_method")
        accepted = {("tracked_landmark_consensus", "paired_positions"): .8,
                    ("tracked_landmark_consensus", "tracker_window"): .8,
                    ("dual_landmark_consensus", "paired_positions"): .82,
                    ("temporal_single_landmark", "temporal_positions"): .80,
                    ("bilateral_wall_translation", "wall_profile_pair"): .80}
        observed = (metrics.get("current_forward_speed_observed") is True
                    and (measurement_source, method) in accepted
                    and metrics.get("current_forward_speed_units") == "viewport_heights_per_second"
                    and _finite(metrics.get("current_forward_speed_confidence"))
                    and metrics["current_forward_speed_confidence"] >= accepted.get((measurement_source, method), math.inf)
                    and _finite(reference) and _finite(window) and 0 < window <= 1.5
                    and _finite(speed) and 0 <= speed <= 1.5
                    and reference - window / 2 > self.blocked_until
                    and reference + window / 2 <= timestamp + 1e-6
                    and 0 <= timestamp - reference <= self.maximum_age)
        if observed and speed == 0:
            # A real stationary landmark consensus is new contrary evidence,
            # not a missing measurement that permits carrying the old speed.
            self._invalidate(observation, "observed_stop_response_not_fitted", barrier=True)
            return
        fresh = observed and (self.seed is None or reference > self.seed["reference_timestamp"] + 1e-9)
        if fresh:
            start = reference
            source = "observed_window_propagated"
            measurement = {"reference_timestamp": reference, "window_seconds": window,
                           "measurement_method": method, "measurement_source": measurement_source,
                           "reference_kind": metrics.get("current_forward_speed_reference_kind")}
        elif self.seed is not None:
            speed = self.seed["speed"]
            start = self.seed["timestamp"]
            source = "recent_estimate"
            measurement = {k: self.seed[k] for k in (
                "reference_timestamp", "window_seconds", "measurement_method", "measurement_source", "reference_kind")}
        else:
            self._invalidate(observation, "no_trustworthy_landmark_seed")
            return
        age = timestamp - measurement["reference_timestamp"]
        if age > self.maximum_age + 1e-9:
            self._invalidate(observation, "last_real_measurement_expired")
            return
        history = self._history(observation, start)
        if history is None:
            self._invalidate(observation, "input_history_does_not_cover_propagation", barrier=True)
            return
        for i, (segment_start, boosted) in enumerate(history):
            segment_end = history[i + 1][0] if i + 1 < len(history) else timestamp
            _, speed = self.model.advance(speed, boosted, segment_end - segment_start)
        self.seed = {"valid": True, "speed": speed, "timestamp": timestamp,
                     "source": source, **measurement, "age_seconds": age,
                     "units": "viewport_heights_per_second",
                     "confidence_kind": "model_estimate_not_observation",
                     "effective_delay_seconds": self.delay,
                     "history_source": metrics["mining_input_history_source"],
                     "terminal_speed_measured": False}
        metrics["mining_forward_seed"] = dict(self.seed)
