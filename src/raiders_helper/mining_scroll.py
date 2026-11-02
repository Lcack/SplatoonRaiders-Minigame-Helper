"""Measured vertical translation of independently visible left/right walls.

This is a gated image measurement, not a movement model. It consumes the
original adaptive floor mask before hole filling, sprite or wake repairs.
Coordinates are the recognizer's canonical 640 by 480 viewport.
"""
import math

import numpy as np


def wall_edges(floor):
    if not isinstance(floor, np.ndarray) or floor.shape != (480, 640):
        raise ValueError("wall scroll needs the original 640x480 floor mask")
    mask = floor > 0
    left = np.argmax(mask, axis=1).astype(float)
    right = 639 - np.argmax(mask[:, ::-1], axis=1).astype(float)
    wide = np.sum(mask, axis=1) >= 70
    # A boundary cut off by the viewport is not a measured visible wall.
    left[~wide | (left <= 2)] = np.nan
    right[~wide | (right >= 637)] = np.nan
    return np.stack([left, right], axis=1)


def _means(values, valid, axis):
    counts = np.sum(valid, axis=axis)
    totals = np.sum(np.where(valid, values, 0.), axis=axis)
    return np.divide(totals, counts, out=np.full(totals.shape, np.inf), where=counts > 0)


def estimate_wall_translation(previous, current, seconds):
    """Estimate a common shift with coverage, uniqueness and segment gates."""
    result = {"accepted": False, "reason": "invalid_interval"}
    if type(seconds) not in (float, int) or not math.isfinite(seconds) or not .025 <= seconds <= .20:
        return result
    if previous.shape != (480, 2) or current.shape != (480, 2):
        raise ValueError("wall profiles must have shape (480,2)")
    ys = np.arange(145, 385)
    shifts = np.arange(-2, min(85, int(math.ceil(seconds * 1.5 * 480)) + 3) + 1)
    differences = np.abs(previous[ys[None, :] + shifts[:, None]] - current[ys][None, :, :])
    valid = np.isfinite(differences)
    capped = np.minimum(differences, 12.)
    coverage = np.mean(valid, axis=1)
    costs = _means(capped, valid, 1)
    costs[coverage < .8] = np.inf
    joint = np.mean(costs, axis=1)
    if not np.any(np.isfinite(joint)):
        return {**result, "reason": "insufficient_visible_wall_coverage"}
    best = int(np.argmin(joint));dy = int(shifts[best])
    side_shifts = shifts[np.argmin(costs, axis=0)]
    outside = np.abs(shifts - dy) > 3
    competitors = joint[outside & np.isfinite(joint)]
    margin = float(np.min(competitors) - joint[best]) if len(competitors) else 0.
    result.update(reason="ambiguous_or_inconsistent_bilateral_shift", integer_pixels=dy,
        residual_pixels=float(joint[best]), ambiguity_margin_pixels=margin,
        side_pixels=side_shifts.tolist(), visible_row_coverage=coverage[best].tolist())
    if dy < 0 or np.max(np.abs(side_shifts - dy)) > 2 or joint[best] >= 2.5 or margin <= .6:
        return result
    segments = []
    for start in (0, 80, 160):
        section = slice(start, start + 80)
        fraction = np.mean(valid[:, section, :], axis=1)
        segment_cost = _means(capped[:, section, :], valid[:, section, :], 1)
        segment_cost[fraction < .65] = np.inf
        combined = np.mean(segment_cost, axis=1)
        chosen = int(np.argmin(combined))
        segment_dy = int(shifts[chosen])
        other = combined[(np.abs(shifts - segment_dy) > 3) & np.isfinite(combined)]
        separation = float(np.min(other) - combined[chosen]) if len(other) else 0.
        supported = bool(np.all(fraction[best] >= .65) and abs(segment_dy - dy) <= 2
                         and combined[chosen] < 2.5 and separation > .3)
        segments.append({"start_row":145 + start, "coverage":fraction[best].tolist(),
                         "pixels":segment_dy, "supports_global_shift":supported,
                         "contradicts_global_shift":bool(np.all(fraction[chosen]>=.65)
                             and abs(segment_dy-dy)>2 and combined[chosen]<2.5 and separation>.3)})
    result["vertical_segments"] = segments
    if (any(min(s["coverage"]) < .65 or s["contradicts_global_shift"] for s in segments)
            or sum(s["supports_global_shift"] for s in segments) < 2):
        result["reason"] = "insufficient_independent_vertical_segments"
        return result
    offset = 0.
    if 0 < best < len(joint) - 1:
        a, b, c = joint[best - 1:best + 2]
        denominator = a + c - 2 * b
        if np.all(np.isfinite([a, b, c])) and denominator > 1e-9:
            offset = float(np.clip((a - c) / (2 * denominator), -.5, .5))
    raw = dy + offset
    # Subpixel fitting around an integer-zero minimum has signed noise.
    # Publish the explicit half-pixel resolution interval, not negative speed.
    near_zero = dy == 0
    displacement = 0. if near_zero else raw
    speed = displacement / 480 / seconds
    if not 0 <= speed <= 1.5:
        result.update(reason="speed_outside_measured_search_domain", raw_pixels=raw)
        return result
    result.update(accepted=True, reason="bilateral_and_segment_agreement", pixels=displacement,
        raw_pixels=raw, speed=speed, zero_motion_within_half_pixel=near_zero,
        zero_resolution_speed=.5 / 480 / seconds if near_zero else None,
        zero_is_quantized_measurement_not_exact_stop_proof=near_zero)
    return result


class WallScrollEstimator:
    def __init__(self):
        self.previous = None

    def observe(self, floor, timestamp, palette, phase="playing", *, blocked=False, measure=True):
        diagnostic = {"accepted":False, "source":"bilateral_wall_translation",
                      "raw_floor_before_repairs":True}
        if phase != "playing" or blocked:
            self.previous = None
            return {"wall_scroll":{**diagnostic, "reason":"non_playing_collision_or_powered"}}
        hue = palette.get("hue") if isinstance(palette, dict) else None
        if (type(timestamp) not in (float, int) or not math.isfinite(timestamp)
                or type(hue) not in (float, int) or not math.isfinite(hue) or not 0 <= hue < 180):
            self.previous = None
            return {"wall_scroll":{**diagnostic, "reason":"invalid_timestamp_or_palette"}}
        current = wall_edges(floor)
        previous = self.previous
        self.previous = (current, timestamp, hue)
        if not measure:
            return {"wall_scroll":{**diagnostic, "reason":"reliable_landmark_preferred"}}
        if previous is None:
            return {"wall_scroll":{**diagnostic, "reason":"need_previous_actual_frame"}}
        old, old_timestamp, old_hue = previous
        seconds = timestamp - old_timestamp
        if seconds <= 0:
            self.previous = None
            return {"wall_scroll":{**diagnostic, "reason":"non_monotonic_capture_time"}}
        difference = abs(hue - old_hue)
        difference = min(difference, 180 - difference)
        if difference > 4:
            return {"wall_scroll":{**diagnostic, "reason":"palette_hue_jump", "hue_change":difference}}
        measured = estimate_wall_translation(old, current, seconds)
        diagnostic.update(measured, window_seconds=seconds,
                          reference_timestamp=(old_timestamp + timestamp) / 2)
        if not measured["accepted"]:
            return {"wall_scroll":diagnostic}
        return {"wall_scroll":diagnostic, "current_forward_speed_observed":True,
            "current_forward_speed":measured["speed"], "current_forward_speed_confidence":.80,
            "current_forward_speed_source":"bilateral_wall_translation",
            "current_forward_speed_method":"wall_profile_pair",
            "current_forward_speed_units":"viewport_heights_per_second",
            "current_forward_speed_confidence_kind":"heuristic_gate_score_not_probability",
            "current_forward_speed_reference_timestamp":(old_timestamp + timestamp) / 2,
            "current_forward_speed_window_seconds":seconds,
            "current_forward_speed_reference_kind":"pair_midpoint_velocity_approximation",
            "current_forward_speed_timestamp":timestamp}
