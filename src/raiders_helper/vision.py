"""Configurable baseline detector. No game-specific templates are bundled."""
import json
import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .models import Entity, Game, Observation


def crop_roi(frame, roi):
    if len(roi) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in roi):
        raise ValueError("ROI 必须为 0..1 的 [x,y,w,h]")
    x, y, w, h = roi
    if w <= 0 or h <= 0 or x + w > 1.000001 or y + h > 1.000001:
        raise ValueError("ROI 超出画面")
    height, width = frame.shape[:2]
    result = frame[int(y * height):round((y + h) * height),
                   int(x * width):round((x + w) * width)]
    if not result.size:
        raise ValueError("ROI 小于一个像素")
    return result


def load_profile(path, game):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["game"] != Game(game).value:
        raise ValueError("配置的游戏与当前选择不一致")
    if data.get("version") != 1:
        raise ValueError("不支持的配置版本")
    data["_root"] = str(path.resolve().parent)
    return data


def create_recognizer(game, profile=None):
    if profile and profile.get("vision_backend") == "mining_live":
        from .mining_vision import MiningLiveRecognizer
        return MiningLiveRecognizer(game, profile)
    if profile and profile.get("vision_backend") == "arcade_fast":
        from .arcade_vision import FastArcadeRecognizer
        return FastArcadeRecognizer(game, profile)
    return Recognizer(game, profile)


class Tracker:
    """One-to-one nearest-neighbour association; no predictions on missing frames."""
    def __init__(self):
        self.previous = ()
        self.timestamp = None
        self.next_id = 0

    def update(self, entities, timestamp):
        dt = timestamp - self.timestamp if self.timestamp is not None else 0
        previous = self.previous if 0 < dt <= 0.25 else ()
        candidates = sorted(
            (math.hypot(e.x - p.x, e.y - p.y), i, j)
            for i, e in enumerate(entities) for j, p in enumerate(previous)
            if e.kind == p.kind
        )
        pairs, used = {}, set()
        for distance, i, j in candidates:
            if distance < 0.15 and i not in pairs and j not in used:
                pairs[i] = j
                used.add(j)
        tracked = []
        for i, e in enumerate(entities):
            if i in pairs:
                p = previous[pairs[i]]
                e = replace(e, track_id=p.track_id, vx=(e.x - p.x) / dt,
                            vy=(e.y - p.y) / dt, samples=p.samples + 1)
            else:
                e = replace(e, track_id=self.next_id)
                self.next_id += 1
            tracked.append(e)
        self.previous, self.timestamp = tuple(tracked), timestamp
        return self.previous


def fill_recognized_obstacle_gaps(entities):
    """Restore floor intervals ONLY where a recognized rock occludes the floor.

    Never extend outer road edges, fill unknown holes, or fill a wall interval.
    This prevents powered rocks being accidentally encoded as impassable road.
    """
    others = [e for e in entities if e.kind != "corridor"]
    obstacles = [e for e in others if e.kind == "obstacle" and e.confidence >= 0.85]
    walls = [e for e in others if e.kind == "wall"]
    rows = {}
    for e in entities:
        if e.kind == "corridor":
            rows.setdefault(e.y, []).append(e)
    for y, intervals in sorted(rows.items()):
        merged = []
        for current in sorted(intervals, key=lambda e: e.x-e.w/2):
            if merged:
                previous = merged[-1]
                left, right = previous.x + previous.w/2, current.x-current.w/2
                known_rock = any(abs(e.y-y) <= e.h/2 and e.x-e.w/2 <= left+1e-6
                                 and right <= e.x+e.w/2+1e-6 for e in obstacles)
                wall = any(abs(e.y-y) <= e.h/2 and e.x+e.w/2 > left and e.x-e.w/2 < right for e in walls)
                if right <= left+1e-6 or (known_rock and not wall):
                    low = previous.x-previous.w/2
                    high = max(previous.x+previous.w/2, current.x+current.w/2)
                    merged[-1] = replace(previous, x=(low+high)/2, w=high-low,
                                         confidence=min(previous.confidence, current.confidence))
                    continue
            merged.append(current)
        others.extend(merged)
    return tuple(others)


class Recognizer:
    def __init__(self, game, profile=None):
        self.game = Game(game)
        self.profile = profile or {"game": self.game.value, "viewport": [0, 0, 1, 1]}
        self.tracker = Tracker()
        self.templates = {}
        self.stable_phase = "unknown"
        self.phase_count = 0
        self.digits = {}
        for spec in self.profile.get("detectors", []):
            if spec["method"] == "template":
                path = Path(self.profile.get("_root", ".")) / spec["path"]
                template = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                if template is None or template.std() < 1:
                    raise ValueError(f"无效或纯色模板: {path}")
                self.templates[spec["path"]] = template
            elif spec["method"] not in ("hsv", "corridor"):
                raise ValueError(f"未知检测器: {spec['method']}")
        for reader in self.profile.get("readouts", []):
            for digit, relative in reader["templates"].items():
                if digit not in "0123456789" or len(digit) != 1:
                    raise ValueError("数字模板标签必须为 0..9")
                path = Path(self.profile.get("_root", ".")) / relative
                image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is None or image.std() < 1:
                    raise ValueError(f"无效数字模板: {path}")
                self.digits[relative] = image

    def read_metrics(self, area):
        metrics = {}
        for reader in self.profile.get("readouts", []):
            digits = []
            for roi in reader["slots"]:
                tile = cv2.cvtColor(crop_roi(area, roi), cv2.COLOR_BGR2GRAY)
                scores = []
                for digit, path in reader["templates"].items():
                    template = self.digits[path]
                    resized = cv2.resize(tile, (template.shape[1], template.shape[0]))
                    score = float(cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)[0, 0]) if resized.std() >= 1 else -1
                    scores.append((score, digit))
                scores.sort(reverse=True)
                if len(scores) < 2 or scores[0][0] < reader.get("threshold", 0.9) or scores[0][0] - scores[1][0] < 0.05:
                    break
                digits.append(scores[0][1])
            if digits and len(digits) == len(reader["slots"]):
                metrics[reader["name"]] = int("".join(digits)) / (10 ** reader.get("decimals", 0))
        return metrics

    def observe(self, packet):
        area = crop_roi(packet.frame, self.profile.get("viewport", [0, 0, 1, 1]))
        # All templates use a canonical playfield resolution.
        width, height = self.profile.get("reference_size", [640, 480])
        area = cv2.resize(area, (width, height))
        found = []
        for spec in self.profile.get("detectors", []):
            roi = spec.get("roi", [0, 0, 1, 1])
            tile = crop_roi(area, roi)
            boxes = []
            threshold = float(spec.get("threshold", 0.85))
            if not 0 <= threshold <= 1:
                raise ValueError("检测阈值必须在 0..1")
            if spec["method"] in ("hsv", "corridor"):
                mask = cv2.inRange(cv2.cvtColor(tile, cv2.COLOR_BGR2HSV),
                                   np.array(spec["lower"]), np.array(spec["upper"]))
                if spec["method"] == "corridor":
                    # Road colour only. Scanline intervals become hard route
                    # constraints; absence of a road interval means no action.
                    for scan_y in spec.get("scan_y", [0.3, 0.4, 0.5]):
                        if not 0 <= scan_y < 1:
                            raise ValueError("scan_y 必须在 0..1 内")
                        row = mask[min(int(scan_y * tile.shape[0]), tile.shape[0] - 1)] > 0
                        edges = np.diff(np.r_[False, row, False].astype(np.int8))
                        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                            if end - start >= spec.get("min_width", 30):
                                boxes.append((int(start), int(scan_y * tile.shape[0]), int(end - start), 1, 0.9))
                    for x, y, w, h, score in boxes:
                        found.append(Entity("corridor", roi[0] + (x + w / 2) / tile.shape[1] * roi[2],
                                            roi[1] + y / tile.shape[0] * roi[3],
                                            w / tile.shape[1] * roi[2], h / tile.shape[0] * roi[3], score))
                    continue
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) >= spec.get("min_area", 20):
                        x, y, w, h = cv2.boundingRect(contour)
                        # A colour hit is a heuristic score, not a calibrated probability.
                        boxes.append((x, y, w, h, 0.9))
            else:
                template = self.templates[spec["path"]]
                th, tw = template.shape[:2]
                if th > tile.shape[0] or tw > tile.shape[1]:
                    continue
                scores = cv2.matchTemplate(tile, template, cv2.TM_CCOEFF_NORMED)
                for _ in range(spec.get("max_objects", 12)):
                    _, score, _, (x, y) = cv2.minMaxLoc(scores)
                    if not math.isfinite(score) or score < threshold:
                        break
                    boxes.append((x, y, tw, th, score))
                    scores[max(0, y - th // 2):y + th, max(0, x - tw // 2):x + tw] = -1
            for x, y, w, h, score in boxes:
                if score >= threshold:
                    found.append(Entity(spec["kind"], roi[0] + (x + w / 2) / tile.shape[1] * roi[2],
                                        roi[1] + (y + h / 2) / tile.shape[0] * roi[3],
                                        w / tile.shape[1] * roi[2], h / tile.shape[0] * roi[3], score))
        if any(spec.get("fill_obstacles") for spec in self.profile.get("detectors", []) if spec["method"] == "corridor"):
            found = list(fill_recognized_obstacle_gaps(found))
        phases = {e.kind.split(":", 1)[1] for e in found if e.kind.startswith("phase:")}
        phase = next(iter(phases)) if len(phases) == 1 else "unknown"
        self.phase_count = self.phase_count + 1 if phase == self.stable_phase else 1
        self.stable_phase = phase
        if self.phase_count < self.profile.get("stable_frames", 3):
            phase = "unknown"
        entities = self.tracker.update(tuple(e for e in found if not e.kind.startswith("phase:")), packet.timestamp)
        required = self.profile.get("required", ["player"])
        scores = [max((e.confidence for e in found if e.kind == kind), default=0.0)
                  for kind in required + ["phase:playing"]]
        confidence = min(scores, default=0.0) if phase == "playing" else 0.0
        notes = () if self.profile.get("detectors") else ("尚未配置实机模板/ROI",)
        return Observation(self.game, packet.timestamp, packet.sequence, phase, confidence, entities, notes, self.read_metrics(area))
