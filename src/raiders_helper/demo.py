"""Synthetic fixtures ONLY. These are not screenshots or a game simulator."""
import math

import cv2
import numpy as np

from .capture.models import FramePacket
from .models import Game


COLORS = {"player": (0, 255, 0), "partner": (255, 255, 0), "ball": (0, 255, 255),
          "obstacle": (0, 0, 255), "reward": (255, 0, 255), "goal": (255, 0, 0),
          "phase:playing": (255, 255, 255)}


def demo_profile(game):
    detectors = []
    for kind, bgr in COLORS.items():
        hue, sat, val = map(int, cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0])
        detectors.append({"kind": kind, "method": "hsv", "lower": [max(0, hue - 2), max(0, sat - 10), 240],
                          "upper": [min(179, hue + 2), min(255, sat + 10), 255], "min_area": 20})
    if Game(game) == Game.MINING:
        detectors.append({"kind": "corridor", "method": "corridor", "lower": [0, 0, 70],
                          "upper": [179, 20, 90], "scan_y": [0.36, 0.43, 0.5]})
    return {"version": 1, "game": Game(game).value, "calibrated": False, "synthetic": True,
            "viewport": [0, 0, 1, 1], "reference_size": [640, 480], "stable_frames": 3,
            "required": ["player"], "detectors": detectors}


def demo_packet(game, sequence, timestamp=None):
    game = Game(game)
    t = sequence / 30
    frame = np.full((480, 640, 3), 24, np.uint8)
    def box(kind, x, y, w=0.05, h=0.05):
        cv2.rectangle(frame, (int((x - w / 2) * 640), int((y - h / 2) * 480)),
                      (int((x + w / 2) * 640), int((y + h / 2) * 480)), COLORS[kind], -1)
    box("phase:playing", 0.03, 0.03, 0.02, 0.02)
    if game == Game.MINING:
        cv2.rectangle(frame, (80, 50), (550, 479), (80, 80, 80), -1)
        box("player", 0.5 + 0.15 * math.sin(t), 0.3)
        box("obstacle", 0.5, 0.5 + 0.1 * math.sin(t * 2), 0.12, 0.08)
        box("reward", 0.75, 0.6)
    elif game in (Game.HOPPING, Game.JUGGLING):
        box("player", 0.3 + 0.05 * math.sin(t), 0.85, 0.1, 0.05)
        box("ball", 0.45 + 0.2 * math.sin(t), 0.2 + 0.5 * abs(math.sin(t * 1.3)))
        if game == Game.JUGGLING:
            box("partner", 0.75, 0.85, 0.1, 0.05)
            box("ball", 0.7 - 0.2 * math.sin(t), 0.2 + 0.5 * abs(math.sin(t * 1.3 + 1)))
    else:
        box("player", 0.3 + 0.1 * math.sin(t), 0.4)
        box("goal", 0.9, 0.8, 0.12, 0.04)
    return FramePacket(frame, t if timestamp is None else timestamp, sequence, 30)
