"""Read an observed MINING result rank, independently of control/runtime.

Ranks are matched to source-indexed A/B/C/S glyphs. Match scores are normalized
correlations, not probabilities. A/B shapes currently come from a real menu,
not observed A/B results. No game score/rank threshold is inferred here.
"""
from dataclasses import dataclass, field
from functools import lru_cache
import math
from pathlib import Path

import cv2
import numpy as np


ASSETS = Path(__file__).resolve().parents[2]/"assets/mining_result"
DEFAULT_VIEWPORT = (1/6, 55/1080, 2/3, 960/1080)
RANK_FIELD = (338, 202, 42, 34)
ANCHOR_ROIS = {"rank": (192, 209, 90, 24), "score": (192, 174, 112, 25),
               "retry": (262, 251, 122, 30), "quit": (262, 286, 100, 30)}


def _load(path):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.std() < 1:
        raise ValueError(f"无效的结算字形/锚点模板：{path}")
    return image


def _ink(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    peak = float(gray.max()) if gray.size else 0.0
    if peak < 70:
        return np.zeros_like(gray, dtype=np.uint8)
    return (gray > max(25, .42*peak)).astype(np.uint8)*255


def _normalise_glyph(image):
    mask = _ink(image)
    if not np.any(mask):
        return None
    x, y, width, height = cv2.boundingRect(mask)
    if not 12 <= width <= 26 or not 10 <= height <= 23:
        return None
    glyph = mask[y:y+height, x:x+width]
    if not .15 <= float(np.mean(glyph > 0)) <= .80:
        return None
    scale = min(24/width, 24/height)
    glyph = cv2.resize(glyph, (round(width*scale), round(height*scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((32, 32), np.uint8)
    top, left = (32-glyph.shape[0])//2, (32-glyph.shape[1])//2
    canvas[top:top+glyph.shape[0], left:left+glyph.shape[1]] = glyph
    return canvas


def _canonical(frame, viewport):
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
            or frame.shape[2] != 3 or not frame.size):
        raise ValueError("frame需为非空uint8 BGR画面")
    if not isinstance(viewport, (list, tuple)) or len(viewport) != 4:
        raise ValueError("viewport需为归一化x/y/w/h")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in viewport):
        raise ValueError("viewport需为有限数")
    x, y, width, height = viewport
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x+width > 1+1e-9 or y+height > 1+1e-9:
        raise ValueError("viewport超出画面")
    fh, fw = frame.shape[:2]
    left, right = round(x*fw), round((x+width)*fw)
    top, bottom = round(y*fh), round((y+height)*fh)
    if right <= left or bottom <= top:
        raise ValueError("viewport为空")
    return cv2.resize(frame[top:bottom, left:right], (640, 480))


def _match(tile, template):
    if tile.std() < 1 or any(a < b for a, b in zip(tile.shape, template.shape)):
        return 0.0
    value = float(cv2.matchTemplate(tile, template, cv2.TM_CCOEFF_NORMED).max())
    return max(0.0, min(1.0, value)) if math.isfinite(value) else 0.0


def _box_score(canonical):
    """Four luminous sides around a mostly black result interior."""
    gray = cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY)
    bright = gray > 80
    top = float(np.max(np.mean(bright[144:157, 153:487], axis=1)))
    bottom = float(np.max(np.mean(bright[326:340, 153:487], axis=1)))
    left = float(np.max(np.mean(bright[158:325, 139:155], axis=0)))
    right = float(np.max(np.mean(bright[158:325, 485:503], axis=0)))
    interior = canonical[158:325, 156:484]
    black = float(np.mean(np.max(interior, axis=2) < 50))
    return min(top, bottom, left, right, black)


@dataclass(frozen=True)
class ResultReading:
    rank: str | None
    confidence: float = 0.0
    reason: str = "未确认结算"
    result_visible: bool = False
    glyph_scores: dict = field(default_factory=dict)
    anchor_scores: dict = field(default_factory=dict)
    box_score: float = 0.0
    match_margin: float | None = None
    numeric_score: int | None = None
    confidence_kind: str = "normalized_template_match_score_not_probability"


class ResultRankReader:
    def __init__(self, asset_directory=ASSETS, *, minimum_match=.82, ambiguity_margin=.10, anchor_threshold=.78):
        self.directory = Path(asset_directory)
        self.templates = {rank:_load(self.directory/f"rank-{rank}.png") for rank in ("A", "B", "C", "S")}
        self.anchors = {name:_load(self.directory/f"anchor-{name}.png") for name in ANCHOR_ROIS}
        self.minimum_match, self.ambiguity_margin, self.anchor_threshold = minimum_match, ambiguity_margin, anchor_threshold

    def inspect(self, frame, viewport=DEFAULT_VIEWPORT):
        try:
            canonical = _canonical(frame, viewport)
        except (ValueError, TypeError):
            return ResultReading(None, reason="画面或viewport无效")
        scores = {}
        for name, (x, y, width, height) in ANCHOR_ROIS.items():
            tile = canonical[y-7:y+height+7, x-7:x+width+7]
            scores[name] = _match(_ink(tile), self.anchors[name])
        box = _box_score(canonical)
        anchors_present = min(scores["rank"], scores["score"]) >= self.anchor_threshold
        context_present = max(scores["retry"], scores["quit"]) >= self.anchor_threshold or box >= .80
        if not anchors_present or not context_present:
            return ResultReading(None, reason="缺少相邻RANK/SCORE与结算选项或边框证据",
                                 anchor_scores=scores, box_score=box)
        x, y, width, height = RANK_FIELD
        glyph = _normalise_glyph(canonical[y:y+height, x:x+width])
        if glyph is None:
            return ResultReading(None, reason="结算等级字形缺失或尺寸不明", result_visible=True,
                                 anchor_scores=scores, box_score=box)
        search = cv2.copyMakeBorder(glyph, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        glyph_scores = {rank:_match(search, template) for rank, template in self.templates.items()}
        ordered = sorted(glyph_scores, key=glyph_scores.get, reverse=True)
        best, second = ordered[:2]
        confidence, margin = glyph_scores[best], glyph_scores[best]-glyph_scores[second]
        accepted = confidence >= self.minimum_match and margin >= self.ambiguity_margin
        return ResultReading(best if accepted else None, confidence,
            "已匹配结算等级；分数OCR尚未实现" if accepted else "等级未知或候选歧义",
            True, glyph_scores, scores, box, margin)


@lru_cache(maxsize=1)
def _reader():
    return ResultRankReader()


def read_result(frame, viewport=DEFAULT_VIEWPORT):
    """Return observed A/B/C/S or None; menu high scores are not results."""
    return _reader().inspect(frame, viewport).rank


def inspect_result(frame, viewport=DEFAULT_VIEWPORT):
    """Return matching diagnostics. confidence is a match score, not probability."""
    return _reader().inspect(frame, viewport)
