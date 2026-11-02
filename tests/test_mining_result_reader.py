"""Source-indexed result glyph/context regressions, not rank-accuracy claims."""
import hashlib
import json

import cv2
import numpy as np
import pytest

from raiders_helper.mining_result_reader import (
    ASSETS, ANCHOR_ROIS, ResultRankReader, inspect_result, read_result,
)


def image(name):
    return cv2.imdecode(np.frombuffer((ASSETS/name).read_bytes(),np.uint8),cv2.IMREAD_COLOR)


@pytest.mark.parametrize("name,rank",[("fixture-C.png","C"),("fixture-S.png","S")])
def test_actual_result_sources_are_read_with_matching_score_not_probability(name,rank):
    frame=image(name)
    assert read_result(frame,(0,0,1,1))==rank
    reading=inspect_result(frame,(0,0,1,1))
    assert reading.result_visible and .82<=reading.confidence<=1
    assert reading.match_margin>=.10
    assert reading.numeric_score is None
    assert reading.confidence_kind=="normalized_template_match_score_not_probability"


def test_real_menu_high_score_a_b_are_not_the_current_round_result():
    reading=inspect_result(image("fixture-menu.png"),(0,0,1,1))
    assert reading.rank is None and not reading.result_visible
    assert reading.box_score==0


@pytest.mark.parametrize("rank",["A","B"])
def test_observed_menu_font_in_synthetic_result_proves_only_shape_pipeline(rank):
    # These fixtures have a visible Chinese synthetic footer. They are NOT
    # observations that the game or our program actually achieved A/B.
    assert read_result(image(f"synthetic-{rank}-on-C.png"),(0,0,1,1))==rank
    source=json.loads((ASSETS/"source.json").read_text(encoding="utf8"))
    glyph=next(row for row in source["records"] if row["path"]==f"rank-{rank}.png")
    composite=next(row for row in source["records"] if row["path"]==f"synthetic-{rank}-on-C.png")
    assert glyph["kind"]=="menu_glyph_only_not_actual_result"
    assert composite["kind"]=="synthetic_glyph_pipeline_not_actual_result"


def test_rank_and_score_words_plus_glyph_without_box_or_options_are_rejected():
    frame=np.zeros((480,640,3),np.uint8)
    real=image("fixture-C.png")
    for x,y,w,h in [ANCHOR_ROIS["rank"],ANCHOR_ROIS["score"],(338,202,42,34)]:
        frame[y:y+h,x:x+w]=real[y:y+h,x:x+w]
    reading=inspect_result(frame,(0,0,1,1))
    assert min(reading.anchor_scores["rank"],reading.anchor_scores["score"])>.9
    assert reading.rank is None and not reading.result_visible


@pytest.mark.parametrize("missing",["rank","score"])
def test_missing_adjacent_text_anchor_rejects_even_with_full_result_box(missing):
    frame=image("fixture-C.png")
    x,y,w,h=ANCHOR_ROIS[missing]
    frame[y-7:y+h+7,x-7:x+w+7]=0
    assert read_result(frame,(0,0,1,1)) is None


def test_four_side_box_can_confirm_early_result_without_retry_quit_yet():
    frame=image("fixture-C.png")
    frame[246:325,200:434]=0
    reading=inspect_result(frame,(0,0,1,1))
    assert max(reading.anchor_scores["retry"],reading.anchor_scores["quit"])<.78
    assert reading.box_score>=.80 and reading.rank=="C"


def test_recognized_options_can_confirm_result_when_box_edge_is_partly_occluded():
    frame=image("fixture-C.png")
    frame[143:158,140:505]=0
    reading=inspect_result(frame,(0,0,1,1))
    assert reading.box_score<.80
    assert max(reading.anchor_scores["retry"],reading.anchor_scores["quit"])>=.78
    assert reading.rank=="C"


@pytest.mark.parametrize("kind",["blank","cross","two_glyphs"])
def test_missing_unknown_and_multi_character_rank_fields_are_rejected(kind):
    frame=image("fixture-C.png")
    frame[202:236,338:380]=0
    if kind=="cross":
        cv2.line(frame,(347,208),(364,226),(240,240,240),2)
        cv2.line(frame,(364,208),(347,226),(240,240,240),2)
    elif kind=="two_glyphs":
        cv2.rectangle(frame,(340,209),(350,224),(240,240,240),2)
        cv2.rectangle(frame,(365,209),(376,224),(240,240,240),2)
    reading=inspect_result(frame,(0,0,1,1))
    assert reading.result_visible and reading.rank is None


def test_duplicate_best_templates_are_ambiguous_even_at_high_match_score():
    reader=ResultRankReader()
    reader.templates["B"]=reader.templates["C"].copy()
    reading=reader.inspect(image("fixture-C.png"),(0,0,1,1))
    assert reading.rank is None and reading.confidence>.99
    assert reading.match_margin==pytest.approx(0,abs=1e-6)


@pytest.mark.parametrize("rank",["C","S"])
@pytest.mark.parametrize("factor",[.65,.85])
def test_font_normalization_tolerates_brightness_change_without_threshold_claims(rank,factor):
    frame=(image(f"fixture-{rank}.png").astype(np.float32)*factor).astype(np.uint8)
    assert read_result(frame,(0,0,1,1))==rank


def test_default_ns_viewport_and_960_width_sampling_keep_source_c_shape():
    canonical=image("fixture-C.png")
    full=np.zeros((1080,1920,3),np.uint8)
    full[55:1015,320:1600]=cv2.resize(canonical,(1280,960),interpolation=cv2.INTER_NEAREST)
    assert read_result(full)=="C"
    half=cv2.resize(full,(960,540),interpolation=cv2.INTER_AREA)
    assert read_result(half)=="C"


@pytest.mark.parametrize("viewport",[[],[0,0,0,1],[0,0,2,1],[-.1,0,1,1],[0,0,float("nan"),1],[True,0,1,1]])
def test_invalid_viewport_never_produces_a_rank(viewport):
    assert read_result(image("fixture-C.png"),viewport) is None


@pytest.mark.parametrize("frame",[None,np.zeros((10,20),np.uint8),np.zeros((10,20,3),float)])
def test_invalid_frame_never_produces_a_rank(frame):
    assert read_result(frame,(0,0,1,1)) is None


def test_packaged_templates_and_fixture_hashes_match_source_manifest():
    manifest=json.loads((ASSETS/"source.json").read_text(encoding="utf8"))
    for record in manifest["records"]:
        assert hashlib.sha256((ASSETS/record["path"]).read_bytes()).hexdigest()==record["sha256"]
    assert manifest["sources"]["S"]["frame_index"]==9360
    assert len(manifest["sources"]["S"]["video_sha256"])==64
