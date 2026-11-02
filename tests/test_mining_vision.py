"""Meaningful MINING perception gates; synthetic scenes are labelled as such."""
from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from raiders_helper.capture.models import FramePacket
from raiders_helper.mining_vision import MiningLiveRecognizer


ROOT=Path(__file__).resolve().parents[1]


def scene(player=True):
    frame=np.zeros((480,640,3),np.uint8)
    cv2.rectangle(frame,(100,35),(539,479),(30,80,140),-1)
    anchor=cv2.imdecode(np.frombuffer((ROOT/"assets/arcade/score-anchor.png").read_bytes(),np.uint8),1)
    frame[2:2+anchor.shape[0],390:390+anchor.shape[1]]=anchor
    if player:
        cv2.rectangle(frame,(303,76),(336,108),(30,220,40),-1)
        cv2.rectangle(frame,(310,87),(329,97),(210,45,20),-1)
    return frame


def recognizer(**values):
    return MiningLiveRecognizer(profile={"viewport":[0,0,1,1],"phase_anchors":False,
                                          "stable_frames":1,**values})


def packet(frame,t=1,sequence=1):
    return FramePacket(frame,t,sequence,60)


def test_robot_seed_full_envelope_and_self_occlusion_floor():
    r=recognizer();obs=r.observe(packet(scene()))
    assert obs.phase=="playing"
    player,=obs.of_kind("player")
    assert abs(player.x-.5)<.002 and abs(player.y-92.5/480)<.002
    assert player.w*640>=32 and player.h*480>=31
    assert all(left<player.x<right for y,left,right in obs.metrics["road_rows"] if .16<y<.22)
    assert r.last_road_mask[92,320]==255


def test_real_cv_number_types_are_accepted_by_controller():
    from raiders_helper.mining_reactive import decide_mining_reactive
    r=recognizer();r.observe(packet(scene()))
    obs=r.observe(packet(scene(),1+1/60,2))
    player,=obs.of_kind("player")
    assert all(type(value) is float for value in (player.x,player.y,player.w,player.h,player.vx,player.vy))
    decision=decide_mining_reactive(obs)
    assert "玩家缺失" not in decision.reason


def test_time_zero_ends_before_result_dialogue():
    frame=scene()
    zero=cv2.imdecode(np.frombuffer((ROOT/"assets/mining/time-zero-strict-0.png").read_bytes(),np.uint8),0)
    frame[1:30,96:190]=cv2.cvtColor(zero,cv2.COLOR_GRAY2BGR)
    obs=recognizer().observe(packet(frame))
    assert obs.metrics["game_ended"] and obs.phase=="result_pending"


def test_every_nonzero_digit_overrides_a_similar_whole_zero_template():
    zero=cv2.imdecode(np.frombuffer((ROOT/"assets/mining/time-zero-strict-1.png").read_bytes(),np.uint8),0)
    for digit in range(1,10):
        glyph=cv2.imdecode(np.frombuffer((ROOT/f"assets/mining/timer-digit-{digit}.png").read_bytes(),np.uint8),0)
        for x in (102,122,162):
            frame=scene()
            frame[1:30,96:190]=cv2.cvtColor(zero,cv2.COLOR_GRAY2BGR)
            frame[4:29,x:x+20]=cv2.cvtColor(glyph,cv2.COLOR_GRAY2BGR)
            obs=recognizer().observe(packet(frame))
            assert not obs.metrics["game_ended"],(digit,x,obs.metrics["time_zero_digits"])
            assert not obs.metrics["time_zero_digits_verified"]


def test_real_six_seconds_is_not_zero_even_when_whole_timer_ncc_exceeds_threshold():
    path=ROOT/"artifacts/mining-supervision/round-04/frame-00012591.jpg"
    if not path.exists():pytest.skip("local NS false-zero evidence not bundled")
    frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    # The raw trace scored .9288; JPEG compression lowers this saved image
    # to .876. Lower only this test's whole-template gate to reproduce its
    # role as a zero candidate; the independent digit gate must reject it.
    obs=MiningLiveRecognizer(profile={"stable_frames":1,"time_zero_threshold":.85}).observe(packet(frame))
    assert obs.metrics["time_zero_score"]>.85
    units=obs.metrics["time_zero_digits"][1]
    assert units["nonzero_digit"]==6 and units["nonzero_score"]>units["zero_score"]
    assert not obs.metrics["game_ended"] and obs.phase!="result_pending"


def test_sprite_blink_is_not_published_as_a_detected_or_tracked_player():
    r=recognizer();obs=r.observe(packet(scene()))
    previous=obs.of_kind("player")[0]
    missing=r.observe(packet(scene(False),1+1/60,2))
    assert not missing.of_kind("player") and missing.phase=="unknown"
    assert missing.metrics["player_detected"] is False
    assert missing.metrics["player_predicted_x"]==previous.x
    assert any("不可见" in note for note in missing.notes)


def test_blue_reacquisition_reports_real_gap_without_inventing_velocity():
    r=recognizer();r.observe(packet(scene()))
    missing=r.observe(packet(scene(False),1.3,2))
    assert not missing.metrics["blue_reacquired"]
    assert missing.metrics["blue_last_seen_age_seconds"]==pytest.approx(.3)
    obs=r.observe(packet(scene(),1.5,3))
    p,=obs.of_kind("player")
    assert p.samples==1 and p.vx==0 and p.vy==0
    assert obs.metrics["blue_reacquired"]
    assert obs.metrics["blue_reacquired_after_seconds"]==pytest.approx(.5)
    assert obs.metrics["blue_last_seen_age_seconds"]==0


def hit_scene(x=303):
    frame=scene(False)
    cv2.rectangle(frame,(x,76),(x+33,108),(90,10,180),-1)
    return frame


def test_collision_appearance_is_separate_from_blue_player_and_needs_identity():
    assert not recognizer().observe(packet(hit_scene())).of_kind("player_hit")
    r=recognizer();r.observe(packet(scene()))
    obs=r.observe(packet(hit_scene(),1.1,2))
    assert not obs.of_kind("player") and obs.phase=="unknown"
    hit,=obs.of_kind("player_hit")
    assert abs(hit.x-.5)<.002 and obs.metrics["collision_suspected"]
    assert obs.metrics["hit_body_detected"] and not obs.metrics["player_detected"]
    assert all(type(value) is float for value in (hit.x,hit.y,hit.w,hit.h))


def test_collision_appearance_rejects_distant_stale_hud_and_orange_fire():
    cases=[(hit_scene(460),1.1),(hit_scene(),2.0)]
    fire=scene(False);cv2.circle(fire,(320,92),18,(0,90,255),-1)
    cases.append((fire,1.1))
    hud=scene(False);cv2.rectangle(hud,(305,2),(338,30),(90,10,180),-1)
    cases.append((hud,1.1))
    for frame,t in cases:
        r=recognizer();r.observe(packet(scene()))
        obs=r.observe(packet(frame,t,2))
        assert not obs.of_kind("player_hit") and not obs.metrics["collision_suspected"]


def test_real_collision_sprite_uses_only_recent_identity():
    directory=ROOT/"artifacts/mining-supervision/round-03"
    names=["frame-00013698.jpg","frame-00013822.jpg"]
    if not all((directory/name).exists() for name in names):pytest.skip("local NS collision evidence not bundled")
    frames=[cv2.imdecode(np.frombuffer((directory/name).read_bytes(),np.uint8),1) for name in names]
    r=MiningLiveRecognizer(profile={"stable_frames":1,"phase_anchors":False})
    assert r.observe(packet(frames[0])).of_kind("player")
    obs=r.observe(packet(frames[1],1.5,2))
    assert obs.metrics["collision_suspected"] and obs.of_kind("player_hit")
    assert not obs.of_kind("player")
    assert not MiningLiveRecognizer(profile={"phase_anchors":False}).observe(packet(frames[1])).of_kind("player_hit")


def test_rank_result_stops_before_retry_and_quit_appear():
    frame=scene()
    rank=cv2.imdecode(np.frombuffer((ROOT/"assets/session/result-rank.png").read_bytes(),np.uint8),1)
    frame[203:203+rank.shape[0],190:190+rank.shape[1]]=rank
    obs=recognizer().observe(packet(frame))
    assert obs.phase=="result_pending" and obs.metrics["game_ended"]
    assert obs.metrics["game_ended_source"]=="result_anchor"


def test_real_early_result_box_stops_with_body_still_visible():
    directory=ROOT/"artifacts/mining-supervision/round-03"
    paths=[directory/"frame-00015212.jpg",directory/"frame-00015274.jpg",directory/"last-frame.png"]
    if not all(path.exists() for path in paths):pytest.skip("local NS result evidence not bundled")
    r=MiningLiveRecognizer(profile={"stable_frames":1})
    observations=[r.observe(packet(cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1),1+i*.1,i)) for i,path in enumerate(paths)]
    assert not observations[0].metrics["game_ended"]
    assert observations[1].phase=="result_pending" and observations[1].metrics["game_ended"]
    assert observations[1].of_kind("player")
    assert observations[2].phase=="result" and observations[2].metrics["game_ended"]


def test_ambiguous_blue_bodies_do_not_choose_one_arbitrarily():
    frame=scene(False)
    for x in (260,360):cv2.rectangle(frame,(x,87),(x+19,97),(210,45,20),-1)
    obs=recognizer().observe(packet(frame))
    assert not obs.of_kind("player") and obs.metrics["ambiguous_player"]


def test_group_of_small_blue_stars_is_not_a_robot_body():
    frame=scene(False)
    for x,y in ((450,75),(475,75),(438,100),(463,100),(488,100)):
        cv2.drawMarker(frame,(x,y),(220,220,40),cv2.MARKER_DIAMOND,12,2)
    obs=recognizer().observe(packet(frame))
    assert not obs.of_kind("player") and obs.phase=="unknown"


def test_black_wake_above_body_is_not_extended_into_forward_road():
    frame=scene()
    frame[35:76,310:330]=0
    r=recognizer();obs=r.observe(packet(frame))
    assert r.last_road_mask[60,320]==0
    assert all(left<.5<right for y,left,right in obs.metrics["road_rows"] if .23<y<.7)


@pytest.mark.parametrize("backend",["legacy","adaptive_inner"])
def test_near_rear_wake_merges_only_local_envelope_and_keeps_binary_evidence(backend):
    frame=scene()
    # Curved trailing cut extends beyond the static sprite repair bbox.
    frame[35:86,307:355]=0
    r=recognizer(floor_backend=backend);obs=r.observe(packet(frame))
    player,=obs.of_kind("player")
    near=[row for row in obs.metrics["road_rows"] if player.y-player.h/2<=row[0]<=player.y]
    assert near and all(left<.2 and right>.8 for _,left,right in near)
    assert obs.metrics["wake_repaired_rows"]
    assert r.last_road_mask[77,340]==0  # pixels are not relabelled as observed floor


@pytest.mark.parametrize("backend",["legacy","adaptive_inner"])
def test_wake_repair_does_not_bridge_forward_branch_divider(backend):
    frame=scene()
    frame[35:270,329:385]=0
    r=recognizer(floor_backend=backend);obs=r.observe(packet(frame))
    assert not obs.metrics["wake_repaired_rows"]
    assert r.last_road_mask[130,350]==0
    # Future rows remain disjoint; they are never merged into the full road.
    assert all(right<.61 or left>.51 for y,left,right in obs.metrics["road_rows"] if .23<y<.5)


def test_multiple_body_wake_gaps_are_each_proven_and_merged():
    r=recognizer()
    p=r.observe(packet(scene())).of_kind("player")[0]
    mask=np.zeros((480,640),np.uint8);mask[35:,100:540]=255
    mask[35:85,285:365]=0
    mask[35:85,321:335]=255  # observed floor island splits the wake in two
    rows=r._road_rows(mask,p,1.1)
    near=[row for row in rows if .15<row[0]<.18]
    assert near and all(left<.2 and right>.8 for _,left,right in near)
    assert sum(abs(y-76/480)<.001 for y,_,_ in r._wake_repairs)==2
    assert mask[76,300]==0 and mask[76,350]==0


def test_multiple_gaps_never_merge_a_divider_continuing_in_front():
    r=recognizer();p=r.observe(packet(scene())).of_kind("player")[0]
    mask=np.zeros((480,640),np.uint8);mask[35:,100:540]=255
    mask[35:85,285:321]=0
    mask[35:270,335:365]=0
    rows=r._road_rows(mask,p,1.1)
    assert all(right<.6 or left>.5 for y,left,right in rows if .15<y<.5)
    assert not any(left<.2 and right>.8 for y,left,right in rows if .15<y<.5)


def test_real_wake_just_outside_blue_bbox_remains_local_occlusion():
    path=ROOT/"artifacts/mining-supervision/round-05/frame-00002731.jpg"
    if not path.exists():pytest.skip("local NS wake evidence not bundled")
    frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    profile={"stable_frames":1,"phase_anchors":False}
    before=MiningLiveRecognizer(profile={**profile,"repair_player_wake":False}).observe(packet(frame))
    after=MiningLiveRecognizer(profile=profile).observe(packet(frame))
    p,=after.of_kind("player")
    near=[row for row in after.metrics["road_rows"] if p.y-p.h/2-.001<=row[0]<=p.y+p.h/2+.001]
    assert all(left<p.x-p.w/2 and right>p.x+p.w/2 for _,left,right in near)
    bottom=p.y+p.h/2
    assert [row for row in before.metrics["road_rows"] if row[0]>bottom]==[row for row in after.metrics["road_rows"] if row[0]>bottom]


def test_rear_gap_far_from_player_or_too_wide_is_not_assumed_wake():
    for left,right in ((440,480),(240,400)):
        frame=scene();frame[35:85,left:right]=0
        obs=recognizer().observe(packet(frame))
        assert not obs.metrics["wake_repaired_rows"]


def test_real_rear_wake_regression_does_not_change_forward_rows():
    path=ROOT/"artifacts/mining-supervision/round-02/frame-00000518.jpg"
    if not path.exists():pytest.skip("local NS round evidence not bundled")
    frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    profile={"stable_frames":1,"phase_anchors":False}
    before=MiningLiveRecognizer(profile={**profile,"repair_player_wake":False}).observe(packet(frame))
    after=MiningLiveRecognizer(profile=profile).observe(packet(frame))
    player,=after.of_kind("player")
    body_top=player.y-player.h/2
    old_near=[row for row in before.metrics["road_rows"] if body_top-.001<=row[0]<=player.y]
    new_near=[row for row in after.metrics["road_rows"] if body_top-.001<=row[0]<=player.y]
    assert any(right<player.x+player.w/2 for _,left,right in old_near)
    assert all(left<player.x-player.w/2 and right>player.x+player.w/2 for _,left,right in new_near)
    assert [row for row in before.metrics["road_rows"] if row[0]>player.y]==[row for row in after.metrics["road_rows"] if row[0]>player.y]


def test_wake_plus_sprite_occlusion_repairs_full_body_but_not_future_rows():
    frame=scene()
    # The 76px combined gap exceeds two body widths, but 34px is the
    # visible sprite itself. It ends at the body bottom, not in front.
    frame[35:109,282:358]=0
    cv2.rectangle(frame,(303,76),(336,108),(30,220,40),-1)
    cv2.rectangle(frame,(310,87),(329,97),(210,45,20),-1)
    before=recognizer(repair_player_wake=False).observe(packet(frame))
    after=recognizer().observe(packet(frame))
    p,=after.of_kind("player")
    body_bottom=p.y+p.h/2
    near=[row for row in after.metrics["road_rows"] if p.y-p.h/2<=row[0]<=body_bottom]
    assert near and all(left<.2 and right>.8 for _,left,right in near)
    assert any(y>p.y for y,_,_ in after.metrics["wake_repaired_rows"])
    assert [row for row in before.metrics["road_rows"] if row[0]>body_bottom]==[row for row in after.metrics["road_rows"] if row[0]>body_bottom]


def test_real_round_three_wake_is_repaired_across_entire_body():
    path=ROOT/"artifacts/mining-supervision/round-03/frame-00012636.jpg"
    if not path.exists():pytest.skip("local NS round evidence not bundled")
    frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    profile={"stable_frames":1,"phase_anchors":False}
    before=MiningLiveRecognizer(profile={**profile,"repair_player_wake":False}).observe(packet(frame))
    after=MiningLiveRecognizer(profile=profile).observe(packet(frame))
    p,=after.of_kind("player")
    bottom=p.y+p.h/2
    near=[row for row in after.metrics["road_rows"] if p.y-p.h/2<=row[0]<=bottom]
    assert all(left<p.x-p.w/2 and right>p.x+p.w/2 for _,left,right in near)
    assert any(y>p.y for y,_,_ in after.metrics["wake_repaired_rows"])
    assert [row for row in before.metrics["road_rows"] if row[0]>bottom]==[row for row in after.metrics["road_rows"] if row[0]>bottom]


def test_real_zero_strict_bank_accepts_both_observed_capture_phases():
    paths=[ROOT/"artifacts/mining-supervision/live-round01.png",
           ROOT/"artifacts/mining-supervision/round-02/last-frame.png"]
    if not all(path.exists() for path in paths):pytest.skip("local NS timer evidence not bundled")
    for path in paths:
        frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
        obs=MiningLiveRecognizer(profile={"stable_frames":1,"phase_anchors":False}).observe(packet(frame))
        assert obs.metrics["game_ended"] and obs.phase=="result_pending"


def test_outer_road_preserves_curves_instead_of_intersecting_all_rows():
    frame=scene(False);frame[35:]=0
    cv2.fillPoly(frame,[np.array([[100,35],[400,35],[600,479],[300,479]],np.int32)],(30,80,140))
    r=recognizer();obs=r.observe(packet(frame))
    rows=obs.metrics["road_rows"]
    assert rows[-1][1]>rows[0][1]+.2
    assert rows[-1][2]>rows[0][2]+.2


def test_compact_interior_boulder_is_hazard_not_road_outer_edge():
    frame=scene()
    cv2.circle(frame,(400,300),25,(5,10,15),-1)
    r=recognizer();obs=r.observe(packet(frame))
    rock,=obs.of_kind("obstacle")
    assert abs(rock.x-400/640)<.01 and abs(rock.y-300/480)<.01
    assert r.last_road_mask[300,400]==255
    near=[row for row in obs.metrics["road_rows"] if abs(row[0]-.58)<.01][0]
    assert near[1]<.2 and near[2]>.8
    assert obs.metrics["obstacles_verified"] is False


def test_near_body_dark_candidate_is_diagnostic_and_forward_boulder_remains():
    frame=scene()
    cv2.circle(frame,(220,92),25,(5,10,15),-1)
    cv2.circle(frame,(400,300),25,(5,10,15),-1)
    obs=recognizer().observe(packet(frame))
    rock,=obs.of_kind("obstacle")
    assert abs(rock.x-400/640)<.01 and rock.y>.5
    uncertain,=obs.metrics["unverified_near_body_dark"]
    assert abs(uncertain["x"]-220/640)<.01 and uncertain["reason"]=="dark_center_not_ahead_of_visible_body"


def test_boulder_rim_is_detected_when_inner_darkness_connects_to_dark_floor():
    frame=scene()
    frame[200:,100:540]=(15,30,65)
    cv2.circle(frame,(400,300),24,(10,15,30),-1)
    cv2.circle(frame,(400,300),25,(40,125,180),4)
    frame[318:331,397:403]=(15,30,65)  # dark connection through one rim sector
    baseline=recognizer(circle_rocks=False).observe(packet(frame))
    assert not baseline.of_kind("obstacle")
    obs=recognizer(circle_rocks=True).observe(packet(frame))
    rock,=obs.of_kind("obstacle")
    assert abs(rock.x-400/640)<.02 and abs(rock.y-300/480)<.02
    assert obs.metrics["circle_rock_added"]==1


def test_circle_supplement_rejects_outside_wall_ring_and_small_stars():
    frame=scene()
    cv2.circle(frame,(65,300),24,(5,10,15),-1)
    cv2.circle(frame,(65,300),25,(180,180,180),4)
    for y in (220,260,300,340):
        cv2.drawMarker(frame,(400,y),(240,240,240),cv2.MARKER_DIAMOND,12,3)
    obs=recognizer(circle_rocks=True).observe(packet(frame))
    assert not obs.of_kind("obstacle")
    assert not obs.metrics["circle_rock_evidence"]


@pytest.mark.parametrize("relative,centers",[
    ("contact-windows/window-02/00155261.png",[(.3844,.5525),(.4519,.8275)]),
    ("frame-00158557.jpg",[(.6056,.3875)]),
])
def test_batch05_actual_missing_small_rims_are_observed_as_aabb(relative,centers):
    folder=ROOT/"artifacts/mining-A-campaign/batch-05/round-0001"
    path=folder/"gameplay"/relative
    if not path.exists():pytest.skip("local NS missed-stone evidence not bundled")
    frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    profile=json.loads((folder/"profile.json").read_text(encoding="utf8"))
    profile.update(phase_anchors=False,stable_frames=1)
    baseline=MiningLiveRecognizer(profile={**profile,"circle_rocks":False}).observe(packet(frame))
    measured=MiningLiveRecognizer(profile={**profile,"circle_rocks":True}).observe(packet(frame))
    assert not baseline.of_kind("obstacle")
    assert len(measured.of_kind("obstacle"))==len(centers)
    for x,y in centers:
        assert any(abs(e.x-x)<.015 and abs(e.y-y)<.015 for e in measured.of_kind("obstacle"))
    assert all(e["rim_sectors"]>=9 for e in measured.metrics["circle_rock_evidence"])
    assert "obstacle_shapes" not in measured.metrics  # no circle collision hypothesis


def test_small_near_ring_requires_current_rim_and_recent_mature_identity():
    from dataclasses import replace
    from raiders_helper.models import Entity
    frame=scene();cv2.circle(frame,(220,100),24,(10,15,30),-1)
    cv2.circle(frame,(220,100),25,(180,180,180),4)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    outer=np.zeros((480,640),np.uint8);outer[35:,100:540]=255
    distance=cv2.distanceTransform(outer,cv2.DIST_L2,3)
    player=Entity("player",.5,92/480,34/640,34/480,.9)
    r=recognizer(retain_tracked_near_rocks=True)
    old=Entity("obstacle",220/640,114/480,56/640,56/480,.82,track_id=7,samples=8,vy=-.3)
    r.tracker.tracks[7]=(old,1.,[])
    denied,_=r._ring_rocks(gray,distance,player,[],near_identity=None)
    found,evidence=r._ring_rocks(gray,distance,player,[],near_identity=lambda c:r._near_rock_identity(c,1.1))
    assert not denied and len(found)==1
    assert evidence[0]["near_rock_identity"]["prior_track_id"]==7
    r.tracker.tracks[7]=(replace(old,samples=2),1.,[])
    young,_=r._ring_rocks(gray,distance,player,[],near_identity=lambda c:r._near_rock_identity(c,1.1))
    assert not young
    blank,_=r._ring_rocks(cv2.cvtColor(scene(),cv2.COLOR_BGR2GRAY),distance,player,[],near_identity=lambda c:{"track_id":7})
    assert not blank


def test_real_problem_rims_add_only_the_three_visually_missing_stones():
    from raiders_helper.models import Entity
    directory=ROOT/"artifacts/mining-supervision/round-06"
    paths=sorted(directory.glob("problem-*.png"))
    if len(paths)!=8:pytest.skip("local NS raw problem evidence not bundled")
    trace=[json.loads(line)["observation"] for line in (directory/"trace.jsonl").read_text(encoding="utf8").splitlines()]
    added={39385,39439,39488}
    expected={39250:2,39255:1,39286:1,39385:1,39439:1,39488:1,39607:1,41290:0}
    for path in paths:
        seq=int(path.name.split("-")[1]);source=next(o for o in trace if o["sequence"]==seq)
        # Context comes from the actual earlier blue observation in the trace,
        # so a red sprite is associated without inventing a current blue body.
        previous=next((o for o in reversed(trace) if o["sequence"]<seq and
                       any(e["kind"]=="player" for e in o["entities"])),None)
        r=MiningLiveRecognizer(profile={"stable_frames":1,"phase_anchors":False,"circle_rocks":True})
        if previous:
            r.last_player=Entity(**next(e for e in previous["entities"] if e["kind"]=="player"))
            r.last_player_time=previous["timestamp"]
        frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
        obs=r.observe(packet(frame,source["timestamp"],seq))
        assert len(obs.of_kind("obstacle"))==expected[seq],(seq,obs.metrics["circle_rock_evidence"])
        assert obs.metrics["circle_rock_added"]==(1 if seq in added else 0)
        if seq in added:
            assert any(abs(e.x-.596)<.025 for e in obs.of_kind("obstacle"))
            assert all(e["rim_sectors"]>=9 for e in obs.metrics["circle_rock_evidence"])


def test_real_wide_road_floor_shadows_are_not_hard_rocks_beside_body():
    path=ROOT/"artifacts/mining-supervision/round-04/frame-00011165.jpg"
    if not path.exists():pytest.skip("local NS false-rock evidence not bundled")
    frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    obs=MiningLiveRecognizer(profile={"stable_frames":1,"phase_anchors":False}).observe(packet(frame))
    p,=obs.of_kind("player")
    assert not obs.of_kind("obstacle")
    uncertain=obs.metrics["unverified_near_body_dark"]
    assert len(uncertain)>=1 and all(e["y"]<=p.y+p.h/2 for e in uncertain)


def test_real_wall_adjacent_body_does_not_remove_true_forward_boulder():
    path=ROOT/"artifacts/mining-supervision/round-05/frame-00002163.jpg"
    if not path.exists():pytest.skip("local NS false-rock evidence not bundled")
    frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    obs=MiningLiveRecognizer(profile={"stable_frames":1,"phase_anchors":False}).observe(packet(frame))
    p,=obs.of_kind("player")
    assert all(e.y>p.y+p.h/2 for e in obs.of_kind("obstacle"))
    assert any(abs(e.x-.534)<.02 and abs(e.y-.427)<.02 for e in obs.of_kind("obstacle"))


def near_rock_scene(y=None,fragment=False):
    frame=scene()
    if y is not None:
        if fragment:
            for x in (415,445):
                cv2.circle(frame,(x,y),8,(10,15,25),-1)
        else:
            cv2.circle(frame,(430,y),20,(10,15,25),-1)
            cv2.circle(frame,(430,y),20,(50,145,190),2)
    return frame


def mature_approaching_rock(r):
    latest=None
    for i,y in enumerate((157,147,137,127)):
        latest=r.observe(packet(near_rock_scene(y),1+i*.08,i+1))
    stone,=latest.of_kind("obstacle")
    assert stone.samples>=4
    return stone


def test_current_near_stone_requires_mature_identity_and_does_not_survive_disappearance():
    r=recognizer(retain_tracked_near_rocks=True)
    previous=mature_approaching_rock(r)
    near=r.observe(packet(near_rock_scene(107),1.40,6))
    stone,=near.of_kind("obstacle")
    assert stone.track_id==previous.track_id and near.metrics["retained_near_body_rocks"]
    missing=r.observe(packet(near_rock_scene(),1.48,7))
    assert not missing.of_kind("obstacle") and not missing.metrics["retained_near_body_rocks"]
    first=recognizer(retain_tracked_near_rocks=True).observe(packet(near_rock_scene(107)))
    assert not first.of_kind("obstacle") and first.metrics["unverified_near_body_dark"]


def test_recent_rock_identity_does_not_promote_smaller_current_fragments():
    r=recognizer(retain_tracked_near_rocks=True);mature_approaching_rock(r)
    obs=r.observe(packet(near_rock_scene(105,fragment=True),1.40,6))
    assert not obs.of_kind("obstacle") and not obs.metrics["retained_near_body_rocks"]


def test_near_rock_identity_expires_and_preserves_original_size_reference():
    from dataclasses import replace
    r=recognizer(retain_tracked_near_rocks=True);stone=mature_approaching_rock(r)
    assert r._near_rock_identity(replace(stone,y=stone.y-.01),1.55) is None
    candidate=replace(stone,y=stone.y-.01,w=stone.w*.90,h=stone.h*.90)
    assert r._near_rock_identity(candidate,1.32) is not None
    tracked=replace(candidate,track_id=stone.track_id,samples=5,vy=0.)
    r.tracker.tracks[stone.track_id]=(tracked,1.32,[])
    # Another 10% shrink would pass if each previous fragment became the
    # new baseline. It must fail against the original intact dimensions.
    assert r._near_rock_identity(replace(tracked,w=tracked.w*.9,h=tracked.h*.9),1.40) is None


def test_real_intact_stone_beside_robot_keeps_identity_after_crossing_body_bottom():
    directory=ROOT/"artifacts/mining-A-campaign/batch-03/round-0001/gameplay"
    window=directory/"contact-windows/window-03"
    paths=sorted(window.glob("*.png"))
    if not paths:pytest.skip("local intact-side-stone evidence not bundled")
    observations={o["sequence"]:o for line in (directory/"trace.jsonl").read_text(encoding="utf8").splitlines()
                  for o in [json.loads(line)["observation"]]}
    r=MiningLiveRecognizer(profile={"stable_frames":1,"phase_anchors":False,"floor_backend":"adaptive_inner",
                                     "retain_tracked_near_rocks":True,"wake_max_body_widths":2.5})
    saved={}
    for path in paths:
        seq=int(path.stem)
        if seq>41309:break
        source=observations[seq]
        frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
        observed=r.observe(packet(frame,source["timestamp"],seq))
        if seq in (41298,41302,41309):saved[seq]=observed
    assert set(saved)=={41298,41302,41309}
    stones={seq:next(e for e in obs.of_kind("obstacle") if abs(e.x-.625)<.02) for seq,obs in saved.items()}
    assert len({stone.track_id for stone in stones.values()})==1
    assert stones[41298].samples>=4
    for seq in (41302,41309):
        assert saved[seq].metrics["retained_near_body_rocks"]
        assert stones[seq].w==pytest.approx(.090625)
    alone=MiningLiveRecognizer(profile={"stable_frames":1,"phase_anchors":False,"floor_backend":"adaptive_inner",
                                         "retain_tracked_near_rocks":True}).observe(packet(frame))
    assert not any(abs(e.x-.625)<.02 for e in alone.of_kind("obstacle"))


@pytest.mark.parametrize("prior_kind",["mature","absent","expired","small","wrong_position"])
def test_real_large_side_boulder_requires_current_rim_and_recent_matching_identity(prior_kind):
    from collections import deque
    from dataclasses import replace
    from raiders_helper.models import Entity
    directory=ROOT/"artifacts/mining-A-campaign/batch-02/round-0005/gameplay"
    path=directory/"frame-00018430.jpg"
    if not path.exists():pytest.skip("local large side-boulder evidence not bundled")
    observations={o["sequence"]:o for line in (directory/"trace.jsonl").read_text(encoding="utf8").splitlines()
                  for o in [json.loads(line)["observation"]]}
    prior=next(Entity(**e) for e in observations[18420]["entities"] if e["kind"]=="obstacle" and e["w"]>.18)
    timestamp=observations[18430]["timestamp"]
    last=observations[18420]["timestamp"]
    frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    r=MiningLiveRecognizer(profile={"stable_frames":1,"phase_anchors":False,"floor_backend":"adaptive_inner",
        "large_rocks":True,"retain_tracked_near_rocks":True})
    if prior_kind=="small":prior=replace(prior,w=prior.w*.6,h=prior.h*.6)
    if prior_kind=="wrong_position":prior=replace(prior,x=.2)
    if prior_kind=="expired":last=timestamp-.31
    if prior_kind!="absent":
        # Seed only the already-recorded measured track at 18420. The
        # current geometry must still be detected from the 18430 image.
        r.tracker.tracks[prior.track_id]=(prior,last,deque([(last,prior.x,prior.y)],maxlen=5))
        r.tracker.last_time=last;r.tracker.next_id=1000
    obs=r.observe(packet(frame,timestamp,18430))
    large=[e for e in obs.of_kind("obstacle") if e.w>.18]
    if prior_kind=="mature":
        assert len(large)==1 and large[0].track_id==prior.track_id
        assert abs(large[0].x-.718125)<.002
        assert obs.metrics["large_rock_evidence"][0]["rim_sectors"]==12
        assert obs.metrics["retained_near_body_rocks"][0]["source"]=="current_large_rim_with_recent_rock_identity"
        # A known identity cannot manufacture an object in a blank frame.
        empty=r.observe(packet(np.zeros_like(frame),timestamp+.08,18435))
        assert not empty.of_kind("obstacle") and not empty.metrics["retained_near_body_rocks"]
    else:
        assert not large and not obs.metrics["large_rock_evidence"]


def test_large_near_ring_still_requires_real_rim_even_with_identity_callback():
    from raiders_helper.models import Entity
    gray=np.full((480,640),100,np.uint8)
    # Labelled synthetic current circle: actual ring pixels, not a carried
    # forward circle produced from the history callback.
    cv2.circle(gray,(430,105),60,30,-1)
    cv2.circle(gray,(430,105),60,180,5)
    floor=np.full((480,640),90,np.float32)
    body=Entity("player",.5,92/480,.05,.07,.88)
    accept=lambda e:{"prior_track_id":7,"prior_samples":10}
    found,evidence=MiningLiveRecognizer._ring_rocks(gray,floor,body,[],large=True,near_identity=accept)
    assert len(found)==1 and evidence[0]["rim_sectors"]>=9
    assert not MiningLiveRecognizer._ring_rocks(gray,floor,body,[],large=True)[0]
    # Remove the ring, leaving only a dark patch. Relaxed positional gates
    # must not turn the same identity into current detection.
    no_rim=np.full((480,640),100,np.uint8)
    assert not MiningLiveRecognizer._ring_rocks(no_rim,floor,body,[],large=True,near_identity=accept)[0]


def test_wall_scallop_and_long_shadow_are_not_boulders():
    frame=scene()
    for y in range(140,440,30):cv2.circle(frame,(105,y),12,(0,0,0),-1)
    cv2.rectangle(frame,(240,280),(430,294),(10,15,25),-1)
    obs=recognizer().observe(packet(frame))
    assert not obs.of_kind("obstacle")


def test_blank_or_unanchored_frame_cannot_be_playing():
    r=recognizer()
    assert r.observe(packet(np.zeros((480,640,3),np.uint8))).phase=="unknown"
    frame=scene();frame[:35]=0
    assert r.observe(packet(frame,1.1,2)).phase=="unknown"


def test_invalid_stability_and_other_game_are_rejected():
    with pytest.raises(ValueError):recognizer(stable_frames=0)
    with pytest.raises(ValueError):MiningLiveRecognizer("hopping",{})


def test_flow_candidate_is_never_promoted_to_measured_world_speed():
    r=recognizer()
    r.observe(packet(scene(),1,1))
    obs=r.observe(packet(scene(),1+1/60,2))
    assert obs.metrics["forward_speed_observed"] is False
    assert "forward_speed" not in obs.metrics


def scroll_entities(speeds=(.25,.251,.249)):
    from raiders_helper.models import Entity
    return [Entity("time_star",.4,.4+index*.1,.025,.03,.8,
                   track_id=index,vx=0.,vy=-speed,samples=5) for index,speed in enumerate(speeds)]


def test_landmark_scroll_observes_consensus_including_zero_without_future_model_fields():
    for speeds,expected in [((.25,.251,.249),.25),((0.,0.,0.),0.),((-.01,0.,.01),0.)]:
        measured=MiningLiveRecognizer._landmark_scroll(scroll_entities(speeds))
        assert measured["current_forward_speed_observed"]
        assert measured["current_forward_speed"]==pytest.approx(expected)
        assert measured["current_forward_speed_units"]=="viewport_heights_per_second"
        assert measured["current_forward_speed_source"]=="tracked_landmark_consensus"
        assert "forward_speed" not in measured


@pytest.mark.parametrize("problem",["too_few","young","body","moving_sideways","disagreement",
                                      "duplicate_id","duplicate_center","nonfinite"])
def test_landmark_scroll_rejects_insufficient_or_nonindependent_evidence(problem):
    from dataclasses import replace
    entities=scroll_entities()
    if problem=="too_few":entities.pop()
    elif problem=="young":entities[-1]=replace(entities[-1],samples=3)
    elif problem=="body":entities[-1]=replace(entities[-1],kind="player")
    elif problem=="moving_sideways":entities[-1]=replace(entities[-1],vx=.1)
    elif problem=="disagreement":entities[-1]=replace(entities[-1],vy=-.6)
    elif problem=="duplicate_id":entities[-1]=replace(entities[-1],track_id=entities[0].track_id)
    elif problem=="duplicate_center":entities[-1]=replace(entities[-1],x=entities[0].x,y=entities[0].y)
    else:entities[-1]=replace(entities[-1],vy=float("nan"))
    measured=MiningLiveRecognizer._landmark_scroll(entities)
    assert not measured["current_forward_speed_observed"]
    assert "current_forward_speed" not in measured


def test_landmark_scroll_opt_in_is_current_only_and_does_not_persist_after_loss(monkeypatch):
    entities=scroll_entities()
    for enabled in (False,True):
        r=recognizer(estimate_landmark_scroll=enabled)
        monkeypatch.setattr(r.tracker,"update",lambda found,t:[e for e in found if e.kind=="player"]+entities)
        first=r.observe(packet(scene()))
        assert first.metrics["current_forward_speed_observed"] is enabled
        assert not first.metrics["forward_speed_observed"] and "forward_speed" not in first.metrics
        monkeypatch.setattr(r.tracker,"update",lambda found,t:[e for e in found if e.kind=="player"]+entities[:2])
        missing=r.observe(packet(scene(),1.1,2))
        assert not missing.metrics["current_forward_speed_observed"]
        assert "current_forward_speed" not in missing.metrics


def paired_scroll_tracks(entities,previous=1.,current=1.1,speed=.35):
    from collections import deque
    return {e.track_id:(e,current,deque([(previous,e.x,e.y+speed*(current-previous)),
                                       (current,e.x,e.y)],maxlen=5)) for e in entities}


def test_pair_scroll_uses_actual_positions_and_reports_measurement_midpoint():
    entities=scroll_entities((.10,.10,.10))  # older tracker window differs
    tracks=paired_scroll_tracks(entities,speed=.35)
    actual=MiningLiveRecognizer._landmark_scroll(entities,tracks=tracks,timestamp=1.1,method="paired_positions")
    assert actual["current_forward_speed_observed"]
    assert actual["current_forward_speed"]==pytest.approx(.35)
    assert actual["current_forward_speed_reference_timestamp"]==pytest.approx(1.05)
    assert actual["current_forward_speed_window_seconds"]==pytest.approx(.1)
    assert actual["current_forward_speed_reference_kind"]=="pair_midpoint_velocity_approximation"
    legacy=MiningLiveRecognizer._landmark_scroll(entities,tracks=tracks,timestamp=1.1)
    assert legacy["current_forward_speed"]==pytest.approx(.1)
    assert legacy["current_forward_speed_reference_kind"]=="median_tracker_window_midpoint_approximation"
    stopped=MiningLiveRecognizer._landmark_scroll(entities,tracks=paired_scroll_tracks(entities,speed=0.),
                                                 timestamp=1.1,method="paired_positions")
    assert stopped["current_forward_speed_observed"] and stopped["current_forward_speed"]==0.


@pytest.mark.parametrize("problem",["missing_history","stale_current","short_window","long_window","different_windows"])
def test_pair_scroll_never_substitutes_an_old_or_mismatched_measurement(problem):
    entities=scroll_entities();tracks=paired_scroll_tracks(entities)
    if problem=="missing_history":tracks={}
    elif problem=="stale_current":tracks=paired_scroll_tracks(entities,previous=.9,current=1.)
    elif problem=="short_window":tracks=paired_scroll_tracks(entities,previous=1.09)
    elif problem=="long_window":tracks=paired_scroll_tracks(entities,previous=.8)
    else:tracks.update(paired_scroll_tracks(entities[-1:],previous=1.01))
    result=MiningLiveRecognizer._landmark_scroll(entities,tracks=tracks,timestamp=1.1,method="paired_positions")
    assert not result["current_forward_speed_observed"] and "current_forward_speed" not in result


def sparse_history(identity=1,x=.4,y=.5,speed=.3,samples=8):
    from raiders_helper.models import Entity
    shape={"source":"current_compact_contour","complete":True}
    return [(t,Entity("obstacle",x,y+speed*(1.3-t),.08,.10,.82,
                     track_id=identity,samples=samples-3+i),shape.copy())
            for i,t in enumerate((1.,1.1,1.2,1.3))]


def test_sparse_actual_window_reference_and_explicit_evidence_sources():
    single=sparse_history();entity=single[-1][1]
    result=MiningLiveRecognizer._sparse_landmark_scroll([entity],{1:single},1.3)
    assert result["current_forward_speed_observed"]
    assert result["current_forward_speed_source"]=="temporal_single_landmark"
    assert result["current_forward_speed_method"]=="temporal_positions"
    assert result["current_forward_speed"]==pytest.approx(.3)
    assert result["current_forward_speed_reference_timestamp"]==pytest.approx(1.15)
    assert result["current_forward_speed_window_seconds"]==pytest.approx(.3)
    second=sparse_history(2,x=.6,y=.7,samples=6)
    dual=MiningLiveRecognizer._sparse_landmark_scroll([entity,second[-1][1]],{1:single,2:second},1.3)
    assert dual["current_forward_speed_source"]=="dual_landmark_consensus"
    assert dual["current_forward_speed_reference_timestamp"]==pytest.approx(1.25)
    assert dual["current_forward_speed_window_seconds"]==pytest.approx(.1)
    stopped=sparse_history(speed=0.)
    zero=MiningLiveRecognizer._sparse_landmark_scroll([stopped[-1][1]],{1:stopped},1.3)
    assert zero["current_forward_speed_observed"] and zero["current_forward_speed"]==0.


@pytest.mark.parametrize("problem",["young","shape_missing","stale","absent","size_jump","sideways","cropped","position_jump","nonfinite"])
def test_sparse_velocity_rejects_weak_or_noncurrent_landmark(problem):
    from dataclasses import replace
    h=sparse_history();t=1.3
    if problem=="young":h=sparse_history(samples=7)
    elif problem=="shape_missing":h[-1]=(h[-1][0],h[-1][1],{})
    elif problem=="stale":t=1.4
    elif problem=="size_jump":h[-1]=(h[-1][0],replace(h[-1][1],w=.04),h[-1][2])
    elif problem=="sideways":h[-1]=(h[-1][0],replace(h[-1][1],x=.42),h[-1][2])
    elif problem=="cropped":h=sparse_history(y=.89)
    elif problem=="position_jump":h[-1]=(h[-1][0],replace(h[-1][1],y=.40),h[-1][2])
    elif problem=="nonfinite":h[-1]=(h[-1][0],replace(h[-1][1],x=float("nan")),h[-1][2])
    entities=[] if problem=="absent" else [h[-1][1]]
    result=MiningLiveRecognizer._sparse_landmark_scroll(entities,{1:h},t)
    assert not result["current_forward_speed_observed"] and "current_forward_speed" not in result


def test_disagreeing_two_landmarks_never_degrade_to_one_favorable_vote():
    a=sparse_history();b=sparse_history(2,x=.6,y=.7,speed=.5)
    result=MiningLiveRecognizer._sparse_landmark_scroll([a[-1][1],b[-1][1]],{1:a,2:b},1.3)
    assert not result["current_forward_speed_observed"]


def test_shape_history_is_only_actual_current_dimensions_and_resets_on_bad_shape():
    r=recognizer(sparse_landmark_scroll=True);h=sparse_history()
    for t,e,shape in h:
        r._landmark_shapes={r._landmark_key(e):shape}
        r._update_landmark_measurements([e],t)
    assert len(r._landmark_measurements[1])==4
    r._landmark_shapes={}
    r._update_landmark_measurements([h[-1][1]],1.4)
    assert 1 not in r._landmark_measurements


def test_real_accelerating_star_pair_has_explicit_reference_time():
    from collections import deque
    from raiders_helper.models import Entity
    path=ROOT/"artifacts/mining-A-campaign/batch-02/round-0001/gameplay/trace.jsonl"
    if not path.exists():pytest.skip("local live raw-position evidence not bundled")
    source={o["sequence"]:o for line in path.read_text(encoding="utf8").splitlines()
            for o in [json.loads(line)["observation"]] if o["sequence"] in (4399,4403)}
    previous,current=source[4399],source[4403]
    old={e["track_id"]:e for e in previous["entities"]}
    entities=[Entity(**e) for e in current["entities"]]
    tracks={e.track_id:(e,current["timestamp"],deque([(previous["timestamp"],old[e.track_id]["x"],old[e.track_id]["y"]),
            (current["timestamp"],e.x,e.y)],maxlen=5)) for e in entities if e.track_id in old}
    result=MiningLiveRecognizer._landmark_scroll(entities,tracks=tracks,timestamp=current["timestamp"],method="paired_positions")
    assert result["current_forward_speed_observed"] and result["current_forward_speed_track_count"]==6
    expected=np.median([(old[e.track_id]["y"]-e.y)/(current["timestamp"]-previous["timestamp"])
                        for e in entities if e.kind=="time_star"])
    assert result["current_forward_speed"]==pytest.approx(expected)
    assert result["current_forward_speed_reference_timestamp"]==pytest.approx((current["timestamp"]+previous["timestamp"])/2)


def test_actual_six_star_consensus_keeps_zr_ramp_separate_from_future_speed():
    path=ROOT/"artifacts/mining-A-campaign/batch-02/round-0001/gameplay/trace.jsonl"
    if not path.exists():pytest.skip("local live landmark evidence not bundled")
    from raiders_helper.models import Entity
    source=next(json.loads(line)["observation"] for line in path.read_text(encoding="utf8").splitlines()
                if json.loads(line)["observation"]["sequence"]==4403)
    measured=MiningLiveRecognizer._landmark_scroll([Entity(**entity) for entity in source["entities"]])
    assert measured["current_forward_speed_observed"] and measured["current_forward_speed_track_count"]==6
    assert measured["current_forward_speed"]==pytest.approx(.3542842890953499)
    assert "forward_speed" not in measured


@pytest.mark.parametrize("sequence",[3277,3281,3286])
def test_observed_fast_left_turn_wake_accepts_explicit_2_5_width_profile(sequence):
    path=ROOT/f"artifacts/mining-A-campaign/batch-02/round-0001/gameplay/contact-windows/window-02/{sequence:08d}.png"
    if not path.exists():pytest.skip("local fast-turn wake evidence not bundled")
    frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    profile={"stable_frames":1,"phase_anchors":False,"floor_backend":"adaptive_inner"}
    before=MiningLiveRecognizer(profile=profile).observe(packet(frame))
    after=MiningLiveRecognizer(profile={**profile,"wake_max_body_widths":2.5}).observe(packet(frame))
    p,=after.of_kind("player")
    assert p==before.of_kind("player")[0]
    rows=[row for row in after.metrics["road_rows"] if p.y-p.h/2-.001<=row[0]<=p.y+p.h/2+.001]
    assert all(left<p.x-p.w/2 and right>p.x+p.w/2 for _,left,right in rows)
    assert [row for row in before.metrics["road_rows"] if row[0]>p.y+p.h/2]==[
        row for row in after.metrics["road_rows"] if row[0]>p.y+p.h/2]


def test_explicit_2_5_wake_width_still_rejects_forward_divider_and_160_pixel_rear_gap():
    for x0,x1,bottom in [(329,385,270),(240,400,85)]:
        frame=scene();frame[35:bottom,x0:x1]=0
        obs=recognizer(wake_max_body_widths=2.5).observe(packet(frame))
        assert not obs.metrics["wake_repaired_rows"]


@pytest.mark.parametrize("hue",[9,27,45,93])
def test_adaptive_floor_follows_changed_hue_and_excludes_gold_wall(hue):
    frame=scene(False)
    floor=cv2.cvtColor(np.uint8([[[hue,235,130]]]),cv2.COLOR_HSV2BGR)[0,0]
    wall=cv2.cvtColor(np.uint8([[[25,150,150]]]),cv2.COLOR_HSV2BGR)[0,0]
    frame[35:]=0;frame[35:,100:540]=wall;frame[35:,120:520]=floor
    cv2.rectangle(frame,(303,76),(336,108),(30,220,40),-1)
    cv2.rectangle(frame,(310,87),(329,97),(210,45,20),-1)
    frame[120:300,440:540]=0  # genuine forward wall indentation stays hard
    obs=recognizer(floor_backend="adaptive_inner").observe(packet(frame))
    assert obs.of_kind("player")
    rows=obs.metrics["road_rows"]
    assert all(left>=119/640 and right<=521/640 for _,left,right in rows)
    assert all(right<=441/640 for y,left,right in rows if .27<y<.58)
    assert not obs.metrics["road_outer_envelope_only"]


def video_mining_frame(seconds):
    path=ROOT/f"artifacts/reference-video-index/mining-source-{seconds:06.1f}s.png"
    if not path.exists():pytest.skip("local user-provided all-S video frames not bundled")
    raw=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    # Source-indexed visual crop, +-2 source pixels; not physical calibration.
    return cv2.resize(raw[26:454,142:710],(640,480))


def test_local_video_floor_palette_changes_without_losing_main_road():
    for sec,expected_hue in [(30,27),(90,45),(150,93),(210,93),(270,9),(300,27)]:
        obs=recognizer(floor_backend="adaptive_inner").observe(packet(video_mining_frame(sec)))
        assert obs.metrics["floor_palette"]["hue"]==expected_hue
        middle=[row for row in obs.metrics["road_rows"] if abs(row[0]-.5)<.001]
        assert middle and middle[0][2]-middle[0][1]>.25,(sec,middle)
        assert len(obs.metrics["road_rows"])>=12


def test_local_cyan_video_floor_keeps_real_body_and_six_separate_time_stars():
    obs=recognizer(floor_backend="adaptive_inner",detect_time_stars=True).observe(packet(video_mining_frame(210)))
    assert obs.phase=="playing" and len(obs.of_kind("player"))==1
    stars=obs.of_kind("time_star")
    assert len(stars)==6
    for x in (454,494):
        for y in (362,402,442):
            assert any(abs(e.x*640-x)<5 and abs(e.y*480-y)<5 for e in stars)
    assert obs.metrics["player_minimum_blue_hue"]==105
    assert not recognizer().observe(packet(video_mining_frame(210))).of_kind("time_star")


def test_real_time_star_glyph_at_body_height_does_not_acquire_player_identity():
    frame=video_mining_frame(210)
    glyph=frame[354:372,442:467].copy()
    floor=cv2.cvtColor(np.uint8([[[93,216,110]]]),cv2.COLOR_HSV2BGR)[0,0]
    frame[65:125,230:295]=floor  # remove the actual robot, preserving star evidence
    frame[82:100,310:335]=glyph
    obs=recognizer(floor_backend="adaptive_inner",detect_time_stars=True).observe(packet(frame))
    assert not obs.of_kind("player") and obs.metrics["player_candidates"]==0
    assert len(obs.of_kind("time_star"))==6


def test_real_ns_six_time_stars_have_distinct_positions_and_not_player_identity():
    path=ROOT/"artifacts/mining-supervision/round-05/frame-00002163.jpg"
    if not path.exists():pytest.skip("local NS star evidence not bundled")
    raw=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    obs=MiningLiveRecognizer(profile={"stable_frames":1,"phase_anchors":False,
          "floor_backend":"adaptive_inner","detect_time_stars":True}).observe(packet(raw))
    assert len(obs.of_kind("player"))==1 and len(obs.of_kind("time_star"))==6
    assert obs.of_kind("player")[0].x>.7
    for x in (258,298):
        for y in (294,334,374):
            assert any(abs(e.x*640-x)<5 and abs(e.y*480-y)<5 for e in obs.of_kind("time_star"))


def powered_source_frame(index):
    path=ROOT/f"artifacts/mining-s-video-study/state-pairs/frame-{index:06d}.png"
    if not path.exists():pytest.skip("local source-indexed powered transition not bundled")
    raw=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    return cv2.resize(raw[26:454,142:710],(640,480))


@pytest.mark.parametrize("before,after",[(2092,2100),(4328,4335),(5730,5738),(8078,8085)])
def test_real_blue_to_powered_keeps_actual_player_and_separates_motion_mode(before,after):
    r=recognizer(floor_backend="adaptive_inner")
    first=r.observe(packet(powered_source_frame(before),before/30,before))
    assert first.of_kind("player") and first.metrics["motion_mode"]=="normal"
    current=r.observe(packet(powered_source_frame(after),after/30,after))
    p,=current.of_kind("player")
    assert current.phase=="playing" and current.metrics["buff_active"]
    assert current.metrics["player_appearance"]=="powered" and not current.metrics["blue_observed"]
    assert current.metrics["motion_mode"]=="powered" and p.w*640>=25 and p.h*480>=30
    # Current pose is measured, but a first green screenshot has no identity.
    alone=recognizer(floor_backend="adaptive_inner").observe(packet(powered_source_frame(after),after/30,after))
    assert not alone.of_kind("player") and not alone.metrics["buff_active"]


def test_powered_continuation_requires_actual_body_and_recent_identity():
    r=recognizer(floor_backend="adaptive_inner")
    r.observe(packet(powered_source_frame(2092),1,1))
    green=powered_source_frame(2100)
    powered=r.observe(packet(green,1.2,2))
    assert powered.metrics["buff_active"]
    assert r.observe(packet(green,1.3,3)).metrics["buff_active"]
    stale=r.observe(packet(green,1.7,4))
    assert not stale.of_kind("player") and not stale.metrics["buff_active"]
    # An arms-only appearance is never a synthetic continuation of the body.
    r=recognizer(floor_backend="adaptive_inner")
    r.observe(packet(powered_source_frame(2092),1,1));r.observe(packet(green,1.2,2))
    arms=green.copy();arms[70:125,345:410]=(30,80,140)
    cv2.rectangle(arms,(355,95),(365,99),(30,220,90),-1)
    cv2.rectangle(arms,(387,95),(397,99),(30,220,90),-1)
    missing=r.observe(packet(arms,1.3,3))
    assert not missing.of_kind("player") and not missing.metrics["buff_active"]


def test_real_normal_ns_bodies_do_not_become_powered_after_identity_is_available():
    directory=ROOT/"artifacts/mining-supervision/round-06"
    paths=sorted(directory.glob("problem-*.png"))
    if len(paths)!=8:pytest.skip("local raw body counterexamples not bundled")
    for path in paths:
        raw=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
        r=MiningLiveRecognizer(profile={"stable_frames":1,"phase_anchors":False,"floor_backend":"adaptive_inner"})
        r.observe(packet(raw,1,1));obs=r.observe(packet(raw,1.06,2))
        assert not obs.metrics["buff_active"] and obs.metrics["powered_body_candidates"]==0


def test_video_ninety_second_large_boulder_has_its_own_explicit_scale():
    frame=video_mining_frame(90)
    baseline=recognizer(floor_backend="adaptive_inner").observe(packet(frame))
    assert not baseline.of_kind("obstacle")
    obs=recognizer(floor_backend="adaptive_inner",large_rocks=True).observe(packet(frame))
    stone,=obs.of_kind("obstacle")
    assert abs(stone.x*640-342)<8 and abs(stone.y*480-246)<8
    assert stone.w*640>150 and stone.h*480>150
    assert obs.metrics["large_rock_added"]==1
    assert obs.metrics["large_rock_evidence"][0]["rim_sectors"]==12
    assert not obs.metrics["circle_rocks_enabled"]


def test_large_ring_on_dark_floor_is_found_without_relaxing_dark_blob_shape():
    frame=scene()
    frame[200:,100:540]=(15,30,65)
    cv2.circle(frame,(360,320),74,(10,15,30),-1)
    cv2.circle(frame,(360,320),75,(40,125,180),5)
    frame[385:400,357:364]=(15,30,65)
    baseline=recognizer().observe(packet(frame))
    assert not baseline.of_kind("obstacle")
    obs=recognizer(large_rocks=True).observe(packet(frame))
    stone,=obs.of_kind("obstacle")
    assert abs(stone.x*640-360)<8 and abs(stone.y*480-320)<8
    assert stone.w*640>135 and obs.metrics["large_rock_added"]==1


def test_large_ring_experiment_does_not_add_rocks_in_eight_raw_problem_frames():
    directory=ROOT/"artifacts/mining-supervision/round-06"
    paths=sorted(directory.glob("problem-*.png"))
    if len(paths)!=8:pytest.skip("local raw giant-rock counterexamples not bundled")
    for path in paths:
        frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
        profile={"stable_frames":1,"phase_anchors":False,"floor_backend":"adaptive_inner"}
        before=MiningLiveRecognizer(profile=profile).observe(packet(frame))
        after=MiningLiveRecognizer(profile={**profile,"large_rocks":True}).observe(packet(frame))
        assert after.metrics["large_rock_added"]==0,path.name
        assert after.of_kind("obstacle")==before.of_kind("obstacle"),path.name


def test_large_scale_rejects_star_groups_and_a_circle_outside_the_road():
    stars=recognizer(floor_backend="adaptive_inner",large_rocks=True,detect_time_stars=True).observe(packet(video_mining_frame(210)))
    assert stars.metrics["large_rock_added"]==0 and len(stars.of_kind("time_star"))==6
    frame=scene();cv2.circle(frame,(50,300),75,(180,180,180),5)
    obs=recognizer(large_rocks=True).observe(packet(frame))
    assert obs.metrics["large_rock_added"]==0


@pytest.mark.parametrize("sequence",[1947,2150])
def test_r8_visible_turning_sprite_does_not_become_wall_or_change_physics_bbox(sequence):
    directory=ROOT/"artifacts/mining-supervision/round-08-adaptive"
    paths=list(directory.glob(f"problem-{sequence:08d}-*.png"))
    if not paths:pytest.skip("local raw R8 turning-sprite evidence not bundled")
    source=next(json.loads(line)["observation"] for line in (directory/"trace.jsonl").read_text(encoding="utf8").splitlines()
                if json.loads(line)["observation"]["sequence"]==sequence)
    old=next(e for e in source["entities"] if e["kind"]=="player")
    raw=cv2.imdecode(np.frombuffer(paths[0].read_bytes(),np.uint8),1)
    obs=MiningLiveRecognizer(profile={"stable_frames":1,"phase_anchors":False,"floor_backend":"adaptive_inner"}).observe(packet(raw))
    p,=obs.of_kind("player")
    assert [p.x,p.y,p.w,p.h]==[old[k] for k in ("x","y","w","h")]
    assert obs.metrics["sprite_occlusion_box"][2]>p.w
    assert obs.metrics["sprite_observed_pixels"]>0
    rows=[r for r in obs.metrics["road_rows"] if p.y-p.h/2-.001<=r[0]<=p.y+p.h/2+.001]
    assert rows and all(left<p.x-p.w/2 and right>p.x+p.w/2 for _,left,right in rows)


def test_sprite_occlusion_preserves_gold_wall_pixels_inside_body_rectangle():
    frame=scene()
    gold=cv2.cvtColor(np.uint8([[[30,150,180]]]),cv2.COLOR_HSV2BGR)[0,0]
    frame[78:108,309:313]=gold
    r=recognizer(floor_backend="adaptive_inner")
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    p=r._player(hsv,1)[0]
    _,visible=r._sprite_occlusion(hsv,p,{"hue":15,"saturation":200},95)
    assert visible[90,310]==0
    rows,_=r._floor_inner_edges(hsv,[[90/480,300/640,540/640]],{"hue":15,"saturation":200},p,visible)
    assert rows[0][1]>=315/640


@pytest.mark.parametrize("directory,sequence",[
    ("mining-hot-live-validation-01/round-0001/gameplay",6529),
    ("mining-hot-live-validation-01/round-0001/gameplay",6824),
    ("mining-A-campaign/batch-01/round-0001/gameplay",16116),
    ("mining-A-campaign/batch-01/round-0001/gameplay",16371),
    ("mining-A-campaign/batch-01/round-0002/gameplay",19177),
    ("mining-A-campaign/batch-01/round-0002/gameplay",19512),
])
def test_real_gold_feet_do_not_cut_a_fake_wall_through_visible_body(directory,sequence):
    directory=ROOT/"artifacts"/directory
    paths=list(directory.glob(f"problem-{sequence:08d}-*.png"))
    if not paths:pytest.skip("local gold-foot evidence not bundled")
    source=next(json.loads(line)["observation"] for line in (directory/"trace.jsonl").read_text(encoding="utf8").splitlines()
                if json.loads(line)["observation"]["sequence"]==sequence)
    old=next(e for e in source["entities"] if e["kind"]=="player")
    frame=cv2.imdecode(np.frombuffer(paths[0].read_bytes(),np.uint8),1)
    obs=MiningLiveRecognizer(profile={"stable_frames":1,"phase_anchors":False,
                                      "floor_backend":"adaptive_inner"}).observe(packet(frame))
    player,=obs.of_kind("player")
    assert [player.x,player.y,player.w,player.h]==[old[k] for k in ("x","y","w","h")]
    body_rows=[row for row in obs.metrics["road_rows"]
               if player.y-player.h/2-.001<=row[0]<=player.y+player.h/2+.001]
    assert body_rows and all(left<player.x-player.w/2 and right>player.x+player.w/2
                             for _,left,right in body_rows)
    front=player.y+player.h/2+.02
    assert [row for row in obs.metrics["road_rows"] if row[0]>front]==[
        row for row in source["metrics"]["road_rows"] if row[0]>front]


def test_actual_left_wall_contact_remains_hard_after_sprite_occlusion_repair():
    path=ROOT/"artifacts/mining-supervision/round-06/problem-00039385-wall_recovery.png"
    if not path.exists():pytest.skip("local actual wall-contact counterexample not bundled")
    frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    obs=MiningLiveRecognizer(profile={"stable_frames":1,"phase_anchors":False,"floor_backend":"adaptive_inner"}).observe(packet(frame))
    p,=obs.of_kind("player")
    rows=[r for r in obs.metrics["road_rows"] if p.y-p.h/2-.001<=r[0]<=p.y+p.h/2+.001]
    assert any(left>=p.x-p.w/2 for _,left,right in rows)


def test_real_result_screen_is_rejected_when_optional_source_is_present():
    # Local source QA is kept separate from synthetic accuracy claims.
    path=ROOT/"artifacts/supervised-mining-input/retry-before-start.png"
    if not path.exists():pytest.skip("local NS evidence not bundled")
    frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    r=MiningLiveRecognizer(profile={"stable_frames":1})
    assert r.observe(packet(frame)).phase=="result"
