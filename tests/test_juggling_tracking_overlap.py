from raiders_helper.fast_tracking import PredictiveTracker
from raiders_helper.models import Entity


def ball(x,y,w,h):return Entity('ball',x,y,w,h,.84)


def test_actual_overlap_shape_change_cannot_reassign_large_fish_to_small_fish():
    # Actual boxes from S-batch-01 seq132205 ->132207, normalized to court.
    tracker=PredictiveTracker(ball_area_ratio_limit=2.6)
    first=tracker.update([ball(.629,.429,.192,.167),ball(.530,.527,.117,.121)],10)
    large_id=first[0].track_id
    after=tracker.update([ball(.695,.496,.072,.092),ball(.567,.464,.175,.223)],10.033)
    assert after[0].track_id!=large_id
    assert after[0].samples==1
    assert after[1].track_id==large_id


def test_small_midair_pose_jitter_does_not_erase_velocity_window():
    tracker=PredictiveTracker(physical_ball_bounces=True)
    for i,x in enumerate((.25,.254,.2505,.2545,.251)):
        result=tracker.update([ball(x,.3+.01*i,.07,.08)],10+.04*i)
    assert tracker.velocity_windows[str(result[0].track_id)]>.15
    assert abs(result[0].vx)<.02


def test_observed_catch_reversal_near_floor_resets_velocity():
    tracker=PredictiveTracker(physical_ball_bounces=True)
    for i,y in enumerate((.75,.79,.83,.76)):
        result=tracker.update([ball(.25,y,.07,.08)],10+.04*i)
    assert result[0].vy<0
    assert tracker.velocity_windows[str(result[0].track_id)]<.05
