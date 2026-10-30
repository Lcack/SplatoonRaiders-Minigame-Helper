import json
import math

import cv2
import numpy as np

from raiders_helper.control.buttons import Button,ControllerState
from raiders_helper.landing import LandingAtlas,decide_landing,estimate_pose,route_from_mask,terrain_mask
from raiders_helper.landing import _line_is_free
from raiders_helper.models import Entity,Game,Observation


def observation(angle=0,**metrics):
    return Observation(Game.LANDING,1,1,"playing",0.95,
                       (Entity("player",0.3,0.5,0.04,0.05,0.95,vx=0,vy=0,samples=3),),
                       metrics={"vehicle_angle_radians":angle,"pose_confidence":0.95,
                                "map_index":0,"map_match_score":0.95,**metrics})


def test_unknown_map_or_pose_holds_input():
    assert decide_landing(observation(map_index=-1),{}).state==ControllerState.neutral()
    assert decide_landing(observation(pose_confidence=0),{}).state==ControllerState.neutral()
    assert decide_landing(observation(float("nan")),{}).state==ControllerState.neutral()


def test_clockwise_counterclockwise_and_aligned_thrust():
    parameters={"waypoints":[[0.6,0.2]]}
    assert decide_landing(observation(),parameters).state.buttons==frozenset({Button.ZL})
    assert decide_landing(observation(),{"waypoints":[[0.1,0.2]]}).state.buttons==frozenset({Button.ZR})
    assert decide_landing(observation(),{"waypoints":[[0.3,0.2]]}).state.buttons==frozenset({Button.ZL,Button.ZR})


def test_uncalibrated_suggestions_are_explicitly_experimental():
    decision=decide_landing(observation(),{"waypoints":[[0.6,0.2]]})
    assert decision.diagnostics["experimental"] is True
    assert decision.diagnostics["target"]==[0.6,0.2]


def test_astar_routes_around_terrain_and_refuses_solid_barrier():
    mask=np.zeros((480,640),np.uint8)
    mask[100:350,290:340]=255
    route=route_from_mask(mask,(0.2,0.4),(0.8,0.4))
    assert len(route)>=3
    assert any(y<0.2 or y>0.75 for _,y in route)
    mask[:,290:340]=255
    assert route_from_mask(mask,(0.2,0.4),(0.8,0.4))==[]


def test_atlas_matches_geometry_not_stage_number(tmp_path):
    manifests=[]
    for index in range(2):
        image=np.zeros((480,640,3),np.uint8)
        cv2.rectangle(image,(150+index*200,100),(190+index*200,400),(20,80,120),-1)
        mask=terrain_mask(image)
        (tmp_path/f"{index}.png").write_bytes(cv2.imencode(".png",mask)[1].tobytes())
        manifests.append({"id":f"layout-{index}","mask":f"{index}.png","goal_hint":[0.9,0.5]})
    (tmp_path/"index.json").write_text(json.dumps({"layouts":manifests}),encoding="utf-8")
    atlas=LandingAtlas(tmp_path/"index.json")
    image=np.zeros((480,640,3),np.uint8)
    cv2.rectangle(image,(350,100),(390,400),(20,80,120),-1)
    result=atlas.match(image)
    assert result["accepted"]
    assert result["map_index"]==1
    assert atlas.match(np.zeros_like(image))["accepted"] is False


def test_pose_does_not_invent_angle_for_blank_crop(tmp_path):
    reference=np.zeros((48,48,3),np.uint8)
    cv2.circle(reference,(24,24),8,(180,30,20),-1)
    cv2.rectangle(reference,(21,8),(27,18),(20,180,30),-1)
    path=tmp_path/"pose.png"
    path.write_bytes(cv2.imencode(".png",reference)[1].tobytes())
    player=Entity("player",0.5,0.5,0.1,0.1,0.9)
    result=estimate_pose(np.zeros((480,640,3),np.uint8),player,path)
    assert result["pose_known"] is False
    assert "vehicle_angle_radians" not in result


def test_pose_rotation_bank_uses_clockwise_positive_radians(tmp_path):
    reference=np.zeros((48,48,3),np.uint8)
    cv2.circle(reference,(24,24),8,(180,30,20),-1)
    cv2.rectangle(reference,(21,5),(27,16),(20,180,30),-1)
    path=tmp_path/"pose.png"
    path.write_bytes(cv2.imencode(".png",reference)[1].tobytes())
    rotated=cv2.warpAffine(reference,cv2.getRotationMatrix2D((24,24),-40,1),(48,48))
    image=np.zeros((480,640,3),np.uint8)
    image[216:264,296:344]=rotated
    result=estimate_pose(image,Entity("player",0.5,0.5,0.08,0.1,0.95),path)
    assert result["pose_known"]
    assert abs(result["vehicle_angle_radians"]-math.radians(40))<math.radians(15)


def test_nonfinite_map_score_is_not_a_valid_match():
    decision=decide_landing(observation(map_match_score=float("nan")),{"waypoints":[[0.6,0.2]]})
    assert decision.state==ControllerState.neutral()


def _atlas_with_mask(tmp_path, mask):
    (tmp_path/"mask.png").write_bytes(cv2.imencode(".png",mask)[1].tobytes())
    (tmp_path/"index.json").write_text(json.dumps({"layouts":[{
        "id":"synthetic-layout","mask":"mask.png","goal_hint":[0.85,0.4]
    }]}),encoding="utf-8")
    return LandingAtlas(tmp_path/"index.json")


def test_navigation_field_is_reused_across_starts_and_equivalent_clearance(tmp_path):
    mask=np.zeros((480,640),np.uint8)
    mask[100:350,290:340]=255
    atlas=_atlas_with_mask(tmp_path,mask)
    first=atlas.route(0,(0.2,0.4),clearance=0.0201)
    second=atlas.route(0,(0.2,0.6),clearance=0.0249)
    assert first and second
    assert len(atlas._navigation_grids)==1
    assert len(atlas._navigation_fields)==1
    assert first[0]!=second[0]
    # Independently inspect dense samples against the inflated blocked grid.
    grid=next(iter(atlas._navigation_grids.values())).grid
    for path in (first,second):
        for a,b in zip(path,path[1:]):
            xx=np.rint(np.linspace(a[0]*79,b[0]*79,1000)).astype(int)
            yy=np.rint(np.linspace(a[1]*59,b[1]*59,1000)).astype(int)
            assert not np.any(grid[yy,xx])


def test_clearance_quantization_rounds_up_instead_of_relaxing_collision_radius(tmp_path):
    mask=np.zeros((480,640),np.uint8)
    mask[100:350,290:340]=255
    atlas=_atlas_with_mask(tmp_path,mask)
    assert atlas.route(0,(0.2,0.4),clearance=0.025)
    assert atlas.route(0,(0.2,0.4),clearance=0.025001)
    assert set(atlas._navigation_grids)=={(0,2),(0,3)}
    smaller=atlas._navigation_grids[(0,2)].grid
    larger=atlas._navigation_grids[(0,3)].grid
    assert np.all(larger[smaller])
    assert np.count_nonzero(larger)>np.count_nonzero(smaller)


def test_navigation_goal_is_keyed_by_snapped_grid_cell(tmp_path):
    atlas=_atlas_with_mask(tmp_path,np.zeros((480,640),np.uint8))
    first=atlas.route(0,(0.2,0.4),(0.85,0.400))
    same_cell=atlas.route(0,(0.3,0.4),(0.85,0.401))
    assert first[-1]==same_cell[-1]
    assert len(atlas._navigation_fields)==1
    different_cell=atlas.route(0,(0.2,0.4),(0.85,0.42))
    assert different_cell[-1]!=first[-1]
    assert len(atlas._navigation_fields)==2


def test_reverse_distance_is_octile_optimal_on_an_empty_grid(tmp_path):
    atlas=_atlas_with_mask(tmp_path,np.zeros((480,640),np.uint8))
    start=(10/79,20/59)
    goal=(50/79,30/59)
    assert atlas.route(0,start,goal)
    field=next(iter(atlas._navigation_fields.values()))
    assert abs(field.distance[20*80+10]-(30+10*math.sqrt(2)))<1e-10


def test_navigation_reuses_unreachable_result_and_returned_routes_are_not_mutable_cache(tmp_path):
    mask=np.zeros((480,640),np.uint8)
    mask[:,290:340]=255
    atlas=_atlas_with_mask(tmp_path,mask)
    assert atlas.route(0,(0.2,0.4))==[]
    assert atlas.route(0,(0.2,0.6))==[]
    assert len(atlas._navigation_fields)==1
    reachable=atlas.route(0,(0.7,0.4))
    assert reachable
    reachable[0][0]=42
    assert all(0<=x<=1 and 0<=y<=1 for x,y in atlas.route(0,(0.7,0.4)))


def test_los_cannot_simplify_through_diagonal_blocked_corner():
    grid=np.zeros((4,4),bool)
    grid[0,1]=True
    assert not _line_is_free(grid,(0,0),(2,2))
    assert not _line_is_free(grid,(2,2),(0,0))
    grid[0,1]=False
    assert _line_is_free(grid,(0,0),(2,2))
