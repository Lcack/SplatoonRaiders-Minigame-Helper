"""Geometry identities and observation contracts; radius is not a game-law claim."""
from dataclasses import replace
import math
import random
import json
from pathlib import Path

import pytest

from raiders_helper.mining_collision import (ObstacleGeometry,rounded_rectangle_entry,
    circle_rectangle_area,segment_rectangle_distance)
from raiders_helper.mining_planner import contact_fraction
from raiders_helper.models import Entity,Observation,Game
from raiders_helper.mining_reactive import decide_mining_reactive,_prefix_rock_contacts
from raiders_helper.mining_planner import Road


def world(**packet_changes):
    entity=Entity('obstacle',.5,.5,46/640,46/480,.95,7,samples=8)
    packet={'track_id':7,'shape':'circle','center_x_pixels':320.,'center_y_pixels':240.,'radius_pixels':20.,
        'coordinate_space':'canonical_640x480_pixels','source':'current_bright_ring','timestamp':1.,
        'rim_sectors':12,'rim_contrast':30.,**packet_changes}
    return Observation(Game.MINING,1.,1,'playing',.95,(entity,),metrics={'obstacle_shapes':[packet]}),entity


def test_exact_rounded_rectangle_sweep_and_corner_relief():
    assert rounded_rectangle_entry(-20,0,20,0,3,4,10)==pytest.approx(.175)
    assert rounded_rectangle_entry(-20,14,20,14,3,4,10)==pytest.approx(.425)
    assert rounded_rectangle_entry(-20,14.01,20,14.01,3,4,10) is None
    assert rounded_rectangle_entry(12,13,12,13,3,4,10) is None
    assert rounded_rectangle_entry(3,4,3,4,3,4,10)==0


def test_disk_rectangle_overlap_exact_areas():
    assert circle_rectangle_area(-20,20,-20,20,10)==pytest.approx(math.pi*100)
    assert circle_rectangle_area(0,20,0,20,10)==pytest.approx(math.pi*25)
    assert circle_rectangle_area(-2,3,-4,5,10)==pytest.approx(45.)
    assert circle_rectangle_area(10,12,-1,1,10)==0
    assert circle_rectangle_area(8,12,8,12,10)==0


def test_circle_overlap_agrees_with_independent_dense_vertical_integration():
    rng=random.Random(83)
    for _ in range(20):
        left,right=sorted(rng.uniform(-13,13) for _ in range(2))
        top,bottom=sorted(rng.uniform(-13,13) for _ in range(2))
        n=10000;dx=(right-left)/n;area=0.
        for i in range(n):
            x=left+(i+.5)*dx
            if abs(x)>=10:continue
            height=math.sqrt(100-x*x)
            area+=max(0.,min(bottom,height)-max(top,-height))*dx
        assert circle_rectangle_area(left,right,top,bottom,10)==pytest.approx(area,abs=.002)


def test_segment_rectangle_clearance_is_continuous_and_corner_exact():
    assert segment_rectangle_distance(-20,0,20,0,3,4)==0
    assert segment_rectangle_distance(-20,14,20,14,3,4)==pytest.approx(10)
    assert segment_rectangle_distance(12,13,12,13,3,4)==pytest.approx(math.sqrt(162))


@pytest.mark.parametrize('change',[{'timestamp':.9},{'source':'predicted_ring'},
    {'coordinate_space':'normalized'},{'rim_sectors':8},{'rim_contrast':14},
    {'radius_pixels':40},{'center_x_pixels':329},{'shape':'box'}])
def test_missing_or_unmatched_current_ring_uses_original_aabb(change):
    obs,entity=world(**change);geometry=ObstacleGeometry(obs,{'circle_collision':{'enabled':True}})
    assert geometry.shape(entity) is None
    for segment in ((.4,.5,.6,.5),(.4,.4,.6,.6),(.1,.1,.2,.2)):
        assert geometry.contact_fraction(*segment,entity,.02,.03)==contact_fraction(*segment,entity,.02,.03)


def test_current_ring_preserves_existing_bbox_axial_extents_and_player_envelope():
    obs,entity=world(center_x_pixels=322.)
    geometry=ObstacleGeometry(obs,{'circle_collision':{'enabled':True,'extra_radius_pixels':0}})
    circle=geometry.shape(entity)
    assert circle.radius_pixels==25.
    assert geometry.bounds(entity)[0]<=entity.x-entity.w/2
    assert geometry.bounds(entity)[1]>=entity.x+entity.w/2
    expected=(circle.center_x_pixels-circle.radius_pixels-8)/640
    fraction=geometry.contact_fraction(.3,.5,.6,.5,entity,8/640,11/480)
    assert .3+(.6-.3)*fraction==pytest.approx(expected)


def test_duplicate_shape_or_entity_identity_is_ambiguous_and_not_cached():
    obs,entity=world();obs.metrics['obstacle_shapes']*=2
    assert not ObstacleGeometry(obs,{'circle_collision':{'enabled':True}}).circles
    clean,entity=world();ambiguous=replace(clean,entities=(entity,entity))
    assert not ObstacleGeometry(ambiguous,{'circle_collision':{'enabled':True}}).circles
    next_frame=replace(clean,timestamp=1.1)
    assert not ObstacleGeometry(next_frame,{'circle_collision':{'enabled':True}}).circles


def test_circle_area_current_contact_and_sweep_use_identical_geometry():
    obs,entity=world();geometry=ObstacleGeometry(obs,{'circle_collision':{'enabled':True}})
    # Original padded AABB corner overlaps, but the exact circle does not.
    x=(320+26)/640;y=(240+26)/480;rx=4/640;ry=4/480
    assert contact_fraction(x,y,x,y,entity,rx,ry)==0
    assert geometry.contact_fraction(x,y,x,y,entity,rx,ry) is None
    assert geometry.overlap_area(x,y,entity,rx,ry)==0
    assert geometry.segment_clearance(x,y,x,y,entity,rx,ry)>0
    assert geometry.overlap_area(.5,.5,entity,rx,ry)>0
    assert geometry.contact_fraction(.5,.5,.5,.5,entity,rx,ry)==0


def test_matching_track_id_does_not_change_star_pickup_geometry():
    obs,entity=world();geometry=ObstacleGeometry(obs,{'circle_collision':{'enabled':True}})
    star=replace(entity,kind='time_star')
    assert geometry.shape(star) is None
    assert geometry.contact_fraction(.4,.5,.6,.5,star,.01,.01)==contact_fraction(.4,.5,.6,.5,star,.01,.01)


def test_prefix_and_current_contact_sets_share_the_circle_corner_geometry():
    obs,entity=world();geometry=ObstacleGeometry(obs,{'circle_collision':{'enabled':True}})
    player=Entity('player',(320+26)/640,(240+26)/480,8/640,8/480,.95)
    prefix=[(player.x,player.y,0.,0.),(player.x,player.y,.1,0.)]
    assert _prefix_rock_contacts(player,[entity],prefix,4/640,4/480)==({0},{0})
    assert _prefix_rock_contacts(player,[entity],prefix,4/640,4/480,geometry)==(set(),set())


def test_recorded_current_ring_guided_search_finds_a_complete_opposite_side_without_changing_hard_envelopes():
    fixture=json.loads((Path(__file__).parent/'fixtures/mining/b6-205935-current-ring.json').read_text(encoding='utf8'))
    raw=fixture['observation'];obs=Observation(**{**raw,'entities':tuple(Entity(**e) for e in raw['entities'])})
    params={**fixture['parameters'],'circle_collision':{'enabled':True,'extra_radius_pixels':0}}
    default=decide_mining_reactive(obs,params)
    assert default.diagnostics['route_status']=='partial'
    params['circle_guided_search']=True
    decision=decide_mining_reactive(obs,params);diag=decision.diagnostics
    assert diag['route_status']=='complete' and diag['mode']=='careful_turn'
    assert diag['guided_circle_search']['side']=='left'
    assert diag['guided_circle_search']['searches']==1
    assert decision.state.left_stick.x==0
    geometry=ObstacleGeometry(obs,params);road=Road(obs.of_kind('corridor'));player=obs.of_kind('player')[0]
    rx=player.w/2+params['wall_margin']+params['position_uncertainty'];ry=player.h/2+params['position_uncertainty']
    for a,b in zip(diag['path'],diag['path'][1:]):
        assert road.swept_safe(a['x'],a['depth'],b['x'],b['depth'],rx,ry)
        assert all(geometry.contact_fraction(a['x'],a['depth'],b['x'],b['depth'],e,rx,ry) is None for e in obs.of_kind('obstacle'))


def simple_world(*objects,x=.35):
    return Observation(Game.MINING,1.,1,'playing',.95,(Entity('player',x,.2,.04,.04,.95,1,samples=5),
        *[Entity('corridor',.5,y,.94,.001,.95) for y in (0,.15,.2,.4,.6,.8,1)],*objects))


def test_lateral_distance_cost_keeps_a_clear_offset_line_without_forcing_zero_when_obstacles_require_steering():
    params={'lookahead_depth':.6,'maximum_steps':30,'maximum_axis':.65,
        'center_attraction_weight':.05,'lateral_travel_weight':.04}
    clear=decide_mining_reactive(simple_world(),params)
    assert clear.diagnostics['route_status']=='complete'
    assert clear.state.left_stick.x==0
    assert all(p['x']==pytest.approx(.35) for p in clear.diagnostics['path'])
    obstacle=Entity('obstacle',.5,.5,.10,.10,.95,2)
    avoiding=decide_mining_reactive(simple_world(obstacle,x=.5),params)
    assert avoiding.diagnostics['route_status']=='complete'
    assert avoiding.state.left_stick.x!=0


def test_straightness_cost_preserves_a_worthwhile_star_detour():
    star=Entity('time_star',.55,.6,.035,.035,.95,3)
    params={'lookahead_depth':.6,'maximum_steps':30,'maximum_axis':.65,
        'center_attraction_weight':.05,'lateral_travel_weight':.04,'star_reward':.04}
    decision=decide_mining_reactive(simple_world(star),params)
    assert decision.diagnostics['route_status']=='complete'
    assert decision.state.left_stick.x>0
    assert len(decision.diagnostics['collected_predicted'])==1


@pytest.mark.parametrize('kind',['obstacle','time_star'])
def test_clear_only_straightness_restores_original_cost_for_forward_objective_or_hazard(kind):
    params={'lookahead_depth':.6,'maximum_steps':30,'star_reward':.04,
        'center_attraction_weight':.05,'lateral_travel_weight':.04,'straightness_clear_only':True}
    clear=decide_mining_reactive(simple_world(),params)
    assert clear.diagnostics['straightness_cost']['effective_center_weight']==.05
    assert clear.diagnostics['straightness_cost']['effective_lateral_travel_weight']==.04
    object=Entity(kind,.55,.6,.035,.035,.95,3)
    occupied=decide_mining_reactive(simple_world(object),params)
    assert occupied.diagnostics['straightness_cost']['effective_center_weight']==1.8
    assert occupied.diagnostics['straightness_cost']['effective_lateral_travel_weight']==0.
    baseline=decide_mining_reactive(simple_world(object),{**params,'center_attraction_weight':1.8,
        'lateral_travel_weight':0.,'straightness_clear_only':False})
    assert occupied.state==baseline.state
    behind=replace(object,y=.03)
    passed=decide_mining_reactive(simple_world(behind),params)
    assert passed.diagnostics['straightness_cost']['effective_center_weight']==.05
