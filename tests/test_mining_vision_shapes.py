"""Current-rim geometry and local rear-hole perception regressions."""
from pathlib import Path
import importlib.util,json,sys
import cv2,numpy as np,pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from raiders_helper.models import Entity
from raiders_helper.capture.models import FramePacket
from raiders_helper.mining_vision import MiningLiveRecognizer as Recognizer

def rock(identity=42,x=.4,y=.5,w=56/640,h=56/480):
    return Entity('obstacle',x,y,w,h,.82,track_id=identity)

def rim(**values):
    return dict(x=.4,y=.5,radius_pixels=25.,rim_sectors=11,rim_contrast=32.,**values)

def test_shape_contract_final_identity_and_unambiguous_pixel_units():
    shapes=Recognizer._current_obstacle_shapes([rock()],[rim()],12.3)
    assert len(shapes)==1
    shape=shapes[0]
    assert shape['track_id']==42 and shape['center_x_pixels']==256 and shape['center_y_pixels']==240
    assert shape['radius_pixels']==25 and shape['timestamp']==12.3
    assert shape['coordinate_space']=='canonical_640x480_pixels'
    assert shape['source']=='current_bright_ring'

@pytest.mark.parametrize('problem',['no_current_obstacle','no_current_rim','ambiguous_obstacles','two_rims','far_center','wrong_size','weak_rim','nonfinite'])
def test_uncertain_current_ring_association_does_not_publish_shape(problem):
    entities=[rock()];evidence=[rim()]
    if problem=='no_current_obstacle':entities=[]
    elif problem=='no_current_rim':evidence=[]
    elif problem=='ambiguous_obstacles':entities.append(rock(identity=43))
    elif problem=='two_rims':evidence.append(rim())
    elif problem=='far_center':entities=[rock(x=.42)]
    elif problem=='wrong_size':entities=[rock(w=.2,h=.3)]
    elif problem=='weak_rim':evidence[0]['rim_sectors']=8
    else:evidence[0]['x']=float('nan')
    assert not Recognizer._current_obstacle_shapes(entities,evidence,12.3)

@pytest.mark.parametrize('filename', ['00155261.png','00155357.png'])
def test_real_small_and_merged_current_rim_matches_final_published_tracks(filename):
    folder=ROOT/'artifacts/mining-A-campaign/batch-05/round-0001'
    path=folder/'gameplay/contact-windows/window-02'/filename
    if not path.exists():pytest.skip('local actual NS evidence not bundled')
    profile=json.loads((folder/'profile.json').read_text(encoding='utf8'))
    profile.update(phase_anchors=False,stable_frames=1,circle_rocks=True,publish_obstacle_shapes=True)
    r=Recognizer(profile=profile)
    frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    o=r.observe(FramePacket(frame,1.,1,60))
    assert len(o.metrics['obstacle_shapes'])==2
    assert {s['track_id'] for s in o.metrics['obstacle_shapes']}=={e.track_id for e in o.of_kind('obstacle')}
    if filename=='00155357.png':assert any(e['merged_existing'] for e in o.metrics['circle_rock_evidence'])
    missing=r.observe(FramePacket(np.zeros_like(frame),1.1,2,60))
    assert not missing.metrics['obstacle_shapes']

def test_actual_205745_tiny_rear_hole_is_repaired_without_moving_physical_body():
    folder=ROOT/'artifacts/mining-A-campaign/batch-06/round-0001'
    path=folder/'gameplay/problem-00205745-wall_recovery.png'
    if not path.exists():pytest.skip('local actual tiny-hole frame not bundled')
    profile=json.loads((folder/'profile.json').read_text(encoding='utf8'))
    profile.update(phase_anchors=False,stable_frames=1)
    frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    old=Recognizer(profile=profile).observe(FramePacket(frame,1.,1,60))
    new=Recognizer(profile={**profile,'repair_tiny_wake_gaps':True}).observe(FramePacket(frame,1.,1,60))
    assert old.of_kind('player')[0]==new.of_kind('player')[0]
    before={y:(a,b) for y,a,b in old.metrics['road_rows']};after={y:(a,b) for y,a,b in new.metrics['road_rows']}
    assert before[86/480][1]==pytest.approx(.36875)
    assert after[86/480][1]>.48
    assert all(before[y]==after[y] for y in before if y!=86/480)


def test_actual_206040_current_compact_gold_toe_does_not_become_right_wall():
    folder=ROOT/'artifacts/mining-A-campaign/batch-06/round-0001'
    path=folder/'gameplay/problem-00206040-wall_recovery.png'
    if not path.exists():pytest.skip('local actual gold toe frame not bundled')
    profile=json.loads((folder/'profile.json').read_text(encoding='utf8'))
    profile.update(phase_anchors=False,stable_frames=1)
    frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    r=Recognizer(profile=profile);o=r.observe(FramePacket(frame,1.,1,60))
    rows={y:(a,b) for y,a,b in o.metrics['road_rows']}
    # The old trace cut this scanline to .840625, through the gold foot.
    assert rows[105/480][1]>=.88
    body,=o.of_kind('player')
    assert body.x==pytest.approx(.81796875) and body.w==pytest.approx(.0515625)
    assert body.h==pytest.approx(.06875)
    assert rows[120/480]==pytest.approx((.4875,.88125))

@pytest.mark.parametrize('problem',['real_forward_divider','six_pixel_gap','distant_five_pixel_gap'])
def test_tiny_repair_still_rejects_forward_or_unattached_gaps(problem):
    r=Recognizer(profile={'phase_anchors':False,'repair_tiny_wake_gaps':True})
    r._wake_repairs=[]
    mask=np.ones((480,640),np.uint8)*255
    player=Entity('player',212/640,92.5/480,34/640,33/480,.9)
    spans=[(0,236),(241,343)]
    if problem=='real_forward_divider':mask[111:117,236:241]=0
    elif problem=='six_pixel_gap':spans=[(0,236),(242,343)]
    else:spans=[(0,238),(243,343)]
    assert r._merge_player_wake(spans,mask,player,86,1.)==spans
