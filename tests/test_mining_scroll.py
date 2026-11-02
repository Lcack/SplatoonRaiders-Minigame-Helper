"""Real timestamp semantics and adversarial wall-profile examples."""
import numpy as np
import pytest

from raiders_helper.mining_scroll import WallScrollEstimator, estimate_wall_translation, wall_edges


def profile():
    y=np.arange(600,dtype=float)
    return np.column_stack([110+.10*y+13*np.sin(y/21)+8*np.sin(y/9),
                            510-.05*y+15*np.cos(y/24)+9*np.sin(y/13)])


def mask(edges):
    result=np.zeros((480,640),np.uint8)
    for y,(a,b) in enumerate(edges):result[y,int(round(a)):int(round(b))+1]=255
    return result


def test_common_vertical_translation_is_measured_from_both_wall_shapes():
    p=profile();r=estimate_wall_translation(p[:480],p[12:492],.1)
    assert r['accepted'] and r['speed']==pytest.approx(.25,abs=.001)
    assert r['side_pixels']==[12,12]
    assert sum(s['supports_global_shift'] for s in r['vertical_segments'])>=2


def test_stationary_signal_has_explicit_quantization_noise_not_negative_speed():
    p=profile()[:480];r=estimate_wall_translation(p,p,.1)
    assert r['accepted'] and r['speed']==0.
    assert r['zero_motion_within_half_pixel']
    assert r['zero_is_quantized_measurement_not_exact_stop_proof']
    assert r['zero_resolution_speed']==pytest.approx(.5/480/.1)
    assert abs(r['raw_pixels'])<=.5


@pytest.mark.parametrize('kind',['vertical','periodic'])
def test_unidentifiable_vertical_or_periodic_profiles_are_not_observations(kind):
    y=np.arange(480,dtype=float)
    p=np.tile([100.,510.],(480,1)) if kind=='vertical' else np.column_stack([100+9*np.sin(y*2*np.pi/12),510+9*np.cos(y*2*np.pi/12)])
    assert not estimate_wall_translation(p,p,.1)['accepted']


def test_two_visible_rows_cannot_pass_even_when_their_match_is_unique():
    p=profile()[:480];current=p.copy();current[145:385]=np.nan;current[[180,320]]=p[[180,320]]
    result=estimate_wall_translation(p,current,.1)
    assert not result['accepted'] and result['reason']=='insufficient_visible_wall_coverage'


def test_disagreement_between_sides_is_rejected():
    p=profile();current=np.column_stack([p[12:492,0],p[4:484,1]])
    assert not estimate_wall_translation(p[:480],current,.1)['accepted']


def test_hidden_longitudinal_segment_cannot_hide_behind_good_overall_coverage():
    p=profile();current=p[12:492].copy();current[145:176]=np.nan
    result=estimate_wall_translation(p[:480],current,.1)
    assert not result['accepted']
    assert result['reason']=='insufficient_independent_vertical_segments'


@pytest.mark.parametrize('seconds',[.0249,.2001,0.,-.1,float('nan'),True])
def test_interval_outside_observed_pair_domain_is_unknown(seconds):
    p=profile();assert not estimate_wall_translation(p[:480],p[12:492],seconds)['accepted']


def test_viewport_cut_edge_is_not_a_visible_wall():
    floor=mask(profile()[:480]);floor[:,:180]=255
    edges=wall_edges(floor)
    assert np.all(np.isnan(edges[:,0]))
    assert not estimate_wall_translation(edges,edges,.1)['accepted']


def test_measurement_reference_is_pair_midpoint_not_capture_instant():
    p=profile();estimator=WallScrollEstimator()
    assert 'current_forward_speed_observed' not in estimator.observe(mask(p[:480]),10.,{'hue':27})
    r=estimator.observe(mask(p[12:492]),10.1,{'hue':27})
    assert r['current_forward_speed_observed'] is True
    assert r['current_forward_speed_reference_timestamp']==pytest.approx(10.05)
    assert r['current_forward_speed_window_seconds']==pytest.approx(.1)
    assert r['current_forward_speed_source']=='bilateral_wall_translation'
    assert r['current_forward_speed_method']=='wall_profile_pair'
    assert r['current_forward_speed_confidence_kind']=='heuristic_gate_score_not_probability'


def test_palette_jump_rejects_the_cross_color_pair_then_can_restart():
    p=profile();estimator=WallScrollEstimator()
    estimator.observe(mask(p[:480]),10.,{'hue':27})
    r=estimator.observe(mask(p[12:492]),10.1,{'hue':40})
    assert r['wall_scroll']['reason']=='palette_hue_jump'
    assert 'current_forward_speed_observed' not in r
    assert estimator.observe(mask(p[24:504]),10.2,{'hue':40})['current_forward_speed_observed']


@pytest.mark.parametrize('phase,blocked',[('unknown',False),('result',False),('playing',True)])
def test_unsafe_frames_reset_measurement_history(phase,blocked):
    p=profile();estimator=WallScrollEstimator()
    estimator.observe(mask(p[:480]),10.,{'hue':27})
    estimator.observe(mask(p[12:492]),10.1,{'hue':27},phase=phase,blocked=blocked)
    r=estimator.observe(mask(p[24:504]),10.2,{'hue':27})
    assert r['wall_scroll']['reason']=='need_previous_actual_frame'


def test_preferred_landmark_keeps_real_frame_history_without_overwriting_observation():
    p=profile();estimator=WallScrollEstimator()
    r=estimator.observe(mask(p[:480]),10.,{'hue':27},measure=False)
    assert set(r)=={'wall_scroll'} and r['wall_scroll']['reason']=='reliable_landmark_preferred'
    assert estimator.observe(mask(p[12:492]),10.1,{'hue':27})['current_forward_speed_observed']


def test_reversed_timestamp_cannot_create_a_fake_future_window():
    p=profile();estimator=WallScrollEstimator()
    estimator.observe(mask(p[:480]),10.,{'hue':27})
    assert estimator.observe(mask(p[12:492]),9.9,{'hue':27})['wall_scroll']['reason']=='non_monotonic_capture_time'
    assert estimator.observe(mask(p[24:504]),10.1,{'hue':27})['wall_scroll']['reason']=='need_previous_actual_frame'
