"""Real independent result screenshots, plus explicitly synthetic rejection cases."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import read_mining_score_offline as score


def image(path):
    return cv2.imdecode(np.frombuffer((ROOT/path).read_bytes(),np.uint8),1)


VALIDATION=[
    ('artifacts/mining-A-campaign/batch-01/round-0003/gameplay/last-frame.png','00730'),
    ('artifacts/mining-A-campaign/batch-01/round-0004/gameplay/last-frame.png','00725'),
    ('artifacts/mining-A-campaign/batch-01/round-0005/gameplay/last-frame.png','00673'),
    ('artifacts/mining-A-campaign/batch-01/round-0006/gameplay/last-frame.png','00715'),
    ('artifacts/mining-supervision/round-03/last-frame.png','00564'),
    ('artifacts/mining-supervision/round-04/last-frame.png','00814'),
    ('artifacts/mining-supervision/round-05/last-frame.png','00698'),
    ('artifacts/mining-A-campaign/batch-02/round-0003/gameplay/last-frame.png','01108'),
]


@pytest.mark.parametrize('path,text',VALIDATION)
def test_independently_viewed_result_screenshots_all_five_digits(path,text):
    data=score.read_score(image(path))
    assert data['text']==text and data['value']==int(text)
    assert len(data['digits'])==5
    assert data['confidence_kind']=='normalized_template_similarity_not_probability'


def test_templates_and_validation_are_source_indexed_and_disjoint():
    index=json.loads((score.ASSETS/'source.json').read_text(encoding='utf8'))
    training_hashes={v['sha256'] for v in index['sources'].values()}
    for entry in index['sources'].values():
        assert entry['visually_verified'] and len(entry['score_text'])==5
        assert hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()==entry['sha256']
    for path,text in VALIDATION:
        digest=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        assert digest not in training_hashes
        assert index['validation'][path]['score_text']==text
        assert index['validation'][path]['sha256']==digest
    for key,entry in index['templates'].items():
        source=index['sources'][entry['source']]
        assert source['score_text'][entry['digit_position_zero_based']]==key
        assert hashlib.sha256((score.ASSETS/entry['file']).read_bytes()).hexdigest()==entry['sha256']


def test_actual_menu_high_scores_are_not_current_round_score():
    frame=image('assets/mining_result/fixture-menu.png')
    assert score.read_score(frame,(0,0,1,1))['value'] is None


@pytest.mark.parametrize('position',[0,2,4])
def test_missing_digit_is_unknown_even_when_hud_still_has_complete_score(position):
    frame=score._canonical(image(VALIDATION[0][0]),score.DEFAULT_VIEWPORT)
    x,y,w,h=score.SCORE_ROI
    frame[y:y+h,x+10+position*20:x+30+position*20]=0
    assert score.read_score(frame,(0,0,1,1))['value'] is None


def test_five_digit_row_without_result_context_is_rejected():
    frame=score._canonical(image(VALIDATION[0][0]),score.DEFAULT_VIEWPORT)
    only=np.zeros_like(frame)
    x,y,w,h=score.SCORE_ROI
    only[y:y+h,x:x+w]=frame[y:y+h,x:x+w]
    assert score.read_score(only,(0,0,1,1))['value'] is None


def test_ambiguous_best_digit_stays_unknown(monkeypatch):
    bank=list(score.template_bank())
    bank[8]=bank[6].copy()
    monkeypatch.setattr(score,'template_bank',lambda *_:bank)
    reading=score.read_score(image(VALIDATION[2][0]))
    assert reading['value'] is None and reading['text'] is None
    assert any(d['margin']<1e-6 for d in reading['digits'])


def test_template_hash_mismatch_is_unknown(tmp_path):
    for path in score.ASSETS.iterdir():
        if path.is_file():shutil.copyfile(path,tmp_path/path.name)
    with (tmp_path/'digit-0.png').open('ab') as stream:stream.write(b'changed')
    assert score.read_score(image(VALIDATION[0][0]),assets=tmp_path)['value'] is None


def test_lower_resolution_below_threshold_is_unknown_not_a_guessed_score():
    frame=cv2.resize(image(VALIDATION[2][0]),(960,540),interpolation=cv2.INTER_AREA)
    reading=score.read_score(frame)
    assert reading['value'] is None


@pytest.mark.parametrize('bad',[None,np.zeros((40,50),np.uint8),np.zeros((40,50,3),np.float64)])
def test_invalid_frame_is_unknown(bad):
    assert score.read_score(bad)['value'] is None
