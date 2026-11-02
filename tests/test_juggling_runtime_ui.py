import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from juggling_runtime_ui import RuntimeUI

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('relative',[
    'S-batch-01/worker-0001/round-0001/result.png',
    'S-batch-01/worker-0003/round-0003/result.png',
    'slide-probe-01/round-0001/result.png'])
def test_real_a_b_c_endings_still_detected_on_fast_path(relative):
    path=ROOT/'artifacts/juggling-live'/relative
    if not path.exists():pytest.skip('actual local result evidence not bundled')
    frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    ui=RuntimeUI(json.loads((ROOT/'profiles/session.local.json').read_text(encoding='utf8')),ROOT/'profiles')
    assert ui.observe(frame,False,10)[0]=='result:retry'
    assert ui.observe(frame,True,10.04)[0]=='result:retry'
    assert ui.scope=='runtime_result_only'


def test_unexpected_menu_is_scanned_again_within_one_second():
    class Matcher:
        def __init__(self,value):self.value=value
        def observe(self,frame):return self.value,{}
    ui=RuntimeUI.__new__(RuntimeUI);ui.full=Matcher('menu:juggling');ui.ending=Matcher('unknown');ui.last_full=None
    assert ui.observe(None,False,10)[0]=='menu:juggling'
    assert ui.observe(None,True,10.04)[0]=='unknown'
    assert ui.observe(None,True,11.01)[0]=='menu:juggling'
