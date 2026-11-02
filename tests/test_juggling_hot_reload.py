from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import juggling_hot_reload as hot


def test_each_round_has_empty_unique_cache_and_its_own_global_round_number(tmp_path):
    a=hot.worker_command(tmp_path,1,Path('profile'),Path('params'),600)
    b=hot.worker_command(tmp_path,2,Path('profile'),Path('params'),600)
    assert '-B' in a and '-X' in a
    assert a[a.index('-X')+1]!=b[b.index('-X')+1]
    assert not Path(a[a.index('-X')+1].split('=',1)[1]).exists()
    assert a[a.index('--round-number')+1]=='1'
    assert b[b.index('--round-number')+1]=='2'
    assert a[a.index('--output')+1]!=b[b.index('--output')+1]


@pytest.mark.parametrize('change',[{'game':'mining'},{'mode':'calibrate'},{'control_enabled':False},{'screen':'playing:juggling'}])
def test_existing_wrong_or_active_session_never_starts_a_second_gui_or_sends_input(monkeypatch,change):
    status={'game':'juggling','mode':'manual','control_enabled':True,'screen':'result:retry',**change}
    monkeypatch.setattr(hot,'probe_session',lambda _:status)
    monkeypatch.setattr(hot.subprocess,'Popen',lambda *a,**kw:pytest.fail('must not open another GUI'))
    with pytest.raises(RuntimeError):hot.ready_session(Path('endpoint'),None)
