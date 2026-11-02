import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from juggling_supervisor import PlayGate,prepare_menu
import juggling_supervisor


def test_old_result_and_false_playing_before_countdown_cannot_complete_new_round():
    gate=PlayGate()
    assert gate.update('result','result:retry')==(False,False)
    assert gate.update('playing','unknown')==(False,False)
    assert gate.update('result','result:retry')==(False,False)
    assert gate.update('countdown','countdown:juggling')==(False,False)
    assert gate.update('playing','unknown')==(True,False)
    assert gate.update('result','result:retry')==(False,True)


def test_attachment_does_not_issue_a_new_game_and_requires_a_playing_frame_before_end():
    gate=PlayGate(attached=True)
    assert gate.update('result','result:retry')==(False,False)
    assert gate.update('playing','unknown')==(True,False)
    assert gate.update('result','result:retry')==(False,True)


def test_known_menu_is_walked_with_observed_cursor_changes_before_confirmation(monkeypatch):
    monkeypatch.setattr(juggling_supervisor.time,'sleep',lambda _:None)
    class Client:
        def __init__(self):self.screens=iter(['unknown','menu:hopping','menu:juggling']);self.commands=[]
        def command(self,value,wait=False):self.commands.append(value)
        def request(self,path):return {'screen':next(self.screens)}
    client=Client()
    assert prepare_menu(client,'menu:mining')=='menu:juggling'
    assert [c['buttons'] for c in client.commands]==[['DPAD_DOWN'],['DPAD_DOWN']]
    assert all('A' not in c['buttons'] for c in client.commands)
