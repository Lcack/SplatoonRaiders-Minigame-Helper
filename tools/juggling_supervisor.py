"""Supervised real JUGGLING identification and autoplay via the task terminal.

The GUI remains the sole hardware owner. Raw observations, policy advice and
actual backend commands are separate evidence; no replay score is a result.
"""
from dataclasses import asdict, replace
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mining_supervisor import LiveClient, LatestLease
from mining_ai_telemetry import ActiveAIReporter
from raiders_helper.control.buttons import ControllerState
from raiders_helper.models import Decision
from raiders_helper.motion_calibration import CalibrationProbe
from raiders_helper.session_runtime import ScreenMatcher
from raiders_helper.trace import state_dict
from raiders_helper.vision import create_recognizer, load_profile
from raiders_helper.juggling import decide_juggling
from raiders_helper.mining_result_reader import inspect_result
from read_mining_score_offline import read_score
from juggling_control_model import SubmittedInputObserver,IdleCenterGate
from juggling_slide_probe import SlideProbe
from juggling_slide_assist import SlideAssist
from juggling_runtime_ui import RuntimeUI


def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')


class PlayGate:
    """A stale result image after A is not the end of the new game."""
    def __init__(self,attached=False):
        self.countdown=attached
        self.played=False

    def update(self,phase,screen):
        self.countdown=self.countdown or screen=='countdown:juggling'
        allowed=self.countdown and phase=='playing' and not screen.startswith(('menu:','result:','countdown:'))
        self.played=self.played or allowed
        return allowed,self.played and screen.startswith('result:')


def prepare_menu(client,screen):
    order=['menu:mining','menu:hopping','menu:juggling','menu:landing']
    for _ in range(4):
        if screen not in order or screen=='menu:juggling':return screen
        button='DPAD_DOWN' if order.index(screen)<2 else 'DPAD_UP'
        previous=screen
        client.command({'command':'press','buttons':[button],'hold_ms':80},wait=True)
        deadline=time.perf_counter()+2
        while time.perf_counter()<deadline:
            time.sleep(.08);screen=client.request('/status').get('screen')
            if screen in order and screen!=previous:break
        else:raise RuntimeError('未确认菜单光标移动，停止且不按A')
    raise RuntimeError('未能闭环定位JUGGLING')


def run_round(client,profile,parameters,output,mode,seconds,number,version,hashes,attached=False):
    output.mkdir(parents=True,exist_ok=False)
    status=client.request('/status')
    if status.get('game')!='juggling' or not status.get('control_enabled'):
        raise ValueError('需要已连接控制器的JUGGLING人工接口')
    client.command({'command':'pause'},wait=True)
    screen=status.get('screen')
    if not attached:screen=prepare_menu(client,screen)
    if screen=='result:quit':
        client.command({'command':'press','buttons':['DPAD_UP'],'hold_ms':80},wait=True)
        time.sleep(.4);screen=client.request('/status').get('screen')
    if attached and screen not in {'playing:juggling','countdown:juggling'}:
        raise ValueError(f'接管当前局需要确认JUGGLING运行/倒计时，当前{screen}')
    if not attached and screen not in {'menu:juggling','result:retry'}:
        raise ValueError(f'需已确认JUGGLING选择或RETRY页，当前{screen}')
    client.command({'command':'pause'},wait=True)
    recognizer=create_recognizer('juggling',profile)
    config=json.loads((ROOT/'profiles/session.local.json').read_text(encoding='utf8'))
    matcher=RuntimeUI(config,ROOT/'profiles')
    probe=CalibrationProbe('juggling',{'enabled':True,'levels':[.4,.7],
        'pulse_seconds':.28,'neutral_seconds':.34,'min_confidence':.8,
        'min_track_samples':3,'play_bounds':[.04,.96]})
    probe.reset_round(number)
    if mode=='slide-probe':probe=SlideProbe(parameters);probe.reset_round(number)
    idle_gate=IdleCenterGate(parameters.get('idle_centering',{}).get('delay_seconds',.3))
    assist=SlideAssist(parameters.get('experimental_slide_rescue',{}))
    loaded={'loaded_version':version,'source_hashes':hashes,'capture_session':status['output'],
        'profile_sha':hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest(),
        'parameters_sha':hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest(),
        'round':number,'run_dir':str(output.resolve()),'game':'juggling'}
    save_json(output/'loaded-version.json',loaded)
    save_json(output/'profile.json',profile);save_json(output/'parameters.json',parameters)
    lease=LatestLease(client);gate=PlayGate(attached);result=None;records=0;last_sequence=-1;last_snapshot=0.
    started=time.perf_counter();frame_times=[];last_console=0.;initial_lives=None
    reporter=ActiveAIReporter(loaded)
    motion=parameters.get('input_response_model',{})
    if parameters.get('experimental_slide_rescue',{}).get('enabled') is True:
        motion={**motion,'slide_model':parameters['experimental_slide_rescue']['model']}
    inputs=SubmittedInputObserver(client,motion) if motion.get('enabled') is True else None
    try:
        reporter.publish(force=True,mode='starting',reason='JUGGLING：独立进程已加载，等待开局倒计时')
        if not attached:client.command({'command':'press','buttons':['A'],'hold_ms':80},wait=True)
        lease.thread.start()
        if inputs is not None:inputs.thread.start()
        with (output/'trace.jsonl').open('w',encoding='utf8') as trace:
            while time.perf_counter()-started<seconds:
                if lease.error:raise RuntimeError(lease.error)
                packet=client.frame()
                if packet.sequence==last_sequence:time.sleep(.004);continue
                last_sequence=packet.sequence
                observed=recognizer.observe(packet);screen,_=matcher.observe(packet.frame,gate.played,time.perf_counter())
                observed.metrics.update(screen_match_ms=matcher.ms,screen_matcher_scope=matcher.scope)
                if screen!='unknown':observed=replace(observed,phase=screen.split(':')[0],confidence=0.)
                allowed,ended=gate.update(observed.phase,screen)
                observed.metrics['observation_age_seconds']=time.perf_counter()-packet.timestamp
                if inputs is not None:inputs.attach(observed)
                observed.metrics['user_reported_initial_lives']=3
                if gate.countdown and observed.metrics.get('lives_observed') and observed.metrics.get('lives_remaining',0)>0:
                    if initial_lives is None:initial_lives=observed.metrics['lives_remaining']
                if not allowed:decision=Decision(reason='JUGGLING：等待确认新局倒计时与游戏运行')
                elif mode in ('probe','slide-probe'):decision=probe.decide(observed)
                else:decision=decide_juggling(observed,parameters)
                decision=idle_gate.apply(observed,decision)
                decision,release_at,release_state=assist.apply(observed,decision,time.perf_counter())
                now=time.perf_counter();age=now-packet.timestamp
                fresh=0<=age<=.2
                state=decision.state if fresh else ControllerState.neutral()
                lease.update(state,release_at=release_at if fresh else None,release_state=release_state if fresh else None)
                identifying=any(k in decision.diagnostics for k in ('slide_probe','calibration_probe'))
                reporter.publish(mode='identification' if identifying else 'autoplay',reason=decision.reason,sequence=packet.sequence)
                row={'live':True,'scope':'advice_join_session_commands_jsonl_for_actual_execution',
                    'client_timestamp':now,'round':number,'mode':mode,'observation':asdict(observed),
                    'decision':{'reason':decision.reason,'state':state_dict(state),'diagnostics':decision.diagnostics},
                    'frame_age_ms':age*1000,'stale_suppressed':not fresh,'screen':screen,
                    'planned_pulse_release_at':release_at if fresh else None}
                trace.write(json.dumps(row,ensure_ascii=False)+'\n');records+=1;frame_times.append(age*1000)
                snapshot_interval=.06 if mode=='slide-probe' and now-started<30 else .4
                if now-last_snapshot>=snapshot_interval:
                    cv2.imencode('.jpg',packet.frame,[cv2.IMWRITE_JPEG_QUALITY,90])[1].tofile(str(output/f'frame-{packet.sequence:08d}.jpg'))
                    last_snapshot=now
                if now-last_console>=1:
                    print(f'{now-started:.1f}s {decision.reason}',flush=True);last_console=now
                if ended:
                    result=screen
                    cv2.imencode('.png',packet.frame)[1].tofile(str(output/'result.png'))
                    break
                if records%20==0:
                    status_now=client.request('/status')
                    if not status_now.get('control',{}).get('paused'):
                        raise RuntimeError('控制所有权已交回其他任务，停止外部AI')
    finally:
        if lease.thread.ident is not None:lease.close()
        else:
            try:client.command({'command':'release'},wait=True)
            except Exception:pass
        reporter.close()
        if inputs is not None:inputs.close()
    summary={'round':number,'mode':mode,'result':result,'frames':records,'seconds':time.perf_counter()-started,
        'stale_frames':sum(v>200 for v in frame_times),'version':version,'session':status['output'],
        'user_reported_initial_lives':3,'initial_lives_detected':initial_lives if not attached else None,
        'first_running_lives_detected':initial_lives,'attached_existing_round':attached,
        'lives_visually_verified':False}
    if result:
        frame=cv2.imdecode(np.frombuffer((output/'result.png').read_bytes(),np.uint8),1)
        summary['result_reading']=asdict(inspect_result(frame))
        assets=Path(profile.get('_root',ROOT/'profiles'))/profile.get('score_assets','../assets/juggling_live_score')
        summary['score_reading']=read_score(frame,assets=assets)
    save_json(output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    if result is None:raise RuntimeError('未确认结算，已释放输入，不自动Retry')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['probe','play','slide-probe'],default='play')
    parser.add_argument('--profile',type=Path,default=ROOT/'profiles/juggling.json')
    parser.add_argument('--parameters',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=1)
    parser.add_argument('--seconds',type=float,default=600)
    parser.add_argument('--attach-current',action='store_true')
    parser.add_argument('--round-number',type=int,default=1)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if not 1<=args.rounds<=100 or not 1<=args.seconds<=1800 or args.round_number<1:raise ValueError('轮数或时间无效')
    profile=load_profile(args.profile,'juggling')
    parameters=json.loads(args.parameters.read_text(encoding='utf8')) if args.parameters else dict(profile.get('policy',{}))
    paths=[*sorted((ROOT/'src/raiders_helper').glob('*.py')),Path(__file__),
           *[ROOT/'tools'/name for name in ('mining_supervisor.py','mining_ai_telemetry.py','read_mining_score_offline.py','juggling_control_model.py','juggling_slide_probe.py','juggling_slide_assist.py','juggling_runtime_ui.py')],
           *sorted((ROOT/'assets/juggling_live_score').glob('*.*'))]
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    version=hashlib.sha256(json.dumps({'source':hashes,'profile':profile,'parameters':parameters},sort_keys=True).encode()).hexdigest()
    client=LiveClient(ROOT/'artifacts/control_endpoint.json',frame_width=960)
    summary={'version':version,'rounds':[],'status':'running'};save_json(args.output/'summary.json',summary)
    try:
        for number in range(args.round_number,args.round_number+args.rounds):
            print(f'JUGGLING 第{number}局 / {args.mode} / 核心 {version[:12]}',flush=True)
            summary['rounds'].append(run_round(client,profile,parameters,args.output/f'round-{number:04d}',args.mode,args.seconds,number,version,hashes,args.attach_current and number==args.round_number))
            save_json(args.output/'summary.json',summary)
            time.sleep(.5)
        summary['status']='completed'
    except BaseException as error:
        summary['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        summary['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        try:client.command({'command':'release'},wait=True)
        except Exception as error:summary['final_release_error']=str(error)
        save_json(args.output/'summary.json',summary)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf8')
    main()
