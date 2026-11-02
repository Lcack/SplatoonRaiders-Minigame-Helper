"""JUGGLING task GUI and fresh-process per-round autonomous control."""
from datetime import datetime
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mining_hot_reload import EndpointClient,atomic_json
from start_mining_hot import active_ai_running,probe_session,starter_lock

ROOT=Path(__file__).resolve().parents[1]
MENUS={'menu:'+name for name in ('mining','hopping','juggling','landing')}


def worker_command(output,number,profile,parameters,seconds):
    return [sys.executable,'-B','-X',f'pycache_prefix={output / ("cache-"+str(number))}',
        str(ROOT/'tools/juggling_supervisor.py'),'--mode','play','--rounds','1',
        '--round-number',str(number),'--seconds',str(seconds),'--profile',str(profile),
        '--parameters',str(parameters),'--output',str(output/f'worker-{number:04d}')]


def ready_session(endpoint,log):
    status=probe_session(endpoint)
    if status is None:
        pythonw=Path(sys.executable).with_name('pythonw.exe')
        executable=str(pythonw if pythonw.exists() else sys.executable)
        subprocess.Popen([executable,'-X','utf8',str(ROOT/'run.py'),'task-terminal','--game','juggling',
            '--profile',str(ROOT/'profiles/juggling.live.local.json'),'--auto-start','--no-record'],
            cwd=ROOT,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            time.sleep(.3)
            status=probe_session(endpoint)
            if status is not None and status.get('control_enabled'):break
        else:raise RuntimeError('任务终端未能连接采集卡与控制器')
    if status.get('game')!='juggling' or status.get('mode')!='manual' or not status.get('control_enabled'):
        raise RuntimeError('现有终端不是已连接的JUGGLING控制接口，请先停止旧任务')
    warmup=time.monotonic()+8
    while status.get('screen')=='unknown' and time.monotonic()<warmup:
        time.sleep(.15);status=probe_session(endpoint)
        if status is None:raise RuntimeError('等待画面时任务终端已停止')
    if status.get('screen') not in MENUS|{'result:retry','result:quit'}:
        raise RuntimeError('需停在四项小游戏选择页或JUGGLING结算页，不会在未知画面按确认')
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds',type=int,default=3,help='默认3局，0表示持续Retry；局间检查stop.request')
    parser.add_argument('--seconds',type=float,default=600)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--profile',type=Path,default=ROOT/'profiles/juggling.live.local.json')
    parser.add_argument('--parameters',type=Path,default=ROOT/'profiles/juggling.ai.local.json')
    parser.add_argument('--stop-at-rank',choices=['S','A','B','C'],default='S')
    args=parser.parse_args()
    if not 0<=args.rounds<=10000 or not 1<=args.seconds<=1800:raise ValueError('轮数或单局时间无效')
    output=(args.output or ROOT/'artifacts'/('hot-juggling-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))).resolve()
    output.mkdir(parents=True,exist_ok=False)
    summary={'status':'starting','requested_rounds':args.rounds,'completed_rounds':0,'rounds':[],
        'reload_boundary':'between_rounds_fresh_process','record_video':False}
    endpoint=ROOT/'artifacts/control_endpoint.json';client=None
    try:
        with (output/'launcher.log').open('ab',buffering=0) as log:
            if active_ai_running(ROOT):raise RuntimeError('已有自主AI在控制，未启动第二个控制者')
            with starter_lock(ROOT/'artifacts/juggling-hot-start.lock'):
                if active_ai_running(ROOT):raise RuntimeError('已有自主AI在控制，未重复启动')
                status=ready_session(endpoint,log);summary['capture_session']=status['output']
                client=EndpointClient(endpoint)
                while args.rounds==0 or summary['completed_rounds']<args.rounds:
                    if (output/'stop.request').exists():
                        summary['status']='stopped_at_round_boundary';break
                    client.command({'command':'release'},wait=True)
                    current=client.request('/status')
                    if current.get('output')!=status['output']:raise RuntimeError('采集会话已改变，停止')
                    number=summary['completed_rounds']+1
                    summary.update(status='worker_running',current_round=number);atomic_json(output/'summary.json',summary)
                    command=worker_command(output,number,args.profile.resolve(),args.parameters.resolve(),args.seconds)
                    code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=log,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).returncode
                    if code:raise RuntimeError(f'第{number}局工作进程退出{code}，停止且不盲按Retry')
                    directory=output/f'worker-{number:04d}'/f'round-{number:04d}'
                    report=json.loads((directory/'summary.json').read_text(encoding='utf8'))
                    if not str(report.get('result','')).startswith('result:'):raise RuntimeError('未确认结算')
                    summary['rounds'].append({'round':number,'directory':str(directory),'version':report['version'],
                        'score':report.get('score_reading',{}).get('value'),'rank':report.get('result_reading',{}).get('rank'),
                        'report':report})
                    summary['completed_rounds']+=1;atomic_json(output/'summary.json',summary)
                    reading=report.get('result_reading',{})
                    if (reading.get('rank')==args.stop_at_rank and reading.get('result_visible') is True
                            and reading.get('confidence',0)>=.8 and reading.get('match_margin',0)>=.1):
                        summary.update(status='rank_target_observed',automated_rank_needs_visual_review=True)
                        break
                    time.sleep(.5)
                else:summary['status']='completed'
    except BaseException as error:
        summary.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        raise
    finally:
        if client is not None:
            try:client.command({'command':'release'},wait=True)
            except Exception as error:summary['final_release_error']=str(error)
        atomic_json(output/'summary.json',summary)


if __name__=='__main__':
    try:main()
    except Exception as error:
        message=str(error).replace('MINING','JUGGLING')
        if sys.stderr is not None:print(message,file=sys.stderr)
        else:
            import tkinter as tk
            from tkinter import messagebox
            root=tk.Tk();root.withdraw();messagebox.showerror('JUGGLING 任务未启动',message);root.destroy()
        raise SystemExit(1)
