"""Actual submitted slide pulses and subsequent observed actor displacement."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path

import numpy as np


def read(path):
    out=[]
    for line in path.read_text(encoding='utf8').splitlines():
        try:out.append(json.loads(line))
        except json.JSONDecodeError:pass
    return out


def analyze(path):
    rows=read(path/'trace.jsonl');loaded=json.loads((path/'loaded-version.json').read_text(encoding='utf8'))
    lo=rows[0]['observation']['timestamp'];hi=rows[-1]['observation']['timestamp']
    events=[c for c in read(Path(loaded['capture_session'])/'commands.jsonl') if c.get('success')
            and c.get('event')=='send_state' and lo<=c['after']<=hi]
    report=[]
    for button,role,axis in [('ZL','player','left_stick'),('ZR','partner','right_stick')]:
        active=None
        for event in events:
            pressed=button in event['state']['buttons']
            if pressed and active is None:active=event
            if not pressed and active is not None:
                a=active['after'];b=event['after'];points=[]
                for row in rows:
                    o=row['observation'];t=o['timestamp']
                    actor=[e for e in o['entities'] if e['kind']==role]
                    if a-.05<=t<=b+.9 and len(actor)==1 and o['phase']!='result':
                        points.append({'t_after_submit':t-a,'x':actor[0]['x'],'y':actor[0]['y'],
                            'sequence':o['sequence'],'lives':o['metrics'].get('lives_remaining')})
                advice=min(rows,key=lambda r:abs(r['client_timestamp']-a))
                case=advice['decision'].get('diagnostics',{}).get('slide_probe',{}).get('probe_id')
                x0=min(points,key=lambda p:abs(p['t_after_submit']-.1))['x'] if points else None
                samples={}
                for dt in (.2,.3,.4,.5,.7,.9):
                    if not points:continue
                    p=min(points,key=lambda p:abs(p['t_after_submit']-dt))
                    if abs(p['t_after_submit']-dt)<.07:samples[str(dt)]=p['x']-x0
                report.append({'button':button,'case':case,'actual_hold_seconds':b-a,
                    'axis':active['state'][axis]['x']/32767,'submit_timestamp':a,'observed_start_x_approx':x0,
                    'displacement_from_approx_100ms_after_submit':samples,'points':points,
                    'scope':'actual backend report and capture positions; about .12s effective delay, clipping/freeze may confound'})
                active=None
    return sorted(report,key=lambda p:p['submit_timestamp'])


def plot(cases,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    n=len(cases);fig,axes=plt.subplots(max(1,(n+1)//2),2,figsize=(10,max(3,2.8*((n+1)//2))),squeeze=False)
    for ax,case in zip(axes.flat,cases):
        p=case['points'];ax.plot([q['t_after_submit'] for q in p],[q['x'] for q in p],'.-',label='画面实测横坐标')
        ax.axvspan(0,case['actual_hold_seconds'],alpha=.2,color='green',label='实际按钮提交区间')
        ax.axvline(.12,color='gray',linestyle='--',label='初步响应延迟')
        ax.set_title(f"{case['case']} / {case['button']}");ax.set_xlabel('距首次实际提交 / 秒');ax.set_ylabel('视口横坐标');ax.grid(alpha=.2)
    for ax in list(axes.flat)[n:]:ax.set_visible(False)
    if n:axes.flat[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('round',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.round);a.output.write_text(json.dumps(result,indent=2),encoding='utf8');plot(result,a.output.with_suffix('.png'))
    print(json.dumps([{k:v for k,v in r.items() if k!='points'} for r in result],indent=2))
