"""Short anchored input/position validation against actual backend reports."""
import argparse
from bisect import bisect_left
import json
from pathlib import Path

import numpy as np


def read(path):return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]


def pairs(rows,role,horizon):
    points=[];losses=[];last_lives=None
    for row in rows:
        o=row['observation'];lives=o['metrics'].get('lives_remaining')
        if lives is not None and last_lives is not None and lives<last_lives:losses.append(o['timestamp'])
        if lives is not None:last_lives=lives
        e=[e for e in o['entities'] if e['kind']==role]
        if (o['phase']=='playing' and len(e)==1 and e[0]['samples']>=3
                and (row['decision'].get('diagnostics',{}).get('policy')=='juggling_interception' or row['mode']=='probe')):
            points.append((o['timestamp'],e[0]))
    times=[p[0] for p in points];result=[];last=-1e9
    for i,(a,e) in enumerate(points):
        if a-last<.12:continue
        j=bisect_left(times,a+horizon)
        if j>=len(points):continue
        b,f=points[j];window=points[i:j+1];low,high=(.08,.43) if role=='player' else (.57,.92)
        if (b-a>horizon+.055 or any(v-u>.12 for u,v in zip(times[i:j],times[i+1:j+1]))
                or any(not low<p['x']<high or p['track_id']!=e['track_id'] for _,p in window)
                or any(a-1<loss<b+1 for loss in losses)):
            continue
        result.append((a,b,f['x']-e['x'],e['vx']*(b-a)));last=a
    return np.array(result)


def command_integral(events,axis,times,tau,deadzone):
    ts=np.array([c['after'] for c in events]);u=np.array([c['state'][axis]['x']/32767 for c in events])
    u=np.sign(u)*np.maximum(0,abs(u)-deadzone)/(1-deadzone)
    pos=np.zeros(len(u));v=np.zeros(len(u))
    for i in range(1,len(u)):
        dt=ts[i]-ts[i-1]
        if tau:
            decay=np.exp(-dt/tau)
            pos[i]=pos[i-1]+u[i-1]*dt+(v[i-1]-u[i-1])*tau*(1-decay)
            v[i]=u[i-1]+(v[i-1]-u[i-1])*decay
        else:pos[i]=pos[i-1]+u[i-1]*dt
    idx=np.clip(np.searchsorted(ts,times,side='right')-1,0,len(ts)-1);dt=np.maximum(0,times-ts[idx])
    x=pos[idx]+u[idx]*dt
    if tau:x+=(v[idx]-u[idx])*tau*(1-np.exp(-dt/tau))
    return x


def fit(rows,events,role):
    p=pairs(rows,role,.18)
    if len(p)<25:return {'accepted':False,'pairs':len(p)}
    axis='left_stick' if role=='player' else 'right_stick';split=int(.65*len(p));best=None
    for tau in (0,.02,.04,.08):
        for deadzone in (0,.05,.10,.15):
            for delay in np.arange(.05,.201,.01):
                x=command_integral(events,axis,p[:,:2]-delay,tau,deadzone);dx=x[:,1]-x[:,0]
                domain=(abs(dx[:split])>.003)
                gain=np.median(p[:split,2][domain]/dx[:split][domain]) if domain.any() else 0
                error=(gain*dx-p[:,2])*640
                objective=np.mean(np.minimum(error[:split]**2,12**2))
                if best is None or objective<best[0]:best=(objective,delay,tau,deadzone,gain,error,dx)
    _,delay,tau,deadzone,gain,error,dx=best
    def stats(e):return {'rmse':float(np.sqrt(np.mean(e*e))),'median_abs':float(np.median(abs(e))),
                         'p90_abs':float(np.percentile(abs(e),90))}
    return {'pairs':len(p),'fit_pairs':split,'holdout_pairs':len(p)-split,'delay':delay,'tau':tau,
            'deadzone':deadzone,'gain':gain,'train_pixels':stats(error[:split]),'holdout_pixels':stats(error[split:]),
            'old_velocity_holdout_pixels':stats((p[split:,3]-p[split:,2])*640),
            'scope':'short anchored free movement; loss windows and edges excluded; not complete calibration'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('rounds',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=[]
    for path in a.rounds:
        rows=read(path/'trace.jsonl');summary=json.loads((path/'summary.json').read_text(encoding='utf8'))
        lo=rows[0]['observation']['timestamp']-1;hi=rows[-1]['observation']['timestamp']+1
        events=[c for c in read(Path(summary['session'])/'commands.jsonl') if c.get('success') is True
                and c.get('event')=='send_state' and lo<=c['after']<=hi and not c['state']['buttons']]
        report.append({'round':str(path),'actors':{role:fit(rows,events,role) for role in ('player','partner')}})
    a.output.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
