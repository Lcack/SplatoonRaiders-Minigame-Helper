"""Experimental past-only per-track gravity memory, evaluated off-line."""
import argparse
from bisect import bisect_left
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from raiders_helper.juggling_prediction import BallTrajectoryObserver
from raiders_helper.models import Entity


def memory_forecast(history,prior):
    if len(history)<5:return None
    h=list(history)[-7:];t=np.array([p[0]-h[-1][0] for p in h])
    if not .13<=-t[0]<=.5 or h[-1][0]-prior['timestamp']>1.2:return None
    x=np.array([p[1] for p in h]);y=np.array([p[2] for p in h]);xx=np.polyfit(t,x,1)
    adjusted=y-.5*prior['gravity']*t*t;yy=np.polyfit(t,adjusted,1)
    expected=np.polyval(yy,t)+.5*prior['gravity']*t*t
    if np.sqrt(np.mean((expected-y)**2))*480>4:return None
    heldout=np.polyfit(t[:-2],adjusted[:-2],1)
    if max(abs(np.polyval(heldout,t[-2:])+.5*prior['gravity']*t[-2:]**2-y[-2:]))*480>5:return None
    vx=xx[0] if np.sqrt(np.mean((np.polyval(xx,t)-x)**2))*640<=3 else None
    return {'timestamp':h[-1][0],'x':h[-1][1],'y':h[-1][2],'vx':vx,
            'vy':float(yy[0]),'gravity':prior['gravity']}


def evaluate(path):
    rows=[json.loads(line)['observation'] for line in path.read_text(encoding='utf8').splitlines()]
    observer=BallTrajectoryObserver();priors={};forecasts=[];tracks={};last={};total=0;accepted=0
    for o in rows:
        es=[Entity(**e) for e in o['entities']];models=observer.observe(es,o['timestamp'],o['phase'])
        if o['phase']!='playing':continue
        for e in es:
            if e.kind!='ball':continue
            total+=1;t=o['timestamp'];tracks.setdefault(e.track_id,[]).append((t,e.x,e.y,e.w,e.h))
            model=models.get(str(e.track_id),{})
            if model.get('accepted'):priors[e.track_id]=model
            prior=priors.get(e.track_id)
            if not prior:continue
            m=memory_forecast(observer.histories[e.track_id],prior)
            if m is None:continue
            accepted+=1
            if t-last.get(e.track_id,-1e9)<.18:continue
            last[e.track_id]=t;forecasts.append((e,m))
    result={'trace':str(path),'observations':total,'accepted':accepted,'fraction':accepted/max(1,total),'horizons':{}}
    for horizon in (.15,.3,.5):
        errors=[]
        for e,m in forecasts:
            h=tracks[e.track_id];ts=[p[0] for p in h];i=bisect_left(ts,m['timestamp']-1e-6);j=bisect_left(ts,m['timestamp']+horizon)
            if j>=len(h) or h[j][0]-m['timestamp']-horizon>.06:continue
            future=h[i:j+1]
            if (len(future)<3 or any(b[0]-a[0]>.13 for a,b in zip(future,future[1:]))
                    or any(p[2]>.8 for p in future)
                    or any((b[2]-a[2])/(b[0]-a[0])>.2 and (c[2]-b[2])/(c[0]-b[0])<-.2 for a,b,c in zip(future,future[1:],future[2:]))):continue
            dt=h[j][0]-m['timestamp'];raw=e.y+e.vy*dt;estimate=m['y']+m['vy']*dt+.5*m['gravity']*dt*dt
            errors.append([(raw-h[j][2])*480,(estimate-h[j][2])*480])
        a=np.array(errors);result['horizons'][str(horizon)]={'pairs':len(a),'rmse_raw_memory':np.sqrt(np.mean(a*a,axis=0)).tolist(),
            'median_raw_memory':np.median(abs(a),axis=0).tolist(),'p90_raw_memory':np.percentile(abs(a),90,axis=0).tolist()}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('traces',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=[evaluate(path) for path in a.traces];a.output.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
