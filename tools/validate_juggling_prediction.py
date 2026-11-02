"""Frozen trace comparison of forecasts against later actual detections.

No future detections enter a fitted model. Future gaps/impulses/low court
positions are excluded from evaluation of the free-flight domain only.
Coverage and exclusions are reported, not called detection accuracy.
"""
import argparse
from bisect import bisect_left
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from raiders_helper.juggling_prediction import BallTrajectoryObserver
from raiders_helper.models import Entity


def evaluate(path):
    rows=[json.loads(line)['observation'] for line in path.read_text(encoding='utf8').splitlines()]
    forecasts=[];observer=BallTrajectoryObserver();tracks={};last_emit={};playing=0;accepted=0
    for obs in rows:
        entities=[Entity(**e) for e in obs['entities']]
        models=observer.observe(entities,obs['timestamp'],obs['phase'])
        if obs['phase']!='playing': continue
        for e in entities:
            if e.kind!='ball':continue
            playing+=1;t=obs['timestamp'];tracks.setdefault(e.track_id,[]).append((t,e.x,e.y,e.w,e.h))
            model=models.get(str(e.track_id),{})
            if not model.get('accepted'):continue
            accepted+=1
            if t-last_emit.get(e.track_id,-1e9)<.18:continue
            last_emit[e.track_id]=t
            forecasts.append((e,model))
    errors={str(h):[] for h in (.15,.3,.5)};excluded={str(h):0 for h in (.15,.3,.5)}
    for e,m in forecasts:
        history=tracks[e.track_id];times=[p[0] for p in history];start=bisect_left(times,m['timestamp']-1e-6)
        for horizon in (.15,.3,.5):
            key=str(horizon);end=bisect_left(times,m['timestamp']+horizon)
            future=history[start:end+1]
            if end>=len(history) or len(future)<3 or abs(history[end][0]-m['timestamp']-horizon)>.06:
                excluded[key]+=1;continue
            velocities=[((b[1]-a[1])/(b[0]-a[0]),(b[2]-a[2])/(b[0]-a[0])) for a,b in zip(future,future[1:])]
            if (any(b[0]-a[0]>.13 for a,b in zip(future,future[1:]))
                    or any(p[2]>.8 for p in future)
                    or any(a[1]>.2 and b[1]<-.2 for a,b in zip(velocities,velocities[1:]))
                    or max(p[3] for p in future)/min(p[3] for p in future)>1.7
                    or max(p[4] for p in future)/min(p[4] for p in future)>1.7):
                excluded[key]+=1;continue
            target=history[end];dt=target[0]-m['timestamp']
            local=[e.x+e.vx*dt,e.y+e.vy*dt]
            quadratic=[m['x']+m['vx']*dt,m['y']+m['vy']*dt+.5*m['gravity']*dt*dt]
            errors[key].append({'local':[(local[0]-target[1])*640,(local[1]-target[2])*480],
                'quadratic':[(quadratic[0]-target[1])*640,(quadratic[1]-target[2])*480]})
    result={'trace':str(path),'ball_observations':playing,'accepted_model_observations':accepted,
            'accepted_fraction':accepted/max(1,playing),'forecasts':len(forecasts),'horizons':{}}
    for horizon,values in errors.items():
        summary={'pairs':len(values),'excluded_free_flight_pairs':excluded[horizon]}
        for name in ('local','quadratic'):
            a=np.array([v[name] for v in values])
            if len(a):summary[name]={'rmse_xy_pixels':np.sqrt(np.mean(a*a,axis=0)).tolist(),
                'median_abs_xy_pixels':np.median(abs(a),axis=0).tolist(),'p90_abs_xy_pixels':np.percentile(abs(a),90,axis=0).tolist()}
        result['horizons'][horizon]=summary
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('traces',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report={'scope':'past-only fit; later actual free-flight detections; not score or complete calibration',
            'traces':[evaluate(path) for path in a.traces]}
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
