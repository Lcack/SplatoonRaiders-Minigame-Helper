"""Fit actual submitted-input/position evidence; never mark calibrated."""
from bisect import bisect_right
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def read_rows(path):
    rows=[]
    for line in path.read_text(encoding='utf8').splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    return rows


def fit_actor(rows,commands,role):
    points=[]
    for row in rows:
        obs=row['observation'];entities=[e for e in obs['entities'] if e['kind']==role]
        if obs['phase']=='playing' and len(entities)==1 and entities[0]['samples']>=3:
            e=entities[0]
            if (.07<e['x']<.44 if role=='player' else .56<e['x']<.94):
                points.append((obs['timestamp'],e['x'],e['track_id']))
    if len(points)<80:return {'accepted':False,'reason':'too_few_points'}
    begin,end=points[0][0],points[-1][0];axis='left_stick' if role=='player' else 'right_stick'
    events=[];last=None
    for c in commands:
        if c.get('success') is not True or c.get('event')!='send_state':continue
        t=c['after'];u=c['state'][axis]['x']/32767
        if t<begin-1 or t>end+.5:continue
        if u!=last:events.append((t,u));last=u
    if len(events)<8:return {'accepted':False,'reason':'too_few_transitions'}
    times=np.array([t-begin for t,x,i in points]);xs=np.array([x for t,x,i in points])
    events=[(t-begin,u) for t,u in events]
    split=np.searchsorted(times,(end-begin)*.65)

    def simulate(p):
        gain,tau,delay,drift,v,xoffset=p
        boundaries=[(t+delay,u) for t,u in events];event_times=[t for t,u in boundaries]
        i=bisect_right(event_times,0)-1;u=boundaries[i][1] if i>=0 else 0.;i+=1
        t=0.;x=xs[0]+xoffset;pred=[]
        def advance(dt):
            nonlocal x,v
            terminal=gain*u+drift;decay=math.exp(-dt/tau)
            x+=terminal*dt+(v-terminal)*tau*(1-decay)
            v=terminal+(v-terminal)*decay
        for target in times:
            while i<len(boundaries) and boundaries[i][0]<=target:
                boundary,new_u=boundaries[i]
                if boundary>t:advance(boundary-t);t=boundary
                u=new_u;i+=1
            if target>t:advance(target-t);t=target
            pred.append(x)
        return np.array(pred)

    bounds=([.03,.008,0.,-.02,-1.,-.02],[2.,.8,.4,.02,1.,.02])
    candidates=[]
    for delay in (.06,.125,.2):
        fit=least_squares(lambda p:(simulate(p)[:split]-xs[:split])*640,
            [.6,.1,delay,0.,0.,0.],bounds=bounds,loss='soft_l1',f_scale=2.,max_nfev=200)
        candidates.append(fit)
    best=min(candidates,key=lambda f:np.mean((simulate(f.x)[:split]-xs[:split])**2))
    predicted=simulate(best.x);residual=(predicted-xs)*640
    gain,tau,delay,drift,v,offset=map(float,best.x)
    return {'scope':'preliminary linear horizontal model; actual commands after -> capture clock; not full calibration',
        'points':len(points),'transitions':len(events),'train_points':int(split),'holdout_points':len(points)-int(split),
        'full_axis_steady_speed':gain,'tau_seconds':tau,'effective_delay_seconds':delay,'neutral_velocity':drift,
        'input_gain_per_second':gain/tau,'drag_per_second':1/tau,
        'train_rmse_pixels':float(np.sqrt(np.mean(residual[:split]**2))),
        'holdout_rmse_pixels':float(np.sqrt(np.mean(residual[split:]**2))),
        'holdout_absolute_error_p95_pixels':float(np.percentile(abs(residual[split:]),95)),
        'fitted_axis_domain':[-.7,.7],'calibrated':False}


def ball_segments(rows):
    histories={};completed=[]
    for row in rows:
        obs=row['observation']
        if obs['phase']!='playing':continue
        for e in obs['entities']:
            if e['kind']!='ball':continue
            t=obs['timestamp'];key=e['track_id'];history=histories.setdefault(key,[])
            if history:
                dt=t-history[-1][0]
                vy=(e['y']-history[-1][2])/dt if dt>0 else 0
                previous=(history[-1][2]-history[-2][2])/(history[-1][0]-history[-2][0]) if len(history)>1 else 0
                if not 0<dt<=.12 or abs(vy)>3 or (previous>.15 and vy<-.15):
                    completed.append(history);history=[];histories[key]=history
            history.append((t,e['x'],e['y'],e['w'],e['h']))
    completed.extend(histories.values());fits=[]
    for h in completed:
        if len(h)<10 or h[-1][0]-h[0][0]<.45:continue
        t=np.array([p[0]-h[0][0] for p in h]);x=np.array([p[1] for p in h]);y=np.array([p[2] for p in h])
        coeff=np.polyfit(t,y,2);pred=np.polyval(coeff,t);error=float(np.sqrt(np.mean((pred-y)**2))*480)
        if .05<2*coeff[0]<5 and error<4:
            fits.append({'start':h[0][0],'end':h[-1][0],'points':len(h),'gravity':float(2*coeff[0]),
                'vertical_rmse_pixels':error,'size_median_pixels':[float(np.median([p[3] for p in h])*640),float(np.median([p[4] for p in h])*480)]})
    return fits


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('round',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    rows=read_rows(args.round/'trace.jsonl');summary=json.loads((args.round/'summary.json').read_text(encoding='utf8'))
    commands=read_rows(Path(summary['session'])/'commands.jsonl')
    report={'actors':{role:fit_actor(rows,commands,role) for role in ('player','partner')},'ball_free_flight_segments':ball_segments(rows),
        'scope':'actual NS source data; no future input, rollout score, or full calibration claim'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'actors':report['actors'],'ball_segments':len(report['ball_free_flight_segments']),
        'gravity_median':float(np.median([x['gravity'] for x in report['ball_free_flight_segments']])) if report['ball_free_flight_segments'] else None},indent=2))


if __name__=='__main__':main()
