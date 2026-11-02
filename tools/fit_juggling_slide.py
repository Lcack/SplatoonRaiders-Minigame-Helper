"""Fit an empirical fixed slide impulse; fourth direction is held out."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def fraction(t,duration):
    phase=np.clip(t/duration,0,1)
    return 2*phase-phase*phase


def fit(cases):
    prepared=[]
    for case in cases:
        sign=1 if case['axis']>0 else -1
        points=case['points'];x0=np.median([p['x'] for p in points if p['t_after_submit']<.08])
        t=np.array([p['t_after_submit'] for p in points]);y=np.array([(p['x']-x0)*sign for p in points])
        prepared.append((t,y))
    train=prepared[:3]
    def errors(p,data):
        distance,duration,delay=p
        return np.concatenate([(distance*fraction(t-delay,duration)-y)*640 for t,y in data])
    model=least_squares(lambda p:errors(p,train),[.297,.35,.12],bounds=([.2,.2,.05],[.38,.55,.2]),loss='soft_l1',f_scale=2)
    def stats(e):return {'rmse_pixels':float(np.sqrt(np.mean(e*e))),'p90_absolute_pixels':float(np.percentile(abs(e),90)),
                         'max_absolute_pixels':float(max(abs(e)))}
    distance,duration,delay=map(float,model.x)
    return {'scope':'empirical .4-axis .08-.11s button pulses; unbounded-by-wall observed cases; not complete physics or cooldown calibration',
        'distance':distance,'duration_seconds':duration,'effective_delay_seconds':delay,
        'initial_speed_estimate':2*distance/duration,'deceleration_estimate':2*distance/duration**2,
        'train_cases':3,'holdout_cases':len(prepared)-3,'train':stats(errors(model.x,train)),
        'holdout':stats(errors(model.x,prepared[3:])),'autonomous_enabled':False,'calibrated':False,
        'tested_axis_magnitude':.4,'tested_hold_seconds':[c['actual_hold_seconds'] for c in cases]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=a.evidence.read_bytes();report=fit(json.loads(raw));report.update(source=str(a.evidence),source_sha256=hashlib.sha256(raw).hexdigest())
    a.output.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
