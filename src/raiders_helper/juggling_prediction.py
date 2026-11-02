"""Short free-flight estimates from current, source-timed ball detections."""
from collections import deque
import math

import numpy as np


def fit_window(history):
    result={'accepted':False,'source':'current_track_quadratic_window',
            'confidence_kind':'model_fit_not_observation_or_probability'}
    if len(history)<8:return {**result,'reason':'insufficient_current_track_history'}
    h=list(history)[-12:];span=h[-1][0]-h[0][0]
    if not .25<=span<=.8:return {**result,'reason':'window_duration'}
    if any(not .015<=b[0]-a[0]<=.12 for a,b in zip(h,h[1:])):return {**result,'reason':'capture_gap'}
    if any(not math.isfinite(v) for p in h for v in p):return {**result,'reason':'nonfinite'}
    for column in (3,4):
        values=[p[column] for p in h]
        if min(values)<=0 or max(values)/min(values)>1.6:return {**result,'reason':'shape_or_identity_change'}
    t=np.array([p[0]-h[-1][0] for p in h]);x=np.array([p[1] for p in h]);y=np.array([p[2] for p in h])
    xx=np.polyfit(t,x,1);yy=np.polyfit(t,y,2)
    error_x=float(np.sqrt(np.mean((np.polyval(xx,t)-x)**2))*640)
    error_y=float(np.sqrt(np.mean((np.polyval(yy,t)-y)**2))*480)
    train_x=np.polyfit(t[:-2],x[:-2],1);train_y=np.polyfit(t[:-2],y[:-2],2)
    check_x=float(np.max(abs(np.polyval(train_x,t[-2:])-x[-2:]))*640)
    check_y=float(np.max(abs(np.polyval(train_y,t[-2:])-y[-2:]))*480)
    gravity=float(2*yy[0]);vx=float(xx[0]);vy=float(yy[1])
    result.update(samples=len(h),window_seconds=span,timestamp=h[-1][0],
        x=float(xx[1]),y=float(yy[2]),vx=vx,vy=vy,gravity=gravity,
        fit_rmse_pixels=[error_x,error_y],last_two_prediction_error_pixels=[check_x,check_y])
    if error_x>2.5 or error_y>2.5 or check_x>4 or check_y>4:
        return {**result,'reason':'fit_or_temporal_holdout_error'}
    if not .05<=gravity<=2.5 or abs(vx)>1.5 or abs(vy)>3:
        return {**result,'reason':'outside_experimental_model_domain'}
    if abs(result['x']-h[-1][1])*640>4 or abs(result['y']-h[-1][2])*480>4:
        return {**result,'reason':'current_position_mismatch'}
    return {**result,'accepted':True,'reason':'current_shape_and_temporal_fit_passed'}


class BallTrajectoryObserver:
    def __init__(self):
        self.histories={};self.last_time=None

    def observe(self,entities,timestamp,phase):
        if phase!='playing' or self.last_time is not None and not 0<timestamp-self.last_time<=.2:
            self.histories.clear()
        self.last_time=timestamp
        if phase!='playing':return {}
        self.histories={key:h for key,h in self.histories.items() if timestamp-h[-1][0]<=.15}
        result={}
        for e in entities:
            if e.kind!='ball':continue
            history=self.histories.setdefault(e.track_id,deque(maxlen=12))
            point=(timestamp,float(e.x),float(e.y),float(e.w),float(e.h))
            if len(history)>=2:
                a,b=list(history)[-2:];dt=timestamp-b[0]
                old_vy=(b[2]-a[2])/(b[0]-a[0]);new_vy=(point[2]-b[2])/dt if dt>0 else 0
                old_vx=(b[1]-a[1])/(b[0]-a[0]);new_vx=(point[1]-b[1])/dt if dt>0 else 0
                bounced=(old_vy>.2 and new_vy<-.2 and abs(point[2]-b[2])>.012
                         or old_vx*new_vx<0 and min(abs(old_vx),abs(new_vx))>.2 and abs(point[1]-b[1])>.008)
                if bounced or not 0<dt<=.12:history.clear()
            history.append(point)
            result[str(e.track_id)]=fit_window(history)
        return result
