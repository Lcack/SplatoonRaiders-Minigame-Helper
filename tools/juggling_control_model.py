"""Pending-motion estimates from successful backend submissions, not advice."""
from bisect import bisect_right
from dataclasses import replace
from collections import deque
import math
import threading
import time

from raiders_helper.control.buttons import StickVector
from juggling_slide_assist import integrate_with_slides


def project_actors(observation,items,observed_until,now,parameters):
    result={'accepted':False,'source':'backend_after_history_with_held_last_input',
            'confidence_kind':'short_motion_model_estimate_not_observation'}
    if not items or not 0<=now-observed_until<=.1:return {**result,'reason':'stale_or_missing_input_history'}
    delay=parameters.get('delay_seconds',.12);dispatch=parameters.get('dispatch_budget_seconds',.015)
    gains=[parameters.get('gain_left',.666),parameters.get('gain_right',.66)]
    if (any(type(v) not in (int,float) or not math.isfinite(v) for v in [delay,dispatch,*gains])
            or not .04<=delay<=.2 or not 0<=dispatch<=.04 or any(not .3<=v<=1 for v in gains)):
        return {**result,'reason':'invalid_response_parameters'}
    begin=observation.timestamp-delay;end=now+dispatch
    seconds=end-begin
    if not .04<=seconds<=.35:return {**result,'reason':'outside_short_prefix_domain'}
    times=[item['after'] for item in items]
    if any(not math.isfinite(t) for t in times) or any(b<a for a,b in zip(times,times[1:])):
        return {**result,'reason':'invalid_submission_times'}
    index=bisect_right(times,begin)-1
    if index<0:return {**result,'reason':'history_does_not_cover_pending_motion'}
    positions={};details={}
    slide_model=parameters.get('slide_model')
    if slide_model is not None and (not isinstance(slide_model,dict)
            or any(type(slide_model.get(k)) not in (int,float) or not math.isfinite(slide_model[k]) for k in ('distance','duration_seconds'))
            or not .2<=slide_model['distance']<=.35 or not .25<=slide_model['duration_seconds']<=.5):
        return {**result,'reason':'invalid_slide_model'}
    for side,role,axis,gain in zip(('left','right'),('player','partner'),('left_stick','right_stick'),gains):
        actor=observation.of_kind(role)
        if len(actor)!=1:return {**result,'reason':'missing_observed_actor'}
        if slide_model is not None:
            displacement=integrate_with_slides(items,side,begin,end,gain,slide_model)
            if displacement is None:return {**result,'reason':'unsupported_input_or_slide_repeat'}
        else:
            displacement=0.;cursor=begin;i=index
            while cursor<end:
                boundary=min(end,times[i+1] if i+1<len(times) else end)
                state=items[i]['state'];value=state.get(axis,{}).get('x')
                if type(value) is not int or not -32767<=value<=32767 or state.get('buttons'):
                    return {**result,'reason':'unsupported_input_or_slide'}
                displacement+=(boundary-cursor)*value/32767*gain
                cursor=boundary;i+=1
        positions[side]=actor[0].x+displacement
        details[side]={'observed_x':actor[0].x,'predicted_displacement':displacement,'gain':gain}
    return {**result,'accepted':True,'reason':'current_actual_history_short_prefix',
            'observation_timestamp':observation.timestamp,'expected_new_input_effect_time':end+delay,
            'seconds_from_observation':seconds,'positions':positions,'details':details,
            'empirical_slide_curve_available':slide_model is not None,
            'held_input_extension_seconds':max(0.,end-observed_until)}


class SubmittedInputObserver:
    def __init__(self,client,parameters):
        self.client=client;self.parameters=parameters;self.items=deque(maxlen=256);self.cursor=0
        self.lock=threading.Lock();self.stop=threading.Event();self.observed_until=0.;self.error=None
        self.thread=threading.Thread(target=self.run,daemon=True,name='juggling-actual-input-history')

    def collect(self):
        response=self.client.request(f'/events?limit=256&after={self.cursor}')
        now=time.perf_counter()
        with self.lock:
            for event in response['events']:
                if self.cursor and event['id']!=self.cursor+1:self.items.clear()
                self.cursor=event['id'];self.items.append({'after':event['timestamp'],'state':event['state']})
            self.observed_until=now;self.error=None

    def run(self):
        while not self.stop.is_set():
            try:self.collect()
            except Exception as error:
                with self.lock:self.error=str(error);self.observed_until=0.
            self.stop.wait(.025)

    def attach(self,observation):
        with self.lock:items=list(self.items);until=self.observed_until;error=self.error
        projection=project_actors(observation,items,until,time.perf_counter(),self.parameters)
        observation.metrics['juggling_actor_input_projection']=projection
        observation.metrics['juggling_actual_input_history']={'source':'successful_backend_after',
            'items':items[-32:],'observed_until':until,'error':error}

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:self.thread.join(timeout=3.2)


class IdleCenterGate:
    """Require a continuous idle interval before starting a middle detour."""
    def __init__(self,seconds=.3):self.seconds=seconds;self.since={};self.last=None

    def apply(self,obs,decision):
        d=decision.diagnostics
        if (obs.phase!='playing' or d.get('policy')!='juggling_interception'
                or self.last is not None and not 0<obs.timestamp-self.last<=.2):
            self.since={}
        self.last=obs.timestamp
        if d.get('policy')!='juggling_interception':return decision
        axes={'left':decision.state.left_stick.x,'right':decision.state.right_stick.x};delayed=[]
        for side in ('left','right'):
            if d.get('targets',{}).get(side) is not None:
                self.since.pop(side,None);continue
            self.since.setdefault(side,obs.timestamp)
            if side in d.get('idle_middle_positions',{}) and obs.timestamp-self.since[side]<self.seconds:
                axes[side]=0;delayed.append(side)
        state=replace(decision.state,left_stick=StickVector(axes['left'],decision.state.left_stick.y),
                      right_stick=StickVector(axes['right'],decision.state.right_stick.y))
        reason=decision.reason
        for side in delayed:
            label='左侧' if side=='left' else '右侧';reason=reason.replace(label+'回中等待',label+'等待目标稳定')
        return replace(decision,state=state,reason=reason,diagnostics={**d,'idle_center_delayed_sides':delayed})
