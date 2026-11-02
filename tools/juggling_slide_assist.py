"""Experimental finite slide rescue using actual-pulse-derived displacement."""
from bisect import bisect_right
from dataclasses import replace
import math

from raiders_helper.control.buttons import Button,StickVector


def distance_at(age,model):
    phase=max(0.,min(1.,age/model['duration_seconds']))
    return model['distance']*(2*phase-phase*phase)


def slide_events(items,side):
    button='ZL' if side=='left' else 'ZR';axis='left_stick' if side=='left' else 'right_stick'
    events=[];held=False
    for item in items:
        pressed=button in item['state'].get('buttons',[])
        if pressed and not held:
            value=item['state'].get(axis,{}).get('x',0)
            if value:events.append((item['after'],1 if value>0 else -1))
        held=pressed
    return events


def integrate_with_slides(items,side,begin,end,gain,model):
    """Input-clock integral; actual submit times, not future advice."""
    times=[p['after'] for p in items];events=slide_events(items,side)
    if any(b[0]-a[0]<1.4 for a,b in zip(events,events[1:])):return None
    axis='left_stick' if side=='left' else 'right_stick'
    boundaries=sorted({begin,end,*[t for t in times if begin<t<end],
        *[t+model['duration_seconds'] for t,_ in events if begin<t+model['duration_seconds']<end]})
    displacement=0.
    for a,b in zip(boundaries,boundaries[1:]):
        index=bisect_right(times,(a+b)/2)-1
        if index<0:return None
        state=items[index]['state'];value=state.get(axis,{}).get('x')
        if type(value) is not int or not -32767<=value<=32767 or set(state.get('buttons',[]))-{'ZL','ZR'}:return None
        active=[(t,sign) for t,sign in events if t<=(a+b)/2<t+model['duration_seconds']]
        if active:
            t,sign=active[-1];displacement+=sign*(distance_at(b-t,model)-distance_at(a-t,model))
        else:displacement+=gain*value/32767*(b-a)
    return displacement


class SlideAssist:
    def __init__(self,config):
        self.config=config;self.active={};self.last={'left':-1e9,'right':-1e9}

    def apply(self,obs,decision,now):
        config=self.config
        if config.get('enabled') is not True:return decision,None,None
        model=config['model'];duration=model['duration_seconds'];delay=config.get('delay_seconds',.12)
        p=obs.metrics.get('juggling_actor_input_projection',{});history=obs.metrics.get('juggling_actual_input_history',{}).get('items',[])
        for side,entry in list(self.active.items()):
            actual=[t for t,_ in slide_events(history,side) if entry['requested_at']-.03<=t<=entry['requested_at']+.2]
            if actual:entry['actual_after']=actual[0]
            if now-entry.get('actual_after',entry['requested_at'])>duration+delay+.04:self.active.pop(side)
        targets=decision.diagnostics.get('targets',{});started=[];rejected=[]
        permitted=obs.phase=='playing' and obs.confidence>=.8 and obs.metrics.get('observation_age_seconds',0)<=.2
        if permitted and p.get('accepted') and p.get('held_input_extension_seconds',1)<.08:
            for side,role in [('left','player'),('right','partner')]:
                target=targets.get(side)
                if side in self.active or not target or now-self.last[side]<config.get('minimum_interval_seconds',1.5):continue
                actor=obs.of_kind(role);balls=[e for e in obs.of_kind('ball') if e.track_id==target['track_id']]
                if len(actor)!=1 or len(balls)!=1 or balls[0].samples<6 or balls[0].vy<.25:continue
                actor=actor[0];ball=balls[0];available=target['available_seconds'];delta=target['target_x']-p['positions'][side]
                if not .06<=available<=.38 or abs(delta)<.1:continue
                margin=(actor.w+ball.w)/2-.01
                if abs(delta)<=.666*available+margin:continue
                sign=1 if delta>0 else -1
                low=actor.w/2 if side=='left' else .5+actor.w/2
                high=.5-actor.w/2 if side=='left' else 1-actor.w/2
                contact=max(low,min(high,p['positions'][side]+sign*distance_at(available,model)))
                stop=max(low,min(high,p['positions'][side]+sign*model['distance']))
                tolerance=max(0.,margin-config.get('position_uncertainty',.025))
                if abs(contact-target['target_x'])>tolerance:continue
                conflicts=[q for q in decision.diagnostics.get('intercepts',[]) if q.get('eligible')
                    and q.get('side')==side and q['track_id']!=target['track_id']
                    and q['time_to_contact']<decision.diagnostics['latency_seconds']+duration+.08
                    and abs(q['target_x']-stop)>.1]
                if conflicts:rejected.append({'side':side,'reason':'another_early_contact_after_slide'});continue
                self.active[side]={'requested_at':now,'sign':sign,'track_id':target['track_id'],
                    'predicted_contact_x':contact,'target_x':target['target_x'],'tolerance':tolerance}
                self.last[side]=now;started.append(side)
        axes={'left':decision.state.left_stick.x,'right':decision.state.right_stick.x};buttons=set(decision.state.buttons)
        deadlines=[]
        for side,entry in self.active.items():
            button=Button.ZL if side=='left' else Button.ZR;deadline=entry['requested_at']+.085
            buttons.discard(button);axes[side]=0
            if not permitted:entry['pulse_cancelled']=True
            if now<deadline and not entry.get('pulse_cancelled'):
                buttons.add(button);axes[side]=round(entry['sign']*.4*32767);deadlines.append(deadline)
        state=replace(decision.state,buttons=frozenset(buttons),left_stick=StickVector(axes['left'],0),right_stick=StickVector(axes['right'],0))
        deadline=min(deadlines) if deadlines else None
        release_state=state
        if deadline is not None:
            released=set(buttons);release_axes=dict(axes)
            for side,entry in self.active.items():
                if entry['requested_at']+.085<=deadline+1e-6:
                    released.discard(Button.ZL if side=='left' else Button.ZR);release_axes[side]=0
            release_state=replace(state,buttons=frozenset(released),left_stick=StickVector(release_axes['left'],0),right_stick=StickVector(release_axes['right'],0))
        reason=decision.reason
        if started:reason+='；'+('、'.join('左侧' if s=='left' else '右侧' for s in started))+'实验滑行救球'
        return replace(decision,state=state,reason=reason,diagnostics={**decision.diagnostics,
            'experimental_slide':{'active':self.active.copy(),'started_sides':started,'rejected':rejected,
                                  'scope':'planned_rescue_not_confirmation_of_catch'}}),deadline,release_state
