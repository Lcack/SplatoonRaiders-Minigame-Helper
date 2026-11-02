"""Bounded slide pulses from observed middle positions; then return to AI."""
import time

from raiders_helper.control.buttons import Button,ControllerState,StickVector
from raiders_helper.juggling import decide_juggling
from raiders_helper.models import Decision


class SlideProbe:
    def __init__(self,parameters):
        self.parameters=parameters
        cases=[(side,sign,duration) for duration in (.1,.3) for side in ('left','right') for sign in (1,-1)]
        config=parameters.get('slide_probe',{});offset=config.get('case_offset',0);count=config.get('case_count',4)
        self.cases=cases[offset:offset+count]
        self.reset_round()

    def reset_round(self,number=None):
        self.index=0;self.stage='seek_middle';self.since=None;self.ready=0;self.number=number

    def decide(self,obs,now=None):
        now=time.perf_counter() if now is None else now
        if self.index>=len(self.cases):return decide_juggling(obs,self.parameters)
        side,sign,duration=self.cases[self.index]
        details={'probe_id':f'{self.index+1}_{side}_{sign:+d}_{duration:g}',
                 'stage':self.stage,'pulse_seconds':duration,'axis':.4*sign,'experimental':True}
        neutral=ControllerState.neutral()
        def out(state,reason):return Decision(state,'滑行标定：'+reason,diagnostics={'slide_probe':details})
        if obs.phase!='playing' or obs.confidence<.8 or obs.metrics.get('observation_age_seconds',1)>.15:
            self.ready=0;self.since=None;self.stage='seek_middle'
            return out(neutral,'等待连续可靠画面')
        role='player' if side=='left' else 'partner';actors=obs.of_kind(role)
        if len(actors)!=1 or actors[0].confidence<.8:
            self.ready=0;self.since=None;self.stage='seek_middle'
            return out(neutral,'角色缺失，取消脉冲')
        actor=actors[0];home=(0 if side=='left' else .5)+(.12 if sign>0 else .38)
        def state(axis,button=False):
            return ControllerState(frozenset({Button.ZL if side=='left' else Button.ZR}) if button else frozenset(),
                StickVector(round(axis*32767),0) if side=='left' else StickVector(),
                StickVector(round(axis*32767),0) if side=='right' else StickVector())
        if self.stage=='seek_middle':
            delta=home-actor.x;self.ready=self.ready+1 if abs(delta)<.025 else 0
            if self.ready>=3:self.stage='settle';self.since=now
            return out(state(max(-.4,min(.4,delta*6))) if abs(delta)>=.025 else neutral,'回到探针起点并等待停止')
        if self.stage=='settle':
            if now-self.since>=.4:self.stage='pulse';self.since=now
            return out(neutral,'中立等待既有输入消退')
        if self.stage=='pulse':
            edge=actor.x-(0 if side=='left' else .5)
            if now-self.since>=duration or sign<0 and edge<.075 or sign>0 and edge>.425:
                self.stage='coast';self.since=now;return out(neutral,'释放并测量滑行余程')
            return out(state(.4*sign,True),'有边界的 ZL/ZR 短脉冲')
        if now-self.since>=.8:
            self.index+=1;self.stage='seek_middle';self.since=None;self.ready=0
        return out(neutral,'记录松开后的实际位移')
