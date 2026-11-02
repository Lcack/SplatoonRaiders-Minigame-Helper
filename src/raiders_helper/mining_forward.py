"""Optional free-flight vertical response; no controller or vision imports.

Coefficients are supplied by an explicit experiment profile, not hidden game
constants. Collision recovery, powered motion and a terminal ZR speed have
not been fitted. Callers must gate the initial state accordingly.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ForwardDynamics:
    zr_acceleration: float
    normal_speed: float
    normal_tau_seconds: float

    def __post_init__(self):
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0
               for v in (self.zr_acceleration,self.normal_speed,self.normal_tau_seconds)):
            raise ValueError("纵向动力学参数必须为有限正数")

    def advance(self, speed, boost, seconds):
        if not math.isfinite(speed) or speed<0 or not math.isfinite(seconds) or seconds<0:
            raise ValueError("纵向初速/时长必须为有限非负数")
        if boost:
            return speed*seconds+.5*self.zr_acceleration*seconds*seconds,speed+self.zr_acceleration*seconds
        decay=math.exp(-seconds/self.normal_tau_seconds)
        distance=self.normal_speed*seconds+(speed-self.normal_speed)*self.normal_tau_seconds*(1-decay)
        return distance,self.normal_speed+(speed-self.normal_speed)*decay

    def time_to_distance(self, speed, boost, distance):
        if not math.isfinite(distance) or distance<0 or not math.isfinite(speed) or speed<0:
            raise ValueError("纵向距离/初速必须为有限非负数")
        if distance==0:return 0.
        if boost:
            # Rationalized quadratic root avoids cancellation at tiny d.
            return 2*distance/(math.sqrt(speed*speed+2*self.zr_acceleration*distance)+speed)
        low,high=0.,distance/self.normal_speed+self.normal_tau_seconds+distance/max(speed,.01)
        for _ in range(42):
            middle=(low+high)/2
            if self.advance(speed,False,middle)[0]<distance:low=middle
            else:high=middle
        return (low+high)/2

    def arrival_time(self, speed, distance, boost=False, switch_after=None):
        if not boost or switch_after is None:
            return self.time_to_distance(speed,boost,distance)
        prefix,next_speed=self.advance(speed,True,switch_after)
        if prefix>=distance:return self.time_to_distance(speed,True,distance)
        return switch_after+self.time_to_distance(next_speed,False,distance-prefix)


def seed_context(obs, parameters):
    """Return a labelled capture-time state, or an explicit legacy fallback."""
    config=parameters.get("forward_dynamics",{})
    if not isinstance(config,dict) or type(config.get("enabled",False)) is not bool:
        raise ValueError("forward_dynamics.enabled必须为布尔值")
    diagnostics={"enabled":config.get("enabled",False),"active":False,"fallback_reason":"disabled"}
    if not diagnostics["enabled"]:return None,None,diagnostics
    keys=("zr_acceleration","normal_speed","normal_tau_seconds")
    if any(k not in config for k in keys):raise ValueError("启用动态纵向模型需要显式提供加速度、普通速度和时间常数")
    model=ForwardDynamics(**{k:config[k] for k in keys})
    metrics=obs.metrics
    if (obs.phase!="playing" or metrics.get("buff_active") or metrics.get("motion_mode")=="powered"
            or metrics.get("player_appearance")=="powered" or obs.of_kind("egg_buff")
            or metrics.get("collision_suspected") or metrics.get("hit_body_detected")):
        diagnostics["fallback_reason"]="collision_or_powered_response_not_fitted"
        return None,None,diagnostics
    seed=metrics.get("mining_forward_seed")
    if not isinstance(seed,dict) or seed.get("valid") is not True:
        diagnostics["fallback_reason"]="missing_valid_capture_time_seed"
        if isinstance(seed,dict):diagnostics["seed_unavailable_reason"]=seed.get("reason")
        return None,None,diagnostics
    speed=seed.get("speed");timestamp=seed.get("timestamp");age=seed.get("age_seconds")
    def finite(v):return type(v) in (int,float) and math.isfinite(v)
    max_age=config.get("maximum_seed_age_seconds",1.5)
    if not finite(max_age) or not 0<max_age<=1.5:raise ValueError("纵向seed有效期必须在0到1.5秒之间")
    if (not finite(speed) or speed<=0 or not finite(timestamp) or abs(timestamp-obs.timestamp)>1e-6
            or not finite(age) or not 0<=age<=max_age
            or seed.get("units")!="viewport_heights_per_second"):
        diagnostics["fallback_reason"]="invalid_stale_or_misaligned_forward_seed"
        return None,None,diagnostics
    diagnostics.update(active=True,fallback_reason=None,seed_speed=speed,seed_timestamp=timestamp,
        seed_source=seed.get("source"),seed_age_seconds=age,reference_timestamp=seed.get("reference_timestamp"),
        confidence_kind="propagated_model_estimate_not_instantaneous_observation",
        zr_acceleration=model.zr_acceleration,normal_speed=model.normal_speed,normal_tau_seconds=model.normal_tau_seconds,
        terminal_speed_measured=False,collision_or_powered_model=False)
    return model,float(speed),diagnostics
