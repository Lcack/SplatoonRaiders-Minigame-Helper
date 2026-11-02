"""Observed Fever gauge and score pickups, independent of catch control."""
from dataclasses import replace

import cv2
import numpy as np

from .fast_tracking import PredictiveTracker
from .models import Entity


class FeverObserver:
    def __init__(self):
        self.active=False
        self.last_full=None
        self.last_active=None
        self.tracker=PredictiveTracker()

    @staticmethod
    def gauge(hsv):
        tile=hsv[10:24,10:132]
        colour=(tile[:,:,1]>=35)&(tile[:,:,2]>=85)
        columns=np.mean(colour,axis=0)>=.3
        # The actual border is excluded; empty black is not gauge progress.
        return float(np.count_nonzero(columns)/len(columns))

    @staticmethod
    def pickups(hsv,entities):
        cyan=cv2.inRange(hsv,(78,80,150),(118,255,255))
        cyan[:35]=0;cyan[445:]=0
        cyan=cv2.morphologyEx(cyan,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count,_,stats,_=cv2.connectedComponentsWithStats(cyan)
        detections=[]
        for x,y,w,h,pixels in stats[1:]:
            if not (7<=w<=38 and 7<=h<=40 and .65<w/h<1.45 and .18<pixels/(w*h)<.9):continue
            cx,cy=(x+w/2)/640,(y+h/2)/480
            if any(e.kind in {'player','partner','ball'}
                   and abs(cx-e.x)<(w/640+e.w)/2+.006
                   and abs(cy-e.y)<(h/480+e.h)/2+.006 for e in entities):continue
            detections.append(Entity('score_pickup',cx,cy,w/640,h/480,.84))
        return detections

    def observe(self,hsv,entities,timestamp,phase):
        fill=self.gauge(hsv)
        if phase!='playing':
            self.active=False;self.last_full=None;self.last_active=None
            self.tracker.update([],timestamp)
            return (),{'fever_active':False,'fever_gauge_fill':fill,'score_pickup_candidates':0}
        if fill>=.95:
            self.active=True;self.last_full=timestamp
        elif self.active and fill<=.03 and self.last_full is not None and timestamp-self.last_full>.5:
            self.active=False
        if self.active:self.last_active=timestamp
        recent=self.last_active is not None and 0<=timestamp-self.last_active<1.
        found=self.pickups(hsv,entities) if recent else []
        tracked=self.tracker.update(found,timestamp)
        # Distinct namespace from the original actor/ball tracker.
        tracked=tuple(replace(e,track_id=e.track_id+100000) for e in tracked)
        return tracked,{'fever_active':self.active,'fever_gauge_fill':fill,
            'fever_state_source':'observed_full_gauge_then_depletion',
            'fever_state_confidence_kind':'temporal_visual_inference_not_probability',
            'score_pickup_candidates':len(tracked),'score_pickup_is_time_bonus':False,
            'score_pickup_collection_confirmed':False,'pickup_priority':'catch_first'}
