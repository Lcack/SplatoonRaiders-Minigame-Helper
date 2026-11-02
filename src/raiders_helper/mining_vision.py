"""Fast, inspectable MINING perception for the supervised NS capture.

The initial HSV/shape thresholds come from source-indexed live screenshots.
Scores are heuristic scores, not probabilities. World speed is not inferred
from player.y; missing/occluded robots are not published as real detections.
"""
import json
import math
from pathlib import Path
import time
from collections import deque

import cv2
import numpy as np

from .fast_tracking import PredictiveTracker
from .models import Entity, Game, Observation
from .mining_scroll import WallScrollEstimator
from .vision import crop_roi


ROOT = Path(__file__).resolve().parents[2]
VIEWPORT = [1/6, 55/1080, 2/3, 960/1080]


def _image(path, flags=cv2.IMREAD_GRAYSCALE):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), flags)
    if image is None:
        raise ValueError(f"无法读取视觉模板：{path}")
    return image


class MiningLiveRecognizer:
    """Reference-size 640x480 robot and road observations.

    ``last_road_mask`` is a uint8 canonical image for diagnostics. ``road_rows``
    metrics contain [y,left,right] normalized coordinates. The black trail
    behind the robot is not treated as a forward obstacle. Road rows are an
    outer floor envelope; unknown interior objects remain an explicit limit.
    """
    def __init__(self, game=Game.MINING, profile=None):
        self.game = Game(game)
        if self.game != Game.MINING:
            raise ValueError("MiningLiveRecognizer 仅支持MINING")
        self.profile = profile or {"game":"mining", "viewport":VIEWPORT}
        self.stable_frames = self.profile.get("stable_frames", 2)
        if type(self.stable_frames) is not int or self.stable_frames < 1:
            raise ValueError("stable_frames必须为正整数")
        self.tracker = PredictiveTracker(max_gap=0.20)
        self.last_player = None
        self.last_player_time = None
        self.last_player_appearance = None
        self.last_blue_time = None
        self.last_canonical = None
        self.last_road_mask = None
        self._phase_count = 0
        self._previous_gray = None
        self._previous_road = None
        self._previous_time = None
        self._near_rock_references = {}
        self._landmark_shapes = {}
        self._landmark_measurements = {}
        self._wall_scroll = WallScrollEstimator() if self.profile.get("estimate_wall_scroll",False) else None
        self._score_anchor = _image(ROOT/"assets/arcade/score-anchor.png")
        self._result_rank = _image(ROOT/"assets/session/result-rank.png")
        self._time_zero_bank = [_image(ROOT/f"assets/mining/time-zero-strict-{i}.png") for i in (0,1)]
        self._timer_digits = [_image(ROOT/f"assets/mining/timer-digit-{digit}.png") for digit in range(10)]
        self._screen_matcher = None
        # Reuse explicitly indexed visual anchors, not a guessed game timer.
        if self.profile.get("phase_anchors", True):
            from .session_runtime import ScreenMatcher
            path = Path(self.profile.get("session_profile", ROOT/"profiles/session.json"))
            if not path.is_absolute():
                path = Path(self.profile.get("_root", "."))/path
            config = json.loads(path.read_text(encoding="utf-8"))
            config["screens"] = [s for s in config["screens"]
                                  if s["name"].startswith("menu:") or s["name"] in
                                  ("countdown:mining", "result:retry", "result:quit")]
            config["stable_frames"] = 1
            self._screen_matcher = ScreenMatcher(config, path.parent)

    @staticmethod
    def _components(mask, minimum=20):
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
        return [(stats[i].astype(int), centers[i]) for i in range(1,count)
                if stats[i,cv2.CC_STAT_AREA] >= minimum]

    def _player(self, hsv, timestamp, minimum_blue_hue=95):
        blue = cv2.inRange(hsv, (minimum_blue_hue,80,65), (140,255,255))
        blue[:40] = 0; blue[170:] = 0
        candidates=[]
        for box,center in self._components(blue,65):
            x,y,w,h,pixels=box
            if 12<=w<=48 and 8<=h<=40 and 1.2<=w/h<=3.5 and pixels<=1300:
                # The blue body is smaller than the full green/blue robot.
                # Fixed expansion is a measured sprite-envelope estimate.
                full=(max(0,x-7),max(0,y-11),min(640,x+w+7),min(480,y+h+11))
                cost=abs(center[1]-92)/80
                if self.last_player is not None and self.last_player_time is not None and 0<timestamp-self.last_player_time<.3:
                    predicted=self.last_player.x+self.last_player.vx*(timestamp-self.last_player_time)
                    cost+=abs(center[0]/640-predicted)*5
                cost-=min(pixels,300)/1000
                candidates.append((cost,full,box))
        candidates.sort(key=lambda item:item[0])
        ambiguous=len(candidates)>1 and candidates[1][0]-candidates[0][0]<.12
        if not candidates or ambiguous:
            return None,len(candidates),ambiguous
        _,(left,top,right,bottom),_=candidates[0]
        left,top,right,bottom=map(int,(left,top,right,bottom))
        return Entity("player",(left+right)/1280,(top+bottom)/960,
                      (right-left)/640,(bottom-top)/480,.88),len(candidates),False

    def _hit_player(self, hsv, timestamp):
        """Observe a magenta collision sprite only beside a recent blue body.

        This is a separate measured appearance, never a substitute blue
        detection. HUD text is outside the ROI; red/orange rock explosions
        cannot seed identity. Without recent identity no hit body is emitted.
        """
        if self.last_player is None or self.last_player_time is None:
            return None,0,False
        age=timestamp-self.last_player_time
        if not 0<age<=self.profile.get("hit_identity_max_age",.75):
            return None,0,False
        magenta=cv2.inRange(hsv,(135,100,65),(179,255,255))
        magenta[:60]=0;magenta[130:]=0
        magenta=cv2.morphologyEx(magenta,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        candidates=[]
        previous=self.last_player
        for box,center in self._components(magenta,90):
            x,y,w,h,pixels=map(int,box)
            if not (12<=w<=48 and 16<=h<=44 and .65<=w/h<=1.8 and pixels/(w*h)>=.20):
                continue
            if abs((y+h/2)/480-previous.y)>.035:
                continue
            # Explicitly bounded association: older motion estimates do not
            # allow a distant purple effect to acquire player identity.
            if abs((x+w/2)/640-previous.x)>max(previous.w*1.5,.065):
                continue
            if not (.5<=w/(previous.w*640)<=1.5 and .5<=h/(previous.h*480)<=1.5):
                continue
            cost=abs((x+w/2)/640-previous.x)+abs((y+h/2)/480-previous.y)
            candidates.append((cost,Entity("player_hit",(x+w/2)/640,(y+h/2)/480,w/640,h/480,.80)))
        candidates.sort(key=lambda item:item[0])
        ambiguous=len(candidates)>1 and candidates[1][0]-candidates[0][0]<.025
        return (None if not candidates or ambiguous else candidates[0][1]),len(candidates),ambiguous

    def _powered_player(self, hsv, timestamp):
        """Observe a filled yellow/green body, anchored to actual prior pose.

        Local all-S video pairs 2092->2100, 4328->4335, 5730->5738,
        8078->8085 show this change. A first green screenshot, stale identity
        or arms-only blink cannot initialize/predict a player here.
        """
        if self.last_player is None or self.last_player_time is None:
            return None,0,False
        age=timestamp-self.last_player_time
        maximum=.25 if self.last_player_appearance=="powered" else .75
        if not 0<age<=maximum:return None,0,False
        mask=cv2.inRange(hsv,(30,160,160),(75,255,255))
        mask[:73]=0;mask[123:]=0
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        candidates=[]
        for box,_ in self._components(mask,250):
            x,y,w,h,pixels=map(int,box)
            if not (14<=w<=44 and 18<=h<=42 and .5<=w/h<=1.6 and pixels/(w*h)>=.40):
                continue
            cx,cy=(x+w/2)/640,(y+h/2)/480
            if abs(cx-self.last_player.x)>max(.10,self.last_player.w*2) or abs(cy-self.last_player.y)>.035:
                continue
            cost=abs(cx-self.last_player.x)+abs(cy-self.last_player.y)
            candidates.append((cost,Entity("player",cx,cy,(w+6)/640,(h+4)/480,.80)))
        candidates.sort(key=lambda item:item[0])
        ambiguous=len(candidates)>1 and candidates[1][0]-candidates[0][0]<.025
        return (None if not candidates or ambiguous else candidates[0][1]),len(candidates),ambiguous

    def _zero_digits(self, timer_mask):
        """Verify each 00.0 digit against observed 1..9 competitors.

        A whole-timer correlation can score TIME06.0 above .92 against
        TIME00.0 because only one character changes. It is not evidence that
        the units digit is zero. Keep per-position, per-class diagnostics.
        """
        decisions=[]
        for x in (102,122,162):
            tile=timer_mask[2:31,x-2:x+22]
            score=lambda template: (float(cv2.matchTemplate(tile,template,cv2.TM_CCOEFF_NORMED).max())
                                    if tile.std()>1 and template.std()>1 else 0.0)
            scores=[score(template) for template in self._timer_digits]
            # Reuse the two actual CRT phases at this exact digit position;
            # a partly hidden zero is not compared to an invented glyph.
            zero=max([scores[0]]+[score(template[3:28,x-96:x-76]) for template in self._time_zero_bank])
            nonzero_digit=max(range(1,10),key=lambda digit:scores[digit])
            competitor=scores[nonzero_digit]
            margin=zero-competitor
            accepted=zero>=self.profile.get("zero_digit_min_score",.80) and margin>=self.profile.get("zero_digit_min_margin",.08)
            decisions.append({"x":x/640,"zero_score":zero,"nonzero_digit":nonzero_digit,
                              "nonzero_score":competitor,"margin":margin,"zero_accepted":accepted})
        return decisions

    @staticmethod
    def road_mask(hsv):
        mask=cv2.inRange(hsv,(4,55,15),(47,255,255))
        mask[:35]=0
        # Fill texture stipples; do not bridge wide black outside regions.
        return cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,9),np.uint8))

    @staticmethod
    def _adaptive_floor(hsv):
        """Estimate the changing floor palette, independent of wall gold.

        Source-indexed all-S local video shows yellow, green, cyan and red
        floor. Its modal saturated hue is dominated by floor area; fixed
        brown thresholds mistake the cyan floor for outside. This opt-in
        branch retains only the main floor component, not isolated wall glow.
        """
        valid=(hsv[:,:,1]>=60)&(hsv[:,:,2]>=35)
        valid[:35]=False
        if np.count_nonzero(valid)<1000:return np.zeros(hsv.shape[:2],np.uint8),{}
        quant=(hsv[:,:,0]//6).astype(np.int32)*16+hsv[:,:,1]//16
        counts=np.bincount(quant[valid],minlength=480)
        mode=int(np.argmax(counts))
        hue,saturation=mode//16*6+3,mode%16*16+8
        delta=np.abs(hsv[:,:,0].astype(np.int16)-hue)
        delta=np.minimum(delta,180-delta)
        mask=((delta<=12)&(hsv[:,:,1]>=saturation-32)&(hsv[:,:,2]>=25)).astype(np.uint8)*255
        mask[:35]=0
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,5),np.uint8))
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        if count<2:return np.zeros_like(mask),{}
        largest=int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))+1
        mask=(labels==largest).astype(np.uint8)*255
        return mask,{"hue":hue,"saturation":saturation,"modal_pixels":int(counts[mode])}

    @staticmethod
    def _sprite_occlusion(hsv, player, palette, minimum_blue_hue):
        """Separate visible turning arms from the physics player envelope.

        The R8 raw frame 1947 has a green arm at x189 while the blue-derived
        physics envelope starts at201. Treating that arm as black outside
        splits a clear road. Expand only from actual coloured components
        overlapping the measured body, never from a predicted pose or wall
        margin. The returned foreground mask also preserves wall highlights
        visible between arms; the whole rectangle is not a wall-erasing mask.
        """
        visible=np.zeros(hsv.shape[:2],np.uint8)
        if player is None:return None,visible
        cx,cy=player.x*640,player.y*480
        left=max(0,round((player.x-player.w/2)*640));right=min(640,round((player.x+player.w/2)*640))
        top=max(35,round((player.y-player.h/2)*480));bottom=min(480,round((player.y+player.h/2)*480))
        x0,x1=max(0,math.floor(cx-40)),min(640,math.ceil(cx+40))
        y0,y1=max(35,math.floor(cy-27)),min(480,math.ceil(cy+27))
        tile=hsv[y0:y1,x0:x1]
        hue=tile[:,:,0]
        foreground=((((hue>=45)&(hue<=85))|((hue>=minimum_blue_hue)&(hue<=140)))&
                    (tile[:,:,1]>=70)&(tile[:,:,2]>=120))
        if "hue" in palette:
            delta=np.abs(hue.astype(np.int16)-palette["hue"])
            foreground&=np.minimum(delta,180-delta)>=10
        mask=cv2.morphologyEx(foreground.astype(np.uint8)*255,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        a,b,c,d=left,top,right,bottom
        for index in range(1,count):
            x,y,w,h,pixels=map(int,stats[index])
            if pixels<4 or pixels>1600 or w>70 or h>48:continue
            overlap=labels[max(0,top-y0):min(y1-y0,bottom-y0),
                           max(0,left-x0):min(x1-x0,right-x0)]
            touches_core=pixels>=12 and np.count_nonzero(overlap==index)>=5
            # R8 frame2150 also has an observed 4x4 green/blue particle at
            # x394..398, disconnected by antialiasing from the x353..388
            # body. Such tiny actual pixels can split the floor too. Allow
            # only compact, nearby green/blue pieces, not gold wall ridges.
            gap_x=max(left-(x0+x+w),(x0+x)-right,0)
            gap_y=max(top-(y0+y+h),(y0+y)-bottom,0)
            peripheral=(w<=12 and h<=12 and math.hypot(gap_x,gap_y)<=16 and
                        np.count_nonzero((labels==index)&foreground&(hue>=50))>=3)
            if not touches_core and not peripheral:continue
            # Keep only genuinely coloured pixels for the wall exclusion,
            # not artificial morphological links or the bounding rectangle.
            chosen=(labels==index)&foreground
            visible[y0:y1,x0:x1][chosen]=255
            a,b,c,d=min(a,x0+x),min(b,y0+y),max(c,x0+x+w),max(d,y0+y+h)
        # The live 6529/6824 frames and campaign-01 16116/19512 show
        # small gold feet below the blue core. These are not covered by
        # the blue/green mask and otherwise become a one-row wall cutting
        # through the body. Admit only observed compact lower components
        # touching the established coloured sprite. Do not mask the whole
        # rectangle: a tall gold wall crossing it must remain an obstacle.
        gold=((hsv[:,:,0]>=12)&(hsv[:,:,0]<=48)&
              (hsv[:,:,1]>=40)&(hsv[:,:,1]<=215)&(hsv[:,:,2]>=45)).astype(np.uint8)*255
        _,gold_labels,gold_stats,_=cv2.connectedComponentsWithStats(gold)
        # Antialiasing separates the gold feet in 16371/19512 by four
        # canonical pixels from the blue/green foreground.
        adjacent=cv2.dilate(visible,np.ones((9,9),np.uint8))
        adjacent_tiny=cv2.dilate(visible,np.ones((13,13),np.uint8))
        for index,box in enumerate(gold_stats[1:],1):
            x,y,w,h,pixels=map(int,box)
            # Actual frame 206040 has a 7x5/15px toe eight pixels beyond
            # the blue-derived box. Keep compactness/current-colour/adjacency
            # requirements; only its small horizontal allowance increases.
            if not (4<=pixels<=160 and w<=26 and h<=14 and y>=cy and
                    left-8<=x and x+w<=right+8 and y+h<=bottom+6):
                continue
            chosen=gold_labels[y:y+h,x:x+w]==index
            if np.count_nonzero(chosen&(adjacent[y:y+h,x:x+w]>0))<3:
                # 206040's tiny measured toe is six pixels from coloured
                # foreground. Do not extend this allowance to larger ridges
                # or chain through gold pixels accepted earlier in the loop.
                if not (w<=8 and h<=6 and pixels<=20 and
                        np.count_nonzero(chosen&(adjacent_tiny[y:y+h,x:x+w]>0))>=3):continue
            visible[y:y+h,x:x+w][chosen]=255
        return Entity("sprite_occlusion",(a+c)/1280,(b+d)/960,(c-a)/640,(d-b)/480,.80),visible

    @staticmethod
    def _floor_inner_edges(hsv, rows, palette, player, sprite_pixels=None):
        """Use the innermost gold ridge highlight when floor and wall overlap.

        Deep green/cyan floor is already separated from gold by hue. Brown
        floor can share wall hue; lower-saturation ridge highlights provide
        extra evidence. Avoid using player pixels as a wall. These measured
        bounds remain experimental, with the applied shifts exposed.
        """
        hue=palette.get("hue")
        if hue is None or 36<=hue<=166:return rows,[]
        saturation=palette["saturation"]
        result=[];adjustments=[]
        for yf,left,right in rows:
            y=round(yf*480);a,b=round(left*640),round(right*640)
            strip=hsv[max(35,y-2):min(480,y+3)]
            # Floor saturation also varies across the CRT's shaded bands.
            # Comparing every row with the frame-wide bright-floor mode
            # mistakes a whole shaded floor band for a wall highlight.
            middle=strip[:,min(b,a+30):max(a+30,b-30)]
            delta=np.abs(middle[:,:,0].astype(np.int16)-hue)
            valid=(np.minimum(delta,180-delta)<=12)&(middle[:,:,2]>=35)&(middle[:,:,1]>=60)
            local_saturation=saturation
            if np.count_nonzero(valid)>=20:
                histogram=np.bincount((middle[:,:,1][valid]//16).astype(int),minlength=16)
                local_saturation=int(np.argmax(histogram))*16+8
            gold=((strip[:,:,0]>=12)&(strip[:,:,0]<=48)&
                  (strip[:,:,1]<=min(215,local_saturation-32))&(strip[:,:,2]>=45))
            if sprite_pixels is not None:
                gold[sprite_pixels[max(35,y-2):min(480,y+3)]>0]=False
            elif player and abs(y-player.y*480)<player.h*240+6:
                lo=max(0,int((player.x-player.w/2)*640)-6)
                hi=min(640,int((player.x+player.w/2)*640)+6)
                gold[:,lo:hi]=False
            occupied=np.mean(gold,axis=0)>=.4
            edge=np.diff(np.r_[False,occupied,False].astype(np.int8))
            runs=[(int(x),int(z)) for x,z in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)) if z-x>=2]
            l=[z for x,z in runs if a<=x<z<=min(b-55,a+60)]
            r=[x for x,z in runs if max(a+55,b-60)<=x<z<=b]
            new_a=max(a,max(l)+3) if l else a
            new_b=min(b,min(r)-3) if r else b
            if new_b-new_a<55:continue
            if new_a!=a or new_b!=b:
                adjustments.append([yf,(new_a-a)/640,(b-new_b)/640])
            result.append([yf,new_a/640,new_b/640])
        return result,adjustments

    def _time_stars(self, hsv, player):
        """Bright cyan diamond candidates, distinct from the blue robot.

        Source: all-S local MINING video 210s and live star groups. Only
        future objects below the full body band are emitted; HUD, the robot
        seed and yellow +time text cannot acquire star identity here.
        """
        mask=cv2.inRange(hsv,(75,70,180),(115,255,255))
        front=math.ceil((player.y+player.h/2)*480)+2 if player else 120
        mask[:front]=0;mask[455:]=0
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        boxes=[]
        for box,_ in self._components(mask,12):
            x,y,w,h,pixels=map(int,box)
            if 5<=w<=23 and 5<=h<=23 and .55<=w/h<=1.8 and pixels<=250 and .12<=pixels/(w*h)<=.85:
                boxes.append((x,y,x+w,y+h))
        # Resampling can separate the bright diamond into several fragments.
        # Merge only within one measured glyph envelope, never neighbouring
        # star positions in a group (the source group is spaced ~40px).
        groups=[]
        for box in boxes:
            left,top,right,bottom=box
            for index,(a,b,c,d) in enumerate(groups):
                width=max(right,c)-min(left,a);height=max(bottom,d)-min(top,b)
                if width<=25 and height<=25 and math.hypot((left+right-a-c)/2,(top+bottom-b-d)/2)<18:
                    groups[index]=(min(left,a),min(top,b),max(right,c),max(bottom,d))
                    break
            else:groups.append(box)
        result=[]
        for a,b,c,d in groups:
            entity=Entity("time_star",(a+c)/1280,(b+d)/960,(c-a)/640,(d-b)/480,.80)
            result.append(entity)
            density=float(np.mean(mask[b:d,a:c]>0))
            complete=(.65<=(c-a)/(d-b)<=1.55 and .20<=density<=.80)
            self._landmark_shapes[self._landmark_key(entity)]={
                "source":"current_cyan_diamond","complete":complete,"density":density}
        return result

    @staticmethod
    def _landmark_key(entity):
        return (entity.kind,entity.x,entity.y,entity.w,entity.h)

    @staticmethod
    def outer_floor(mask):
        """Keep connected road silhouettes, filling only enclosed inner holes."""
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        result=np.zeros_like(mask)
        major=[c for c in contours if cv2.contourArea(c)>=1000]
        cv2.drawContours(result,major,-1,255,cv2.FILLED)
        return result

    def _near_rock_identity(self, candidate, timestamp):
        """Associate an actual near-body contour; never generate a missing rock."""
        if not self.profile.get("retain_tracked_near_rocks",False) or timestamp is None:
            return None
        matches=[]
        for track_id,(previous,last,_) in self.tracker.tracks.items():
            dt=timestamp-last
            if previous.kind!="obstacle" or previous.samples<4 or not 0<dt<=.20:
                continue
            reference=self._near_rock_references.get(track_id,(previous.w,previous.h,last))
            width_ratio,height_ratio=candidate.w/reference[0],candidate.h/reference[1]
            if not (.85<=width_ratio<=1.20 and .85<=height_ratio<=1.20):
                continue
            dx=abs(candidate.x-(previous.x+previous.vx*dt))
            dy=abs(candidate.y-(previous.y+previous.vy*dt))
            if abs(candidate.x-previous.x)>.025 or dx>.025 or dy>.035:
                continue
            matches.append((dx+dy,track_id,previous,reference,last))
        matches.sort(key=lambda item:item[0])
        if not matches or (len(matches)>1 and matches[1][0]-matches[0][0]<.01):
            return None
        _,track_id,previous,reference,last=matches[0]
        # Keep the first intact dimensions, not a shrinking sequence of
        # fragment sizes. This is a limited continuity gate, not a general
        # visual classifier of all possible rock-breaking animations.
        self._near_rock_references[track_id]=(reference[0],reference[1],timestamp)
        return {"prior_track_id":track_id,"prior_samples":previous.samples,
                "identity_age_seconds":timestamp-last,"reference_width":reference[0],
                "reference_height":reference[1],"source":"current_compact_contour_with_recent_rock_identity"}

    def _rocks(self, hsv, outer, player, gray=None, timestamp=None):
        # Actual NS boulders have a compact dark center inside a lighter rim.
        # Wall scallops and long floor shadows must not become separate rocks.
        distance=cv2.distanceTransform(outer,cv2.DIST_L2,3)
        dark=((hsv[:,:,2]<80)&(distance>18)).astype(np.uint8)*255
        dark[:40]=0;dark[455:]=0
        if player:
            x0=max(0,round((player.x-player.w/2)*640)-5)
            x1=min(640,round((player.x+player.w/2)*640)+5)
            y0=max(0,round((player.y-player.h/2)*480)-5)
            y1=min(480,round((player.y+player.h/2)*480)+5)
            dark[y0:y1,x0:x1]=0
        dark=cv2.morphologyEx(dark,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(dark,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        result=[]
        self._unverified_near_body_dark=[]
        self._retained_near_body_rocks=[]
        if timestamp is not None:
            self._near_rock_references={key:value for key,value in self._near_rock_references.items()
                                       if 0<=timestamp-value[2]<=.20}
        for c in contours:
            x,y,w,h=cv2.boundingRect(c)
            area=cv2.contourArea(c);perimeter=cv2.arcLength(c,True)
            compact=4*math.pi*area/(perimeter*perimeter) if perimeter else 0
            if 14<=w<=140 and 14<=h<=140 and .55<w/h<1.8 and area>=100 and area/(w*h)>.35 and compact>.30:
                pad=7
                x0,y0=max(0,x-pad),max(0,y-pad)
                x1,y1=min(640,x+w+pad),min(480,y+h+pad)
                candidate=Entity("obstacle",(x0+x1)/1280,(y0+y1)/960,
                                 (x1-x0)/640,(y1-y0)/480,.82)
                self._landmark_shapes[self._landmark_key(candidate)]={
                    "source":"current_compact_contour","complete":bool(
                        compact>=.55 and area/(w*h)>=.50 and .70<=w/h<=1.40),
                    "compactness":compact,"fill_ratio":area/(w*h)}
                if player is not None and candidate.y<=player.y+player.h/2:
                    # A previously observed stone beside the robot can still
                    # be intact; passing its center does not prove contact.
                    # Without both current shape and recent identity, keep
                    # near-body dark wake/shadow candidates diagnostic only.
                    identity=self._near_rock_identity(candidate,timestamp)
                    if identity is not None:
                        result.append(candidate)
                        self._retained_near_body_rocks.append({"x":candidate.x,"y":candidate.y,
                            "w":candidate.w,"h":candidate.h,**identity})
                        continue
                    self._unverified_near_body_dark.append({
                        "x":candidate.x,"y":candidate.y,"w":candidate.w,"h":candidate.h,
                        "reason":"dark_center_not_ahead_of_visible_body"})
                    continue
                result.append(candidate)
        self._circle_rock_evidence=[]
        self._circle_rock_added=0
        self._circle_rock_ms=0.0
        if gray is not None and self.profile.get("circle_rocks",False):
            started=time.perf_counter()
            near_identity=(lambda candidate:self._near_rock_identity(candidate,timestamp)) if self.profile.get("retain_tracked_near_rocks",False) else None
            extra,evidence=self._ring_rocks(gray,distance,player,result,near_identity=near_identity)
            self._circle_rock_ms=(time.perf_counter()-started)*1000
            self._circle_rock_evidence=evidence
            self._circle_rock_added=len(extra)
            for ring in evidence:
                if ring.get("near_rock_identity") is not None:
                    diameter=2*(ring["radius_pixels"]+3)
                    self._retained_near_body_rocks.append({"x":ring["x"],"y":ring["y"],
                        "w":diameter/640,"h":diameter/480,**ring["near_rock_identity"],
                        "source":"current_small_rim_with_recent_rock_identity"})
            result.extend(extra)
        self._large_rock_evidence=[]
        self._large_rock_added=0
        self._large_rock_ms=0.0
        if gray is not None and self.profile.get("large_rocks",False):
            started=time.perf_counter()
            near_identity=(lambda candidate:self._near_rock_identity(candidate,timestamp)) if (
                player is not None and self.profile.get("retain_tracked_near_rocks",False)) else None
            extra,evidence=self._ring_rocks(gray,distance,player,result,large=True,near_identity=near_identity)
            self._large_rock_ms=(time.perf_counter()-started)*1000
            self._large_rock_evidence=evidence
            self._large_rock_added=len(extra)
            for ring in evidence:
                if ring.get("near_rock_identity") is not None and not ring["merged_existing"]:
                    diameter=2*(ring["radius_pixels"]+3)
                    self._retained_near_body_rocks.append({"x":ring["x"],"y":ring["y"],
                        "w":diameter/640,"h":diameter/480,"rim_sectors":ring["rim_sectors"],
                        **ring["near_rock_identity"],"source":"current_large_rim_with_recent_rock_identity"})
            # A large ring can enclose several dark lobes. Its measured rim
            # is the obstacle envelope; do not retain tiny lobe duplicates.
            for large in extra:
                result=[e for e in result if not (abs(e.x-large.x)+e.w/2<=large.w/2 and
                                                   abs(e.y-large.y)+e.h/2<=large.h/2)]
            result.extend(extra)
        for entity in result:
            rings=[ring for ring in self._circle_rock_evidence+self._large_rock_evidence
                   if math.hypot((entity.x-ring["x"])*640,(entity.y-ring["y"])*480)<8]
            if rings:
                ring=max(rings,key=lambda r:r["rim_sectors"])
                self._landmark_shapes[self._landmark_key(entity)]={
                    "source":"current_bright_ring","complete":bool(ring["rim_sectors"]>=10 and ring["rim_contrast"]>=20),
                    "rim_sectors":ring["rim_sectors"],"rim_contrast":ring["rim_contrast"]}
        return result

    @staticmethod
    def _ring_rocks(gray, floor_distance, player, existing, large=False, near_identity=None):
        """Supplement compact dark blobs with a bright, closed stone rim.

        R6 problem frames 39385/39439/39488 show a real round stone merging
        into dark floor texture under the old V<80 test. Smooth away the CRT
        texture, search circles at half size, then demand an interior floor
        position and a brighter ring in at least 9 of 12 directions. A large
        irregular dark floor region never becomes a rock by relaxing its
        original shape test.
        """
        if not np.any(floor_distance>38) or gray[120:455].std()<2:
            return [],[]
        smooth=cv2.GaussianBlur(gray,(0,0),1.8)
        small=cv2.resize(smooth,(320,240),interpolation=cv2.INTER_AREA)
        # The independent large scale comes from the user video at 90s:
        # its ~78px outer radius cannot be found by the 16..36px search.
        # It has a separate opt-in switch; enabling it does not enable the
        # original small-circle experiment.
        circles=cv2.HoughCircles(small,cv2.HOUGH_GRADIENT,dp=1.2,minDist=35 if large else 18,
                                 param1=65,param2=38 if large else 19,
                                 minRadius=24 if large else 8,maxRadius=55 if large else 18)
        if circles is None:return [],[]
        front=(math.ceil((player.y+player.h/2)*480)+2 if player else 120)
        result=[];evidence=[]
        for raw_x,raw_y,raw_radius in circles[0]*2:
            x,y,radius=map(float,(raw_x,raw_y,raw_radius))
            if not 40<y<450 or not 0<=round(x)<640:
                continue
            # The whole inferred circle must fit inside the actual outer
            # road. Wall scallops and outside black pockets cannot qualify.
            inside_floor=float(floor_distance[round(y),round(x)])>=radius+2
            needs_identity=y<=front or not inside_floor
            if needs_identity:
                # Actual side boulders 18430/58010 keep a bright intact rim,
                # but the nearby body/wake cuts their floor-distance circle.
                # Only a mature observed rock may bypass these two
                # positional gates; unknown wall rings retain the old gate.
                if (near_identity is None or player is None or y-radius>front or
                        float(floor_distance[round(y),round(x)])<18):
                    continue
            x0,x1=max(0,int(x-radius*1.2)),min(640,int(x+radius*1.2)+1)
            y0,y1=max(0,int(y-radius*1.2)),min(480,int(y+radius*1.2)+1)
            yy,xx=np.ogrid[y0:y1,x0:x1]
            radial=np.sqrt((xx-x)**2+(yy-y)**2)
            angular=((np.arctan2(yy-y,xx-x)+math.pi)/(2*math.pi)*12).astype(int)%12
            tile=smooth[y0:y1,x0:x1]
            inside=radial<radius*.65
            ring=(radial>radius*.82)&(radial<radius*1.08)
            contrast=float(np.mean(tile[ring])-np.mean(tile[inside]))
            if contrast<15:continue
            supported=0
            for sector in range(12):
                inner=tile[inside&(angular==sector)]
                rim=tile[ring&(angular==sector)]
                supported+=bool(len(inner) and len(rim) and float(np.mean(rim)-np.mean(inner))>8)
            if supported<9:continue
            pad=radius+3
            candidate=Entity("obstacle",x/640,y/480,2*pad/640,2*pad/480,.80)
            identity=near_identity(candidate) if needs_identity else None
            if needs_identity and identity is None:
                continue
            comparison=result if large else existing+result
            duplicate=any(math.hypot((candidate.x-e.x)*640,(candidate.y-e.y)*480)<
                          max(18,min(e.w*640,e.h*480)*.55) for e in comparison)
            evidence.append({"x":candidate.x,"y":candidate.y,"radius_pixels":radius,
                             "rim_contrast":contrast,"rim_sectors":supported,
                             "merged_existing":duplicate,"source":"large_smoothed_bright_ring" if large else "smoothed_bright_ring",
                             "near_rock_identity":identity})
            if not duplicate:result.append(candidate)
        return result,evidence

    @staticmethod
    def _restore_player_floor(mask, player):
        """Fill only the visible robot's occlusion with floor on both sides.

        Do not widen an outside edge and do not erase a black forward region.
        The black wake above the sprite remains outside the repaired bbox.
        """
        if player is None:return mask
        left=max(0,round((player.x-player.w/2)*640))
        right=min(640,round((player.x+player.w/2)*640))
        top=max(35,round((player.y-player.h/2)*480))
        bottom=min(480,round((player.y+player.h/2)*480))
        mask=mask.copy()
        for y in range(top,bottom):
            a=np.flatnonzero(mask[y,max(0,left-15):left])
            b=np.flatnonzero(mask[y,right:min(640,right+15)])
            if len(a) and len(b):
                mask[y,max(0,left-15)+int(a[-1]):right+int(b[0])+1]=255
        return mask

    def _merge_player_wake(self, spans, mask, player, y, timestamp=None):
        """Join a narrow *rear* wake only where forward floor proves a bridge.

        This changes the local road envelope, not the binary image. A narrow
        gap alone is insufficient: it must follow the current/recent body x,
        remain within its rear/body envelope, and end before the forward floor band. Thus
        a forward black indentation or continuing branch divider stays hard.
        """
        if player is None or not self.profile.get("repair_player_wake",True):
            return spans
        cx,cy,width,height=player.x*640,player.y*480,player.w*640,player.h*480
        body_top,body_bottom=cy-height/2,cy+height/2
        if not cy-height<=y<=body_bottom:
            return spans
        maximum=width*self.profile.get("wake_max_body_widths",2.0)
        anchors=[cx]
        if (timestamp is not None and self.last_player is not None and self.last_player_time is not None
                and 0<=timestamp-self.last_player_time<=.2):
            anchors.append(self.last_player.x*640)
        front_y=min(474,math.ceil(cy+height/2)+2)
        front=np.mean(mask[front_y:min(480,front_y+5)]>0,axis=0)>=.4
        spans=list(spans)
        index=0
        while index<len(spans)-1:
            (left,end),(begin,right)=spans[index:index+2]
            gap=begin-end
            # At a body scanline the black wake and non-floor sprite can be
            # one connected gap. Count only the portion outside the measured
            # sprite envelope against the wake-width limit. Otherwise a
            # curved 45px wake plus a 34px sprite is rejected as a 79px wake.
            sprite_overlap=(max(0,min(begin,cx+width/2)-max(end,cx-width/2))
                            if body_top<=y<=body_bottom else 0)
            if not 0<gap<=maximum+sprite_overlap:
                index+=1
                continue
            # Allow only the few antialiased pixels just outside the measured
            # blue-derived sprite bbox, rather than jumping to another road.
            attachment=self.profile.get("wake_attachment_margin_pixels",3)
            if self.profile.get("repair_tiny_wake_gaps",False) and gap<=5:
                # Raw NS 205745 leaves a five-pixel rear hole seven pixels
                # beyond the blue-derived physical bbox (four beyond the
                # observed sprite envelope). Only this tiny local gap gets
                # four extra pixels of reach; the real forward-floor bridge
                # below is still mandatory, and wider gaps are unchanged.
                attachment=max(attachment,7)
            if not any(end-width*.5-attachment<=anchor<=begin+width*.5+attachment for anchor in anchors):
                index+=1
                continue
            # Require real floor across both gap and its immediate shoulders.
            # A divider that continues in front of the body cannot pass this.
            bridge=front[max(left,end-5):min(right,begin+5)]
            if not len(bridge) or float(np.mean(bridge))<.8:
                index+=1
                continue
            # A wake can contain a brown sprite/texture island, splitting one
            # local occlusion into several gaps. Prove and join each original
            # gap, then recheck its next neighbour; selecting just one longest
            # pair leaves a fake wall through the body at adjacent scanlines.
            self._wake_repairs.append([y/480,end/640,begin/640])
            spans[index:index+2]=[(left,right)]
        return spans

    def _road_rows(self, mask, player, timestamp=None):
        values=self.profile.get("scan_y")
        if values is None:
            # Include the player's full envelope, and sample forward densely.
            values=([.14,.16,.18,.20,.22,.25,.28,.32,.37,.43,.50,.58,.68,.80,.91])
            if player:
                values += [max(.08,player.y-player.h/2),player.y,
                           min(.95,player.y+player.h/2)]
        rows=[]
        self._wake_repairs=[]
        previous=None
        for yf in sorted(set(float(y) for y in values if 0<y<.97)):
            y=round(yf*479)
            strip=mask[max(35,y-2):min(480,y+3)]
            if strip.size==0:continue
            occupied=np.mean(strip>0,axis=0)>=.4
            # Short closed gaps are pixel texture, not distinct road branches.
            edges=np.diff(np.r_[False,occupied,False].astype(np.int8))
            # Retain short shoulders long enough to test a local rear wake;
            # they are not standalone corridors unless the bridge is proven.
            spans=[(int(a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=12]
            spans=self._merge_player_wake(spans,mask,player,y,timestamp)
            spans=[(left,right) for left,right in spans if right-left>=55]
            if not spans:continue
            if previous:
                # Follow the connected outer-road envelope, not a stray item.
                candidates=[(a,b) for a,b in spans if min(b,previous[1])-max(a,previous[0])>0]
                spans=candidates or spans
            left,right=max(spans,key=lambda span:span[1]-span[0])
            previous=(left,right)
            rows.append([y/480,left/640,right/640])
        return rows

    def _flow(self, gray, road, timestamp):
        result={"forward_speed_observed":False,"forward_speed_confidence":0.0,
                "forward_speed_source":"background_flow", "forward_speed_units":"viewport_heights_per_second"}
        if self._previous_time is not None and .008<=timestamp-self._previous_time<=.20:
            mask=self._previous_road.copy();mask[:150]=0;mask[440:]=0
            points=cv2.goodFeaturesToTrack(self._previous_gray,maxCorners=100,qualityLevel=.03,minDistance=10,mask=mask)
            if points is not None and len(points)>=12:
                nxt,status,error=cv2.calcOpticalFlowPyrLK(self._previous_gray,gray,points,None,
                    winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.03))
                good=(status.ravel()>0)&(error.ravel()<15)
                if np.count_nonzero(good)>=12:
                    before,after=points[good].reshape(-1,2),nxt[good].reshape(-1,2)
                    transform,inliers=cv2.estimateAffinePartial2D(before,after,method=cv2.RANSAC,ransacReprojThreshold=1.5)
                    if transform is not None:
                        fraction=float(np.mean(inliers))
                        dt=timestamp-self._previous_time
                        dx,dy=float(transform[0,2]),float(transform[1,2])
                        result.update(background_dx_pixels=dx,background_dy_pixels=dy,
                            background_flow_inlier_fraction=fraction,background_flow_points=int(len(before)))
                        # This is only a visual proxy until a moving-world
                        # landmark check rules out screen-anchored CRT texture.
                        result["forward_speed_candidate"]=-dy/dt/480
        self._previous_gray,self._previous_road,self._previous_time=gray,road,timestamp
        return result

    @staticmethod
    def _current_obstacle_shapes(entities, evidence, timestamp):
        """Only current measured bright rims associated to final track IDs.

        This records visible geometry, not a calibrated physical collision
        circle. Consumers must retain their conservative radius hypothesis.
        """
        obstacles=[e for e in entities if e.kind=="obstacle" and type(e.track_id) is int and e.track_id>=0]
        associated={}
        for ring in evidence:
            values=[ring.get(key) for key in ("x","y","radius_pixels","rim_sectors","rim_contrast")]
            if not all(isinstance(value,(int,float)) and math.isfinite(value) for value in values):continue
            x,y,radius,sectors,contrast=values
            if radius<=0 or sectors<9 or contrast<15:continue
            diameter=2*(radius+3)
            matches=[]
            for entity in obstacles:
                if not all(math.isfinite(value) for value in (entity.x,entity.y,entity.w,entity.h)):continue
                separation=math.hypot((entity.x-x)*640,(entity.y-y)*480)
                side=max(entity.w*640,entity.h*480)
                if separation<=8 and abs(side-diameter)<=max(12,.25*diameter):matches.append(entity)
            if len(matches)!=1:continue
            entity=matches[0]
            associated.setdefault(entity.track_id,[]).append({
                "track_id":entity.track_id,"shape":"circle","center_x_pixels":x*640,
                "center_y_pixels":y*480,"radius_pixels":radius,
                "coordinate_space":"canonical_640x480_pixels","source":"current_bright_ring",
                "rim_sectors":int(sectors),"rim_contrast":contrast,"timestamp":timestamp,
                "association":"unique_current_obstacle",
                "confidence_kind":"visual_rim_evidence_not_measured_collision_box"})
        # Multiple current rims for one published track are ambiguous too.
        return [items[0] for items in associated.values() if len(items)==1]

    def _update_landmark_measurements(self, entities, timestamp):
        # Retain only actual detections and their measured dimensions. The
        # predictive track's position/velocity is never a shape observation.
        self._landmark_measurements={key:history for key,history in self._landmark_measurements.items()
                                     if history and 0<timestamp-history[-1][0]<=.20}
        for entity in entities:
            if entity.kind not in {"obstacle","time_star"}:continue
            shape=self._landmark_shapes.get(self._landmark_key(entity),{})
            if not shape.get("complete"):
                self._landmark_measurements.pop(entity.track_id,None)
                continue
            history=self._landmark_measurements.setdefault(entity.track_id,deque(maxlen=4))
            history.append((timestamp,entity,shape))

    @staticmethod
    def _temporal_landmark_candidate(history, timestamp, minimum_samples=8):
        if len(history)<4:return None
        history=list(history)[-4:]
        last=history[-1][1]
        if last.samples<minimum_samples or abs(history[-1][0]-timestamp)>1e-7:return None
        if any(not shape.get("complete") for _,_,shape in history):return None
        if any(not .025<=b[0]-a[0]<=.20 for a,b in zip(history,history[1:])):return None
        span=history[-1][0]-history[0][0]
        if not .12<=span<=.60:return None
        entities=[e for _,e,_ in history]
        if any(e.confidence<.8 or e.track_id!=last.track_id or e.kind!=last.kind for e in entities):return None
        if any(not math.isfinite(v) for e in entities for v in (e.x,e.y,e.w,e.h)):return None
        if max(e.x for e in entities)-min(e.x for e in entities)>3/640:return None
        for key in ("w","h"):
            sizes=[getattr(e,key) for e in entities]
            if min(sizes)<=0 or max(sizes)/min(sizes)>1.15:return None
        # No cropped object, HUD, body-adjacent fragment or wake can supply
        # the lower-evidence observation, even if its identity is mature.
        if any(e.y-e.h/2<.25 or e.y+e.h/2>.93 or e.x-e.w/2<=0 or e.x+e.w/2>=1
               for e in entities):return None
        times=[t-history[0][0] for t,_,_ in history];ys=[e.y for e in entities]
        mean_t=sum(times)/4;mean_y=sum(ys)/4
        denominator=sum((t-mean_t)**2 for t in times)
        slope=sum((t-mean_t)*(y-mean_y) for t,y in zip(times,ys))/denominator
        residual=max(abs(y-(mean_y+slope*(t-mean_t))) for t,y in zip(times,ys))*480
        if residual>2:return None
        speed=-(ys[-1]-ys[0])/span
        if not -.02<=speed<=1.5:return None
        pairs=[(-(b[1].y-a[1].y)/(b[0]-a[0]),(a[0]+b[0])/2,b[0]-a[0])
               for a,b in zip(history,history[1:])]
        for (va,ta,dta),(vb,tb,dtb) in zip(pairs,pairs[1:]):
            # Experimental acceleration bounds plus a 2px localization
            # allowance; a single quantized frame is not acceleration truth.
            tolerance=2/480*(1/dta+1/dtb)
            if vb-va>.35*(tb-ta)+tolerance or vb-va< -1.5*(tb-ta)-tolerance:return None
        return {"entity":last,"speed":max(0.0,speed),"reference":(history[0][0]+timestamp)/2,
                "span":span,"residual_pixels":residual,"last_pair":history[-2:],
                "shape_source":history[-1][2]["source"]}

    @staticmethod
    def _sparse_landmark_scroll(entities, histories, timestamp):
        result={"current_forward_speed_observed":False,"current_forward_speed_confidence":0.0,
                "sparse_landmark_scroll_enabled":True,"sparse_landmark_shape_track_count":0}
        candidates=[]
        for entity in entities:
            if entity.kind not in {"obstacle","time_star"}:continue
            candidate=MiningLiveRecognizer._temporal_landmark_candidate(histories.get(entity.track_id,()),timestamp,6)
            if candidate is not None:candidates.append(candidate)
        result["sparse_landmark_shape_track_count"]=len(candidates)
        groups={}
        for candidate in candidates:
            a,b=candidate["last_pair"];dt=b[0]-a[0]
            vx=(b[1].x-a[1].x)/dt;speed=-(b[1].y-a[1].y)/dt
            if abs(vx)<.03 and -.02<=speed<=1.5:
                groups.setdefault(a[0],[]).append((candidate,max(0.0,speed)))
        pair=next((group for group in groups.values() if len(group)==2),None)
        if pair is not None:
            speeds=[value for _,value in pair];speed=float(np.mean(speeds))
            a,b=(item[0]["entity"] for item in pair)
            if (math.hypot((a.x-b.x)*640,(a.y-b.y)*480)<12 or
                    abs(speeds[0]-speeds[1])>max(.025,.05*speed)):return result
            selected=[item[0] for item in pair]
            start=selected[0]["last_pair"][0][0];span=timestamp-start;reference=(start+timestamp)/2
            source="dual_landmark_consensus";method="paired_positions";confidence=.82
            reference_kind="pair_midpoint_velocity_approximation"
        elif len(candidates)==1 and candidates[0]["entity"].samples>=8:
            selected=candidates;candidate=selected[0];speed=candidate["speed"]
            reference=candidate["reference"];span=candidate["span"]
            source="temporal_single_landmark";method="temporal_positions";confidence=.80
            reference_kind="temporal_window_midpoint_velocity_approximation"
        else:return result
        result.update(current_forward_speed_observed=True,current_forward_speed=float(speed),
            current_forward_speed_confidence=confidence,current_forward_speed_source=source,
            current_forward_speed_method=method,current_forward_speed_units="viewport_heights_per_second",
            current_forward_speed_confidence_kind="heuristic_gate_score_not_probability",
            current_forward_speed_reference_timestamp=reference,current_forward_speed_window_seconds=span,
            current_forward_speed_reference_kind=reference_kind,
            current_forward_speed_track_ids=[item["entity"].track_id for item in selected],
            current_forward_speed_track_count=len(selected),
            current_forward_speed_shape_sources=[item["shape_source"] for item in selected],
            current_forward_speed_temporal_residual_pixels=max(item["residual_pixels"] for item in selected))
        return result

    @staticmethod
    def _landmark_scroll(entities, *, tracks=None, timestamp=None, method="tracker_window"):
        """Current scroll from independent tracked stones/stars, never body y.

        This measures the tracker window, not future ZR terminal speed.
        Confidence is a gate score, not a calibrated probability. No value
        is retained when the current frame loses landmark agreement.
        """
        result={"current_forward_speed_observed":False,"current_forward_speed_confidence":0.0,
                "current_forward_speed_source":"tracked_landmark_consensus",
                "current_forward_speed_units":"viewport_heights_per_second",
                "current_forward_speed_confidence_kind":"heuristic_gate_score_not_probability",
                "current_forward_speed_method":method,
                "current_forward_speed_track_ids":[],"current_forward_speed_track_count":0}
        if method not in {"tracker_window","paired_positions"}:return result
        candidates=[];identities=set();measurements={}
        for entity in entities:
            if (entity.kind not in {"obstacle","time_star"} or entity.samples<4 or
                    type(entity.track_id) is not int or entity.track_id<0 or entity.track_id in identities):
                continue
            values=(entity.x,entity.y,entity.vx,entity.vy,entity.confidence)
            if not all(isinstance(value,(int,float)) and math.isfinite(value) for value in values):
                continue
            vx,vy=entity.vx,entity.vy
            reference=None
            stored=(tracks or {}).get(entity.track_id)
            if stored is not None and len(stored[2])>=2:
                history=stored[2]
                first=history[-2] if method=="paired_positions" else history[0]
                last=history[-1];span=last[0]-first[0]
                if (timestamp is not None and abs(last[0]-timestamp)<1e-7 and span>0):
                    reference={"start":first[0],"midpoint":(first[0]+last[0])/2,"span":span}
                    if method=="paired_positions":
                        vx=(last[1]-first[1])/span;vy=(last[2]-first[2])/span
            if method=="paired_positions" and (reference is None or not .025<=reference["span"]<=.20):
                continue
            if entity.confidence<.8 or abs(vx)>=.03 or not -.02<=-vy<=1.5:
                continue
            # Two detector envelopes at the same center are one landmark.
            if any(math.hypot((entity.x-other.x)*640,(entity.y-other.y)*480)<12 for other in candidates):
                continue
            identities.add(entity.track_id);candidates.append(entity)
            measurements[entity.track_id]={"speed":-vy,"reference":reference}
        if method=="paired_positions" and candidates:
            # Use the same two observed capture instants for every vote,
            # rather than combine velocities from differently old windows.
            groups={}
            for entity in candidates:
                start=measurements[entity.track_id]["reference"]["start"]
                groups.setdefault(start,[]).append(entity)
            candidates=max(groups.items(),key=lambda item:(len(item[1]),item[0]))[1]
        result["current_forward_speed_track_ids"]=[entity.track_id for entity in candidates]
        result["current_forward_speed_track_count"]=len(candidates)
        if len(candidates)<3:return result
        speeds=[measurements[entity.track_id]["speed"] for entity in candidates]
        median=float(np.median(speeds));spread=float(max(speeds)-min(speeds))
        result.update(current_forward_speed_range=spread,current_forward_speed_median_candidate=median)
        if spread>max(.06,.2*median):return result
        result.update(current_forward_speed_observed=True,current_forward_speed_confidence=.85,
                      current_forward_speed=max(0.0,median))
        references=[measurements[entity.track_id]["reference"] for entity in candidates]
        if all(reference is not None for reference in references):
            result.update(current_forward_speed_reference_timestamp=float(np.median([r["midpoint"] for r in references])),
                          current_forward_speed_window_seconds=max(r["span"] for r in references),
                          current_forward_speed_reference_kind=("pair_midpoint_velocity_approximation" if method=="paired_positions"
                              else "median_tracker_window_midpoint_approximation"))
        return result

    def observe(self, packet):
        started=time.perf_counter()
        area=cv2.resize(crop_roi(packet.frame,self.profile.get("viewport",VIEWPORT)),(640,480))
        self.last_canonical=area
        gray=cv2.cvtColor(area,cv2.COLOR_BGR2GRAY)
        hsv=cv2.cvtColor(area,cv2.COLOR_BGR2HSV)
        screen="unknown"
        if self._screen_matcher:
            screen,_=self._screen_matcher.observe(packet.frame)
        hud=gray[:34,370:530]
        anchor=float(cv2.matchTemplate(hud,self._score_anchor,cv2.TM_CCOEFF_NORMED).max())
        rank_score=float(cv2.matchTemplate(gray[198:239,185:298],self._result_rank,cv2.TM_CCOEFF_NORMED).max())
        timer_color=((((hsv[:,:,0]<=4)|(hsv[:,:,0]>=176))&(hsv[:,:,1]>=180)&(hsv[:,:,2]>=100)) |
                     ((hsv[:,:,1]<=60)&(hsv[:,:,2]>=180))).astype(np.uint8)*255
        timer_tile=timer_color[:33,92:196]
        time_zero_score=(max(float(cv2.matchTemplate(timer_tile,template,cv2.TM_CCOEFF_NORMED).max())
                             for template in self._time_zero_bank) if timer_tile.std()>1 else 0.0)
        zero_digits=self._zero_digits(timer_color)
        digits_zero=all(digit["zero_accepted"] for digit in zero_digits)
        timer_ended=time_zero_score>=self.profile.get("time_zero_threshold",.90) and digits_zero
        rank_ended=rank_score>=self.profile.get("result_rank_threshold",.90)
        ended=bool(screen.startswith("result:") or ((timer_ended or rank_ended) and anchor>=.65 and not screen.startswith(("menu:","countdown:"))))
        adaptive=self.profile.get("floor_backend","legacy")=="adaptive_inner"
        floor,palette=self._adaptive_floor(hsv) if adaptive else (self.road_mask(hsv),{})
        floor_hue=palette.get("hue",0)
        blue_minimum=min(108,max(95,floor_hue+12)) if adaptive and 80<=floor_hue<=105 else 95
        player,count,ambiguous=self._player(hsv,packet.timestamp,blue_minimum)
        powered,power_count,power_ambiguous=self._powered_player(hsv,packet.timestamp) if adaptive else (None,0,False)
        appearance="powered" if powered is not None else "blue" if player is not None else None
        blue_observed=player is not None and powered is None
        if powered is not None:player=powered
        hit,hit_count,hit_ambiguous=self._hit_player(hsv,packet.timestamp) if player is None else (None,0,False)
        body=player if player is not None else hit
        occlusion,sprite_pixels=self._sprite_occlusion(hsv,body,palette,blue_minimum) if adaptive else (body,None)
        road=self._restore_player_floor(self.outer_floor(floor),occlusion)
        self.last_road_mask=road
        rows=self._road_rows(road,body,packet.timestamp)
        rows,inner_adjustments=self._floor_inner_edges(hsv,rows,palette,body,sprite_pixels) if adaptive else (rows,[])
        self._landmark_shapes={}
        found=[] if player is None else [player]
        if hit:found.append(hit)
        rocks=self._rocks(hsv,road,body,gray,packet.timestamp)
        found.extend(rocks)
        stars=self._time_stars(hsv,body) if self.profile.get("detect_time_stars",False) else []
        found.extend(stars)
        found.extend(Entity("corridor",(left+right)/2,y,right-left,1/480,.88)
                     for y,left,right in rows)
        tracked=self.tracker.update(found,packet.timestamp)
        if self.profile.get("sparse_landmark_scroll",False):
            self._update_landmark_measurements(tracked,packet.timestamp)
        detected=next((e for e in tracked if e.kind=="player"),None)
        blue_gap=(max(0.0,float(packet.timestamp-self.last_blue_time))
                  if self.last_blue_time is not None else None)
        blue_reacquired=blue_observed and blue_gap is not None and blue_gap>.20
        if detected:
            self.last_player,self.last_player_time=detected,packet.timestamp
            self.last_player_appearance=appearance
            if blue_observed:self.last_blue_time=packet.timestamp
        visible_phase=screen.split(":",1)[0] if screen!="unknown" else None
        ready=detected is not None and len(rows)>=5 and anchor>=.65 and visible_phase is None
        self._phase_count=self._phase_count+1 if ready else 0
        phase=("result" if visible_phase=="result" else "result_pending") if ended else (visible_phase or ("playing" if self._phase_count>=self.stable_frames else "unknown"))
        confidence=.86 if phase=="playing" else 0.0
        metrics={"vision_ms":(time.perf_counter()-started)*1000,"hud_anchor_score":anchor,
                 "game_ended":ended,"time_zero_score":time_zero_score,"result_rank_score":rank_score,
                 "time_zero_digits":zero_digits,"time_zero_digits_verified":digits_zero,
                 "game_ended_source":"result_anchor" if ended and (rank_ended or visible_phase=="result") else "time_zero" if ended else None,
                 "player_candidates":count,"ambiguous_player":ambiguous,"player_detected":detected is not None,
                 "player_minimum_blue_hue":blue_minimum,
                 "player_appearance":appearance,"blue_observed":blue_observed,
                 "buff_active":powered is not None,"motion_mode":"powered" if powered is not None else "normal" if blue_observed else "unknown",
                 "powered_body_candidates":power_count,"ambiguous_powered_body":power_ambiguous,
                 "blue_observation_gap_seconds":blue_gap if blue_observed else None,
                 "blue_last_seen_age_seconds":0.0 if blue_observed else blue_gap,
                 "blue_reacquired":blue_reacquired,
                 "blue_reacquired_after_seconds":blue_gap if blue_reacquired else None,
                 "collision_suspected":hit is not None,"hit_body_detected":hit is not None,
                 "hit_body_candidates":hit_count,"ambiguous_hit_body":hit_ambiguous,
                 "road_rows":rows,"road_outer_envelope_only":not adaptive,"obstacles_verified":False,
                 "floor_backend":"adaptive_inner" if adaptive else "legacy",
                 "floor_palette":palette,"road_inner_edge_adjustments":inner_adjustments,
                 "sprite_occlusion_box":[occlusion.x,occlusion.y,occlusion.w,occlusion.h] if occlusion is not None else None,
                 "sprite_observed_pixels":int(np.count_nonzero(sprite_pixels)) if sprite_pixels is not None else None,
                 "road_rows_verified":False,"road_mask_source":"floor_silhouette",
                 "time_star_candidates":len(stars),"time_star_detection_enabled":bool(self.profile.get("detect_time_stars",False)),
                 "wake_repaired_rows":self._wake_repairs,
                 "wake_tiny_gap_repair_enabled":bool(self.profile.get("repair_tiny_wake_gaps",False)),
                 "unverified_near_body_dark":self._unverified_near_body_dark,
                 "retained_near_body_rocks":self._retained_near_body_rocks,
                 "retain_tracked_near_rocks":bool(self.profile.get("retain_tracked_near_rocks",False)),
                 "circle_rock_added":self._circle_rock_added,"circle_rock_evidence":self._circle_rock_evidence,
                 "circle_rocks_enabled":bool(self.profile.get("circle_rocks",False)),
                 "circle_rock_ms":self._circle_rock_ms,
                 "large_rocks_enabled":bool(self.profile.get("large_rocks",False)),
                 "large_rock_added":self._large_rock_added,"large_rock_evidence":self._large_rock_evidence,
                 "large_rock_ms":self._large_rock_ms,
                 "rock_candidates":len(rocks),"rock_detector":"dark_compact_or_bright_ring_interior_candidates"}
        if self.profile.get("publish_obstacle_shapes",False):
            metrics["obstacle_shapes"]=self._current_obstacle_shapes(tracked,
                self._circle_rock_evidence+self._large_rock_evidence,packet.timestamp)
        if hit:
            metrics.update(hit_body_x=hit.x,hit_body_y=hit.y,hit_body_w=hit.w,hit_body_h=hit.h,
                           hit_body_identity_age=packet.timestamp-self.last_player_time)
        if self.profile.get("estimate_background_flow",False):
            metrics.update(self._flow(gray,road,packet.timestamp))
        else:
            metrics.update(forward_speed_observed=False,forward_speed_confidence=0.0,
                           background_flow_enabled=False)
        metrics.update(current_forward_speed_observed=False,current_forward_speed_confidence=0.0,
                       landmark_scroll_enabled=bool(self.profile.get("estimate_landmark_scroll",False)))
        if metrics["landmark_scroll_enabled"] and phase=="playing":
            metrics.update(self._landmark_scroll(tracked,tracks=self.tracker.tracks,timestamp=packet.timestamp,
                method=self.profile.get("landmark_scroll_method","tracker_window")))
            if (self.profile.get("sparse_landmark_scroll",False) and not metrics["current_forward_speed_observed"]
                    and metrics.get("current_forward_speed_track_count",0)<3):
                metrics.update(self._sparse_landmark_scroll(tracked,self._landmark_measurements,packet.timestamp))
            if metrics["current_forward_speed_observed"]:
                metrics["current_forward_speed_timestamp"]=packet.timestamp
        metrics["wall_scroll_enabled"]=self._wall_scroll is not None
        if self._wall_scroll is not None:
            observed=bool(metrics.get("current_forward_speed_observed"))
            wall=self._wall_scroll.observe(floor,packet.timestamp,palette,phase=phase,
                blocked=hit is not None or powered is not None,measure=not observed)
            # Reliable landmark observations have priority; the estimator
            # still records this actual raw floor for the next capture pair.
            if observed:metrics["wall_scroll"]=wall.get("wall_scroll",{})
            else:
                metrics.update(wall)
                if metrics.get("current_forward_speed_observed"):
                    metrics["current_forward_speed_timestamp"]=packet.timestamp
        if detected is None and self.last_player and self.last_player_time is not None and 0<packet.timestamp-self.last_player_time<.15:
            dt=packet.timestamp-self.last_player_time
            metrics.update(player_predicted_x=self.last_player.x+self.last_player.vx*dt,
                           player_predicted_y=self.last_player.y,player_prediction_age=dt)
        notes=("MINING实机颜色/形状原型；石头为道路内暗块/明亮圆环候选，未测准确率；匹配分数不是正确率",)
        if detected is None:notes+=("蓝色机体本帧不可见或身份歧义；预测位置仅诊断，不发布为真实检测",)
        if hit:notes+=("本帧检出近期蓝色机体附近的紫红碰撞外观，单列player_hit；尚未验证碰撞状态准确率",)
        return Observation(self.game,packet.timestamp,packet.sequence,phase,confidence,tracked,notes,metrics)
