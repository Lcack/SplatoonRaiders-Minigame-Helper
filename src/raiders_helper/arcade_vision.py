"""Fast reference-video perception for Hopping/Juggling/Landing.

This is a transparent colour/shape baseline calibrated to the saved reference,
not a trained universal detector. Similar colours, overlaps and different capture
settings can fail. All bundled profiles are advisory-only and uncalibrated.
"""
from pathlib import Path
import time

import cv2
import numpy as np

from .fast_tracking import PredictiveTracker
from .models import Entity, Game, Observation
from .vision import crop_roi


class FastArcadeRecognizer:
    def __init__(self, game, profile):
        self.game, self.profile = Game(game), profile
        area_gate=profile.get('ball_area_ratio_limit') if self.game==Game.JUGGLING else None
        physical_bounces=self.game==Game.JUGGLING and profile.get('physical_ball_bounces') is True
        self.tracker = PredictiveTracker(ball_area_ratio_limit=area_gate,physical_ball_bounces=physical_bounces)
        self.phase_count = 0
        self._life_candidate = None
        self._life_repeats = 0
        self._life_first = None
        self.fever_observer = None
        self.ball_observer = None
        if self.game==Game.JUGGLING and profile.get('track_ball_dynamics') is True:
            from .juggling_prediction import BallTrajectoryObserver
            self.ball_observer=BallTrajectoryObserver()
        if self.game==Game.JUGGLING and profile.get('read_fever_gauge') is True:
            from .juggling_fever import FeverObserver
            self.fever_observer=FeverObserver()
        self.stable_frames = profile.get("stable_frames", 2)
        if isinstance(self.stable_frames, bool) or not isinstance(self.stable_frames, int) or self.stable_frames < 1:
            raise ValueError("stable_frames 必须为正整数")
        self.previous_player = None
        self.atlas = None
        self.last_pose = None
        self.last_map = {}
        self.map_frame = -999
        self.map_signature = None
        self.score_anchor = None
        self.object_templates={}
        for kind,key in (("egg_pickup","egg_template"),("golden_enemy","golden_template")):
            if profile.get(key):
                path=Path(profile.get("_root","."))/profile[key]
                template=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_GRAYSCALE)
                self.object_templates[kind]=(template,cv2.flip(template,1))
        anchor = profile.get("score_anchor")
        if anchor:
            path = Path(profile.get("_root", ".")) / anchor
            self.score_anchor = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if self.game == Game.LANDING:
            from .landing import LandingAtlas
            path = Path(profile.get("_root", ".")) / profile["landing_atlas"]
            self.atlas = LandingAtlas(path)
            self.profile.setdefault("policy", {})["atlas_registry"] = str(path.resolve())
        self.kernel = np.ones((3, 3), np.uint8)

    def components(self, mask, min_area=20, close=(3, 3)):
        if close:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(close, np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        return [tuple(map(int, box)) for box in stats[1:] if box[4] >= min_area]

    @staticmethod
    def entity(kind, box, score=0.86, pad=0):
        x, y, w, h = box[:4]
        left, top = max(0, x-pad), max(0, y-pad)
        right, bottom = min(640, x+w+pad), min(480, y+h+pad)
        return Entity(kind, (left+right)/1280, (top+bottom)/960,
                      (right-left)/640, (bottom-top)/480, score)

    @staticmethod
    def overlap(box, entity, margin=0):
        x, y, w, h = box[:4]
        return (x+w > (entity.x-entity.w/2)*640-margin and x < (entity.x+entity.w/2)*640+margin
                and y+h > (entity.y-entity.h/2)*480-margin and y < (entity.y+entity.h/2)*480+margin)

    def modal_panel(self, hsv):
        """Reject reference game's central stage/result/pause-style panels."""
        white=cv2.inRange(hsv,(0,0,155),(179,85,255))
        white[:90]=0;white[440:]=0;white[:,:55]=0;white[:,585:]=0
        white=cv2.morphologyEx(white,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(white,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour)
            if 140<=w<=500 and 65<=h<=330:
                top=np.mean(np.any(white[y:y+4,x:x+w],axis=0)>0)
                bottom=np.mean(np.any(white[y+h-4:y+h,x:x+w],axis=0)>0)
                left=np.mean(np.any(white[y:y+h,x:x+4],axis=1)>0)
                right=np.mean(np.any(white[y:y+h,x+w-4:x+w],axis=1)>0)
                if min(top,bottom,left,right)>.65:
                    return True
        return False

    def match_object(self, area, kind, box, padding=18):
        templates=self.object_templates.get(kind)
        if not templates:
            return None
        x,y,w,h=box[:4]
        x0,y0=max(85,x-padding),max(35,y-padding)
        x1,y1=min(555,x+w+padding),min(450,y+h+padding)
        if x1 <= x0 or y1 <= y0:
            return None
        tile=cv2.cvtColor(area[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
        best=None
        for template in templates:
            th,tw=template.shape
            if tile.shape[0]<th or tile.shape[1]<tw:
                continue
            _,score,_,location=cv2.minMaxLoc(cv2.matchTemplate(tile,template,cv2.TM_CCOEFF_NORMED))
            if score>=0.82 and (best is None or score>best[0]):
                best=(score,(x0+location[0],y0+location[1],tw,th,tw*th))
        return best

    @staticmethod
    def juggling_life_pixels(hsv):
        """Five fixed HUD slots observed on the actual NS, not a life rule."""
        red=(((hsv[:,:,0]<=9)|(hsv[:,:,0]>=171))&(hsv[:,:,1]>=140)&(hsv[:,:,2]>=90))
        return [int(np.count_nonzero(red[8:27,250+20*i:270+20*i])) for i in range(5)]

    def juggling_lives(self,hsv,anchor_score):
        pixels=self.juggling_life_pixels(hsv)
        occupied=[count>=100 for count in pixels]
        count=sum(occupied)
        valid=anchor_score>=.65 and occupied==([True]*count+[False]*(5-count))
        candidate=count if valid else None
        self._life_repeats=self._life_repeats+1 if candidate==self._life_candidate else 1
        self._life_candidate=candidate
        result={'life_slot_red_pixels':pixels,'lives_observed':False,
                'life_count_source':'actual_red_heart_HUD_slots','life_confidence_kind':'heuristic_gate_not_probability'}
        if candidate is not None and self._life_repeats>=(8 if candidate==0 else 3):
            if self._life_first is None and candidate>0:self._life_first=candidate
            result.update(lives_remaining=candidate,lives_observed=True,lives_first_observed=self._life_first)
        return result

    def observe(self, packet):
        started = time.perf_counter()
        area = cv2.resize(crop_roi(packet.frame, self.profile.get("viewport", [1/6, 55/1080, 2/3, 960/1080])), (640, 480))
        hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
        overlay=self.modal_panel(hsv)
        blue = cv2.inRange(hsv, (95, 100, 100), (135, 255, 255))
        green_blue = cv2.inRange(hsv, (40, 85, 90), (135, 255, 255))
        yellow = cv2.inRange(hsv, (22, 100, 130), (42, 255, 255))
        red = cv2.inRange(hsv, (0, 105, 100), (12, 255, 255)) | cv2.inRange(hsv, (170, 105, 100), (179, 255, 255))
        for mask in (blue, green_blue, yellow, red):
            mask[:35] = 0  # HUD is not a sprite.
        if self.game == Game.JUGGLING:
            red[435:] = 0
        else:
            red[450:] = 0
        blue_boxes = self.components(blue, 30)
        body_boxes = self.components(green_blue, 100, (5, 5))
        yellow_boxes = self.components(yellow, 80, (5, 5))
        found = []
        partner_post_supplement=[]
        squid_count=0
        ambiguous=False
        robots = [b for b in blue_boxes if b[2] >= (12 if self.game == Game.LANDING else 22) and b[3] >= 12 and b[4] >= (100 if self.game == Game.LANDING else 200)
                  and b[2] <= 80 and b[3] <= 65 and b[1] >= (360 if self.game == Game.JUGGLING else 35)]
        if self.game == Game.JUGGLING:
            robots = [b for b in robots if b[0]+b[2]/2 < 320]
        powered = False
        if robots:
            robot = max(robots, key=lambda b: b[4])
            center = (robot[0]+robot[2]/2, robot[1]+robot[3]/2)
            connected = [b for b in body_boxes if b[0] <= center[0] <= b[0]+b[2]
                         and b[1] <= center[1] <= b[1]+b[3] and 20 <= b[2] < 120 and b[3] < 110]
            player = self.entity("player", max(connected, key=lambda b: b[4]) if connected else robot, pad=3 if connected else 12)
            found.append(player)
        elif self.game == Game.HOPPING:
            buff_robots = [b for b in yellow_boxes if b[2] >= 45 and b[3] >= 40 and b[1] > 130 and b[2] < 140 and b[3] < 140]
            if buff_robots:
                if self.previous_player:
                    robot = min(buff_robots, key=lambda b: abs((b[0]+b[2]/2)/640-self.previous_player.x)+abs((b[1]+b[3]/2)/480-self.previous_player.y))
                else:
                    robot = max(buff_robots, key=lambda b: b[4])
                player = self.entity("player", robot)
                found.extend((player, self.entity("egg_buff", robot, 0.82)))
                powered = True
        player = next((e for e in found if e.kind == "player"), None)
        if player:
            self.previous_player = player
        if self.game in (Game.HOPPING, Game.JUGGLING):
            squids = [b for b in body_boxes if 16 <= b[2] <= 55 and 26 <= b[3] <= 75
                      and 0.14*640 < b[0]+b[2]/2 < 0.86*640 and b[4] >= 130
                      and (not player or not self.overlap(b, player, 2))]
            if self.game == Game.JUGGLING:
                partner_left=310 if self.profile.get('juggling_partner_full_half') is True else 330
                faces = [b for b in blue_boxes if b[0] > partner_left and b[0]+b[2]/2>320 and b[1] > 385 and b[4] >= 25 and b[2] >= 7 and b[3] >= 8]
                squids = [b for b in body_boxes if b[0] > partner_left and b[0]+b[2]/2>320 and b[1] > 380 and 12 <= b[2] <= 50
                          and 17 <= b[3] <= 65 and any(b[0] <= f[0]+f[2]/2 <= b[0]+b[2]
                          and b[1] <= f[1]+f[3]/2 <= b[1]+b[3] for f in faces)]
                if not squids and self.profile.get('juggling_partner_full_half') is True:
                    near=[f for f in faces if f[0]<375 and f[1]>400]
                    if near:
                        # The green CRT fringe of the middle post can join
                        # the squid body. Limit it by the current blue face,
                        # rather than publishing the post as part of actor.
                        x0=max(310,min(f[0] for f in near)-24);x1=min(400,max(f[0]+f[2] for f in near)+8)
                        y0=max(402,min(f[1] for f in near)-8);y1=min(465,max(f[1]+f[3] for f in near)+5)
                        local=np.zeros_like(green_blue);local[y0:y1,x0:x1]=green_blue[y0:y1,x0:x1]
                        squids=[b for b in self.components(local,130,(3,3)) if b[0]+b[2]/2>320
                                and 12<=b[2]<=50 and 17<=b[3]<=55]
                        partner_post_supplement=[list(b[:4]) for b in squids]
            if powered:
                squids += [b for b in yellow_boxes if 18 <= b[2] <= 55 and 32 <= b[3] <= 80
                           and b[2]/b[3] < 1.15 and not self.overlap(b, player, 3)]
            if squids:
                squid_count=len(squids)
                ordered=sorted(squids,key=lambda b:b[4],reverse=True)
                ambiguous=len(ordered)>1 and ordered[1][4] >= ordered[0][4]*0.65
                kind = "partner" if self.game == Game.JUGGLING else "ball"
                # A single squid is expected. Ambiguous candidates are exposed
                # in notes instead of silently inventing two controlled actors.
                selected = max(squids, key=lambda b: b[4])
                found.append(self.entity(kind, selected))
            bright = ((hsv[:, :, 2] >= 140) & (hsv[:, :, 1] >= 80) | (hsv[:, :, 2] >= 205)).astype(np.uint8)*255
            bright[:45] = 0
            bright[435 if self.game == Game.JUGGLING else 450:] = 0
            if self.game == Game.HOPPING:
                bright[:, :85] = 0
                bright[:, 555:] = 0
                red[:, :85] = 0
                red[:, 555:] = 0
            fish = []
            supplemented=[]
            for box in self.components(bright, 50, (3, 17)):
                x, y, w, h, pixels = box
                red_count = cv2.countNonZero(red[y:y+h, x:x+w])
                ordinary=12 <= w < 145 and 18 <= h < 125 and 0.25 < w/h < 3.1 and red_count >= 65 and red_count/(w*h) >= 0.05
                supplement=False
                if (not ordinary and self.game==Game.JUGGLING
                        and self.profile.get('juggling_small_fish_supplement') is True
                        and 12<=w<145 and 8<=h<125 and .25<w/h<8
                        and red_count>=20 and red_count/(w*h)>=.03):
                    # Actual rotating platform fish can have <65 red pixels
                    # or be thinner than18 px. Require an observed coloured
                    # body as well: red falling hearts and green +30 text
                    # cannot qualify alone. No stale track creates an object.
                    body_count=cv2.countNonZero(green_blue[y:y+h,x:x+w])
                    tile=hsv[y:y+h,x:x+w]
                    pale_count=int(np.count_nonzero((tile[:,:,1]<=100)&(tile[:,:,2]>=150)))
                    supplement=body_count>=10 and pale_count>=12
                    if supplement:supplemented.append({'box':list(box[:4]),'red_pixels':red_count,
                        'body_pixels':body_count,'pale_face_pixels':pale_count})
                if ordinary or supplement:
                    fish.append(box)
                    found.append(self.entity("ball" if self.game == Game.JUGGLING else "enemy", box, 0.84, pad=2))
            if self.game == Game.HOPPING:
                for box in yellow_boxes:
                    if box[4] >= 200 and 25 <= box[2] <= 100 and 12 <= box[3] <= 65 and box[1] < 350:
                        if not any(self.overlap(box, e, 3) for e in found if e.kind in ("player", "ball")):
                            matched=self.match_object(area,"golden_enemy",box,24)
                            if matched:
                                found.append(self.entity("golden_enemy",matched[1],max(0.82,matched[0])))
                egg_matches=[]
                for box in fish:
                    if box[2]<=85 and box[3]<=80 and 0.55<box[2]/box[3]<1.65:
                        matched=self.match_object(area,"egg_pickup",box)
                        if matched and not any(self.overlap(matched[1],e) for e in found if e.kind in ("player","ball")):
                            if not any(abs(matched[1][0]-other[1][0])<12 and abs(matched[1][1]-other[1][1])<12 for other in egg_matches):
                                egg_matches.append(matched)
                for score,box in egg_matches:
                    found=[e for e in found if not (e.kind=="enemy" and self.overlap(box,e))]
                    found.append(self.entity("egg_pickup",box,max(0.82,score)))
        metrics = {}
        anchor_score = 0.0
        if self.score_anchor is not None:
            hud = cv2.cvtColor(area[:34, 370:530], cv2.COLOR_BGR2GRAY)
            if hud.shape[0] >= self.score_anchor.shape[0] and hud.shape[1] >= self.score_anchor.shape[1]:
                anchor_score = float(cv2.matchTemplate(hud, self.score_anchor, cv2.TM_CCOEFF_NORMED).max())
        tracked = self.tracker.update(found, packet.timestamp)
        if self.game == Game.LANDING and player:
            from .landing import estimate_pose
            pose_path = self.atlas.path.parent / self.atlas.manifest.get("pose_reference", "robot_upright.png")
            pose = estimate_pose(area, player, pose_path)
            metrics.update({k: v for k, v in pose.items() if isinstance(v, (float, int, bool))})
            angle = pose.get("vehicle_angle_radians")
            if angle is not None:
                if self.last_pose and 0 < packet.timestamp-self.last_pose[0] < 0.15:
                    dt=packet.timestamp-self.last_pose[0]
                    delta=(angle-self.last_pose[1]+np.pi)%(2*np.pi)-np.pi
                    # Quantized 10-degree pose bins must not become an
                    # unbounded derivative spike at a bin boundary.
                    raw=float(np.clip(delta/dt,-8,8))
                    smoothed=0.25*raw+0.75*self.last_pose[2]
                    metrics["angular_velocity"]=smoothed
                else:
                    smoothed=0.0
                self.last_pose=(packet.timestamp,angle,smoothed)
            signature = cv2.resize(cv2.inRange(hsv, (5, 60, 30), (40, 255, 230)), (32, 24))
            changed = self.map_signature is None or np.mean(np.abs(signature.astype(float)-self.map_signature)) > 8
            if changed or packet.sequence-self.map_frame >= self.profile.get("map_match_every", 6) or not self.last_map.get("accepted"):
                self.last_map = self.atlas.match(area)
                self.map_frame = packet.sequence
                self.map_signature = signature.astype(float)
            metrics.update({k: v for k, v in self.last_map.items() if k in ("map_index", "map_match_score", "map_match_margin")})
        ready = player is not None and anchor_score >= 0.65 and not ambiguous and not overlay
        if self.game == Game.JUGGLING:
            ready = ready and any(e.kind == "partner" for e in tracked)
        self.phase_count = self.phase_count+1 if ready else 0
        phase = "overlay" if overlay else ("playing" if self.phase_count >= self.stable_frames else "unknown")
        confidence = 0.82 if phase == "playing" else 0.0
        metrics["hud_anchor_score"] = anchor_score
        if self.game==Game.JUGGLING:
            metrics['small_fish_supplement_enabled']=self.profile.get('juggling_small_fish_supplement') is True
            metrics['small_fish_supplement']=supplemented
            metrics['ball_velocity_windows']=dict(self.tracker.velocity_windows)
            metrics['partner_post_supplement']=partner_post_supplement
            if self.profile.get('read_live_lives') is True:metrics.update(self.juggling_lives(hsv,anchor_score))
        metrics["squid_candidates"] = squid_count
        metrics["robot_candidates"] = len(robots)
        metrics["ambiguous_targets"] = ambiguous
        if self.fever_observer is not None:
            pickups,fever=self.fever_observer.observe(hsv,tracked,packet.timestamp,phase)
            tracked=tuple(tracked)+pickups
            metrics.update(fever)
        if self.ball_observer is not None:
            metrics['ball_flight_models']=self.ball_observer.observe(tracked,packet.timestamp,phase)
        metrics["vision_ms"] = (time.perf_counter()-started)*1000
        notes = ("参考视频颜色/形状启发式；匹配得分不是正确率；重叠与陌生画面可能误检",)
        if ambiguous:
            notes += ("多个相近大小的角色候选，暂停建议等待消歧",)
        return Observation(self.game, packet.timestamp, packet.sequence, phase, confidence, tracked, notes, metrics)
