"""Small-object predictive assignment without a heavyweight inference runtime."""
from collections import deque
from dataclasses import replace
import math


def assignment(cost):
    """Minimum-cost rectangular assignment, rows <= columns, O(n^3)."""
    if not cost:
        return []
    n, m = len(cost), len(cost[0])
    u, v, p, way = [0.]*(n+1), [0.]*(m+1), [0]*(m+1), [0]*(m+1)
    for i in range(1, n+1):
        p[0], j0 = i, 0
        minimum, used = [math.inf]*(m+1), [False]*(m+1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, m+1):
                if not used[j]:
                    current = cost[i0-1][j-1]-u[i0]-v[j]
                    if current < minimum[j]:
                        minimum[j], way[j] = current, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(m+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0], j0 = p[j1], j1
    return [(p[j]-1, j-1) for j in range(1, m+1) if p[j]]


class PredictiveTracker:
    def __init__(self, max_gap=0.15, *, ball_area_ratio_limit=None, physical_ball_bounces=False):
        self.tracks = {}
        self.next_id = 0
        self.last_time = None
        self.max_gap = max_gap
        self.ball_area_ratio_limit=ball_area_ratio_limit
        self.physical_ball_bounces=physical_ball_bounces
        self.velocity_windows={}

    def update(self, detections, timestamp):
        self.velocity_windows={}
        if self.last_time is not None and not 0 < timestamp-self.last_time <= 0.25:
            self.tracks.clear()
        self.last_time = timestamp
        self.tracks = {key: value for key, value in self.tracks.items() if timestamp-value[1] <= self.max_gap}
        previous = list(self.tracks.items())
        costs = []
        for e in detections:
            row = []
            for _, (old, last, _) in previous:
                dt = timestamp-last
                predicted_x, predicted_y = old.x+old.vx*dt, old.y+old.vy*dt
                distance = math.hypot(e.x-predicted_x, e.y-predicted_y)
                size = abs(e.w-old.w)+abs(e.h-old.h)
                shape_ok=True
                if e.kind==old.kind=='ball' and self.ball_area_ratio_limit is not None:
                    areas=(e.w*e.h,old.w*old.h)
                    shape_ok=min(areas)>0 and max(areas)/min(areas)<=self.ball_area_ratio_limit
                row.append(distance+0.08*size if e.kind == old.kind and shape_ok and distance < 0.08+0.8*dt else 1000.)
            costs.append(row+[0.11]*len(detections))
        matches = dict(assignment(costs))
        result = []
        for i, e in enumerate(detections):
            j = matches[i]
            if j < len(previous) and costs[i][j] < 0.11:
                track_id, (old, last, history) = previous[j]
                dt = timestamp-last
                raw_vx, raw_vy = (e.x-old.x)/dt, (e.y-old.y)/dt
                # React immediately to a measured bounce instead of averaging
                # upward and downward samples into a misleading zero velocity.
                bounced = old.samples >= 2 and ((raw_vy*old.vy < 0 and abs(e.y-old.y) > 0.002)
                                               or (raw_vx*old.vx < 0 and abs(e.x-old.x) > 0.002))
                if e.kind=='ball' and self.physical_ball_bounces:
                    # Ordinary sprite rotation and the flight apex do not
                    # create an instantaneous physical velocity reversal.
                    bounced=old.samples>=2 and (
                        old.vy>.2 and raw_vy<-.2 and e.y>.68 and abs(e.y-old.y)>.01
                        or old.vx*raw_vx<0 and min(abs(old.vx),abs(raw_vx))>.2
                           and (e.x<.12 or e.x>.88) and abs(e.x-old.x)>.008)
                if bounced:
                    history = deque([(last, old.x, old.y)], maxlen=5)
                history.append((timestamp, e.x, e.y))
                first = history[0]
                span = timestamp-first[0]
                vx = (e.x-first[1])/span if span else raw_vx
                vy = (e.y-first[2])/span if span else raw_vy
                e = replace(e, track_id=track_id, vx=vx, vy=vy, samples=old.samples+1)
            else:
                track_id = self.next_id
                self.next_id += 1
                history = deque([(timestamp, e.x, e.y)], maxlen=5)
                e = replace(e, track_id=track_id)
            self.tracks[track_id] = (e, timestamp, history)
            if e.kind=='ball':self.velocity_windows[str(track_id)]=timestamp-history[0][0]
            result.append(e)
        # Missed objects are kept briefly only for association, never published
        # as if they had been detected in this frame.
        return tuple(result)
