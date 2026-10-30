"""Fast LANDING layout/pose analysis and experimental feedback suggestions.

The atlas is derived from one recorded run, not a complete map catalogue.
Control gains are heuristics until measured; suggestions never bypass the
runtime's separate hardware-calibration gate.
"""
from collections import OrderedDict
from functools import lru_cache
import heapq
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .control.buttons import Button, ControllerState
from .models import Decision

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATLAS = ROOT / "assets/landing_atlas/index.json"


def canonical_frame(frame):
    if frame.shape[:2] == (480, 640):
        return frame
    height, width = frame.shape[:2]
    # Source video's game viewport, not a universal capture-card calibration.
    return cv2.resize(frame[round(height*55/1080):round(height*1015/1080),
                            round(width/6):round(width*5/6)], (640, 480))


def terrain_mask(frame):
    """Brown structure pixels, excluding HUD and most small dynamic sprites."""
    frame = canonical_frame(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((7, 65, 35)), np.array((40, 255, 255)))
    mask[:35] = 0
    mask[445:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros(mask.shape, np.uint8)
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        width, height = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # A small compact orange/red fish fragment is not a terrain wall.
        if area >= 45 and (width >= 24 or height >= 32 or area >= 300):
            clean[labels == i] = 255
    # Fill holes in connected brown walls, while keeping open passages open.
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(clean)
    cv2.drawContours(result, contours, -1, 255, cv2.FILLED)
    return result


def _read_image(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), flags)


def _feature(mask):
    reduced = cv2.resize(mask, (80, 60), interpolation=cv2.INTER_AREA)
    return reduced > 80


def _platform_hints(mask):
    hints = []
    for x0, x1, x in ((0, 50, 0.055), (590, 640, 0.94)):
        rows = np.count_nonzero(mask[100:390, x0:x1], axis=1)
        candidates = np.flatnonzero(rows > 12)
        y = (100+int(candidates[0]))/480 - 0.045 if len(candidates) else 0.45
        hints.append((x, float(max(0.15, min(0.8, y)))))
    return hints


def _nearest_free(grid, cell, radius=8):
    h, w = grid.shape
    x, y = cell
    candidates = [(abs(dx)+abs(dy), x+dx, y+dy)
                  for dy in range(-radius, radius+1) for dx in range(-radius, radius+1)
                  if 0 <= x+dx < w and 0 <= y+dy < h and not grid[y+dy, x+dx]]
    if not candidates:
        return None
    _, x, y = min(candidates)
    return x, y


def _clearance_radius(clearance):
    # Geometry depends on this integer dilation radius, not on every small
    # frame-to-frame change in the detector bbox or the raw float clearance.
    return max(1, math.ceil(clearance*80))


def _routing_grid(mask, radius):
    grid = _feature(mask).astype(np.uint8)
    grid = cv2.dilate(grid, np.ones((2*radius+1, 2*radius+1), np.uint8)) > 0
    grid[:3] = True
    grid[55:] = True
    return grid


def _line_is_free(grid, a, b):
    """Supercover traversal between cell centers, including touched corners.

    The old rounded samples could jump across a blocked cell at a diagonal
    corner. Check both adjacent cells when the segment touches a grid corner,
    using the same no-corner-cut rule as the shortest-path graph.
    """
    x, y = a
    target_x, target_y = b
    if grid[y, x] or grid[target_y, target_x]:
        return False
    nx, ny = abs(target_x-x), abs(target_y-y)
    sx = 1 if target_x>x else -1
    sy = 1 if target_y>y else -1
    ix = iy = 0
    while ix<nx or iy<ny:
        decision = (1+2*ix)*ny-(1+2*iy)*nx
        if decision == 0:
            if grid[y, x+sx] or grid[y+sy, x]:
                return False
            x, y = x+sx, y+sy
            ix, iy = ix+1, iy+1
        elif decision < 0:
            x += sx
            ix += 1
        else:
            y += sy
            iy += 1
        if grid[y, x]:
            return False
    return True


def _simplify_route(path, visible):
    if not path:
        return ()
    simplified, index = [path[0]], 0
    while index<len(path)-1:
        target = index+1
        for candidate in range(index+2, len(path)):
            if not visible(path[index], path[candidate]):
                break
            target = candidate
        simplified.append(path[target])
        index = target
    return tuple((x/79, y/59) for x, y in simplified)


class _NavigationGrid:
    """One static inflated grid shared by all goal fields for this radius."""
    def __init__(self, grid):
        self.grid = grid
        self.visibility = {}

    def visible(self, a, b):
        # Visibility is symmetric and independent of the destination. Starts
        # change every frame but most route suffixes recur on the same map.
        key = (a, b) if a<=b else (b, a)
        result = self.visibility.get(key)
        if result is None:
            result = _line_is_free(self.grid, a, b)
            if len(self.visibility)>=8192:
                self.visibility.clear()
            self.visibility[key] = result
        return result


class _NavigationField:
    """Reverse Dijkstra tree: subsequent starts only follow decreasing costs."""
    def __init__(self, navigation_grid, goal):
        self.navigation_grid = navigation_grid
        self.goal = goal
        grid = navigation_grid.grid
        height, width = grid.shape
        self.width = width
        self.distance = np.full(height*width, math.inf)
        self.next_cell = np.full(height*width, -1, dtype=np.int32)
        goal_id = goal[1]*width+goal[0]
        self.distance[goal_id] = 0.0
        self.next_cell[goal_id] = goal_id
        blocked = grid.ravel().tolist()
        queue = [(0.0, goal_id)]
        moves = ((1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),
                 (1,1,math.sqrt(2)),(1,-1,math.sqrt(2)),
                 (-1,1,math.sqrt(2)),(-1,-1,math.sqrt(2)))
        while queue:
            cost, current = heapq.heappop(queue)
            if cost > self.distance[current]:
                continue
            x, y = current%width, current//width
            for dx, dy, weight in moves:
                nx, ny = x+dx, y+dy
                if not 0<=nx<width or not 0<=ny<height:
                    continue
                neighbor = ny*width+nx
                if blocked[neighbor] or (dx and dy and (blocked[y*width+nx] or blocked[ny*width+x])):
                    continue
                candidate = cost+weight
                if candidate < self.distance[neighbor]:
                    self.distance[neighbor] = candidate
                    self.next_cell[neighbor] = current
                    heapq.heappush(queue, (candidate, neighbor))
        self.routes = {}

    def route(self, start):
        cell_id = start[1]*self.width+start[0]
        if cell_id not in self.routes:
            path = []
            current = cell_id
            for _ in range(len(self.next_cell)):
                next_id = int(self.next_cell[current])
                if next_id < 0:
                    path = []
                    break
                path.append((current%self.width, current//self.width))
                if next_id == current:
                    break
                # A shortest-path successor must strictly reduce distance.
                if self.distance[next_id] >= self.distance[current]:
                    path = []
                    break
                current = next_id
            else:
                path = []
            self.routes[cell_id] = _simplify_route(path, self.navigation_grid.visible)
        # Return a fresh container: callers cannot corrupt future cached routes.
        return [list(point) for point in self.routes[cell_id]]


def route_from_mask(mask, start, goal, clearance=0.025):
    """Small-grid A* with no diagonal corner cutting and LOS simplification."""
    grid = _routing_grid(mask, _clearance_radius(clearance))
    begin = _nearest_free(grid, (round(start[0]*79), round(start[1]*59)))
    end = _nearest_free(grid, (round(goal[0]*79), round(goal[1]*59)))
    if begin is None or end is None:
        return []
    queue = [(0.0, begin)]
    costs, parents = {begin: 0.0}, {}
    while queue:
        _, current = heapq.heappop(queue)
        if current == end:
            break
        x, y = current
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            nx, ny = x+dx, y+dy
            if not 0 <= nx < 80 or not 0 <= ny < 60 or grid[ny, nx]:
                continue
            if dx and dy and (grid[y, nx] or grid[ny, x]):
                continue
            cost = costs[current] + (math.sqrt(2) if dx and dy else 1)
            next_cell = (nx, ny)
            if cost < costs.get(next_cell, math.inf):
                costs[next_cell], parents[next_cell] = cost, current
                heuristic = math.hypot(nx-end[0], ny-end[1])
                heapq.heappush(queue, (cost+heuristic, next_cell))
    if end not in costs:
        return []
    path = [end]
    while path[-1] != begin:
        path.append(parents[path[-1]])
    path.reverse()
    return [list(point) for point in _simplify_route(path, lambda a,b: _line_is_free(grid,a,b))]


class LandingAtlas:
    def __init__(self, path=DEFAULT_ATLAS):
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = ROOT/self.path
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        self.layouts = self.manifest["layouts"]
        self.masks = [_read_image(self.path.parent/e["mask"], cv2.IMREAD_GRAYSCALE) for e in self.layouts]
        if any(mask is None for mask in self.masks):
            raise ValueError("LANDING atlas contains an unreadable mask")
        self.features = np.stack([_feature(mask) for mask in self.masks])
        self._navigation_grids = OrderedDict()
        self._navigation_fields = OrderedDict()

    def match(self, canonical_bgr, minimum_score=0.60, minimum_margin=0.08):
        feature = _feature(terrain_mask(canonical_bgr))
        intersection = np.count_nonzero(self.features & feature, axis=(1,2))
        total = np.count_nonzero(self.features, axis=(1,2))+np.count_nonzero(feature)
        # Symmetric overlap tolerates a robot obscuring part of a small launch
        # platform; unlike matching black background it still penalizes extras.
        scores = 2*intersection/np.maximum(1, total)
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        score = float(scores[best])
        margin = score-float(scores[int(order[1])]) if len(order)>1 else score
        accepted = score >= minimum_score and margin >= minimum_margin
        return {"map_index": best if accepted else -1, "map_id": self.layouts[best]["id"] if accepted else None,
                "map_match_score": score, "map_match_margin": margin,
                "accepted": accepted, "best_candidate": self.layouts[best]["id"]}

    def route(self, map_index, start, goal=None, clearance=0.025):
        entry = self.layouts[map_index]
        goal = tuple(goal or entry["goal_hint"])
        radius = _clearance_radius(clearance)
        grid_key = (map_index, radius)
        if grid_key not in self._navigation_grids:
            if len(self._navigation_grids)>=16:
                old_key, _ = self._navigation_grids.popitem(last=False)
                for field_key in list(self._navigation_fields):
                    if field_key[:2] == old_key:
                        del self._navigation_fields[field_key]
            self._navigation_grids[grid_key] = _NavigationGrid(_routing_grid(self.masks[map_index], radius))
        navigation_grid = self._navigation_grids[grid_key]
        self._navigation_grids.move_to_end(grid_key)
        grid = navigation_grid.grid
        begin = _nearest_free(grid, (round(start[0]*79), round(start[1]*59)))
        end = _nearest_free(grid, (round(goal[0]*79), round(goal[1]*59)))
        if begin is None or end is None:
            return []
        # Quantize only to actual grid geometry. The requested clearance is
        # always rounded UP, and different snapped goals get separate fields.
        key = (map_index, radius, end)
        if key not in self._navigation_fields:
            if len(self._navigation_fields)>=64:
                self._navigation_fields.popitem(last=False)
            self._navigation_fields[key] = _NavigationField(navigation_grid, end)
        field = self._navigation_fields[key]
        self._navigation_fields.move_to_end(key)
        return field.route(begin)


def build_atlas(index_path=None, output_dir=None):
    index_path = Path(index_path or ROOT/"maps/landing/index.json")
    output = Path(output_dir or ROOT/"artifacts/landing_atlas")
    output.mkdir(parents=True, exist_ok=True)
    registry = json.loads(index_path.read_text(encoding="utf-8"))
    layouts = []
    for entry in registry["layouts"]:
        full = _read_image(ROOT/entry["image"])
        canonical = canonical_frame(full)
        mask = terrain_mask(canonical)
        mask_name = entry["id"]+"-mask.png"
        cv2.imencode(".png", mask)[1].tofile(str(output/mask_name))
        start, goal = _platform_hints(mask)
        waypoints = route_from_mask(mask, start, goal)
        layouts.append({"id": entry["id"], "observed_stage": entry["observed_stage"],
                        "source_image": entry["image"], "source_seconds": entry["seconds"],
                        "mask": mask_name, "start_hint": start, "goal_hint": goal,
                        "waypoints": waypoints, "strategy_status": "experimental_geometry_only"})
    # An upright, pre-launch robot from the same reference clip. Pose convention
    # is relative to this visibly upright sprite, not an inferred physics angle.
    reference = canonical_frame(_read_image(ROOT/"artifacts/landing_map_candidates/frame-00057780.png"))
    patch = _robot_patch(reference, (0.055, 0.46))
    cv2.imencode(".png", patch)[1].tofile(str(output/"robot_upright.png"))
    manifest = {"version":1, "scope":"one_reference_run_not_complete_map_catalog",
                "canonical_size":[640,480], "source_url":registry["source_url"],
                "source_sha256":registry["source_sha256"], "pose_reference":"robot_upright.png",
                "layouts":layouts}
    (output/"index.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest


def _robot_patch(frame, center):
    height,width=frame.shape[:2]
    cx,cy=round(center[0]*width),round(center[1]*height)
    # Recenter on the distinctive dark-blue body, not the orientation-dependent
    # green outline or an exhaust jet included in the detector bbox.
    x0,y0=max(0,cx-23),max(0,cy-23)
    tile=frame[y0:min(height,cy+24),x0:min(width,cx+24)]
    hsv=cv2.cvtColor(tile,cv2.COLOR_BGR2HSV)
    blue=cv2.inRange(hsv,np.array((105,100,55)),np.array((132,255,255)))
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(blue)
    if count>1:
        candidates=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=8]
        if candidates:
            i=max(candidates,key=lambda i:stats[i,cv2.CC_STAT_AREA])
            cx,cy=round(x0+centroids[i][0]),round(y0+centroids[i][1])
    padded=cv2.copyMakeBorder(frame,24,24,24,24,cv2.BORDER_CONSTANT)
    return padded[cy:cy+48,cx:cx+48].copy()


def _pose_features(patch):
    hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    blue=cv2.inRange(hsv,np.array((105,100,55)),np.array((132,255,255))).astype(np.float32)/255
    green=cv2.inRange(hsv,np.array((35,80,45)),np.array((103,255,255))).astype(np.float32)/255
    features=np.stack((blue,green),axis=-1)
    features=cv2.GaussianBlur(features,(3,3),0.6)
    return features.reshape(-1)


@lru_cache(maxsize=4)
def _pose_bank(reference_path):
    reference=_read_image(reference_path)
    if reference is None:
        raise ValueError("无法读取 LANDING 姿态模板")
    vectors=[]
    angles=np.arange(-180,180,10,dtype=float)
    for degrees in angles:
        # OpenCV positive rotation is counterclockwise; our positive is clockwise.
        matrix=cv2.getRotationMatrix2D((24,24),-degrees,1)
        rotated=cv2.warpAffine(reference,matrix,(48,48))
        vector=_pose_features(rotated)
        vectors.append(vector/max(1e-9,float(np.linalg.norm(vector))))
    return angles,np.stack(vectors)


def estimate_pose(canonical_bgr, player_entity, reference_path=None):
    """Return radians, upright=0 / clockwise positive, only if bank is decisive."""
    path=Path(reference_path or DEFAULT_ATLAS.parent/"robot_upright.png")
    if not path.is_file():
        return {"pose_confidence":0.0,"pose_known":False}
    patch=_robot_patch(canonical_bgr,(player_entity.x,player_entity.y))
    vector=_pose_features(patch)
    norm=float(np.linalg.norm(vector))
    if norm<3:
        return {"pose_confidence":0.0,"pose_known":False}
    angles,bank=_pose_bank(str(path.resolve()))
    scores=bank@(vector/norm)
    best=int(np.argmax(scores))
    score=float(scores[best])
    differences=np.abs((angles-angles[best]+180)%360-180)
    alternatives=scores[differences>=50]
    margin=score-float(np.max(alternatives)) if alternatives.size else score
    known=score>=0.58 and margin>=0.025
    result={"pose_confidence":score,"pose_margin":margin,"pose_known":known}
    if known:
        result["vehicle_angle_radians"]=math.radians(float(angles[best]))
    return result


@lru_cache(maxsize=4)
def _atlas(path):
    return LandingAtlas(path)


def _angle_delta(target,current):
    return (target-current+math.pi)%(2*math.pi)-math.pi


def decide_landing(obs, parameters=None):
    parameters=parameters or {}
    base={"experimental":parameters.get("calibrated") is not True,"mode":"waiting"}
    if obs.phase!="playing" or obs.confidence<0.8:
        return Decision(reason="LANDING：等待可靠游戏画面",diagnostics=base)
    players=obs.of_kind("player")
    if len(players)!=1:
        return Decision(reason="LANDING：机器人缺失或不唯一",diagnostics=base)
    player=players[0]
    angle=obs.metrics.get("vehicle_angle_radians")
    pose_score=obs.metrics.get("pose_confidence",1.0 if angle is not None else 0.0)
    if angle is None or not math.isfinite(angle) or not math.isfinite(pose_score) or pose_score<parameters.get("minimum_pose_score",0.58):
        return Decision(reason="LANDING：朝向不明确，等待姿态识别",diagnostics=base)
    map_index=obs.metrics.get("map_index",-1)
    map_score=obs.metrics.get("map_match_score",0.0)
    if not math.isfinite(map_index) or not math.isfinite(map_score) or map_index<0 or int(map_index)!=map_index or map_score<parameters.get("minimum_map_score",0.60):
        return Decision(reason="LANDING：地图未可靠匹配，等待地形识别",diagnostics=base)
    goals=obs.of_kind("goal")
    goal=(goals[0].x,goals[0].y-0.02) if len(goals)==1 else None
    path=parameters.get("waypoints")
    map_id=parameters.get("map_id")
    if path is None:
        try:
            atlas=_atlas(str(parameters.get("atlas_registry",DEFAULT_ATLAS)))
            index=int(map_index)
            if not 0<=index<len(atlas.layouts):
                return Decision(reason="LANDING：地图索引超出本地图库",diagnostics=base)
            map_id=atlas.layouts[index]["id"]
            path=atlas.route(index,(player.x,player.y),goal,
                             parameters.get("clearance",max(player.w/2,0.018)))
        except (OSError,ValueError,KeyError):
            return Decision(reason="LANDING：地图库尚未构建或已失效",diagnostics=base)
    if not path:
        return Decision(reason="LANDING：地形包络内暂未找到安全路线",diagnostics=base)
    points=[(float(p[0]),float(p[1])) for p in path]
    if not all(math.isfinite(x) and math.isfinite(y) and 0<=x<=1 and 0<=y<=1 for x,y in points):
        return Decision(reason="LANDING：路线坐标无效",diagnostics=base)
    # Replanning starts at the current robot position; select the first waypoint
    # far enough ahead, not the final destination behind a wall.
    target=next((p for p in points if math.hypot(p[0]-player.x,p[1]-player.y)>0.045),points[-1])
    dx,dy=target[0]-player.x,target[1]-player.y
    speed=math.hypot(player.vx,player.vy)
    final_distance=math.hypot(points[-1][0]-player.x,points[-1][1]-player.y)
    braking=final_distance<parameters.get("braking_distance",0.12) and speed>parameters.get("landing_speed",0.10)
    aspect=parameters.get("coordinate_aspect_ratio",640/480)
    if braking:
        desired=math.atan2(-player.vx*aspect,player.vy) if speed>0.01 else 0.0
        mode="braking"
    else:
        # Upward support plus lateral/vertical correction; final landing asks
        # for upright posture rather than blindly pointing at the squid.
        tx=dx-parameters.get("velocity_damping",0.6)*player.vx
        ty=dy-parameters.get("velocity_damping",0.6)*player.vy
        # A distant horizontal waypoint should not demand a 90-degree roll
        # and discard all upward support. The support/tilt gains are explicit
        # experimental control heuristics, not measured gravity constants.
        support=parameters.get("vertical_support",0.18)
        desired=math.atan2(tx*aspect,max(0.04,-ty+support)) if abs(tx)+abs(ty)>0.01 else 0.0
        tilt=parameters.get("maximum_navigation_tilt_radians",0.75)
        desired=max(-tilt,min(tilt,desired))
        if final_distance<0.06:
            desired=0.0
        mode="landing" if final_distance<0.1 else "navigate"
    angular=obs.metrics.get("angular_velocity",0.0)
    if not math.isfinite(angular):
        angular=0.0
    error=_angle_delta(desired,angle)-parameters.get("angular_damping",0.18)*angular
    tolerance=parameters.get("angle_tolerance_radians",0.18)
    buttons=set()
    if error>tolerance:
        buttons.add(Button.ZL)
        action="左喷射，顺时针调整"
    elif error < -tolerance:
        buttons.add(Button.ZR)
        action="右喷射，逆时针调整"
    else:
        # Both jets point along the current heading. Stop thrust when already
        # moving sufficiently toward the waypoint; unlike LLM planning this is
        # a fixed-size numeric calculation.
        toward=(player.vx*dx+player.vy*dy)/max(0.01,math.hypot(dx,dy))
        needs_lift=dy < -0.025 or player.vy>0.08
        needs_forward=toward<parameters.get("cruise_speed",0.15) and dy<0.06
        if braking or needs_lift or needs_forward:
            buttons.update((Button.ZL,Button.ZR))
            action="双喷射制动" if braking else "双喷射推进"
        else:
            action="松开喷射，利用惯性"
    # Dynamic hazards in the immediate body neighborhood override aggressive
    # propulsion; detailed enemy/wind dynamics remain a model-calibration task.
    nearby=[e for e in obs.entities if e.kind in ("enemy","obstacle")
            and math.hypot(e.x-player.x,e.y-player.y)<(e.w+player.w)/2+0.05]
    if nearby and len(buttons)==2 and not braking:
        buttons.clear()
        mode="hazard_wait"
        action="近身动态障碍，松开喷射并重新判断"
    diagnostics={**base,"mode":mode,"map_id":map_id,"map_index":int(map_index),
                 "target":list(target),"desired_angle_radians":desired,"angle_error_radians":error,
                 "route":points,"pose_confidence":pose_score,"map_match_score":map_score,
                 "nearby_hazards":len(nearby)}
    return Decision(ControllerState(frozenset(buttons)),"LANDING："+action,
                    min(obs.confidence,float(map_score),float(pose_score)),diagnostics)
