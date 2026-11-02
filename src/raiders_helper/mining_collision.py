"""Optional current-ring circle hypothesis with unchanged player bounds.

Visible rings are evidence about appearance, not measured collision boxes.
Unknown, stale or ambiguous shapes retain their original AABB semantics.
"""
from dataclasses import dataclass
import math

from raiders_helper.mining_planner import contact_fraction as box_contact


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def _slab_entry(x0,y0,x1,y1,half_x,half_y):
    entry,leave=0.,1.
    for start,end,radius in ((x0,x1,half_x),(y0,y1,half_y)):
        delta=end-start
        if abs(delta)<1e-14:
            if abs(start)>radius:return None
        else:
            a,b=(-radius-start)/delta,(radius-start)/delta
            entry=max(entry,min(a,b));leave=min(leave,max(a,b))
            if entry>leave:return None
    return entry


def _disk_entry(x0,y0,x1,y1,radius):
    c=x0*x0+y0*y0-radius*radius
    if c<=0:return 0.
    dx,dy=x1-x0,y1-y0;a=dx*dx+dy*dy
    if a<=1e-24:return None
    b=x0*dx+y0*dy;discriminant=b*b-a*c
    if discriminant<0:return None
    entry=(-b-math.sqrt(discriminant))/a
    return entry if 0<=entry<=1 else None


def rounded_rectangle_entry(x0,y0,x1,y1,half_x,half_y,radius):
    """Exact line contact with rectangle Minkowski-summed by a disk."""
    values=[_slab_entry(x0,y0,x1,y1,half_x+radius,half_y),
            _slab_entry(x0,y0,x1,y1,half_x,half_y+radius)]
    values.extend(_disk_entry(x0-sx*half_x,y0-sy*half_y,x1-sx*half_x,y1-sy*half_y,radius)
                  for sx in (-1,1) for sy in (-1,1))
    present=[value for value in values if value is not None]
    return min(present) if present else None


def segment_rectangle_distance(x0,y0,x1,y1,half_x,half_y):
    """Exact minimum distance to an axis-aligned rectangle, in pixel units."""
    dx,dy=x1-x0,y1-y0;cuts={0.,1.}
    for start,delta,radius in ((x0,dx,half_x),(y0,dy,half_y)):
        if abs(delta)>1e-14:
            cuts.update(t for edge in (-radius,radius) if 0<(t:=(edge-start)/delta)<1)
    cuts=sorted(cuts);best=math.inf
    def distance_squared(t):
        return max(abs(x0+dx*t)-half_x,0.)**2+max(abs(y0+dy*t)-half_y,0.)**2
    for lower,upper in zip(cuts,cuts[1:]):
        middle=(lower+upper)/2;terms=[]
        for start,delta,radius in ((x0,dx,half_x),(y0,dy,half_y)):
            position=start+delta*middle
            terms.append((delta,start-math.copysign(radius,position)) if abs(position)>radius else (0.,0.))
        quadratic=sum(a*a for a,b in terms)
        optimum=(-sum(a*b for a,b in terms)/quadratic if quadratic>1e-24 else middle)
        best=min(best,distance_squared(lower),distance_squared(upper),distance_squared(max(lower,min(upper,optimum))))
    return math.sqrt(max(0.,best))


def circle_rectangle_area(left,right,top,bottom,radius):
    """Exact disk/rectangle intersection area by piecewise integration."""
    left=max(-radius,left);right=min(radius,right)
    if left>=right or top>=radius or bottom<=-radius:return 0.
    cuts={left,right}
    for edge in (top,bottom):
        if abs(edge)<radius:
            x=math.sqrt(max(0.,radius*radius-edge*edge))
            cuts.update(t for t in (-x,x) if left<t<right)
    def primitive(x):
        return .5*(x*math.sqrt(max(0.,radius*radius-x*x))+radius*radius*math.asin(max(-1.,min(1.,x/radius))))
    cuts=sorted(cuts);area=0.
    for a,b in zip(cuts,cuts[1:]):
        height=math.sqrt(max(0.,radius*radius-((a+b)/2)**2))
        if min(bottom,height)<=max(top,-height):continue
        integral=primitive(b)-primitive(a)
        upper=bottom*(b-a) if bottom<height else integral
        lower=top*(b-a) if top>-height else -integral
        area+=upper-lower
    return max(0.,area)


@dataclass(frozen=True)
class Circle:
    center_x_pixels:float
    center_y_pixels:float
    radius_pixels:float
    visual_radius_pixels:float


class ObstacleGeometry:
    def __init__(self,observation=None,parameters=None):
        config=(parameters or {}).get('circle_collision',{})
        if not isinstance(config,dict) or type(config.get('enabled',False)) is not bool:
            raise ValueError('circle_collision.enabled must be boolean')
        self.enabled=config.get('enabled',False);self.circles={};self.rejected={}
        padding=config.get('extra_radius_pixels',0.)
        if not _finite(padding) or not 0<=padding<=12:raise ValueError('circle radius extra padding must be within 0..12 pixels')
        self.padding=padding
        if not self.enabled or observation is None:return
        entities={}
        for entity in observation.entities:
            if (entity.kind in {'obstacle','rock'} and all(_finite(v) for v in
                    (entity.x,entity.y,entity.w,entity.h,entity.confidence))
                    and entity.w>0 and entity.h>0 and entity.confidence>=.8):
                entities.setdefault(entity.track_id,[]).append(entity)
        packets=observation.metrics.get('obstacle_shapes',[])
        if not isinstance(packets,(list,tuple)):return
        packets_by_id={}
        for packet in packets:
            if isinstance(packet,dict) and type(packet.get('track_id')) is int:
                packets_by_id.setdefault(packet['track_id'],[]).append(packet)
        for identity,items in packets_by_id.items():
            if identity<0 or len(items)!=1 or len(entities.get(identity,()))!=1:
                self.rejected[identity]='ambiguous_current_identity';continue
            p=items[0];e=entities[identity][0]
            required=('timestamp','center_x_pixels','center_y_pixels','radius_pixels','rim_sectors','rim_contrast')
            if (any(not _finite(p.get(key)) for key in required) or p.get('shape')!='circle'
                    or p.get('coordinate_space')!='canonical_640x480_pixels' or p.get('source')!='current_bright_ring'
                    or abs(p['timestamp']-observation.timestamp)>1e-6 or p['rim_sectors']<9 or p['rim_contrast']<15):
                self.rejected[identity]='missing_current_ring_contract';continue
            cx,cy,r=p['center_x_pixels'],p['center_y_pixels'],p['radius_pixels']
            if (not 0<=cx<=640 or not 0<=cy<=480 or not 0<r<=120
                    or math.hypot(cx-e.x*640,cy-e.y*480)>8
                    or abs(2*(r+3)-max(e.w*640,e.h*480))>max(12,.25*2*(r+3))):
                self.rejected[identity]='current_ring_does_not_match_entity';continue
            # The detector's box already normally adds 3 px around its rim.
            # Keep both original axial extents even if ring and box centers differ.
            radius=max(r+3,e.w*640/2+abs(cx-e.x*640),e.h*480/2+abs(cy-e.y*480))+padding
            self.circles[identity]=Circle(cx,cy,radius,r)

    def shape(self,entity):
        return self.circles.get(entity.track_id) if entity.kind in {'obstacle','rock'} else None

    def bounds(self,entity):
        c=self.shape(entity)
        if c:return ((c.center_x_pixels-c.radius_pixels)/640,(c.center_x_pixels+c.radius_pixels)/640,
                     (c.center_y_pixels-c.radius_pixels)/480,(c.center_y_pixels+c.radius_pixels)/480)
        return entity.x-entity.w/2,entity.x+entity.w/2,entity.y-entity.h/2,entity.y+entity.h/2

    def contact_fraction(self,x0,y0,x1,y1,entity,rx,ry):
        c=self.shape(entity)
        if c is None:return box_contact(x0,y0,x1,y1,entity,rx,ry)
        ax,ay,bx,by=x0*640-c.center_x_pixels,y0*480-c.center_y_pixels,x1*640-c.center_x_pixels,y1*480-c.center_y_pixels
        if (min(ax,bx)>c.radius_pixels+rx*640 or max(ax,bx)<-c.radius_pixels-rx*640
                or min(ay,by)>c.radius_pixels+ry*480 or max(ay,by)<-c.radius_pixels-ry*480):return None
        return rounded_rectangle_entry(ax,ay,bx,by,rx*640,ry*480,c.radius_pixels)

    def overlap_area(self,x,y,entity,rx,ry):
        c=self.shape(entity)
        if c is None:
            width=max(0.,min(x+rx,entity.x+entity.w/2)-max(x-rx,entity.x-entity.w/2))
            height=max(0.,min(y+ry,entity.y+entity.h/2)-max(y-ry,entity.y-entity.h/2))
            return width*height
        cx,cy=x*640-c.center_x_pixels,y*480-c.center_y_pixels
        return circle_rectangle_area(cx-rx*640,cx+rx*640,cy-ry*480,cy+ry*480,c.radius_pixels)/(640*480)

    def segment_clearance(self,x0,y0,x1,y1,entity,rx,ry):
        c=self.shape(entity)
        if c:
            distance=segment_rectangle_distance(x0*640-c.center_x_pixels,y0*480-c.center_y_pixels,
                x1*640-c.center_x_pixels,y1*480-c.center_y_pixels,rx*640,ry*480)-c.radius_pixels
        else:
            distance=segment_rectangle_distance((x0-entity.x)*640,(y0-entity.y)*480,
                (x1-entity.x)*640,(y1-entity.y)*480,(rx+entity.w/2)*640,(ry+entity.h/2)*480)
        return max(0.,distance)/640

    def diagnostics(self):
        return {'enabled':self.enabled,'model':'current_visible_ring_circle_collision_hypothesis',
            'collision_shape_measured':False,'extra_radius_pixels':self.padding,
            'matched_track_ids':sorted(self.circles),'rejected':self.rejected,
            'unknown_shape':'original_aabb','player_and_wall_envelope':'unchanged',
            'circles':{identity:vars(circle) for identity,circle in self.circles.items()}}


BOX_GEOMETRY=ObstacleGeometry()
