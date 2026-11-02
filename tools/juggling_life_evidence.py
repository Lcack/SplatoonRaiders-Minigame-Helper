"""Offline HUD evidence and short miss storyboards from actual snapshots."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def canonical(frame):
    h,w=frame.shape[:2]
    return cv2.resize(frame[round(h*55/1080):round(h*1015/1080),round(w/6):round(w*5/6)],(640,480))


def life_pixels(area):
    hsv=cv2.cvtColor(area,cv2.COLOR_BGR2HSV)
    mask=(((hsv[:,:,0]<=9)|(hsv[:,:,0]>=171))&(hsv[:,:,1]>=140)&(hsv[:,:,2]>=90))
    return [int(mask[8:27,250+20*i:270+20*i].sum()) for i in range(5)]


def evidence(directory):
    rows=[]
    for line in (directory/'trace.jsonl').read_text(encoding='utf8').splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    byseq={r['observation']['sequence']:r for r in rows};frames=[];events=[];previous=None
    for path in sorted(directory.glob('frame-*.jpg')):
        seq=int(path.stem.split('-')[1]);row=byseq.get(seq)
        if row is None:continue
        image=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
        pixels=life_pixels(canonical(image));count=sum(p>=100 for p in pixels)
        timestamp=row['observation']['timestamp'];item={'timestamp':timestamp,'sequence':seq,'lives':count,
            'slot_pixels':pixels,'path':str(path.resolve())};frames.append(item)
        if row['observation']['phase']=='playing':
            if previous is not None and 0<count<previous:
                events.append({'timestamp':timestamp,'sequence':seq,'before':previous,'after':count,
                    'scope':'first saved snapshot of HUD decrease; event can precede this by snapshot interval'})
            if count>0:previous=count
    panels=[]
    for event in events:
        tiles=[]
        for delta in (-1.2,-.6,.05):
            f=min(frames,key=lambda f:abs(f['timestamp']-event['timestamp']-delta))
            image=cv2.imdecode(np.frombuffer(Path(f['path']).read_bytes(),np.uint8),1)
            image=cv2.resize(image,(480,270));cv2.putText(image,f"{event['before']}->{event['after']} seq{f['sequence']}",(5,258),0,.4,(255,255,255),1);tiles.append(image)
        panels.append(np.hstack(tiles))
    if panels:cv2.imencode('.jpg',np.vstack(panels))[1].tofile(str(directory/'miss-storyboard.jpg'))
    report={'observed_initial_lives':max((f['lives'] for f in frames),default=None),'events':events,'frames':frames}
    (directory/'life-evidence.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args()
    report=evidence(args.directory);print(json.dumps({'initial_lives':report['observed_initial_lives'],'events':report['events']},indent=2))
