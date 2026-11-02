"""Source-index actual manually checked JUGGLING result glyphs."""
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from read_mining_score_offline import normalise,score_components,read_score
from raiders_helper.mining_result_reader import _canonical,DEFAULT_VIEWPORT

ROOT=Path(__file__).resolve().parents[1]
output=ROOT/'assets/juggling_live_score'
output.mkdir(parents=True,exist_ok=True)
index={'schema_version':1,'label_method':'Actual JUGGLING result screenshots manually reviewed; unseen digits retain source-indexed MINING glyphs',
       'templates':{},'sources':{}}
sources=[('baseline-02/round-0001','03345','B'),('baseline-02/round-0002','09420','A'),
         ('confirm-01/round-0001','04125','B'),('confirm-01/round-0002','06660','B'),
         ('hot-validation-01/worker-0001/round-0001','08280','A')]
for directory,text,rank in sources:
    path=ROOT/'artifacts/juggling-live'/directory/'result.png'
    raw=path.read_bytes();frame=cv2.imdecode(np.frombuffer(raw,np.uint8),1)
    parts=score_components(_canonical(frame,DEFAULT_VIEWPORT))
    assert parts is not None and len(parts)==5
    index['sources'][directory]={'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),
                                'score_text':text,'rank':rank,'manually_verified':True}
    for digit,(mask,box) in zip(text,parts):
        if digit in index['templates']:continue
        target=output/(digit+'.png');cv2.imencode('.png',normalise(mask))[1].tofile(str(target))
        index['templates'][digit]={'file':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                                   'source':directory,'box_in_score_roi':box}
previous=ROOT/'artifacts/mining-score-reader'
original=json.loads((previous/'source.json').read_text(encoding='utf8'))
for digit in ('7','8'):
    if digit in index['templates']:continue
    entry=original['templates'][digit];target=output/(digit+'.png')
    shutil.copyfile(previous/entry['file'],target)
    index['templates'][digit]={'file':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                               'original_index':'artifacts/mining-score-reader/source.json','original_entry':entry}
(output/'source.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf8')
for directory,expected,_ in sources+[('confirm-01/round-0003','05235','B')]:
    path=ROOT/'artifacts/juggling-live'/directory/'result.png'
    frame=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
    result=read_score(frame,assets=output)
    print(directory,result['text'],'expected',expected,'min_match',round(min(d['match'] for d in result['digits']),3))
