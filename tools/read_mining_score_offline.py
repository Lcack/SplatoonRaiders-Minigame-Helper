"""Read all five score digits from a result screenshot, without live control.

Templates come only from visually checked result digits, never timer glyphs
or menu high scores. Matching values are similarities, not probabilities;
unknown results stay unknown and rank A is still verified visually.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from raiders_helper.mining_result_reader import DEFAULT_VIEWPORT, _canonical, _ink, _match, inspect_result

ASSETS = ROOT / "artifacts/mining-score-reader"
SCORE_ROI = (335,170,111,28)


def score_components(canonical):
    x,y,w,h = SCORE_ROI
    tile = _ink(canonical[y:y+h,x:x+w])
    boxes = sorted(cv2.boundingRect(c) for c in cv2.findContours(tile,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0])
    boxes = [b for b in boxes if b[2]*b[3] >= 12]
    if len(boxes) != 5:
        return None
    # Five complete, evenly spaced cells: no clipped leading/trailing digit.
    for i,(left,top,width,height) in enumerate(boxes):
        if not (6 <= width <= 20 and 10 <= height <= 23 and 2 <= top <= 12
                and abs(left+width/2-(20+20*i)) <= 4):
            return None
    return [(tile[top:top+height,left:left+width],(left,top,width,height))
            for left,top,width,height in boxes]


def normalise(mask):
    height,width = mask.shape
    scale = min(24/width,24/height)
    glyph = cv2.resize(mask,(round(width*scale),round(height*scale)),interpolation=cv2.INTER_AREA)
    out = np.zeros((32,32),np.uint8)
    top,left = (32-glyph.shape[0])//2,(32-glyph.shape[1])//2
    out[top:top+glyph.shape[0],left:left+glyph.shape[1]] = glyph
    return out


@lru_cache(maxsize=2)
def template_bank(directory=ASSETS):
    directory = Path(directory)
    source = json.loads((directory/"source.json").read_text(encoding="utf8"))
    if source.get("schema_version") != 1 or set(source["templates"]) != set(map(str,range(10))):
        raise ValueError("score digit source index is incomplete")
    bank = []
    for digit in range(10):
        entry = source["templates"][str(digit)]
        raw = (directory/entry["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("score digit template hash mismatch")
        image = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (32,32) or image.std() < 1:
            raise ValueError("invalid score digit template")
        bank.append(image)
    return bank


def read_score(frame, viewport=DEFAULT_VIEWPORT, *, minimum_match=.90, minimum_margin=.08, assets=ASSETS):
    result = {"value":None,"text":None,"digits":[],"reason":"未确认完整结算画面",
              "scope":"offline five-digit result score; not rank inference or menu high score",
              "confidence_kind":"normalized_template_similarity_not_probability"}
    context = inspect_result(frame,viewport)
    if not context.result_visible:
        return result
    canonical = _canonical(frame,viewport)
    components = score_components(canonical)
    if components is None:
        result["reason"] = "结算分数字段不是完整五位或位置异常"
        return result
    try:
        bank = template_bank(Path(assets))
    except (OSError,ValueError,KeyError) as error:
        result["reason"] = "结算分数字形不可用："+str(error)
        return result
    for mask,box in components:
        glyph = cv2.copyMakeBorder(normalise(mask),2,2,2,2,cv2.BORDER_CONSTANT,value=0)
        scores = sorted(((digit,_match(glyph,template)) for digit,template in enumerate(bank)),key=lambda pair:-pair[1])
        result["digits"].append({"digit":scores[0][0],"match":scores[0][1],
            "margin":scores[0][1]-scores[1][1],"box_in_score_roi":list(box)})
    if not all(item["match"] >= minimum_match and item["margin"] >= minimum_margin for item in result["digits"]):
        result["reason"] = "至少一位字形不清或候选存在歧义"
        return result
    result["text"] = ''.join(str(item["digit"]) for item in result["digits"])
    result["value"] = int(result["text"])
    result["reason"] = "完整五位结算分数已匹配；等级另行确认"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image",type=Path)
    args = parser.parse_args()
    frame = cv2.imdecode(np.frombuffer(args.image.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf8")
    print(json.dumps(read_score(frame),ensure_ascii=False,indent=2))


if __name__ == "__main__":main()
