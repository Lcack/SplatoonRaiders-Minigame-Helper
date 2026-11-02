"""Summarise real round traces; observations are not confirmed collisions."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from raiders_helper.mining_result_reader import inspect_result
from read_mining_score_offline import read_score as score_candidate


def read_rows(path):
    rows = []
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A currently running worker may have an incomplete tail.
    return rows


def report(batch):
    results, tiles = [], []
    for directory in sorted(batch.glob('round-*')):
        game = directory / 'gameplay'
        rows = read_rows(game / 'trace.jsonl')
        playing = [r for r in rows if r['observation']['phase'] == 'playing']
        modes = Counter(r['decision']['diagnostics'].get('mode', 'none') for r in playing)
        reasons = Counter(r['decision']['reason'] for r in playing if not r['decision']['diagnostics'].get('mode'))
        ages = [r['frame_age_ms'] for r in playing]
        hit_times = [r['observation']['timestamp'] for r in rows if r['observation']['metrics'].get('hit_body_detected')]
        episodes = sum(i == 0 or t-hit_times[i-1] > .4 for i,t in enumerate(hit_times))
        item = {'round': directory.name, 'frames': len(rows), 'playing_frames': len(playing),
                'modes': dict(modes), 'neutral_reasons': dict(reasons),
                'decision_age_ms_p50_p90_max': [float(x) for x in np.percentile(ages, [50,90,100])] if ages else [],
                'playing_over_200ms': sum(x > 200 for x in ages),
                'purple_body_candidate_episodes': episodes,
                'scope': 'advice and visual candidates; not confirmed collisions or executed inputs'}
        version = directory / 'loaded-version.json'
        if version.exists():
            item['version_id'] = json.loads(version.read_text(encoding='utf-8'))['version_id']
        image_path = game / 'last-frame.png'
        if image_path.exists():
            frame = cv2.imdecode(np.frombuffer(image_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            item['result_reading'] = asdict(inspect_result(frame))
            item['score_candidate'] = score_candidate(frame) if item['result_reading']['result_visible'] else None
            item['result_image'] = str(image_path.resolve())
            tile = cv2.resize(frame, (640,360))
            cv2.putText(tile, directory.name, (10,25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            tiles.append(tile)
        results.append(item)
    output = {'batch': str(batch.resolve()), 'rounds': results}
    (batch / 'evaluation.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    if tiles:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)])
        cv2.imencode('.jpg', sheet)[1].tofile(str(batch / 'results.jpg'))
    return output


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.batch), ensure_ascii=False, indent=2))
