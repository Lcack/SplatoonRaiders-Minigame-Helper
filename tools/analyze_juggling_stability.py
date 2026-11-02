"""Count actual backend direction reversals over comparable early-game time."""
import argparse
import json
from pathlib import Path


def analyze(path,seconds):
    rows=[json.loads(s) for s in (path/'trace.jsonl').read_text(encoding='utf8').splitlines()]
    playing=[r for r in rows if r['decision'].get('diagnostics',{}).get('policy')=='juggling_interception']
    lo=playing[0]['observation']['timestamp'];hi=min(playing[-1]['observation']['timestamp'],lo+seconds)
    session=Path(json.loads((path/'summary.json').read_text(encoding='utf8'))['session'])
    events=[c for c in [json.loads(s) for s in (session/'commands.jsonl').read_text(encoding='utf8').splitlines()]
        if c.get('success') and c.get('event')=='send_state' and lo<=c['after']<=hi]
    result={'round':str(path),'seconds':hi-lo,'scope':'actual backend axes; reversals can be necessary, not classified mistakes','actors':{}}
    for axis in ('left_stick','right_stick'):
        sign=0;last=lo;reversals=[];last_value=0.;changes=0
        for e in events:
            value=e['state'][axis]['x']/32767
            if abs(value-last_value)>.05:changes+=1;last_value=value
            current=1 if value>.1 else -1 if value<-.1 else 0
            if current:
                if sign and current!=sign:reversals.append(e['after']-last)
                if current!=sign:last=e['after'];sign=current
        result['actors'][axis]={'meaningful_axis_changes':changes,'direction_reversals':len(reversals),
            'quick_reversals_under_250ms':sum(v<.25 for v in reversals),'reversals_per_second':len(reversals)/(hi-lo)}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('rounds',type=Path,nargs='+');p.add_argument('--seconds',type=float,default=90);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=[analyze(path,a.seconds) for path in a.rounds];a.output.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
