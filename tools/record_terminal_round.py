"""Record exactly one actual MINING/JUGGLING terminal-window round.

No desktop-wide capture, synthetic UI, video speed changes, or game inputs are
generated here. The existing coordinator owns all controls. The recording is
trimmed against the three actual countdown glyphs visible in the window video.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from raiders_helper.vision import crop_roi
from mining_supervisor import LiveClient


def write_json(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    try:temp.replace(path)
    except PermissionError:
        if path.name!='terminal_recording.json':raise
        # The GUI may briefly hold this optional heartbeat on Windows. The
        # next update retries; video/control must never stop for display I/O.


def countdown_start(video,layout,game='mining'):
    config=json.loads((ROOT/'profiles/session.local.json').read_text(encoding='utf8'))
    countdown=next(s for s in config['screens'] if s['name']=='countdown:'+game)
    specs=countdown['all_of'][0]['any_of']
    templates=[cv2.imdecode(np.frombuffer((ROOT/'profiles'/s['path']).read_bytes(),np.uint8),0) for s in specs]
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    x,y,w,h=layout['preview_rect'];runs=[[],[],[]];found={};peaks=[0.,0.,0.]
    count=0
    try:
        while count<fps*20:
            ok,frame=cap.read()
            if not ok:break
            preview=frame[y:y+h,x:x+w]
            canonical=cv2.resize(crop_roi(preview,config['viewport']),(640,480))
            gray=cv2.cvtColor(canonical,cv2.COLOR_BGR2GRAY)
            for i,(spec,template) in enumerate(zip(specs,templates)):
                tile=crop_roi(gray,spec['roi'])
                score=float(cv2.matchTemplate(tile,template,cv2.TM_CCOEFF_NORMED).max())
                peaks[i]=max(peaks[i],score)
                if score>=.80:runs[i].append(count/fps)
                else:runs[i]=[]
                if len(runs[i])>=3 and i not in found:found[i]=runs[i][0]
            count+=1
            if len(found)==3:break
    finally:cap.release()
    if set(found)!={0,1,2} or not found[0]<found[1]<found[2]:
        raise RuntimeError(f'录像中未确认完整3/2/1顺序，保留原片等待检查：{found}, peaks={peaks}')
    return found[0],{'times_3_2_1':found,'template_peaks':peaks,'fps':fps}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=1,choices=[1])
    parser.add_argument('--game',choices=['mining','juggling'],default='mining')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    ffmpeg=ROOT/'artifacts/recording-deps/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
    client=LiveClient(ROOT/'artifacts/control_endpoint.json')
    status=client.request('/status');session=Path(status['output'])
    layout=json.loads((session/'terminal_layout.json').read_text(encoding='utf8'))
    report_path=session/'terminal_recording.json'
    width,height=layout['client_size'];raw=args.output/'window-preroll.mp4';final=args.output/(args.game.upper()+'-终端自主运行.mp4')
    metadata={'capture':'actual_gdigrab_named_terminal_window','layout':layout,'audio':False,'speed':1.0,'batch':str(args.batch),'game':args.game,'rounds':1}
    progress=args.output/'encoder-progress.txt'
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    recorder=None;coordinator=None
    with (args.output/'encoder.log').open('w',encoding='utf8') as log:
        command=[str(ffmpeg),'-hide_banner','-loglevel','warning','-f','gdigrab','-framerate','30','-draw_mouse','0',
            '-video_size',f'{width}x{height}','-i','title='+layout['window_title'],'-an','-c:v','libx264',
            '-preset','ultrafast','-crf','20','-threads','2','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-pix_fmt','yuv420p','-movflags','+faststart','-progress',str(progress),str(raw)]
        try:
            recorder=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=log,creationflags=flags)
            ready_by=time.monotonic()+10
            while time.monotonic()<ready_by:
                if recorder.poll() is not None:raise RuntimeError('窗口编码器未能启动，未启动游戏')
                if progress.exists() and 'frame=' in progress.read_text(encoding='utf8',errors='replace'):break
                time.sleep(.05)
            else:raise RuntimeError('窗口编码器未报告第一帧，未启动游戏')
            write_json(report_path,{'running':True,'updated_unix':time.time(),'output':str(final)})
            coordinator_log=(args.output/'coordinator.log').open('w',encoding='utf8')
            mining_command=[sys.executable,str(ROOT/'tools/mining_hot_reload.py'),
                '--endpoint',str(ROOT/'artifacts/control_endpoint.json'),'--profile',str(ROOT/'profiles/mining.supervised.local.json'),
                '--parameters',str(ROOT/'profiles/mining.ai.local.json'),'--output',str(args.batch),
                '--rounds','1','--seconds-per-round','600','--round-pause-seconds','3']
            juggling_command=[sys.executable,str(ROOT/'tools/juggling_hot_reload.py'),'--rounds','1','--seconds','600',
                '--profile',str(ROOT/'profiles/juggling.live.local.json'),'--parameters',str(ROOT/'profiles/juggling.ai.local.json'),
                '--output',str(args.batch)]
            coordinator=subprocess.Popen(juggling_command if args.game=='juggling' else mining_command,
                stdout=coordinator_log,stderr=subprocess.STDOUT,cwd=ROOT,creationflags=flags)
            print('终端窗口录像已开始；仅运行并录制一局 '+args.game.upper()+'。',flush=True)
            entered=False;played=False;heartbeat=0.;deadline=time.monotonic()+660;coordinator_done=None
            while time.monotonic()<deadline:
                if recorder.poll() is not None:raise RuntimeError('窗口编码器意外停止')
                current=client.request('/status');screen=current.get('screen','')
                entered=entered or screen.startswith('countdown:')
                played=played or (entered and screen.startswith('playing:'))
                if time.monotonic()-heartbeat>=.5:
                    write_json(report_path,{'running':True,'updated_unix':time.time(),'output':str(final)})
                    heartbeat=time.monotonic()
                if entered and played and screen.startswith('result:'):
                    metadata['ending_screen']=screen
                    time.sleep(1.5)
                    break
                code=coordinator.poll()
                if code is not None:
                    if coordinator_done is None:coordinator_done=time.monotonic()
                    if code or time.monotonic()-coordinator_done>3:raise RuntimeError('单局自主程序结束但未确认完整结算，保留已录片段')
                time.sleep(.1)
            else:raise RuntimeError('未在限定时间确认第一局结束')
        finally:
            if recorder is not None and recorder.poll() is None:
                recorder.stdin.write(b'q\n');recorder.stdin.flush()
                try:recorder.wait(timeout=10)
                except subprocess.TimeoutExpired:recorder.terminate();recorder.wait(timeout=5)
            write_json(report_path,{'running':False,'completed':bool(metadata.get('ending_screen')),'updated_unix':time.time(),'output':str(final)})
            if not metadata.get('ending_screen') and coordinator is not None and coordinator.poll() is None:
                subprocess.run(['taskkill','/PID',str(coordinator.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=flags)
                client.command({'command':'release'},wait=True)
    start,evidence=countdown_start(raw,layout,args.game);metadata.update(trim_start_seconds=start,countdown=evidence)
    subprocess.run([str(ffmpeg),'-hide_banner','-loglevel','error','-ss',str(start),'-i',str(raw),'-an',
        '-c:v','libx264','-preset','fast','-crf','20','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(final)],
        check=True,creationflags=flags)
    write_json(args.output/'recording.json',metadata)
    print('一局终端录像已完成：'+str(final),flush=True)
    code=coordinator.wait();coordinator_log.close()
    print(f'单局自主程序退出码：{code}',flush=True)
    return code


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf8')
    raise SystemExit(main())
