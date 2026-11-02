"""A live raw frame is independent of old GUI analysis and owns its timestamp."""
from concurrent.futures import ThreadPoolExecutor
import http.client
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np
import pytest

from raiders_helper.capture.models import FramePacket
from raiders_helper.debug_api import DebugHub


def packet(value, sequence, stamp=None):
    return FramePacket(np.full((48,64,3),value,np.uint8), time.perf_counter() if stamp is None else stamp,sequence,60)


def decode(data):
    return cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)


def test_capture_provider_beats_old_analyzed_frame_and_keeps_screen_separate(tmp_path):
    newest=[packet(90,20)]
    hub=DebugHub(tmp_path,capture_provider=lambda:newest[0])
    hub.publish(status={"screen":"menu:mining"},frame=packet(20,2).frame,sequence=2,captured_at=1.)
    image,meta=hub.frame()
    assert np.all(decode(image)==90) and meta['sequence']==20
    assert meta['captured_at']==newest[0].timestamp and meta['source']=='capture_worker_latest'
    assert hub.status()['screen']=='menu:mining'
    assert hub.status()['frames']['capture']['sequence']==20
    newest[0]=packet(100,21)
    image,meta=hub.frame(max_width=320)
    assert meta['sequence']==21 and np.all(decode(image)==100)
    hub.close()


def test_empty_provider_does_not_fall_back_to_analyzed_capture(tmp_path):
    hub=DebugHub(tmp_path,capture_provider=lambda:None)
    hub.publish(frame=packet(20,2).frame,sequence=2,captured_at=1.)
    with pytest.raises(FileNotFoundError):hub.frame()
    assert hub.status()['frames']['capture']=={'available':False,'source':'capture_worker_latest'}
    hub.close()


def test_stopped_capture_keeps_original_age_instead_of_becoming_fresh(tmp_path):
    old=packet(30,3,time.perf_counter()-10)
    hub=DebugHub(tmp_path,capture_provider=lambda:old)
    _,first=hub.frame();_,second=hub.frame()
    assert first['captured_at']==second['captured_at']==old.timestamp
    assert first['sequence']==second['sequence']==3
    assert first['age_seconds']>=10 and second['age_seconds']>=first['age_seconds']
    assert hub.status()['frames']['capture']['age_seconds']>=10
    hub.close()


def test_analysis_request_never_calls_raw_provider(tmp_path):
    def fail():raise AssertionError('analysis must not use raw provider')
    hub=DebugHub(tmp_path,capture_provider=fail)
    hub.publish(frame=packet(60,2).frame,sequence=2,captured_at=1.,kind='analysis')
    image,meta=hub.frame(kind='analysis')
    assert meta['sequence']==2 and meta['source']=='memory'
    assert np.all(decode(image)==60)
    hub.close()


def test_provider_runs_outside_hub_lock_and_close_rejects_late_frame(tmp_path):
    entered,unblock=threading.Event(),threading.Event()
    def provider():
        entered.set()
        assert unblock.wait(2)
        return packet(70,7)
    hub=DebugHub(tmp_path,capture_provider=provider)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(hub.frame)
        assert entered.wait(1)
        # This lock-taking call must finish while provider is still blocked.
        assert hub.submit({'command':'pause'})['status']=='queued'
        hub.close()
        unblock.set()
        with pytest.raises(FileNotFoundError):future.result(timeout=1)
    with pytest.raises(FileNotFoundError):hub.frame()
    assert hub.status()['frames']=={}


def test_close_during_png_encode_rejects_result(tmp_path,monkeypatch):
    entered,unblock=threading.Event(),threading.Event()
    encode=cv2.imencode
    def slow(*args,**kwargs):
        entered.set();assert unblock.wait(2)
        return encode(*args,**kwargs)
    monkeypatch.setattr(cv2,'imencode',slow)
    hub=DebugHub(tmp_path,capture_provider=lambda:packet(80,8))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(hub.frame)
        assert entered.wait(1)
        hub.close();unblock.set()
        with pytest.raises(FileNotFoundError):future.result(timeout=1)


def test_simultaneous_requests_cannot_mix_pixel_and_sequence_snapshots(tmp_path,monkeypatch):
    count=0;lock=threading.Lock();barrier=threading.Barrier(2)
    encode=cv2.imencode
    def provider():
        nonlocal count
        with lock:
            count+=1
            return packet(count*20,count,float(count))
    def concurrent(*args,**kwargs):
        barrier.wait(timeout=2)
        return encode(*args,**kwargs)
    monkeypatch.setattr(cv2,'imencode',concurrent)
    hub=DebugHub(tmp_path,capture_provider=provider)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(hub.frame) for _ in range(2)]
        for future in futures:
            image,meta=future.result(timeout=3)
            assert np.all(decode(image)==meta['sequence']*20)
            assert meta['captured_at']==float(meta['sequence'])
    hub.close()


def test_request_owns_copy_before_encoding(tmp_path,monkeypatch):
    source=packet(40,4)
    encode=cv2.imencode
    def mutate_original(*args,**kwargs):
        source.frame[:]=0
        return encode(*args,**kwargs)
    monkeypatch.setattr(cv2,'imencode',mutate_original)
    hub=DebugHub(tmp_path,capture_provider=lambda:source)
    image,meta=hub.frame()
    assert np.all(decode(image)==40) and meta['sequence']==4
    hub.close()


def test_http_endpoint_returns_latest_source_and_matching_timestamp_headers(tmp_path):
    source=packet(90,99)
    hub=DebugHub(tmp_path,capture_provider=lambda:source)
    endpoint=hub.start()
    address=urlsplit(endpoint['url'])
    connection=http.client.HTTPConnection(address.hostname,address.port,timeout=2)
    try:
        connection.request('GET','/frame?kind=capture&max_width=960',
                           headers={endpoint['token_header']:endpoint['token']})
        response=connection.getresponse();image=response.read()
        assert response.status==200
        assert response.getheader('X-Frame-Source')=='capture_worker_latest'
        assert response.getheader('X-Frame-Sequence')=='99'
        assert abs(float(response.getheader('X-Frame-Captured-At'))-source.timestamp)<1e-8
        assert np.all(decode(image)==90)
    finally:
        connection.close();hub.close()
