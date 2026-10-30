"""Save the public reference once, with provenance and a local decode check.

Run with the project's local tooling packages, without changing conda base:
python tools/download_reference.py
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.youtube.com/watch?v=l3dMQV1sar8"
TARGET = ROOT / "artifacts/reference_videos/utagaming-l3dMQV1sar8"


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    sys.path[:0] = [str(ROOT / "artifacts/tooling/yt_dlp"),
                   str(ROOT / "artifacts/tooling/ffmpeg")]
    import cv2
    import imageio_ffmpeg
    from yt_dlp import YoutubeDL

    TARGET.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestvideo[ext=mp4][vcodec^=avc1][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]",
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "merge_output_format": "mp4",
        "outtmpl": str(TARGET / "reference.%(ext)s"),
        "cachedir": str(ROOT / "artifacts/tooling/cache"),
        "noplaylist": True,
        "writeinfojson": True,
        "writedescription": True,
        "noprogress": True,
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
    }
    with YoutubeDL(options) as downloader:
        info = downloader.extract_info(URL, download=True)

    path = TARGET / "reference.mp4"
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError("下载文件无法打开")
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not fps > 0 or count < 1:
            raise RuntimeError("无有效视频帧")
        duration = count / fps
        if abs(duration - info["duration"]) > 2:
            raise RuntimeError("本地视频时长与来源不符")
        samples = []
        for seconds in (0, duration / 2, duration - 1):
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"无法解码 {seconds:.1f} 秒处的帧")
            samples.append(round(seconds, 3))
    finally:
        capture.release()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "source_url": URL, "video_id": info["id"], "title": info["title"],
        "uploader": info.get("uploader"), "upload_date": info.get("upload_date"),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_file": path.name, "bytes": path.stat().st_size, "sha256": digest,
        "width": width, "height": height, "fps": fps, "frame_count": count,
        "duration_seconds": duration, "decode_checked_seconds": samples,
        "chapters": info.get("chapters", []),
        "selected_formats": [{"format_id": f.get("format_id"), "vcodec": f.get("vcodec"),
                              "acodec": f.get("acodec")} for f in info.get("requested_formats", [])],
    }
    (TARGET / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
