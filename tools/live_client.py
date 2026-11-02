"""Read/control a running Raiders terminal through its loopback capability API."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description="读取当前采集画面/状态、暂停自动流程并发送短时手柄输入")
    source=parser.add_mutually_exclusive_group()
    source.add_argument("--session",type=Path,help="运行目录，内有control_endpoint.json")
    source.add_argument("--endpoint",type=Path,help="发现文件；默认artifacts/control_endpoint.json")
    parser.add_argument("--timeout",type=float,default=8,help="请求和命令完成等待上限（秒）")
    parser.add_argument("--no-wait",action="store_true",help="只返回排队回执，不等待主循环执行")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("status")
    frame=sub.add_parser("frame")
    frame.add_argument("--kind",choices=["capture","analysis"],default="capture")
    frame.add_argument("--output",type=Path,required=True)
    events=sub.add_parser("events")
    events.add_argument("--limit",type=int,default=64)
    events.add_argument("--after",type=int,default=0)
    for name in ("pause","resume","release"):
        sub.add_parser(name)
    press=sub.add_parser("press")
    press.add_argument("--buttons",nargs="*",default=[])
    press.add_argument("--left-x",type=int,default=0)
    press.add_argument("--left-y",type=int,default=0)
    press.add_argument("--right-x",type=int,default=0)
    press.add_argument("--right-y",type=int,default=0)
    press.add_argument("--duration",type=float,default=.12,help="短脉冲秒数，0.03–1.0；先pause再press")
    args=parser.parse_args(argv)
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout需为有限正数")
        path=(args.session/"control_endpoint.json") if args.session else (args.endpoint or ROOT/"artifacts/control_endpoint.json")
        endpoint=json.loads(path.read_text(encoding="utf-8"))
        if not endpoint.get("running"):
            raise ValueError("调试接口已关闭")
        url=endpoint["url"]
        address=urlsplit(url)
        if (address.scheme!="http" or address.hostname not in {"127.0.0.1","localhost"}
                or address.username or address.password or address.path not in {"","/"}
                or address.query or address.fragment or not address.port):
            raise ValueError("发现文件必须指向本机HTTP接口")
        token_header=endpoint.get("token_header","X-Raiders-Token")
        if token_header not in {"X-Raiders-Token","X-Tableturf-Token"}:
            raise ValueError("发现文件包含未知认证头")
        opener=build_opener(ProxyHandler({}))

        def request(route,payload=None):
            data=None if payload is None else json.dumps(payload,allow_nan=False).encode("utf-8")
            req=Request(url.rstrip("/")+route,data=data,headers={
                token_header:endpoint["token"],"Content-Type":"application/json"})
            with opener.open(req,timeout=args.timeout) as response:
                body=response.read()
                return (body,dict(response.headers)) if route.startswith("/frame") else json.loads(body)

        if args.command=="frame":
            data,headers=request("/frame?kind="+args.kind)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_bytes(data)
            result={"path":str(args.output.resolve()),"sequence":headers.get("X-Frame-Sequence"),
                    "age_seconds":headers.get("X-Frame-Age-Seconds")}
        elif args.command=="status":
            result=request("/status")
        elif args.command=="events":
            result=request(f"/events?limit={args.limit}&after={args.after}")
        else:
            payload={"command":args.command}
            if args.command=="press":
                if not math.isfinite(args.duration) or not .03<=args.duration<=1:
                    raise ValueError("duration需为0.03–1.0秒")
                payload.update(buttons=args.buttons,hold_ms=round(args.duration*1000),
                               left_stick={"x":args.left_x,"y":args.left_y},
                               right_stick={"x":args.right_x,"y":args.right_y})
            result=request("/commands",payload)
            deadline=time.monotonic()+args.timeout
            while not args.no_wait and result["status"] in {"queued","running"} and time.monotonic()<deadline:
                time.sleep(.05)
                result=request("/commands/"+result["id"])
            if not args.no_wait and result["status"]!="succeeded":
                print(json.dumps(result,ensure_ascii=False,indent=2))
                return 1
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except (OSError,ValueError,KeyError,TypeError,HTTPError,URLError) as error:
        # Never print discovery metadata or authentication headers.
        print(json.dumps({"error":str(error)},ensure_ascii=False))
        return 1


if __name__=="__main__":
    raise SystemExit(main())
