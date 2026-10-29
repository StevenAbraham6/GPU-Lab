import argparse
import json
import os
import sys
import time
from pathlib import Path
from .catalog import TESTS,config,select_tests
from .devices import discover,select_gpu
from .preflight import check
from .runner import DEMO_GPU,cancel,continuation,create_run
from .store import ACTIVE,Store
from .timing import format_duration,run_timing

def output(value):print(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))

def parser():
    p=argparse.ArgumentParser(prog="gpu-lab",description="GPU performance, power and video benchmarking. No CPU fallback.")
    p.add_argument("--data-dir",help="Persistent run storage (or GPU_LAB_HOME)")
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("devices",help="Discover physical GPU and OpenCL mappings")
    sub.add_parser("tests",help="List versioned workloads")
    for name in ("run","doctor"):
        q=sub.add_parser(name,help="Launch detached benchmarks" if name=="run" else "Run preflight checks without benchmarking")
        q.add_argument("tests",nargs="*",default=["all"]);q.add_argument("--gpu");q.add_argument("--profile",choices=["quick","standard","extended"],default="quick")
        q.add_argument("--config",help="JSON configuration file; overrides profile defaults")
        q.add_argument("--demo",action="store_true",help="Explicit synthetic UI demonstration; never hardware data")
        if name=="run":
            q.add_argument("--name");q.add_argument("--description",default="");q.add_argument("--tag",action="append",default=[])
            q.add_argument("--allow-partial",action="store_true",help="Explicitly skip blocked per-test dependencies; global blocks still stop")
            q.add_argument("--wait",action="store_true",help="Watch the detached run until completion")
    q=sub.add_parser("serve");q.add_argument("--host",default="127.0.0.1");q.add_argument("--port",type=int,default=8080);q.add_argument("--token-file")
    sub.add_parser("list")
    for name in ("show","wait","cancel","resume","reproduce"):
        q=sub.add_parser(name);q.add_argument("run_id")
    q=sub.add_parser("edit");q.add_argument("run_id");q.add_argument("--json",required=True,help="Metadata patch JSON file")
    q=sub.add_parser("export");q.add_argument("run_id");q.add_argument("--format",choices=["pdf","html","csv","json"],default="pdf");q.add_argument("--output",required=True)
    q=sub.add_parser("import");q.add_argument("file")
    q=sub.add_parser("compare");q.add_argument("baseline");q.add_argument("candidate")
    return p

def wait(store,rid):
    width=0
    try:
        while True:
            store.recover();run=store.get(rid,samples=False)
            timing=run_timing(run)
            current_elapsed=format_duration(timing["current_elapsed_seconds"]) if run["current_test"] else "-"
            run_elapsed=format_duration(timing["run_elapsed_seconds"])
            line=(f"{run['status']} | {run['current_test'] or '-'} | "
                  f"{len(run['results'])}/{len(run['tests'])} groups | "
                  f"current elapsed {current_elapsed} | total elapsed {run_elapsed}")
            print("\r"+line.ljust(width),end="",flush=True);width=max(width,len(line))
            if run["status"] not in ACTIVE:
                print(flush=True)
                if run.get("error"):print(run["error"],file=sys.stderr)
                return 0 if run["status"]=="completed" else 2
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDetached run continues. Use gpu-lab cancel RUN_ID to stop it.");return 0

def main():
    args=parser().parse_args();store=Store(args.data_dir)
    try:
        cmd=args.command
        if cmd=="devices":output(discover())
        elif cmd=="tests":output(TESTS)
        elif cmd in ("run","doctor"):
            overrides=json.loads(Path(args.config).read_text()) if args.config else {}
            overrides.pop("profile",None)
            cfg=config(args.profile,**overrides);tests=select_tests(args.tests)
            if cmd=="doctor":
                gpu=DEMO_GPU if args.demo else select_gpu(discover(),args.gpu)
                result=check(gpu,tests,cfg,store.root);output(result)
                return 2 if result["state"]=="blocked" else 0
            meta=dict(description=args.description,tags=args.tag)
            if args.name:meta["name"]=args.name
            rid=create_run(store,args.gpu,tests,cfg,meta,args.demo,allow_partial=args.allow_partial)
            print(rid,flush=True)
            if args.wait:return wait(store,rid)
        elif cmd=="serve":
            from .server import serve
            token=Path(args.token_file).read_text().strip() if args.token_file else os.environ.get("GPU_LAB_TOKEN")
            serve(store,args.host,args.port,token)
        elif cmd=="list":output(store.list())
        elif cmd=="show":store.recover();output(store.get(args.run_id,samples=False))
        elif cmd=="wait":return wait(store,args.run_id)
        elif cmd=="cancel":cancel(store,args.run_id);output({"cancel_requested":args.run_id})
        elif cmd in ("resume","reproduce"):print(continuation(store,args.run_id,cmd=="resume"))
        elif cmd=="edit":output(store.edit_metadata(args.run_id,json.loads(Path(args.json).read_text())))
        elif cmd=="export":
            from .reports import report_csv,report_html,report_pdf
            run=store.get(args.run_id)
            data=dict(pdf=report_pdf,html=report_html,csv=report_csv,json=lambda _:json.dumps(store.export(args.run_id),indent=2).encode())[args.format](run)
            path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);print(path.resolve())
        elif cmd=="import":print(store.import_run(json.loads(Path(args.file).read_text())))
        elif cmd=="compare":
            from .compare import comparison
            output(comparison(store.get(args.baseline,samples=False),store.get(args.candidate,samples=False)))
        return 0
    except (ValueError,KeyError,OSError) as e:
        print(f"GPU Lab: {e}",file=sys.stderr);return 2

if __name__=="__main__":sys.exit(main())
