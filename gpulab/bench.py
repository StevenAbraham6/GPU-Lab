"""One subprocess per test: contains a driver crash/OOM and releases all allocations."""
import json
import sys
import traceback
import os
import signal
import threading
import time
from .util import atomic_json, now

def main():
    # Kill this benchmark's process group if its durable supervisor disappears.
    parent=int(os.environ.get('GPU_LAB_PARENT_PID',os.getppid()))
    def stop_group(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        try:os.killpg(os.getpgrp(),signal.SIGTERM)
        finally:raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop_group)
    def watch_parent():
        while True:
            time.sleep(1)
            if os.getppid()!=parent:
                os.kill(os.getpid(),signal.SIGTERM);return
    threading.Thread(target=watch_parent,daemon=True).start()
    if os.getppid()!=parent:
        stop_group(signal.SIGTERM,None)
    spec=json.loads(open(sys.argv[1]).read());output=sys.argv[2]
    result=dict(test_id=spec["test_id"],started=now(),cases=[],status="error",backend=spec["gpu"].get("vendor"))
    try:
        test=spec["test_id"]
        if test=="matrix":
            from .torch_backend import matrix
            result["cases"]=matrix(spec["gpu"],spec["config"])
        elif test=="video":
            from .video import video
            result["cases"]=video(spec["gpu"],spec["config"],spec["directory"])
        else:
            from .opencl_backend import OpenCL
            backend=OpenCL(spec["gpu"])
            backend.gpu=spec["gpu"];backend.directory=spec["directory"]
            result["cases"]=getattr(backend,test)(spec["config"])
        statuses={c["status"] for c in result["cases"]}
        result["status"]="failed_validation" if "failed_validation" in statuses else "error" if "error" in statuses else "passed" if statuses=={"passed"} else "partial" if "passed" in statuses else "unsupported"
    except Exception as e:
        result.update(status="failed_validation" if isinstance(e,ArithmeticError) else "error",reason=str(e),traceback=traceback.format_exc())
    result["finished"]=now();atomic_json(output,result)

if __name__=="__main__":main()
