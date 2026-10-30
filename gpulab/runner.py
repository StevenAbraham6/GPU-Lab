import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from . import __version__
from .catalog import BY_ID, config, metric, select_tests
from .devices import backend_python, discover, select_gpu
from .preflight import check
from .store import ACTIVE, Store
from .telemetry import integrate, sample
from .util import atomic_json, fingerprint, now, process_marker
from .updates import launch_guard

DEMO_GPU=dict(id="demo:0",name="Demonstration GPU — synthetic data",vendor="demo",simulated=True,vram_bytes=24*1024**3)

def create_run(store, gpu_selector=None, names=None, cfg=None, metadata=None, simulated=False, parent=None, allow_partial=False):
    with launch_guard(store):
        return _create_run(store,gpu_selector,names,cfg,metadata,simulated,parent,allow_partial)

def _create_run(store, gpu_selector=None, names=None, cfg=None, metadata=None, simulated=False, parent=None, allow_partial=False):
    cfg_values=dict(cfg or {});profile=cfg_values.pop('profile','quick')
    cfg=config(profile,**cfg_values);tests=select_tests(names)
    inventory=discover() if not simulated else dict(host={"hostname":"demonstration","os":"simulated","cpu":"simulated"},gpus=[DEMO_GPU])
    gpu=DEMO_GPU if simulated else select_gpu(inventory,gpu_selector)
    snapshot=dict(host=inventory["host"],gpu=gpu,app_version=__version__,captured=now(),
                  environment={key:os.environ[key] for key in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL','RUSTICL_ENABLE') if key in os.environ})
    meta=dict(name=f"{gpu['name']} · {cfg['profile']}",description="",tags=[],notes="",hardware=dict(host=inventory["host"],gpu=gpu))
    meta.update(metadata or {})
    cfg={**cfg,"allow_partial":bool(allow_partial)}
    rid=store.create(meta,snapshot,cfg,tests,gpu,simulated,parent)
    launch(store,rid)
    return rid

def launch(store,rid):
    folder=store.directory(rid)
    with (folder/"worker.log").open("ab") as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('backend_entry.py')),"worker",str(store.root),rid],stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    store.update(rid,pid=p.pid,marker=process_marker(p.pid))
    threading.Thread(target=p.wait,daemon=True).start()

def cancel(store,rid):
    run=store.get(rid,samples=False)
    if run["source"]!="local" or run["status"] not in ACTIVE:
        raise ValueError("Only active local runs can be cancelled")
    store.update(rid,cancel=1,status="cancelling")

class Collector:
    def __init__(self,store,rid,gpu,cfg):
        self.store,self.rid,self.gpu,self.cfg=store,rid,gpu,cfg
        self.stop=threading.Event();self.hot=threading.Event();self.test_id=None
        self.lock=threading.Lock();self.thread=threading.Thread(target=self.loop,daemon=True)
    def take(self):
        with self.lock:
            begin=time.time();reading=sample(self.gpu);end=time.time()
            self.store.sample(self.rid,self.test_id,(begin+end)/2,reading)
            if reading.get("temperature_c") is not None and reading["temperature_c"]>=self.cfg["max_temp"]:self.hot.set()
            return reading
    def loop(self):
        while not self.stop.is_set():
            try:self.take()
            except Exception as e:
                print("Telemetry collection error:",e,flush=True)
            self.stop.wait(self.cfg["sample_interval"])
    def start(self):self.thread.start()
    def close(self):self.stop.set();self.thread.join(timeout=10)

def efficiency(store,rid,result,cfg):
    for case in result.get("cases",[]):
        intervals=case.get("intervals",[])
        energies=[];total_work=0;total_time=0
        details=[]
        for interval in intervals:
            start,end=interval["start"],interval["end"]
            samples=store.samples(rid,start-10,end+10)
            info=integrate(samples,start,end,cfg["sample_interval"])
            # Short phases cannot resolve one-second averaged hardware sensors reliably.
            if end-start<max(2*cfg["sample_interval"],2):
                info.update(energy_j=None,average_power_w=None,coverage="phase too short for sensor sampling")
            details.append(info)
            if info["energy_j"] is not None and info["energy_j"]>0:
                energies.append(info["energy_j"]);total_work+=interval["work"];total_time+=end-start
        case["power"]=dict(intervals=details,scope="GPU device telemetry only; never whole-server power")
        if intervals and len(energies)==len(intervals):
            energy=sum(energies);unit=intervals[0]["work_unit"]
            case["metrics"].extend([
                metric("average GPU power",energy/total_time,"W","context"),
                metric("energy for measured work",energy,"J","context"),
                metric("work per joule",total_work/energy,unit+"/J","higher"),
                metric("energy per work unit",energy/total_work,"J/"+unit,"lower")])
            case["power"].update(energy_j=energy,average_power_w=energy/total_time,power_scope=details[0]["power_scope"],covered=True)
        else:
            case["power"].update(covered=False,reason="No fully covered energy window; see per-interval coverage. Performance remains valid.")
    return result

def demo_result(test,cfg):
    """Explicit fixture for product evaluation. Never enters real hardware adapters."""
    cases=[]
    examples={"arithmetic":[("FP32",12500,"GFLOP/s","higher"),("INT32",8100,"GOP/s","higher")],
              "matrix":[("FP32 · square",24.8,"TFLOP/s","higher"),("FP16 · square",102.4,"TFLOP/s","higher"),("INT8 · square",190,"TOP/s","higher")],
              "memory":[("copy · 64 MiB",320,"GB/s","higher"),("dependent access",180,"ns/access","lower")],
              "transfer":[("upload · 64 MiB",24.5,"GB/s","higher"),("download · 64 MiB",23.9,"GB/s","higher")],
              "latency":[("host completion",32.4,"µs","lower")],
              "video":[("H264 · 1080p encode",440,"frames/s","higher"),("HEVC · 4K encode",110,"frames/s","higher")],
              "concurrency":[("2 concurrent jobs",16800,"GFLOP/s","higher")],
              "stability":[("sustained FP32",12300,"GFLOP/s","higher")],
              "integrity":[("VRAM mismatches",0,"errors","lower")]}
    for name,value,unit,direction in examples[test]:
        metrics=[metric("throughput" if direction=="higher" else "latency" if unit in ("µs","ns/access") else "mismatches",value,unit,direction)]
        if direction=="higher":metrics += [metric("average GPU power",100,"W","context"),metric("work per joule",value/100,unit.replace("/s","/J"),"higher")]
        if test=="video":metrics += [metric("VMAF",94.2,"score"),metric("output bitrate",8,"Mbit/s","context")]
        cases.append(dict(name=name,status="passed",settings={"fixture":"synthetic-v1","profile":cfg["profile"]},metrics=metrics,intervals=[],
                          power=dict(covered=True,power_scope="SIMULATED",average_power_w=100),note="SIMULATED: not a GPU measurement"))
    return dict(test_id=test,status="passed",started=now(),finished=now(),cases=cases,simulated=True)

def continuation(store,rid,unfinished_only=True):
    old=store.get(rid,samples=False)
    if old["status"] in ACTIVE:raise ValueError("Wait for or cancel the active run first")
    successful={r["test_id"] for r in old["results"] if r["status"]=="passed"}
    names=[t for t in old["tests"] if not unfinished_only or t not in successful]
    if not names:raise ValueError("No unfinished tests; use reproduce to repeat all tests")
    meta={**old["metadata"],"name":old["metadata"]["name"]+(" · continuation" if unfinished_only else " · repeat")}
    return create_run(store,old["gpu"]["id"],names,old["config"],meta,old["simulated"],rid,old["config"].get("allow_partial",False))

def work(store,rid):
    run=store.get(rid,samples=False);gpu=run["gpu"];cfg=run["config"]
    store.update(rid,pid=os.getpid(),marker=process_marker(os.getpid()),heartbeat=now())
    done=threading.Event()
    def heartbeat():
        while not done.wait(3):store.update(rid,heartbeat=now())
    threading.Thread(target=heartbeat,daemon=True).start()
    lockfile=(store.root/"locks"/(fingerprint(gpu["id"])+".lock")).open("a+")
    collector=None;child=None
    try:
        while True:
            if store.get(rid,samples=False)["cancel"]:
                store.update(rid,status="cancelled");return
            try:fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(.5)
        store.update(rid,status="preflight")
        report=check(gpu,run["tests"],cfg,store.root)
        if store.get(rid,samples=False)["cancel"]:
            store.update(rid,status="cancelled");return
        atomic_json(store.directory(rid)/"preflight.json",report)
        store.audit(rid,dict(action="preflight",report=report))
        if report["state"]=="blocked" and not cfg.get("allow_partial"):
            store.update(rid,status="blocked",error="Preflight blocked. Review findings; resolve dependencies or explicitly allow partial runs.");return
        global_blocks=[c for c in report["checks"] if c["state"]=="blocked" and not c["tests"]]
        if global_blocks:
            store.update(rid,status="blocked",error="; ".join(c["detail"] for c in global_blocks));return
        blocked={t for c in report["checks"] if c["state"]=="blocked" for t in c["tests"]}
        store.update(rid,status="running",started_at=now())
        if not run["simulated"]:
            collector=Collector(store,rid,gpu,cfg);collector.start()
        for test in run["tests"]:
            store.update(rid,current_test=test,current_test_started_at=now())
            if store.get(rid,samples=False)["cancel"]:
                store.update(rid,status="cancelled",current_test_started_at=None);return
            if test in blocked:
                reasons=[c["detail"] for c in report["checks"] if test in c["tests"] and c["state"]=="blocked"]
                store.result(rid,dict(test_id=test,status="missing_dependency",reason="; ".join(reasons),cases=[]))
                store.update(rid,current_test_started_at=None)
                continue
            if run["simulated"]:
                for watt in (82,108,99):
                    store.sample(rid,test,time.time(),dict(power_w=watt,temperature_c=60,clock_mhz=1500,memory_used_mib=1024,
                                 utilization_pct=95,power_scope='SIMULATED',source='synthetic demonstration fixture'))
                    time.sleep(.07)
                store.result(rid,demo_result(test,cfg))
                store.update(rid,current_test_started_at=None)
                continue
            collector.test_id=test;collector.take()
            folder=store.directory(rid);spec=folder/(test+"-spec.json");out=folder/(test+"-raw.json")
            atomic_json(spec,dict(test_id=test,gpu=gpu,config=cfg,directory=str(folder/"video")))
            python=backend_python(gpu,store.root) if test=="matrix" else sys.executable
            with (folder/(test+".log")).open("ab") as log:
                env=os.environ.copy();env['GPU_LAB_PARENT_PID']=str(os.getpid())
                child=subprocess.Popen([python,str(Path(__file__).with_name('backend_entry.py')),"bench",str(spec),str(out)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
            started=time.monotonic();aborted=None
            while child.poll() is None:
                if store.get(rid,samples=False)["cancel"]:aborted="cancelled"
                elif collector.hot.is_set():aborted="temperature ceiling reached"
                elif time.monotonic()-started>cfg["timeout"]:aborted="test timeout"
                if aborted:
                    terminate(child);break
                time.sleep(.25)
            collector.take()
            if aborted:
                store.result(rid,dict(test_id=test,status="cancelled" if aborted=="cancelled" else "error",reason=aborted,cases=[]))
                store.update(rid,status="cancelled" if aborted=="cancelled" else "error",error=aborted,current_test_started_at=None);return
            if child.returncode!=0 or not out.exists():
                result=dict(test_id=test,status="error",reason=f"Benchmark process exited {child.returncode}; inspect {test}.log",cases=[])
            else:
                result=json.loads(out.read_text());result=efficiency(store,rid,result,cfg)
            store.result(rid,result)
            store.update(rid,current_test_started_at=None)
            child=None
        results=store.get(rid,samples=False)["results"]
        states={r["status"] for r in results}
        final="completed" if states=={"passed"} else "failed_validation" if "failed_validation" in states else "partial" if "passed" in states or "partial" in states else "error"
        store.update(rid,status=final,current_test=None,current_test_started_at=None)
    except BaseException as e:
        if child and child.poll() is None:terminate(child)
        store.update(rid,status="error",error=str(e))
        raise
    finally:
        if collector:collector.close()
        done.set();fcntl.flock(lockfile,fcntl.LOCK_UN);lockfile.close()
        atomic_json(store.directory(rid)/"run.json",store.export(rid))

def terminate(child):
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait()
