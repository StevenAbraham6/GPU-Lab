import concurrent.futures
import math
import statistics
import time
from pathlib import Path
from .catalog import metric
from .measurement import event_seconds, stats, timed_loop

class OpenCL:
    def __init__(self, gpu):
        import numpy as np
        import pyopencl as cl
        self.np, self.cl = np, cl
        item = gpu.get("opencl")
        if not item:
            raise RuntimeError("No OpenCL mapping for selected GPU")
        dev = cl.get_platforms()[item["platform_index"]].get_devices()[item["device_index"]]
        if not dev.type & cl.device_type.GPU or dev.name.strip() != item["name"] or dev.driver_version != item["driver"]:
            raise RuntimeError("OpenCL device mapping changed; rediscover devices")
        # PCI mapping is captured in the inventory; reject any changed mapping at runtime.
        from .probe import opencl_probe
        current = next((d for d in opencl_probe()["devices"] if d["platform_index"] == item["platform_index"] and d["device_index"] == item["device_index"]), None)
        if current is None or current.get("bus") != item.get("bus"):
            raise RuntimeError("Physical OpenCL device mapping changed")
        self.dev = dev
        self.ctx = cl.Context([dev])
        self.queue = self.new_queue()
        self.program = cl.Program(self.ctx, (Path(__file__).parent / "kernels/core.cl").read_text()).build(options=[])
        self.k = {name: cl.Kernel(self.program, name) for name in ("arithmetic_fp", "arithmetic_int", "read_memory", "write_memory", "copy_memory", "chase", "tiny", "fill_pattern", "verify_pattern")}

    def new_queue(self):
        return self.cl.CommandQueue(self.ctx, properties=self.cl.command_queue_properties.PROFILING_ENABLE)

    def buffer(self, size):
        return self.cl.Buffer(self.ctx, self.cl.mem_flags.READ_WRITE, int(size))

    def execution_check(self):
        out = self.buffer(4)
        self.k["tiny"](self.queue, (1,), None, out).wait()
        a = self.np.zeros(1, dtype=self.np.uint32)
        self.cl.enqueue_copy(self.queue, a, out).wait()
        if a[0] != 42:
            raise RuntimeError("Tiny kernel correctness failed")
        return dict(device=self.dev.name, correct=True)

    def budget(self, cfg, buffers=3):
        return min(cfg["buffer_mib"]*1024**2, self.dev.max_mem_alloc_size//2,
                   int(self.dev.global_mem_size*cfg["memory_fraction"] / buffers))

    def arithmetic_job(self, kind="fp32"):
        np, cl = self.np, self.cl
        n, iterations = 262144, 1024
        out = self.buffer(n*16)
        kernel = self.k["arithmetic_fp" if kind == "fp32" else "arithmetic_int"]
        def fn(queue=None):
            return event_seconds(kernel(queue or self.queue, (n,), None, out, np.uint32(iterations)))
        fn()
        result = np.zeros((64,4), dtype=np.float32 if kind == "fp32" else np.uint32)
        cl.enqueue_copy(self.queue, result, out).wait()
        ids = np.arange(64, dtype=np.uint32)
        if kind == "fp32":
            ref = np.stack([.01*(ids%17), .02*(ids%13), .03*(ids%11), .04*(ids%7)], axis=1).astype(np.float32)
            # Simulate explicit FMA with float64 intermediate then round to float32.
            for _ in range(iterations):
                ref = (ref.astype(np.float64)*float(np.float32(1.000001)) + float(np.float32(.000001))).astype(np.float32)
            correct = np.allclose(result, ref, rtol=2e-5, atol=2e-6)
        else:
            ref = np.stack([ids, ids+1, ids+2, ids+3], axis=1)
            for _ in range(iterations):
                ref = ref*np.uint32(1664525)+np.uint32(1013904223)
            correct = np.array_equal(ref, result)
        if not correct:
            raise ArithmeticError(f"{kind} arithmetic validation failed")
        return fn, n*iterations*8/1e9, out, dict(items=n, independent_accumulators=4, iterations=iterations, precision=kind,
                                               arithmetic="explicit FMA" if kind=="fp32" else "uint32 wrapping multiply/add")

    def arithmetic(self, cfg):
        cases = []
        for kind in ("fp32", "int32"):
            fn, work, buf, settings = self.arithmetic_job(kind)
            cases.append(timed_loop(fn, cfg, work, "GFLOP/s" if kind=="fp32" else "GOP/s", kind.upper(), settings))
        return cases

    def memory(self, cfg):
        np, cl = self.np, self.cl
        maximum = 2**int(math.log2(self.budget(cfg)))
        sizes = sorted(set([min(maximum, n) for n in (256*1024, 4*1024**2, 32*1024**2, maximum)]))
        cases = []
        for size in sizes:
            n = size//4; gs=min(65536,n)
            a,b,out=self.buffer(size),self.buffer(size),self.buffer(gs*4)
            cl.enqueue_fill_buffer(self.queue, a, np.float32(1), 0, size).wait()
            for op in ("read", "write", "copy", "strided-copy"):
                def fn(op=op):
                    if op=="read": ev=self.k["read_memory"](self.queue,(gs,),None,a,out,np.uint32(n))
                    elif op=="write": ev=self.k["write_memory"](self.queue,(gs,),None,b,np.uint32(n))
                    else: ev=self.k["copy_memory"](self.queue,(gs,),None,a,b,np.uint32(n),np.uint32(1 if op=="copy" else 33))
                    return event_seconds(ev)
                fn()
                host=np.empty(gs if op=="read" else min(n,4096),dtype=np.float32)
                cl.enqueue_copy(self.queue,host,out if op=="read" else b).wait()
                expected=np.full(gs,n//gs,dtype=np.float32) if op=="read" else (np.arange(len(host))%1024).astype(np.float32) if op=="write" else np.ones(len(host),np.float32)
                if not np.array_equal(host,expected): raise ArithmeticError(f"Memory {op} correctness failed")
                work=size*(2 if "copy" in op else 1)/1e9
                cases.append(timed_loop(fn,cfg,work,"GB/s",f"{op} · {size//1024} KiB",dict(operation=op,buffer_bytes=size,byte_count="read+write" if "copy" in op else "one stream")))
            # One cycle through a random permutation: dependent accesses cannot be prefetched from a fixed stride.
            count = min(n, 16*1024**2)
            if any(c.get("settings",{}).get("operation")=="pointer chase" and c["settings"]["buffer_bytes"]==count*4 for c in cases):continue
            permutation=np.random.default_rng(17).permutation(count).astype(np.uint32)
            links=np.empty(count,np.uint32); links[permutation]=np.roll(permutation,-1)
            cl.enqueue_copy(self.queue,a,links).wait()
            steps=min(count,262144)
            seconds=event_seconds(self.k["chase"](self.queue,(1,),None,a,out,np.uint32(steps)))
            got=np.zeros(1,np.uint32);cl.enqueue_copy(self.queue,got,out).wait()
            expected_pos=0
            for _ in range(steps):expected_pos=int(links[expected_pos])
            if got[0]!=expected_pos:raise ArithmeticError("Pointer chase validation failed")
            timings=[]
            for _ in range(max(5,cfg["repeats"])):
                timings.append(event_seconds(self.k["chase"](self.queue,(1,),None,a,out,np.uint32(steps)))*1e9/steps)
            cases.append(dict(name=f"dependent access · {count*4//1024} KiB",status="passed",settings=dict(buffer_bytes=count*4,steps=steps,operation="pointer chase",seed=17),
                              metrics=[metric("dependent access latency",statistics.median(timings),"ns/access","lower",statistics=stats(timings))],intervals=[]))
        return cases

    def transfer(self,cfg):
        np,cl=self.np,self.cl
        maximum=2**int(math.log2(self.budget(cfg)))
        cases=[]
        for size in sorted(set(min(maximum,s) for s in (4096,1024**2,16*1024**2,maximum))):
            hostbuf=cl.Buffer(self.ctx,cl.mem_flags.READ_WRITE|cl.mem_flags.ALLOC_HOST_PTR,size)
            host,_=cl.enqueue_map_buffer(self.queue,hostbuf,cl.map_flags.READ|cl.map_flags.WRITE,0,(size//4,),np.uint32,is_blocking=True)
            host[:]=np.arange(size//4,dtype=np.uint32)
            target=self.buffer(size);other=self.buffer(size)
            cl.enqueue_copy(self.queue,target,host).wait();cl.enqueue_copy(self.queue,other,host).wait()
            q2=self.new_queue()
            for direction in ("upload","download","duplex"):
                def fn():
                    if direction=="upload":return event_seconds(cl.enqueue_copy(self.queue,target,host,is_blocking=False))
                    if direction=="download":return event_seconds(cl.enqueue_copy(self.queue,host,target,is_blocking=False))
                    # Distinct host staging buffers avoid a host read/write race.
                    up=cl.enqueue_copy(self.queue,target,host,is_blocking=False)
                    down=cl.enqueue_copy(q2,download_host,other,is_blocking=False)
                    up.wait();down.wait()
                    return (max(up.profile.end,down.profile.end)-min(up.profile.start,down.profile.start))*1e-9
                downbuf=cl.Buffer(self.ctx,cl.mem_flags.READ_WRITE|cl.mem_flags.ALLOC_HOST_PTR,size)
                download_host,_=cl.enqueue_map_buffer(q2,downbuf,cl.map_flags.READ|cl.map_flags.WRITE,0,(size//4,),np.uint32,is_blocking=True)
                case=timed_loop(fn,cfg,size*(2 if direction=="duplex" else 1)/1e9,"GB/s",f"{direction} · {size//1024} KiB",dict(direction=direction,bytes=size,host_memory="OpenCL ALLOC_HOST_PTR mapped",byte_count="both directions" if direction=="duplex" else "one direction"))
                verify=np.empty_like(host);cl.enqueue_copy(self.queue,verify,target).wait()
                if not np.array_equal(verify,np.arange(size//4,dtype=np.uint32)):raise ArithmeticError("Transfer correctness failed")
                if direction=="duplex" and not np.array_equal(download_host,verify):raise ArithmeticError("Duplex download correctness failed")
                latency=[]
                for _ in range(20):
                    begin=time.perf_counter();fn();latency.append((time.perf_counter()-begin)*1e6)
                case["metrics"].append(metric("completion latency",statistics.median(latency),"µs","lower",statistics=stats(latency)))
                cases.append(case)
                download_host.base.release(q2).wait()
            host.base.release(self.queue).wait()
        return cases

    def latency(self,cfg):
        out=self.buffer(4);kernel=self.k["tiny"]
        for _ in range(100):kernel(self.queue,(1,),None,out).wait()
        submit,complete,device=[],[],[]
        for _ in range(1000):
            begin=time.perf_counter_ns();ev=kernel(self.queue,(1,),None,out);sent=time.perf_counter_ns();ev.wait();end=time.perf_counter_ns()
            submit.append((sent-begin)/1000);complete.append((end-begin)/1000);device.append((ev.profile.end-ev.profile.start)/1000)
        self.execution_check()
        return [dict(name="one-work-item round trip",status="passed",settings=dict(samples=1000,warmup_dispatches=100),intervals=[],
                     metrics=[metric(name,statistics.median(values),"µs","lower",statistics=stats(values)) for name,values in (("host submission",submit),("host completion",complete),("device execution",device))])]

    def concurrency(self,cfg):
        cases=[];baseline=None
        for jobs in (1,2,4):
            # Separate kernel objects are required: setting arguments is not thread-safe.
            jobspec=[]
            for i in range(jobs):
                fn,work,buf,settings=self.arithmetic_job()
                kernel=self.cl.Kernel(self.program,"arithmetic_fp");queue=self.new_queue()
                def invoke(kernel=kernel,queue=queue,buf=buf):
                    return event_seconds(kernel(queue,(settings["items"],),None,buf,self.np.uint32(settings["iterations"])))
                jobspec.append((invoke,buf))
            rates=[];jobtimes=[];intervals=[]
            with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
                def roundtrip():
                    start=time.perf_counter();futures=[pool.submit(x[0]) for x in jobspec]
                    times=[f.result() for f in futures];return time.perf_counter()-start,times
                for _ in range(3):roundtrip()
                for repeat in range(cfg["repeats"]):
                    start=time.time();begin=time.perf_counter();rounds=0
                    while rounds==0 or time.perf_counter()-begin<cfg["duration"]:
                        elapsed,times=roundtrip();jobtimes.extend(times);rounds+=1
                    end=time.time();rate=work*jobs*rounds/(end-start);rates.append(rate)
                    intervals.append(dict(start=start,end=end,work=work*jobs*rounds,work_unit="GFLOP"))
            result=statistics.median(rates)
            if baseline is None:baseline=result
            cases.append(dict(name=f"{jobs} concurrent jobs",status="passed",settings=dict(jobs=jobs,precision="fp32",items=settings["items"],iterations=settings["iterations"]),intervals=intervals,
                              metrics=[metric("aggregate throughput",result,"GFLOP/s",statistics=stats(rates)),metric("per-job slowdown",baseline/(result/jobs),"×","lower"),metric("job device duration",statistics.median(jobtimes)*1e3,"ms","lower",statistics=stats([v*1e3 for v in jobtimes]))]))
        if cfg.get("concurrency_media"):
            from .video import concurrent_video
            cases.extend(concurrent_video(self.gpu,cfg,self.directory,compute_factory=self.arithmetic_job))
        return cases

    def stability(self,cfg):
        cases=[];phases=cfg["soak_workloads"];phase_cfg={**cfg,"soak_seconds":cfg["soak_seconds"]/len(phases)}
        for phase in phases:
            if phase=="compute":cases.extend(self.compute_soak(phase_cfg))
            elif phase=="memory":cases.extend(self.memory_soak(phase_cfg))
            else:
                from .video import concurrent_video
                cases.extend(concurrent_video(self.gpu,phase_cfg,self.directory,soak=True))
        return cases

    def compute_soak(self,cfg):
        fn,work,buf,settings=self.arithmetic_job()
        reference=self.np.empty(64*4,self.np.float32);self.cl.enqueue_copy(self.queue,reference,buf).wait()
        begin=time.time();series=[];intervals=[]
        while time.time()-begin<cfg["soak_seconds"]:
            start=time.time();count=0
            while count==0 or time.time()-start<min(5,cfg["duration"]):fn();count+=1
            end=time.time();series.append(dict(elapsed=end-begin,throughput=work*count/(end-start)))
            intervals.append(dict(start=start,end=end,work=work*count,work_unit="GFLOP"))
            # Compare with the validated pre-soak reference, not a potentially faulty fresh result.
            actual=self.np.empty(64*4,self.np.float32);self.cl.enqueue_copy(self.queue,actual,buf).wait()
            if not self.np.array_equal(actual,reference):raise ArithmeticError("Sustained workload output changed")
        q=max(1,len(series)//4);early=statistics.mean(x["throughput"] for x in series[:q]);late=statistics.mean(x["throughput"] for x in series[-q:])
        return [dict(name="sustained FP32",status="passed",settings={**settings,"soak_seconds":cfg["soak_seconds"]},intervals=intervals,series=series,
                     metrics=[metric("sustained throughput",statistics.mean(x["throughput"] for x in series),"GFLOP/s"),metric("late / early throughput",late/early*100,"%","context")])]

    def memory_soak(self,cfg):
        np,cl=self.np,self.cl;size=2**int(math.log2(self.budget(cfg)));n=size//4;gs=min(n,65536)
        a,b=self.buffer(size),self.buffer(size)
        cl.enqueue_fill_buffer(self.queue,a,np.float32(1.0),0,size).wait()
        begin=time.time();series=[];intervals=[]
        while time.time()-begin<cfg['soak_seconds']:
            start=time.time();count=0
            while count==0 or time.time()-start<min(5,cfg['duration']):
                event_seconds(self.k['copy_memory'](self.queue,(gs,),None,a,b,np.uint32(n),np.uint32(1)));count+=1
            end=time.time();work=size*2*count/1e9
            intervals.append(dict(start=start,end=end,work=work,work_unit='GB'));series.append(dict(elapsed=end-begin,throughput=work/(end-start)))
            spot=np.empty(min(n,4096),np.float32);cl.enqueue_copy(self.queue,spot,b).wait()
            if not np.array_equal(spot,np.ones_like(spot)):raise ArithmeticError('Memory soak validation failed')
        q=max(1,len(series)//4);early=statistics.mean(x['throughput'] for x in series[:q]);late=statistics.mean(x['throughput'] for x in series[-q:])
        return [dict(name='sustained VRAM copy',status='passed',settings=dict(buffer_bytes=size,soak_seconds=cfg['soak_seconds'],byte_count='read+write'),intervals=intervals,series=series,
                     metrics=[metric('sustained bandwidth',statistics.mean(x['throughput'] for x in series),'GB/s'),metric('late / early throughput',late/early*100,'%','context')])]

    def integrity(self,cfg):
        np,cl=self.np,self.cl
        size=int(min(self.dev.max_mem_alloc_size*.8,self.dev.global_mem_size*cfg["memory_fraction"],cfg["buffer_mib"]*1024**2))//4*4
        buf=self.buffer(size);n=size//4;gs=min(n,65536);counts=self.buffer(gs*4);host=np.empty(gs,np.uint32)
        start=time.time();errors=0
        def pattern(i,p):
            if p==0:return 0
            if p==1:return 0xffffffff
            if p==2:return 0xaaaaaaaa if i&1 else 0x55555555
            v=(i ^ (p*0x9e3779b9))&0xffffffff;v^=v>>16;v=(v*0x85ebca6b)&0xffffffff;v^=v>>13;return v
        for iteration in range(cfg["integrity_passes"]):
            for p in (0,1,2,3+iteration):
                self.k["fill_pattern"](self.queue,(gs,),None,buf,np.uint32(n),np.uint32(p)).wait()
                self.k["verify_pattern"](self.queue,(gs,),None,buf,counts,np.uint32(n),np.uint32(p)).wait()
                cl.enqueue_copy(self.queue,host,counts).wait();errors+=int(host.astype(np.uint64).sum())
                spot=np.empty(min(n,1024),np.uint32);cl.enqueue_copy(self.queue,spot,buf).wait()
                if not np.array_equal(spot,np.array([pattern(i,p) for i in range(len(spot))],np.uint32)):raise ArithmeticError("Host memory spot check failed")
        # Deliberately inject one error and prove the GPU verifier detects exactly one.
        cl.enqueue_fill_buffer(self.queue,buf,np.uint32(0),0,size).wait()
        bad=np.array([1],np.uint32);cl.enqueue_copy(self.queue,buf,bad).wait()
        self.k["verify_pattern"](self.queue,(gs,),None,buf,counts,np.uint32(n),np.uint32(0)).wait()
        cl.enqueue_copy(self.queue,host,counts).wait()
        if int(host.astype(np.uint64).sum())!=1:raise ArithmeticError("Verifier failed deliberate single-error self-test")
        return [dict(name="allocated VRAM pattern check",status="passed" if errors==0 else "failed_validation",settings=dict(bytes=size,passes=cfg["integrity_passes"],patterns_per_pass=4,verifier_self_test="one injected error detected"),intervals=[],
                     metrics=[metric("mismatches",errors,"errors","lower"),metric("tested allocation",size/1024**2,"MiB","context"),metric("elapsed",time.time()-start,"s","context")])]
