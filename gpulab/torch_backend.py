import math
import numpy as np
from .catalog import metric
from .measurement import timed_loop
from .probe import torch_device

def matrix(gpu, cfg):
    torch, runtime, dev = torch_device(gpu)
    runtime.set_device(dev)
    policy = {}
    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if matmul is not None:
        for setting in ("allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"):
            if hasattr(matmul, setting):
                setattr(matmul, setting, False); policy[setting] = False
    cases = []
    for precision in cfg["precisions"]:
        for size in cfg["matrix_sizes"]:
            for shape in ("square", "rectangular", "batched"):
                n = size if shape != "batched" else min(size, 512)
                batch = 8 if shape == "batched" else 1
                m,k,p = (n,n,n) if shape != "rectangular" else (n,n//2,n//4)
                settings = dict(precision=precision, m=m, k=k, n=p, batch=batch, shape=shape, seed=2026,
                                math_policy=policy, library="PyTorch vendor GEMM", torch_version=torch.__version__,
                                input_distribution="seeded uniform [-0.125,0.125]" if precision!="int8" else "seeded integers [-3,3]",
                                accumulation="INT32" if precision=="int8" else "FP32 requested by disabled reduced-precision reductions; dispatch not independently profiled" if precision in ("fp16","bf16") else "FP32")
                name=f"{precision.upper()} · {shape} · {batch}×{m}×{k}×{p}"
                # Batched size is capped and must not run identical duplicate cases.
                if any(c["name"] == name for c in cases):
                    continue
                if precision=="int8" and shape=="batched":
                    cases.append(dict(name=name,status="unsupported",reason="No portable accelerated batched INT8 adapter in v0.1",settings=settings,metrics=[],intervals=[]));continue
                try:
                    dtype=dict(fp32=torch.float32,fp16=torch.float16,bf16=torch.bfloat16,int8=torch.int8)[precision]
                    element=1 if precision=="int8" else 4 if precision=="fp32" else 2
                    estimated=batch*(m*k+k*p+m*p)*max(element,4)
                    try:free,_=runtime.mem_get_info(dev)
                    except (AttributeError, TypeError):free=gpu.get("vram_bytes",0)
                    if free and estimated>free*.6:
                        raise NotImplementedError("Estimated allocation exceeds 60% of currently free VRAM")
                    rng=np.random.default_rng(2026)
                    ashape=(batch,m,k) if batch>1 else (m,k);bshape=(batch,k,p) if batch>1 else (k,p)
                    if precision=="int8":
                        ah=rng.integers(-3,4,ashape,dtype=np.int8);bh=rng.integers(-3,4,bshape,dtype=np.int8)
                    else:
                        ah=(rng.random(ashape,dtype=np.float32)-.5)*.25;bh=(rng.random(bshape,dtype=np.float32)-.5)*.25
                    a=torch.from_numpy(ah).to(device=dev,dtype=dtype);b=torch.from_numpy(bh).to(device=dev,dtype=dtype)
                    out=torch.empty((batch,m,p) if batch>1 else (m,p),device=dev,dtype=torch.int32 if precision=="int8" else dtype)
                    if precision=="int8" and not hasattr(torch,"_int_mm"):
                        raise NotImplementedError("Private torch._int_mm API not available in this PyTorch build")
                    def operation():
                        if precision=="int8":
                            return torch._int_mm(a,b,out=out)
                        if batch>1:return torch.bmm(a,b,out=out)
                        return torch.mm(a,b,out=out)
                    try:
                        operation();runtime.synchronize()
                    except (RuntimeError,TypeError,NotImplementedError) as e:
                        raise NotImplementedError(f"Requested GPU GEMM path unavailable: {e}") from e
                    # Validate using the actual rounded inputs; CPU float64/int64 reference.
                    rows=np.linspace(0,m-1,min(m,8),dtype=int);cols=np.linspace(0,p-1,min(p,8),dtype=int)
                    aslice=a[0] if batch>1 else a;bslice=b[0] if batch>1 else b;oslice=out[0] if batch>1 else out
                    ar=aslice[rows.tolist(),:].to(torch.float64).cpu().numpy()
                    br=bslice[:,cols.tolist()].to(torch.float64).cpu().numpy()
                    reference=ar@br;actual=oslice[rows.tolist(),:][:,cols.tolist()].to(torch.float64).cpu().numpy()
                    error=float(np.max(np.abs(reference-actual)))
                    scale=max(float(np.max(np.abs(reference))),1e-6)
                    tolerance=dict(fp32=5e-5,fp16=.003,bf16=.025,int8=0)[precision]
                    if not np.isfinite(actual).all() or error>max(1e-6 if precision!="int8" else 0,scale*tolerance):
                        raise ArithmeticError(f"GEMM failed reference check: max error {error}, reference scale {scale}")
                    start_event=runtime.Event(enable_timing=True);end_event=runtime.Event(enable_timing=True)
                    def fn():
                        start_event.record();operation();end_event.record();end_event.synchronize()
                        return start_event.elapsed_time(end_event)/1000
                    work=2*batch*m*k*p/1e12
                    case=timed_loop(fn,cfg,work,"TOP/s" if precision=="int8" else "TFLOP/s",name,settings)
                    case["metrics"].append(metric("maximum reference error",error,"absolute","lower"))
                    case["validation"]=dict(checked_outputs=len(rows)*len(cols),sampled_batch=0,reference="CPU float64 from rounded inputs",relative_to_max_tolerance=tolerance)
                    cases.append(case)
                except NotImplementedError as e:
                    cases.append(dict(name=name,status="unsupported",reason=str(e)[:2000],settings=settings,metrics=[],intervals=[]))
                except ArithmeticError as e:
                    cases.append(dict(name=name,status="failed_validation",reason=str(e),settings=settings,metrics=[],intervals=[]))
                finally:
                    # No GPU arrays survive into the next allocation-budget check.
                    if 'aslice' in locals():del aslice,bslice,oslice
                    if 'a' in locals():del a
                    if 'b' in locals():del b
                    if 'out' in locals():del out
                    if 'ah' in locals():del ah
                    if 'bh' in locals():del bh
                    runtime.empty_cache()
    return cases
