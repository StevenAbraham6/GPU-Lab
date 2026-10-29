"""Versioned workload and metric definitions shared by CLI, UI and reports."""
from copy import deepcopy

TESTS = [
    dict(id="arithmetic", group="compute", title="General arithmetic", backend="opencl",
         summary="Register-heavy FP32 FMA and INT32 arithmetic on general compute units.",
         method="Independent accumulators per work item; fixed iteration count; FP contraction controlled. Integer operations are modulo 2^32. Outputs are checked against host references. No matrix acceleration is requested.",
         boundary="OpenCL event timing excludes compilation and allocation; efficiency uses continuous-loop wall time."),
    dict(id="matrix", group="compute", title="Dense matrix math", backend="torch",
         summary="FP32, FP16, BF16 and INT8 GEMM with correctness checks for each precision.",
         method="Vendor PyTorch matmul on fixed seeded matrices. TF32 disabled, FP16/BF16 reduced-precision reductions disabled where exposed; accumulation policy recorded. INT8 uses the probed private torch._int_mm API and INT32 output; unavailable paths are skipped. Dense operations count a multiply and add as two operations. Library dispatch is not proof of a specific hardware instruction.",
         boundary="GPU event time excludes input creation/transfers. Sustained-loop wall throughput is used for efficiency."),
    dict(id="memory", group="memory", title="VRAM & cache behavior", backend="opencl",
         summary="Read, write, copy and strided throughput plus dependent pointer-chase latency.",
         method="Working sets sweep from cache-sized to large allocations. Read/write count one byte stream; copy counts read plus write. Pointer chase is a serial dependency chain through a seeded permutation; reported latency includes instruction overhead and is not a pure DRAM latency measurement.",
         boundary="GPU event timing. Cache residency is inferred from size, not asserted from hardware counters."),
    dict(id="transfer", group="transfer", title="Host ↔ GPU transfers", backend="opencl",
         summary="Pinned host-memory upload, download and simultaneous-direction transfer.",
         method="ALLOC_HOST_PTR mapped buffers are filled before timing. Message sizes sweep from 4 KiB to the configured buffer limit. Separate profiling queues measure concurrent directions. Reported duplex throughput is the sum of both directions over their combined span.",
         boundary="Event transfer time and wall completion latency; host allocation and pinning excluded. Host/NUMA/PCIe affect these system-level results."),
    dict(id="latency", group="latency", title="Small-job latency", backend="opencl",
         summary="Submission, device execution and round-trip completion overhead.",
         method="Dispatch a one-work-item kernel and wait for it repeatedly. Measure host enqueue duration, GPU event duration, and host completion duration independently. Report median and p95; this is a driver/runtime/system measurement.",
         boundary="Compilation and warm-up excluded. Individual host timings include Python wrapper overhead."),
    dict(id="video", group="video", title="Hardware video", backend="ffmpeg",
         summary="H.264, HEVC and AV1 hardware encode, decode and transcode, with quality analysis.",
         method="Use explicitly selected NVENC/NVDEC or VAAPI devices. Fixed seeded moving-detail test clip or user-supplied reference. Same resolution, bit rate, frame rate, GOP and pixel format. Encode quality is checked after timing using PSNR/SSIM and VMAF if available. Presets are recorded, not assumed equivalent across vendors. Decode uses the generated hardware-encoded stream; this input dependency is recorded.",
         boundary="Process wall time includes startup and encode uploads. Decode readback is included; transcode retains hardware frames. CPU reference preparation and quality analysis excluded. FPS is throughput, not per-frame latency."),
    dict(id="concurrency", group="concurrency", title="Concurrent workloads", backend="opencl",
         summary="Compute queues, hardware video streams and mixed compute/video sharing one GPU.",
         method="Identical arithmetic jobs with distinct outputs run through independent OpenCL queues and host threads. Media cases use one, two and four hardware encoders; a mixed case adds one arithmetic loop. Video input preparation and process startup are recorded boundaries. Report aggregate throughput and per-job slowdown against the single-job baseline.",
         boundary="Wall time from submitting concurrent jobs until all complete, including host scheduling."),
    dict(id="stability", group="stability", title="Sustained workloads", backend="opencl",
         summary="Compute, memory and hardware video phases with time-series throughput.",
         method="Split the configured total soak duration across selected compute, memory and video phases. Check compute/memory outputs, record throughput and telemetry, compare early and late intervals. Video uses the first configured codec, resolution and mode, with preparation excluded and encode process boundaries included.",
         boundary="Wall time per interval, including checks; GPU power collected independently. User temperature ceiling stops the run if exposed sensor reaches it."),
    dict(id="integrity", group="integrity", title="VRAM integrity", backend="opencl",
         summary="Address-dependent pattern verification across a configurable memory allocation.",
         method="Allocate bounded VRAM; write zero, all-one, alternating and address-hashed patterns in multiple passes, then GPU-verify every word and read mismatch counts. Host spot-checks verify the verifier. Cannot allocate driver-reserved memory; this is not a hardware certification or an ECC scrub.",
         boundary="Reports tested MiB, patterns, passes, mismatch counts and elapsed time. Other workloads' correctness checks remain enabled."),
]
BY_ID = {t["id"]: t for t in TESTS}
PROFILES = {
    "quick": dict(duration=2.0, warmup=0.5, repeats=3, buffer_mib=64, matrix_sizes=[1024, 2048], soak_seconds=30, video_seconds=3),
    "standard": dict(duration=10.0, warmup=2.0, repeats=3, buffer_mib=512, matrix_sizes=[4096, 8192], soak_seconds=1800, video_seconds=10),
    "extended": dict(duration=30.0, warmup=5.0, repeats=5, buffer_mib=1024, matrix_sizes=[4096, 8192, 12288], soak_seconds=3600, video_seconds=20),
}

def config(profile="quick", **overrides):
    if profile not in PROFILES:
        raise ValueError("Unknown profile")
    value = deepcopy(PROFILES[profile])
    value.update(profile=profile, sample_interval=1.0, precisions=["fp32", "fp16", "bf16", "int8"],
                 codecs=["h264", "hevc", "av1"], resolutions=["1920x1080", "3840x2160"],
                 video_modes=["quality", "low-latency"], reference=None, max_temp=90.0,
                 memory_fraction=0.5, integrity_passes=2, timeout=7200,
                 concurrency_media=True, soak_workloads=["compute","memory","video"])
    unknown=set(overrides)-set(value)-{'allow_partial'}
    if unknown:raise ValueError('Unknown configuration fields: '+', '.join(sorted(unknown)))
    value.update({k: v for k, v in overrides.items() if v is not None})
    if not isinstance(value['memory_fraction'],(int,float)) or isinstance(value['memory_fraction'],bool) or not (0.05 <= value["memory_fraction"] <= 0.8):
        raise ValueError("memory_fraction must be 0.05–0.8")
    for key, low, high in [("duration", .1, 600), ("warmup", 0, 60), ("repeats", 1, 20),
                           ("buffer_mib", 1, 65536), ("soak_seconds", 1, 86400),
                           ("video_seconds", 1, 600), ("sample_interval", .1, 10),
                           ("max_temp", 40, 110), ("integrity_passes", 1, 20), ("timeout", 5, 172800)]:
        if not isinstance(value[key],(float,int)) or isinstance(value[key],bool) or not low <= value[key] <= high:
            raise ValueError(f"{key} must be {low}–{high}")
    for key in ("repeats", "buffer_mib", "integrity_passes", "timeout"):
        if value[key]!=int(value[key]):raise ValueError(f'{key} must be an integer')
        value[key] = int(value[key])
    if not value["matrix_sizes"] or any(not isinstance(n, int) or not 128 <= n <= 32768 for n in value["matrix_sizes"]):
        raise ValueError("Matrix sizes must be integers from 128 to 32768")
    if not value["precisions"] or set(value["precisions"]) - {"fp32", "fp16", "bf16", "int8"}:
        raise ValueError("Invalid matrix precision")
    if not value["codecs"] or set(value["codecs"]) - {"h264", "hevc", "av1"}:
        raise ValueError("Invalid codec")
    if not value["resolutions"] or set(value["resolutions"]) - {"1280x720", "1920x1080", "3840x2160"}:
        raise ValueError("Invalid video resolution")
    if not value["video_modes"] or set(value["video_modes"]) - {"quality", "low-latency"}:
        raise ValueError("Invalid video mode")
    if not value["soak_workloads"] or set(value["soak_workloads"])-{"compute","memory","video"}:
        raise ValueError("Invalid soak workload")
    if not isinstance(value["concurrency_media"],bool):raise ValueError("concurrency_media must be boolean")
    return value

def select_tests(names=None):
    names = names or ["all"]
    selected = []
    for name in names:
        found = [t["id"] for t in TESTS if name in ("all", t["id"], t["group"])]
        if not found:
            raise ValueError(f"Unknown test/group: {name}")
        selected.extend(found)
    return list(dict.fromkeys(selected))

def metric(name, value, unit, direction="higher", **extra):
    return dict(name=name, value=float(value), unit=unit, direction=direction, **extra)
