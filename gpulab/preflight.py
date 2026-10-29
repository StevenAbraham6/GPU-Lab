import importlib.util
import shutil
import os
import sys
from pathlib import Path
from .catalog import BY_ID
from .devices import probe, backend_python
from .telemetry import sample
from .util import command, now

def check(gpu, tests, cfg, root, execute=True):
    items = []
    def add(name, state, detail, tests=None):
        items.append(dict(name=name, state=state, detail=detail, tests=tests or []))
    add("Platform", "ready" if sys.platform.startswith("linux") else "warning",
        "Linux is the supported GPU target. macOS supports the UI, reports and explicitly simulated demonstrations.")
    usage = shutil.disk_usage(root)
    required = 5 * 1024**3 if "video" in tests else 512 * 1024**2
    add("Disk space", "ready" if usage.free >= required else "blocked", f"{usage.free / 1024**3:.1f} GiB free; need {required / 1024**3:.1f} GiB")
    if gpu.get("simulated"):
        add("Simulation", "warning", "DEMONSTRATION ONLY: all measurements are synthetic and never hardware results.")
        return dict(at=now(), state="ready" if usage.free >= required else "blocked", checks=items, simulated=True)
    if not gpu.get("id"):
        add("Selected GPU", "blocked", "No physical GPU selected")
    else:
        add("Selected GPU", "ready", f"{gpu['name']} · {gpu['id']}")
    cl_tests = [t for t in tests if BY_ID[t]["backend"] == "opencl"]
    if cl_tests:
        if not gpu.get("opencl"):
            add("OpenCL GPU runtime", "blocked", "Selected physical GPU has no mapped OpenCL GPU device. Install its vendor ICD/runtime. CPU OpenCL devices are excluded.", cl_tests)
        elif execute:
            result = probe("opencl-execute", gpu)
            add("OpenCL execution & correctness", "blocked" if result.get("error") else "ready", result.get("error") or "GPU kernel output validated", cl_tests)
    if "matrix" in tests:
        result = probe("torch", gpu, root) if execute else {}
        if result.get("error"):
            add("Matrix backend", "blocked", result["error"] + f" [Python: {backend_python(gpu, root)}]", ["matrix"])
        else:
            for precision in cfg["precisions"]:
                mode = result.get("precisions", {}).get(precision, {})
                add(f"Matrix {precision.upper()}", "ready" if mode.get("supported") else "warning",
                    "Tiny GPU execution probe passed" if mode.get("supported") else "Unsupported: " + mode.get("reason", "not probed"), ["matrix"])
            if execute and not any(result.get("precisions", {}).get(p, {}).get("supported") for p in cfg["precisions"]):
                add("Requested matrix precisions", "blocked", "None passed execution", ["matrix"])
    telemetry = sample(gpu)
    add("GPU power telemetry", "ready" if telemetry["power_w"] is not None else "warning",
        f"{telemetry['power_w']} W · {telemetry['power_scope']} · {telemetry['source']}" if telemetry["power_w"] is not None else "No readable power sensor. Performance can run; efficiency will be unavailable.")
    t = telemetry.get("temperature_c")
    add("Starting temperature", "blocked" if t is not None and t >= cfg["max_temp"] else "ready" if t is not None else "warning",
        f"{t} °C; user stop ceiling {cfg['max_temp']} °C" if t is not None else "Temperature unavailable; thermal stop cannot be enforced from telemetry")
    if gpu.get("vendor") == "nvidia":
        selector=gpu.get('uuid') or gpu.get('bus')
        rc, out, _ = command(["nvidia-smi", "-i", selector, "--query-compute-apps=pid,process_name", "--format=csv,noheader"]) if selector else (1,'','')
        add("Competing GPU compute processes", "warning" if rc or out else "ready", out or ("Process inventory unavailable" if rc else "No NVIDIA compute processes reported; graphics/other activity may still exist"))
    else:
        busy = telemetry.get("utilization_pct")
        add("GPU activity", "warning" if busy is None or busy > 5 else "ready", f"Reported utilization: {busy}% (not a complete process inventory)")
    if "video" in tests:
        rc, encoders, _ = command(["ffmpeg", "-hide_banner", "-encoders"])
        filters = command(["ffmpeg", "-hide_banner", "-filters"])[1]
        add("FFmpeg & FFprobe", "ready" if rc == 0 and shutil.which("ffprobe") else "blocked", "Both are required; codec probes below validate runtime support", ["video"])
        add("Video quality metrics", "ready" if "libvmaf" in filters else "warning", "VMAF available" if "libvmaf" in filters else "VMAF missing from FFmpeg; PSNR/SSIM will be reported. Install a libvmaf-enabled FFmpeg for perceptual quality.", ["video"])
        if gpu.get("vendor") != "nvidia" and not gpu.get("render_node"):
            add("Video device mapping", "blocked", "No DRM render node mapped to this GPU", ["video"])
        ref = cfg.get("reference")
        if ref and not Path(ref).is_file():
            add("Video reference", "blocked", "Reference file does not exist on the server", ["video"])
        if execute and rc == 0:
            from .video import probe_video
            for codec in cfg["codecs"]:
                p = probe_video(gpu, codec)
                add(f"Hardware encoder {codec.upper()}", "ready" if p["supported"] else "warning", p.get("reason") or p["encoder"], ["video"])
    if gpu.get("vram_bytes"):
        proposed = cfg["buffer_mib"] * 1024**2 * 3
        add("VRAM allocation budget", "ready" if proposed < gpu["vram_bytes"] * cfg["memory_fraction"] else "warning",
            f"Detected {gpu['vram_bytes']/1024**3:.1f} GiB total; workloads bound allocations and fail explicitly on OOM; current free VRAM is runtime-dependent")
    return dict(at=now(), state="blocked" if any(i["state"] == "blocked" for i in items) else "warning" if any(i["state"] == "warning" for i in items) else "ready", checks=items)
