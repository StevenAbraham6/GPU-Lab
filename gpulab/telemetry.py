"""Device-reported power, never inferred from TDP or utilization."""
import csv
import glob
import math
import time
from pathlib import Path
from .devices import read
from .util import command

def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) and n >= 0 else None
    except (ValueError, TypeError):
        return None

def sample(gpu):
    result = dict(power_w=None, temperature_c=None, clock_mhz=None, memory_used_mib=None,
                  utilization_pct=None, power_limit_w=None, source=None, power_scope="unknown")
    if gpu.get("vendor") == "nvidia":
        selector=gpu.get('uuid') or gpu.get('bus')
        if not selector:
            result['error']='No unambiguous SMI mapping for this OpenCL GPU';return result
        rc, out, err = command(["nvidia-smi", "-i", selector,
                              "--query-gpu=power.draw,temperature.gpu,clocks.sm,memory.used,utilization.gpu,power.limit",
                              "--format=csv,noheader,nounits"], timeout=5)
        if rc == 0:
            row = next(csv.reader(out.splitlines()), [])
            if len(row) == 6:
                for key, v in zip(("power_w", "temperature_c", "clock_mhz", "memory_used_mib", "utilization_pct", "power_limit_w"), row):
                    result[key] = number(v.strip())
                result.update(source="nvidia-smi power.draw", power_scope="NVIDIA device-reported board power")
        else:
            result["error"] = err[-500:]
    else:
        path = gpu.get("sysfs")
        if path:
            for hw in sorted(Path(path).glob("hwmon/hwmon*")):
                power_path = next((p for p in (hw / "power1_average", hw / "power1_input") if p.exists()), None)
                if power_path:
                    p = number(read(power_path))
                    result["power_w"] = p / 1e6 if p is not None else None
                    result.update(source=str(power_path), power_scope=f"Device-reported {read(hw / 'power1_label', 'unverified power domain')}")
                # Use edge temperature if labelled; otherwise first exposed sensor, with source retained.
                temp_path = hw / "temp1_input"
                t = number(read(temp_path))
                result["temperature_c"] = t / 1000 if t is not None else None
                result["temperature_sensor"] = read(hw / "temp1_label", "temp1")
                cap = number(read(hw / "power1_cap")); result["power_limit_w"] = cap / 1e6 if cap is not None else None
                freq = number(read(hw / "freq1_input")); result["clock_mhz"] = freq / 1e6 if freq is not None else None
            used = number(read(Path(path) / "mem_info_vram_used"))
            result["memory_used_mib"] = used / 1024**2 if used is not None else None
            result["utilization_pct"] = number(read(Path(path) / "gpu_busy_percent"))
    return result

def integrate(samples, start, end, interval=1.0):
    """Integrate timestamped samples; require coverage at BOTH measurement boundaries.

    The collector takes a sample immediately before every phase. Endpoint interpolation
    uses bracketing samples. Large holes and unbracketed boundaries invalidate energy.
    """
    valid = sorted((s for s in samples if s.get("power_w") is not None), key=lambda s: s["ts"])
    info = dict(energy_j=None, average_power_w=None, peak_power_w=None, sample_count=0,
                coverage="unavailable", power_scope=None)
    if end <= start or len(valid) < 2:
        return info
    before = [s for s in valid if s["ts"] <= start]
    after = [s for s in valid if s["ts"] >= end]
    if not before or not after:
        info["coverage"] = "unbracketed measurement interval"
        return info
    span = [before[-1]] + [s for s in valid if start < s["ts"] < end] + [after[0]]
    if any(b["ts"] - a["ts"] > max(3 * interval, 3) for a, b in zip(span, span[1:])):
        info["coverage"] = "sampling gap"
        return info
    scopes = {s.get("power_scope") for s in span}
    if len(scopes) != 1:
        info["coverage"] = "power domain changed"
        return info
    energy, peak = 0.0, 0.0
    for a, b in zip(span, span[1:]):
        lo, hi = max(start, a["ts"]), min(end, b["ts"])
        if hi <= lo or b["ts"] == a["ts"]:
            continue
        slope = (b["power_w"] - a["power_w"]) / (b["ts"] - a["ts"])
        pl = a["power_w"] + slope * (lo - a["ts"])
        ph = a["power_w"] + slope * (hi - a["ts"])
        energy += (pl + ph) * .5 * (hi - lo)
        peak = max(peak, pl, ph)
    info.update(energy_j=energy, average_power_w=energy/(end-start), peak_power_w=peak,
                sample_count=len(span), coverage="covered", power_scope=span[0].get("power_scope"),
                note="GPU telemetry estimate; includes idle gaps inside this phase; vendor sensor domains may differ.")
    return info
