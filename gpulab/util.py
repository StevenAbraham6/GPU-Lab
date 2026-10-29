import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

def home():
    return Path(os.environ.get("GPU_LAB_HOME", "~/.local/share/gpu-lab")).expanduser().resolve()

def command(args, timeout=15, env=None):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, "", str(e)

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

def process_marker(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" + stat[19]
    except OSError:
        rc, out, _ = command(["ps", "-p", str(pid), "-o", "lstart="], timeout=2)
        return out if rc == 0 and out else None

def host_info():
    mem = None
    try:
        mem = int(Path("/proc/meminfo").read_text().split("MemTotal:")[1].split()[0]) * 1024
    except OSError:
        rc, out, _ = command(["sysctl", "-n", "hw.memsize"])
        if rc == 0:
            mem = int(out)
    cpu = platform.processor()
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return dict(hostname=platform.node(), os=platform.platform(), kernel=platform.release(),
                architecture=platform.machine(), cpu=cpu, logical_cpus=os.cpu_count(), ram_bytes=mem,
                python=platform.python_version())
