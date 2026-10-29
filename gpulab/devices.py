import csv
import json
import os
import shutil
import sys
from pathlib import Path
from .util import command, host_info, home

VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}

def read(path, default=None):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default

def normalize_bus(value):
    if not value:
        return None
    parts = value.lower().split(":")
    if len(parts) == 3:
        return f"{int(parts[0], 16):04x}:{parts[1]}:{parts[2]}"
    return value.lower()

def backend_python(gpu, root=None):
    vendor = gpu.get("vendor", "")
    override = os.environ.get("GPU_LAB_TORCH_" + vendor.upper())
    candidate = Path(root or home()) / "backends" / vendor / "bin" / "python"
    return override or (str(candidate) if candidate.exists() else sys.executable)

def probe(kind, gpu=None, root=None, timeout=45):
    python = backend_python(gpu or {}, root) if kind == "torch" else sys.executable
    args = [python, str(Path(__file__).with_name('backend_entry.py')), "probe", kind]
    if gpu:
        args += [json.dumps(gpu)]
    rc, out, err = command(args, timeout)
    try:
        return json.loads(out.splitlines()[-1]) if rc == 0 else {"error": (err or out)[-3000:]}
    except (ValueError, IndexError):
        return {"error": (err or out or "Probe returned no JSON")[-3000:]}

def discover():
    devices = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        if "-" in card.name:
            continue
        pci = (card / "device").resolve()
        vendor = VENDORS.get(read(pci / "vendor"))
        if not vendor:
            continue
        bus = normalize_bus(pci.name)
        rc, desc, _ = command(["lspci", "-s", bus], timeout=3)
        render = next(iter((pci / "drm").glob("renderD*")), None)
        driver = (pci / "driver").resolve().name if (pci / "driver").exists() else None
        total = read(pci / "mem_info_vram_total")
        devices.append(dict(id=bus, bus=bus, vendor=vendor, name=desc.split(": ", 1)[-1] if rc == 0 else f"{vendor.upper()} {read(pci / 'device', '')}",
                            driver=driver, sysfs=str(pci), render_node=f"/dev/dri/{render.name}" if render else None,
                            vram_bytes=int(total) if total else None,
                            pcie_speed=read(pci / "current_link_speed"), pcie_width=read(pci / "current_link_width"),
                            numa_node=read(pci / "numa_node")))
    rc, out, _ = command(["nvidia-smi", "--query-gpu=uuid,pci.bus_id,name,memory.total,driver_version,index", "--format=csv,noheader,nounits"])
    if rc == 0:
        for row in csv.reader(out.splitlines()):
            if len(row) != 6:
                continue
            uuid, bus, name, mem, driver, ordinal = [x.strip() for x in row]
            bus = normalize_bus(bus)
            gpu = next((g for g in devices if g["bus"] == bus), None)
            if gpu is None:
                gpu = dict(id=bus, bus=bus, vendor="nvidia"); devices.append(gpu)
            gpu.update(name=name, uuid=uuid, nvidia_index=int(ordinal), vram_bytes=int(float(mem)) * 1024**2, driver=driver)
    cl = probe("opencl")
    for item in cl.get("devices", []):
        gpu = next((g for g in devices if item.get("bus") and item["bus"] == g.get("bus")), None)
        if gpu is None:
            # A unique normalized name is safe; ambiguous matches require an explicit mapping.
            matches = [g for g in devices if g["vendor"] == item["vendor"] and item["name"].lower() in g["name"].lower()]
            gpu = matches[0] if len(matches) == 1 else None
        if gpu is None:
            gpu = dict(id=item.get("bus") or f"opencl:{item['platform_index']}:{item['device_index']}",
                       bus=item.get("bus"), vendor=item["vendor"], name=item["name"], vram_bytes=item["vram_bytes"])
            devices.append(gpu)
        gpu.update(opencl=item)
        if gpu["vendor"] != "nvidia":
            gpu["pci_description"] = gpu["name"]
            gpu["name"] = item.get("board_name") or item["name"]
        if not gpu.get("driver"):
            gpu["driver"] = item["driver"]
    return dict(host=host_info(), gpus=devices, opencl_error=cl.get("error"),
                ffmpeg=shutil.which("ffmpeg"), ffprobe=shutil.which("ffprobe"))

def select_gpu(inventory, selector=None):
    gpus = inventory["gpus"]
    if selector is None:
        if len(gpus) != 1:
            raise ValueError("Select a GPU explicitly with --gpu (use gpu-lab devices).")
        return gpus[0]
    matches = [g for g in gpus if selector in (g["id"], g.get("uuid"), g.get("name"))]
    if not matches:
        raise ValueError("Selected GPU was not detected")
    return matches[0]
