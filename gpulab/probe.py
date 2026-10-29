"""Risky driver imports and tiny execution probes live outside the web process."""
import json
import sys
import ctypes
from pathlib import Path
from .devices import normalize_bus

def opencl_probe():
    import pyopencl as cl
    items = []
    for pi, platform in enumerate(cl.get_platforms()):
        for di, d in enumerate(platform.get_devices()):
            if not d.type & cl.device_type.GPU:
                continue
            v = d.vendor.lower()
            vendor = "nvidia" if "nvidia" in v else "amd" if any(s in v for s in ("amd", "advanced micro")) else "intel" if "intel" in v else "other"
            bus = None
            try:
                info = d.pci_bus_info_khr
                bus = f"{info.pci_domain:04x}:{info.pci_bus:02x}:{info.pci_device:02x}.{info.pci_function}"
            except Exception:
                try:
                    if vendor == "nvidia":
                        bus = f"0000:{d.pci_bus_id_nv:02x}:{d.pci_slot_id_nv >> 3:02x}.{d.pci_slot_id_nv & 7}"
                    elif vendor == "amd":
                        top = d.topology_amd
                        bus = f"0000:{top.bus:02x}:{top.device:02x}.{top.function}"
                except Exception:
                    pass
            try:board_name=d.board_name_amd.strip()
            except Exception:board_name=None
            items.append(dict(platform_index=pi, device_index=di, name=d.name.strip(), board_name=board_name, vendor=vendor,
                              bus=normalize_bus(bus), vram_bytes=d.global_mem_size, max_alloc_bytes=d.max_mem_alloc_size,
                              driver=d.driver_version, platform=platform.name, version=d.version))
    return {"devices": items}

def runtime_bus(index,vendor):
    """Resolve runtime ordinal to PCI identity using its already loaded shared library."""
    target="libamdhip64" if vendor=="amd" else "libcudart"
    function="hipDeviceGetPCIBusId" if vendor=="amd" else "cudaDeviceGetPCIBusId"
    try:
        paths={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if target in line and '/' in line}
        for path in paths:
            lib=ctypes.CDLL(path);fn=getattr(lib,function)
            fn.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int];fn.restype=ctypes.c_int
            buf=ctypes.create_string_buffer(64)
            if fn(buf,64,index)==0:return normalize_bus(buf.value.decode())
    except (OSError,AttributeError):pass
    return None

def torch_device(gpu):
    import torch
    vendor = gpu["vendor"]
    if vendor == "intel":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("Intel PyTorch XPU not available; this adapter is experimental")
        runtime, kind = torch.xpu, "xpu"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA/ROCm GPU available in this Python environment")
        if bool(torch.version.hip) != (vendor == "amd"):
            raise RuntimeError("Wrong PyTorch build for selected GPU vendor")
        runtime, kind = torch.cuda, "cuda"
    candidates = []
    for i in range(runtime.device_count()):
        p = runtime.get_device_properties(i)
        bus = normalize_bus(getattr(p, "pci_bus_id", None)) or runtime_bus(i,vendor)
        uuid = str(getattr(p, "uuid", ""))
        same_uuid = gpu.get("uuid") and uuid.lower().replace("gpu-", "") == gpu["uuid"].lower().replace("gpu-", "")
        if same_uuid or (bus and bus == gpu.get("bus")):
            return torch, runtime, f"{kind}:{i}"
        if p.name.strip().lower() == gpu["name"].strip().lower():
            candidates.append(i)
    # Single-device processes still require matching model name; never just use ordinal zero.
    if len(candidates) == 1:
        return torch, runtime, f"{kind}:{candidates[0]}"
    raise RuntimeError("Cannot unambiguously map selected physical GPU to PyTorch. Isolate it with CUDA_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES and verify its model name.")

def torch_probe(gpu):
    torch, rt, dev = torch_device(gpu)
    rt.set_device(dev)
    modes = {}
    for precision in ("fp32", "fp16", "bf16", "int8"):
        try:
            dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16, int8=torch.int8)[precision]
            a = torch.ones((128, 128), dtype=dtype, device=dev)
            b = a.clone()
            out = torch._int_mm(a, b) if precision == "int8" else a @ b
            rt.synchronize()
            if not bool((out == 128).all().item()):
                raise RuntimeError("Execution returned incorrect result")
            modes[precision] = dict(supported=True, output_dtype=str(out.dtype))
        except Exception as e:
            modes[precision] = dict(supported=False, reason=str(e)[:1000])
    return dict(device=dev, device_name=rt.get_device_name(dev), version=torch.__version__,
                cuda=getattr(torch.version, "cuda", None), hip=getattr(torch.version, "hip", None),
                precisions=modes, note="Short execution probe; hardware-instruction dispatch not independently profiled")

def main():
    try:
        kind = sys.argv[1]
        if kind == "opencl":
            result = opencl_probe()
        elif kind == "torch":
            result = torch_probe(json.loads(sys.argv[2]))
        elif kind == "opencl-execute":
            from .opencl_backend import OpenCL
            cl = OpenCL(json.loads(sys.argv[2]))
            result = cl.execution_check()
        else:
            raise ValueError("Unknown probe")
    except Exception as e:
        result = {"error": str(e)}
    print(json.dumps(result))

if __name__ == "__main__":
    main()
