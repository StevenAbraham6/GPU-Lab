# GPU Lab

A local, open-source workspace for comparing GPU **performance, energy efficiency, memory, hardware video and sustained behavior**. Launch the same tests from the CLI or web UI. Keep historical runs, edit their descriptions, compare results, and download PDF, HTML, CSV and JSON reports.

**Status: v0.1.0, ready for application evaluation and hardware qualification.** Application tests run without a GPU; physical NVIDIA L4 / AMD Radeon AI PRO R9700 / R9600 validation is still required. An implemented adapter is not a claim that every precision, codec or sensor works on every card. Preflight and runtime checks expose missing support rather than silently using the CPU.

## Installation in one command

From a copy of this repository on an Ubuntu GPU server:

```bash
bash install.sh --backend auto --service
```

This installs OS userspace prerequisites, a private Python environment, the CLI/web application, separate CUDA/ROCm PyTorch environments for detected vendors, and a private FFmpeg + VMAF video toolchain. It also starts the loopback web UI as a user systemd service. There is no Node.js build step, external database or cloud account.

If you have the release archive instead, this is one shell command after copying it to the server:

```bash
tar -xzf gpu-lab-0.1.0.tar.gz && bash gpu-lab-0.1.0/install.sh --backend auto --service
```

From a new server, install directly from the published GitHub source:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/StevenAbraham6/GPU-Lab/main/install.sh) --backend auto --service
```

The bootstrap installs the source published on `main`; local development changes are not uploaded by the installer. For a private repository, clone with your normal Git authentication and run `bash install.sh`; do not put credentials into a public bootstrap URL. For repeatable deployments, set `GPU_LAB_REF` to a published release tag.

### Host prerequisites and boundaries

- Primary GPU target: **Ubuntu 24.04 x86-64**, with a driver/kernel combination supported by the GPU vendor. The dependency installer also uses apt on Debian-family systems, but those OS/GPU combinations require qualification.
- Working vendor kernel driver, device permissions and OpenCL GPU ICD/runtime. Installing Python packages does not install a kernel driver.
- AMD Radeon must use an AMD-supported OS/kernel/driver combination. R9700 and R9600 appear in the ROCm 10.1 hardware list; that does not make an old kernel compatible. See [AMD hardware/OS support](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html).
- NVIDIA: `nvidia-smi` must work. The default math environment uses a CUDA 12.8 PyTorch build; it needs a compatible driver. The CUDA toolkit/compiler is not required for those PyTorch wheels.
- AMD: `/dev/kfd` and the GPU's DRM render node must be accessible. Install the matching vendor OpenCL runtime/ICD as well as the kernel driver. `rocm-opencl-runtime` is available through the appropriate AMD apt repository; the installer does not register or replace vendor repositories.
- `/dev/dri/renderD*` access is required for AMD/Intel VAAPI media. NVIDIA uses NVENC/NVDEC through its driver.
- Internet access, sudo for apt prerequisites, and enough disk space for Python/FFmpeg builds and GPU wheels. Allow **15–30 GiB** for a full two-vendor installation and build caches, plus run artifacts. Building video tools can take several minutes.
- A display server is not required. The application does not change power limits, clocks, drivers, Secure Boot or BIOS settings.

**A GPU with no driver may require driver installation, Secure Boot enrollment and a reboot before any benchmark can work.** GPU Lab installs the entire application in one command; it cannot eliminate those host requirements. It detects and explains them through `doctor`/preflight instead of claiming a successful hardware setup.

For NVIDIA on Ubuntu, follow the [Ubuntu server driver instructions](https://ubuntu.com/server/docs/nvidia-drivers-installation/). For AMD, follow the [ROCm/Radeon compatibility and installation documentation](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html). Do not mix driver branches or disable Secure Boot merely to satisfy an application dependency.

### Installer options

```bash
bash install.sh --backend nvidia        # CUDA matrix backend
bash install.sh --backend amd           # ROCm matrix backend
bash install.sh --backend both          # isolated environments coexist
bash install.sh --backend none          # no PyTorch environment; OpenCL/video still install
bash install.sh --skip-video-build      # distro FFmpeg; VMAF may be missing
bash install.sh --no-system-packages    # admin already supplied apt prerequisites
bash install.sh --app-only              # UI, reports and demonstration on macOS/Linux
```

Default paths:

| Purpose | Location / override |
|---|---|
| Application, venv, private FFmpeg | `~/.local/share/gpu-lab-app` / `GPU_LAB_PREFIX` |
| SQLite journal, runs, backend environments | `~/.local/share/gpu-lab` / `GPU_LAB_HOME` |
| Launcher | `~/.local/bin/gpu-lab` / `GPU_LAB_BIN` |
| Service | `~/.config/systemd/user/gpu-lab.service` |

If `~/.local/bin` is not in your shell PATH, invoke `~/.local/bin/gpu-lab` directly or add that directory to PATH.

The default userspace math pins are NVIDIA `torch==2.7.1` from the official `cu128` index and AMD `torch[device-all]==2.14.0+rocm10.1.0` from AMD's documented index. These are isolated, **not a guarantee that the host driver is suitable**. Override `GPU_LAB_NVIDIA_INDEX`, `GPU_LAB_NVIDIA_TORCH`, `GPU_LAB_AMD_INDEX` and `GPU_LAB_AMD_TORCH` when qualifying another supported stack. NVIDIA's default pin needs a compatible Python wheel (Python 3.10–3.12 is the intended installation range); select `GPU_LAB_PYTHON=python3.12` if the host default is newer.

Existing environments can be used via `GPU_LAB_TORCH_NVIDIA=/path/to/python` and `GPU_LAB_TORCH_AMD=/path/to/python`. Set these variables in the service environment too if you use a service. Device/backend versions are attached to results.

### Web UI and remote access

```bash
gpu-lab serve
```

Open **http://127.0.0.1:8080** on the server. From a laptop, create a tunnel:

```bash
ssh -L 8080:127.0.0.1:8080 ubuntu@YOUR_SERVER
```

Then open http://127.0.0.1:8080 on the laptop. The UI provides Overview, New benchmark, Run library and Compare runs. It has no analytics or third-party web assets.

To bind outside loopback, supply a token; an unauthenticated non-loopback listener is rejected:

```bash
GPU_LAB_TOKEN='YOUR_LONG_RANDOM_TOKEN' gpu-lab serve --host 0.0.0.0 --port 8080
```

Use HTTPS through a reverse proxy or an SSH tunnel on untrusted networks. The UI accepts the token through a login form and uses an HttpOnly SameSite cookie; API clients can use `Authorization: Bearer ...`. Tokens are not placed in URLs. Browser mutations reject foreign origins. This is a single trusted-operator workspace, not a multi-tenant hosted service.

### Persistent service

`--service` installs a **user** systemd unit. To keep it running after logout:

```bash
sudo loginctl enable-linger "$USER"
systemctl --user status gpu-lab
journalctl --user -u gpu-lab -f
systemctl --user restart gpu-lab
```

The unit uses `KillMode=process`: restarting the UI does not intentionally kill detached benchmark workers. A reboot cannot preserve an in-flight GPU operation; the run will be marked interrupted and completed test records remain available.

## First hardware run

```bash
gpu-lab devices
gpu-lab doctor all --gpu 0000:60:00.0 --profile quick
gpu-lab run all --gpu 0000:60:00.0 --profile quick --name 'L4 baseline' --wait
```

Use the **actual ID returned by `devices`**; `0000:60:00.0` is an example. Selection is required when more than one GPU is detected. CPU OpenCL devices are excluded. Physical PCI/UUID mappings are preferred; ambiguous PyTorch mappings fail explicitly.

Start with `quick` to qualify execution and correctness. Use `standard` for comparative measurements after the GPU has a stable environment. Run the same settings on the AMD server, export its JSON bundle, and import it into the same UI for comparison.

### Individual tests and groups

```bash
gpu-lab tests
gpu-lab run arithmetic --gpu GPU_ID --wait
gpu-lab run matrix --gpu GPU_ID --config examples/compute.json --wait
gpu-lab run memory transfer latency --gpu GPU_ID --wait
gpu-lab run video --gpu GPU_ID --config examples/video.json --wait
gpu-lab run concurrency stability integrity --gpu GPU_ID --profile standard --wait
gpu-lab run compute --gpu GPU_ID --profile standard --wait
```

`compute` expands to `arithmetic` and `matrix`. `all` expands to all nine groups. FFT/reduction/stencil application tests, AI model inference, graphics and ray tracing are deliberately outside this suite.

### Preflight

Preflight checks run automatically in the durable worker and can also be launched independently in the CLI/UI. They check:

- Device identity and mapping, OpenCL GPU execution/correctness.
- Matrix backend availability and a tiny execution check for each requested precision.
- Available GPU power/temperature telemetry and the configured temperature ceiling.
- Disk space and allocation budget.
- NVIDIA compute processes, or AMD utilization as a partial activity check.
- FFmpeg/FFprobe, VMAF filter availability and hardware encoder smoke probes.
- Reference clip existence; GPU Lab reservations are displayed in the UI.

**Ready** means the check passed. **Warning** means some interpretation or partial capability is required. **Blocked** means the requested run cannot start as configured. Active jobs on the same GPU queue behind one another using an OS file lock.

```bash
gpu-lab run all --gpu GPU_ID --allow-partial
```

`--allow-partial` explicitly skips blocked per-test dependencies and records `missing_dependency`. Global low-disk/high-temperature blocks still stop the run. Per-codec/precision runtime limitations are recorded in the relevant case. No data is substituted.

## What the suite measures

| ID | Workload | Output and direction |
|---|---|---|
| `arithmetic` | Register-heavy FP32 FMA and wrapping INT32 multiply/add | GFLOP/s or GOP/s: **higher** |
| `matrix` | Dense square/rectangular/batched GEMM, FP32/FP16/BF16/INT8 where available | TFLOP/s or TOP/s: **higher** |
| `memory` | Read/write/copy, strided copy, working-set sweeps, dependent pointer chase | GB/s: **higher**; ns/access: **lower** |
| `transfer` | Pinned host upload/download and simultaneous directions | GB/s: **higher**; completion µs: **lower** |
| `latency` | Host submission, device execution, completion of a tiny kernel | Median/p95 µs: **lower** |
| `video` | Hardware encode, decode/readback and transcode; quality after timing | FPS and quality: **higher**, subject to bitrate/quality tradeoff |
| `concurrency` | 1/2/4 compute queues, 1/2/4 video streams, mixed compute/video | Aggregate throughput: **higher**; per-job slowdown: **lower** |
| `stability` | Time-bounded compute/memory/video phases | Sustained throughput and early/late behavior; contextual health results |
| `integrity` | Address-dependent VRAM patterns, mismatch count, injected-error verifier self-test | Errors: **lower**; tested MiB: **coverage**, not a performance score |

All arithmetic outputs are used. Correctness checks run before performance reporting. Integrity verification includes a deliberately injected error to prove the GPU checker can detect one; that injected error is not counted as a hardware fault.

### Floating point and integer interpretation

- FP32 GEMM disables TF32. Reduced-precision FP16/BF16 reductions are disabled where the backend exposes those settings. Input, output, requested accumulation behavior and library versions are recorded.
- FP16/BF16 general execution is not the same thing as a verified matrix-instruction path. Library invocation alone is **not proof of a specific hardware unit**.
- INT8 uses the probed private `torch._int_mm` interface with INT32 output. It is version-sensitive and is not assumed available on ROCm or Intel. Batched INT8 has no portable accelerated implementation in this version and is explicitly unsupported. No slow float conversion is substituted for INT8.
- A matrix multiply-add counts as **two operations**. Dense and sparse headline specifications must not be compared interchangeably. Sparse benchmarking is not implemented.
- Numerical checks sample outputs against CPU references derived from rounded inputs. VRAM tests check their allocated region; neither is a certification of all possible calculations or every physical memory location.

### Power and efficiency

Power is **GPU device telemetry**, never total server power and never a TDP estimate:

- NVIDIA: `nvidia-smi power.draw`, temperature, SM clocks, VRAM usage, utilization, power limit.
- AMD: readable DRM/hwmon power/temperature/frequency files and device utilization/VRAM files, with the exact sensor path and label recorded. This avoids assuming AMD SMI is installed. Missing sensors remain unavailable.
- Intel: common OpenCL/VAAPI execution can be discovered, but its full matrix and telemetry adapter is experimental and **not qualified**.

Energy is trapezoidal integration of timestamped watts. Each measured phase must have samples bracketing both ends, no large sampling holes and a stable sensor domain. Phases shorter than two sample periods (at least two seconds) do not get an energy score.

```text
energy_J       = integral(power_W × time_s)
average_W      = energy_J / measured_wall_seconds
work_per_joule = completed_work / energy_J
energy_per_work_unit = energy_J / completed_work
```

Kernel throughput uses GPU event timing. Efficiency uses **completed work and wall time from the matching sustained loop**, including host dispatch gaps. Warm-up, compilation, CPU clip preparation and quality analysis are excluded from these energy windows. Video process startup and transfers are included and disclosed.

Sensor scope is important: an AMD reported package/PPT domain and NVIDIA reported board power are not automatically identical. Comparison suppresses efficiency rankings when domains differ. Even matching labels are telemetry estimates, not calibrated external measurements. Cross-vendor power-efficiency conclusions require confirming the sensor boundaries. The app does not subtract idle power or automatically change limits.

### Video methodology

- Explicit NVENC/NVDEC on NVIDIA, with each FFmpeg process isolated to the selected GPU UUID; VAAPI with the selected DRM render node on AMD (and an experimental Intel path). No `-hwaccel auto` or CPU encoder fallback.
- H.264, HEVC and AV1 are independently probed. The current formats are **SDR, 8-bit, 4:2:0**, 30 fps. Main resolutions: 1080p and 4K; 720p is also configurable. HDR, 10-bit and 4:4:4 are not implemented.
- Default targets: 4 Mbit/s at 720p, 8 Mbit/s at 1080p, 25 Mbit/s at 4K; GOP 60, B frames disabled. NVENC uses p4 with hq/ll tune; VAAPI uses driver default + CBR/GOP/B-frame constraints. The two named modes do not currently request different VAAPI-specific low-latency tuning, and reports disclose this.
- Synthetic `testsrc2` is a reproducible **calibration clip**, not representative visual-content coverage. Use the same real SDR reference on each server for substantive encoder quality comparisons. Its normalized pixel hash is recorded.
- Encode wall FPS includes CPU reference decode and upload. Decode FPS includes hardware decode and CPU readback. Transcode retains hardware frames. These boundaries are shown; host CPU/I/O can influence results.
- Decode/transcode inputs are generated by that GPU's encoder. Their encoded bitstream hashes therefore commonly differ across GPUs; comparison flags mismatched inputs. They are useful pipeline measurements, not a controlled common-bitstream decoder ranking.
- PSNR/SSIM and pinned **VMAF v0.6.1** run after encoding, outside timing/power windows. Missing VMAF is visible. The installer builds a private libvmaf-enabled FFmpeg by default.
- FPS is throughput, **not per-frame latency**. Concurrent clip completion time is also not per-frame latency.
- Compare quality at a matching bitrate and latency policy. The tool does not implement automatic quality-target or rate-distortion curve search. Cross-vendor preset names are not equivalent; video tradeoff rows are shown without an automatic winner.
- Encoded output/log files are retained locally. Prepared lossless references are removed after a successful test to reduce disk use; original user references are never changed.

### Profiles and customization

| Profile | Repeats | Per-case timed loop | Max buffer request | Matrix square sizes | Total soak budget | Clip length |
|---|---:|---:|---:|---|---:|---:|
| `quick` | 3 | 2 s | 64 MiB | 1024, 2048 | 30 s | 3 s |
| `standard` | 3 | 10 s | 512 MiB | 4096, 8192 | 1800 s | 10 s |
| `extended` | 5 | 30 s | 1024 MiB | 4096, 8192, 12288 | 3600 s | 20 s |

Every selected precision and shape adds cases. A full run takes substantially longer than a single case, especially with video preparation/quality analysis. The soak budget is split across `soak_workloads`; each interval/clip can overrun the budget slightly. It is not an exact scheduling deadline.

JSON configuration files or the UI's Advanced configuration override defaults:

```json
{
  "duration": 10,
  "warmup": 2,
  "repeats": 3,
  "buffer_mib": 512,
  "matrix_sizes": [4096, 8192],
  "precisions": ["fp32", "fp16", "bf16", "int8"],
  "sample_interval": 1,
  "max_temp": 90,
  "memory_fraction": 0.5,
  "integrity_passes": 2,
  "soak_seconds": 1800,
  "soak_workloads": ["compute", "memory", "video"],
  "concurrency_media": true,
  "codecs": ["h264", "hevc", "av1"],
  "resolutions": ["1920x1080", "3840x2160"],
  "video_modes": ["quality", "low-latency"],
  "video_seconds": 10,
  "reference": null,
  "timeout": 7200
}
```

`max_temp` is a **user-configurable stop ceiling**, not a claim that 90°C is suitable for every card. Set it for your hardware. A stop requires a readable temperature sensor. Allocation budgets are bounded by reported device limits; OOMs still fail explicitly. `buffer_mib` caps integrity allocation as well as bandwidth buffers, so the default integrity test does not cover all VRAM.

## Durable execution and history

```bash
gpu-lab run all --gpu GPU_ID --name 'Run A'       # prints ID, returns immediately
gpu-lab list
gpu-lab show RUN_ID
gpu-lab wait RUN_ID                             # Ctrl-C detaches; does not cancel
gpu-lab cancel RUN_ID
gpu-lab resume RUN_ID                           # linked run of unfinished/failed groups
gpu-lab reproduce RUN_ID                        # linked fresh run of every original group
```

- A detached supervisor owns the run, not the browser or HTTP request.
- One subprocess per test contains most driver crashes/OOMs and releases allocations on exit.
- SQLite WAL is the authoritative journal. Each completed test is committed and has a JSON snapshot. Telemetry is committed as it arrives.
- The live report and `gpu-lab wait` show current-test and whole-run elapsed time.
- Workers update a heartbeat and identify their PID using boot ID/start time. Dead workers are marked interrupted when the UI/CLI inspects history.
- Cancellation terminates the benchmark process group, including FFmpeg children. A benchmark also monitors its supervisor to clean up after supervisor loss. A GPU driver hang may require administrator recovery; software cannot guarantee recovery from a hung kernel.
- An interrupted **active test** may lose its uncommitted cases; earlier completed test groups survive. Continuation creates a new linked run, rather than merging separate thermal conditions into one record.
- Browser refresh/disconnection and UI service restart do not intentionally cancel benchmarks. A host reboot interrupts active work.

## Reports, metadata and comparisons

```bash
gpu-lab edit RUN_ID --json examples/metadata.json
gpu-lab export RUN_ID --format pdf  --output results/run-a.pdf
gpu-lab export RUN_ID --format html --output results/run-a.html
gpu-lab export RUN_ID --format csv  --output results/run-a.csv
gpu-lab export RUN_ID --format json --output results/run-a.json
gpu-lab import results/another-server.json
gpu-lab compare BASELINE_ID CANDIDATE_ID
```

The UI has the same downloads and actions. Run history supports name/GPU/tag search and hardware/demo/active filters. Import a **JSON bundle** for comparisons across servers. PDFs are presentation artifacts, not importable measurement containers.

- Names, descriptions, notes, tags and hardware metadata are editable before/after execution. Edits produce a revision and preserve the previous metadata.
- Original detected hardware, selected workloads, raw results and provenance remain separate from editable descriptions. Downloads use current metadata plus retained original evidence.
- PDFs use stable group ordering, units, direction labels, repeated table headers, page numbers and methodology/settings appendices. HTML has expandable case details. CSV includes unit/direction/settings fields. JSON includes results, telemetry and metadata history.
- Higher throughput/work per joule and lower latency/energy per identical work unit are desirable. Raw watts, temperature, tested memory coverage and overall run energy are contextual values, not universal winners.
- Comparison matches test/case/metric/settings. Unsupported/missing cases stay visible. Different workload settings, suite versions or power domains suppress ranking. Hardware and simulated data cannot be ranked against one another.
- Kernel/library/driver differences are part of the measured stack; results are not isolated silicon specifications. CPU/NUMA/PCIe differences particularly affect transfer and small-job results.
- Repeatability is shown as median, range, coefficient of variation and p95 where meaningful. The UI does not yet provide statistical significance tests; do not treat small noisy differences as decisive.
- Bundle fingerprints detect accidental content changes; they are **not cryptographic signatures or proof of authenticity**. Import trusted bundles only. Local video/log files are not embedded in the JSON bundle; measurements and telemetry are.

### Demonstration without a GPU

```bash
gpu-lab run all --demo --name 'UI demonstration' --wait
gpu-lab serve
```

Every screen/report explicitly marks this data **SIMULATED**. Demonstrations never establish hardware compatibility or performance. The preview PDF in `outputs/` is also synthetic.

## Troubleshooting

| Finding | Action |
|---|---|
| No GPU detected | Check PCI inventory, host driver and passthrough. macOS is UI/demo only. |
| No mapped OpenCL device | Run `clinfo`; install the vendor's matching GPU ICD. A CPU ICD is insufficient. |
| NVIDIA driver unavailable | Fix `nvidia-smi` first; verify kernel module matches `uname -r` and Secure Boot signing. |
| AMD permission failure | Check `/dev/kfd` and render-node group permissions. Log out/in after an administrator changes video/render group membership. |
| PyTorch wrong vendor | Use the separate backend environment or `GPU_LAB_TORCH_VENDOR` override. CUDA and ROCm torch wheels must not replace each other in one environment. |
| Ambiguous GPU mapping | Use stable PCI/UUID IDs. Remove stale device masks or isolate one matching physical GPU explicitly. Never guess ordinal zero. |
| INT8 unsupported | Inspect the probe reason; that PyTorch/backend path may not exist. Do not substitute FP16 or label scalar INT8 as accelerated matrix throughput. |
| Hardware codec unsupported | Inspect the encoder smoke probe and test log. Codec/format/driver capabilities differ. |
| Missing VMAF | Use the default private video build or supply a libvmaf-enabled FFmpeg earlier in the worker PATH. |
| Efficiency unavailable | Check sensor availability, interval length, coverage and sampling gaps. Extend the timed workload. |
| Power domains differ | Review sensor documentation before a cross-vendor efficiency conclusion; no automatic ranking is produced. |
| Run queued | Another GPU Lab worker holds the lock for this GPU. |
| Browser was closed | Reopen Run library; execution is independent. |
| Service exits after logout | Enable user lingering, or use an administrator-managed service. |
| Worker interrupted | Inspect logs; resume creates a new linked run. |

Logs and snapshots live under `$GPU_LAB_HOME/runs/RUN_ID/`. Back up the entire data directory while the app/workers are stopped, or use SQLite's backup facilities; do not copy only an active database file and omit its WAL.

## Development, verification and extension

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m unittest discover -s tests -v
node --check gpulab/static/app.js
bash -n install.sh scripts/build-video.sh
clang -x cl -cl-std=CL1.2 -fsyntax-only gpulab/kernels/core.cl
python3 scripts/make-release.py
```

The application tests cover durable demonstration execution, metadata provenance, interrupted workers, energy integration/coverage rejection, conservative comparisons, device selection, HTTP authentication/origin checks and PDF/CSV/HTML generation. They do **not** validate GPU performance. See [hardware qualification](docs/HARDWARE-VALIDATION.md) before publishing comparative results.

Architecture:

```text
CLI / Web UI
      ↓
Shared catalog + preflight + SQLite journal
      ↓
Detached supervisor → one subprocess per test
      ├── OpenCL portable GPU kernels
      ├── isolated CUDA / ROCm matrix environments
      └── explicitly mapped FFmpeg hardware video
      ↓
Timestamped GPU telemetry + completed test records
      ↓
Run library / comparison / PDF / HTML / CSV / JSON
```

`catalog.py` defines workload IDs and methods; `devices.py`/`probe.py` map identities; `opencl_backend.py`, `torch_backend.py` and `video.py` execute workloads; `runner.py` owns durability/telemetry; `store.py` owns provenance; `reports.py` and `static/` present the same results. `server.py` exposes the authenticated REST API.

Intel extension: qualify its OpenCL and VAAPI device mapping, then add/validate XPU matrix and telemetry paths using the same result schema. The existing XPU probe is experimental. A production Intel support claim requires physical validation; nothing falls back to CPU.

Not implemented: AI-model benchmarks, graphics/ray tracing, sparse/FP64/FP8 math, multi-GPU communication, power-limit sweeps, calibrated board power, automatic equal-quality encoder search, per-frame video latency, HDR/10-bit testing, scheduled runs or multi-user role management. These are explicit extensions, not hidden capabilities.

License: MIT for GPU Lab. Dependencies and vendor drivers retain their own licenses. The app uses only local storage and does not upload benchmark data.
