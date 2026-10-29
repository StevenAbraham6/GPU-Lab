# Hardware qualification checklist

Application tests and a synthetic demonstration do not establish GPU support. Record validation evidence for each actual OS/kernel/driver/runtime/GPU combination before interpreting comparative reports.

## Qualification matrix

| Target | Implementation path | Physical qualification |
|---|---|---|
| NVIDIA L4 | OpenCL GPU, CUDA PyTorch, NVENC/NVDEC, NVIDIA telemetry | Pending |
| AMD Radeon AI PRO R9700 | OpenCL GPU, ROCm PyTorch, VAAPI, hwmon telemetry | Pending |
| AMD Radeon AI PRO R9600 | OpenCL GPU, ROCm PyTorch, VAAPI, hwmon telemetry | Pending |
| Future Intel GPU | OpenCL/VAAPI, experimental XPU mapping, partial generic hwmon discovery | Pending; not a supported production target |

## Procedure

1. Install on a vendor-supported host. Save `gpu-lab devices`, `uname -a`, driver/runtime versions and permissions.
2. Run `gpu-lab doctor all --gpu GPU_ID`. Review every finding. Resolve mappings and dependencies before recording a baseline.
3. Run each group with the quick profile. Check GPU processes/utilization externally; confirm only the selected GPU is active.
4. Verify arithmetic and VRAM checker self-tests pass. Review GEMM precision/accumulation settings and CPU reference errors. If an INT8 path is missing, record unsupported; do not reinterpret another test as an INT8 result.
5. Confirm memory/transfer byte counts and actual allocation sizes. Use the same sizes for all cards; keep capacity sweeps separate.
6. Confirm hardware video device selection. Review FFmpeg commands/logs, output frame counts, reference pixel hash and PSNR/SSIM/VMAF. The built-in moving test pattern is calibration content, not representative video coverage.
7. Run standard measurements at least three times under stable conditions. Keep ambient/cooling notes, power policy, PCIe configuration and competing activity in metadata.
8. Verify telemetry sensor documentation: whole-board vs chip/package, averaging cadence, units and scope. Cross-vendor domain mismatches must not receive an automatic efficiency winner.
9. During a long run, close the browser and restart only the UI service; confirm the run continues. Test cancellation, then test a supervisor termination on a disposable run. Confirm no benchmark process tree remains active and completed groups survive.
10. Export PDF/HTML/CSV/JSON. Check the PDF layout and direction labels. Import the JSON into another workspace and confirm measured values/provenance remain intact.
11. Compare two runs with identical settings and two deliberately mismatched runs. Check that incompatible rows are visible without a ranking.
12. Record tested, unsupported and untested capabilities explicitly. Do not mark an entire vendor qualified based on one successful kernel.

## Measurement limitations to retain in published results

- Results reflect hardware plus driver/runtime/library stack.
- GPU telemetry is an estimate. Sensor domains and sampling differ across vendors.
- Host launch/transfer/video pipeline results can be CPU, NUMA, PCIe or I/O limited.
- Native matrix libraries and video presets differ; method/configuration disclosure is mandatory.
- Thermal stop behavior depends on a readable sensor. User-defined ceilings are not vendor certifications.
- VRAM checks cover allocated memory only; sampled numerical checks do not prove every output correct.
- Separate run continuations have separate thermal states and timestamps; do not silently merge them.
