# Verification record - v0.1.0

Validated in the development workspace on macOS ARM64 using Python 3.12:

- 23 automated application tests passed, including queued cancellation, execution after HTTP server shutdown, report downloads, metadata provenance, imported suite-version preservation, energy integration and conservative comparison behavior.
- Python modules compiled; JavaScript and installer shell syntax checks passed.
- OpenCL C 1.2 source passed a Clang syntax check. This is not an OpenCL runtime or performance qualification.
- The one-command installer completed in a clean app-only virtual environment with isolated installation/data/bin directories.
- A source release was extracted, installed and launched outside the repository directory to check relocation and package resources.
- Browser checks covered mobile/desktop layout, historical reports, metadata editing, run reproduction, preflight findings and pairwise comparison.
- The synthetic PDF report was rendered and visually inspected, including its summary, charts, table pagination and methodology/configuration sections.

Not executed in this environment:

- Physical NVIDIA L4, AMD R9700 or AMD R9600 benchmarks.
- Linux apt/service installation, CUDA/ROCm wheel execution or the Linux FFmpeg/VMAF compilation path.
- GPU power-domain calibration or driver-specific hardware codec verification.

The first server deployment must follow [HARDWARE-VALIDATION.md](HARDWARE-VALIDATION.md). Synthetic fixtures are not performance evidence. Source distribution uses the configured GitHub repository and the local release archive; publishing the source does not qualify GPU hardware execution.
