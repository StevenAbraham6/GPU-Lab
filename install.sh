#!/usr/bin/env bash
# One-command installer. Kernel drivers remain an explicit host prerequisite.
set -euo pipefail
GPU_LAB_REPO="${GPU_LAB_REPO:-https://github.com/StevenAbraham6/GPU-Lab.git}"
GPU_LAB_REF="${GPU_LAB_REF:-main}"
GPU_LAB_PREFIX="${GPU_LAB_PREFIX:-$HOME/.local/share/gpu-lab-app}"
GPU_LAB_DATA="${GPU_LAB_HOME:-$HOME/.local/share/gpu-lab}"
GPU_LAB_BIN="${GPU_LAB_BIN:-$HOME/.local/bin}"
backend=auto
service=0
video_build=1
app_only=0
system_packages=1
usage() {
  cat <<'EOF'
GPU Lab installer
  bash install.sh [--backend auto|nvidia|amd|both|none] [--service]
                  [--app-only] [--skip-video-build] [--no-system-packages]
  --service             Start the UI as a persistent user systemd service.
  --app-only            UI/reports only; no GPU runtimes or video compilation.
  --skip-video-build    Use distribution FFmpeg (VMAF may be unavailable).
  --no-system-packages  Prerequisites already supplied; do not run apt.
Variables: GPU_LAB_PREFIX, GPU_LAB_HOME, GPU_LAB_REPO, GPU_LAB_REF,
           GPU_LAB_NVIDIA_INDEX, GPU_LAB_NVIDIA_TORCH,
           GPU_LAB_AMD_INDEX, GPU_LAB_AMD_TORCH.
EOF
}
while (($#)); do
  case "$1" in
    --backend) backend="${2:?Missing backend}"; shift 2;;
    --service) service=1; shift;;
    --app-only) app_only=1; backend=none; video_build=0; shift;;
    --skip-video-build) video_build=0; shift;;
    --no-system-packages) system_packages=0; shift;;
    -h|--help) usage; exit 0;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2;;
  esac
done
case "$backend" in auto|nvidia|amd|both|none);; *) usage; exit 2;; esac
if [[ "$app_only" == 0 && "$(uname -s)" != Linux ]]; then
  printf 'GPU execution targets Linux. On macOS use: bash install.sh --app-only\n' >&2; exit 2
fi
if [[ "$system_packages" == 1 && "$(uname -s)" == Linux ]]; then
  if ! command -v apt-get >/dev/null; then
    printf 'Automatic host prerequisites target Ubuntu/Debian. See README for manual setup.\n' >&2;exit 2
  fi
  elevate=()
  if [[ $EUID != 0 ]]; then elevate=(sudo); fi
  "${elevate[@]}" apt-get update
  "${elevate[@]}" apt-get install -y python3 python3-venv python3-pip git curl ca-certificates pciutils \
    build-essential pkg-config ocl-icd-libopencl1 ocl-icd-opencl-dev clinfo \
    ffmpeg vainfo mesa-va-drivers libva-dev libdrm-dev libnuma1
  if [[ "$video_build" == 1 ]]; then
    "${elevate[@]}" apt-get install -y meson ninja-build nasm yasm
  fi
fi
if [[ "$backend" == auto ]]; then
  have_nvidia=0;have_amd=0
  if command -v lspci >/dev/null; then
    if lspci -Dn | awk '$2 ~ /^03/ && $3 ~ /^10de:/ {found=1} END{exit !found}'; then have_nvidia=1; fi
    if lspci -Dn | awk '$2 ~ /^03/ && $3 ~ /^1002:/ {found=1} END{exit !found}'; then have_amd=1; fi
  fi
  if [[ "$have_nvidia" == 1 && "$have_amd" == 1 ]]; then backend=both
  elif [[ "$have_nvidia" == 1 ]]; then backend=nvidia
  elif [[ "$have_amd" == 1 ]]; then backend=amd
  else backend=none; fi
fi
script_dir=""
if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
mkdir -p "$GPU_LAB_PREFIX" "$GPU_LAB_DATA" "$GPU_LAB_BIN"
if [[ -n "$script_dir" && -f "$script_dir/pyproject.toml" ]]; then
  source_dir="$script_dir"
else
  source_dir="$GPU_LAB_PREFIX/source"
  if [[ -d "$source_dir/.git" ]]; then
    git -C "$source_dir" fetch origin "$GPU_LAB_REF"
    git -C "$source_dir" checkout --detach FETCH_HEAD
  else
    git clone --depth 1 --branch "$GPU_LAB_REF" "$GPU_LAB_REPO" "$source_dir"
  fi
fi
python_bin="${GPU_LAB_PYTHON:-python3}"
"$python_bin" -c 'import sys; assert (3,10) <= sys.version_info[:2], "Python >=3.10 required"'
"$python_bin" -m venv "$GPU_LAB_PREFIX/venv"
py="$GPU_LAB_PREFIX/venv/bin/python"
"$py" -m pip install --upgrade pip wheel setuptools
extra=""
if [[ "$app_only" == 0 ]]; then extra='[opencl]'; fi
"$py" -m pip install "$source_dir$extra"
# Install the app package into its own venv. Both math stacks coexist in separate environments.
for vendor in nvidia amd; do
  if [[ "$backend" != "$vendor" && "$backend" != both ]]; then continue; fi
  runtime="$GPU_LAB_DATA/backends/$vendor"
  "$python_bin" -m venv "$runtime"
  "$runtime/bin/python" -m pip install --upgrade pip
  "$runtime/bin/python" -m pip install numpy
  if [[ "$vendor" == nvidia ]]; then
    "$runtime/bin/python" -m pip install --index-url "${GPU_LAB_NVIDIA_INDEX:-https://download.pytorch.org/whl/cu128}" "${GPU_LAB_NVIDIA_TORCH:-torch==2.7.1}"
  else
    "$runtime/bin/python" -m pip install --index-url "${GPU_LAB_AMD_INDEX:-https://stable.repo.amd.com/rocm/whl-next/}" "${GPU_LAB_AMD_TORCH:-torch[device-all]==2.14.0+rocm10.1.0}"
  fi
done
if [[ "$video_build" == 1 && "$app_only" == 0 ]]; then
  bash "$source_dir/scripts/build-video.sh" "$GPU_LAB_PREFIX/tools"
fi
# Quote executable paths with Python rather than interpolating shell source.
"$py" - "$GPU_LAB_PREFIX" "$GPU_LAB_DATA" "$GPU_LAB_BIN/gpu-lab" <<'PY'
import pathlib,shlex,sys
prefix,data,dest=map(pathlib.Path,sys.argv[1:])
q=shlex.quote
text='#!/usr/bin/env bash\nset -e\n'
text+='export GPU_LAB_HOME="${GPU_LAB_HOME:-'+str(data).replace('"','\\"').replace('$','\\$').replace('`','\\`')+'}"\n'
text+='export PATH='+q(str(prefix/'tools/bin'))+':"$PATH"\n'
text+='exec '+q(str(prefix/'venv/bin/python'))+' -m gpulab "$@"\n'
dest.write_text(text);dest.chmod(0o755)
PY
if [[ "$service" == 1 ]]; then
  if ! command -v systemctl >/dev/null; then printf 'systemd is unavailable; run gpu-lab serve manually.\n' >&2;exit 2;fi
  mkdir -p "$HOME/.config/systemd/user"
  "$py" - "$HOME/.config/systemd/user/gpu-lab.service" "$GPU_LAB_BIN/gpu-lab" <<'PY'
from pathlib import Path
import sys
exe=sys.argv[2].replace('\\','\\\\').replace('"','\\"').replace('%','%%')
Path(sys.argv[1]).write_text('[Unit]\nDescription=GPU Lab web workspace\nAfter=network.target\n\n[Service]\nType=simple\nExecStart="'+exe+'" serve --host 127.0.0.1 --port 8080\nRestart=on-failure\nRestartSec=3\nKillMode=process\n\n[Install]\nWantedBy=default.target\n')
PY
  systemctl --user daemon-reload
  systemctl --user enable --now gpu-lab
fi
"$GPU_LAB_BIN/gpu-lab" devices
printf '\nGPU Lab installed. Backend selection: %s\n' "$backend"
printf 'Launch: %s serve\n' "$GPU_LAB_BIN/gpu-lab"
printf 'CLI:    %s --help\n' "$GPU_LAB_BIN/gpu-lab"
printf 'Open:   http://127.0.0.1:8080 (SSH tunnel for a remote server)\n'
printf 'Data:   %s\n' "$GPU_LAB_DATA"
printf '\nKernel drivers / OpenCL ICDs are host prerequisites, not replaced by this installer.\n'
printf 'Run gpu-lab doctor --gpu GPU_ID to check this machine before benchmarking.\n'
printf 'For a systemd user service to survive logout: sudo loginctl enable-linger "$USER"\n'
