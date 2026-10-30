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
update_app=0
network_service=0
open_network_service=0
usage() {
  cat <<'EOF'
GPU Lab installer
  bash install.sh [--backend auto|nvidia|amd|both|none] [--service]
                  [--app-only] [--update-app] [--skip-video-build] [--no-system-packages]
  --service             Start the UI as a persistent user systemd service.
  --network-service     Configure the service for authenticated network access (implies --service).
  --open-network-service Configure network access without a token (implies --service).
  --app-only            UI/reports only; no GPU runtimes or video compilation.
  --update-app          Update only the app and browser-update setup; keep GPU runtimes/tools.
  --skip-video-build    Use distribution FFmpeg (VMAF may be unavailable).
  --no-system-packages  Prerequisites already supplied; do not run apt.
Variables: GPU_LAB_PREFIX, GPU_LAB_HOME, GPU_LAB_REPO, GPU_LAB_REF, GPU_LAB_UPDATE_BRANCH,
           GPU_LAB_NVIDIA_INDEX, GPU_LAB_NVIDIA_TORCH,
           GPU_LAB_AMD_INDEX, GPU_LAB_AMD_TORCH.
EOF
}
while (($#)); do
  case "$1" in
    --backend) backend="${2:?Missing backend}"; shift 2;;
    --service) service=1; shift;;
    --network-service) service=1; network_service=1; shift;;
    --open-network-service) service=1; open_network_service=1; shift;;
    --app-only) app_only=1; backend=none; video_build=0; shift;;
    --skip-video-build) video_build=0; shift;;
    --no-system-packages) system_packages=0; shift;;
    --update-app) update_app=1; shift;;
    -h|--help) usage; exit 0;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2;;
  esac
done
if [[ "$update_app" == 1 ]]; then backend=none;video_build=0;system_packages=0;fi
if [[ "$network_service" == 1 && "$open_network_service" == 1 ]]; then
  printf 'Choose either --network-service or --open-network-service.\n' >&2; exit 2
fi
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
if [[ "$update_app" == 1 ]]; then
  extra=""
  if "$py" -c 'import importlib.util; assert importlib.util.find_spec("pyopencl") is not None' 2>/dev/null; then extra='[opencl]';fi
fi
# During the one-time upgrade, avoid replacing packages underneath a live service.
service_stopped=0
restore_service() {
  if [[ "$service_stopped" == 1 ]]; then systemctl --user start gpu-lab || true;fi
}
trap restore_service EXIT
if [[ "$update_app" == 1 && "$service" == 1 ]] && command -v systemctl >/dev/null && systemctl --user is-active --quiet gpu-lab; then
  systemctl --user stop gpu-lab
  service_stopped=1
fi
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
"$py" - "$GPU_LAB_PREFIX" "$GPU_LAB_DATA" "$GPU_LAB_BIN/gpu-lab" "$source_dir" "$service" "$extra" "${GPU_LAB_UPDATE_BRANCH:-main}" <<'PY'
import json,pathlib,shlex,subprocess,sys
prefix,data,dest,source=map(lambda value:pathlib.Path(value).expanduser().absolute(),sys.argv[1:5])
deployment=prefix/'deployment.json'
previous=json.loads(deployment.read_text()) if deployment.exists() else {}
commit=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],capture_output=True,text=True)
configuration=dict(source=str(source.resolve()),python=str(prefix/'venv/bin/python'),updater_python=str(pathlib.Path(sys._base_executable).resolve()),
                   branch=sys.argv[7],service='gpu-lab.service' if sys.argv[5]=='1' else previous.get('service'),
                   opencl=bool(sys.argv[6]),installed_commit=commit.stdout.strip() if commit.returncode==0 else None,installation_uncertain=False)
deployment.write_text(json.dumps(configuration,indent=2)+'\n');deployment.chmod(0o600)
q=shlex.quote
text='#!/usr/bin/env bash\nset -e\n'
text+='export GPU_LAB_HOME="${GPU_LAB_HOME:-'+str(data).replace('"','\\"').replace('$','\\$').replace('`','\\`')+'}"\n'
text+='export GPU_LAB_DEPLOYMENT='+q(str(deployment))+'\n'
text+='export PATH='+q(str(prefix/'tools/bin'))+':"$PATH"\n'
# Isolate imports from the current checkout while retaining relative CLI file paths.
text+='exec '+q(str(prefix/'venv/bin/python'))+' -I -m gpulab "$@"\n'
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
  if [[ "$network_service" == 1 ]]; then
    mkdir -p "$HOME/.config/systemd/user/gpu-lab.service.d"
    "$py" - "$GPU_LAB_PREFIX/network-token" "$HOME/.config/systemd/user/gpu-lab.service.d/network.conf" "$GPU_LAB_BIN/gpu-lab" <<'PY'
import os, pathlib, secrets, sys
token, override, exe = map(lambda value: pathlib.Path(value).expanduser().absolute(), sys.argv[1:])
try:
    fd = os.open(token, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
except FileExistsError:
    if not token.is_file() or not token.read_text().strip():
        raise SystemExit('Existing network token is empty or not a regular file; resolve it before proceeding.')
    token.chmod(0o600)
else:
    with os.fdopen(fd, 'w') as out:
        out.write(secrets.token_urlsafe(32) + '\n')
def quote(value):
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'
override.write_text('[Service]\nExecStart=\nExecStart=' + quote(exe) + ' serve --host 0.0.0.0 --port 8080 --token-file ' + quote(token) + '\n')
PY
    printf 'Network access enabled on port 8080. Token file: %s/network-token\n' "$GPU_LAB_PREFIX"
  fi
  if [[ "$open_network_service" == 1 ]]; then
    mkdir -p "$HOME/.config/systemd/user/gpu-lab.service.d"
    "$py" - "$HOME/.config/systemd/user/gpu-lab.service.d/network.conf" "$GPU_LAB_BIN/gpu-lab" <<'PY'
from pathlib import Path
import sys
exe = sys.argv[2].replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%')
Path(sys.argv[1]).write_text('[Service]\nExecStart=\nExecStart="' + exe + '" serve --host 0.0.0.0 --port 8080 --open-access\n')
PY
    printf 'Network access enabled on port 8080 without authentication.\n'
  fi
  systemctl --user daemon-reload
  systemctl --user enable --now gpu-lab
  systemctl --user restart gpu-lab
  service_stopped=0
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
