#!/usr/bin/env bash
# Private, pinned FFmpeg toolchain; does not replace /usr/bin/ffmpeg.
set -euo pipefail
prefix="${1:?Usage: build-video.sh PREFIX}"
if [[ -x "$prefix/bin/ffmpeg" ]] && "$prefix/bin/ffmpeg" -hide_banner -filters 2>/dev/null | grep -q libvmaf; then
  printf 'Using existing private FFmpeg with libvmaf: %s\n' "$prefix/bin/ffmpeg";exit 0
fi
mkdir -p "$prefix/src" "$prefix/lib/pkgconfig" "$prefix/bin"
jobs="$(getconf _NPROCESSORS_ONLN)"
if ((jobs>8)); then jobs=8; fi
export PKG_CONFIG_PATH="$prefix/lib/pkgconfig:$prefix/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"
checkout() {
  local repository="$1" ref="$2" folder="$3"
  if [[ ! -d "$folder/.git" ]]; then git clone --depth 1 --branch "$ref" "$repository" "$folder"; fi
  # Existing source must match the requested tag, rather than silently reusing a different version.
  git -C "$folder" fetch --depth 1 origin "refs/tags/$ref:refs/tags/$ref"
  git -C "$folder" checkout --detach "$ref"
}
checkout https://github.com/Netflix/vmaf.git v3.0.0 "$prefix/src/vmaf"
if [[ ! -d "$prefix/src/vmaf/libvmaf/build" ]]; then
  meson setup "$prefix/src/vmaf/libvmaf/build" "$prefix/src/vmaf/libvmaf" --prefix "$prefix" --libdir lib --buildtype release -Ddefault_library=static -Denable_tests=false -Denable_docs=false
fi
meson compile -C "$prefix/src/vmaf/libvmaf/build" -j "$jobs"
meson install -C "$prefix/src/vmaf/libvmaf/build"
checkout https://github.com/FFmpeg/nv-codec-headers.git n12.2.72.0 "$prefix/src/nv-codec-headers"
make -C "$prefix/src/nv-codec-headers" PREFIX="$prefix" install
checkout https://github.com/FFmpeg/FFmpeg.git n7.1.1 "$prefix/src/ffmpeg"
cd "$prefix/src/ffmpeg"
./configure --prefix="$prefix" --pkg-config-flags=--static --extra-cflags="-I$prefix/include" --extra-ldflags="-L$prefix/lib" \
  --enable-libvmaf --enable-vaapi --enable-nvenc --enable-nvdec --enable-ffnvcodec \
  --disable-doc --disable-debug --disable-ffplay
make -j "$jobs"
make install
"$prefix/bin/ffmpeg" -hide_banner -version
