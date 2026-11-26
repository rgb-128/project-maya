#!/usr/bin/env bash
# Universal Pascal (sm_60 / sm_61) installer for Project Maya (p100-pascal-952-port).
# Works on any Linux x86_64 PC: single-CPU desktops or multi-socket servers,
# with 1 to 4+ NVIDIA Pascal GPUs (Tesla P100 sm_60, P40/P4/GTX 10-series/Titan Xp sm_61).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
CUDA_ARCH=""
JOBS="$(nproc 2>/dev/null || echo 4)"
CHECK_ONLY=0
INSTALL_DEPS=0

usage() {
    cat <<'EOF'
Usage: ./install-pascal.sh [options]

Auto-detects your NVIDIA Pascal GPU(s) (sm_60 or sm_61), CUDA 12.x toolkit,
and CPU topology (single-socket or multi-socket), then builds Project Maya
with native Pascal CUDA & host SIMD optimizations enabled.

Options:
  --check             Detect hardware & toolchain only (no build)
  --install-deps      Install missing build tools via apt/dnf/pacman (uses sudo)
  --arch 60|61        Override CUDA SM architecture (60=P100, 61=P40/P4/1080Ti)
  --build-dir DIR     Custom CMake build directory (default: ./build)
  --jobs N            Parallel compile jobs (default: nproc)
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)        CHECK_ONLY=1; shift ;;
        --install-deps) INSTALL_DEPS=1; shift ;;
        --arch)         CUDA_ARCH="${2:-}"; shift 2 ;;
        --build-dir)    BUILD_DIR="${2:-}"; shift 2 ;;
        --jobs)         JOBS="${2:-}"; shift 2 ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

echo "======================================================================"
echo "  Project Maya (p100-pascal-952-port) Pascal Installer"
echo "======================================================================"

# 1. Detect CPU / RAM / NUMA topology
SOCKETS="$(lscpu 2>/dev/null | awk -F: '/^Socket\(s\):/ {gsub(/ /,"",$2); print $2}' || echo 1)"
[[ -z "$SOCKETS" ]] && SOCKETS=1
CORES="$(nproc 2>/dev/null || echo 4)"
RAM_GIB="$(awk '/MemTotal:/ {printf "%.0f", $2/1048576}' /proc/meminfo 2>/dev/null || echo "?")"
NUMA_NODES="$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l || echo 1)"
SIMD="SSE/Scalar"
if grep -qE '\bavx512f\b' /proc/cpuinfo 2>/dev/null; then SIMD="AVX-512"
elif grep -qE '\bavx2\b' /proc/cpuinfo 2>/dev/null; then SIMD="AVX2"
elif grep -qE '\bavx\b' /proc/cpuinfo 2>/dev/null; then SIMD="AVX1+F16C"
fi

echo "[CPU]  ${SOCKETS} socket(s), ${CORES} logical thread(s), ${NUMA_NODES} NUMA node(s), ${SIMD}, ${RAM_GIB} GiB RAM"

# 2. Detect NVIDIA Pascal GPUs
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERR]  nvidia-smi not found. Install NVIDIA driver 580.x (last branch supporting Pascal)." >&2
    exit 1
fi

GPU_LIST="$(nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || true)"
if [[ -z "$GPU_LIST" ]]; then
    echo "[ERR]  No NVIDIA GPUs detected by nvidia-smi." >&2
    exit 1
fi

GPU_COUNT=0
HAS_60=0
HAS_61=0
TOTAL_VRAM_MIB=0
echo "[GPUs]"
while IFS=',' read -r idx name cc mem; do
    idx="$(echo "$idx" | xargs)"
    name="$(echo "$name" | xargs)"
    cc="$(echo "$cc" | xargs)"
    mem_num="$(echo "$mem" | grep -oE '[0-9]+' | head -n1 || echo 0)"
    TOTAL_VRAM_MIB=$((TOTAL_VRAM_MIB + mem_num))
    GPU_COUNT=$((GPU_COUNT + 1))
    echo "       GPU ${idx}: ${name} (compute_cap ${cc}, ${mem_num} MiB)"
    [[ "$cc" == "6.0" ]] && HAS_60=1
    [[ "$cc" == "6.1" ]] && HAS_61=1
done <<< "$GPU_LIST"

echo "[VRAM] ${GPU_COUNT} GPU(s), $((TOTAL_VRAM_MIB / 1024)) GiB total VRAM"

if [[ -z "$CUDA_ARCH" ]]; then
    if [[ "$HAS_60" -eq 1 && "$HAS_61" -eq 0 ]]; then
        CUDA_ARCH="60"
    elif [[ "$HAS_61" -eq 1 && "$HAS_60" -eq 0 ]]; then
        CUDA_ARCH="61"
    else
        CUDA_ARCH="60"
    fi
fi

# 3. Locate CUDA 12.x nvcc
NVCC_BIN=""
find_cuda12() {
    local candidates=()
    command -v nvcc >/dev/null 2>&1 && candidates+=("$(command -v nvcc)")
    for d in /usr/local/cuda-12*/bin/nvcc /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc; do
        [[ -x "$d" ]] && candidates+=("$d")
    done
    for c in "${candidates[@]}"; do
        local ver
        ver="$("$c" --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}' || true)"
        if [[ "$ver" == 12.* ]]; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

if NVCC_BIN="$(find_cuda12)"; then
    CUDA_VER="$("$NVCC_BIN" --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
    CUDA_ROOT="$(cd "$(dirname "$NVCC_BIN")/.." && pwd)"
    echo "[CUDA] Found CUDA ${CUDA_VER}: ${NVCC_BIN}"
else
    echo "[ERR]  CUDA 12.x (nvcc) not found!" >&2
    echo "       Pascal (sm_60/sm_61) requires CUDA Toolkit 12.x (e.g., 12.4–12.9) and NVIDIA driver 580.x." >&2
    exit 1
fi

# 4. Optional build dependency installation
if [[ "$INSTALL_DEPS" -eq 1 ]]; then
    echo "[DEPS] Installing build dependencies..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y build-essential cmake ninja-build git python3 libnuma-dev
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y gcc gcc-c++ cmake ninja-build git python3 numactl-devel
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --needed base-devel cmake ninja git python numactl
    else
        echo "[WARN] Unsupported package manager; please install cmake >= 3.24, g++, ninja (or make), git, python3."
    fi
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "[ERR]  cmake not found. Re-run with --install-deps or install cmake >= 3.24." >&2
    exit 1
fi

GEN_ARGS=()
if command -v ninja >/dev/null 2>&1; then
    GEN_ARGS=(-G Ninja)
    echo "[GEN]  Ninja"
else
    echo "[GEN]  Unix Makefiles"
fi

echo "[PLAN] Arch: sm_${CUDA_ARCH} | Build dir: ${BUILD_DIR} | Jobs: ${JOBS}"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo "[OK]   Hardware & toolchain check passed (--check mode; skipping build)."
    exit 0
fi

# 5. Initialize submodules if needed & build
if [[ -f "${SCRIPT_DIR}/.gitmodules" ]] && [[ ! -f "${SCRIPT_DIR}/third_party/llama.cpp/CMakeLists.txt" ]]; then
    echo "[GIT]  Initializing submodules..."
    git -C "${SCRIPT_DIR}" submodule update --init --recursive
fi

export PATH="${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"

cmake "${GEN_ARGS[@]}" \
    -S "${SCRIPT_DIR}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${NVCC_BIN}" \
    -DSTRATA_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DSTRATA_NATIVE_EXPERTS=ON \
    -DSTRATA_BUILD_TESTS=OFF

cmake --build "${BUILD_DIR}" --target maya -j"${JOBS}"

echo ""
echo "======================================================================"
echo "  Build Complete!"
echo "======================================================================"
echo "  Binary: ${BUILD_DIR}/maya"
echo ""
echo "  Quick start (auto-adapts to ${SOCKETS} CPU socket(s) & ${GPU_COUNT} Pascal GPU(s)):"
echo "    ./scripts/run-maya-pascal.sh -- --setup"
echo "    ./scripts/run-maya-pascal.sh -- --config /path/to/maya-config.json"
echo "======================================================================"
