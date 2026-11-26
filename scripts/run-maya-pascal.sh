#!/usr/bin/env bash
# Hardware-adaptive Pascal launcher for Project Maya (p100-pascal-952-port).
# Wraps ./maya.sh with optimal Pascal (sm_60 / sm_61) GPU & CPU/NUMA env vars
# automatically tuned for ANY Linux PC:
#   - 1 CPU socket (desktop/workstation) OR 2+ CPU sockets (NUMA server)
#   - 1, 2, 3, or 4 NVIDIA Pascal GPUs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPUS="auto"
DRY_RUN=0
MAYA_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./scripts/run-maya-pascal.sh [pascal-options] [-- maya.sh args...]

Automatically configures Pascal (sm_60/sm_61) kernel flags and single- vs
multi-socket NUMA memory policies for your PC, then runs ./maya.sh.

Pascal wrapper options:
  --gpus N|0,1,...|auto  Number of GPUs (1..4) or explicit CUDA indices (default: auto)
  --dry-run              Print the resolved environment and command without executing
  -h, --help             Show this help

Examples:
  # Interactive setup (downloads/packs a model and starts the web UI + API on :8080)
  ./scripts/run-maya-pascal.sh -- --setup

  # Start a previously configured model
  ./scripts/run-maya-pascal.sh -- --config /path/to/maya-config.json --port 8080

  # Run on only 1 or 2 GPUs explicitly
  ./scripts/run-maya-pascal.sh --gpus 1 -- --setup
  ./scripts/run-maya-pascal.sh --gpus 2 -- --config /path/to/maya-config.json

  # Run built-in speed benchmark
  ./scripts/run-maya-pascal.sh -- --bench
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)     GPUS="${2:-}"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        --)         shift; MAYA_ARGS+=("$@"); break ;;
        *)          MAYA_ARGS+=("$1"); shift ;;
    esac
done

# Detect GPUs, NUMA topology, and system RAM
AVAIL_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 1)"
[[ "$AVAIL_GPUS" -lt 1 ]] && AVAIL_GPUS=1
NUMA_NODES="$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l || echo 1)"
RAM_GIB="$(awk '/MemTotal:/ {printf "%.0f", $2/1048576}' /proc/meminfo 2>/dev/null || echo 64)"

CUDA_DEVS=""
NUM_GPUS=1
if [[ "$GPUS" == "auto" ]]; then
    NUM_GPUS="$AVAIL_GPUS"
    [[ "$NUM_GPUS" -gt 4 ]] && NUM_GPUS=4
    CUDA_DEVS="$(seq -s, 0 $((NUM_GPUS - 1)))"
elif [[ "$GPUS" =~ ^[0-9]+$ ]]; then
    NUM_GPUS="$GPUS"
    CUDA_DEVS="$(seq -s, 0 $((NUM_GPUS - 1)))"
else
    CUDA_DEVS="$GPUS"
    NUM_GPUS="$(echo "$CUDA_DEVS" | tr ',' '\n' | wc -l)"
fi

# Choose NUMA policy cleanly for single-CPU desktops vs multi-socket servers
ARENA_NUMA="0"
TP_NUMA="0"
if [[ "$NUMA_NODES" -ge 2 && "$NUM_GPUS" -ge 2 ]]; then
    TP_NUMA="1"
    if [[ "$RAM_GIB" -ge 280 ]]; then
        ARENA_NUMA="mirror"
    else
        ARENA_NUMA="interleave"
    fi
fi

# Enable Pascal-tuned kernels + lossless routing (all portable across 1..4 GPUs & 1..2+ sockets)
ENV_VARS=(
    "CUDA_VISIBLE_DEVICES=${CUDA_DEVS}"
    "CUDA_DEVICE_ORDER=PCI_BUS_ID"
    "CUDA_DEVICE_MAX_CONNECTIONS=32"
    "MAYA_P100_MMVQ=1"
    "MAYA_P100_HFMA2=1"
    "MAYA_MTP_DVOCAB_MAX=16384"
    "MAYA_VERIFY_DEVICE_PLAN=1"
    "MAYA_VERIFY_PIPE=0"
    "MAYA_POOL_FAST=1"
    "MAYA_ARENA_NUMA=${ARENA_NUMA}"
    "MAYA_TP_NUMA=${TP_NUMA}"
    "MAYA_TP_P2P=1"
    "MAYA_GLM_NUMA=local"
    "MAYA_GLM_CPU_SHARED=socket"
    "MAYA_GLM_PASCAL_MOE=1"
    "MAYA_GLM_MV_Q8_P100=1"
    "MAYA_GLM_MV_P100=1"
    "MAYA_GLM_PASCAL_MV=1"
    "MAYA_GLM_PREFILL_PASCAL=1"
    "MAYA_GLM_PREFILL_PASCAL_FP16=1"
    "MAYA_GLM_PREFILL_PASCAL_DENSE=1"
    "MAYA_GLM_STREAM_FETCH=1"
    "MAYA_GLM_ROUTE_FAST=1"
    "MAYA_GLM_PREFETCH_N=1"
    "MAYA_GLM_PF_CE=1"
    "MAYA_GLM_P2P=0"
    "MAYA_GLM_RESIDENT_TOPK=0"
    "MAYA_GLM_MISS_SKIP_W=0"
    "MAYA_GLM_DISK_SKIP_W=0"
)

CMD=(env "${ENV_VARS[@]}" "${SCRIPT_DIR}/maya.sh" "${MAYA_ARGS[@]}")

echo "[maya-pascal] GPUs=${CUDA_DEVS} (${NUM_GPUS}) | NUMA=${NUMA_NODES} node(s) | RAM=${RAM_GIB} GiB | arena_numa=${ARENA_NUMA}"
if [[ "$DRY_RUN" -eq 1 ]]; then
    printf ' %q' "${CMD[@]}"
    echo ""
    exit 0
fi

exec "${CMD[@]}"
