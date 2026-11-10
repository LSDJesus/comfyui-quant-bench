#!/usr/bin/env bash
# ComfyUI quantization bench: the community kit. Plans, downloads, runs and
# packages the fit-and-speed pass for one GPU. Safe to re-run: downloads and
# finished runs are skipped.
#
#   ./run.sh                          everything for this card
#   ./run.sh --families zimage,sdxl   only some model families (less to download)
#   ./run.sh --gpu 1                  pick a GPU without being asked
#   ./run.sh --name yourhandle        how to credit you on the results page (default: ask)
#   ./run.sh --yes                    don't ask before downloading
#   ./run.sh --clean                  delete the downloaded models (keeps results/)
set -euo pipefail
cd "$(dirname "$0")"

FAMILIES=""; GPU=""; NAME=""; NAME_SET=0; YES=0; CLEAN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --families) FAMILIES="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --name) NAME="$2"; NAME_SET=1; shift 2 ;;
    --yes|-y) YES=1; shift ;;
    --clean) CLEAN=1; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "Unknown option $1 (see ./run.sh --help)"; exit 1 ;;
  esac
done

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
die() { printf '\n\033[31m%s\033[0m\n' "$*" >&2; exit 1; }
ask() { [ "$YES" = 1 ] && return 0; read -rp "$1 [Y/n] " r; [ -z "$r" ] || [[ "$r" =~ ^[Yy] ]]; }
cfg() { sed -n "s/.*\"$1\": *\"\{0,1\}\([^\",]*\).*/\1/p" kit/config.json; }

if [ "$CLEAN" = 1 ]; then
  D="${QBENCH_DATA:-./data}"
  ask "Delete the downloaded models in $D (results/ stays)?" && rm -rf "$D"/cold "$D"/flash/models "$D"/flash/downloads/_quant_bench/{bakes,sources,cond,ref} "$D"/.hf "$D"/.cache && echo "Deleted."
  exit 0
fi

# ── 1. prerequisites ─────────────────────────────────────────────────────────
command -v docker >/dev/null || die "Docker isn't installed. Linux: https://docs.docker.com/engine/install/  Windows: Docker Desktop with the WSL2 backend."
docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is missing (the 'docker compose' command)."
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: install the NVIDIA driver first."
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
[ "${DRIVER%%.*}" -ge 580 ] || die "NVIDIA driver $DRIVER is too old: this kit uses CUDA 13, which needs driver 580 or newer."
WSL=0; grep -qi microsoft /proc/version 2>/dev/null && WSL=1
if [ "$WSL" = 1 ] && [[ "$(pwd -P)" == /mnt/[a-z]/* ]]; then
  die "This folder is on a Windows drive ($(pwd -P)). Clone it inside WSL instead (e.g. ~/comfyui-quant-bench): the Windows-drive bridge is ~20x slower and would skew the load times."
fi

# ── 2. which GPU ─────────────────────────────────────────────────────────────
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader,nounits)
if [ -z "$GPU" ]; then
  if [ "${#GPUS[@]}" -eq 1 ]; then GPU=0; else
    bold "GPUs in this machine:"; printf '  %s\n' "${GPUS[@]}"
    read -rp "Which index should be tested? " GPU
  fi
fi
LINE=${GPUS[$GPU]:-}; [ -n "$LINE" ] || die "No GPU with index $GPU."
GPU_NAME=$(echo "$LINE" | cut -d, -f2 | sed 's/^ *//')
GPU_UUID=$(echo "$LINE" | cut -d, -f4 | tr -d ' ')
DATA="${QBENCH_DATA:-./data}"; export QBENCH_DATA="$DATA"
mkdir -p "$DATA" results
IDFILE="$DATA/.kit_id_${GPU_UUID}"
if [ -f "$IDFILE" ]; then ID=$(cat "$IDFILE"); else
  SLUG=$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | sed -E 's/nvidia|geforce|laptop gpu//g; s/[^a-z0-9]+//g')
  ID="${SLUG}-$(od -An -N2 -tx1 /dev/urandom | tr -d ' \n')"; echo "$ID" > "$IDFILE"
fi
bold "Testing GPU $GPU: $GPU_NAME (results id $ID)"

if [ "$NAME_SET" = 0 ] && [ "$YES" = 0 ] && [ ! -f "$DATA/.contributor.json" ]; then
  read -rp "Name or handle to credit on the results page (Enter for anonymous): " NAME
fi
CPU=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ *//' || true)
OS=$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -sr)
PLATFORM=$([ "$WSL" = 1 ] && echo "Windows (WSL2)" || echo "Linux")
if [ ! -f "$DATA/.contributor.json" ] || [ "$NAME_SET" = 1 ]; then
  printf '{"name": "%s", "host_platform": "%s", "os": "%s", "cpu": "%s", "kit": %s}\n' \
    "${NAME//\"/}" "$PLATFORM" "${OS//\"/}" "${CPU//\"/}" "$(cfg kit_version)" > "$DATA/.contributor.json"
fi

export QBENCH_GPU="$GPU" QBENCH_FAST_DISK="$WSL" QBENCH_UID="$(id -u)" QBENCH_GID="$(id -g)"
RUN=(docker compose run --rm bench)

# ── 3. the container (first time: ~10 GB, a few minutes) ─────────────────────
bold "Building the container image (same ComfyUI, torch and packages as the published results)..."
docker compose build

# ── 4. plan + download ──────────────────────────────────────────────────────
bold "Planning..."
"${RUN[@]}" python /opt/kit/plan.py --families "$FAMILIES"
NEED=$(cat "$DATA/.plan_need")
FREE=$(df -B1 --output=avail "$DATA" | tail -1 | tr -d ' ')
[ "$NEED" -lt "$FREE" ] || die "Not enough disk space here: need $((NEED / 1000000000)) GB more, $((FREE / 1000000000)) GB free. Try --families to test fewer models."
if [ "$NEED" -gt 0 ]; then
  ask "Download $((NEED / 1000000000)) GB of model files into $DATA?" || die "Stopped before downloading."
  "${RUN[@]}" python /opt/kit/fetch.py
fi

# ── 5. run ──────────────────────────────────────────────────────────────────
bold "Running the bench. Close games, video and anything else using this GPU, and leave the PC alone: the desktop and other apps share the card and slow it down."
[ "$WSL" = 1 ] && echo "WSL2 gets half your RAM by default; models that don't fit the card stream from that RAM. More: https://learn.microsoft.com/windows/wsl/wsl-config"
MODELS_ARG=(); [ -n "$FAMILIES" ] && MODELS_ARG=(--models "$FAMILIES")
"${RUN[@]}" python /opt/bench/fit.py --gpu "$ID" --no-te "${MODELS_ARG[@]}" || echo "fit.py stopped early; packaging what finished (run ./run.sh again to continue)."

# ── 6. package ──────────────────────────────────────────────────────────────
"${RUN[@]}" python /opt/kit/package.py --gpu "$ID"
bold "Done. Send results/qbench-$ID.zip back:"
echo "  open $(cfg results_url)"
echo "  and drag the zip into the form. Thank you!"
echo "Free the disk space afterwards with: ./run.sh --clean"
