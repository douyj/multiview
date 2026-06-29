"""
scripts/run_singleview_v18_training.sh --overwrite

scripts/run_singleview_v18_training.sh \
  --test-views 12,15,18,21,24 \
  --overwrite
"""

#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-3}"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-CT}"

SINO_CONFIG="${SINO_CONFIG:-configs/sino_singleview_v18.yaml}"
IMAGE_CONFIG="${IMAGE_CONFIG:-configs/image_singleview_fista_v18.yaml}"

SINO_ROOT="${SINO_ROOT:-./multiview_dataset/sino_singleview_v18}"
FISTA_ROOT="${FISTA_ROOT:-./multiview_dataset/fista_singleview_v18}"

TRAIN_SPLITS="${TRAIN_SPLITS:-train,valid,test}"
TRAIN_VIEW="${TRAIN_VIEW:-18}"
TEST_VIEWS="${TEST_VIEWS:-}"

SINO_BATCH_SIZE="${SINO_BATCH_SIZE:-8}"
FISTA_BATCH_SIZE="${FISTA_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FISTA_NUM_LAYERS="${FISTA_NUM_LAYERS:-7}"
FISTA_CKPT="${FISTA_CKPT:-}"

SKIP_SINO_TRAIN=0
SKIP_SINO_GEN=0
SKIP_FISTA=0
SKIP_IMAGE_TRAIN=0
OVERWRITE=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_singleview_v18_training.sh [options]

This runs the fixed-view18 training pipeline:
  1. Train sino-domain view18 model
  2. Generate view18 sino intermediate dataset
  3. Generate view18 FISTA CT dataset
  4. Train image-domain view18 model
  5. Optionally test different views with the trained view18 models

Options:
  --device ID              CUDA_VISIBLE_DEVICES value. Default: 3
  --train-view N           Fixed training view. Default: 18
  --train-splits SPLITS    Splits for generated training data. Default: train,valid,test
  --test-views VIEWS       Optional cross-view test after training, e.g. 12,15,18,21,24

  --sino-config PATH       Default: configs/sino_singleview_v18.yaml
  --image-config PATH      Default: configs/image_singleview_fista_v18.yaml
  --sino-root PATH         Default: ./multiview_dataset/sino_singleview_v18
  --fista-root PATH        Default: ./multiview_dataset/fista_singleview_v18

  --sino-batch-size N      Default: 8
  --fista-batch-size N     Default: 2
  --num-workers N          Default: 0
  --fista-num-layers N     Default: 7
  --fista-ckpt PATH        Optional LFISTNet checkpoint

  --skip-sino-train        Reuse existing sino-domain checkpoint
  --skip-sino-gen          Reuse existing sino intermediate dataset
  --skip-fista             Reuse existing FISTA dataset
  --skip-image-train       Reuse existing image-domain checkpoint
  --overwrite              Pass --overwrite to FISTA generation
  -h, --help               Show this help

Environment overrides are also supported, for example:
  DEVICE=3 TEST_VIEWS=12,15,18,21,24 scripts/run_singleview_v18_training.sh
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --train-view) TRAIN_VIEW="$2"; shift 2 ;;
    --train-splits) TRAIN_SPLITS="$2"; shift 2 ;;
    --test-views) TEST_VIEWS="$2"; shift 2 ;;
    --sino-config) SINO_CONFIG="$2"; shift 2 ;;
    --image-config) IMAGE_CONFIG="$2"; shift 2 ;;
    --sino-root) SINO_ROOT="$2"; shift 2 ;;
    --fista-root) FISTA_ROOT="$2"; shift 2 ;;
    --sino-batch-size) SINO_BATCH_SIZE="$2"; shift 2 ;;
    --fista-batch-size) FISTA_BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --fista-num-layers) FISTA_NUM_LAYERS="$2"; shift 2 ;;
    --fista-ckpt) FISTA_CKPT="$2"; shift 2 ;;
    --skip-sino-train) SKIP_SINO_TRAIN=1; shift ;;
    --skip-sino-gen) SKIP_SINO_GEN=1; shift ;;
    --skip-fista) SKIP_FISTA=1; shift ;;
    --skip-image-train) SKIP_IMAGE_TRAIN=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -f "$CONDA_SH" ]]; then
  echo "ERROR: conda init script not found: $CONDA_SH" >&2
  exit 1
fi

latest_file() {
  local pattern="$1"
  local found
  found="$(find . -path "$pattern" -type f | sort | tail -n 1 || true)"
  if [[ -z "$found" ]]; then
    echo "ERROR: no file found for pattern: $pattern" >&2
    exit 1
  fi
  echo "${found#./}"
}

# Some conda hooks reference optional backup variables.
set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

export CUDA_VISIBLE_DEVICES="$DEVICE"

echo "================================================================"
echo "Single-view training pipeline"
echo "device       : $CUDA_VISIBLE_DEVICES"
echo "train_view   : $TRAIN_VIEW"
echo "train_splits : $TRAIN_SPLITS"
echo "test_views   : ${TEST_VIEWS:-<none>}"
echo "sino_config  : $SINO_CONFIG"
echo "image_config : $IMAGE_CONFIG"
echo "sino_root    : $SINO_ROOT"
echo "fista_root   : $FISTA_ROOT"
echo "================================================================"

if [[ "$SKIP_SINO_TRAIN" -eq 0 ]]; then
  echo
  echo "[1/4] Train fixed-view sino-domain model"
  python SinoDomain_train_multiview.py --config "$SINO_CONFIG"
else
  echo
  echo "[1/4] Sino-domain training skipped"
fi

SINO_CKPT="$(latest_file "*/outputs/sino_domain_singleview_v18_*/checkpoints/sino_multiview_best_main.pth")"
echo "sino_ckpt: $SINO_CKPT"

if [[ "$SKIP_SINO_GEN" -eq 0 ]]; then
  echo
  echo "[2/4] Generate fixed-view sino dataset"
  python generate_multiview_sino_dataset.py \
    --config "$SINO_CONFIG" \
    --ckpt "$SINO_CKPT" \
    --out-root "$SINO_ROOT" \
    --splits "$TRAIN_SPLITS" \
    --views "$TRAIN_VIEW" \
    --batch-size "$SINO_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"
else
  echo
  echo "[2/4] Sino dataset generation skipped"
fi

if [[ "$SKIP_FISTA" -eq 0 ]]; then
  echo
  echo "[3/4] Generate fixed-view FISTA CT dataset"
  fista_cmd=(
    python generate_multiview_fista_dataset.py
    --sino-root "$SINO_ROOT"
    --out-root "$FISTA_ROOT"
    --splits "$TRAIN_SPLITS"
    --views "$TRAIN_VIEW"
    --batch-size "$FISTA_BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --num-layers "$FISTA_NUM_LAYERS"
  )

  if [[ -n "$FISTA_CKPT" ]]; then
    fista_cmd+=(--fista-ckpt "$FISTA_CKPT")
  fi

  if [[ "$OVERWRITE" -eq 1 ]]; then
    fista_cmd+=(--overwrite)
  fi

  "${fista_cmd[@]}"
else
  echo
  echo "[3/4] FISTA generation skipped"
fi

if [[ "$SKIP_IMAGE_TRAIN" -eq 0 ]]; then
  echo
  echo "[4/4] Train fixed-view image-domain model"
  python ImageDomain_train_multiview.py --config "$IMAGE_CONFIG"
else
  echo
  echo "[4/4] Image-domain training skipped"
fi

IMAGE_CKPT="$(latest_file "*/outputs/image_singleview_fista_ours_v18_*/checkpoints/image_multiview_best_psnr.pth")"
echo "image_ckpt: $IMAGE_CKPT"

if [[ -n "$TEST_VIEWS" ]]; then
  echo
  echo "[extra] Cross-view test with fixed-view18 models"
  scripts/run_custom_views_pipeline.sh \
    --views "$TEST_VIEWS" \
    --splits test \
    --image-split test \
    --sino-config "$SINO_CONFIG" \
    --sino-ckpt "$SINO_CKPT" \
    --sino-root "$SINO_ROOT" \
    --fista-root "$FISTA_ROOT" \
    --image-config "$IMAGE_CONFIG" \
    --image-ckpt "$IMAGE_CKPT" \
    --overwrite
fi

echo
echo "Single-view training pipeline complete."
