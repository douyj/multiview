"""
scripts/run_custom_views_pipeline.sh \
  --views 13,16,20,23 \
  --splits test \
  --image-split test \
  --overwrite

scripts/run_custom_views_pipeline.sh \
  --views 13,16,20,23 \
  --splits test \
  --image-split test \
  --image-save-dir outputs/custom_test_views_13_16_20_23 \
  --overwrite
"""

#!/usr/bin/env bash
set -euo pipefail

VIEWS="${VIEWS:-13,16,20,23}"
SPLITS="${SPLITS:-test}"
IMAGE_SPLIT="${IMAGE_SPLIT:-test}"
DEVICE="${DEVICE:-3}"

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-CT}"

SINO_CONFIG="${SINO_CONFIG:-configs/sino_multiview_v12to24.yaml}"
SINO_CKPT="${SINO_CKPT:-outputs/sino_domain_multiview_v12to24_20260623/checkpoints/sino_multiview_best_main.pth}"
SINO_ROOT="${SINO_ROOT:-./multiview_dataset/sino_v12to24}"

FISTA_ROOT="${FISTA_ROOT:-./multiview_dataset/fista_v12to24}"
FISTA_CKPT="${FISTA_CKPT:-}"
FISTA_NUM_LAYERS="${FISTA_NUM_LAYERS:-7}"

IMAGE_CONFIG="${IMAGE_CONFIG:-configs/image_multiview_fista_v12to24.yaml}"
IMAGE_CKPT="${IMAGE_CKPT:-outputs/image_multiview_fista_ours_v12to24_20260623/checkpoints/image_multiview_best_psnr.pth}"
IMAGE_SAVE_DIR="${IMAGE_SAVE_DIR:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

SINO_BATCH_SIZE="${SINO_BATCH_SIZE:-8}"
FISTA_BATCH_SIZE="${FISTA_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"

SKIP_SINO=0
SKIP_FISTA=0
SKIP_IMAGE_TEST=0
OVERWRITE=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_custom_views_pipeline.sh [options]

Required idea:
  --views 13,16,20,23

Options:
  --views VIEWS              Comma-separated view counts. Default: 13,16,20,23
  --splits SPLITS            Comma-separated splits for sino/FISTA generation. Default: test
  --image-split SPLIT        Split for ImageDomain_test_multiview.py. Default: test
  --device DEVICE            CUDA_VISIBLE_DEVICES value. Default: 3

  --sino-config PATH         Sino config yaml.
  --sino-ckpt PATH           Sino checkpoint.
  --sino-root PATH           Sino intermediate output root.

  --fista-root PATH          FISTA CT output root.
  --fista-ckpt PATH          Optional LFISTNet checkpoint.
  --fista-num-layers N       FISTA layers. Default: 7

  --image-config PATH        Image-domain config yaml.
  --image-ckpt PATH          Image-domain checkpoint.
  --image-save-dir PATH      Optional ImageDomain_test output dir.
  --run-id ID                Suffix for default image output dir. Default: timestamp

  --sino-batch-size N        Default: 8
  --fista-batch-size N       Default: 2
  --num-workers N            Default: 0

  --skip-sino                Do not run sino-domain generation.
  --skip-fista               Do not run FISTA generation.
  --skip-image-test          Do not run image-domain test.
  --overwrite                Pass --overwrite to FISTA generation.
  -h, --help                 Show this help.

Environment overrides are also supported, for example:
  VIEWS=14,17,22 DEVICE=3 scripts/run_custom_views_pipeline.sh
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --views) VIEWS="$2"; shift 2 ;;
    --splits) SPLITS="$2"; shift 2 ;;
    --image-split) IMAGE_SPLIT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --sino-config) SINO_CONFIG="$2"; shift 2 ;;
    --sino-ckpt) SINO_CKPT="$2"; shift 2 ;;
    --sino-root) SINO_ROOT="$2"; shift 2 ;;
    --fista-root) FISTA_ROOT="$2"; shift 2 ;;
    --fista-ckpt) FISTA_CKPT="$2"; shift 2 ;;
    --fista-num-layers) FISTA_NUM_LAYERS="$2"; shift 2 ;;
    --image-config) IMAGE_CONFIG="$2"; shift 2 ;;
    --image-ckpt) IMAGE_CKPT="$2"; shift 2 ;;
    --image-save-dir) IMAGE_SAVE_DIR="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --sino-batch-size) SINO_BATCH_SIZE="$2"; shift 2 ;;
    --fista-batch-size) FISTA_BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --skip-sino) SKIP_SINO=1; shift ;;
    --skip-fista) SKIP_FISTA=1; shift ;;
    --skip-image-test) SKIP_IMAGE_TEST=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$VIEWS" ]]; then
  echo "ERROR: --views is empty." >&2
  exit 2
fi

if [[ ! -f "$CONDA_SH" ]]; then
  echo "ERROR: conda init script not found: $CONDA_SH" >&2
  exit 1
fi

# Some conda activate/deactivate hooks reference optional backup variables.
# Keep nounset disabled only while conda mutates the shell environment.
set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

export CUDA_VISIBLE_DEVICES="$DEVICE"

echo "================================================================"
echo "Custom multiview pipeline"
echo "views          : $VIEWS"
echo "splits         : $SPLITS"
echo "image_split    : $IMAGE_SPLIT"
echo "device         : $CUDA_VISIBLE_DEVICES"
echo "sino_config    : $SINO_CONFIG"
echo "sino_ckpt      : $SINO_CKPT"
echo "sino_root      : $SINO_ROOT"
echo "fista_root     : $FISTA_ROOT"
echo "image_config   : $IMAGE_CONFIG"
echo "image_ckpt     : $IMAGE_CKPT"
echo "================================================================"

if [[ "$SKIP_SINO" -eq 0 ]]; then
  echo
  echo "[1/3] Sino-domain generation"
  python generate_multiview_sino_dataset.py \
    --config "$SINO_CONFIG" \
    --ckpt "$SINO_CKPT" \
    --out-root "$SINO_ROOT" \
    --splits "$SPLITS" \
    --views "$VIEWS" \
    --batch-size "$SINO_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"
else
  echo
  echo "[1/3] Sino-domain generation skipped"
fi

if [[ "$SKIP_FISTA" -eq 0 ]]; then
  echo
  echo "[2/3] FISTA CT generation"
  fista_cmd=(
    python generate_multiview_fista_dataset.py
    --sino-root "$SINO_ROOT"
    --out-root "$FISTA_ROOT"
    --splits "$SPLITS"
    --views "$VIEWS"
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
  echo "[2/3] FISTA CT generation skipped"
fi

if [[ "$SKIP_IMAGE_TEST" -eq 0 ]]; then
  echo
  echo "[3/3] Image-domain test"
  if [[ -z "$IMAGE_SAVE_DIR" ]]; then
    ckpt_name="$(basename "$IMAGE_CKPT" .pth)"
    ckpt_dir="$(dirname "$IMAGE_CKPT")"
    exp_dir="$(dirname "$ckpt_dir")"
    safe_views="${VIEWS//,/v}"
    IMAGE_SAVE_DIR="$exp_dir/${IMAGE_SPLIT}_results_${ckpt_name}_views_${safe_views}_${RUN_ID}"
  fi

  echo "image_save_dir : $IMAGE_SAVE_DIR"

  image_cmd=(
    python ImageDomain_test_multiview.py
    --config "$IMAGE_CONFIG"
    --ckpt "$IMAGE_CKPT"
    --split "$IMAGE_SPLIT"
    --views "$VIEWS"
    --save-dir "$IMAGE_SAVE_DIR"
  )

  "${image_cmd[@]}"
else
  echo
  echo "[3/3] Image-domain test skipped"
fi

echo
echo "Pipeline complete."
