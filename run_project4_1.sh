#!/usr/bin/env bash
set -euo pipefail

# Configurable parameters
DATA_DIR="${DATA_DIR:-/dtu/datasets1/02516/potholes}"
MODE="${MODE:-fast}"            # fast or quality
MAX_DIM="${MAX_DIM:-500}"       # resize longer side to at most MAX_DIM before Selective Search
MAX_PROPOSALS="${MAX_PROPOSALS:-2000}"
WORKERS="${WORKERS:-8}"
K_SELECTED="${K_SELECTED:-1000}"

echo "[1/4] Exploring dataset & plotting GT examples..."
python src/01_explore_data.py --data-dir "$DATA_DIR"

echo "[2/4] Extracting Selective Search proposals (mode=$MODE, max_dim=$MAX_DIM)..."
python src/02_extract_proposals.py \
  --data-dir "$DATA_DIR" \
  --mode "$MODE" \
  --max-dim "$MAX_DIM" \
  --max-proposals "$MAX_PROPOSALS" \
  --workers "$WORKERS"

PROPOSALS_NPZ="outputs/proposals_${MODE}_maxdim${MAX_DIM}.npz"

echo "[3/4] Evaluating proposals on the training set..."
python src/03_evaluate_proposals.py \
  --data-dir "$DATA_DIR" \
  --proposals "$PROPOSALS_NPZ" \
  --split train

echo "[4/4] Preparing labeled proposals for detector training (K=$K_SELECTED)..."
python src/04_prepare_proposals.py \
  --data-dir "$DATA_DIR" \
  --proposals "$PROPOSALS_NPZ" \
  --num-proposals "$K_SELECTED" \
  --pos-iou 0.5 \
  --neg-iou-low 0.0 \
  --neg-iou-high 0.3

echo "Done! Check outputs/ and outputs/figures/."
