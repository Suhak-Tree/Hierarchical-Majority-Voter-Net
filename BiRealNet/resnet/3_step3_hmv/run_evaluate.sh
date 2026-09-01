#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/ReActNet/pytorch_cifar100/data}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-../2_step2/models/checkpoint.pth.tar}"
HMV_CHECKPOINT="${HMV_CHECKPOINT:-./models_hmv_g64_spatial_full/model_best.pth.tar}"

python3 evaluate_hmv.py \
  --data "$DATA_ROOT" \
  --reference_checkpoint "$STAGE2_CHECKPOINT" \
  --hmv_checkpoint "$HMV_CHECKPOINT" \
  --output ./hmv_eval_g64_spatial \
  --group_size 64 \
  --wiring spatial \
  --batch_size 32 \
  --workers 8
