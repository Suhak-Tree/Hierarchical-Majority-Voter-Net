#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/ReActNet/pytorch_cifar100/data}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-../2_step2/models/checkpoint.pth.tar}"

python3 train_hmv.py \
  --data "$DATA_ROOT" \
  --resume "$STAGE2_CHECKPOINT" \
  --save ./models_hmv_g64_spatial_smoke \
  --objective ce \
  --stage threshold \
  --group_size 64 \
  --wiring spatial \
  --batch_size 8 \
  --workers 2 \
  --epochs 1 \
  --max_train_batches 2 \
  --max_val_batches 2 \
  --print_freq 1
