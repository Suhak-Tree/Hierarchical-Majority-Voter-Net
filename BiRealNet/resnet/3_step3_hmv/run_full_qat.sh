#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/ReActNet/pytorch_cifar100/data}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-../2_step2/models/checkpoint.pth.tar}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT to the CIFAR-100 ResNet-18 checkpoint}"
THRESHOLD_CHECKPOINT="${THRESHOLD_CHECKPOINT:-./models_hmv_g64_spatial_threshold/model_best.pth.tar}"

python3 train_hmv.py \
  --data "$DATA_ROOT" \
  --resume "$STAGE2_CHECKPOINT" \
  --hmv_resume "$THRESHOLD_CHECKPOINT" \
  --teacher_checkpoint "$TEACHER_CHECKPOINT" \
  --save ./models_hmv_g64_spatial_full \
  --objective kd \
  --stage full \
  --group_size 64 \
  --wiring spatial \
  --batch_size 32 \
  --workers 8 \
  --epochs 256 \
  --learning_rate 0.001 \
  --weight_decay 0
