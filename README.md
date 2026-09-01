HMV 실행 및 학습 명령어

## Smoke test

```bash
cd /workspace/resnet/3_step3

DATA_ROOT=/workspace/pytorch_cifar100/data \
STAGE2_CHECKPOINT=/workspace/resnet/2_step2/models/checkpoint.pth.tar \
bash run_smoke.sh

ls -lh ./models_hmv_g64_spatial_smoke
head -n 20 ./models_hmv_g64_spatial_smoke/thresholds_latest.csv

python3 evaluate_hmv.py \
  --data /workspace/pytorch_cifar100/data \
  --reference_checkpoint /workspace/resnet/2_step2/models/checkpoint.pth.tar \
  --hmv_checkpoint ./models_hmv_g64_spatial_smoke/checkpoint.pth.tar \
  --output ./hmv_eval_smoke \
  --group_size 64 \
  --wiring spatial \
  --batch_size 8 \
  --workers 2 \
  --max_batches 2

cat ./hmv_eval_smoke/evaluation_summary.json
head -n 20 ./hmv_eval_smoke/learned_thresholds.csv
head -n 20 ./hmv_eval_smoke/layer_operator_match.csv
```

## 학습 1: Threshold 학습 후 전체 재학습

### 1단계: Threshold 학습

```bash
bash -o pipefail -c \
  'bash run_threshold_qat.sh 2>&1 | tee logs/threshold_qat.log'
```

### 2단계: 전체 재학습

```bash
export DATA_ROOT=/workspace/pytorch_cifar100/data
export STAGE2_CHECKPOINT=/workspace/resnet/2_step2/models/model_best.pth.tar
export TEACHER_CHECKPOINT=/workspace/pytorch_cifar100/checkpoint/resnet18/Monday_24_March_2025_02h_46m_14s/resnet18-200-regular.pth

ls -lh "$STAGE2_CHECKPOINT"
ls -lh "$TEACHER_CHECKPOINT"

bash -o pipefail -c \
  'bash run_threshold_qat.sh 2>&1 | tee logs/threshold_qat.log'
```

## 학습 2: Threshold 및 Binary Weight 동시 학습

```bash
cd /workspace/resnet/3_step3
mkdir -p logs

python3 train_hmv.py \
  --data /workspace/pytorch_cifar100/data \
  --resume /workspace/resnet/2_step2/models/model_best.pth.tar \
  --save ./models_hmv_g64_spatial_full_direct \
  --objective kd \
  --teacher_checkpoint /workspace/pytorch_cifar100/checkpoint/resnet18/Monday_24_March_2025_02h_46m_14s/resnet18-200-regular.pth \
  --stage full \
  --train_l1_threshold 1 \
  --train_l2_threshold 1 \
  --group_size 64 \
  --wiring spatial \
  --batch_size 32 \
  --workers 8 \
  --epochs 256 \
  --learning_rate 0.001 \
  --weight_decay 0 \
  2>&1 | tee logs/full_qat_direct.log
```
