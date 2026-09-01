# Bi-RealNet PopBin -> G64 Hierarchical Majority Voter

이 폴더는 공식 ReActNet 저장소(`a6cd08d4605d94135faea6fd73354041e4130b52`)와
사용자가 전달한 `HMV Code.zip`을 대조해 정리한 **Bi-RealNet 우선 실험용 최종 overlay**입니다.
기존의 여러 `3_step3`, `3_step3_try` 파일을 섞어 쓰지 않습니다.

## 1. 이번 최종본에서 확정한 명세

- 대상: CIFAR-100 Bi-RealNet-18, 총 16개 `HardBinaryConv`
- 초기 가중치: 기존 `resnet/2_step2/models/checkpoint.pth.tar`
- 적용 범위: 모든 binary convolution layer
- HMV: 2-level, Level-1 group size `G=64`
- 기본 wiring: spatial (기존 flatten 순서를 유지해 같은 kernel/channel 인접 항을 먼저 묶음)
- Level-1 threshold: layer별 scalar 정수 `T1_l`, 범위 `0..65`
- Level-2 threshold: layer별 scalar 정수 `T2_l`, 범위 `0..M_l+1`
- 판정: positive match/vote 수가 threshold 이상이면 `+1`, 아니면 `-1`
- 초기값: `T1=33`, `T2=floor(M/2)+1` (동률은 `-1`)
- forward: 반올림 및 clamp된 비음수 정수 threshold만 사용
- backward: STE와 Bi-Real polynomial surrogate 사용
- zero padding: 유효 비트 수를 별도 mask로 계산해 padding 0을 `-1` bit로 오해하지 않음
- remainder: 유효 비트 수 `r`에 맞춰 strict-majority 기준을 보정

| Binary layers | Reduction N | G64 group count M | Initial T1 | Initial T2 |
|---|---:|---:|---:|---:|
| L1-L5 | 576 | 9 | 33 | 5 |
| L6-L9 | 1152 | 18 | 33 | 10 |
| L10-L13 | 2304 | 36 | 33 | 19 |
| L14-L16 | 4608 | 72 | 33 | 37 |

현재 Bi-RealNet 코드의 최대 reduction은 4608입니다. 9216은 이 모델의 현재 16개
binary convolution 명세가 아니므로, ReActNet/MobileNet 확장 때 모델에서 다시 자동 추출해야 합니다.

## 2. 서버에 필요한 외부 파일

코드 작성에 추가 정보는 필요하지 않았습니다. 다만 실제 학습에는 ZIP에 포함되지 않은 다음 파일이 필요합니다.

1. 필수: 학습 완료된 `2_step2/models/checkpoint.pth.tar`
2. KD 사용 시 필수: `pytorch_cifar100`의 CIFAR-100 ResNet-18 teacher checkpoint
3. CIFAR-100: root 아래 `cifar-100-python/{train,test,meta}`

공식 ReActNet의 2단계 학습은 `DistributionLoss` 기반 KD를 사용합니다. 따라서 기존 recipe를
정확히 존중하는 본 실험의 기본 objective도 `kd`입니다. `ce`는 teacher 없이 smoke test를 하거나
추가 ablation을 할 때만 사용합니다.

## 3. 설치

서버에 공식 ReActNet과 `pytorch_cifar100`이 있다고 가정합니다.

```bash
cd /workspace/ReActNet
cp -r /업로드한/경로/ReActNet_HMV_G64/resnet/3_step3_hmv ./resnet/
cd ./resnet/3_step3_hmv
```

기존 `resnet/1_step1`, `resnet/2_step2`, checkpoint는 수정하거나 덮어쓰지 않습니다.

## 4. 가장 먼저 할 smoke test

```bash
cd /workspace/ReActNet/resnet/3_step3_hmv
export DATA_ROOT=/workspace/ReActNet/pytorch_cifar100/data
export STAGE2_CHECKPOINT=/workspace/ReActNet/resnet/2_step2/models/checkpoint.pth.tar
bash run_smoke.sh
```

이 실행은 2 train batch와 2 validation batch만 처리하는 구조 검증입니다. 논문 결과가 아닙니다.
성공 조건은 checkpoint mismatch가 없고, 16개 layer threshold가 export되며, forward/backward가
오류 없이 끝나는 것입니다.

## 5. 권장 학습 순서

### A. Threshold-only QAT

모든 기존 PopBin parameter를 고정하고 16개 layer의 L1/L2 threshold 32개만 학습합니다.

```bash
export DATA_ROOT=/workspace/ReActNet/pytorch_cifar100/data
export STAGE2_CHECKPOINT=/workspace/ReActNet/resnet/2_step2/models/checkpoint.pth.tar
export TEACHER_CHECKPOINT=/workspace/ReActNet/pytorch_cifar100/checkpoint/resnet18/실제경로/resnet18-200-regular.pth
bash run_threshold_qat.sh
```

### B. HMV-aware full QAT

Threshold-only best checkpoint에서 시작해 threshold와 기존 PopBin weight를 함께 fine-tuning합니다.
새 stage이므로 optimizer와 epoch는 의도적으로 다시 시작합니다.

```bash
export THRESHOLD_CHECKPOINT=./models_hmv_g64_spatial_threshold/model_best.pth.tar
bash run_full_qat.sh
```

OOM이 발생하면 먼저 `--batch_size 32`를 `16`, 그다음 `8`로 낮춥니다. 학습 의미는 바뀌지
않지만 batch size별 결과는 실험표에 반드시 기록합니다.

같은 stage의 중단 지점부터 정확히 이어갈 때만 다음 두 인자를 함께 사용합니다.

```bash
--hmv_resume ./models_hmv_g64_spatial_full/checkpoint.pth.tar --continue_run
```

## 6. tmux에서 실행하고 다음 날 확인하기

```bash
tmux new -s hmv_g64
cd /workspace/ReActNet/resnet/3_step3_hmv
mkdir -p logs
export DATA_ROOT=/workspace/ReActNet/pytorch_cifar100/data
export STAGE2_CHECKPOINT=/workspace/ReActNet/resnet/2_step2/models/checkpoint.pth.tar
export TEACHER_CHECKPOINT=/workspace/ReActNet/pytorch_cifar100/checkpoint/resnet18/실제경로/resnet18-200-regular.pth
bash -o pipefail -c 'bash run_threshold_qat.sh 2>&1 | tee logs/threshold_qat.log'
```

분리: `Ctrl-b`, 이어서 `d`

```bash
tmux attach -t hmv_g64
tail -n 100 logs/threshold_qat.log
tail -f logs/threshold_qat.log
```

## 7. 최종 평가

```bash
export HMV_CHECKPOINT=./models_hmv_g64_spatial_full/model_best.pth.tar
bash run_evaluate.sh
```

평가는 같은 실제 CIFAR-100 sample에서 다음을 모두 계산합니다.

1. `reference PopBin`: 원래 2_step2 checkpoint와 flat PopBin operator
2. `adapted flat`: HMV-QAT로 바뀐 weight를 flat PopBin operator에 넣은 결과
3. `learned HMV`: 학습된 weight와 layer별 정수 threshold를 사용한 HMV 결과
4. reference/HMV end-to-end class prediction agreement
5. reference/adapted-flat prediction agreement (threshold-only에서는 100%여야 함)
6. adapted-flat/HMV end-to-end class prediction agreement
7. 각 layer에서 flat PopBin output과 HMV output의 operator match/mismatch rate

생성 파일:

- `evaluation_summary.json`: 세 경로의 top-1/top-5, loss, prediction agreement
- `learned_thresholds.csv`: 16개 layer의 최종 정수 L1/L2 threshold
- `layer_operator_match.csv`: layer별 match, mismatch, positive, tie rate
- `thresholds_latest.{csv,json}`: 학습 중 최신 하드웨어 설정값
- `training_history.csv`: epoch별 accuracy/loss/LR

## 8. 파일 역할

- `birealnet_hmv.py`: PopBin 호환 Bi-RealNet과 학습 가능한 정수 HMV
- `train_hmv.py`: threshold-only/full QAT, CE/KD 선택, checkpoint 저장
- `evaluate_hmv.py`: baseline/HMV 정확도 및 일치율 통합 평가
- `experiment_utils.py`: CIFAR-100 loader, checkpoint, CSV/JSON 공통 기능
- `tests/test_hmv.py`: tie, threshold, padding, 16-layer 명세 단위 테스트
- `run_*.sh`: 서버 실행 명령을 고정한 wrapper

제외한 전달본 파일은 PTQ 탐색본, 9/18/36-bit 실험본, single-layer 실험본, 서로 다른
`birealnet_v1.py` 변형, 빈 `train,py`입니다. 연구 기록으로는 보존하되 이번 G64 QAT 실행 경로에는
섞지 않는 것이 안전합니다. ReActNet/MobileNet HMV는 Bi-RealNet 결과를 확정한 뒤 별도 2차 작업으로
옮기는 것이 현재 실험 순서와 맞습니다.
