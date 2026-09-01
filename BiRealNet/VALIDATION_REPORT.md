# Local validation report

Validation date: 2026-08-31

## Passed locally

- Python `compileall` for every delivered `.py` file
- Six CPU tensor unit tests
  - default 32:32 L1 tie maps to negative
  - learned `T1=32` can map the same tie to positive
  - spatial zero-padding uses a validity mask
  - exported thresholds are rounded, clamped, and nonnegative
  - Bi-RealNet-18 contains 16 expected binary layers and correct N/M values
  - both L1 and L2 thresholds receive finite surrogate gradients
- Full Bi-RealNet forward on a `(1, 3, 32, 32)` tensor
- Full threshold-only backward through all 16 HMV layers
- Threshold-only training keeps BatchNorm running statistics frozen
- Per-layer operator-statistics collection through all 16 layers
- State-dict compatibility against the supplied `resnet/2_step2/birealnet.py`
  - old PopBin keys: 202
  - delivered model keys: 234
  - retained old keys: all 202
  - intentionally added keys: 32 thresholds only
- Synthetic DataParallel-style `2_step2` checkpoint load
- `train_hmv.py --help` and `evaluate_hmv.py --help` imports

## Must still be run on the server

- Real `checkpoint.pth.tar` load, because the checkpoint was not in `HMV Code.zip`
- CIFAR-100 two-batch smoke test
- CUDA memory check at batch sizes 32, 16, and 8
- Full threshold-only and full-QAT runs
- Final real-data accuracy, prediction agreement, and layer match-rate export

Local tests used PyTorch 2.13 CPU only for arithmetic verification. The target container's existing
PyTorch/torchvision versions should be recorded in the experiment log with `pip freeze` and
`python3 -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"`.
