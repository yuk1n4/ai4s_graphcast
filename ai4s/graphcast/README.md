# GraphCast — AI4S Multi-Platform Weather Prediction

PyTorch-based implementation of Google DeepMind's GraphCast global weather forecasting
model (0.25 deg resolution). Trained and validated on NVIDIA H100, Huawei Ascend 910C,
Hygon BW1000, and T-Head PPU-ZW810E.

## Quick Start

```bash
# Download data (one-time)
curl -LO <release-url>/graphcast_25km_data.tar.gz
tar -xzf graphcast_25km_data.tar.gz

# Install dependencies
pip install -r requirements.txt

# Run inference
bash scripts/infer.sh

# Run training (4-GPU model parallel, 3 steps)
DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 STEPS=3 bash scripts/train.sh
```

## Supported Platforms

| Platform | Device | Backend | FlagGems | Status |
|----------|--------|---------|----------|--------|
| NVIDIA H100 | cuda | native_cuda | — | ✓ |
| Huawei Ascend 910C | npu | native_npu / flagos_ascend | 6/7 ops | ✓ |
| Hygon BW1000 | cuda (HIP) | native_hygon / flagos_hygon | 4/7 ops | ✓ |
| T-Head PPU-ZW810E | cuda (HGGC) | native_ptg / flagos_ptg | 5/7 ops | ✓ |

See `ai4s/docs/multi-platform-guide.md` for accelerator compliance requirements.

## Project Structure

```
ai4s/graphcast/
├── README.md
├── requirements.txt
├── configs/
│   ├── training/          # 25km and 100km training YAML configs
│   └── inference/         # operational 25km inference YAML
├── src/graphcast_compat/  # model, backend, data loading, normalization
├── scripts/
│   ├── train.sh           # training entry point → training/train_mp.sh
│   ├── infer.sh           # inference entry point → run_25km_inference.sh
│   ├── training/          # model parallel, DDP, single-GPU training
│   └── inference/         # inference scripts
├── tools/                 # FlagGems candidate staging (third_party/)
└── data/                  # cases, stats, checkpoints (downloaded separately)
```

## Training

```bash
# Native (baseline B)
bash scripts/train.sh

# FlagGems (baseline C) — Ascend example
GRAPHCAST_BACKEND=flagos_ascend \
GRAPHCAST_FLAGOS_MODE=flaggems \
GRAPHCAST_FLAGGEMS_OPS=addmm,silu,layer_norm,cat,index,add \
  bash scripts/train.sh
```

See platform-specific guides in the parent `docs/platforms/` directory for
Docker setup, GPU checks, and exact commands per accelerator.

## Inference

```bash
bash scripts/infer.sh
```

## License

Apache 2.0. See repository root `LICENSE`.
