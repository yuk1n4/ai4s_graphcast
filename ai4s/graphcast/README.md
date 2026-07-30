# GraphCast — AI4S Multi-Platform Weather Prediction

PyTorch-based implementation of Google DeepMind's GraphCast global weather forecasting
model (0.25 deg resolution). Trained and validated on NVIDIA H100, Huawei Ascend 910C,
Hygon BW1000, and T-Head PPU-ZW810E.

## Quick Start

```bash
# One command — downloads data, runs inference (~30 sec)
bash quickstart.sh

# Or step by step:
# 1. Download data (one-time, ~1 GB)
bash quickstart.sh infer    # auto-downloads if data/ missing

# 2. Run training (4-GPU model parallel, 3 steps)
bash quickstart.sh train
```

The first run downloads the data archive (~1 GB) from GitHub Releases automatically.

## quickstart.sh Usage

```bash
bash quickstart.sh [MODE] [OPTIONS...]
```

### Modes

| Command | Action | Typical time |
|---------|--------|-------------|
| `bash quickstart.sh` | **Inference** (default) — fastest env check | ~30 sec |
| `bash quickstart.sh infer` | Inference only | ~30 sec |
| `bash quickstart.sh train` | Training only (4-GPU, 3 steps) | ~1-10 min |
| `bash quickstart.sh all` | Train then infer | ~2-15 min |

### Options (forwarded to train.sh / infer.sh)

| Option | Default | Description |
|--------|---------|-------------|
| `--device DEV` | `cuda` | Accelerator: `cuda` (NVIDIA/Hygon/PPU) or `npu` (Ascend) |
| `--steps N` | `3` | Training steps |
| `--flagos-mode MODE` | `off` | `flagems` to enable FlagGems operator replacement |
| `--flaggems-ops OPS` | (all 6) | Comma-separated list of FlagGems operators |

### Examples

```bash
# Quick environment check (inference, default mode)
bash quickstart.sh

# Help
bash quickstart.sh --help

# 10-step native training
bash quickstart.sh train --steps 10

# Ascend FlagGems training (6 ops, excl index_add_)
bash quickstart.sh train --device npu --flagos-mode flaggems \
    --flaggems-ops "addmm,silu,layer_norm,cat,index,add"

# PPU FlagGems training (5 ops, excl cat + index_add_)
bash quickstart.sh train --flagos-mode flaggems \
    --flaggems-ops "addmm,silu,layer_norm,index,add"

# Full pipeline: train 10 steps then infer
bash quickstart.sh all --steps 10
```

## Supported Platforms

| Platform | Device | Backend | FlagGems | Status |
|----------|--------|---------|----------|--------|
| NVIDIA H100 | `cuda` | native_cuda | — | ✓ |
| Huawei Ascend 910C | `npu` | native_npu / flagos_ascend | 6/7 ops | ✓ |
| Hygon BW1000 | `cuda` (HIP) | native_hygon / flagos_hygon | 4/7 ops | ✓ |
| T-Head PPU-ZW810E | `cuda` (HGGC) | native_ptg / flagos_ptg | 5/7 ops | ✓ |

See `ai4s/docs/multi-platform-guide.md` for accelerator compliance requirements.

## Project Structure

```
ai4s/graphcast/
├── quickstart.sh           # one-click data download + train/infer
├── README.md
├── requirements.txt
├── configs/
│   ├── training/           # 25km and 100km training YAML
│   └── inference/          # operational 25km inference YAML
├── src/graphcast_compat/   # model, backend, data loading, normalization
├── scripts/
│   ├── train.sh            # training entry → training/train_mp.sh
│   ├── infer.sh            # inference entry → inference/run_25km_inference.sh
│   ├── training/           # model parallel, DDP, single-GPU
│   └── inference/          # inference scripts
├── tools/                  # FlagGems candidate staging
└── data/                   # cases, stats, checkpoints, wheelhouse (auto-downloaded)
```

## Manual Training

```bash
# Native (baseline B)
bash scripts/train.sh

# FlagGems (baseline C)
GRAPHCAST_BACKEND=flagos_ptg \
GRAPHCAST_FLAGOS_MODE=flaggems \
GRAPHCAST_FLAGGEMS_OPS=addmm,silu,layer_norm,index,add \
  bash scripts/train.sh
```

## Manual Inference

```bash
bash scripts/infer.sh
```

## Data

The quickstart script auto-downloads data from:
```
https://github.com/yuk1n4/ai4s_graphcast/releases/download/v1.0-data-25km/graphcast_25km_data.tar.gz
```

Archive contents (~1 GB):
```
data/
├── cases/operational_25km/   # inputs.nc, targets.nc, forcings.nc
├── stats/                    # normalization statistics
├── checkpoints/              # pretrained weights
└── wheelhouse/               # offline Python packages (Ascend + PPU)
```

Platform-specific guides in `graphcast_ds/docs/platforms/` cover Docker setup,
GPU checks, and exact commands per accelerator.

## License

Apache 2.0. See repository root `LICENSE`.

