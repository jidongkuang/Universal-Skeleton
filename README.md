# Universal Skeleton Action Recognition

PyTorch implementation of **Towards Universal Skeleton-Based Action Recognition**.

This repository is a cleaned, GitHub-ready release of the main training pipeline used for universal skeleton-based action recognition under heterogeneous skeleton sources and open-vocabulary supervision.

## Overview

The released code focuses on the core setting described in the paper:

- heterogeneous training across `NTU RGB+D 120` 3D skeletons, `NTU RGB+D 120` 2D pose sequences, and `HumanML3D`
- a multi-stream spatio-temporal encoder
- multi-grained motion-text alignment
- long-tail multi-label evaluation on `HumanML3D`

## Highlights

- Minimal standalone research codebase instead of a full internal experiment dump
- Standard project layout with separated `datasets`, `models`, `utils`, and `scripts`
- Environment-variable-based dataset configuration for easier reproduction
- Small metadata files included, while large datasets and checkpoints stay external

## Repository Structure

```text
.
|-- assets/                   # figures and release assets
|-- data/
|   |-- annotations/          # label grouping metadata
|   `-- text/                 # label text maps
|-- datasets/                 # dataset loaders and preprocessing
|-- models/                   # model definitions
|-- scripts/                  # runnable shell entrypoints
|-- third_party/clip/         # vendored CLIP implementation
|-- utils/                    # logging, paths, prompt helpers
|-- config.py                 # central experiment config
|-- train.py                  # main training entry
|-- requirements.txt
`-- README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Preparation

This repository does not bundle datasets, pretrained checkpoints, or large cached artifacts.

Set the required environment variables before training:

```bash
export HUMANML3D_ROOT=/path/to/HumanML3D
export HOV_NTU120_3D=/path/to/NTU120_CSub.npz
export HOV_NTU120_2D=/path/to/ntu120_hrnet.pkl
export HOV_NTU120_2D_MEAN=/path/to/ntu120_2d_Mean.npy
export HOV_NTU120_2D_STD=/path/to/ntu120_2d_Std.npy
export HOV_DEVICE=0
export HOV_OUTPUT_DIR=work_dirs/hov_main
```

Expected `HumanML3D` layout:

- `annotations_actions_400.json`
- `mean_std/new_Mean.npy`
- `mean_std/new_Std.npy`
- `new_joints/`
- `texts/`
- `split/new_train_longtail.txt`
- `split/new_val_longtail.txt`

## Training

Run the main training script with:

```bash
bash scripts/train_main.sh
```

Or directly:

```bash
python train.py
```

Logs and checkpoints are written to `work_dirs/` by default.

## Reproducibility Notes

- The current release keeps the main experiment path only and removes many internal ablation variants.
- The project vendors the CLIP implementation under `third_party/clip/`.
- Large datasets and generated artifacts are ignored by `.gitignore` so this folder can be used as a standalone GitHub repository.

## Release Notes

- `assets/` currently contains placeholders for figures and method illustrations.
- The repository metadata now points to the GitHub account `jidongkuang`.
- Update the final arXiv URL in `CITATION.cff` before publishing.

## Citation

See `CITATION.cff` for machine-readable citation metadata. Update the arXiv field before release.
