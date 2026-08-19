<h1 align="center">Toward Universal Skeleton-Based Action Recognition across Heterogeneous Skeletons and Open Vocabularies</h1>

<p align="center">
  Jidong Kuang &middot; Hongsong Wang &middot; Jie Gui &middot; Yuan Yan Tang &middot; James Tin-Yau Kwok
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.17013"><img src="https://img.shields.io/badge/arXiv-2604.17013-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="#requirements"><img src="https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white" alt="Python 3.9"></a>
  <a href="#requirements"><img src="https://img.shields.io/badge/PyTorch-2.1.1%2Bcu118-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.1.1 with CUDA 11.8"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="MIT License"></a>
</p>

Official PyTorch implementation of [**Toward Universal Skeleton-Based Action Recognition across Heterogeneous Skeletons and Open Vocabularies**](https://arxiv.org/abs/2604.17013).

## Pipeline

<p align="center">
  <img src="assets/pipeline_overview.png" width="95%" alt="Universal skeleton action recognition pipeline">
  <br>
  <b>Overview of heterogeneous skeleton unification and multi-grained motion-text alignment.</b>
</p>

## Overview

This work studies action recognition across heterogeneous skeleton formats and
open vocabularies. The released pipeline provides:

- a unified 31-joint representation for Kinect v1, Kinect v2, COCO-17, and
  SMPL-22 skeletons;
- deterministic kinematic imputation for joints absent from a source format;
- joint, bone, and motion streams with a shared spatio-temporal Transformer;
- global, stream-specific, and fine-grained motion-text alignment; and
- joint training on NTU RGB+D 60/120 3D skeletons, 2D poses, and HumanML3D.

## Requirements

The code was tested with:

- Python 3.9
- PyTorch 2.1.1 + CUDA 11.8

```bash
git clone https://github.com/jidongkuang/Universal-Skeleton.git
cd Universal-Skeleton
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

Raw datasets and generated caches are not distributed with this repository.
The HumanML3D 400-class annotations, fixed train/validation splits, label maps,
and normalization statistics are included. Prepare the large source files in
the following layout:

```text
data/
|-- ntu/
|   `-- NTU60_CS.npz
|-- nturgb/
|   |-- ntu60_2d.pkl
|   |-- ntu60_2d_Mean.npy
|   `-- ntu60_2d_Std.npy
`-- HumanML3D/
    |-- new_joints/
    `-- texts/
```

See [Data Preparation](docs/data_preparation.md) for official sources, expected
schemas, split sizes, and path overrides.

Dataset paths can be overridden with `HOV_NTU_3D`, `HOV_NTU_2D`,
`HOV_NTU_2D_MEAN`, `HOV_NTU_2D_STD`, `HOV_NTU_CACHE_DIR`, and
`HUMANML3D_ROOT`; see `config.py` for all options. Profile-specific names such
as `HOV_NTU60_3D` are also supported.

## Unified Skeleton Representation

Every source format is canonicalized to `(3, T, 31, 2)` before entering the
shared encoder.

| Source | Native joints | Coordinates | Mapping key |
|---|---:|---:|---|
| Kinect v1 / NW-UCLA | 20 | 3D | `kinect_v1_20` |
| Kinect v2 / NTU | 25 | 3D | `kinect_v2_25` |
| COCO pose / NTU-2D | 17 | 2D | `coco17` |
| SMPL / HumanML3D | 22 | 3D | `smpl22` |

For 2D inputs, the depth channel is zero. Missing trunk joints are interpolated,
while terminal joints are extrapolated along the kinematic chain:

```text
p_target = p_anchor + rho * (p_anchor - p_parent)
```

We use `rho=0.25` for hands, `0.15` for feet, `0.18` for hand tips, and
`0.32` for thumbs. Rules execute only when their anchors are valid and never
overwrite observed joints. A single observed person is replicated into the
second person slot. Validity masks guard the imputation process and are not
passed to the encoder as gates or attention masks.

The complete joint mapping, topology, and imputation program are defined in
`datasets/unified_skeleton.py`. The main training entry directly loads Kinect
v2, COCO-17, and SMPL data; the tested Kinect v1 mapping can be integrated with
an NW-UCLA loader.

## Training

Run the NTU-60 3D + NTU-60 2D + HumanML3D configuration:

```bash
bash scripts/train_table2.sh
```

The general training entry also supports the NTU-120 profile:

```bash
HOV_NTU_NUM_CLASSES=120 bash scripts/train_main.sh
```

Set the output directory or GPU through environment variables when needed:

```bash
HOV_DEVICE=0 HOV_OUTPUT_DIR=work_dirs/ntu60 \
HOV_NTU_NUM_CLASSES=60 python train.py
```

Logs and checkpoints are written to `work_dirs/` by default.

## Pretrained Model

The epoch-360 checkpoint is stored separately because of its size:

| File | Download | Size | SHA256 |
|---|---|---:|---|
| `universal_skeleton_ntu60_humanml3d_epoch360.pth` | [Google Drive](https://drive.google.com/file/d/17J3wiBKD0vpyJZSu7wlnIcn3u8cBouQh/view?usp=sharing) | 306,534,726 bytes | `6a8861948728df8c44a8949a3d9b68a20e6bcdf245593bd74fd46fbc3959fc09` |

Verify the checksum before evaluation.

## Evaluation

After preparing the datasets and downloading the checkpoint, run:

```bash
bash scripts/eval_table2.sh \
  /path/to/universal_skeleton_ntu60_humanml3d_epoch360.pth
```

The command evaluates one checkpoint on all three test sets and writes the
same metrics to `work_dirs/table2_eval.json`. With seed 0, the expected output
is:

| NTU-60 3D | NTU-60 2D | HumanML3D | Many-shot | Medium-shot | Few-shot |
|---:|---:|---:|---:|---:|---:|
| 87.2202 | 90.4834 | 62.7656 | 72.5089 | 53.7938 | 49.6440 |

The first two columns use the NTU-60 cross-subject split. HumanML3D uses the
included validation split and accepts a prediction when it matches any label
assigned to the motion.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover source-to-unified mappings, kinematic imputation, data shapes,
cache validation, NTU profiles, text labels, model inputs, checkpoint loading,
evaluation interfaces, and bone topology.

## Acknowledgements

This repository includes the [CLIP](https://github.com/openai/CLIP) text
encoder implementation. We thank the authors of NTU RGB+D, NW-UCLA, and
HumanML3D for making their datasets available to the research community.

## License

This project is released under the [MIT License](LICENSE).

## Citation

If you find this work useful, please cite:

```bibtex
@misc{kuang2026universalskeletonbasedactionrecognition,
  title         = {Toward Universal Skeleton-Based Action Recognition across Heterogeneous Skeletons and Open Vocabularies},
  author        = {Jidong Kuang and Hongsong Wang and Jie Gui and Yuan Yan Tang and James Tin-Yau Kwok},
  year          = {2026},
  eprint        = {2604.17013},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2604.17013}
}
```

## Contact

For questions, please contact `jidongkuang@seu.edu.cn`.
