# Data Preparation

The released full configuration uses NTU RGB+D 60 3D skeletons, NTU-60 2D
poses, and HumanML3D motions. Raw datasets are not redistributed. Follow the
licenses and access rules of the original datasets:

- [NTU RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D)
- [MMAction2 skeleton data preparation](https://mmaction2.readthedocs.io/en/latest/dataset_zoo/skeleton.html)

## Included Metadata

The repository includes the experiment-specific files that are small enough to
redistribute:

```text
data/
|-- annotations/humanml3d_label_groups.json
|-- humanml3d/
|   |-- annotations_actions_400.json.gz
|   |-- mean_std/
|   |   |-- new_Mean.npy
|   |   `-- new_Std.npy
|   `-- split/
|       |-- new_train_longtail.txt
|       `-- new_val_longtail.txt
|-- nturgb/
|   |-- ntu60_2d_Mean.npy
|   `-- ntu60_2d_Std.npy
`-- text/
    |-- humanml3d400_label_map.txt
    `-- ntu120_label_map.txt
```

The HumanML3D loader reads the compressed annotation file directly. The
metadata root can be overridden with `HOV_HUMANML3D_METADATA_ROOT`.

## NTU-60 3D

Prepare the cross-subject split as one NPZ file. The loader expects these
members:

| Member | Shape | Description |
|---|---|---|
| `x_train` | `(40091, 300, 150)` | 2 people x 25 joints x 3 coordinates |
| `y_train` | `(40091, 60)` | one-hot labels |
| `x_test` | `(16487, 300, 150)` | cross-subject test skeletons |
| `y_test` | `(16487, 60)` | one-hot labels |

The common `NTU60_CS.npz` layout produced by 2s-AGCN-style NTU preprocessing is
supported directly. On first access, the four NPY members are extracted into a
memory-mapped cache; ensure the cache location has about 10 GB of free space.

Set the source and cache paths:

```bash
export HOV_NTU60_3D=/path/to/NTU60_CS.npz
export HOV_NTU60_CACHE_DIR=/path/to/ntu60_cache
```

## NTU-60 2D

The 2D annotation file follows the MMAction2/PYSKL pickle structure:

```text
{
  "split": {
    "xsub_train": [frame_dir, ...],
    "xsub_val": [frame_dir, ...]
  },
  "annotations": [
    {
      "frame_dir": str,
      "label": int,
      "total_frames": int,
      "keypoint": float array shaped (P, T, 17, 2),
      "keypoint_score": optional float array shaped (P, T, 17)
    },
    ...
  ]
}
```

The selected split must contain 40,091 training and 16,487 validation samples.
The normalization arrays are included. Place the pickle at
`data/nturgb/ntu60_2d.pkl` or set:

```bash
export HOV_NTU60_2D=/path/to/ntu60_2d.pkl
```

## HumanML3D

Generate HumanML3D by following its official repository. Only the motion and
caption directories are read from the external dataset root:

```text
/path/to/HumanML3D/
|-- new_joints/    # one (T, 22, 3) NPY array per motion
`-- texts/         # HumanML3D caption files
```

Set:

```bash
export HUMANML3D_ROOT=/path/to/HumanML3D
```

With the included split and malformed-record filtering used by the released
loader, the effective index contains 21,035 training items and 9,271 validation
items. The loader reports skipped malformed records as a warning instead of
silently discarding them.

## Path Check

For the repository-local layout, the required large files are:

```text
data/ntu/NTU60_CS.npz
data/nturgb/ntu60_2d.pkl
data/HumanML3D/new_joints/
data/HumanML3D/texts/
```

Environment variables take precedence over these defaults. All paths used in a
run are printed at the beginning of `train.log`.
