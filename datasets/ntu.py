import os
import pickle
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch

from datasets.transforms import random_rot, valid_crop_resize
from datasets.unified_skeleton import canonicalize, replicate_people


def _extract_npz_member(npz_path, member_name, cache_dir):
    npz_path = Path(npz_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / member_name
    with zipfile.ZipFile(npz_path, "r") as archive:
        try:
            expected_size = archive.getinfo(member_name).file_size
        except KeyError as error:
            raise ValueError(
                f"{npz_path} does not contain required member {member_name!r}"
            ) from error

        if target.is_file() and target.stat().st_size == expected_size:
            return target
        if target.exists() and not target.is_file():
            raise ValueError(f"cache target is not a regular file: {target}")

        temporary = cache_dir / f"{member_name}.tmp.{os.getpid()}"
        print(f"Extracting {member_name} from {npz_path} to {target}", flush=True)
        try:
            with archive.open(member_name, "r") as source, open(
                temporary, "wb"
            ) as output:
                shutil.copyfileobj(source, output, length=64 * 1024 * 1024)
            if temporary.stat().st_size != expected_size:
                raise OSError(
                    f"incomplete extraction of {member_name} from {npz_path}"
                )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def _load_npz_member_mmap(npz_path, member_name, cache_dir):
    return np.load(
        _extract_npz_member(npz_path, member_name, cache_dir), mmap_mode="r"
    )


class NTU3DDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        label_features,
        split="train",
        p_interval=(0.95,),
        random_rotation=False,
        window_size=128,
        cache_dir=None,
    ):
        self.data_path = str(data_path)
        self.label_features = label_features.detach().cpu().numpy()
        self.split = split
        self.window_size = window_size
        self.p_interval = list(p_interval)
        self.random_rotation = random_rotation
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(self.data_path).with_suffix("")
        )
        self._load_data()

    def _load_data(self):
        if self.split == "train":
            data_member, label_member = "x_train.npy", "y_train.npy"
        elif self.split == "test":
            data_member, label_member = "x_test.npy", "y_test.npy"
        else:
            raise NotImplementedError("split only supports train/test")

        self.data = _load_npz_member_mmap(
            self.data_path, data_member, self.cache_dir
        )
        labels = _load_npz_member_mmap(
            self.data_path, label_member, self.cache_dir
        )
        if self.data.ndim != 3:
            raise ValueError(f"unexpected NTU-3D data shape {self.data.shape}")
        if labels.ndim != 2:
            raise ValueError(f"unexpected NTU-3D label shape {labels.shape}")
        if len(self.data) != len(labels):
            raise ValueError(
                "NTU-3D data and label files contain different sample counts"
            )

        positive_labels = labels > 0
        labels_per_sample = positive_labels.sum(axis=1)
        if not np.all(labels_per_sample == 1):
            invalid_index = int(np.flatnonzero(labels_per_sample != 1)[0])
            raise ValueError(
                "NTU-3D labels must be one-hot; "
                f"sample {invalid_index} has {labels_per_sample[invalid_index]} "
                "positive entries"
            )
        self.labels = positive_labels.argmax(axis=1).astype(np.int64, copy=False)
        if self.labels.size and self.labels.max() >= len(self.label_features):
            raise ValueError(
                f"NTU-3D contains label {self.labels.max()}, but only "
                f"{len(self.label_features)} text labels were loaded"
            )

        _, self.num_frames, flattened = self.data.shape
        if flattened != 2 * 25 * 3:
            raise ValueError(f"unexpected NTU-3D flattened width {flattened}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        source = np.asarray(self.data[index], dtype=np.float32).reshape(
            self.num_frames, 2, 25, 3
        )
        observed = np.isfinite(source).all(axis=-1) & (
            np.abs(source).sum(axis=-1) > 1e-8
        )
        valid_frame_num = int(observed.any(axis=(1, 2)).sum())
        skeleton = canonicalize(source, "kinect_v2_25", observed)
        skeleton = replicate_people(skeleton, person_axis=1)
        data_numpy = skeleton.coordinates.transpose(3, 0, 2, 1)
        label = self.labels[index]
        label_feature = self.label_features[label]
        data_numpy = valid_crop_resize(
            data_numpy, valid_frame_num, self.p_interval, self.window_size
        )
        if self.random_rotation:
            data_numpy = random_rot(data_numpy)
        return data_numpy, label, label_feature


class NTU2DDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        label_features,
        mean_path,
        std_path,
        split="xsub_train",
        p_interval=(0.95,),
        window_size=128,
        normalization=True,
        score_threshold=0.0,
    ):
        self.data_path = str(data_path)
        self.label_features = label_features.detach().cpu().numpy()
        self.mean_path = str(mean_path)
        self.std_path = str(std_path)
        self.split = split
        self.window_size = window_size
        self.p_interval = list(p_interval)
        self.normalization = normalization
        self.score_threshold = float(score_threshold)
        if self.score_threshold < 0.0:
            raise ValueError("score_threshold must be non-negative")
        self._load_data()

    def _load_data(self):
        with open(self.data_path, "rb") as file:
            data = pickle.load(file)

        split, annotations = data["split"], data["annotations"]
        selected = set(split[self.split])
        self.data = [item for item in annotations if item["frame_dir"] in selected]
        if self.data:
            labels = [item["label"] for item in self.data]
            min_label = min(labels)
            max_label = max(labels)
            if min_label < 0:
                raise ValueError(f"NTU-2D contains negative label {min_label}")
            if max_label >= len(self.label_features):
                raise ValueError(
                    f"NTU-2D contains label {max_label}, but only "
                    f"{len(self.label_features)} text labels were loaded"
                )
        self.mean = np.load(self.mean_path)
        self.std = np.load(self.std_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        source = np.asarray(item["keypoint"], dtype=np.float32)
        if source.ndim != 4 or source.shape[-2:] != (17, 2):
            raise ValueError(f"unexpected NTU-2D keypoint shape {source.shape}")

        if "keypoint_score" in item:
            confidence = np.asarray(item["keypoint_score"], dtype=np.float32)
            if confidence.shape != source.shape[:-1]:
                raise ValueError("NTU-2D keypoint and confidence shapes do not match")
            observed = confidence > self.score_threshold
        else:
            observed = np.abs(source).sum(axis=-1) > 1e-8
        observed &= np.isfinite(source).all(axis=-1)

        if self.normalization:
            source = (source - self.mean) / self.std

        source = source.transpose(1, 0, 2, 3)
        observed = observed.transpose(1, 0, 2)
        skeleton = canonicalize(source, "coco17", observed)
        skeleton = replicate_people(skeleton, person_axis=1)
        data_numpy = skeleton.coordinates.transpose(3, 0, 2, 1)

        label = item["label"]
        label_feature = self.label_features[label]
        valid_frame_num = item["total_frames"]
        data_numpy = valid_crop_resize(
            data_numpy, valid_frame_num, self.p_interval, self.window_size
        )
        return data_numpy, label, label_feature
