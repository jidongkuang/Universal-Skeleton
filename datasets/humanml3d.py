import codecs as cs
import gzip
import json
import os
import random
import warnings
from collections import Counter
from os.path import join as pjoin

import numpy as np
import orjson
import torch
from tqdm import tqdm

from datasets import param_util
from datasets.transforms import valid_crop_resize
from datasets.unified_skeleton import canonicalize, replicate_people


EXPECTED_DATA_ERRORS = (IndexError, KeyError, OSError, ValueError)


def _warn_skipped_samples(split_file, skipped_errors):
    if not skipped_errors:
        return
    summary = ", ".join(
        f"{error_name}={count}"
        for error_name, count in sorted(skipped_errors.items())
    )
    warnings.warn(
        f"Skipped malformed HumanML3D samples listed in {split_file}: {summary}",
        stacklevel=2,
    )


def _unify_smpl_motion(motion):
    source = np.asarray(motion, dtype=np.float32)[:, None, :, :]
    if source.shape[-2:] != (22, 3):
        raise ValueError(f"unexpected HumanML3D motion shape {source.shape}")
    observed = np.isfinite(source).all(axis=-1)
    skeleton = canonicalize(source, "smpl22", observed)
    skeleton = replicate_people(skeleton, person_axis=1)
    return skeleton.coordinates.transpose(3, 0, 2, 1)


def _load_action_annotations(metadata_root):
    json_path = pjoin(metadata_root, "annotations_actions_400.json")
    gzip_path = f"{json_path}.gz"
    if os.path.isfile(json_path):
        opener = open
        annotation_path = json_path
    elif os.path.isfile(gzip_path):
        opener = gzip.open
        annotation_path = gzip_path
    else:
        raise FileNotFoundError(
            "HumanML3D action annotations not found at "
            f"{json_path} or {gzip_path}"
        )
    with opener(annotation_path, "rb") as file:
        return orjson.loads(file.read())


def load_label_groups(classification_file_path):
    with open(classification_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    head_ids = {int(item["label"]) for item in data.get("head", [])}
    medium_ids = {int(item["label"]) for item in data.get("medium", [])}
    tail_ids = {int(item["label"]) for item in data.get("tail", [])}
    return head_ids, medium_ids, tail_ids


class HumanML3DTrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        label_features,
        split="train",
        window_size=128,
        p_interval=(0.5, 1.0),
        unit_length=4,
        normalization=True,
        metadata_root=None,
    ):
        self.data_root = str(data_root)
        self.metadata_root = str(metadata_root or data_root)
        self.label_features = label_features.detach().cpu().numpy()
        self.window_size = window_size
        self.p_interval = list(p_interval)
        self.unit_length = unit_length
        self.normalization = normalization
        self.max_motion_length = 196
        self.max_length = 20
        self.pointer = 0
        self.dataset_name = "t2m"
        self.motion_dir = pjoin(self.data_root, "new_joints")
        self.text_dir = pjoin(self.data_root, "texts")
        self.meta_dir = pjoin(self.metadata_root, "mean_std")
        self.split_file = pjoin(
            self.metadata_root, "split", f"new_{split}_longtail.txt"
        )

        self.annotations_actions = _load_action_annotations(self.metadata_root)

        self.mean = np.load(pjoin(self.meta_dir, "new_Mean.npy"))
        self.std = np.load(pjoin(self.meta_dir, "new_Std.npy"))
        self.kinematic_chain = param_util.t2m_kinematic_chain
        self._build_index()

    def _build_index(self):
        data_dict = {}
        id_list = []
        with cs.open(self.split_file, "r") as file:
            for line in file.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        skipped_errors = Counter()
        min_motion_len = 40
        fps = 20

        for name in tqdm(id_list, desc="Loading HumanML3D train"):
            try:
                motion = np.load(pjoin(self.motion_dir, f"{name}.npy"))
                if len(motion) < min_motion_len or len(motion) >= 200:
                    continue

                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, f"{name}.txt")) as file:
                    action = self.annotations_actions[name]
                    for line_number, line in enumerate(file.readlines()):
                        line_split = line.strip().split("#")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        text_dict = {
                            "caption": action["annotations"][line_number]["processed_label_text"],
                            "label": action["annotations"][line_number]["label"],
                            "tokens": line_split[1].split(" "),
                        }

                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            motion_segment = motion[
                                int(f_tag * fps): int(to_tag * fps)
                            ]
                            if (
                                len(motion_segment) < min_motion_len
                                or len(motion_segment) >= 200
                            ):
                                continue
                            new_name = (
                                random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                + "_"
                                + name
                            )
                            while new_name in data_dict:
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                    + "_"
                                    + name
                                )
                            data_dict[new_name] = {
                                "motion": motion_segment,
                                "length": len(motion_segment),
                                "text": [text_dict],
                            }
                            new_name_list.append(new_name)
                            length_list.append(len(motion_segment))

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except EXPECTED_DATA_ERRORS as error:
                skipped_errors[type(error).__name__] += 1
                continue

        _warn_skipped_samples(self.split_file, skipped_errors)
        self.name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda item: item[1])
        )
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        self.pointer = np.searchsorted(self.length_arr, length)
        self.max_length = length

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        text_data = random.choice(text_list)
        label = random.choice(text_data["label"])

        coin = np.random.choice(["single", "single", "double"])
        if coin == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        if self.normalization:
            motion = (motion - self.mean) / self.std

        data_numpy = _unify_smpl_motion(motion)
        label_feature = self.label_features[label]
        data_numpy = valid_crop_resize(
            data_numpy, m_length, self.p_interval, self.window_size
        )
        return data_numpy, label, label_feature


class HumanML3DEvalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        split="val",
        window_size=128,
        p_interval=(0.95,),
        unit_length=4,
        normalization=True,
        metadata_root=None,
    ):
        self.data_root = str(data_root)
        self.metadata_root = str(metadata_root or data_root)
        self.window_size = window_size
        self.p_interval = list(p_interval)
        self.unit_length = unit_length
        self.normalization = normalization
        self.max_motion_length = 196
        self.max_length = 20
        self.pointer = 0
        self.motion_dir = pjoin(self.data_root, "new_joints")
        self.text_dir = pjoin(self.data_root, "texts")
        self.meta_dir = pjoin(self.metadata_root, "mean_std")
        self.split_file = pjoin(
            self.metadata_root, "split", f"new_{split}_longtail.txt"
        )

        self.annotations_actions = _load_action_annotations(self.metadata_root)

        self.mean = np.load(pjoin(self.meta_dir, "new_Mean.npy"))
        self.std = np.load(pjoin(self.meta_dir, "new_Std.npy"))
        self._build_index()

    def _build_index(self):
        data_dict = {}
        id_list = []
        with cs.open(self.split_file, "r") as file:
            for line in file.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        skipped_errors = Counter()
        min_motion_len = 40
        fps = 20

        for name in tqdm(id_list, desc="Loading HumanML3D eval"):
            try:
                motion = np.load(pjoin(self.motion_dir, f"{name}.npy"))
                if len(motion) < min_motion_len or len(motion) >= 200:
                    continue

                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, f"{name}.txt")) as file:
                    action = self.annotations_actions[name]
                    for _, line in enumerate(file.readlines()):
                        line_split = line.strip().split("#")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        text_dict = {
                            "label_list": action["labels"],
                            "tokens": line_split[1].split(" "),
                        }

                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            motion_segment = motion[
                                int(f_tag * fps): int(to_tag * fps)
                            ]
                            if (
                                len(motion_segment) < min_motion_len
                                or len(motion_segment) >= 200
                            ):
                                continue
                            new_name = (
                                random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                + "_"
                                + name
                            )
                            while new_name in data_dict:
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                    + "_"
                                    + name
                                )
                            data_dict[new_name] = {
                                "motion": motion_segment,
                                "length": len(motion_segment),
                                "text": [text_dict],
                            }
                            new_name_list.append(new_name)
                            length_list.append(len(motion_segment))

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except EXPECTED_DATA_ERRORS as error:
                skipped_errors[type(error).__name__] += 1
                continue

        _warn_skipped_samples(self.split_file, skipped_errors)
        self.name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda item: item[1])
        )
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        self.pointer = np.searchsorted(self.length_arr, length)
        self.max_length = length

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        text_data = random.choice(text_list)
        label_list = text_data["label_list"]

        multi_hot = np.zeros(400, dtype=np.float32)
        multi_hot[label_list] = 1.0

        coin = np.random.choice(["single", "single", "double"])
        if coin == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        if self.normalization:
            motion = (motion - self.mean) / self.std

        data_numpy = _unify_smpl_motion(motion)
        data_numpy = valid_crop_resize(data_numpy, m_length, self.p_interval, self.window_size)
        return data_numpy, multi_hot
