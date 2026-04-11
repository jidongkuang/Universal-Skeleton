import codecs as cs
import json
import random
from os.path import join as pjoin

import numpy as np
import orjson
import torch
from tqdm import tqdm

from datasets import param_util
from datasets.transforms import valid_crop_resize


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
    ):
        self.data_root = str(data_root)
        self.label_features = label_features.numpy()
        self.window_size = window_size
        self.p_interval = list(p_interval)
        self.unit_length = unit_length
        self.normalization = normalization
        self.max_motion_length = 196
        self.max_length = 20
        self.pointer = 0
        self.dataset_name = "t2m"
        self.joint_sequence = [
            0, 3, 12, 15, 13,
            16, 18, 20, 14, 17,
            19, 21, 1, 4, 7,
            10, 2, 5, 8, 11,
            9, 22, 22, 22, 22,
            22, 22, 22, 22, 22, 6, 22, 22,
        ]
        self.motion_dir = pjoin(self.data_root, "new_joints")
        self.text_dir = pjoin(self.data_root, "texts")
        self.meta_dir = pjoin(self.data_root, "mean_std")
        self.split_file = pjoin(self.data_root, "split", f"new_{split}_longtail.txt")

        with open(pjoin(self.data_root, "annotations_actions_400.json"), "rb") as ff:
            self.annotations_actions = orjson.loads(ff.read())

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
                            try:
                                n_motion = motion[int(f_tag * fps): int(to_tag * fps)]
                                if len(n_motion) < min_motion_len or len(n_motion) >= 200:
                                    continue
                                new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                while new_name in data_dict:
                                    new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                data_dict[new_name] = {"motion": n_motion, "length": len(n_motion), "text": [text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except Exception:
                                continue

                if flag:
                    data_dict[name] = {"motion": motion, "length": len(motion), "text": text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception:
                continue

        self.name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
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

        data_numpy = np.expand_dims(motion, axis=0)
        data_numpy = np.pad(data_numpy, ((0, 1), (0, 0), (0, 11), (0, 0)), mode="constant")
        data_numpy = data_numpy[:, :, self.joint_sequence, :]
        data_numpy = data_numpy.transpose(3, 1, 2, 0)
        label_feature = self.label_features[label]
        data_numpy = valid_crop_resize(data_numpy, m_length, self.p_interval, self.window_size)
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
    ):
        self.data_root = str(data_root)
        self.window_size = window_size
        self.p_interval = list(p_interval)
        self.unit_length = unit_length
        self.normalization = normalization
        self.max_motion_length = 196
        self.max_length = 20
        self.pointer = 0
        self.joint_sequence = [
            0, 3, 12, 15, 13,
            16, 18, 20, 14, 17,
            19, 21, 1, 4, 7,
            10, 2, 5, 8, 11,
            9, 22, 22, 22, 22,
            22, 22, 22, 22, 22, 6, 22, 22,
        ]
        self.motion_dir = pjoin(self.data_root, "new_joints")
        self.text_dir = pjoin(self.data_root, "texts")
        self.meta_dir = pjoin(self.data_root, "mean_std")
        self.split_file = pjoin(self.data_root, "split", f"new_{split}_longtail.txt")

        with open(pjoin(self.data_root, "annotations_actions_400.json"), "rb") as ff:
            self.annotations_actions = orjson.loads(ff.read())

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
                        text_dict = {"label_list": action["labels"], "tokens": line_split[1].split(" ")}

                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * fps): int(to_tag * fps)]
                                if len(n_motion) < min_motion_len or len(n_motion) >= 200:
                                    continue
                                new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                while new_name in data_dict:
                                    new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                data_dict[new_name] = {"motion": n_motion, "length": len(n_motion), "text": [text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except Exception:
                                continue

                if flag:
                    data_dict[name] = {"motion": motion, "length": len(motion), "text": text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception:
                continue

        self.name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
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

        data_numpy = np.expand_dims(motion, axis=0)
        data_numpy = np.pad(data_numpy, ((0, 1), (0, 0), (0, 11), (0, 0)), mode="constant")
        data_numpy = data_numpy[:, :, self.joint_sequence, :]
        data_numpy = data_numpy.transpose(3, 1, 2, 0)
        data_numpy = valid_crop_resize(data_numpy, m_length, self.p_interval, self.window_size)
        return data_numpy, multi_hot
