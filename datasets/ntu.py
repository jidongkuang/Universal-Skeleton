import pickle

import numpy as np
import torch

from datasets.transforms import random_rot, valid_crop_resize


class NTU3DDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        label_features,
        split="train",
        p_interval=(0.95,),
        random_rotation=False,
        window_size=128,
    ):
        self.data_path = str(data_path)
        self.label_features = label_features.numpy()
        self.split = split
        self.window_size = window_size
        self.p_interval = list(p_interval)
        self.random_rotation = random_rotation
        self._load_data()

    def _load_data(self):
        npz_data = np.load(self.data_path, mmap_mode="r")
        if self.split == "train":
            self.data = npz_data["x_train"]
            self.labels = np.where(npz_data["y_train"] > 0)[1]
        elif self.split == "test":
            self.data = npz_data["x_test"]
            self.labels = np.where(npz_data["y_test"] > 0)[1]
        else:
            raise NotImplementedError("split only supports train/test")

        n, t, _ = self.data.shape
        self.data = self.data.reshape((n, t, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        self.data = np.pad(self.data, ((0, 0), (0, 0), (0, 0), (0, 8), (0, 0)), mode="constant")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.labels[index]
        label_feature = self.label_features[label]
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        data_numpy = valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
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
    ):
        self.data_path = str(data_path)
        self.label_features = label_features.numpy()
        self.mean_path = str(mean_path)
        self.std_path = str(std_path)
        self.split = split
        self.window_size = window_size
        self.p_interval = list(p_interval)
        self.normalization = normalization
        self.joint_sequence = [
            17, 17, 17, 17, 5,
            7, 9, 17, 6, 8,
            10, 17, 11, 13, 15,
            17, 12, 14, 16, 17,
            17, 17, 17, 17, 17,
            0, 1, 2, 3, 4, 17, 17, 17,
        ]
        self._load_data()

    def _load_data(self):
        with open(self.data_path, "rb") as file:
            data = pickle.load(file)

        split, annotations = data["split"], data["annotations"]
        selected = set(split[self.split])
        self.data = [item for item in annotations if item["frame_dir"] in selected]
        self.mean = np.load(self.mean_path)
        self.std = np.load(self.std_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        data_numpy = item["keypoint"]

        if self.normalization:
            data_numpy = (data_numpy - self.mean) / self.std

        if data_numpy.shape[0] == 1:
            data_numpy = np.pad(data_numpy, ((0, 1), (0, 0), (0, 16), (0, 1)), mode="constant")
        else:
            data_numpy = np.pad(data_numpy, ((0, 0), (0, 0), (0, 16), (0, 1)), mode="constant")

        data_numpy = data_numpy[:, :, self.joint_sequence, :]
        data_numpy = data_numpy.transpose(3, 1, 2, 0)

        label = item["label"]
        label_feature = self.label_features[label]
        valid_frame_num = item["total_frames"]
        data_numpy = valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        return data_numpy, label, label_feature
