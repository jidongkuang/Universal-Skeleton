import random

import numpy as np
import torch
import torch.nn.functional as F


def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    c, _, v, m = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias:end - bias, :, :]
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), 64), valid_size)
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]

    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(c * v * m, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(
        data,
        size=(c * v * m, window),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    data = data.contiguous().view(c, v, m, window).permute(0, 3, 1, 2).contiguous().numpy()
    return data


def _rot(rot):
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = torch.zeros(rot.shape[0], 1)
    ones = torch.ones(rot.shape[0], 1)

    r1 = torch.stack((ones, zeros, zeros), dim=-1)
    rx2 = torch.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), dim=-1)
    rx3 = torch.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), dim=-1)
    rx = torch.cat((r1, rx2, rx3), dim=1)

    ry1 = torch.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), dim=-1)
    r2 = torch.stack((zeros, ones, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), dim=-1)
    ry = torch.cat((ry1, r2, ry3), dim=1)

    rz1 = torch.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones), dim=-1)
    rz2 = torch.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim=1)

    return rz.matmul(ry).matmul(rx)


def random_rot(data_numpy, theta=0.3):
    data_torch = torch.from_numpy(data_numpy)
    c, t, v, m = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(t, c, v * m)
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot] * t, dim=0)
    rot = _rot(rot)
    data_torch = torch.matmul(rot, data_torch)
    return data_torch.view(t, c, v, m).permute(1, 0, 2, 3).contiguous().numpy()


def random_choose(data_numpy, size):
    _, t, _, _ = data_numpy.shape
    if t <= size:
        return data_numpy
    begin = random.randint(0, t - size)
    return data_numpy[:, begin:begin + size, :, :]
