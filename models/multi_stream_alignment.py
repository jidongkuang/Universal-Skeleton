import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from datasets.unified_skeleton import (
    BONE_PARENTS,
    NUM_UNIFIED_JOINTS,
    NUM_UNIFIED_PEOPLE,
)


class ContrastiveLoss(nn.Module):
    def __init__(self, tau=0.4):
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = tau

    def similarity(self, first, second):
        first = F.normalize(first)
        second = F.normalize(second)
        return torch.mm(first, second.t())

    def semi_loss(self, first, second):
        reflection_similarity = torch.exp(
            self.similarity(first, first) / self.tau
        )
        cross_similarity = torch.exp(
            self.similarity(first, second) / self.tau
        )
        positive_similarity = cross_similarity.diag()
        denominator = (
            reflection_similarity.sum(1)
            + cross_similarity.sum(1)
            - reflection_similarity.diag()
        )
        return -torch.log(positive_similarity / denominator)

    def forward(self, first, second, mean=True):
        loss = (
            self.semi_loss(first, second) + self.semi_loss(second, first)
        ) * 0.5
        return loss.mean() if mean else loss


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b]", stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class ModalityEmbedding(nn.Module):
    def __init__(self, temporal_input_size, spatial_input_size, hidden_size):
        super().__init__()
        self.temporal_embedding = nn.Sequential(
            nn.Linear(temporal_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
        )
        self.spatial_embedding = nn.Sequential(
            nn.Linear(spatial_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, temporal_src, spatial_src):
        return (
            self.temporal_embedding(temporal_src),
            self.spatial_embedding(spatial_src),
        )


class EmbeddingFusion(nn.Module):
    def __init__(self, temporal_input_size, spatial_input_size, hidden_size):
        super().__init__()
        self.temporal_fusion = nn.Linear(temporal_input_size, hidden_size, bias=False)
        self.spatial_fusion = nn.Linear(spatial_input_size, hidden_size, bias=False)

    def forward(self, temporal_src, spatial_src):
        return self.temporal_fusion(temporal_src), self.spatial_fusion(spatial_src)


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, num_spatial_tokens):
        super().__init__()
        self.position_encoding = PositionalEncoding(hidden_size)
        self.spatial_position = torch.nn.Parameter(
            torch.zeros(1, num_spatial_tokens, hidden_size)
        )

        temporal_layer = TransformerEncoderLayer(
            hidden_size,
            num_heads,
            hidden_size,
            batch_first=True,
            dropout=0.0,
        )
        self.temporal_encoder_1 = TransformerEncoder(temporal_layer, num_layers)
        self.temporal_encoder_2 = TransformerEncoder(temporal_layer, num_layers)

        spatial_layer = TransformerEncoderLayer(
            hidden_size,
            num_heads,
            hidden_size,
            batch_first=True,
            dropout=0.0,
        )
        self.spatial_encoder_1 = TransformerEncoder(spatial_layer, num_layers)
        self.spatial_encoder_2 = TransformerEncoder(spatial_layer, num_layers)

    def forward(self, temporal_src, spatial_src):
        temporal_out = self.temporal_encoder_1(self.position_encoding(temporal_src))
        temporal_out = self.temporal_encoder_2(
            self.position_encoding(temporal_out) + temporal_src
        )
        temporal_global = temporal_out.amax(dim=1)

        batch_size = temporal_src.shape[0]
        spatial_position = self.spatial_position.expand(batch_size, -1, -1)
        spatial_out = self.spatial_encoder_1(spatial_src + spatial_position)
        spatial_out = self.spatial_encoder_2(
            spatial_out + spatial_position + spatial_src
        )
        spatial_global = spatial_out.amax(dim=1)

        fused = torch.cat([temporal_global, spatial_global], dim=1)
        return fused, temporal_global, spatial_global, temporal_out, spatial_out


class BaseEncoder(nn.Module):
    def __init__(
        self,
        temporal_input_size,
        spatial_input_size,
        hidden_size,
        num_heads,
        num_layers,
        num_spatial_tokens,
    ):
        super().__init__()
        self.joint_embedding = ModalityEmbedding(
            temporal_input_size, spatial_input_size, hidden_size
        )
        self.bone_embedding = ModalityEmbedding(
            temporal_input_size, spatial_input_size, hidden_size
        )
        self.motion_embedding = ModalityEmbedding(
            temporal_input_size, spatial_input_size, hidden_size
        )
        self.fusion = EmbeddingFusion(hidden_size, hidden_size, hidden_size)
        self.encoder = SpatioTemporalTransformer(
            hidden_size, num_heads, num_layers, num_spatial_tokens
        )

    def forward(
        self,
        joint_temporal,
        joint_spatial,
        bone_temporal,
        bone_spatial,
        motion_temporal,
        motion_spatial,
    ):
        joint_temporal, joint_spatial = self.joint_embedding(
            joint_temporal, joint_spatial
        )
        bone_temporal, bone_spatial = self.bone_embedding(
            bone_temporal, bone_spatial
        )
        motion_temporal, motion_spatial = self.motion_embedding(
            motion_temporal, motion_spatial
        )

        fused_temporal = (joint_temporal + bone_temporal + motion_temporal) / 3
        fused_spatial = (joint_spatial + bone_spatial + motion_spatial) / 3
        fused_temporal, fused_spatial = self.fusion(fused_temporal, fused_spatial)
        return self.encoder(fused_temporal, fused_spatial)


class MultiStreamAlignmentModel(nn.Module):
    def __init__(
        self,
        temporal_input_size,
        spatial_input_size,
        hidden_size,
        num_heads,
        num_layers,
        num_joints=NUM_UNIFIED_JOINTS,
        num_people=NUM_UNIFIED_PEOPLE,
    ):
        super().__init__()
        self.embedding_dim = 2 * hidden_size
        self.num_joints = int(num_joints)
        self.num_people = int(num_people)
        if self.num_joints != len(BONE_PARENTS):
            raise ValueError("num_joints must match the unified skeleton topology")
        self.bone_parents = BONE_PARENTS

        self.backbone = BaseEncoder(
            temporal_input_size,
            spatial_input_size,
            hidden_size,
            num_heads,
            num_layers,
            self.num_people * self.num_joints,
        )
        self.global_projector = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(True),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(True),
            nn.Linear(self.embedding_dim, 768),
        )
        self.temporal_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 768),
        )
        self.spatial_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 768),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _build_modality(self, data_input, modality="joint"):
        n, c, t, v, m = data_input.shape
        if v != self.num_joints or m != self.num_people:
            raise ValueError(
                f"expected {self.num_joints} joints and {self.num_people} people, "
                f"got {v} joints and {m} people"
            )
        if modality == "joint":
            temporal = data_input.permute(0, 2, 4, 3, 1).reshape(n, t, m * v * c)
            spatial = data_input.permute(0, 4, 3, 2, 1).reshape(n, m * v, t * c)
        elif modality == "bone":
            bone = torch.zeros_like(data_input)
            for joint, parent in enumerate(self.bone_parents):
                if joint != parent:
                    bone[:, :, :, joint, :] = (
                        data_input[:, :, :, joint, :] - data_input[:, :, :, parent, :]
                    )
            temporal = bone.permute(0, 2, 4, 3, 1).reshape(n, t, m * v * c)
            spatial = bone.permute(0, 4, 3, 2, 1).reshape(n, m * v, t * c)
        elif modality == "motion":
            motion = torch.zeros_like(data_input)
            motion[:, :, :-1, :, :] = (
                data_input[:, :, 1:, :, :] - data_input[:, :, :-1, :, :]
            )
            temporal = motion.permute(0, 2, 4, 3, 1).reshape(n, t, m * v * c)
            spatial = motion.permute(0, 4, 3, 2, 1).reshape(n, m * v, t * c)
        else:
            raise ValueError(f"unsupported modality {modality!r}")
        return temporal, spatial

    def forward(self, data, text_features=None):
        joint_temporal, joint_spatial = self._build_modality(data, "joint")
        bone_temporal, bone_spatial = self._build_modality(data, "bone")
        motion_temporal, motion_spatial = self._build_modality(data, "motion")
        outputs = self.backbone(
            joint_temporal,
            joint_spatial,
            bone_temporal,
            bone_spatial,
            motion_temporal,
            motion_spatial,
        )
        fused, temporal_global, spatial_global, temporal_tokens, spatial_tokens = (
            outputs
        )
        visual = self.global_projector(fused)
        return (
            visual,
            text_features,
            temporal_global,
            spatial_global,
            temporal_tokens,
            spatial_tokens,
        )
