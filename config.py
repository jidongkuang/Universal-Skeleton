from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from datasets.unified_skeleton import (
    NUM_COORDINATE_CHANNELS,
    NUM_UNIFIED_JOINTS,
    NUM_UNIFIED_PEOPLE,
)
from utils.paths import env_int, env_path, repo_path


SUPPORTED_NTU_CLASS_COUNTS = (60, 120)


def _profile_path(generic_env: str, profile_env: str, default: Path) -> Path:
    return env_path(generic_env, env_path(profile_env, default))


@dataclass
class TrainConfig:
    seed: int = 0
    device: int = env_int("HOV_DEVICE", 0)
    epochs: int = 400
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    window_size: int = 128
    hidden_size: int = 1024
    num_heads: int = 1
    num_layers: int = 2
    train_workers: int = 16
    eval_workers: int = 16
    eval_batch_size: int = 64
    save_checkpoints: bool = True
    normalization: bool = True
    clip_model_name: str = "ViT-L/14@336px"
    output_dir: Path = field(
        default_factory=lambda: env_path(
            "HOV_OUTPUT_DIR", repo_path("work_dirs", "hov_main")
        )
    )
    ntu_num_classes: int = field(
        default_factory=lambda: env_int("HOV_NTU_NUM_CLASSES", 120)
    )
    ntu_3d_path: Optional[Path] = None
    ntu_3d_cache_dir: Optional[Path] = None
    ntu_2d_path: Optional[Path] = None
    ntu_2d_mean_path: Optional[Path] = None
    ntu_2d_std_path: Optional[Path] = None
    humanml3d_root: Path = field(
        default_factory=lambda: env_path(
            "HUMANML3D_ROOT", repo_path("data", "HumanML3D")
        )
    )
    humanml3d_metadata_root: Path = field(
        default_factory=lambda: env_path(
            "HOV_HUMANML3D_METADATA_ROOT", repo_path("data", "humanml3d")
        )
    )
    label_group_path: Path = field(
        default_factory=lambda: repo_path(
            "data", "annotations", "humanml3d_label_groups.json"
        )
    )
    ntu_label_map_path: Path = field(
        default_factory=lambda: env_path(
            "HOV_NTU_LABEL_MAP",
            repo_path("data", "text", "ntu120_label_map.txt"),
        )
    )
    humanml3d_label_map_path: Path = field(
        default_factory=lambda: repo_path(
            "data", "text", "humanml3d400_label_map.txt"
        )
    )
    humanml3d_train_split: str = "train"
    humanml3d_eval_split: str = "val"
    ntu3d_train_split: str = "train"
    ntu3d_eval_split: str = "test"
    ntu2d_train_split: str = "xsub_train"
    ntu2d_eval_split: str = "xsub_val"
    num_temporal_segments: int = 4

    def __post_init__(self):
        if self.ntu_num_classes not in SUPPORTED_NTU_CLASS_COUNTS:
            raise ValueError(
                f"ntu_num_classes must be one of {SUPPORTED_NTU_CLASS_COUNTS}, "
                f"got {self.ntu_num_classes}"
            )
        if self.num_temporal_segments <= 0:
            raise ValueError("num_temporal_segments must be positive")
        if self.window_size % self.num_temporal_segments != 0:
            raise ValueError(
                "window_size must be divisible by num_temporal_segments"
            )

        if self.ntu_num_classes == 60:
            defaults = {
                "ntu_3d_path": repo_path("data", "ntu", "NTU60_CS.npz"),
                "ntu_3d_cache_dir": repo_path("data", "cache", "ntu60_cs"),
                "ntu_2d_path": repo_path("data", "nturgb", "ntu60_2d.pkl"),
                "ntu_2d_mean_path": repo_path("data", "nturgb", "ntu60_2d_Mean.npy"),
                "ntu_2d_std_path": repo_path("data", "nturgb", "ntu60_2d_Std.npy"),
            }
        else:
            defaults = {
                "ntu_3d_path": repo_path("data", "ntu", "NTU120_CSub.npz"),
                "ntu_3d_cache_dir": repo_path("data", "cache", "ntu120_csub"),
                "ntu_2d_path": repo_path("data", "nturgb", "ntu120_hrnet.pkl"),
                "ntu_2d_mean_path": repo_path("data", "nturgb", "ntu120_2d_Mean.npy"),
                "ntu_2d_std_path": repo_path("data", "nturgb", "ntu120_2d_Std.npy"),
            }

        env_suffixes = {
            "ntu_3d_path": "3D",
            "ntu_3d_cache_dir": "CACHE_DIR",
            "ntu_2d_path": "2D",
            "ntu_2d_mean_path": "2D_MEAN",
            "ntu_2d_std_path": "2D_STD",
        }
        for field_name, suffix in env_suffixes.items():
            configured = getattr(self, field_name)
            if configured is None:
                configured = _profile_path(
                    f"HOV_NTU_{suffix}",
                    f"HOV_NTU{self.ntu_num_classes}_{suffix}",
                    defaults[field_name],
                )
            setattr(self, field_name, Path(configured).expanduser())

    @property
    def ntu_name(self) -> str:
        return f"ntu{self.ntu_num_classes}"

    @property
    def num_joints(self) -> int:
        return NUM_UNIFIED_JOINTS

    @property
    def num_people(self) -> int:
        return NUM_UNIFIED_PEOPLE

    @property
    def temporal_input_size(self) -> int:
        return self.num_people * self.num_joints * NUM_COORDINATE_CHANNELS

    @property
    def spatial_input_size(self) -> int:
        return self.window_size * NUM_COORDINATE_CHANNELS
