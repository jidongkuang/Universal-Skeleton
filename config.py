from dataclasses import dataclass, field
from pathlib import Path

from datasets.unified_skeleton import (
    NUM_COORDINATE_CHANNELS,
    NUM_UNIFIED_JOINTS,
    NUM_UNIFIED_PEOPLE,
)
from utils.paths import env_int, env_path, repo_path


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
    output_dir: Path = field(default_factory=lambda: env_path("HOV_OUTPUT_DIR", repo_path("work_dirs", "hov_main")))
    ntu120_3d_path: Path = field(default_factory=lambda: env_path("HOV_NTU120_3D", repo_path("data", "ntu", "NTU120_CSub.npz")))
    ntu120_3d_cache_dir: Path = field(default_factory=lambda: env_path("HOV_NTU_CACHE_DIR", repo_path("data", "cache", "ntu120_csub")))
    ntu120_2d_path: Path = field(default_factory=lambda: env_path("HOV_NTU120_2D", repo_path("data", "nturgb", "ntu120_hrnet.pkl")))
    ntu120_2d_mean_path: Path = field(default_factory=lambda: env_path("HOV_NTU120_2D_MEAN", repo_path("data", "nturgb", "ntu120_2d_Mean.npy")))
    ntu120_2d_std_path: Path = field(default_factory=lambda: env_path("HOV_NTU120_2D_STD", repo_path("data", "nturgb", "ntu120_2d_Std.npy")))
    humanml3d_root: Path = field(default_factory=lambda: env_path("HUMANML3D_ROOT", repo_path("data", "HumanML3D")))
    label_group_path: Path = field(default_factory=lambda: repo_path("data", "annotations", "humanml3d_label_groups.json"))
    ntu120_label_map_path: Path = field(default_factory=lambda: repo_path("data", "text", "ntu120_label_map.txt"))
    humanml3d_label_map_path: Path = field(default_factory=lambda: repo_path("data", "text", "humanml3d400_label_map.txt"))
    humanml3d_train_split: str = "train"
    humanml3d_eval_split: str = "val"
    ntu3d_train_split: str = "train"
    ntu3d_eval_split: str = "test"
    ntu2d_train_split: str = "xsub_train"
    ntu2d_eval_split: str = "xsub_val"
    num_temporal_segments: int = 4

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
