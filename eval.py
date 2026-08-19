from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import SUPPORTED_NTU_CLASS_COUNTS, TrainConfig
from datasets.humanml3d import HumanML3DEvalDataset, load_label_groups
from datasets.ntu import NTU2DDataset, NTU3DDataset
from models.multi_stream_alignment import MultiStreamAlignmentModel
from third_party import clip
from utils.evaluation import evaluate_humanml3d, evaluate_single_label
from utils.text_prompt import TextCLIP, load_label_texts


DATASET_CHOICES = ("ntu3d", "ntu2d", "humanml3d")


def parse_args():
    default_device = (
        f"cuda:{os.environ.get('HOV_DEVICE', '0')}"
        if torch.cuda.is_available()
        else "cpu"
    )
    parser = argparse.ArgumentParser(
        description="Evaluate a unified skeleton action-recognition checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--ntu-num-classes",
        type=int,
        choices=SUPPORTED_NTU_CLASS_COUNTS,
        default=60,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_CHOICES,
        default=list(DATASET_CHOICES),
    )
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for a machine-readable copy of the metrics.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(value):
    value = f"cuda:{value}" if str(value).isdigit() else str(value)
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_checkpoint(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, Mapping) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint must contain a model state dictionary")

    model.load_state_dict(state_dict, strict=True)
    return checkpoint_path


def encode_label_features(cfg, device):
    ntu_tokens, humanml3d_tokens = load_label_texts(
        cfg.ntu_label_map_path,
        cfg.humanml3d_label_map_path,
        cfg.ntu_num_classes,
    )
    clip_model, _ = clip.load(cfg.clip_model_name, "cpu")
    del clip_model.visual
    text_encoder = TextCLIP(clip_model).to(device).eval()
    with torch.inference_mode():
        ntu_features = text_encoder(ntu_tokens.to(device)).cpu()
        humanml3d_features = text_encoder(humanml3d_tokens.to(device)).cpu()
    del text_encoder, clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"ntu": ntu_features, "humanml3d": humanml3d_features}


def build_eval_loaders(cfg, label_features, selected_datasets, device):
    loader_options = {
        "batch_size": cfg.eval_batch_size,
        "num_workers": cfg.eval_workers,
        "shuffle": False,
        "pin_memory": device.type == "cuda",
    }
    loaders = {}

    if "ntu3d" in selected_datasets:
        dataset = NTU3DDataset(
            cfg.ntu_3d_path,
            label_features["ntu"],
            split=cfg.ntu3d_eval_split,
            p_interval=(0.95,),
            random_rotation=False,
            window_size=cfg.window_size,
            cache_dir=cfg.ntu_3d_cache_dir,
        )
        loaders["ntu3d"] = DataLoader(dataset, **loader_options)

    if "ntu2d" in selected_datasets:
        dataset = NTU2DDataset(
            cfg.ntu_2d_path,
            label_features["ntu"],
            cfg.ntu_2d_mean_path,
            cfg.ntu_2d_std_path,
            split=cfg.ntu2d_eval_split,
            p_interval=(0.95,),
            window_size=cfg.window_size,
            normalization=cfg.normalization,
        )
        loaders["ntu2d"] = DataLoader(dataset, **loader_options)

    if "humanml3d" in selected_datasets:
        dataset = HumanML3DEvalDataset(
            cfg.humanml3d_root,
            split=cfg.humanml3d_eval_split,
            window_size=cfg.window_size,
            p_interval=(0.95,),
            normalization=cfg.normalization,
            metadata_root=cfg.humanml3d_metadata_root,
        )
        loaders["humanml3d"] = DataLoader(dataset, **loader_options)

    return loaders


def build_model(cfg):
    return MultiStreamAlignmentModel(
        cfg.temporal_input_size,
        cfg.spatial_input_size,
        cfg.hidden_size,
        cfg.num_heads,
        cfg.num_layers,
        cfg.num_joints,
        cfg.num_people,
    )


def evaluate(args):
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")

    set_seed(args.seed)
    device = resolve_device(args.device)
    selected_datasets = tuple(dict.fromkeys(args.datasets))
    cfg = TrainConfig(
        ntu_num_classes=args.ntu_num_classes,
        eval_batch_size=args.batch_size,
        eval_workers=args.workers,
    )

    print(f"Encoding label text with {cfg.clip_model_name} ...", flush=True)
    label_features = encode_label_features(cfg, device)
    loaders = build_eval_loaders(
        cfg, label_features, selected_datasets, device
    )

    model = build_model(cfg)
    checkpoint_path = load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    print(f"Loaded checkpoint: {checkpoint_path}", flush=True)

    metrics = {}
    if "ntu3d" in loaders:
        metrics[f"NTU-{cfg.ntu_num_classes} (3D) x-sub"] = 100.0 * (
            evaluate_single_label(
                model,
                loaders["ntu3d"],
                label_features["ntu"],
                device,
                description=f"Eval NTU-{cfg.ntu_num_classes} 3D",
            )
        )
    if "ntu2d" in loaders:
        metrics[f"NTU-{cfg.ntu_num_classes} (2D) x-sub"] = 100.0 * (
            evaluate_single_label(
                model,
                loaders["ntu2d"],
                label_features["ntu"],
                device,
                description=f"Eval NTU-{cfg.ntu_num_classes} 2D",
            )
        )
    if "humanml3d" in loaders:
        head_ids, medium_ids, tail_ids = load_label_groups(cfg.label_group_path)
        humanml3d = evaluate_humanml3d(
            model,
            loaders["humanml3d"],
            label_features["humanml3d"],
            device,
            head_ids,
            medium_ids,
            tail_ids,
        )
        metrics.update(
            {
                "HumanML3D overall": 100.0 * humanml3d["overall_acc"],
                "HumanML3D many-shot": 100.0 * humanml3d["head_acc"],
                "HumanML3D medium-shot": 100.0 * humanml3d["medium_acc"],
                "HumanML3D few-shot": 100.0 * humanml3d["tail_acc"],
            }
        )

    print("\nEvaluation results (%)")
    for name, value in metrics.items():
        print(f"{name:<32} {value:8.4f}")

    report = {
        "checkpoint": str(checkpoint_path),
        "ntu_num_classes": cfg.ntu_num_classes,
        "seed": args.seed,
        "metrics_percent": metrics,
    }
    if args.json_output is not None:
        output_path = args.json_output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved JSON report: {output_path}")
    return report


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
