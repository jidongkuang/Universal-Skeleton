from __future__ import annotations

from collections.abc import Collection

import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.inference_mode()
def evaluate_single_label(
    model,
    loader,
    label_features,
    device,
    *,
    description="Eval dataset",
):
    """Evaluate top-1 recognition accuracy for a single-label dataset."""

    model.eval()
    label_features = F.normalize(label_features.to(device), dim=-1)
    total_correct = 0
    total_samples = 0

    for data, label, _ in tqdm(loader, desc=description):
        data = data.to(device=device, dtype=torch.float32, non_blocking=True)
        label = label.to(device=device, dtype=torch.long, non_blocking=True)
        visual, _, _, _, _, _ = model(data, None)
        logits = F.normalize(visual, dim=-1) @ label_features.t()
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == label).sum().item()
        total_samples += label.size(0)

    if total_samples == 0:
        raise ValueError("cannot evaluate an empty dataset")
    return total_correct / total_samples


@torch.inference_mode()
def evaluate_humanml3d(
    model,
    loader,
    label_features,
    device,
    head_ids: Collection[int],
    medium_ids: Collection[int],
    tail_ids: Collection[int],
    *,
    description="Eval HumanML3D",
):
    """Evaluate the HumanML3D multi-label and frequency-group metrics."""

    model.eval()
    label_features = F.normalize(label_features.to(device), dim=-1)
    head_ids = set(head_ids)
    medium_ids = set(medium_ids)
    tail_ids = set(tail_ids)

    total_correct = 0
    total_samples = 0
    group_totals = {"head": 0, "medium": 0, "tail": 0}
    group_correct = {"head": 0, "medium": 0, "tail": 0}

    for data, multi_hot in tqdm(loader, desc=description):
        data = data.to(device=device, dtype=torch.float32, non_blocking=True)
        multi_hot = multi_hot.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        visual, _, _, _, _, _ = model(data, None)
        logits = F.normalize(visual, dim=-1) @ label_features.t()
        predictions = logits.argmax(dim=1)
        correct_mask = multi_hot.gather(1, predictions.unsqueeze(1)).squeeze(1)
        total_correct += correct_mask.sum().item()
        total_samples += multi_hot.size(0)

        for predicted_label, is_correct in zip(
            predictions.tolist(), correct_mask.tolist()
        ):
            if predicted_label in head_ids:
                group = "head"
            elif predicted_label in medium_ids:
                group = "medium"
            elif predicted_label in tail_ids:
                group = "tail"
            else:
                continue
            group_totals[group] += 1
            group_correct[group] += int(is_correct > 0.5)

    if total_samples == 0:
        raise ValueError("cannot evaluate an empty HumanML3D dataset")

    metrics = {"overall_acc": total_correct / total_samples}
    for group in ("head", "medium", "tail"):
        total = group_totals[group]
        metrics[f"{group}_acc"] = group_correct[group] / total if total else 0.0
    return metrics
