import json
import os
import random
import time
from collections import OrderedDict
from math import cos, pi

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import TrainConfig
from datasets.humanml3d import HumanML3DEvalDataset, HumanML3DTrainDataset, load_label_groups
from datasets.ntu import NTU2DDataset, NTU3DDataset
from datasets.unified_skeleton import BODY_PARTS
from models.multi_stream_alignment import ContrastiveLoss, MultiStreamAlignmentModel
from third_party import clip
from utils.logger import build_logger
from utils.text_prompt import TextCLIP, load_label_texts


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(2)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.output_dir = cfg.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.output_dir))
        self.logger = build_logger(self.output_dir / "train.log")
        self.loss_fn = ContrastiveLoss()
        self.best_results = {
            "ntu3d": {"epoch": -1, "acc": -1.0},
            "ntu2d": {"epoch": -1, "acc": -1.0},
            "humanml3d": {
                "overall": {"epoch": -1, "acc": -1.0},
                "head": {"epoch": -1, "acc": -1.0},
                "medium": {"epoch": -1, "acc": -1.0},
                "tail": {"epoch": -1, "acc": -1.0},
            },
        }

    def load_text_features(self):
        ntu_tokens, humanml3d_tokens = load_label_texts(
            self.cfg.ntu_label_map_path,
            self.cfg.humanml3d_label_map_path,
            self.cfg.ntu_num_classes,
        )
        clip_model, _ = clip.load(self.cfg.clip_model_name, "cpu")
        del clip_model.visual
        text_encoder = TextCLIP(clip_model).cuda(self.device)

        with torch.no_grad():
            ntu_features = text_encoder(ntu_tokens.cuda(self.device)).detach().cpu()
            humanml3d_features = text_encoder(humanml3d_tokens.cuda(self.device)).detach().cpu()

        self.label_features = {
            "ntu": ntu_features,
            "humanml3d": humanml3d_features,
        }

    def load_data(self):
        self.head_ids, self.medium_ids, self.tail_ids = load_label_groups(self.cfg.label_group_path)

        self.datasets = {
            "ntu3d_train": NTU3DDataset(
                self.cfg.ntu_3d_path,
                self.label_features["ntu"],
                split=self.cfg.ntu3d_train_split,
                p_interval=(0.5, 1.0),
                random_rotation=True,
                window_size=self.cfg.window_size,
                cache_dir=self.cfg.ntu_3d_cache_dir,
            ),
            "ntu3d_test": NTU3DDataset(
                self.cfg.ntu_3d_path,
                self.label_features["ntu"],
                split=self.cfg.ntu3d_eval_split,
                p_interval=(0.95,),
                random_rotation=False,
                window_size=self.cfg.window_size,
                cache_dir=self.cfg.ntu_3d_cache_dir,
            ),
            "ntu2d_train": NTU2DDataset(
                self.cfg.ntu_2d_path,
                self.label_features["ntu"],
                self.cfg.ntu_2d_mean_path,
                self.cfg.ntu_2d_std_path,
                split=self.cfg.ntu2d_train_split,
                p_interval=(0.5, 1.0),
                window_size=self.cfg.window_size,
                normalization=self.cfg.normalization,
            ),
            "ntu2d_test": NTU2DDataset(
                self.cfg.ntu_2d_path,
                self.label_features["ntu"],
                self.cfg.ntu_2d_mean_path,
                self.cfg.ntu_2d_std_path,
                split=self.cfg.ntu2d_eval_split,
                p_interval=(0.95,),
                window_size=self.cfg.window_size,
                normalization=self.cfg.normalization,
            ),
            "humanml3d_train": HumanML3DTrainDataset(
                self.cfg.humanml3d_root,
                self.label_features["humanml3d"],
                split=self.cfg.humanml3d_train_split,
                window_size=self.cfg.window_size,
                p_interval=(0.5, 1.0),
                normalization=self.cfg.normalization,
            ),
            "humanml3d_test": HumanML3DEvalDataset(
                self.cfg.humanml3d_root,
                split=self.cfg.humanml3d_eval_split,
                window_size=self.cfg.window_size,
                p_interval=(0.95,),
                normalization=self.cfg.normalization,
            ),
        }

        train_dataset = ConcatDataset([
            self.datasets["ntu3d_train"],
            self.datasets["ntu2d_train"],
            self.datasets["humanml3d_train"],
        ])
        self.logger.info(
            "Dataset sizes: "
            f"{self.cfg.ntu_name}-3D train/test="
            f"{len(self.datasets['ntu3d_train'])}/{len(self.datasets['ntu3d_test'])}, "
            f"{self.cfg.ntu_name}-2D train/test="
            f"{len(self.datasets['ntu2d_train'])}/{len(self.datasets['ntu2d_test'])}, "
            "HumanML3D train/test="
            f"{len(self.datasets['humanml3d_train'])}/{len(self.datasets['humanml3d_test'])}"
        )

        self.data_loaders = {
            "train": DataLoader(train_dataset, batch_size=self.cfg.batch_size, num_workers=self.cfg.train_workers, shuffle=True),
            "ntu3d_test": DataLoader(self.datasets["ntu3d_test"], batch_size=self.cfg.eval_batch_size, num_workers=self.cfg.eval_workers, shuffle=False),
            "ntu2d_test": DataLoader(self.datasets["ntu2d_test"], batch_size=self.cfg.eval_batch_size, num_workers=self.cfg.eval_workers, shuffle=False),
            "humanml3d_test": DataLoader(self.datasets["humanml3d_test"], batch_size=self.cfg.eval_batch_size, num_workers=self.cfg.eval_workers, shuffle=False),
        }

    def load_model(self):
        self.model = MultiStreamAlignmentModel(
            self.cfg.temporal_input_size,
            self.cfg.spatial_input_size,
            self.cfg.hidden_size,
            self.cfg.num_heads,
            self.cfg.num_layers,
            self.cfg.num_joints,
            self.cfg.num_people,
        ).cuda(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)

    def adjust_learning_rate(self, epoch):
        if epoch < 15:
            lr = self.cfg.learning_rate * epoch / 15
        else:
            lr = (self.cfg.learning_rate) * (1 + cos(pi * (epoch - 15) / (self.cfg.epochs - 15))) / 2
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.writer.add_scalar("train/lr", lr, epoch)

    def save_model(self, save_path):
        if self.cfg.save_checkpoints:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k, v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, save_path)

    def train_one_epoch(self, epoch):
        self.model.train()
        self.adjust_learning_rate(epoch)
        running_loss = []

        for data, _, label_text in tqdm(self.data_loaders["train"], desc=f"Train {epoch}"):
            data = data.type(torch.FloatTensor).cuda(self.device)
            label_text = label_text.cuda(self.device)

            visual, text, visual_t, visual_s, t_out, s_out = self.model(data, label_text)
            visual_t = self.model.temporal_projector(visual_t)
            visual_s = self.model.spatial_projector(visual_s)

            loss_local = (self.loss_fn(visual_t, label_text) + self.loss_fn(visual_s, label_text)) * 0.5
            loss_ts_consistency = self.loss_fn(visual_t, visual_s)

            loss_temporal_segments = 0
            segment_len = t_out.shape[1] // self.cfg.num_temporal_segments
            for i in range(self.cfg.num_temporal_segments):
                temporal_segment = t_out[:, i * segment_len:(i + 1) * segment_len, :].amax(dim=1)
                projected_segment = self.model.temporal_projector(temporal_segment)
                loss_temporal_segments += self.loss_fn(projected_segment, text)
            loss_temporal_segments /= self.cfg.num_temporal_segments

            loss_spatial_parts = 0
            for person in range(self.cfg.num_people):
                for indices in BODY_PARTS.values():
                    token_indices = [
                        joint + person * self.cfg.num_joints for joint in indices
                    ]
                    part_features = s_out[:, token_indices, :].amax(dim=1)
                    projected_part = self.model.spatial_projector(part_features)
                    loss_spatial_parts += self.loss_fn(projected_part, text)
            loss_spatial_parts /= self.cfg.num_people * len(BODY_PARTS)

            loss_ts_parts = (loss_temporal_segments + loss_spatial_parts) / 2
            loss = self.loss_fn(visual, text) + loss_local + 0.2 * loss_ts_consistency + 0.5 * loss_ts_parts

            running_loss.append(loss.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        epoch_loss = torch.stack(running_loss).mean().item()
        self.writer.add_scalar("train/loss", epoch_loss, epoch)
        self.logger.info(f"epoch [{epoch}] loss: {epoch_loss:.4f}")

    def test_dataset(self, loader, label_features):
        self.model.eval()
        label_features = F.normalize(label_features.cuda(self.device), dim=-1)
        total_correct = 0
        total_samples = 0
        for data, label, _ in tqdm(loader, desc="Eval dataset"):
            data = data.type(torch.FloatTensor).cuda(self.device)
            label = label.type(torch.LongTensor).cuda(self.device)
            visual, _, _, _, _, _ = self.model(data, None)
            visual = F.normalize(visual, dim=-1)
            logits = visual @ label_features.t()
            preds = logits.argmax(dim=1)
            total_correct += (preds == label).sum().item()
            total_samples += label.size(0)
        return total_correct / total_samples

    def test_humanml3d(self, loader, label_features):
        self.model.eval()
        label_features = F.normalize(label_features.cuda(self.device), dim=-1)

        total_correct = 0
        total_samples = 0
        total_preds_in_head = correct_preds_in_head = 0
        total_preds_in_medium = correct_preds_in_medium = 0
        total_preds_in_tail = correct_preds_in_tail = 0

        for data, multi_hot in tqdm(loader, desc="Eval HumanML3D"):
            data = data.type(torch.FloatTensor).cuda(self.device)
            multi_hot = multi_hot.type(torch.float32).cuda(self.device)
            visual, _, _, _, _, _ = self.model(data, None)
            visual = F.normalize(visual, dim=-1)
            logits = visual @ label_features.t()
            preds = logits.argmax(dim=1)
            correct_mask = multi_hot.gather(1, preds.unsqueeze(1)).squeeze(1)
            total_correct += correct_mask.sum().item()
            total_samples += multi_hot.size(0)

            for i in range(multi_hot.size(0)):
                predicted_label_id = preds[i].item()
                is_correct = correct_mask[i].item() > 0.5
                if predicted_label_id in self.head_ids:
                    total_preds_in_head += 1
                    if is_correct:
                        correct_preds_in_head += 1
                elif predicted_label_id in self.medium_ids:
                    total_preds_in_medium += 1
                    if is_correct:
                        correct_preds_in_medium += 1
                elif predicted_label_id in self.tail_ids:
                    total_preds_in_tail += 1
                    if is_correct:
                        correct_preds_in_tail += 1

        return {
            "overall_acc": total_correct / total_samples if total_samples > 0 else 0.0,
            "head_acc": correct_preds_in_head / total_preds_in_head if total_preds_in_head > 0 else 0.0,
            "medium_acc": correct_preds_in_medium / total_preds_in_medium if total_preds_in_medium > 0 else 0.0,
            "tail_acc": correct_preds_in_tail / total_preds_in_tail if total_preds_in_tail > 0 else 0.0,
        }

    def evaluate(self, epoch):
        acc_ntu3d = self.test_dataset(self.data_loaders["ntu3d_test"], self.label_features["ntu"])
        acc_ntu2d = self.test_dataset(self.data_loaders["ntu2d_test"], self.label_features["ntu"])
        acc_humanml3d = self.test_humanml3d(self.data_loaders["humanml3d_test"], self.label_features["humanml3d"])

        self.writer.add_scalar("test/ntu3d", acc_ntu3d, epoch)
        self.writer.add_scalar("test/ntu2d", acc_ntu2d, epoch)
        self.writer.add_scalar("test/humanml3d", acc_humanml3d["overall_acc"], epoch)

        self.logger.info(f"epoch [{epoch}] NTU3D acc: {acc_ntu3d * 100:.4f}%")
        self.logger.info(f"epoch [{epoch}] NTU2D acc: {acc_ntu2d * 100:.4f}%")
        self.logger.info(
            f"epoch [{epoch}] HumanML3D overall/head/medium/tail: "
            f"{acc_humanml3d['overall_acc'] * 100:.4f}% / "
            f"{acc_humanml3d['head_acc'] * 100:.4f}% / "
            f"{acc_humanml3d['medium_acc'] * 100:.4f}% / "
            f"{acc_humanml3d['tail_acc'] * 100:.4f}%"
        )

        if acc_ntu3d > self.best_results["ntu3d"]["acc"]:
            self.best_results["ntu3d"] = {"epoch": epoch, "acc": acc_ntu3d}
            self.save_model(self.output_dir / "best_ntu3d.pth")
        if acc_ntu2d > self.best_results["ntu2d"]["acc"]:
            self.best_results["ntu2d"] = {"epoch": epoch, "acc": acc_ntu2d}
            self.save_model(self.output_dir / "best_ntu2d.pth")
        if acc_humanml3d["overall_acc"] > self.best_results["humanml3d"]["overall"]["acc"]:
            self.best_results["humanml3d"]["overall"] = {"epoch": epoch, "acc": acc_humanml3d["overall_acc"]}
            self.save_model(self.output_dir / "best_humanml3d_overall.pth")
        for key in ["head", "medium", "tail"]:
            metric_key = f"{key}_acc"
            if acc_humanml3d[metric_key] > self.best_results["humanml3d"][key]["acc"]:
                self.best_results["humanml3d"][key] = {"epoch": epoch, "acc": acc_humanml3d[metric_key]}

    def run(self):
        set_seed(self.cfg.seed)
        self.logger.info(json.dumps(self.cfg.__dict__, indent=4, default=str))
        self.load_text_features()
        self.load_data()
        self.load_model()

        start_time = time.time()
        for epoch in range(self.cfg.epochs):
            self.train_one_epoch(epoch)
            if epoch % 8 == 0 and epoch >= 10:
                with torch.no_grad():
                    self.evaluate(epoch)
        self.logger.info(f"Finished in {(time.time() - start_time) / 3600:.2f} hours")


def main():
    trainer = Trainer(TrainConfig())
    trainer.run()


if __name__ == "__main__":
    main()
