import torch

from third_party import clip


def _load_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.rstrip().lstrip() for line in file.readlines()]


def load_label_texts(ntu120_label_map_path, humanml3d_label_map_path):
    ntu120_labels = _load_lines(ntu120_label_map_path)
    humanml3d_labels = _load_lines(humanml3d_label_map_path)

    ntu120_tokens = torch.cat([clip.tokenize(label) for label in ntu120_labels])
    humanml3d_tokens = torch.cat([clip.tokenize(label) for label in humanml3d_labels])
    return ntu120_tokens, humanml3d_tokens


class TextCLIP(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.float()

    def forward(self, text):
        return self.model.encode_text(text)
