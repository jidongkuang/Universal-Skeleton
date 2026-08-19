import torch

from third_party import clip


HUMANML3D_NUM_CLASSES = 400


def _load_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file]
    empty_lines = [index for index, line in enumerate(lines, start=1) if not line]
    if empty_lines:
        raise ValueError(
            f"label map {file_path} contains an empty label at line {empty_lines[0]}"
        )
    return lines


def load_label_texts(ntu_label_map_path, humanml3d_label_map_path, ntu_num_classes):
    ntu_labels = _load_lines(ntu_label_map_path)
    humanml3d_labels = _load_lines(humanml3d_label_map_path)

    if len(ntu_labels) < ntu_num_classes:
        raise ValueError(
            f"NTU label map has {len(ntu_labels)} labels, but "
            f"{ntu_num_classes} are required"
        )
    ntu_labels = ntu_labels[:ntu_num_classes]
    if len(humanml3d_labels) != HUMANML3D_NUM_CLASSES:
        raise ValueError(
            f"HumanML3D label map has {len(humanml3d_labels)} labels, but "
            f"{HUMANML3D_NUM_CLASSES} are required"
        )

    ntu_tokens = torch.cat([clip.tokenize(label) for label in ntu_labels])
    humanml3d_tokens = torch.cat([clip.tokenize(label) for label in humanml3d_labels])
    return ntu_tokens, humanml3d_tokens


class TextCLIP(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.float()

    def forward(self, text):
        return self.model.encode_text(text)
