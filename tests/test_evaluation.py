import tempfile
import unittest
from pathlib import Path

import torch

from eval import load_checkpoint, resolve_device
from models.multi_stream_alignment import MultiStreamAlignmentModel


class EvaluationTest(unittest.TestCase):
    def _build_small_model(self):
        return MultiStreamAlignmentModel(
            temporal_input_size=2 * 31 * 3,
            spatial_input_size=8 * 3,
            hidden_size=8,
            num_heads=1,
            num_layers=1,
        )

    def test_cpu_device_is_supported(self):
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_raw_state_dict_checkpoint_loads_strictly(self):
        source = self._build_small_model()
        target = self._build_small_model()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            torch.save(source.state_dict(), checkpoint)
            loaded_path = load_checkpoint(target, checkpoint)

        self.assertEqual(loaded_path, checkpoint.resolve())
        for source_parameter, target_parameter in zip(
            source.parameters(), target.parameters()
        ):
            torch.testing.assert_close(source_parameter, target_parameter)

    def test_checkpoint_with_state_dict_key_loads(self):
        source = self._build_small_model()
        target = self._build_small_model()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            torch.save({"state_dict": source.state_dict()}, checkpoint)
            load_checkpoint(target, checkpoint)

        for source_parameter, target_parameter in zip(
            source.parameters(), target.parameters()
        ):
            torch.testing.assert_close(source_parameter, target_parameter)


if __name__ == "__main__":
    unittest.main()
