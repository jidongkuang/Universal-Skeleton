import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import TrainConfig
from utils.text_prompt import load_label_texts


class ConfigAndTextTest(unittest.TestCase):
    def test_ntu60_profile_uses_cross_subject_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            config = TrainConfig(ntu_num_classes=60)

        self.assertEqual(config.ntu_name, "ntu60")
        self.assertEqual(config.ntu_3d_path.name, "NTU60_CS.npz")
        self.assertEqual(config.ntu_3d_cache_dir.name, "ntu60_cs")
        self.assertEqual(config.ntu_2d_path.name, "ntu60_2d.pkl")

    def test_generic_path_override_takes_precedence(self):
        with patch.dict(
            os.environ,
            {
                "HOV_NTU_3D": "/tmp/generic.npz",
                "HOV_NTU60_3D": "/tmp/profile.npz",
            },
            clear=True,
        ):
            config = TrainConfig(ntu_num_classes=60)

        self.assertEqual(config.ntu_3d_path, Path("/tmp/generic.npz"))

    def test_unsupported_ntu_class_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be one of"):
            TrainConfig(ntu_num_classes=61)

    def test_temporal_segments_must_evenly_partition_window(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            TrainConfig(window_size=127, num_temporal_segments=4)

    def test_ntu60_text_candidates_are_first_60_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ntu_path = root / "ntu.txt"
            humanml3d_path = root / "humanml3d.txt"
            ntu_path.write_text("\n".join(f"action {i}" for i in range(120)))
            humanml3d_path.write_text(
                "\n".join(f"motion {i}" for i in range(400))
            )

            ntu_tokens, humanml3d_tokens = load_label_texts(
                ntu_path, humanml3d_path, 60
            )

        self.assertEqual(ntu_tokens.shape, (60, 77))
        self.assertEqual(humanml3d_tokens.shape, (400, 77))

    def test_short_ntu_label_map_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ntu_path = root / "ntu.txt"
            humanml3d_path = root / "humanml3d.txt"
            ntu_path.write_text("walk\nrun\n")
            humanml3d_path.write_text(
                "\n".join(f"motion {i}" for i in range(400))
            )

            with self.assertRaisesRegex(ValueError, "2 labels"):
                load_label_texts(ntu_path, humanml3d_path, 60)

    def test_humanml3d_label_count_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ntu_path = root / "ntu.txt"
            humanml3d_path = root / "humanml3d.txt"
            ntu_path.write_text("\n".join(f"action {i}" for i in range(120)))
            humanml3d_path.write_text("walk\nrun\n")

            with self.assertRaisesRegex(ValueError, "has 2 labels"):
                load_label_texts(ntu_path, humanml3d_path, 60)

    def test_empty_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ntu_path = root / "ntu.txt"
            humanml3d_path = root / "humanml3d.txt"
            ntu_path.write_text("walk\n\nrun\n")
            humanml3d_path.write_text(
                "\n".join(f"motion {i}" for i in range(400))
            )

            with self.assertRaisesRegex(ValueError, "empty label at line 2"):
                load_label_texts(ntu_path, humanml3d_path, 2)


if __name__ == "__main__":
    unittest.main()
