import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from datasets.humanml3d import _unify_smpl_motion
from datasets.ntu import NTU2DDataset, NTU3DDataset


class DatasetShapeTest(unittest.TestCase):
    def test_humanml3d_motion_is_unified_and_replicated(self):
        motion = np.arange(6 * 22 * 3, dtype=np.float32).reshape(6, 22, 3)
        output = _unify_smpl_motion(motion)
        self.assertEqual(output.shape, (3, 6, 31, 2))
        np.testing.assert_array_equal(output[..., 0], output[..., 1])

    def test_ntu3d_loader_returns_unified_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ntu.npz"
            source = np.zeros((1, 8, 2, 25, 3), dtype=np.float32)
            source[:, :, 0] = 1.0
            flattened = source.reshape(1, 8, -1)
            labels = np.ones((1, 1), dtype=np.float32)
            np.savez(path, x_train=flattened, y_train=labels, x_test=flattened, y_test=labels)

            dataset = NTU3DDataset(
                path,
                torch.zeros(1, 768),
                split="train",
                p_interval=(1.0,),
                window_size=8,
            )
            output, label, label_feature = dataset[0]
            self.assertEqual(output.shape, (3, 8, 31, 2))
            self.assertEqual(label, 0)
            self.assertEqual(label_feature.shape, (768,))
            np.testing.assert_array_equal(output[..., 0], output[..., 1])

    def test_ntu3d_loader_refreshes_truncated_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ntu.npz"
            cache_dir = root / "cache"
            cache_dir.mkdir()
            source = np.ones((1, 8, 2, 25, 3), dtype=np.float32)
            labels = np.ones((1, 1), dtype=np.float32)
            np.savez(
                path,
                x_train=source.reshape(1, 8, -1),
                y_train=labels,
                x_test=source.reshape(1, 8, -1),
                y_test=labels,
            )
            (cache_dir / "x_train.npy").write_bytes(b"truncated")

            dataset = NTU3DDataset(
                path,
                torch.zeros(1, 768),
                split="train",
                p_interval=(1.0,),
                window_size=8,
                cache_dir=cache_dir,
            )

            self.assertEqual(len(dataset), 1)
            self.assertGreater((cache_dir / "x_train.npy").stat().st_size, 9)

    def test_ntu3d_loader_rejects_non_one_hot_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ntu.npz"
            source = np.ones((1, 8, 2, 25, 3), dtype=np.float32)
            labels = np.ones((1, 2), dtype=np.float32)
            np.savez(
                path,
                x_train=source.reshape(1, 8, -1),
                y_train=labels,
                x_test=source.reshape(1, 8, -1),
                y_test=labels,
            )

            with self.assertRaisesRegex(ValueError, "must be one-hot"):
                NTU3DDataset(
                    path,
                    torch.zeros(2, 768),
                    split="train",
                    p_interval=(1.0,),
                    window_size=8,
                )

    def test_ntu2d_loader_returns_unified_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "ntu2d.pkl"
            mean_path = root / "mean.npy"
            std_path = root / "std.npy"
            keypoint = np.ones((1, 8, 17, 2), dtype=np.float32)
            payload = {
                "split": {"xsub_train": ["sample"]},
                "annotations": [
                    {
                        "frame_dir": "sample",
                        "keypoint": keypoint,
                        "keypoint_score": np.ones(keypoint.shape[:-1], dtype=np.float32),
                        "label": 0,
                        "total_frames": 8,
                    }
                ],
            }
            with open(data_path, "wb") as file:
                pickle.dump(payload, file)
            np.save(mean_path, np.zeros((17, 2), dtype=np.float32))
            np.save(std_path, np.ones((17, 2), dtype=np.float32))

            dataset = NTU2DDataset(
                data_path,
                torch.zeros(1, 768),
                mean_path,
                std_path,
                split="xsub_train",
                p_interval=(1.0,),
                window_size=8,
            )
            output, label, label_feature = dataset[0]
            self.assertEqual(output.shape, (3, 8, 31, 2))
            self.assertEqual(label, 0)
            self.assertEqual(label_feature.shape, (768,))
            self.assertTrue(np.all(output[2] == 0.0))
            np.testing.assert_array_equal(output[..., 0], output[..., 1])


if __name__ == "__main__":
    unittest.main()
