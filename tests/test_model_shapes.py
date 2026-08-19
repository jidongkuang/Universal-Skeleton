import unittest

import torch

from models.multi_stream_alignment import MultiStreamAlignmentModel


class ModelShapeTest(unittest.TestCase):
    def setUp(self):
        self.model = MultiStreamAlignmentModel(
            temporal_input_size=2 * 31 * 3,
            spatial_input_size=8 * 3,
            hidden_size=8,
            num_heads=1,
            num_layers=1,
        ).eval()

    def test_forward_uses_31_joints_and_two_people(self):
        data = torch.randn(2, 3, 8, 31, 2)
        with torch.no_grad():
            visual, _, temporal, spatial, temporal_tokens, spatial_tokens = self.model(data)
        self.assertEqual(visual.shape, (2, 768))
        self.assertEqual(temporal.shape, (2, 8))
        self.assertEqual(spatial.shape, (2, 8))
        self.assertEqual(temporal_tokens.shape, (2, 8, 8))
        self.assertEqual(spatial_tokens.shape, (2, 62, 8))

    def test_bone_topology_places_upper_spine_in_trunk_chain(self):
        data = torch.zeros(1, 3, 8, 31, 2)
        data[:, 0] = torch.arange(31).view(1, 1, 31, 1)
        temporal, _ = self.model._build_modality(data, "bone")
        bone = temporal.reshape(1, 8, 2, 31, 3)
        self.assertTrue(torch.all(bone[..., 0, :] == 0))
        self.assertTrue(torch.all(bone[..., 20, 0] == -10))
        self.assertTrue(torch.all(bone[..., 30, 0] == 29))

    def test_legacy_33_joint_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 31 joints"):
            self.model(torch.randn(1, 3, 8, 33, 2))

    def test_unknown_modality_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported modality"):
            self.model._build_modality(
                torch.randn(1, 3, 8, 31, 2), "unknown"
            )


if __name__ == "__main__":
    unittest.main()
