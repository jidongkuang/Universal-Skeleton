import unittest

import numpy as np

from datasets.unified_skeleton import (
    BODY_PARTS,
    NUM_UNIFIED_JOINTS,
    SOURCE_TO_UNIFIED,
    canonicalize,
    replicate_people,
)


class UnifiedSkeletonTest(unittest.TestCase):
    def test_joint_count_mappings_and_parts(self):
        self.assertEqual(NUM_UNIFIED_JOINTS, 31)
        self.assertEqual(
            SOURCE_TO_UNIFIED["kinect_v2_25"],
            tuple((index, index) for index in range(25)),
        )
        self.assertEqual(
            SOURCE_TO_UNIFIED["coco17"],
            (
                (0, 25), (1, 26), (2, 27), (3, 28), (4, 29),
                (5, 4), (6, 8), (7, 5), (8, 9), (9, 6), (10, 10),
                (11, 12), (12, 16), (13, 13), (14, 17), (15, 14),
                (16, 18),
            ),
        )
        partition = [joint for part in BODY_PARTS.values() for joint in part]
        self.assertEqual(len(BODY_PARTS), 4)
        self.assertEqual(sorted(partition), list(range(31)))

    def test_coco_fixed_order_imputation(self):
        source = np.zeros((1, 1, 17, 2), dtype=np.float32)
        source[0, 0] = np.asarray(
            [
                (0.0, 6.2), (-0.2, 6.4), (0.2, 6.4),
                (-0.4, 6.2), (0.4, 6.2),
                (-2.0, 4.0), (2.0, 4.0),
                (-3.0, 3.0), (3.0, 3.0),
                (-4.0, 2.0), (4.0, 2.0),
                (-1.0, 0.0), (1.0, 0.0),
                (-1.0, -2.0), (1.0, -2.0),
                (-1.0, -4.0), (1.0, -4.0),
            ],
            dtype=np.float32,
        )
        observed = np.ones(source.shape[:-1], dtype=bool)
        result = canonicalize(source, "coco17", observed)
        joints = result.coordinates[0, 0]

        np.testing.assert_allclose(joints[0], (0.0, 0.0, 0.0))
        np.testing.assert_allclose(joints[20], (0.0, 4.0, 0.0))
        np.testing.assert_allclose(joints[1], (0.0, 2.0, 0.0))
        np.testing.assert_allclose(joints[30], (0.0, 3.0, 0.0))
        np.testing.assert_allclose(joints[3], (0.0, 6.28, 0.0), atol=1e-6)
        np.testing.assert_allclose(joints[2], (0.0, 5.14, 0.0), atol=1e-6)
        np.testing.assert_allclose(joints[7], (-4.25, 1.75, 0.0))
        np.testing.assert_allclose(joints[15], (-1.0, -4.3, 0.0), atol=1e-6)
        np.testing.assert_allclose(joints[21], (-4.295, 1.705, 0.0), atol=1e-6)
        np.testing.assert_allclose(joints[22], (-4.33, 1.67, 0.0), atol=1e-6)
        self.assertTrue(result.valid.all())
        self.assertTrue(np.all(result.coordinates[..., 2] == 0.0))

    def test_kinect_v1_mapping_matches_table_si(self):
        source = np.arange(20 * 3, dtype=np.float32).reshape(1, 20, 3)
        observed = np.ones(source.shape[:-1], dtype=bool)
        result = canonicalize(source, "kinect_v1_20", observed, impute=False)
        for source_index, target_index in SOURCE_TO_UNIFIED["kinect_v1_20"]:
            np.testing.assert_array_equal(
                result.coordinates[..., target_index, :], source[..., source_index, :]
            )

    def test_smpl_mapping_matches_table_si(self):
        source = np.arange(22 * 3, dtype=np.float32).reshape(1, 22, 3)
        observed = np.ones(source.shape[:-1], dtype=bool)
        result = canonicalize(source, "smpl22", observed, impute=False)
        for source_index, target_index in SOURCE_TO_UNIFIED["smpl22"]:
            np.testing.assert_array_equal(
                result.coordinates[..., target_index, :], source[..., source_index, :]
            )

    def test_observed_joints_are_never_overwritten(self):
        rng = np.random.default_rng(4)
        source = rng.normal(size=(2, 2, 25, 3)).astype(np.float32)
        observed = np.ones(source.shape[:-1], dtype=bool)
        result = canonicalize(source, "kinect_v2_25", observed)
        np.testing.assert_array_equal(result.coordinates[..., :25, :], source)
        self.assertFalse(result.imputed[..., :25].any())
        np.testing.assert_allclose(
            result.coordinates[..., 30, :],
            0.5 * (source[..., 1, :] + source[..., 20, :]),
        )
        for face_index in range(25, 30):
            np.testing.assert_array_equal(
                result.coordinates[..., face_index, :], source[..., 3, :]
            )

    def test_missing_anchors_remain_invalid(self):
        source = np.zeros((1, 1, 17, 2), dtype=np.float32)
        source[..., 0, :] = (1.0, 2.0)
        observed = np.zeros(source.shape[:-1], dtype=bool)
        observed[..., 0] = True
        result = canonicalize(source, "coco17", observed)
        self.assertFalse(result.valid[..., 0].item())
        self.assertFalse(result.valid[..., 7].item())
        self.assertTrue(result.valid[..., 3].item())
        np.testing.assert_array_equal(
            result.coordinates[..., 3, :],
            np.asarray([[[1.0, 2.0, 0.0]]], dtype=np.float32),
        )

    def test_single_person_is_replicated(self):
        source = np.zeros((3, 2, 25, 3), dtype=np.float32)
        source[:, 0] = 1.0
        observed = np.zeros(source.shape[:-1], dtype=bool)
        observed[:, 0] = True
        result = replicate_people(
            canonicalize(source, "kinect_v2_25", observed), person_axis=1
        )
        self.assertEqual(result.coordinates.shape, (3, 2, 31, 3))
        np.testing.assert_array_equal(result.coordinates[:, 0], result.coordinates[:, 1])
        np.testing.assert_array_equal(result.observed[:, 0], result.observed[:, 1])
        np.testing.assert_array_equal(result.imputed[:, 0], result.imputed[:, 1])

    def test_two_people_are_preserved(self):
        source = np.ones((1, 2, 25, 3), dtype=np.float32)
        source[:, 1] = 2.0
        observed = np.ones(source.shape[:-1], dtype=bool)
        canonical = canonicalize(source, "kinect_v2_25", observed)
        result = replicate_people(canonical, person_axis=1)
        np.testing.assert_array_equal(result.coordinates, canonical.coordinates)


if __name__ == "__main__":
    unittest.main()
