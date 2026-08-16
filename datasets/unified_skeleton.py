"""Unified 31-joint skeleton mapping and fixed kinematic imputation."""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np


UNIFIED_JOINTS: tuple[str, ...] = (
    "pelvis",
    "spine",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "shoulder_center",
    "left_hand_tip",
    "left_thumb",
    "right_hand_tip",
    "right_thumb",
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "upper_spine",
)
NUM_UNIFIED_JOINTS = len(UNIFIED_JOINTS)
NUM_UNIFIED_PEOPLE = 2
NUM_COORDINATE_CHANNELS = 3


# Source index -> unified index, matching Table SI in the supplementary material.
SOURCE_TO_UNIFIED: Mapping[str, tuple[tuple[int, int], ...]] = {
    "kinect_v1_20": (
        (0, 0), (1, 1), (2, 20), (3, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (8, 4), (9, 5), (10, 6), (11, 7),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (16, 12), (17, 13), (18, 14), (19, 15),
    ),
    "kinect_v2_25": tuple((index, index) for index in range(25)),
    "coco17": (
        (0, 25), (1, 26), (2, 27), (3, 28), (4, 29),
        (5, 4), (6, 8), (7, 5), (8, 9), (9, 6), (10, 10),
        (11, 12), (12, 16), (13, 13), (14, 17), (15, 14), (16, 18),
    ),
    "smpl22": (
        (0, 0), (1, 12), (2, 16), (3, 1), (4, 13), (5, 17),
        (6, 30), (7, 14), (8, 18), (9, 20), (10, 15), (11, 19),
        (12, 2), (15, 3), (16, 4), (17, 8), (18, 5), (19, 9),
        (20, 6), (21, 10),
    ),
}

SOURCE_JOINT_COUNTS: Mapping[str, int] = {
    "kinect_v1_20": 20,
    "kinect_v2_25": 25,
    "coco17": 17,
    "smpl22": 22,
}

BODY_PARTS: Mapping[str, tuple[int, ...]] = {
    "head": (2, 3, 25, 26, 27, 28, 29),
    "arms": (4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 24),
    "torso": (0, 1, 20, 30),
    "legs": (12, 13, 14, 15, 16, 17, 18, 19),
}

# Parent for each joint in the kinematic tree used to construct bone features.
BONE_PARENTS: tuple[int, ...] = (
    0, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14,
    0, 16, 17, 18, 30, 7, 7, 11, 11, 3, 25, 25, 26, 27, 1,
)

RHO_HAND = 0.25
RHO_FOOT = 0.15
RHO_HAND_TIP = 0.18
RHO_THUMB = 0.32


@dataclass(frozen=True)
class UnifiedSkeleton:
    coordinates: np.ndarray
    observed: np.ndarray
    imputed: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        return np.logical_or(self.observed, self.imputed)


def _validate_source(
    coordinates: np.ndarray,
    source_format: str,
    observed: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if source_format not in SOURCE_TO_UNIFIED:
        raise KeyError(
            f"unknown source format {source_format!r}; expected one of "
            f"{sorted(SOURCE_TO_UNIFIED)}"
        )

    source = np.asarray(coordinates, dtype=np.float32)
    if source.ndim < 2:
        raise ValueError("coordinates must include joint and coordinate axes")
    expected_joints = SOURCE_JOINT_COUNTS[source_format]
    if source.shape[-2] != expected_joints:
        raise ValueError(
            f"{source_format} has {source.shape[-2]} joints, expected {expected_joints}"
        )
    if source.shape[-1] not in (2, 3):
        raise ValueError("coordinate dimension must be 2 or 3")

    finite = np.isfinite(source).all(axis=-1)
    if observed is None:
        if source_format in {"smpl22", "kinect_v1_20"}:
            source_observed = finite
        else:
            source_observed = finite & (np.abs(source).sum(axis=-1) > 1e-8)
    else:
        source_observed = np.asarray(observed, dtype=bool)
        if source_observed.shape != source.shape[:-1]:
            raise ValueError(
                f"observed mask shape {source_observed.shape} does not match "
                f"coordinates {source.shape[:-1]}"
            )
        source_observed = source_observed & finite

    source = np.where(source_observed[..., None], source, 0.0)
    if source.shape[-1] == 2:
        source = np.pad(source, [(0, 0)] * (source.ndim - 1) + [(0, 1)])
    return source.astype(np.float32, copy=False), source_observed


def _write_estimate(
    coordinates: np.ndarray,
    valid: np.ndarray,
    imputed: np.ndarray,
    target: int,
    estimate: np.ndarray,
    anchors_valid: np.ndarray,
) -> None:
    write = (~valid[..., target]) & anchors_valid
    coordinates[..., target, :] = np.where(
        write[..., None], estimate, coordinates[..., target, :]
    )
    imputed[..., target] |= write
    valid[..., target] |= write


def _midpoint(
    coordinates: np.ndarray,
    valid: np.ndarray,
    imputed: np.ndarray,
    target: int,
    first: int,
    second: int,
) -> None:
    estimate = 0.5 * (coordinates[..., first, :] + coordinates[..., second, :])
    _write_estimate(
        coordinates,
        valid,
        imputed,
        target,
        estimate,
        valid[..., first] & valid[..., second],
    )


def _extend(
    coordinates: np.ndarray,
    valid: np.ndarray,
    imputed: np.ndarray,
    target: int,
    anchor: int,
    parent: int,
    ratio: float,
) -> None:
    estimate = coordinates[..., anchor, :] + ratio * (
        coordinates[..., anchor, :] - coordinates[..., parent, :]
    )
    _write_estimate(
        coordinates,
        valid,
        imputed,
        target,
        estimate,
        valid[..., anchor] & valid[..., parent],
    )


def _copy(
    coordinates: np.ndarray,
    valid: np.ndarray,
    imputed: np.ndarray,
    target: int,
    source: int,
) -> None:
    _write_estimate(
        coordinates,
        valid,
        imputed,
        target,
        coordinates[..., source, :],
        valid[..., source],
    )


def _mean_observed(
    coordinates: np.ndarray,
    valid: np.ndarray,
    imputed: np.ndarray,
    target: int,
    sources: Sequence[int],
) -> None:
    indices = np.asarray(tuple(sources), dtype=np.int64)
    source_valid = valid[..., indices]
    count = source_valid.sum(axis=-1)
    estimate = (
        coordinates[..., indices, :] * source_valid[..., None]
    ).sum(axis=-2) / np.maximum(count[..., None], 1)
    _write_estimate(coordinates, valid, imputed, target, estimate, count > 0)


def kinematic_impute(skeleton: UnifiedSkeleton) -> UnifiedSkeleton:
    """Apply the fixed-order imputation program from Appendix A."""

    coordinates = skeleton.coordinates.copy()
    observed = skeleton.observed.copy()
    imputed = skeleton.imputed.copy()
    valid = skeleton.valid.copy()

    _midpoint(coordinates, valid, imputed, 0, 12, 16)
    _midpoint(coordinates, valid, imputed, 20, 4, 8)
    _midpoint(coordinates, valid, imputed, 1, 0, 20)
    _midpoint(coordinates, valid, imputed, 30, 1, 20)
    _mean_observed(coordinates, valid, imputed, 3, (25, 26, 27, 28, 29))
    _midpoint(coordinates, valid, imputed, 2, 20, 3)

    _extend(coordinates, valid, imputed, 7, 6, 5, RHO_HAND)
    _extend(coordinates, valid, imputed, 11, 10, 9, RHO_HAND)
    _extend(coordinates, valid, imputed, 15, 14, 13, RHO_FOOT)
    _extend(coordinates, valid, imputed, 19, 18, 17, RHO_FOOT)
    _extend(coordinates, valid, imputed, 21, 7, 6, RHO_HAND_TIP)
    _extend(coordinates, valid, imputed, 22, 7, 6, RHO_THUMB)
    _extend(coordinates, valid, imputed, 23, 11, 10, RHO_HAND_TIP)
    _extend(coordinates, valid, imputed, 24, 11, 10, RHO_THUMB)

    for target in (25, 26, 27, 28, 29):
        _copy(coordinates, valid, imputed, target, 3)

    coordinates = np.where(valid[..., None], coordinates, 0.0).astype(np.float32)
    return UnifiedSkeleton(coordinates, observed, imputed)


def canonicalize(
    coordinates: np.ndarray,
    source_format: str,
    observed: Optional[np.ndarray] = None,
    *,
    impute: bool = True,
) -> UnifiedSkeleton:
    """Map a source skeleton to the unified 31-joint, 3-channel layout."""

    source, source_observed = _validate_source(coordinates, source_format, observed)
    output_shape = source.shape[:-2] + (NUM_UNIFIED_JOINTS, NUM_COORDINATE_CHANNELS)
    output = np.zeros(output_shape, dtype=np.float32)
    unified_observed = np.zeros(output_shape[:-1], dtype=bool)

    for source_index, target_index in SOURCE_TO_UNIFIED[source_format]:
        present = source_observed[..., source_index]
        output[..., target_index, :] = np.where(
            present[..., None], source[..., source_index, :], 0.0
        )
        unified_observed[..., target_index] = present

    skeleton = UnifiedSkeleton(
        coordinates=output,
        observed=unified_observed,
        imputed=np.zeros_like(unified_observed),
    )
    return kinematic_impute(skeleton) if impute else skeleton


def replicate_people(
    skeleton: UnifiedSkeleton,
    *,
    max_people: int = NUM_UNIFIED_PEOPLE,
    person_axis: int = -3,
) -> UnifiedSkeleton:
    """Replicate one observed person to fill the fixed member dimension."""

    if max_people < 1:
        raise ValueError("max_people must be positive")
    axis = person_axis if person_axis >= 0 else skeleton.coordinates.ndim + person_axis
    if axis < 0 or axis >= skeleton.coordinates.ndim - 2:
        raise ValueError("person axis cannot be the joint or coordinate axis")

    mask_axis = axis
    reduce_axes = tuple(index for index in range(skeleton.valid.ndim) if index != mask_axis)
    active = np.flatnonzero(skeleton.valid.any(axis=reduce_axes))
    if len(active) > max_people:
        active = active[:max_people]

    if len(active) == 0:
        indices = np.zeros(max_people, dtype=np.int64)
    elif len(active) == 1:
        indices = np.repeat(active, max_people)
    else:
        indices = active

    coordinates = np.take(skeleton.coordinates, indices, axis=axis)
    observed = np.take(skeleton.observed, indices, axis=mask_axis)
    imputed = np.take(skeleton.imputed, indices, axis=mask_axis)

    if len(active) == 0:
        coordinates[...] = 0.0
        observed[...] = False
        imputed[...] = False
    return UnifiedSkeleton(coordinates, observed, imputed)


def _validate_constants() -> None:
    if NUM_UNIFIED_JOINTS != 31:
        raise ValueError("the unified skeleton must contain exactly 31 joints")
    if len(BONE_PARENTS) != NUM_UNIFIED_JOINTS:
        raise ValueError("bone parent table must cover all unified joints")
    partition = [joint for part in BODY_PARTS.values() for joint in part]
    if sorted(partition) != list(range(NUM_UNIFIED_JOINTS)):
        raise ValueError("body parts must partition all unified joints exactly once")
    for source_format, mapping in SOURCE_TO_UNIFIED.items():
        source_indices = [source for source, _ in mapping]
        target_indices = [target for _, target in mapping]
        if len(source_indices) != len(set(source_indices)):
            raise ValueError(f"duplicate source mapping in {source_format}")
        if len(target_indices) != len(set(target_indices)):
            raise ValueError(f"duplicate target mapping in {source_format}")


_validate_constants()
