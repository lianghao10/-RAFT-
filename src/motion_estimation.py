from __future__ import annotations

import cv2
import numpy as np


IDENTITY_AFFINE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def sample_flow_points(
    flow: np.ndarray,
    valid_mask: np.ndarray | None = None,
    sample_step: int = 8,
    random_samples: int | None = None,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = flow.shape[:2]
    if valid_mask is None:
        valid_mask = np.ones((h, w), dtype=bool)

    yy, xx = np.mgrid[0:h:sample_step, 0:w:sample_step]
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    keep = valid_mask[coords[:, 1], coords[:, 0]]
    coords = coords[keep]

    if random_samples and coords.shape[0] > random_samples:
        rng = np.random.default_rng(seed)
        coords = coords[rng.choice(coords.shape[0], random_samples, replace=False)]

    sampled_flow = flow[coords[:, 1], coords[:, 0]]
    finite = np.isfinite(sampled_flow).all(axis=1)
    coords = coords[finite].astype(np.float32)
    sampled_flow = sampled_flow[finite].astype(np.float32)
    return coords, coords + sampled_flow


def estimate_affine_from_flow(
    flow: np.ndarray,
    valid_mask: np.ndarray | None = None,
    sample_step: int = 8,
    ransac_threshold: float = 3.0,
    min_points: int = 12,
) -> tuple[np.ndarray, int]:
    src_pts, dst_pts = sample_flow_points(flow, valid_mask, sample_step)
    if src_pts.shape[0] < min_points:
        return IDENTITY_AFFINE.copy(), 0

    matrix, inliers = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
    )
    if matrix is None:
        return IDENTITY_AFFINE.copy(), 0
    return matrix.astype(np.float32), int(inliers.sum()) if inliers is not None else 0
