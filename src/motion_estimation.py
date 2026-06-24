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
    """从 RAFT 稠密光流中采样匹配点对 p=(x,y), p'=(x+u,y+v)。"""
    h, w = flow.shape[:2]
    if valid_mask is None:
        valid_mask = np.ones((h, w), dtype=bool)

    # 稠密光流包含每个像素的运动，直接全部用于 RANSAC 计算量太大；
    # 因此默认按网格间隔采样，兼顾速度、稳定性和可复现实验。
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
    """使用 RANSAC 从采样光流点中估计 2x3 全局仿射运动矩阵。"""
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
