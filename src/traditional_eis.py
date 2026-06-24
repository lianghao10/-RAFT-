from __future__ import annotations

import time

import cv2
import numpy as np

from .motion_estimation import IDENTITY_AFFINE
from .trajectory_smoothing import transform_to_params


def estimate_lk_transforms(frames: list[np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    transforms: list[np.ndarray] = []
    start = time.perf_counter()

    for i in range(len(frames) - 1):
        prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3,
        )

        if prev_pts is None or len(prev_pts) < 8:
            matrix = IDENTITY_AFFINE.copy()
        else:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            if curr_pts is None or status is None:
                matrix = IDENTITY_AFFINE.copy()
            else:
                good_prev = prev_pts[status.ravel() == 1]
                good_curr = curr_pts[status.ravel() == 1]
                if len(good_prev) < 8:
                    matrix = IDENTITY_AFFINE.copy()
                else:
                    matrix, _ = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC)
                    if matrix is None:
                        matrix = IDENTITY_AFFINE.copy()
        transforms.append(transform_to_params(matrix))

    total = time.perf_counter() - start
    return np.asarray(transforms, dtype=np.float32), {"mean_flow_time": total / max(1, len(transforms))}
