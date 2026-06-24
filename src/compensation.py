from __future__ import annotations

import cv2
import numpy as np

from .crop import fix_border
from .trajectory_smoothing import params_to_transform


def stabilize_frames(
    frames: list[np.ndarray],
    smoothed_transforms: np.ndarray,
    zoom_margin: float = 1.04,
) -> list[np.ndarray]:
    """根据平滑后的运动参数对每帧执行仿射补偿，并通过轻微放大隐藏黑边。"""
    if not frames:
        return []
    h, w = frames[0].shape[:2]
    output = [fix_border(frames[0], zoom_margin)]
    for idx, transform_params in enumerate(smoothed_transforms):
        matrix = params_to_transform(transform_params)
        stabilized = cv2.warpAffine(
            frames[idx + 1],
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        output.append(fix_border(stabilized, zoom_margin))
    return output
