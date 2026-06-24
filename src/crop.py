from __future__ import annotations

import cv2
import numpy as np


def fix_border(frame: np.ndarray, zoom_margin: float = 1.04) -> np.ndarray:
    """Scale the frame slightly around its center to hide warp borders."""
    h, w = frame.shape[:2]
    transform = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0, zoom_margin)
    return cv2.warpAffine(frame, transform, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def crop_ratio_from_zoom(zoom_margin: float) -> float:
    if zoom_margin <= 0:
        return 1.0
    return float(min(1.0, 1.0 / (zoom_margin * zoom_margin)))
