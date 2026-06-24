from __future__ import annotations

import cv2
import numpy as np


class ForegroundMasker:
    """Build valid-pixel masks used before sampling dense optical flow."""

    def __init__(
        self,
        mode: str = "flow_percentile",
        border_ratio: float = 0.2,
        percentile_low: float = 10,
        percentile_high: float = 90,
    ) -> None:
        self.mode = mode
        self.border_ratio = border_ratio
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self._mog2 = cv2.createBackgroundSubtractorMOG2(history=200, detectShadows=True)

    def build(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        h, w = flow.shape[:2]
        mask = np.ones((h, w), dtype=bool)
        if self.mode == "none":
            return mask

        if self.mode in {"border", "border_flow_percentile"}:
            border = np.zeros((h, w), dtype=bool)
            bh = max(1, int(h * self.border_ratio))
            bw = max(1, int(w * self.border_ratio))
            border[:bh, :] = True
            border[h - bh :, :] = True
            border[:, :bw] = True
            border[:, w - bw :] = True
            mask &= border

        if self.mode in {"flow_percentile", "border_flow_percentile"}:
            mag = np.linalg.norm(flow, axis=2)
            valid_mag = mag[np.isfinite(mag)]
            if valid_mag.size:
                lower = np.percentile(valid_mag, self.percentile_low)
                upper = np.percentile(valid_mag, self.percentile_high)
                mask &= (mag >= lower) & (mag <= upper)

        if self.mode == "mog2":
            fg_mask = self._mog2.apply(frame)
            mask &= fg_mask == 0

        return mask
