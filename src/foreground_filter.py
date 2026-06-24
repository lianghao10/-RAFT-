from __future__ import annotations

import cv2
import numpy as np


class ForegroundMasker:
    """生成光流采样前的有效区域掩码，用于尽量保留背景运动。"""

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
            # 很多手持自拍视频中主体位于中心，边缘区域更可能代表背景运动。
            border = np.zeros((h, w), dtype=bool)
            bh = max(1, int(h * self.border_ratio))
            bw = max(1, int(w * self.border_ratio))
            border[:bh, :] = True
            border[h - bh :, :] = True
            border[:, :bw] = True
            border[:, w - bw :] = True
            mask &= border

        if self.mode in {"flow_percentile", "border_flow_percentile"}:
            # 过滤幅值过大或过小的异常光流，降低独立运动前景对全局运动估计的干扰。
            mag = np.linalg.norm(flow, axis=2)
            valid_mag = mag[np.isfinite(mag)]
            if valid_mag.size:
                lower = np.percentile(valid_mag, self.percentile_low)
                upper = np.percentile(valid_mag, self.percentile_high)
                mask &= (mag >= lower) & (mag <= upper)

        if self.mode == "mog2":
            # MOG2 将画面分为前景和背景，这里只保留背景区域参与仿射估计。
            fg_mask = self._mog2.apply(frame)
            mask &= fg_mask == 0

        return mask
