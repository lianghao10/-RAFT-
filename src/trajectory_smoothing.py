from __future__ import annotations

import numpy as np


def moving_average(curve: np.ndarray, radius: int) -> np.ndarray:
    """使用边缘填充的滑动平均滤波平滑一维运动曲线。"""
    if curve.size == 0 or radius <= 0:
        return curve.copy()
    window_size = 2 * radius + 1
    padded = np.pad(curve, (radius, radius), mode="edge")
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    return np.convolve(padded, kernel, mode="same")[radius:-radius]


def smooth_trajectory(trajectory: np.ndarray, radius: int = 15) -> np.ndarray:
    if trajectory.size == 0:
        return trajectory.copy()
    smoothed = np.zeros_like(trajectory, dtype=np.float32)
    for i in range(trajectory.shape[1]):
        smoothed[:, i] = moving_average(trajectory[:, i], radius)
    return smoothed


def smooth_transforms(transforms: np.ndarray, radius: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将逐帧运动累积为轨迹，平滑轨迹后再得到新的逐帧补偿参数。"""
    if transforms.size == 0:
        empty = transforms.astype(np.float32)
        return empty, empty, empty
    trajectory = np.cumsum(transforms, axis=0).astype(np.float32)
    smoothed_trajectory = smooth_trajectory(trajectory, radius)
    difference = smoothed_trajectory - trajectory
    return (transforms + difference).astype(np.float32), trajectory, smoothed_trajectory


def transform_to_params(matrix: np.ndarray) -> np.ndarray:
    """从 2x3 仿射矩阵中提取水平位移 dx、垂直位移 dy 和旋转角 da。"""
    dx = float(matrix[0, 2])
    dy = float(matrix[1, 2])
    da = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    return np.array([dx, dy, da], dtype=np.float32)


def params_to_transform(params: np.ndarray) -> np.ndarray:
    """将 dx、dy、da 还原成 OpenCV warpAffine 使用的 2x3 矩阵。"""
    dx, dy, da = [float(v) for v in params]
    cos_a = np.cos(da)
    sin_a = np.sin(da)
    return np.array([[cos_a, -sin_a, dx], [sin_a, cos_a, dy]], dtype=np.float32)
