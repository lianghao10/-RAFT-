from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def high_freq_energy(trajectory: np.ndarray, window: int = 15) -> float:
    if trajectory.size == 0:
        return 0.0
    from .trajectory_smoothing import smooth_trajectory

    smooth = smooth_trajectory(trajectory, radius=window)
    residual = trajectory - smooth
    return float(np.mean(residual * residual))


def stability_score(raw_trajectory: np.ndarray, smooth_trajectory: np.ndarray) -> float:
    before = high_freq_energy(raw_trajectory)
    after = high_freq_energy(smooth_trajectory)
    if before <= 1e-8:
        return 1.0
    return float(1.0 - after / before)


def append_metrics(path: str | Path, row: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_name",
        "method",
        "crop_ratio",
        "stability_score",
        "mean_flow_time",
        "mean_total_time",
        "fps",
        "notes",
    ]
    exists = output.exists()
    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})
