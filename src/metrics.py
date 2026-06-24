from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime

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
        "input_video",
        "output_video",
        "method",
        "raft_model",
        "mask_mode",
        "sample_step",
        "resize",
        "device",
        "crop_ratio",
        "stability_score",
        "mean_flow_time",
        "mean_total_time",
        "fps",
        "notes",
    ]
    exists = output.exists()
    if exists:
        first_line = output.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
        current_header = first_line[0].split(",") if first_line else []
        if current_header != fieldnames:
            backup = output.with_name(f"{output.stem}_legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}{output.suffix}")
            output.rename(backup)
            exists = False
    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def save_run_log(results_dir: str | Path, row: dict[str, object]) -> Path:
    """保存每次实验的参数和指标，方便后续论文整理实验表格。"""
    log_dir = Path(results_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_stem = Path(str(row.get("video_name", "video"))).stem
    method = str(row.get("method", "method")).replace("/", "_")
    output = log_dir / f"{timestamp}_{video_stem}_{method}.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
    return output
