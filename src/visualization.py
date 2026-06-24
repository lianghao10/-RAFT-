from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_trajectory_plots(
    trajectory: np.ndarray,
    smoothed: np.ndarray,
    output_dir: str | Path,
    video_stem: str,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = ["dx", "dy", "da"]
    for idx, name in enumerate(names):
        plt.figure(figsize=(9, 4))
        plt.plot(trajectory[:, idx], label="raw", linewidth=1.2)
        plt.plot(smoothed[:, idx], label="smoothed", linewidth=1.5)
        plt.title(f"{video_stem} {name} trajectory")
        plt.xlabel("Frame")
        plt.ylabel(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"{video_stem}_{name}.png", dpi=150)
        plt.close()


def make_frame_compare(
    original: list[np.ndarray],
    traditional: list[np.ndarray] | None,
    raft: list[np.ndarray] | None,
    output_path: str | Path,
    max_frames: int = 4,
) -> None:
    candidates = [("Original", original), ("Traditional EIS", traditional), ("RAFT", raft)]
    rows = [(title, frames) for title, frames in candidates if frames]
    if not rows:
        return

    frame_count = min(len(original), max_frames)
    indices = np.linspace(0, len(original) - 1, frame_count, dtype=int)
    tiles = []
    for title, frames in rows:
        selected = []
        for idx in indices:
            frame = frames[min(idx, len(frames) - 1)].copy()
            cv2.putText(frame, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            selected.append(frame)
        tiles.append(np.concatenate(selected, axis=1))
    compare = np.concatenate(tiles, axis=0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), compare)
