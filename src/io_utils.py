from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def list_videos(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    return sorted(p for p in root.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)


def read_video(path: str | Path, resize_width: int | None = None) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if resize_width and resize_width > 0 and frame.shape[1] > resize_width:
            scale = resize_width / frame.shape[1]
            frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if len(frames) < 2:
        raise ValueError("At least two frames are required for stabilization.")
    return frames, float(fps)


def write_video(path: str | Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError("No frames to write.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise OSError(f"Cannot create video writer: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
