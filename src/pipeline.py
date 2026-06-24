from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .compensation import stabilize_frames
from .crop import crop_ratio_from_zoom
from .foreground_filter import ForegroundMasker
from .io_utils import read_video, write_video
from .metrics import append_metrics, stability_score
from .motion_estimation import estimate_affine_from_flow
from .raft_flow import RAFTFlowEstimator, save_flow_visualization
from .traditional_eis import estimate_lk_transforms
from .trajectory_smoothing import smooth_transforms, transform_to_params
from .visualization import save_trajectory_plots


def stabilize_traditional(
    input_path: str | Path,
    output_path: str | Path,
    results_dir: str | Path,
    resize: int | None = None,
    smoothing_radius: int = 15,
    crop_zoom_margin: float = 1.04,
) -> dict[str, object]:
    start_total = time.perf_counter()
    frames, source_fps = read_video(input_path, resize)
    transforms, timing = estimate_lk_transforms(frames)
    smoothed_transforms, trajectory, smoothed_trajectory = smooth_transforms(transforms, smoothing_radius)
    stabilized = stabilize_frames(frames, smoothed_transforms, crop_zoom_margin)
    write_video(output_path, stabilized, source_fps)

    video_stem = Path(input_path).stem
    save_trajectory_plots(trajectory, smoothed_trajectory, Path(results_dir) / "trajectories", video_stem)

    total_time = time.perf_counter() - start_total
    row = {
        "video_name": Path(input_path).name,
        "method": "traditional",
        "crop_ratio": f"{crop_ratio_from_zoom(crop_zoom_margin):.4f}",
        "stability_score": f"{stability_score(trajectory, smoothed_trajectory):.4f}",
        "mean_flow_time": f"{timing['mean_flow_time']:.6f}",
        "mean_total_time": f"{total_time / max(1, len(frames)):.6f}",
        "fps": f"{len(frames) / max(total_time, 1e-8):.3f}",
        "notes": "Shi-Tomasi + LK + affine RANSAC",
    }
    append_metrics(Path(results_dir) / "metrics.csv", row)
    return {"frames": stabilized, "metrics": row}


def estimate_raft_transforms(
    frames: list[np.ndarray],
    estimator: RAFTFlowEstimator,
    results_dir: str | Path,
    video_stem: str,
    mask_mode: str,
    sample_step: int,
    save_flow_every: int,
    border_ratio: float,
    percentile_low: float,
    percentile_high: float,
) -> tuple[np.ndarray, dict[str, float]]:
    masker = ForegroundMasker(mask_mode, border_ratio, percentile_low, percentile_high)
    transforms = []
    flow_times = []
    inlier_counts = []

    for i in tqdm(range(len(frames) - 1), desc="RAFT flow"):
        flow, flow_time = estimator.estimate(frames[i], frames[i + 1])
        flow_times.append(flow_time)

        if i == 0 or (save_flow_every > 0 and (i + 1) % save_flow_every == 0):
            flow_path = Path(results_dir) / "flow_vis" / f"{video_stem}_frame_{i + 1:04d}_flow.png"
            save_flow_visualization(flow, flow_path)

        valid_mask = masker.build(frames[i], flow)
        matrix, inlier_count = estimate_affine_from_flow(flow, valid_mask, sample_step)
        inlier_counts.append(inlier_count)
        transforms.append(transform_to_params(matrix))

    return np.asarray(transforms, dtype=np.float32), {
        "mean_flow_time": float(np.mean(flow_times)) if flow_times else 0.0,
        "mean_inliers": float(np.mean(inlier_counts)) if inlier_counts else 0.0,
    }


def stabilize_raft(
    input_path: str | Path,
    output_path: str | Path,
    results_dir: str | Path,
    raft_weights: str | None = None,
    raft_model: str = "small",
    raft_iters: int = 12,
    resize: int | None = 640,
    mask: str = "flow_percentile",
    sample_step: int = 8,
    smoothing_radius: int = 15,
    crop_zoom_margin: float = 1.04,
    save_flow_every: int = 50,
    border_ratio: float = 0.2,
    percentile_low: float = 10,
    percentile_high: float = 90,
) -> dict[str, object]:
    print(f"Processing video: {input_path}")
    start_total = time.perf_counter()
    frames, source_fps = read_video(input_path, resize)
    estimator = RAFTFlowEstimator(raft_weights, model_name=raft_model, iters=raft_iters)
    video_stem = Path(input_path).stem

    transforms, timing = estimate_raft_transforms(
        frames,
        estimator,
        results_dir,
        video_stem,
        mask,
        sample_step,
        save_flow_every,
        border_ratio,
        percentile_low,
        percentile_high,
    )
    smoothed_transforms, trajectory, smoothed_trajectory = smooth_transforms(transforms, smoothing_radius)
    stabilized = stabilize_frames(frames, smoothed_transforms, crop_zoom_margin)
    write_video(output_path, stabilized, source_fps)
    save_trajectory_plots(trajectory, smoothed_trajectory, Path(results_dir) / "trajectories", video_stem)

    total_time = time.perf_counter() - start_total
    row = {
        "video_name": Path(input_path).name,
        "method": f"raft_{mask}",
        "crop_ratio": f"{crop_ratio_from_zoom(crop_zoom_margin):.4f}",
        "stability_score": f"{stability_score(trajectory, smoothed_trajectory):.4f}",
        "mean_flow_time": f"{timing['mean_flow_time']:.6f}",
        "mean_total_time": f"{total_time / max(1, len(frames)):.6f}",
        "fps": f"{len(frames) / max(total_time, 1e-8):.3f}",
        "notes": f"RAFT {raft_model}, step={sample_step}, mean_inliers={timing['mean_inliers']:.1f}",
    }
    append_metrics(Path(results_dir) / "metrics.csv", row)
    return {"frames": stabilized, "metrics": row}
