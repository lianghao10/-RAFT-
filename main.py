from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.io_utils import list_videos, read_video
from src.pipeline import stabilize_raft, stabilize_traditional
from src.visualization import make_frame_compare


def load_config(path: str | None) -> dict:
    """读取默认配置文件；命令行参数会覆盖这里的配置。"""
    if not path:
        path = "configs/default.yaml"
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="传统 EIS 与 RAFT 深度光流视频防抖程序。")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", choices=["traditional", "raft", "both"], default=None)
    parser.add_argument("--input", help="Single input video path.")
    parser.add_argument("--output", help="Single output video path.")
    parser.add_argument("--input_dir", help="Batch input directory.")
    parser.add_argument("--output_dir", default="data/output_videos")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--raft_weights", default=None)
    parser.add_argument("--raft_model", choices=["small", "large"], default=None)
    parser.add_argument("--raft_iters", type=int, default=None)
    parser.add_argument("--mask", choices=["none", "border", "flow_percentile", "border_flow_percentile", "mog2"], default=None)
    parser.add_argument("--sample_step", type=int, default=None)
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--smoothing_radius", type=int, default=None)
    parser.add_argument("--save_flow_every", type=int, default=None)
    parser.add_argument("--crop_zoom_margin", type=float, default=None)
    return parser.parse_args()


def cfg_value(args: argparse.Namespace, config: dict, key: str, default):
    """优先使用命令行参数；如果未传入，则读取 yaml 配置；最后使用默认值。"""
    value = getattr(args, key)
    return config.get(key, default) if value is None else value


def run_one(input_path: Path, output_path: Path | None, args: argparse.Namespace, config: dict) -> None:
    """处理单个视频，并根据 method 选择 traditional、raft 或 both。"""
    method = cfg_value(args, config, "method", "raft")
    results_dir = Path(args.results_dir)
    resize = cfg_value(args, config, "resize", 640)
    smoothing_radius = cfg_value(args, config, "smoothing_radius", 15)
    crop_zoom_margin = cfg_value(args, config, "crop_zoom_margin", 1.04)
    original_frames = None
    traditional_frames = None
    raft_frames = None

    if method in {"traditional", "both"}:
        trad_output = output_path if method == "traditional" else None
        trad_output = trad_output or Path(args.output_dir) / f"{input_path.stem}_traditional_stab.mp4"
        result = stabilize_traditional(input_path, trad_output, results_dir, resize, smoothing_radius, crop_zoom_margin)
        traditional_frames = result["frames"]

    if method in {"raft", "both"}:
        raft_output = output_path if method == "raft" else None
        raft_output = raft_output or Path(args.output_dir) / f"{input_path.stem}_raft_stab.mp4"
        result = stabilize_raft(
            input_path=input_path,
            output_path=raft_output,
            results_dir=results_dir,
            raft_weights=args.raft_weights,
            raft_model=cfg_value(args, config, "raft_model", "small"),
            raft_iters=cfg_value(args, config, "raft_iters", 12),
            resize=resize,
            mask=cfg_value(args, config, "mask", "flow_percentile"),
            sample_step=cfg_value(args, config, "sample_step", 8),
            smoothing_radius=smoothing_radius,
            crop_zoom_margin=crop_zoom_margin,
            save_flow_every=cfg_value(args, config, "save_flow_every", 50),
            border_ratio=config.get("border_ratio", 0.2),
            percentile_low=config.get("percentile_low", 10),
            percentile_high=config.get("percentile_high", 90),
        )
        raft_frames = result["frames"]

    if method == "both":
        # both 模式会额外生成 原始帧 / 传统 EIS / RAFT 的横向对比图。
        original_frames, _ = read_video(input_path, resize)
        compare_path = results_dir / "frames_compare" / f"{input_path.stem}_compare.png"
        make_frame_compare(original_frames, traditional_frames, raft_frames, compare_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.input:
        # 单视频模式：适合论文中的单个样例实验。
        run_one(Path(args.input), Path(args.output) if args.output else None, args, config)
        return
    if args.input_dir:
        # 批处理模式：遍历目录中的常见视频格式。
        for video in list_videos(args.input_dir):
            run_one(video, None, args, config)
        return
    raise SystemExit("Please provide --input or --input_dir.")


if __name__ == "__main__":
    main()
