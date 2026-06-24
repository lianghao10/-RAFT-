# RAFT Optical Flow Video Stabilization

This project implements a course-paper friendly video stabilization pipeline with two methods:

- Traditional EIS baseline: Shi-Tomasi corners, LK pyramid optical flow, affine motion estimation, trajectory smoothing, and `warpAffine` compensation.
- RAFT improved method: dense RAFT optical flow, background-motion filtering, affine RANSAC motion estimation, trajectory smoothing, and motion compensation.

The code focuses on using public pretrained RAFT models for inference. It does not train RAFT from scratch.

## Project Structure

```text
video_stabilization_raft/
├── README.md
├── requirements.txt
├── main.py
├── configs/default.yaml
├── weights/
├── data/
│   ├── input_videos/
│   └── output_videos/
├── src/
│   ├── traditional_eis.py
│   ├── raft_flow.py
│   ├── motion_estimation.py
│   ├── foreground_filter.py
│   ├── trajectory_smoothing.py
│   ├── compensation.py
│   ├── crop.py
│   ├── metrics.py
│   ├── visualization.py
│   ├── io_utils.py
│   └── pipeline.py
└── results/
    ├── frames_compare/
    ├── flow_vis/
    ├── trajectories/
    ├── logs/
    └── metrics.csv
```

## Environment

Python 3.10 or newer is recommended.

```bash
cd video_stabilization_raft
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the PyTorch build that matches your CUDA version from the official PyTorch website first, then install the remaining requirements.

## RAFT Weights

By default, the program uses torchvision's public pretrained RAFT weights. On the first run, torchvision may download the weights automatically.

You can also pass a local checkpoint:

```bash
python main.py --method raft --input data/input_videos/test.mp4 --raft_weights weights/raft-things.pth
```

The code automatically chooses:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

The console prints messages such as:

```text
Using device: cuda
Loaded RAFT checkpoint: torchvision DEFAULT weights
Processing video: data/input_videos/test.mp4
```

## Single Video

Traditional baseline:

```bash
python main.py --method traditional --input data/input_videos/test.mp4 --output data/output_videos/test_traditional_stab.mp4
```

RAFT improved method:

```bash
python main.py --method raft --input data/input_videos/test.mp4 --output data/output_videos/test_raft_stab.mp4 --mask flow_percentile --sample_step 8 --resize 640
```

Run both methods and generate a frame comparison image:

```bash
python main.py --method both --input data/input_videos/test.mp4 --resize 640 --mask flow_percentile
```

## Batch Processing

```bash
python main.py --method raft --input_dir data/input_videos --output_dir data/output_videos --mask mog2
```

Supported video suffixes include `.mp4`, `.avi`, `.mov`, `.mkv`, and `.m4v`.

## Mask Modes

```text
none
border
flow_percentile
border_flow_percentile
mog2
```

Recommended settings for the paper experiments:

```bash
python main.py --method raft --input data/input_videos/test.mp4 --mask flow_percentile --sample_step 8 --resize 640
python main.py --method raft --input data/input_videos/test.mp4 --mask border --sample_step 8 --resize 640
python main.py --method raft --input data/input_videos/test.mp4 --mask mog2 --sample_step 8 --resize 640
```

## Outputs

Stabilized videos:

```text
data/output_videos/test_traditional_stab.mp4
data/output_videos/test_raft_stab.mp4
```

Trajectory plots:

```text
results/trajectories/test_dx.png
results/trajectories/test_dy.png
results/trajectories/test_da.png
```

RAFT flow visualizations:

```text
results/flow_vis/test_frame_0001_flow.png
results/flow_vis/test_frame_0050_flow.png
```

Frame comparison image when using `--method both`:

```text
results/frames_compare/test_compare.png
```

Metrics:

```text
results/metrics.csv
```

The CSV contains:

```text
video_name, method, crop_ratio, stability_score, mean_flow_time, mean_total_time, fps, notes
```

## Metrics

- `crop_ratio`: retained image area after border zoom. Higher is better.
- `stability_score`: estimated reduction of high-frequency trajectory energy. Higher is better.
- `mean_flow_time`: average optical-flow or motion-estimation time per frame pair.
- `mean_total_time`: average total processing time per frame.
- `fps`: processing speed.

## Notes for Experiments

For a clean paper comparison, run:

```bash
python main.py --method traditional --input data/input_videos/test.mp4
python main.py --method raft --input data/input_videos/test.mp4 --mask flow_percentile --sample_step 8 --resize 640
python main.py --method both --input data/input_videos/test.mp4 --mask flow_percentile --sample_step 8 --resize 640
```

Then use:

- Original video
- Traditional EIS output
- RAFT output
- `results/trajectories/*.png`
- `results/flow_vis/*.png`
- `results/frames_compare/*_compare.png`
- `results/metrics.csv`

## Common Issues

If RAFT runs too slowly on CPU, use:

```bash
python main.py --method raft --input data/input_videos/test.mp4 --resize 320 --raft_model small --raft_iters 6
```

If automatic torchvision weight download fails, download the weights manually or configure network access, then pass the local path with `--raft_weights`.

If black borders are visible, increase:

```bash
--crop_zoom_margin 1.08
```

This preserves visual stability at the cost of a smaller crop ratio.
