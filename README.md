# 基于 RAFT 深度光流的视频防抖

本项目实现了一个用于课程论文实验的视频电子防抖系统，包含两条处理路线：

- 传统 EIS baseline：Shi-Tomasi 角点检测、LK 金字塔光流跟踪、仿射运动估计、轨迹平滑和 `warpAffine` 运动补偿。
- RAFT 改进方法：使用预训练 RAFT 估计稠密光流，再通过背景运动筛选、RANSAC 仿射矩阵估计、轨迹平滑和运动补偿输出稳定视频。

本项目重点是调用公开预训练 RAFT 模型进行推理，并将其接入视频防抖流程；不包含从零训练 RAFT 网络。

## 项目结构

```text
video_stabilization_raft/
|-- README.md
|-- requirements.txt
|-- main.py
|-- configs/
|   `-- default.yaml
|-- weights/
|   `-- README.md
|-- data/
|   |-- input_videos/
|   `-- output_videos/
|-- src/
|   |-- traditional_eis.py       # 传统 Shi-Tomasi + LK 防抖 baseline
|   |-- raft_flow.py             # RAFT 光流推理与光流可视化
|   |-- motion_estimation.py     # 由稠密光流估计全局仿射运动
|   |-- foreground_filter.py     # 前景过滤 / 背景区域筛选
|   |-- trajectory_smoothing.py  # 运动轨迹平滑
|   |-- compensation.py          # warpAffine 运动补偿
|   |-- crop.py                  # 动态裁剪 / 边界修复
|   |-- metrics.py               # 评价指标与 metrics.csv 输出
|   |-- visualization.py         # 轨迹图、对比图输出
|   |-- io_utils.py              # 视频读写
|   `-- pipeline.py              # 防抖主流程
`-- results/
    |-- frames_compare/
    |-- flow_vis/
    |-- trajectories/
    |-- logs/
    `-- metrics.csv
```

## 环境安装

建议使用 Python 3.10 或更新版本。

```bash
cd video_stabilization_raft
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果电脑有 NVIDIA GPU，建议先到 PyTorch 官网安装与你 CUDA 版本匹配的 PyTorch，然后再安装其余依赖。

## RAFT 权重说明

默认情况下，程序会使用 `torchvision` 提供的公开预训练 RAFT 权重。第一次运行时，`torchvision` 可能会自动下载权重。

如果你已经有本地权重文件，可以通过 `--raft_weights` 指定：

```bash
python main.py --method raft --input data/input_videos/test.mp4 --raft_weights weights/raft-things.pth
```

程序会自动选择运行设备：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

运行时控制台会输出类似信息：

```text
Using device: cuda
Loaded RAFT checkpoint: torchvision DEFAULT weights
Processing video: data/input_videos/test.mp4
```

## 单视频运行

运行传统 baseline：

```bash
python main.py --method traditional --input data/input_videos/test.mp4 --output data/output_videos/test_traditional_stab.mp4
```

运行 RAFT 改进方法：

```bash
python main.py --method raft --input data/input_videos/test.mp4 --output data/output_videos/test_raft_stab.mp4 --mask flow_percentile --sample_step 8 --resize 640
```

同时运行传统方法和 RAFT 方法，并生成帧序列对比图：

```bash
python main.py --method both --input data/input_videos/test.mp4 --resize 640 --mask flow_percentile
```

## 批量处理

将多个视频放入 `data/input_videos/` 后运行：

```bash
python main.py --method raft --input_dir data/input_videos --output_dir data/output_videos --mask mog2
```

支持的视频后缀包括 `.mp4`、`.avi`、`.mov`、`.mkv` 和 `.m4v`。

## 前景过滤 / 背景筛选模式

可通过 `--mask` 选择不同筛选策略：

```text
none                   不做筛选，直接使用采样光流点
border                 优先使用画面边缘区域，适合主体位于中心的场景
flow_percentile        按光流幅值分位数过滤异常运动点
border_flow_percentile 同时使用边缘区域和光流幅值过滤
mog2                   使用 OpenCV MOG2 背景建模筛选背景区域
```

论文实验中推荐至少比较：

```bash
python main.py --method raft --input data/input_videos/test.mp4 --mask flow_percentile --sample_step 8 --resize 640
python main.py --method raft --input data/input_videos/test.mp4 --mask border --sample_step 8 --resize 640
python main.py --method raft --input data/input_videos/test.mp4 --mask mog2 --sample_step 8 --resize 640
```

## 输出结果

稳定后的视频：

```text
data/output_videos/test_traditional_stab.mp4
data/output_videos/test_raft_stab.mp4
```

轨迹平滑前后对比图：

```text
results/trajectories/test_dx.png
results/trajectories/test_dy.png
results/trajectories/test_da.png
```

RAFT 光流可视化图：

```text
results/flow_vis/test_frame_0001_flow.png
results/flow_vis/test_frame_0050_flow.png
```

使用 `--method both` 时生成帧序列对比图：

```text
results/frames_compare/test_compare.png
```

评价指标：

```text
results/metrics.csv
```

CSV 字段如下：

```text
video_name, method, crop_ratio, stability_score, mean_flow_time, mean_total_time, fps, notes
```

## 评价指标说明

- `crop_ratio`：裁剪后保留的有效画面面积比例，数值越高表示画幅损失越小。
- `stability_score`：基于运动轨迹高频能量变化估计的稳定性提升，数值越高表示高频抖动减少越明显。
- `mean_flow_time`：平均每对相邻帧的光流估计或运动估计耗时。
- `mean_total_time`：平均每帧总处理耗时。
- `fps`：整体处理速度。

## 建议的论文实验流程

为了形成“原始视频 / 传统 EIS / RAFT 改进方法”的对照实验，可以按下面步骤运行：

```bash
python main.py --method traditional --input data/input_videos/test.mp4
python main.py --method raft --input data/input_videos/test.mp4 --mask flow_percentile --sample_step 8 --resize 640
python main.py --method both --input data/input_videos/test.mp4 --mask flow_percentile --sample_step 8 --resize 640
```

论文中可使用以下材料：

- 原始视频截图
- 传统 EIS 输出视频
- RAFT 改进方法输出视频
- `results/trajectories/*.png`
- `results/flow_vis/*.png`
- `results/frames_compare/*_compare.png`
- `results/metrics.csv`

## 常见问题

如果 CPU 上运行 RAFT 太慢，可以降低分辨率和迭代次数：

```bash
python main.py --method raft --input data/input_videos/test.mp4 --resize 320 --raft_model small --raft_iters 6
```

如果自动下载 `torchvision` 权重失败，可以手动下载公开 RAFT 权重，然后使用：

```bash
python main.py --method raft --input data/input_videos/test.mp4 --raft_weights weights/raft-things.pth
```

如果输出视频仍然有黑边，可以适当增大裁剪放大系数：

```bash
--crop_zoom_margin 1.08
```

该参数可以减少边缘黑边，但会降低 `crop_ratio`，即保留画面面积会变小。
