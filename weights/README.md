# 权重文件目录

如果你已经下载了 RAFT 预训练权重，可以放在这个目录中，例如：

```text
weights/raft-things.pth
weights/raft-sintel.pth
```

运行时通过 `--raft_weights` 指定本地权重：

```bash
python main.py --method raft --input data/input_videos/test.mp4 --raft_weights weights/raft-things.pth
```

如果不传入 `--raft_weights`，程序会默认使用 `torchvision` 提供的公开预训练 RAFT 权重。
