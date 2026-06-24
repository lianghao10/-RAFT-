from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch


class RAFTFlowEstimator:
    """Torchvision RAFT wrapper returning H x W x 2 dense flow arrays."""

    def __init__(
        self,
        weights_path: str | None = None,
        model_name: str = "small",
        device: str | None = None,
        iters: int = 12,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.iters = iters
        self.model_name = model_name
        self.transforms = None
        self.model = self._load_model(weights_path)
        self.model.to(self.device).eval()
        print(f"Using device: {self.device}")
        if weights_path:
            print(f"Loaded RAFT checkpoint: {weights_path}")
        else:
            print("Loaded RAFT checkpoint: torchvision DEFAULT weights")

    def _load_model(self, weights_path: str | None):
        from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small

        if self.model_name == "large":
            weights = None if weights_path else Raft_Large_Weights.DEFAULT
            model = raft_large(weights=weights, progress=True)
        else:
            weights = None if weights_path else Raft_Small_Weights.DEFAULT
            model = raft_small(weights=weights, progress=True)

        if weights is not None:
            self.transforms = weights.transforms()

        if weights_path:
            checkpoint = torch.load(weights_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned, strict=False)
        return model

    @staticmethod
    def _to_tensor(frame: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        return tensor.unsqueeze(0)

    @staticmethod
    def _pad_to_multiple(tensor: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, h, w = tensor.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h or pad_w:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
        return tensor, (pad_h, pad_w)

    @torch.inference_mode()
    def estimate(self, frame_t: np.ndarray, frame_t1: np.ndarray) -> tuple[np.ndarray, float]:
        image1 = self._to_tensor(frame_t).to(self.device)
        image2 = self._to_tensor(frame_t1).to(self.device)
        image1, (pad_h, pad_w) = self._pad_to_multiple(image1)
        image2, _ = self._pad_to_multiple(image2)
        if self.transforms is not None:
            image1, image2 = self.transforms(image1, image2)
        else:
            image1 = image1 / 127.5 - 1.0
            image2 = image2 / 127.5 - 1.0

        start = time.perf_counter()
        flow_predictions = self.model(image1, image2, num_flow_updates=self.iters)
        elapsed = time.perf_counter() - start

        flow = flow_predictions[-1][0].permute(1, 2, 0).detach().cpu().numpy()
        if pad_h:
            flow = flow[:-pad_h, :, :]
        if pad_w:
            flow = flow[:, :-pad_w, :]
        return flow.astype(np.float32), elapsed


def flow_to_bgr(flow: np.ndarray) -> np.ndarray:
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def save_flow_visualization(flow: np.ndarray, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), flow_to_bgr(flow))
