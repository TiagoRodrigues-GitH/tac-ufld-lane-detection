"""Input representations for the preprocessing ablations.

One ``Preprocessor`` turns RGB frames (float in [0, 1], after resizing and
augmentation) into the model input channels, and ``channel_stats`` gives the
matching normalisation. Training, evaluation, streaming inference and the
ONNX/TensorRT runners all call this module, and the configuration is stored
in every checkpoint, so the representation cannot drift between stages.

Modes (``data.preprocessing.mode``) and model input channels:

========== ==== ===============================================================
mode       C    content
========== ==== ===============================================================
rgb        3    original behaviour (default)
gray       1    luminance (BT.601)                            [conv1 changes]
gray3      3    luminance replicated to 3 channels            [same model]
edge       1    Sobel gradient magnitude of the luminance      [conv1 changes]
canny      1    Canny edge map                                [conv1 changes]
hough      1    probabilistic-Hough line segments drawn from
                the Canny map (near-horizontal ones dropped)  [conv1 changes]
rgb_edge   4    RGB + Sobel (or Canny) channel                [conv1 changes]
========== ==== ===============================================================

``pre_ops`` (``blur``, ``clahe``, ``equalize``) run first, in the listed order.
Modes with C != 3 change the first convolution; ImageNet weights are adapted
(see ``tac_ufld.models.resnet.adapt_conv1``), so those runs are separate
configurations, not drop-in replacements for the RGB baseline.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from tac_ufld.config import PreprocessConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GRAY_MEAN = float(np.mean(IMAGENET_MEAN))
GRAY_STD = float(np.mean(IMAGENET_STD))

_CHANNELS = {"rgb": 3, "gray": 1, "gray3": 3, "edge": 1, "canny": 1, "hough": 1, "rgb_edge": 4}


def channels_for_mode(mode: str) -> int:
    if mode not in _CHANNELS:
        raise ValueError(f"unknown preprocessing mode '{mode}'")
    return _CHANNELS[mode]


def channel_stats(cfg: PreprocessConfig, normalize: str) -> tuple[list[float], list[float]]:
    """Per-channel (mean, std) used after preprocessing. ``normalize: none``
    leaves RGB/gray channels in [0, 1]; feature channels always use
    ``feature_mean`` / ``feature_std``."""
    img_mean = list(IMAGENET_MEAN) if normalize == "imagenet" else [0.0, 0.0, 0.0]
    img_std = list(IMAGENET_STD) if normalize == "imagenet" else [1.0, 1.0, 1.0]
    feat_mean, feat_std = [cfg.feature_mean], [cfg.feature_std]
    mode = cfg.mode
    if mode in ("rgb", "gray3"):
        return img_mean, img_std
    if mode == "gray":
        return ([GRAY_MEAN], [GRAY_STD]) if normalize == "imagenet" else ([0.0], [1.0])
    if mode in ("edge", "canny", "hough"):
        return feat_mean, feat_std
    if mode == "rgb_edge":
        return img_mean + feat_mean, img_std + feat_std
    raise ValueError(f"unknown preprocessing mode '{mode}'")


class Preprocessor:
    """RGB frames (T, 3, H, W) in [0, 1] -> (T, C, H, W) in [0, 1], unnormalised."""

    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg
        self.channels = channels_for_mode(cfg.mode)
        self.identity = cfg.mode == "rgb" and not cfg.pre_ops

    # ------------------------------------------------------------ per image

    def _pre(self, rgb: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        for op in cfg.pre_ops:
            if op == "blur":
                rgb = cv2.GaussianBlur(rgb, (cfg.blur_ksize, cfg.blur_ksize), 0)
            elif op in ("clahe", "equalize"):
                lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
                if op == "clahe":
                    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_tile, cfg.clahe_tile))
                    lab[..., 0] = clahe.apply(lab[..., 0])
                else:
                    lab[..., 0] = cv2.equalizeHist(lab[..., 0])
                rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return rgb

    def _sobel(self, gray: np.ndarray) -> np.ndarray:
        g = gray.astype(np.float32) / 255.0
        k = self.cfg.sobel_ksize
        mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=k), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=k))
        if self.cfg.edge_normalization == "max":
            peak = float(mag.max())
            return mag / peak if peak > 0 else mag
        return np.clip(mag / self.cfg.edge_clip, 0.0, 1.0)

    def _canny(self, gray: np.ndarray) -> np.ndarray:
        c = self.cfg
        return cv2.Canny(gray, c.canny_low, c.canny_high, L2gradient=c.canny_l2)

    def _hough(self, gray: np.ndarray) -> np.ndarray:
        c = self.cfg
        edges = self._canny(gray)
        out = np.zeros_like(gray)
        segments = cv2.HoughLinesP(edges, 1, np.pi / 180, c.hough_threshold,
                                   minLineLength=c.hough_min_line_length, maxLineGap=c.hough_max_line_gap)
        if segments is not None:
            min_angle = np.deg2rad(c.hough_min_angle_deg)
            for x1, y1, x2, y2 in np.asarray(segments).reshape(-1, 4):  # (N,1,4) in OpenCV 4, (N,4) in 5
                angle = abs(np.arctan2(y2 - y1, x2 - x1))
                angle = min(angle, np.pi - angle)
                if angle >= min_angle:
                    cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 255, c.hough_thickness)
        return out

    def image(self, rgb_u8: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 RGB -> (H, W, C) float32 in [0, 1]."""
        rgb = self._pre(rgb_u8)
        mode = self.cfg.mode
        if mode == "rgb":
            return rgb.astype(np.float32) / 255.0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if mode == "gray":
            return (gray.astype(np.float32) / 255.0)[..., None]
        if mode == "gray3":
            return np.repeat((gray.astype(np.float32) / 255.0)[..., None], 3, axis=2)
        if mode == "edge":
            return self._sobel(gray)[..., None]
        if mode == "canny":
            return (self._canny(gray).astype(np.float32) / 255.0)[..., None]
        if mode == "hough":
            return (self._hough(gray).astype(np.float32) / 255.0)[..., None]
        if mode == "rgb_edge":
            edge = self._sobel(gray) if self.cfg.edge_source == "sobel" else self._canny(gray).astype(np.float32) / 255.0
            return np.concatenate([rgb.astype(np.float32) / 255.0, edge[..., None]], axis=2)
        raise ValueError(mode)

    # ------------------------------------------------------------ batches

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """(T, 3, H, W) float [0, 1] -> (T, C, H, W) float [0, 1]."""
        if self.identity:
            return frames
        out = []
        for frame in frames:
            u8 = (frame.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
            out.append(torch.from_numpy(self.image(u8).transpose(2, 0, 1).copy()))
        return torch.stack(out)


def normalize_channels(frames: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    """(..., C, H, W) -> standardised per channel."""
    m = torch.tensor(mean, dtype=frames.dtype).view(-1, 1, 1)
    s = torch.tensor(std, dtype=frames.dtype).view(-1, 1, 1)
    return (frames - m) / s
