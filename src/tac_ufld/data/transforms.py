"""Image loading, normalisation and photometric augmentation."""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image

from tac_ufld.config import AugmentationConfig

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_frame(path: str, img_w: int, img_h: int) -> torch.Tensor:
    """RGB image resized to (img_h, img_w), float32 in [0, 1], CHW.
    PIL handles non-ASCII Windows paths (cv2.imread does not)."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize((img_w, img_h), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def normalize(frames: torch.Tensor, mode: str) -> torch.Tensor:
    """ImageNet normalisation as in the official UFLD transforms."""
    if mode == "none":
        return frames
    return (frames - IMAGENET_MEAN) / IMAGENET_STD


class PhotometricAugmenter:
    """Colour jitter, Gaussian noise, random erasing and history-frame dropout,
    sampled once per clip so every frame of a sequence is changed identically.
    Ported from ``ELASTemporalDataset._augment_sequence`` of the ELAS script.
    No geometric augmentation: labels stay exact."""

    def __init__(self, cfg: AugmentationConfig) -> None:
        self.cfg = cfg

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if not cfg.enabled or random.random() > cfg.prob:
            return frames
        brightness = 1.0 + random.uniform(-cfg.brightness, cfg.brightness)
        contrast = 1.0 + random.uniform(-cfg.contrast, cfg.contrast)
        saturation = 1.0 + random.uniform(-cfg.saturation, cfg.saturation)
        mean = frames.mean(dim=(2, 3), keepdim=True)
        frames = (frames - mean) * contrast + mean
        gray = frames.mean(dim=1, keepdim=True)
        frames = (gray + (frames - gray) * saturation) * brightness
        if cfg.noise_std > 0:
            frames = frames + torch.randn_like(frames) * cfg.noise_std
        frames = frames.clamp(0.0, 1.0)

        if random.random() < cfg.erasing_prob:
            _, _, h, w = frames.shape
            area = random.uniform(*cfg.erasing_scale) * h * w
            aspect = random.uniform(0.5, 2.0)
            eh = max(1, min(h, int((area * aspect) ** 0.5)))
            ew = max(1, min(w, int((area / aspect) ** 0.5)))
            y0, x0 = random.randint(0, h - eh), random.randint(0, w - ew)
            frames[:, :, y0:y0 + eh, x0:x0 + ew] = torch.rand(frames.shape[0], 3, eh, ew)

        if frames.shape[0] > 1 and random.random() < cfg.frame_dropout_prob:
            frames[random.randrange(frames.shape[0] - 1)] = frames[-1].clone()
        return frames
