"""Image loading, normalisation and photometric augmentation."""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
import torch
from PIL import Image

from tac_ufld.config import AugmentationConfig

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# RGB <-> YIQ, used for hue rotation (rotating the IQ plane rotates hue).
_RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
_YIQ2RGB = torch.linalg.inv(_RGB2YIQ)


def load_frame(path: str, img_w: int, img_h: int) -> torch.Tensor:
    """RGB image resized to (img_h, img_w), float32 in [0, 1], CHW.
    PIL handles non-ASCII Windows paths (cv2.imread does not)."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize((img_w, img_h), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def rgb_array_to_tensor(rgb: np.ndarray, img_w: int, img_h: int) -> torch.Tensor:
    """(H, W, 3) uint8 RGB array (e.g. a video frame) -> the same tensor
    ``load_frame`` produces for the image on disk (PIL bilinear resize)."""
    img = Image.fromarray(rgb).resize((img_w, img_h), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def normalize(frames: torch.Tensor, mode: str) -> torch.Tensor:
    """ImageNet normalisation as in the official UFLD transforms."""
    if mode == "none":
        return frames
    return (frames - IMAGENET_MEAN) / IMAGENET_STD


def _per_frame(frames: torch.Tensor, fn) -> torch.Tensor:
    out = [torch.from_numpy(np.ascontiguousarray(fn(f.permute(1, 2, 0).contiguous().numpy()).transpose(2, 0, 1)))
           for f in frames]
    return torch.stack(out)


def _motion_kernel(size: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    c = (size - 1) / 2.0
    dx, dy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    cv2.line(kernel, (int(round(c - dx * c)), int(round(c - dy * c))),
             (int(round(c + dx * c)), int(round(c + dy * c))), 1.0, 1)
    return kernel / max(kernel.sum(), 1e-6)


def _shadow_mask(h: int, w: int) -> np.ndarray:
    """Random quadrilateral spanning the image height (a cast shadow band)."""
    x_top = sorted(random.uniform(0, w) for _ in range(2))
    x_bot = sorted(random.uniform(0, w) for _ in range(2))
    poly = np.array([[x_top[0], 0], [x_top[1], 0], [x_bot[1], h - 1], [x_bot[0], h - 1]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(mask, [poly], 1.0)
    return mask


class PhotometricAugmenter:
    """Colour jitter, Gaussian noise, random erasing and history-frame dropout,
    sampled once per clip so every frame of a sequence is changed identically.
    Ported from ``ELASTemporalDataset._augment_sequence`` of the ELAS script.

    Optional extensions (all off by default, and then they draw no random
    numbers, so existing runs are reproduced exactly): gamma, hue rotation,
    cast shadows, Gaussian blur and motion blur."""

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
        frames = self._extended(frames)
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

    def _extended(self, frames: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.gamma > 0:
            g = math.exp(random.uniform(-math.log1p(cfg.gamma), math.log1p(cfg.gamma)))
            frames = frames.clamp(0.0, 1.0) ** g
        if cfg.hue > 0:
            angle = random.uniform(-cfg.hue, cfg.hue) * 2 * math.pi
            c, s = math.cos(angle), math.sin(angle)
            rot = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=frames.dtype)
            m = (_YIQ2RGB @ rot @ _RGB2YIQ).to(frames.dtype)
            frames = torch.einsum("ij,tjhw->tihw", m, frames)
        if cfg.shadow_prob > 0 and random.random() < cfg.shadow_prob:
            strength = random.uniform(*cfg.shadow_strength)
            mask = torch.from_numpy(_shadow_mask(frames.shape[2], frames.shape[3]))
            frames = frames * (1.0 - strength * mask)
        if cfg.blur_prob > 0 and random.random() < cfg.blur_prob:
            sigma = random.uniform(*cfg.blur_sigma)
            frames = _per_frame(frames, lambda a: cv2.GaussianBlur(a, (0, 0), sigma).reshape(a.shape))
        if cfg.motion_blur_prob > 0 and random.random() < cfg.motion_blur_prob:
            size = random.randint(*cfg.motion_blur_kernel) | 1
            kernel = _motion_kernel(size, random.uniform(0.0, 180.0))
            frames = _per_frame(frames, lambda a: cv2.filter2D(a, -1, kernel).reshape(a.shape))
        return frames
