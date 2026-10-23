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


def load_frame(path: str, img_w: int, img_h: int, draft: bool = False) -> torch.Tensor:
    """RGB image resized to (img_h, img_w), float32 in [0, 1], CHW.
    PIL handles non-ASCII Windows paths (cv2.imread does not).
    ``draft``: JPEG only, decode at the smallest DCT scale that is still >= the
    target size (much faster for large sources), then resize as usual."""
    with Image.open(path) as img:
        if draft and img.format == "JPEG":
            img.draft("RGB", (img_w, img_h))
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


def degrade_current_frame(frames: torch.Tensor, cfg: AugmentationConfig, op: str, rng=random,
                          generator: torch.Generator | None = None, scope: str = "current") -> torch.Tensor:
    """Apply one degradation ``op`` to the LAST frame of the clip (RGB in [0, 1]); earlier frames are untouched.

    ``op``: occlude | blur | darken | noise (the training set) or shadow | glare (held-out: never used in
    training unless configured, so the robustness test can measure corruptions the models have not seen).
    ``scope="clip"`` (robustness evaluation only) applies the SAME op with the SAME parameters to every frame
    of the clip: a persistent occluder, as a vehicle ahead or glare is over 67-133 ms, when the history cannot
    simply be read instead of the current frame.

    ``scope="independent"`` (robustness evaluation only, added 2026-10-19) applies the same op to every frame
    but draws its parameters and content independently per frame: fresh sensor noise, occluders at other
    places, fluctuating blur and brightness. Unlike ``clip``, the history then holds evidence the current frame
    lacks, which is the regime where temporal fusion can genuinely help under a lasting condition.

    ``rng`` (the ``random`` module or a ``random.Random``) and ``generator``
    (a ``torch.Generator``, None = global) make it reproducible: training uses
    the global streams, the robustness evaluation a fixed seed per frame."""
    if scope == "current":
        return _degrade_last(frames, cfg, op, rng, generator)
    if scope == "independent":
        out = frames.clone()
        for i in range(frames.shape[0]):  # oldest -> current, each frame with its own draws
            out[i] = _degrade_last(frames[i:i + 1], cfg, op, rng, generator)[0]
        return out
    if scope != "clip":
        raise ValueError(f"unknown degradation scope '{scope}'")
    rng_state = rng.getstate()
    gen_state = generator.get_state() if generator is not None else None
    out = frames.clone()
    for i in range(frames.shape[0]):  # replay the same random draws for every frame
        rng.setstate(rng_state)
        if generator is not None:
            generator.set_state(gen_state)
        out[i] = _degrade_last(frames[i:i + 1], cfg, op, rng, generator)[0]
    return out


def _degrade_last(frames: torch.Tensor, cfg: AugmentationConfig, op: str, rng,
                  generator: torch.Generator | None) -> torch.Tensor:
    frames = frames.clone()
    cur = frames[-1]
    c, h, w = cur.shape
    if op == "occlude":  # solid boxes (vehicles, objects) centred in the road band
        for _ in range(rng.randint(*cfg.current_occlusion_boxes)):
            area = rng.uniform(*cfg.current_occlusion_scale) * h * w
            aspect = rng.uniform(0.5, 2.0)
            bh = max(1, min(h, int((area / aspect) ** 0.5)))
            bw = max(1, min(w, int((area * aspect) ** 0.5)))
            cy = rng.uniform(*cfg.current_occlusion_band) * (h - 1)
            cx = rng.uniform(0.0, w - 1.0)
            y0 = int(max(0, min(h - bh, cy - bh / 2)))
            x0 = int(max(0, min(w - bw, cx - bw / 2)))
            cur[:, y0:y0 + bh, x0:x0 + bw] = torch.rand(c, 1, 1, generator=generator)
    elif op == "blur":
        sigma = rng.uniform(*cfg.current_blur_sigma)
        cur = _per_frame(cur.unsqueeze(0), lambda a: cv2.GaussianBlur(a, (0, 0), sigma).reshape(a.shape))[0]
    elif op == "darken":
        cur = cur * rng.uniform(*cfg.current_darken)
    elif op == "noise":
        noise = torch.randn(cur.shape, generator=generator, dtype=cur.dtype)
        cur = (cur + noise * rng.uniform(*cfg.current_noise_std)).clamp(0.0, 1.0)
    elif op == "shadow":  # held-out: a cast shadow band across the road (trees, bridges, buildings)
        x_top = sorted(rng.uniform(0, w) for _ in range(2))
        x_bot = sorted(rng.uniform(0, w) for _ in range(2))
        poly = np.array([[x_top[0], 0], [x_top[1], 0], [x_bot[1], h - 1], [x_bot[0], h - 1]], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(mask, [poly], 1.0)
        cur = cur * (1.0 - rng.uniform(0.5, 0.8) * torch.from_numpy(mask))
    elif op == "glare":  # held-out: sun glare / headlights, a saturated blob in the upper road band
        cy, cx = rng.uniform(0.35, 0.7) * (h - 1), rng.uniform(0.2, 0.8) * (w - 1)
        radius = rng.uniform(0.15, 0.35) * w
        ys, xs = torch.meshgrid(torch.arange(h, dtype=cur.dtype), torch.arange(w, dtype=cur.dtype), indexing="ij")
        blob = torch.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * (radius / 2) ** 2))
        cur = (cur + rng.uniform(0.6, 1.0) * blob).clamp(0.0, 1.0)
    else:
        raise ValueError(f"unknown current-frame degradation '{op}'")
    frames[-1] = cur
    return frames


class PhotometricAugmenter:
    """Colour jitter, Gaussian noise, random erasing and history-frame dropout,
    sampled once per clip so every frame of a sequence is changed identically.
    Ported from ``ELASTemporalDataset._augment_sequence`` of the ELAS script.

    Optional extensions (all off by default, and then they draw no random
    numbers, so existing runs are reproduced exactly): gamma, hue rotation,
    cast shadows, Gaussian blur, motion blur, and current-frame degradation
    (``current_frame_prob``: only the LAST frame of the clip, whose lanes are
    the target, is occluded / blurred / darkened / made noisy, so a temporal
    model has to take the lanes from its clean history frames)."""

    def __init__(self, cfg: AugmentationConfig) -> None:
        self.cfg = cfg

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        frames = self._photometric(frames)
        if self.cfg.enabled and self.cfg.current_frame_prob > 0:
            frames = self._degrade_current(frames)
        return frames

    def _degrade_current(self, frames: torch.Tensor) -> torch.Tensor:
        """Degrade frames[-1] only (history untouched), or, for ``current_frame_persistent_share`` of the degraded
        clips, every frame with the same parameters. The share draws a random number only when it is > 0, so
        configs without it reproduce earlier runs exactly."""
        cfg = self.cfg
        if random.random() >= cfg.current_frame_prob:
            return frames
        op = random.choice(cfg.current_frame_ops)
        persistent = (cfg.current_frame_persistent_share > 0 and frames.shape[0] > 1
                      and random.random() < cfg.current_frame_persistent_share)
        return degrade_current_frame(frames, cfg, op, scope="clip" if persistent else "current")

    def _photometric(self, frames: torch.Tensor) -> torch.Tensor:
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
