"""Light CIPO detection head on the lane model's own backbone features (design: ``docs/CIPO.md`` section 3).

* ``FeatureTap``   forward hooks that capture the backbone's multi-scale features WHILE the lane model runs, so one
                   backbone pass gives the lanes and the detection features, and no lane-model file changes:
                   ResNet (UFLD family): layer2 / layer3 / layer4 = strides 8 / 16 / 32 (128 / 256 / 512 ch for R18);
                   light backbone: its third and fourth blocks = strides 8 / 16 (96 / 128 ch).
* ``CipoHead``     a tiny FPN (1x1 laterals to ``width`` channels, nearest upsampling to stride 8, sum, one
                   depthwise-separable 3x3 block) and a CenterNet-style head (Zhou et al. 2019): two depthwise-
                   separable blocks, then 1x1 outputs for a heatmap per importance level, the sub-cell centre
                   offset and the log box size.
* ``cipo_loss``    penalty-reduced focal loss on the heatmaps + L1 on offset and log size at object centres.
* ``decode``       NMS-free: 3x3 max-pool peaks of the class-agnostic heatmap + top-k (the same goal as YOLOv10's
                   NMS-free design, Wang et al. 2024: no NMS in the deployed graph).

Only single-frame lane models are supported: with a temporal model the backbone also encodes the history frames,
and which frame the head should read is a design question of its own (``docs/CIPO.md`` section 3).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tac_ufld.models.lite import LightweightBackbone
from tac_ufld.models.resnet import ResNetBackbone


def _out_channels(block: nn.Module) -> int:
    convs = [m for m in block.modules() if isinstance(m, nn.Conv2d)]
    return convs[-1].out_channels


class FeatureTap:
    """Hooks a lane model's backbone; ``tap(x)`` returns (lane-model output, [features fine -> coarse])."""

    def __init__(self, lane_model: nn.Module) -> None:
        self.model = lane_model
        backbones = [m for m in lane_model.modules() if isinstance(m, (ResNetBackbone, LightweightBackbone))]
        if len(backbones) != 1:
            raise ValueError(f"expected exactly one ResNet or light backbone in {type(lane_model).__name__}, "
                             f"found {len(backbones)}")
        if getattr(lane_model, "temporal", False):
            raise ValueError("FeatureTap supports single-frame lane models only (see the module docstring)")
        backbone = backbones[0]
        self._captured: list[torch.Tensor | None] = []
        self._handles = []
        if isinstance(backbone, ResNetBackbone):
            self.kind = "resnet"
            self.strides = (8, 16, 32)
            self.channels = tuple(_out_channels(layer) for layer in (backbone.layer2, backbone.layer3, backbone.layer4))
            self._captured = [None]
            self._handles.append(backbone.register_forward_hook(self._store_tuple))
        else:
            self.kind = "lite"
            self.strides = (8, 16)
            self.channels = (_out_channels(backbone.net[2]), _out_channels(backbone.net[3]))
            self._captured = [None, None]
            for i, block in enumerate((backbone.net[2], backbone.net[3])):
                self._handles.append(block.register_forward_hook(self._store_at(i)))

    def _store_tuple(self, _module, _inp, out) -> None:
        self._captured[0] = out

    def _store_at(self, i: int):
        def hook(_module, _inp, out) -> None:
            self._captured[i] = out
        return hook

    def __call__(self, x: torch.Tensor):
        self._captured = [None] * len(self._captured)
        out = self.model(x)
        feats = list(self._captured[0]) if self.kind == "resnet" else list(self._captured)
        if any(f is None for f in feats) or feats[0].shape[0] != x.shape[0]:
            raise RuntimeError("the backbone hooks did not capture one feature map per input image")
        return out, feats

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def _dw_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class CipoHead(nn.Module):
    """Tiny FPN + centre head. Input: features fine -> coarse (the first one sets the output stride)."""

    def __init__(self, in_channels: tuple[int, ...], width: int = 64, num_levels: int = 4,
                 prior: float = 0.01) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in in_channels)
        self.smooth = _dw_block(width, width)
        self.tower = nn.Sequential(_dw_block(width, width), _dw_block(width, width))
        self.heat = nn.Conv2d(width, num_levels, 1)
        self.box = nn.Conv2d(width, 4, 1)          # (offset x, offset y, log w, log h)
        nn.init.constant_(self.heat.bias, -math.log((1 - prior) / prior))   # focal-loss prior (RetinaNet)
        nn.init.normal_(self.box.weight, std=0.001)
        nn.init.zeros_(self.box.bias)
        self.num_levels = num_levels

    def forward(self, feats: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        size = feats[0].shape[-2:]
        x = self.lateral[0](feats[0])
        for lateral, f in zip(self.lateral[1:], feats[1:]):
            x = x + F.interpolate(lateral(f), size=size, mode="nearest")
        x = self.tower(self.smooth(x))
        box = self.box(x)
        return {"heat": self.heat(x), "offset": torch.sigmoid(box[:, :2]), "log_size": box[:, 2:]}


# -- loss -------------------------------------------------------------------------------------------------------------

def focal_heatmap_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None,
                       alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet's penalty-reduced pixel-wise focal loss, normalised by the number of peaks.
    ``weight`` (B, h, w) scales the negatives (0 inside ignore regions)."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target.ge(1 - 1e-4).float()
    neg = 1.0 - pos
    pos_loss = torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = torch.log(1 - p) * p ** alpha * (1 - target) ** beta * neg
    if weight is not None:
        neg_loss = neg_loss * weight.unsqueeze(1)
    num_pos = pos.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos.clamp(min=1.0)


def _gather(feature: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """(B, C, h, w) at flat indices (B, K) -> (B, K, C)."""
    b, c = feature.shape[:2]
    flat = feature.reshape(b, c, -1).permute(0, 2, 1)
    return flat.gather(1, ind.unsqueeze(-1).expand(-1, -1, c))


def cipo_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], offset_weight: float = 1.0,
              size_weight: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    heat = focal_heatmap_loss(out["heat"], batch["heat"], batch.get("weight"))
    mask = batch["mask"].unsqueeze(-1)
    denom = mask.sum().clamp(min=1.0) * 2
    off = (torch.abs(_gather(out["offset"].float(), batch["ind"]) - batch["offset"]) * mask).sum() / denom
    size = (torch.abs(_gather(out["log_size"].float(), batch["ind"]) - batch["log_size"]) * mask).sum() / denom
    total = heat + offset_weight * off + size_weight * size
    return total, {"heat": float(heat.detach()), "offset": float(off.detach()), "size": float(size.detach())}


# -- decoding ---------------------------------------------------------------------------------------------------------

@torch.no_grad()
def decode(out: dict[str, torch.Tensor], stride: int, top_k: int = 100, min_score: float = 0.01
           ) -> list[dict[str, np.ndarray]]:
    """Per image: ``boxes`` (K, 4) xyxy in model pixels, ``scores`` (K,) = max level score at the peak,
    ``levels`` (K,) argmax level 1..L, ``level_scores`` (K, L). Peaks = 3x3 local maxima of the class-agnostic
    heatmap (no NMS)."""
    heat = torch.sigmoid(out["heat"].float())
    agnostic, level = heat.max(dim=1)
    peaks = agnostic == F.max_pool2d(agnostic.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    scored = (agnostic * peaks).flatten(1)
    b, _, h, w = heat.shape
    k = min(top_k, h * w)
    scores, ind = scored.topk(k, dim=1)
    offset = _gather(out["offset"].float(), ind)
    size = torch.exp(_gather(out["log_size"].float(), ind).clamp(max=10.0))
    level_scores = _gather(heat, ind)
    ys, xs = (ind // w).float(), (ind % w).float()
    cx, cy = (xs + offset[..., 0]) * stride, (ys + offset[..., 1]) * stride
    bw, bh = size[..., 0] * stride, size[..., 1] * stride
    boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)
    lvl = level.flatten(1).gather(1, ind) + 1
    results = []
    for i in range(b):
        keep = scores[i] >= min_score
        results.append({"boxes": boxes[i][keep].cpu().numpy().astype(np.float64),
                        "scores": scores[i][keep].cpu().numpy().astype(np.float64),
                        "levels": lvl[i][keep].cpu().numpy(),
                        "level_scores": level_scores[i][keep].cpu().numpy().astype(np.float64)})
    return results


class HeadProbe(nn.Module):
    """Wraps a head with fixed feature shapes so ``evaluation.efficiency.count_macs`` can measure it alone."""

    def __init__(self, head: CipoHead, channels: tuple[int, ...], strides: tuple[int, ...]) -> None:
        super().__init__()
        self.head, self.channels, self.strides = head, channels, strides

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, h, w = x.shape
        feats = [x.new_zeros(b, c, math.ceil(h / s), math.ceil(w / s)) for c, s in zip(self.channels, self.strides)]
        return self.head(feats)
