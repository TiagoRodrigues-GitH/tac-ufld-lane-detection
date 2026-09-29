"""Batched inference producing decoded row-anchor predictions."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tac_ufld.decoding import decode_logits


@dataclass
class Predictions:
    exist: np.ndarray           # (N, A, L) existence probability
    x: np.ndarray               # (N, A, L) x position, model pixels
    current_weight: np.ndarray  # (N,) weight on the current frame after fusion (NaN if n/a)
    loss: float | None = None   # mean comparable (focal) loss, when a loss_fn is given


def _current_frame_weight(outputs: dict, batch_size: int) -> np.ndarray:
    """Share of the fused feature coming from the current frame:
    v02/v04 softmax weight, v03 mean gate, v05 1 - residual gate."""
    if "fusion_weights" in outputs:
        return np.full(batch_size, float(outputs["fusion_weights"][-1]), dtype=np.float32)
    if "gates" in outputs:
        return outputs["gates"][:, -1].float().mean(dim=(1, 2, 3)).cpu().numpy()
    if "gate" in outputs:
        return (1.0 - outputs["gate"].float().mean(dim=(1, 2, 3))).cpu().numpy()
    return np.full(batch_size, np.nan, dtype=np.float32)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    img_w: int,
    amp: bool = True,
    static_history: bool = False,
    loss_fn=None,
) -> Predictions:
    """Run ``model`` over ``loader`` (which must not shuffle). ``static_history``
    replaces every history frame by the current frame (temporal ablation)."""
    model.eval()
    exist_all, x_all, weight_all, losses = [], [], [], []
    autocast = torch.autocast("cuda") if (amp and device.startswith("cuda")) else nullcontext()
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        if static_history and images.shape[1] > 1:
            images = images[:, -1:].expand_as(images).contiguous()
        with autocast:
            outputs = model(images)
        logits = outputs["logits"]
        exist, bins = decode_logits(logits)
        grid = logits.shape[1] - 1
        exist_all.append(exist.cpu().numpy())
        x_all.append((bins * (img_w - 1) / (grid - 1)).cpu().numpy())
        weight_all.append(_current_frame_weight(outputs, images.shape[0]))
        if loss_fn is not None:
            gpu_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            gpu_batch["images"] = images
            _, logs = loss_fn(model, outputs, gpu_batch, include_temporal=False)
            losses.append((logs["focal"], images.shape[0]))
    mean_loss = None
    if losses:
        mean_loss = sum(l * n for l, n in losses) / sum(n for _, n in losses)
    return Predictions(
        exist=np.concatenate(exist_all).astype(np.float32),
        x=np.concatenate(x_all).astype(np.float32),
        current_weight=np.concatenate(weight_all).astype(np.float32),
        loss=mean_loss,
    )
