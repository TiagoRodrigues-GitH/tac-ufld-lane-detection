"""Parameter count, MACs, latency, FPS and peak memory (ported from the ELAS script)."""

from __future__ import annotations

import time

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Parameters used at inference: the training-only auxiliary segmentation branch of UFLD is excluded, as in
    the UFLD paper."""
    return sum(p.numel() for name, p in model.named_parameters() if ".aux_" not in f".{name}")


def count_macs(model: nn.Module, dummy: torch.Tensor) -> int:
    """Multiply-accumulates of one forward pass: Conv2d, Linear, multi-head attention and the lite head's
    anchor-row pooling (element-wise ops, normalisation and grid sampling are not counted).

    ``nn.MultiheadAttention`` runs its projections functionally, so the Linear hook never sees them (its
    ``out_proj`` is never *called*); before 2026-10-18 the attention fusions (ufld_v08, lite_v07) were therefore
    under-counted. Its hook adds the Q/K/V and output projections and the two attention products."""
    from tac_ufld.models.lite import AnchorRowPool

    macs = [0]

    def conv_hook(m: nn.Conv2d, _inp, out):
        macs[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def linear_hook(m: nn.Linear, _inp, out):
        macs[0] += out.numel() * m.in_features

    def attention_hook(m: nn.MultiheadAttention, inp, _out):
        q, k = inp[0], inp[1]
        if not m.batch_first:  # (L, B, E) -> (B, L, E)
            q, k = q.transpose(0, 1), k.transpose(0, 1)
        b, lq, e = q.shape
        lk = k.shape[1]
        macs[0] += b * (lq * e * e + 2 * lk * e * e + lq * e * e)  # Q, K, V and output projections
        macs[0] += b * 2 * lq * lk * e                              # Q K^T and attention x V

    def row_pool_hook(m: AnchorRowPool, inp, out):
        b, c, h, w = inp[0].shape
        anchors, grid = out.shape[-2], out.shape[-1]
        macs[0] += b * c * (anchors * h * w + anchors * w * grid)

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.MultiheadAttention):
            handles.append(m.register_forward_hook(attention_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, AnchorRowPool):
            handles.append(m.register_forward_hook(row_pool_hook))
    try:
        with torch.no_grad():
            model(dummy)
    finally:
        for h in handles:
            h.remove()
    return macs[0]


@torch.no_grad()
def measure_efficiency(model: nn.Module, num_frames: int, img_h: int, img_w: int,
                       device: str, runs: int = 50, warmup: int = 10, in_channels: int = 3) -> dict[str, float]:
    """Batch-1 latency on ``device``. Temporal models re-encode every history
    frame per call, so this is an upper bound for a streaming deployment that
    caches history features (see ``tac_ufld.inference.streaming``)."""
    model.eval()
    dummy = torch.randn(1, num_frames, in_channels, img_h, img_w, device=device)
    macs = count_macs(model, dummy)
    for _ in range(warmup):
        model(dummy)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(runs):
        model(dummy)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) / runs * 1000.0
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else float("nan")
    return {
        "params_millions": count_parameters(model) / 1e6,
        "gmacs": macs / 1e9,
        "latency_ms": latency_ms,
        "fps": 1000.0 / latency_ms if latency_ms > 0 else float("nan"),
        "peak_gpu_memory_mb": peak,
        "device": device,
        "input_frames": num_frames,
    }
