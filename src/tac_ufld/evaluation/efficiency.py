"""Parameter count, MACs, latency, FPS and peak memory (ported from the ELAS script)."""

from __future__ import annotations

import time

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: nn.Module, dummy: torch.Tensor) -> int:
    """Multiply-accumulates of Conv2d and Linear layers for one forward pass."""
    macs = [0]

    def conv_hook(m: nn.Conv2d, _inp, out):
        macs[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def linear_hook(m: nn.Linear, _inp, out):
        macs[0] += out.numel() * m.in_features

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
    try:
        with torch.no_grad():
            model(dummy)
    finally:
        for h in handles:
            h.remove()
    return macs[0]


@torch.no_grad()
def measure_efficiency(model: nn.Module, num_frames: int, img_h: int, img_w: int,
                       device: str, runs: int = 50, warmup: int = 10) -> dict[str, float]:
    """Batch-1 latency on ``device``. Temporal models re-encode every history
    frame per call, so this is an upper bound for a streaming deployment that
    caches history features."""
    model.eval()
    dummy = torch.randn(1, num_frames, 3, img_h, img_w, device=device)
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
