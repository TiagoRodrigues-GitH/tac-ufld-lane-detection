"""Inference backends with one interface, so streaming, benchmarking and the
numerical checks treat PyTorch, ONNX Runtime and TensorRT the same way.

A backend exposes the model split at the point where features can be cached:

* ``encode(frame)``  (1, C, H, W) -> {"current": f, "history": g}
  per-frame features (``history`` is what later steps reuse; for most models
  it is the same tensor as ``current``);
* ``head(history, current)`` -> logits (1, G+1, A, L): fusion + classifier,
  ``history`` ordered oldest -> newest;
* ``full(clip)`` (1, T, C, H, W) -> logits: the unsplit model (recompute mode).

Tensors are numpy float32 arrays at the boundary for the exported backends
and torch tensors for PyTorch; ``to_numpy`` normalises outputs.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from torch import nn


class Backend(Protocol):
    name: str
    temporal: bool

    def encode(self, frame): ...
    def head(self, history: list, current): ...
    def full(self, clip): ...
    def synchronize(self) -> None: ...


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


class TorchBackend:
    """PyTorch eval-mode backend. ``precision``: fp32 | fp16 (CUDA only)."""

    def __init__(self, model: nn.Module, device: str = "cpu", precision: str = "fp32") -> None:
        if precision not in ("fp32", "fp16"):
            raise ValueError("precision must be fp32 or fp16")
        if precision == "fp16" and not device.startswith("cuda"):
            raise ValueError("fp16 inference needs a CUDA device")
        self.model = model.eval().to(device)
        if precision == "fp16":
            self.model = self.model.half()
        self.device, self.precision = device, precision
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.temporal = bool(getattr(model, "temporal", False))
        self.name = f"torch-{precision}"

    def _in(self, x) -> torch.Tensor:
        t = x if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x))
        return t.to(self.device, dtype=self.dtype, non_blocking=True)

    @torch.no_grad()
    def encode(self, frame) -> dict[str, torch.Tensor]:
        return self.model.encode_frames(self._in(frame))

    @torch.no_grad()
    def head(self, history: list, current) -> torch.Tensor:
        return self.model.forward_features([self._in(h) for h in history], self._in(current))["logits"]

    @torch.no_grad()
    def full(self, clip) -> torch.Tensor:
        return self.model(self._in(clip))["logits"]

    @torch.no_grad()
    def step(self, current, state):
        """Recurrent models only: one update of the carried hidden state."""
        out = self.model.recurrent_step(self._in(current), None if state is None else self._in(state))
        return out["logits"], out["state"]

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
