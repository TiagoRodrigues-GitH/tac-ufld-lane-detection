"""Temporal fusion modules added in v0.4 (shared by the UFLD and lite families).

* ``AlignedResidualFusion`` (``ufld_v06``): lite v0.5's fusion ported to UFLD.
  Each history feature map is warped onto the current one by a learned
  per-cell flow before fusing, so lane markings that moved between frames
  are not blurred together (v0.2-v0.4 average unaligned feature maps). A
  per-pixel gate, closed at initialisation, mixes the aligned history into
  the current features.
* ``ConvGRUFusion`` (``ufld_v07``, ``lite_v06``): a small convolutional GRU at
  the backbone's output resolution. Features are reduced to ``hidden``
  channels, the GRU runs over the frames oldest -> current, and the final
  hidden state is projected back and ADDED to the current features. The
  projection is zero-initialised, so after the warm start the model computes
  exactly its single-frame baseline and learns how much memory to use.

Both return ``(fused, extras)`` like the other fusions. ``ConvGRUFusion``
also exposes ``step``/``readout`` so a deployment can carry ONE hidden-state
tensor per stream instead of a window of past feature maps
(``StreamingLaneDetector(mode="carry")``).
"""

from __future__ import annotations

import torch
from torch import nn

from tac_ufld.models.lite import ResidualTemporalFusion


class AlignedResidualFusion(ResidualTemporalFusion):
    """Warp every history frame onto the current frame, compress the warped
    history, gate it into the current features (see ``ResidualTemporalFusion``)."""

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fused, gate, flows = super().forward(features)
        return fused, {"gate": gate, "flows": flows}


class ConvGRUFusion(nn.Module):
    """Convolutional GRU over per-frame features (B, T, C, h, w) -> fused (B, C, h, w)."""

    recurrent = True

    def __init__(self, channels: int, hidden: int = 64, kernel: int = 3) -> None:
        super().__init__()
        self.hidden = hidden
        pad = kernel // 2
        self.reduce = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.BatchNorm2d(hidden),
                                    nn.ReLU(inplace=True))
        self.gates = nn.Conv2d(2 * hidden, 2 * hidden, kernel, padding=pad)      # update z, reset r
        self.candidate = nn.Conv2d(2 * hidden, hidden, kernel, padding=pad)
        self.project = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def init_state(self, features: torch.Tensor) -> torch.Tensor:
        b, _, h, w = features.shape
        return features.new_zeros(b, self.hidden, h, w)

    def step(self, features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """One GRU update with the features of the next frame."""
        x = self.reduce(features)
        z, r = torch.sigmoid(self.gates(torch.cat([x, state], dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, r * state], dim=1)))
        return (1.0 - z) * state + z * candidate

    def readout(self, current: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return current + self.project(state)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state = self.init_state(features[:, 0])
        for t in range(features.shape[1]):
            state = self.step(features[:, t], state)
        return self.readout(features[:, -1], state), {"state_norm": state.detach().abs().mean(dim=(1, 2, 3))}
