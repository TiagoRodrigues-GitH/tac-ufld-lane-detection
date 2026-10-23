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
* ``OcclusionGatedConvGRUFusion`` (``lite_v06_occ``, added 2026-10-19): the
  same ConvGRU with its update gate multiplied by a per-cell visibility of
  the incoming frame, so an occluded frame does not overwrite the memory.

* ``TemporalAttentionFusion`` (``ufld_v08``, ``lite_v07``, added 2026-10-05):
  attention over past frames, the family of TGC-Net and LaneTCA. Every cell
  of the current feature map queries the cells of all frames in the window
  (keys and values pooled to at most ``kv_cells`` per frame, each frame with
  a learned embedding), so a cell can take context from any place and time
  instead of the same cell (gates), a learned displacement (flow) or a
  running state (GRU). The read-out is zero-initialised and added to the
  current features: after the warm start the model is its baseline.

All return ``(fused, extras)`` like the other fusions. ``ConvGRUFusion``
also exposes ``step``/``readout`` so a deployment can carry ONE hidden-state
tensor per stream instead of a window of past feature maps
(``StreamingLaneDetector(mode="carry")``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tac_ufld.models.lite import ResidualTemporalFusion


class AlignedResidualFusion(ResidualTemporalFusion):
    """Warp every history frame onto the current frame, compress the warped
    history, gate it into the current features (see ``ResidualTemporalFusion``)."""

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fused, gate, flows = super().forward(features)
        return fused, {"gate": gate, "flows": flows}


class ConvGRUFusion(nn.Module):
    """Convolutional GRU over per-frame features (B, T, C, h, w) -> fused (B, C, h, w).

    What it does: a GRU whose matrix products are 3x3 convolutions, so it keeps one memory vector PER CELL of the
    feature map. For each frame (oldest -> current) an update gate z decides how much of the memory to overwrite
    and a reset gate r how much of it to use when proposing the new content. The final memory is projected back
    to the backbone width and added to the current features; the projection starts at zero, so the warm-started
    model begins exactly as its single-frame baseline. In deployment only the memory tensor is carried between
    frames (``step``/``readout``), not a window of past feature maps."""

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


class OcclusionGatedConvGRUFusion(ConvGRUFusion):
    """ConvGRU whose memory write is gated by a per-cell visibility of the incoming frame (``lite_v06_occ``).

    What it does: a 3x3 convolution on the frame's reduced features predicts a visibility v in [0, 1] per cell,
    and the GRU update gate becomes z * v, so ``state = (1 - z v) state + z v candidate``. Where a frame is
    occluded, blurred or dark (v -> 0) the memory is kept instead of being overwritten with bad evidence - the idea
    of the occlusion-aware memories of OMR (ECCV 2024) and SIGMA-Lane, reduced to one extra 3x3 conv. v looks at
    the incoming frame only (not at the memory), so it acts as a frame-quality estimate; it has no direct label and
    is learned through the lane loss, helped by the current-frame degradation of the training recipe. It starts
    open (bias +3, v = 0.95 everywhere) and, as in ``ConvGRUFusion``, the read-out starts at zero, so the
    warm-started model is exactly its single-frame baseline."""

    def __init__(self, channels: int, hidden: int = 64, kernel: int = 3, visibility_bias: float = 3.0) -> None:
        super().__init__(channels, hidden, kernel)
        self.visibility = nn.Conv2d(hidden, 1, kernel, padding=kernel // 2)
        nn.init.zeros_(self.visibility.weight)
        nn.init.constant_(self.visibility.bias, visibility_bias)

    def gated_step(self, features: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One GRU update and the visibility (B, 1, h, w) that gated it."""
        x = self.reduce(features)
        z, r = torch.sigmoid(self.gates(torch.cat([x, state], dim=1))).chunk(2, dim=1)
        visibility = torch.sigmoid(self.visibility(x))
        candidate = torch.tanh(self.candidate(torch.cat([x, r * state], dim=1)))
        write = z * visibility
        return (1.0 - write) * state + write * candidate, visibility

    def step(self, features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.gated_step(features, state)[0]

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state = self.init_state(features[:, 0])
        for t in range(features.shape[1]):
            state, visibility = self.gated_step(features[:, t], state)
        return self.readout(features[:, -1], state), {
            "state_norm": state.detach().abs().mean(dim=(1, 2, 3)),
            "visibility": visibility.detach().mean(dim=(1, 2, 3)),   # of the current frame
        }


class TemporalAttentionFusion(nn.Module):
    """Cross-frame attention: queries from the current frame, keys/values from every frame of the window.

    (B, T, C, h, w) -> fused (B, C, h, w). Features are reduced to ``dim`` channels; a depthwise 3x3 convolution
    adds position information (conditional positional encoding, Chu et al. 2023), so the module works at any
    feature-map size; keys and values are average-pooled to at most ``kv_cells`` (rows, cols) per frame, which
    keeps the cost at queries x (T x kv cells) whatever the input size.
    """

    def __init__(self, channels: int, num_frames: int, dim: int = 128, heads: int = 4,
                 kv_cells: tuple[int, int] = (12, 20)) -> None:
        super().__init__()
        self.kv_cells = kv_cells
        self.reduce = nn.Sequential(nn.Conv2d(channels, dim, 1, bias=False), nn.BatchNorm2d(dim))
        self.position = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.frame_embedding = nn.Parameter(torch.zeros(num_frames, dim))
        nn.init.normal_(self.frame_embedding, std=0.02)
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.project = nn.Conv2d(dim, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, t, c, h, w = features.shape
        x = self.reduce(features.reshape(b * t, c, h, w))
        x = x + self.position(x)
        dim = x.shape[1]
        query = x.reshape(b, t, dim, h, w)[:, -1].flatten(2).transpose(1, 2)          # (B, h*w, dim)
        kv = F.adaptive_avg_pool2d(x, (min(h, self.kv_cells[0]), min(w, self.kv_cells[1])))
        cells = kv.shape[-2] * kv.shape[-1]
        kv = kv.reshape(b, t, dim, cells).transpose(2, 3) + self.frame_embedding[-t:].view(1, t, 1, dim)
        kv = self.norm_kv(kv.reshape(b, t * cells, dim))
        out, weights = self.attention(self.norm_q(query), kv, kv, need_weights=True)  # weights (B, h*w, T*cells)
        out = out.transpose(1, 2).reshape(b, dim, h, w)
        history_share = weights[..., : (t - 1) * cells].sum(-1).mean(-1).detach()   # attention on past frames
        return features[:, -1] + self.project(out), {"history_attention": history_share}
