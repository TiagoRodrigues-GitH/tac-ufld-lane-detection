"""Lightweight models from the ELAS development script
(``tac_ufld_elas_pipeline_version_01_corrected.py``).

* ``lite_baseline`` = ``SingleFrameUFLDLikeModel``: 4-block CNN
  (``LightweightBackbone``) + ``LanePixelHead`` (per-lane, per-anchor MLP
  over pooled features, (G+1)-way UFLD-style classification).
* ``lite_v05``      = the warped residual temporal fusion
  (``LearnableFlowWarp`` + ``ResidualTemporalFusion``), previously labelled
  v02/v03/v04.
* ``lite_v06``      = the same backbone and head with a ConvGRU recurrent
  fusion (``tac_ufld.models.fusion.ConvGRUFusion``), added in v0.4.
* ``lite_tsm``      = the same backbone and head with an online temporal
  shift inside the backbone (``LiteTSM``, added 2026-10-19): no fusion
  module, no extra parameters or MACs.

Change for a fair comparison (audit C3): v05 now uses the SAME current-frame
backbone and head as ``lite_baseline`` and is warm-started from it, so the
only difference between the pair is the temporal pathway. The half-width
backbone is gone; the cheap ``TinyHistoryEncoder`` for history frames is
kept as an option (``model.lite_history_encoder: tiny``). ``LanePixelHead``
now returns logits in the official (B, G+1, A, L) layout; its unused
``y_head`` was dropped (its loss weight was 0).

Fix of 2026-10-18 (review): the head pooled the feature map to (A, G) with
``AdaptiveAvgPool2d``, i.e. anchors spread evenly over the whole image,
while the row anchors only cover the road band. Most anchors therefore read
the wrong image rows (the top ELAS anchor read the sky). ``AnchorRowPool``
reads each anchor at its own row; ``model.lite_anchor_rows: uniform``
reproduces the old head. Every lite result obtained before this date used
the misaligned head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    """Two 3x3 conv + BatchNorm + ReLU layers (the first may have stride 2, halving the resolution), with light
    spatial dropout. The building block of the light backbone."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.05) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConvBlock(nn.Module):
    """MobileNet-style block: a per-channel (depthwise) 3x3 conv followed by a 1x1 (pointwise) conv. About 8-9x
    fewer multiply-adds than a plain 3x3 conv with the same channels; used by the optional tiny history encoder."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.05) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightweightBackbone(nn.Module):
    """Four stride-2 ConvBlocks (32/64/96/128 channels), output stride 16; trained from scratch (no ImageNet).

    About 0.46 M parameters versus 11.2 M for the ResNet-18 of UFLD. Its receptive field is about 120 px, so every
    output cell sees only a local neighbourhood: the head must read the cells at the right image rows
    (``AnchorRowPool``)."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 128, dropout: float = 0.05) -> None:
        super().__init__()
        self.out_channels = feature_dim
        self.net = nn.Sequential(
            ConvBlock(in_channels, 32, 2, dropout), ConvBlock(32, 64, 2, dropout),
            ConvBlock(64, 96, 2, dropout), ConvBlock(96, feature_dim, 2, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyHistoryEncoder(nn.Module):
    """Cheap depthwise-separable encoder for history frames (same output shape as the backbone).

    Optional (``model.lite_history_encoder: tiny``): past frames only need to supply context, so a cheaper encoder
    can replace the shared backbone for them. Off by default, because a different encoder for the history would
    confound the comparison with the single-frame baseline."""

    def __init__(self, in_channels: int = 3, out_channels: int = 128, mid: int = 12, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            DepthwiseSeparableConvBlock(in_channels, mid, 2, dropout),
            DepthwiseSeparableConvBlock(mid, mid * 2, 2, dropout),
            DepthwiseSeparableConvBlock(mid * 2, mid * 2, 2, dropout),
            DepthwiseSeparableConvBlock(mid * 2, out_channels, 2, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _adaptive_bins(in_size: int, out_size: int) -> np.ndarray:
    """(out, in) averaging matrix with the bins of PyTorch's adaptive average pooling."""
    m = np.zeros((out_size, in_size), dtype=np.float32)
    for i in range(out_size):
        start, end = (i * in_size) // out_size, -(-((i + 1) * in_size) // out_size)
        m[i, start:end] = 1.0 / (end - start)
    return m


def _row_interpolation(anchor_fracs: tuple[float, ...], in_rows: int) -> np.ndarray:
    """(A, h) linear-interpolation matrix: row ``a`` reads the feature map at the image row of anchor ``a``.

    ``anchor_fracs`` are anchor rows as fractions of the input height (pixel-centre convention); feature row ``i``
    covers image rows ``[i, i+1) * H / h``, so its centre sits at fraction ``(i + 0.5) / h``."""
    pos = np.clip(np.asarray(anchor_fracs, dtype=np.float64) * in_rows - 0.5, 0, in_rows - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, in_rows - 1)
    w_hi = pos - lo
    m = np.zeros((len(pos), in_rows), dtype=np.float32)
    np.add.at(m, (np.arange(len(pos)), lo), 1.0 - w_hi)
    np.add.at(m, (np.arange(len(pos)), hi), w_hi)
    return m


class AnchorRowPool(nn.Module):
    """(B, C, h, w) -> (B, C, A, G): features AT each row anchor, averaged into G horizontal cells.

    What it does: for every row anchor it interpolates the feature map at that anchor's image row, then averages
    the columns into the G grid cells (the bins of ``AdaptiveAvgPool2d``). Two fixed matrix products, so it exports
    to ONNX/TensorRT unchanged.

    Why it exists (review 2026-10-18): ``AdaptiveAvgPool2d((A, G))`` spreads the A anchors evenly over the WHOLE
    feature map, but the anchors only cover the road band (ELAS: 59-99.5 % of the height). With that pooling the
    top anchor (y = 226 of 384) read features of rows 0-77 (sky): the light head could not see the lane at most
    anchors, and only the anchors near the bottom of the image were aligned."""

    def __init__(self, anchor_fracs: list[float] | tuple[float, ...], griding_num: int) -> None:
        super().__init__()
        self.grid = griding_num
        self.anchor_fracs = tuple(float(f) for f in anchor_fracs)  # plain floats: no state-dict entry
        self._cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def extra_repr(self) -> str:
        return f"anchors={len(self.anchor_fracs)}, grid={self.grid}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Matrices are built from Python ints with NumPy (cached per size/device/dtype), so an ONNX trace records
        # them as constants.
        h, w = int(x.shape[-2]), int(x.shape[-1])
        key = (h, w, x.device, x.dtype)
        if key not in self._cache:
            rows = torch.from_numpy(_row_interpolation(self.anchor_fracs, h)).to(device=x.device, dtype=x.dtype)
            cols = torch.from_numpy(_adaptive_bins(w, self.grid).T.copy()).to(device=x.device, dtype=x.dtype)
            self._cache[key] = (rows, cols)
        rows, cols = self._cache[key]
        return torch.matmul(torch.matmul(rows, x), cols)


class LanePixelHead(nn.Module):
    """Row-anchor classification head of the light models (UFLD-style output, local instead of global features).

    What it does: a 3x3 conv reduces the backbone features to ``mid`` channels; the map is pooled to one feature
    row per anchor and G cells per row (A, G); a 1x1 conv gives each lane slot its own projection; one MLP, shared
    by every (lane, anchor) pair, reads the G cells of that anchor row and outputs G+1 logits (G cells + "no
    lane"), returned in the official UFLD layout (B, G+1, A, L). The "no lane" bias starts at +1 (conservative).

    ``anchor_fracs`` (anchor rows as fractions of the image height) selects the anchor-aligned pooling
    (``AnchorRowPool``); ``None`` keeps the original ``AdaptiveAvgPool2d((A, G))``, which is misaligned with the
    anchors and is kept only to reproduce runs made before 2026-10-18 (``model.lite_anchor_rows: uniform``)."""

    def __init__(self, feature_dim: int, num_lanes: int, num_row_anchors: int, griding_num: int,
                 mid: int = 96, hidden: int = 192, dropout: float = 0.15,
                 anchor_fracs: list[float] | None = None) -> None:
        super().__init__()
        self.lanes, self.anchors, self.grid, self.mid = num_lanes, num_row_anchors, griding_num, mid
        self.pre = nn.Sequential(
            nn.Conv2d(feature_dim, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
        )
        if anchor_fracs is None:
            self.pool = nn.AdaptiveAvgPool2d((num_row_anchors, griding_num))
        else:
            if len(anchor_fracs) != num_row_anchors:
                raise ValueError(f"{len(anchor_fracs)} anchor rows given for {num_row_anchors} row anchors")
            self.pool = AnchorRowPool(anchor_fracs, griding_num)
        self.lane_proj = nn.Conv2d(mid, num_lanes * mid, 1)
        self.mlp = nn.Sequential(
            nn.Linear(mid * griding_num, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Linear(hidden, griding_num + 1)
        nn.init.zeros_(self.cls_head.weight)
        with torch.no_grad():
            self.cls_head.bias.zero_()
            self.cls_head.bias[-1] = 1.0

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b = feat.shape[0]
        h = self.lane_proj(self.pool(self.pre(feat)))                    # (B, L*mid, A, G)
        h = h.reshape(b, self.lanes, self.mid, self.anchors, self.grid)
        h = h.permute(0, 1, 3, 2, 4).reshape(b * self.lanes * self.anchors, self.mid * self.grid)
        logits = self.cls_head(self.mlp(h)).reshape(b, self.lanes, self.anchors, self.grid + 1)
        return logits.permute(0, 3, 2, 1).contiguous()                   # (B, G+1, A, L)


class LiteSingleFrame(nn.Module):
    """Light single-frame baseline (``lite_baseline``): ``LightweightBackbone`` + ``LanePixelHead`` on the current
    frame only. The reference that every light temporal model is compared against."""

    temporal = False

    def __init__(self, backbone: LightweightBackbone, head: LanePixelHead) -> None:
        super().__init__()
        self.backbone, self.head = backbone, head

    def encode_frames(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"current": self.backbone(x)}

    def forward_features(self, history: list[torch.Tensor], current: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.head(current)}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() == 5:
            x = x[:, -1]
        return {"logits": self.head(self.backbone(x))}


class LearnableFlowWarp(nn.Module):
    """Predict a per-pixel offset between f(t-k) and f(t); warp f(t-k) onto f(t).

    What it does: a small CNN looks at the past and current feature maps side by side and predicts, for every
    cell, a 2-D displacement (at most ``max_disp`` cells, bounded by tanh); the past map is resampled at the
    displaced positions (bilinear ``grid_sample``), so a marking that moved between frames lines up with its
    current position before fusion. Learned without flow labels (only through the lane loss), so the "flow" is
    whatever displacement helps detection - it can also move features vertically, which is why the capacity
    control ``lite_v05_static`` (same module, history = current frame) is needed to read its gains."""

    def __init__(self, channels: int, max_disp: float = 8.0, hidden: int = 64) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.flow_net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 3, padding=1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1, bias=False), nn.BatchNorm2d(hidden // 2), nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, 2, 3, padding=1),
        )

    def forward(self, feat_prev: torch.Tensor, feat_cur: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = feat_prev.shape
        flow = torch.tanh(self.flow_net(torch.cat([feat_prev, feat_cur], dim=1))) * self.max_disp
        ys, xs = torch.meshgrid(
            torch.arange(h, device=flow.device, dtype=flow.dtype),
            torch.arange(w, device=flow.device, dtype=flow.dtype), indexing="ij",
        )
        sample = torch.stack([xs, ys], dim=0).unsqueeze(0) + flow
        grid = torch.stack(
            [sample[:, 0] / max(w - 1, 1) * 2 - 1, sample[:, 1] / max(h - 1, 1) * 2 - 1], dim=-1
        )
        warped = F.grid_sample(feat_prev, grid.to(feat_prev.dtype), mode="bilinear",
                               padding_mode="border", align_corners=True)
        return warped, flow


class ResidualTemporalFusion(nn.Module):
    """fused = f_cur + gate * (history - f_cur); per-pixel gate starts closed.

    What it does: every past frame is warped onto the current one (``LearnableFlowWarp``), the warped maps are
    compressed into one history map (1x1 conv), and a per-cell gate in [0, 1] decides how much of it replaces the
    current features. The gate bias starts at -2 (gate ~0.12), so training starts close to the single-frame
    model and opens the gate only where the past helps (e.g. a marking hidden in the current frame)."""

    def __init__(self, channels: int, num_frames: int, hidden: int = 64, max_disp: float = 8.0,
                 gate_bias_init: float = -2.0) -> None:
        super().__init__()
        self.warp = LearnableFlowWarp(channels, max_disp=max_disp, hidden=hidden)
        self.history_conv = nn.Sequential(
            nn.Conv2d(channels * max(num_frames - 1, 1), channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 3, padding=1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.constant_(self.gate_net[-1].bias, gate_bias_init)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        f_cur = features[:, -1]
        warped, flows = [], []
        for k in range(features.shape[1] - 1):
            w_k, flow_k = self.warp(features[:, k], f_cur)
            warped.append(w_k)
            flows.append(flow_k)
        history = self.history_conv(torch.cat(warped, dim=1))
        gate = torch.sigmoid(self.gate_net(torch.cat([f_cur, history], dim=1)))
        return f_cur + gate * (history - f_cur), gate, flows


def _fusion_outputs(result) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``ResidualTemporalFusion`` returns (fused, gate, flows); the v0.4
    fusions return (fused, extras)."""
    if len(result) == 2 and isinstance(result[1], dict):
        return result
    fused, gate, flows = result
    return fused, {"gate": gate, "flows": flows}


class LiteTemporal(nn.Module):
    """Lite temporal model: the lite baseline's backbone and head around a
    temporal fusion (default: the v0.5 warped residual fusion; ``fusion=``
    any module mapping (B, T, C, h, w) -> (fused, extras), e.g. the ConvGRU
    of lite_v06).

    Streaming: the fusion depends on the current frame and is recomputed
    every step; the per-frame history features (backbone or tiny encoder
    output) depend on their own frame only and are cached."""

    temporal = True

    def __init__(self, backbone: LightweightBackbone, head: LanePixelHead, num_frames: int,
                 history_encoder: str = "shared", in_channels: int = 3, dropout: float = 0.05,
                 fusion: nn.Module | None = None) -> None:
        super().__init__()
        self.backbone, self.head = backbone, head
        channels = backbone.out_channels
        self.history_encoder = (TinyHistoryEncoder(in_channels, channels, dropout=dropout)
                                if history_encoder == "tiny" else None)
        self.fusion = fusion if fusion is not None else ResidualTemporalFusion(
            channels, num_frames, hidden=64, max_disp=8.0)

    def encode_frames(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        current = self.backbone(x)
        history = current if self.history_encoder is None else self.history_encoder(x)
        return {"current": current, "history": history}

    def forward_features(self, history: list[torch.Tensor], current: torch.Tensor) -> dict[str, torch.Tensor]:
        if not history:
            return {"logits": self.head(current)}
        fused, extras = _fusion_outputs(self.fusion(torch.stack([*history, current], dim=1)))
        return {"logits": self.head(fused), **extras}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t, c, h, w = x.shape
        feat_cur = self.backbone(x[:, -1])
        if t == 1:
            return {"logits": self.head(feat_cur)}
        encoder = self.history_encoder or self.backbone
        hist = encoder(x[:, :-1].reshape(b * (t - 1), c, h, w))
        feats = torch.cat([hist.reshape(b, t - 1, *feat_cur.shape[1:]), feat_cur.unsqueeze(1)], dim=1)
        fused, extras = _fusion_outputs(self.fusion(feats))
        return {"logits": self.head(fused), **extras}

    def per_frame_logits(self, x: torch.Tensor, indices: tuple[int, ...]) -> list[torch.Tensor]:
        return [self.head(self.backbone(x[:, i])) for i in indices]

    def fusion_parameters(self) -> list[nn.Parameter]:
        params = list(self.fusion.parameters())
        if self.history_encoder is not None:
            params += list(self.history_encoder.parameters())
        return params

    # -- recurrent fusion only (lite_v06): carry one hidden state per stream

    @property
    def recurrent(self) -> bool:
        return bool(getattr(self.fusion, "recurrent", False))

    def recurrent_step(self, current: torch.Tensor, state: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """One streaming step with a carried hidden state (``None`` = start of
        stream), updated from the backbone features of the current frame."""
        if not self.recurrent:
            raise TypeError("recurrent_step needs a recurrent fusion (lite_v06)")
        state = self.fusion.init_state(current) if state is None else state
        state = self.fusion.step(current, state)
        return {"logits": self.head(self.fusion.readout(current, state)), "state": state}


LiteWarpTemporal = LiteTemporal  # v0.3 name, kept for existing imports


def temporal_shift(x: torch.Tensor, fold: int) -> torch.Tensor:
    """Online (uni-directional) temporal shift of a clip (B, T, C, h, w): the first ``fold`` channels of frame t
    are those of frame t-1; the first frame of the clip receives zeros there. The other channels are untouched."""
    past = torch.cat([torch.zeros_like(x[:, :1, :fold]), x[:, :-1, :fold]], dim=1)
    return torch.cat([past, x[:, :, fold:]], dim=2)


class LiteTSM(nn.Module):
    """Lite backbone with an online temporal shift (``lite_tsm``; TSM, Lin et al., ICCV 2019, and its online
    edge-device version).

    What it does: at the input of backbone blocks 2-4, 1/``fold_div`` of the channels are replaced by the same
    channels of the previous frame at the same layer (zeros for the first frame of a clip), so every block mixes
    information across time with its ordinary convolutions. The layers and their weights are exactly those of
    ``lite_baseline`` (the state dict is identical, so the warm start is a strict full load): zero extra parameters
    and MACs. Unlike the fusions of ``LiteTemporal`` the shift changes the current frame's features from the first
    shifted block on, so the warm-started model does not start as its baseline.

    Temporal reach: with shifts in 3 blocks the current output depends on exactly 3 previous frames (one more per
    shifted block) and on nothing older, but a ``num_frames``-clip only supplies ``num_frames - 1``; the deeper
    shifts see zeros in training (first-frame padding).

    Streaming, two exact splits:

    * window (``encode_frames`` / ``forward_features``, streaming modes "cached"/"recompute", ONNX encoder/head):
      the per-frame cache holds block-1 output (the input of the first shifted block, a function of its frame
      only) and the head re-runs blocks 2-4 over the window. Exactly the trained computation, but blocks 2-4 run
      ``num_frames`` times per frame.
    * carry (``recurrent_step``, streaming mode "carry", ONNX ``step``): the online TSM deployment. The state is
      the shifted slice of every shifted block (3 small tensors); each frame costs one backbone pass, the
      baseline's cost. The memory is finite: carry equals the clip forward over the last ``len(shift_blocks) + 1``
      frames of the chain (4 with the default), so it matches the 3-frame training clip only up to the deepest
      shift, which sees a real slice of frame t-3 where training saw zeros (the train/deploy gap of online TSM,
      small but not zero).
    """

    temporal = True
    recurrent = True  # carries per-layer slices between frames (streaming mode "carry")

    def __init__(self, backbone: LightweightBackbone, head: LanePixelHead,
                 shift_blocks: tuple[int, ...] = (1, 2, 3), fold_div: int = 8) -> None:
        super().__init__()
        self.backbone, self.head = backbone, head
        blocks = list(backbone.net)
        if not shift_blocks or sorted(shift_blocks) != list(shift_blocks) or not 0 < shift_blocks[0] < len(blocks):
            raise ValueError(f"shift_blocks must be increasing block indices in [1, {len(blocks) - 1}]")
        self.shift_blocks = tuple(int(i) for i in shift_blocks)  # 0-based: (1, 2, 3) = blocks 2-4
        self.fold_div = int(fold_div)
        self.folds = tuple(blocks[i].block[0].in_channels // self.fold_div for i in self.shift_blocks)
        if min(self.folds) < 1:
            raise ValueError(f"fold_div {fold_div} leaves no channel to shift")

    def extra_repr(self) -> str:
        return f"shift_blocks={self.shift_blocks}, folds={self.folds}"

    def _stem(self, x: torch.Tensor) -> torch.Tensor:
        """Blocks before the first shift: a function of the frame only (what the window cache stores)."""
        for block in self.backbone.net[:self.shift_blocks[0]]:
            x = block(x)
        return x

    def _shifted(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, T, C, h, w) stem features -> (B, T, C', h', w') backbone features, shifting at every shifted block."""
        b, t = feats.shape[:2]
        x = feats
        for i in range(self.shift_blocks[0], len(self.backbone.net)):
            if i in self.shift_blocks:
                x = temporal_shift(x, self.folds[self.shift_blocks.index(i)])
            y = self.backbone.net[i](x.reshape(b * t, *x.shape[2:]))
            x = y.reshape(b, t, *y.shape[1:])
        return x

    def _clip_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        b, t = x.shape[:2]
        stem = self._stem(x.reshape(b * t, *x.shape[2:]))
        return self._shifted(stem.reshape(b, t, *stem.shape[1:]))

    def encode_frames(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self._stem(x)
        return {"current": stem, "history": stem}

    def forward_features(self, history: list[torch.Tensor], current: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.head(self._shifted(torch.stack([*history, current], dim=1))[:, -1])}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.head(self._clip_features(x)[:, -1])}

    def per_frame_logits(self, x: torch.Tensor, indices: tuple[int, ...]) -> list[torch.Tensor]:
        """Output at each clip position, i.e. with the causal context the clip gives that frame (the shift makes
        a frame's features depend on its predecessors, so there is no independent single-frame pass)."""
        feats = self._clip_features(x)
        return [self.head(feats[:, i]) for i in indices]

    def recurrent_step(self, current: torch.Tensor, state: tuple[torch.Tensor, ...] | None
                       ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        """One online step: ``current`` = stem features of the new frame (``encode_frames``), ``state`` = the
        previous frame's slices (``None`` = start of stream, zeros as for the first frame of a clip). Returns the
        logits and this frame's slices."""
        x, new_state = current, []
        for i in range(self.shift_blocks[0], len(self.backbone.net)):
            if i in self.shift_blocks:
                k = self.shift_blocks.index(i)
                fold = self.folds[k]
                past = x.new_zeros(x.shape[0], fold, *x.shape[2:]) if state is None else state[k]
                new_state.append(x[:, :fold].clone())
                x = torch.cat([past.to(x.dtype), x[:, fold:]], dim=1)
            x = self.backbone.net[i](x)
        return {"logits": self.head(x), "state": tuple(new_state)}
