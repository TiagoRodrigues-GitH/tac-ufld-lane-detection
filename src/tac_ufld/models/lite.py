"""Lightweight models from the ELAS development script
(``tac_ufld_elas_pipeline_version_01_corrected.py``).

* ``lite_baseline`` = ``SingleFrameUFLDLikeModel``: 4-block CNN
  (``LightweightBackbone``) + ``LanePixelHead`` (per-lane, per-anchor MLP
  over pooled features, (G+1)-way UFLD-style classification).
* ``lite_v05``      = the warped residual temporal fusion
  (``LearnableFlowWarp`` + ``ResidualTemporalFusion``), previously labelled
  v02/v03/v04.

Change for a fair comparison (audit C3): v05 now uses the SAME current-frame
backbone and head as ``lite_baseline`` and is warm-started from it, so the
only difference between the pair is the temporal pathway. The half-width
backbone is gone; the cheap ``TinyHistoryEncoder`` for history frames is
kept as an option (``model.lite_history_encoder: tiny``). ``LanePixelHead``
now returns logits in the official (B, G+1, A, L) layout; its unused
``y_head`` was dropped (its loss weight was 0).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
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
    """Four stride-2 ConvBlocks (32/64/96/128 channels), output stride 16."""

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
    """Cheap depthwise-separable encoder for history frames (same output shape)."""

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


class LanePixelHead(nn.Module):
    """Row-anchor classification head: pool features to (A, G), project per
    lane, run a shared MLP per (lane, anchor) and output G+1 logits (G cells
    + "no lane"). The "no lane" bias starts at +1 (conservative start)."""

    def __init__(self, feature_dim: int, num_lanes: int, num_row_anchors: int, griding_num: int,
                 mid: int = 96, hidden: int = 192, dropout: float = 0.15) -> None:
        super().__init__()
        self.lanes, self.anchors, self.grid, self.mid = num_lanes, num_row_anchors, griding_num, mid
        self.pre = nn.Sequential(
            nn.Conv2d(feature_dim, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((num_row_anchors, griding_num))
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
    temporal = False

    def __init__(self, backbone: LightweightBackbone, head: LanePixelHead) -> None:
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() == 5:
            x = x[:, -1]
        return {"logits": self.head(self.backbone(x))}


class LearnableFlowWarp(nn.Module):
    """Predict a per-pixel offset between f(t-k) and f(t); warp f(t-k) onto f(t)."""

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
    """fused = f_cur + gate * (history - f_cur); per-pixel gate starts closed."""

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


class LiteWarpTemporal(nn.Module):
    temporal = True

    def __init__(self, backbone: LightweightBackbone, head: LanePixelHead, num_frames: int,
                 history_encoder: str = "shared") -> None:
        super().__init__()
        self.backbone, self.head = backbone, head
        channels = backbone.out_channels
        self.history_encoder = TinyHistoryEncoder(out_channels=channels) if history_encoder == "tiny" else None
        self.fusion = ResidualTemporalFusion(channels, num_frames, hidden=64, max_disp=8.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t, c, h, w = x.shape
        feat_cur = self.backbone(x[:, -1])
        if t == 1:
            return {"logits": self.head(feat_cur)}
        encoder = self.history_encoder or self.backbone
        hist = encoder(x[:, :-1].reshape(b * (t - 1), c, h, w))
        feats = torch.cat([hist.reshape(b, t - 1, *feat_cur.shape[1:]), feat_cur.unsqueeze(1)], dim=1)
        fused, gate, flows = self.fusion(feats)
        return {"logits": self.head(fused), "gate": gate, "flows": flows}

    def per_frame_logits(self, x: torch.Tensor, indices: tuple[int, ...]) -> list[torch.Tensor]:
        return [self.head(self.backbone(x[:, i])) for i in indices]

    def fusion_parameters(self) -> list[nn.Parameter]:
        params = list(self.fusion.parameters())
        if self.history_encoder is not None:
            params += list(self.history_encoder.parameters())
        return params
