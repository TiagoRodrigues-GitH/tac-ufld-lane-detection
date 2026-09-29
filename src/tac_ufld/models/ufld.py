"""UFLD (Ultra-Fast Lane Detection, Qin et al., ECCV 2020) and the
supervisor's temporal variants built on it.

``UFLDNet`` re-implements the official ``parsingNet`` (github.com/cfzd/
Ultra-Fast-Lane-Detection, ``model/model.py``) with ``use_aux=False``:
ResNet backbone -> 1x1 conv to 8 channels -> flatten -> Linear(., 2048) ->
ReLU -> Linear(2048, (G+1)*A*L), reshaped to (B, G+1, A, L). The only
generalisation is that the flatten size is derived from the input size
instead of the hard-coded 1800 (= 8*9*25 for 288x800).

The temporal models (``TemporalWeightedFusion`` = v02/v04,
``GatedTemporalFusion`` = v03) are ported unchanged in structure from the
supervisor's notebook (``TACUFLDTemporalModel``,
``TACUFLDGatedTemporalModel``): every frame goes through the SAME UFLD
backbone, features are fused, then the UFLD head classifies the fused map.
All of them hold the UFLD network as ``self.ufld`` so baseline weights
warm-start the temporal models with a strict state-dict load.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tac_ufld.models.resnet import ResNetBackbone


def _official_init(module: nn.Module) -> None:
    """``real_init_weights`` from the official repo (applied to ``cls``)."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, 0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class UFLDNet(nn.Module):
    def __init__(
        self, num_lanes: int, num_row_anchors: int, griding_num: int,
        img_h: int, img_w: int, backbone: str = "18", pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.cls_dim = (griding_num + 1, num_row_anchors, num_lanes)
        self.model = ResNetBackbone(backbone, pretrained=pretrained)
        self.pool = nn.Conv2d(ResNetBackbone.out_channels, 8, 1)
        self.flat_dim = 8 * math.ceil(img_h / 32) * math.ceil(img_w / 32)
        self.cls = nn.Sequential(
            nn.Linear(self.flat_dim, 2048), nn.ReLU(), nn.Linear(2048, math.prod(self.cls_dim))
        )
        _official_init(self.cls)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[-1]

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        flat = self.pool(features).flatten(1)
        return self.cls(flat).view(-1, *self.cls_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(x))


class UFLDSingleFrame(nn.Module):
    """Single-frame UFLD baseline. Accepts (B, 3, H, W) or (B, T, 3, H, W)
    (uses the last, i.e. current, frame)."""

    temporal = False

    def __init__(self, ufld: UFLDNet) -> None:
        super().__init__()
        self.ufld = ufld

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() == 5:
            x = x[:, -1]
        return {"logits": self.ufld(x)}


class TemporalWeightedFusion(nn.Module):
    """v02/v04: one learnable softmax weight per frame (supervisor's design).
    For 3 frames the weights start at (0.10, 0.20, 0.70) as in the notebook."""

    def __init__(self, num_frames: int) -> None:
        super().__init__()
        if num_frames == 3:
            init = torch.log(torch.tensor([0.10, 0.20, 0.70]))
        else:
            init = torch.linspace(-1.0, 1.0, num_frames)
        self.temporal_logits = nn.Parameter(init.clone())

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        weights = F.softmax(self.temporal_logits, dim=0)
        fused = (features * weights.view(1, -1, 1, 1, 1)).sum(dim=1)
        return fused, {"fusion_weights": weights.detach()}


class GatedTemporalFusion(nn.Module):
    """v03: per-pixel softmax gates over frames predicted from the stacked
    features (supervisor's design). A fixed bias favours the current frame
    at initialisation ((-2, -1, 2) for 3 frames, as in the notebook)."""

    def __init__(self, feature_dim: int, num_frames: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(feature_dim * num_frames, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_frames, 1),
        )
        bias = torch.tensor([-2.0, -1.0, 2.0]) if num_frames == 3 else torch.linspace(-2.0, 2.0, num_frames)
        self.register_buffer("initial_bias", bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, t, c, h, w = features.shape
        logits = self.gate_net(features.reshape(b, t * c, h, w)) + self.initial_bias.view(1, t, 1, 1)
        gates = F.softmax(logits, dim=1).unsqueeze(2)  # (B, T, 1, H, W)
        return (features * gates).sum(dim=1), {"gates": gates}


class UFLDTemporal(nn.Module):
    """Shared-backbone temporal UFLD: encode every frame with ``ufld.model``,
    fuse, classify with the UFLD head."""

    temporal = True

    def __init__(self, ufld: UFLDNet, fusion: nn.Module) -> None:
        super().__init__()
        self.ufld = ufld
        self.fusion = fusion

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t, c, h, w = x.shape
        feats = self.ufld.encode(x.reshape(b * t, c, h, w))
        fused, extra = self.fusion(feats.reshape(b, t, *feats.shape[1:]))
        return {"logits": self.ufld.classify(fused), **extra}

    def per_frame_logits(self, x: torch.Tensor, indices: tuple[int, ...]) -> list[torch.Tensor]:
        """Single-frame predictions of selected frames (temporal-consistency loss)."""
        return [self.ufld(x[:, i]) for i in indices]

    def fusion_parameters(self) -> list[nn.Parameter]:
        return list(self.fusion.parameters())
