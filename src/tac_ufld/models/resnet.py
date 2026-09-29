"""ResNet-18/34 backbone, layer-for-layer identical to ``torchvision.models``.

The official UFLD code wraps ``torchvision.models.resnet18`` and returns the
outputs of ``layer2``, ``layer3`` and ``layer4``. This module reproduces
that architecture with the same parameter names, so the official torchvision
ImageNet checkpoints load with ``strict=True`` (verified by
``tests/test_models.py`` when the weights can be downloaded). Keeping it
local avoids a torchvision dependency (no wheels for every Python/CUDA
combination) without changing the network.
"""

from __future__ import annotations

import torch
from torch import nn

IMAGENET_WEIGHTS = {
    "18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
}
_LAYERS = {"18": (2, 2, 2, 2), "34": (3, 4, 6, 3)}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetBackbone(nn.Module):
    """Returns (layer2, layer3, layer4) features like the official UFLD wrapper."""

    out_channels = 512

    def __init__(self, depth: str = "18", pretrained: bool = True) -> None:
        super().__init__()
        if depth not in _LAYERS:
            raise ValueError(f"unsupported ResNet depth '{depth}' (supported: {sorted(_LAYERS)})")
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        blocks, in_ch = [], 64
        for i, (n, width) in enumerate(zip(_LAYERS[depth], (64, 128, 256, 512))):
            stride = 1 if i == 0 else 2
            layer = [BasicBlock(in_ch, width, stride)] + [BasicBlock(width, width) for _ in range(n - 1)]
            blocks.append(nn.Sequential(*layer))
            in_ch = width
        self.layer1, self.layer2, self.layer3, self.layer4 = blocks
        self._init_weights()
        if pretrained:
            load_imagenet_weights(self, depth)

    def _init_weights(self) -> None:  # torchvision defaults
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4


def load_imagenet_weights(module: ResNetBackbone, depth: str) -> None:
    """Download (cached under ~/.cache/torch/hub) and load torchvision's
    ImageNet-1k weights; the classifier head (``fc``) is dropped."""
    try:
        state = torch.hub.load_state_dict_from_url(IMAGENET_WEIGHTS[depth], map_location="cpu", progress=True)
    except Exception as exc:  # network errors surface with an actionable message
        raise RuntimeError(
            f"Could not download ImageNet weights for ResNet-{depth} ({exc}). "
            f"Connect to the internet once, or set model.pretrained: false."
        ) from exc
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    module.load_state_dict(state, strict=True)
