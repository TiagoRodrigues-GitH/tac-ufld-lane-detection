"""Model shapes, warm starts and the ResNet/torchvision weight compatibility."""

from __future__ import annotations

import pytest
import torch

from tac_ufld.config import PROJECT_ROOT, load_config
from tac_ufld.models.registry import VARIANTS, build_model, training_order, warm_start
from tac_ufld.models.resnet import ResNetBackbone


@pytest.fixture(scope="module")
def cfg():
    return load_config(PROJECT_ROOT / "configs" / "elas_smoke.yaml", {
        "data.img_h": 64, "data.img_w": 96, "data.griding_num": 24, "data.num_row_anchors": 8,
        "model.pretrained": False, "device": "cpu"})


@pytest.mark.parametrize("key", list(VARIANTS))
def test_forward_shapes(cfg, key):
    model = build_model(key, cfg, pretrained=False).eval()
    x = torch.randn(2, cfg.data.num_frames, 3, 64, 96)
    out = model(x)
    assert out["logits"].shape == (2, 25, 8, 2)
    if VARIANTS[key].temporal:
        per = model.per_frame_logits(x, (1, 2))
        assert len(per) == 2 and per[0].shape == (2, 25, 8, 2)


def test_warm_starts(cfg):
    base = build_model("ufld_baseline", cfg, pretrained=False)
    v02 = build_model("ufld_v02", cfg)
    assert warm_start(v02, base.state_dict()) == ["ufld"]
    assert torch.equal(v02.ufld.cls[0].weight, base.ufld.cls[0].weight)
    v04 = build_model("ufld_v04", cfg)
    assert warm_start(v04, v02.state_dict()) == ["<all>"]
    lite = build_model("lite_baseline", cfg)
    v05 = build_model("lite_v05", cfg)
    assert warm_start(v05, lite.state_dict()) == ["backbone", "head"]


def test_training_order_puts_references_first(cfg):
    order = training_order(["ufld_v03", "lite_v05", "ufld_v04"], cfg)
    assert order.index("ufld_baseline") < order.index("ufld_v03")
    assert order.index("lite_baseline") < order.index("lite_v05")


def test_temporal_ablation_changes_nothing_for_single_frame_models(cfg):
    model = build_model("ufld_baseline", cfg, pretrained=False).eval()
    x = torch.randn(1, 3, 3, 64, 96)
    static = x[:, -1:].expand_as(x)
    assert torch.allclose(model(x)["logits"], model(static)["logits"])


@pytest.mark.network
def test_imagenet_weights_load_strictly():
    try:
        ResNetBackbone("18", pretrained=True)
    except RuntimeError as exc:
        if "Could not download" in str(exc):
            pytest.skip("offline")
        raise
