"""Temporal attention fusion (ufld_v08, lite_v07): starts as the baseline, any feature size, gradients flow."""

from __future__ import annotations

import pytest
import torch

from tac_ufld.models.fusion import TemporalAttentionFusion
from tac_ufld.models.registry import VARIANTS, build_model, training_order


def test_zero_initialised_readout_returns_the_current_features():
    fusion = TemporalAttentionFusion(channels=32, num_frames=3, dim=16, heads=2).eval()
    feats = torch.randn(2, 3, 32, 6, 10)
    fused, extras = fusion(feats)
    assert torch.equal(fused, feats[:, -1])
    share = extras["history_attention"]
    assert share.shape == (2,) and bool(((share >= 0) & (share <= 1)).all())


@pytest.mark.parametrize("size", [(6, 10), (24, 40)])  # pooled keys above kv_cells, plain keys below
def test_any_feature_size_and_gradients_reach_the_attention(size):
    fusion = TemporalAttentionFusion(channels=16, num_frames=3, dim=16, heads=2, kv_cells=(12, 20))
    torch.nn.init.normal_(fusion.project.weight, std=0.1)  # as after some training
    feats = torch.randn(2, 3, 16, *size, requires_grad=True)
    fused, _ = fusion(feats)
    assert fused.shape == (2, 16, *size)
    fused.square().mean().backward()
    assert fusion.attention.in_proj_weight.grad is not None and feats.grad[:, 0].abs().sum() > 0  # history used


@pytest.mark.parametrize("variant", ["ufld_v08", "lite_v07"])
def test_attention_variants_build_and_warm_start_from_their_baseline(variant, tiny_config):
    cfg = tiny_config
    spec = VARIANTS[variant]
    assert spec.temporal and spec.fusion == "attention"
    assert training_order([variant], cfg)[0] == spec.warm_start
    model = build_model(variant, cfg, pretrained=False).eval()
    x = torch.rand(1, cfg.data.num_frames, 3, cfg.data.img_h, cfg.data.img_w)
    out = model(x)
    assert "logits" in out and "history_attention" in out
