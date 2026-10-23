"""Official UFLD auxiliary segmentation (model.ufld_aux): training-only output, labels, loss, parameter count."""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from tac_ufld.data.targets import seg_from_cls
from tac_ufld.evaluation.efficiency import count_parameters
from tac_ufld.losses import Hyperparams, VariantLoss
from tac_ufld.models.registry import VARIANTS, build_model


def with_aux(cfg, on: bool = True):
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, ufld_aux=on))
    return cfg


def test_aux_output_only_in_training_for_baseline_and_temporal(tiny_config):
    cfg = with_aux(tiny_config)
    d = cfg.data
    x = torch.rand(2, d.num_frames, 3, d.img_h, d.img_w)
    for variant in ["ufld_baseline", "ufld_v03", "ufld_v08"]:
        model = build_model(variant, cfg, pretrained=False)
        out = model.train()(x)
        assert out["aux_seg"].shape == (2, d.num_lanes + 1, -(-d.img_h // 8), -(-d.img_w // 8)), variant
        assert "aux_seg" not in model.eval()(x), variant


def test_reported_parameters_exclude_the_aux_branch(tiny_config):
    with_branch = build_model("ufld_baseline", with_aux(tiny_config), pretrained=False)
    without = build_model("ufld_baseline", with_aux(tiny_config, False), pretrained=False)
    assert sum(p.numel() for p in with_branch.parameters()) > sum(p.numel() for p in without.parameters())
    assert count_parameters(with_branch) == count_parameters(without)


def test_seg_label_follows_the_row_anchor_targets():
    anchors = np.linspace(20, 60, 5).astype(np.float32)          # rows in model pixels (64 x 96 input)
    cls = np.full((5, 2), 24, dtype=np.int64)                      # G = 24: "no lane" everywhere ...
    cls[:, 1] = 12                                                 # ... except slot 1 at cell 12 (x = 12*95/23 = 49.6 px)
    seg = seg_from_cls(cls, anchors, img_w=96, img_h=64, griding_num=24)
    assert seg.shape == (8, 12)
    rows = np.flatnonzero((seg == 2).any(axis=1))
    assert rows.min() >= 2 and rows.max() <= 7                     # only between the first and last anchor
    assert set(np.flatnonzero(seg[4] == 2)) == {6}                 # x = 49.6 px -> cell floor(49.6 / 8) = 6
    assert not (seg == 1).any()                                    # absent lane: background


def test_loss_adds_the_aux_term(tiny_config):
    cfg = with_aux(tiny_config)
    d = cfg.data
    model = build_model("ufld_baseline", cfg, pretrained=False).train()
    x = torch.rand(2, d.num_frames, 3, d.img_h, d.img_w)
    out = model(x)
    a, lanes = d.num_row_anchors, d.num_lanes
    batch = {"images": x, "cls": torch.full((2, a, lanes), d.griding_num), "exist": torch.zeros(2, a, lanes),
             "x": torch.zeros(2, a, lanes), "valid": torch.ones(2, a, lanes, dtype=torch.bool),
             "seg": torch.zeros(2, *out["aux_seg"].shape[-2:], dtype=torch.long)}
    hp = Hyperparams(lr=1e-3, lr_fusion=1e-3, weight_decay=0.0, lambda_temporal=0.0, lambda_coord=0.0)
    total, logs = VariantLoss(VARIANTS["ufld_baseline"], cfg.train.loss, hp, d.img_w)(model, out, batch)
    assert "aux_seg" in logs and logs["total"] > logs["focal"]
    total.backward()
    assert model.ufld.aux_combine[-1].weight.grad is not None
