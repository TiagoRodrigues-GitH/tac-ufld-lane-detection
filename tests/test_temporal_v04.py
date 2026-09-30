"""v0.4 temporal improvements: current-frame degradation, backbone learning
rate / freezing, cross-dataset initialisation, aligned and recurrent fusion,
equal-training controls, the Kalman reference, carried recurrent state and
the ablation runner's grid / reuse options."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from tac_ufld.config import PROJECT_ROOT, AugmentationConfig, ConfigError, load_config
from tac_ufld.data.transforms import PhotometricAugmenter
from tac_ufld.evaluation.predictor import Predictions
from tac_ufld.evaluation.tracking import KalmanParams, LaneKalmanTracker, sequence_order, track_predictions
from tac_ufld.inference.loading import load_model
from tac_ufld.inference.streaming import StreamingLaneDetector
from tac_ufld.losses import Hyperparams
from tac_ufld.models.registry import (
    VARIANTS, backbone_module, build_model, init_from_checkpoint, resolve_spec, training_order, warm_start,
)
from tests.test_streaming import _cfg, _checkpoint, _frames

# ------------------------------------------------------- current-frame degradation


def _clip(t: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.rand(t, 3, 48, 64, generator=g)


def test_current_frame_degradation_changes_only_the_current_frame():
    cfg = AugmentationConfig(prob=0.0, current_frame_prob=1.0)  # photometric off: isolate the degradation
    aug = PhotometricAugmenter(cfg)
    for op in ("occlude", "blur", "darken", "noise"):
        cfg.current_frame_ops = [op]
        clip = _clip()
        random.seed(1)
        torch.manual_seed(1)
        out = aug(clip.clone())
        assert torch.equal(out[:-1], clip[:-1]), op          # history untouched
        assert not torch.equal(out[-1], clip[-1]), op        # current degraded
        assert out.shape == clip.shape and float(out.min()) >= 0.0


def test_occlusion_boxes_lie_in_the_road_band():
    cfg = AugmentationConfig(prob=0.0, current_frame_prob=1.0, current_frame_ops=["occlude"],
                             current_occlusion_boxes=[1, 1], current_occlusion_scale=[0.01, 0.01],
                             current_occlusion_band=[0.8, 1.0])
    clip = torch.zeros(2, 3, 100, 100)
    for s in range(20):
        random.seed(s)
        torch.manual_seed(s)
        changed = (PhotometricAugmenter(cfg)(clip.clone())[-1] != 0).any(dim=0).nonzero()
        if len(changed):
            assert changed[:, 0].float().mean() >= 70  # box centre in the lower band


def test_degradation_off_draws_no_random_numbers():
    """current_frame_prob = 0 must reproduce the v0.3 augmentation exactly."""
    cfg = AugmentationConfig()
    random.seed(3)
    torch.manual_seed(3)
    a = PhotometricAugmenter(cfg)(_clip())
    state = (random.getstate(), torch.get_rng_state())
    random.seed(3)
    torch.manual_seed(3)
    b = PhotometricAugmenter(cfg)._photometric(_clip())
    assert torch.equal(a, b)
    assert state[0] == random.getstate() and torch.equal(state[1], torch.get_rng_state())


def test_new_config_options_are_validated():
    base = PROJECT_ROOT / "configs" / "elas_pilot.yaml"
    for bad in ({"data.augmentation.current_frame_ops": ["melt"]}, {"data.augmentation.current_frame_prob": 1.5},
                {"train.freeze_backbone_stages": 6}, {"train.lr_backbone_mult": 0.0}, {"model.ufld_head_hidden": 4},
                {"model.init_scope": "head"}, {"evaluation.kalman_grid": {"speed": [1.0]}}):
        with pytest.raises(ConfigError):
            load_config(base, bad)
    load_config(base, {"hpo.search_space": {"lr_backbone_mult": {"low": 0.05, "high": 1.0, "log": True}}})


# ------------------------------------------------------------- trainer controls


def _trainer(cfg, variant: str):
    from tac_ufld.data.targets import make_row_anchors
    from tac_ufld.evaluation.evaluator import Evaluator
    from tac_ufld.training.trainer import Trainer

    t = cfg.train
    hp = Hyperparams(t.lr, t.lr_fusion, t.weight_decay, t.lambda_temporal, t.lambda_coord, t.lr_backbone_mult)
    d = cfg.data
    model = build_model(variant, cfg)
    ev = Evaluator(cfg, make_row_anchors(d.img_h, d.num_row_anchors, d.row_anchor_range))
    return Trainer(model, resolve_spec(variant, cfg), cfg, hp, "cpu", [], [], [], ev, 1, 0), model


def test_backbone_learning_rate_group():
    trainer, model = _trainer(_cfg(), "ufld_v06")
    assert len(trainer._optimizer().param_groups) == 2               # default: unchanged grouping
    trainer, model = _trainer(_cfg(**{"train.lr_backbone_mult": 0.1}), "ufld_v06")
    groups = trainer._optimizer().param_groups
    assert len(groups) == 3
    backbone_ids = {id(p) for p in backbone_module(model).parameters()}
    assert {id(p) for p in groups[1]["params"]} == backbone_ids
    assert groups[1]["lr"] == pytest.approx(groups[0]["lr"] * 0.1)


@pytest.mark.parametrize("variant,stages", [("ufld_baseline", 2), ("lite_baseline", 2)])
def test_frozen_stages_keep_weights_and_batchnorm_statistics(variant, stages):
    cfg = _cfg(**{"train.freeze_backbone_stages": stages})
    trainer, model = _trainer(cfg, variant)
    frozen = [p for s in trainer.frozen for p in s.parameters()]
    assert frozen and not any(p.requires_grad for p in frozen)
    stats = {k: v.clone() for k, v in model.state_dict().items() if "running" in k}
    model.train()
    for s in trainer.frozen:
        s.eval()
    model(torch.randn(2, 3, cfg.data.img_h, cfg.data.img_w))
    frozen_names = {n for n, m in model.named_modules() for s in trainer.frozen if m is s}
    changed = [k for k, v in model.state_dict().items() if "running" in k and not torch.equal(stats[k], v)]
    assert changed and not [k for k in changed if any(k.startswith(f + ".") for f in frozen_names)]


def test_init_from_official_ufld_checkpoint(tmp_path):
    cfg = _cfg()
    donor = build_model("ufld_baseline", cfg).ufld          # an UFLDNet = official parsingNet layout
    state = {f"module.{k}": v + 1.0 for k, v in donor.state_dict().items() if v.is_floating_point()}
    state["module.cls.2.weight"] = torch.zeros(7, 2048)      # other head size: must be skipped
    path = tmp_path / "culane_18.pth"
    torch.save({"model": state, "optimizer": None}, path)
    model = build_model("ufld_baseline", cfg)
    info = init_from_checkpoint(model, str(path), "all_matching")
    assert "ufld.cls.2.weight" in info["skipped"] and "ufld.model.conv1.weight" in info["loaded"]
    assert torch.equal(model.ufld.model.conv1.weight, donor.model.conv1.weight + 1.0)
    backbone_only = build_model("ufld_baseline", cfg)
    before = backbone_only.ufld.pool.weight.clone()
    info = init_from_checkpoint(backbone_only, str(path), "backbone")
    assert all(k.startswith("ufld.model.") for k in info["loaded"])
    assert torch.equal(backbone_only.ufld.pool.weight, before)


# --------------------------------------------------------------- new variants


def test_new_variants_are_registered_and_paired():
    cfg = _cfg()
    order = training_order(["ufld_v07", "ufld_baseline_ct", "lite_v06"], cfg)
    assert order.index("ufld_baseline") < order.index("ufld_baseline_ct")
    assert order.index("lite_baseline") < order.index("lite_v06")
    for key in ("ufld_v02", "ufld_v03", "ufld_v04", "ufld_v06", "ufld_v07", "lite_v05", "lite_v06"):
        spec = VARIANTS[key]
        assert spec.temporal and spec.reference.endswith("_baseline")
        assert spec.budget_reference == f"{spec.family}_baseline_ct"
    ct = build_model("ufld_baseline_ct", cfg)
    assert warm_start(ct, build_model("ufld_baseline", cfg).state_dict()) == ["<all>"]


@pytest.mark.parametrize("variant", ["ufld_v07", "lite_v06"])
def test_recurrent_models_start_as_their_baseline(variant):
    cfg = _cfg()
    spec = VARIANTS[variant]
    base = build_model(spec.reference, cfg).eval()
    for p in base.parameters():  # non-trivial head so the comparison means something
        if p.dim() > 1 and not p.any():
            torch.nn.init.normal_(p, 0.0, 0.05)
    model = build_model(variant, cfg).eval()
    warm_start(model, base.state_dict())
    x = torch.randn(2, cfg.data.num_frames, 3, cfg.data.img_h, cfg.data.img_w)
    with torch.no_grad():
        assert torch.allclose(model(x)["logits"], base(x[:, -1])["logits"], atol=1e-5)


@pytest.mark.parametrize("variant", ["ufld_v07", "lite_v06"])
def test_carry_mode_equals_window_mode_until_the_window_is_full(tmp_path, variant):
    cfg = _cfg()
    loaded = load_model(_checkpoint(tmp_path, variant, cfg))
    window, carry = StreamingLaneDetector(loaded, mode="cached"), StreamingLaneDetector(loaded, mode="carry")
    step, t_frames = cfg.data.temporal_step, cfg.data.num_frames
    exact_until = (t_frames - 1) * step + step - 1        # chains no longer than the training window
    differs = False
    for t, frame in enumerate(_frames(14)):
        a, b = window.infer_frame(frame, frame_index=t), carry.infer_frame(frame, frame_index=t)
        if t <= exact_until and t >= (t_frames - 1) * step:
            np.testing.assert_allclose(a.exist, b.exist, atol=1e-5)
            np.testing.assert_allclose(a.x_model, b.x_model, atol=1e-3)
        elif t > exact_until:
            differs |= not (np.allclose(a.exist, b.exist, atol=1e-7) and np.allclose(a.x_model, b.x_model, atol=1e-5))
        assert carry.cache_size() <= step
    assert differs  # beyond the window the carried state holds older frames


def test_carry_mode_rejects_non_recurrent_models(tmp_path):
    cfg = _cfg()
    with pytest.raises(ValueError, match="recurrent"):
        StreamingLaneDetector(load_model(_checkpoint(tmp_path, "lite_v05", cfg)), mode="carry")


# -------------------------------------------------------------- Kalman tracker


def test_kalman_reduces_noise_and_is_causal():
    rng = np.random.default_rng(0)
    truth = 100.0 + 0.5 * np.arange(60)
    meas = truth + rng.normal(0, 4.0, 60)
    tr = LaneKalmanTracker(KalmanParams(q=0.05, r=16.0, alpha=0.0))
    out = np.array([tr.update(np.ones((1, 1)), np.full((1, 1), m), i)[1][0, 0] for i, m in enumerate(meas)])
    assert np.abs(out[20:] - truth[20:]).mean() < 0.7 * np.abs(meas[20:] - truth[20:]).mean()
    tr2 = LaneKalmanTracker(KalmanParams(q=0.05, r=16.0, alpha=0.0))
    prefix = [tr2.update(np.ones((1, 1)), np.full((1, 1), m), i)[1][0, 0] for i, m in enumerate(meas[:30])]
    np.testing.assert_allclose(prefix, out[:30])  # future frames never change past outputs


def test_kalman_reinitialises_on_jumps_and_gaps_and_bridges_misses():
    p = KalmanParams(q=0.1, r=16.0, alpha=0.5, gate_px=30.0, max_gap=5)
    tr = LaneKalmanTracker(p)
    for i in range(10):
        tr.update(np.ones((1, 1)), np.full((1, 1), 100.0), i)
    _, x = tr.update(np.ones((1, 1)), np.full((1, 1), 200.0), 10)        # jump > gate: follow at once
    assert x[0, 0] == pytest.approx(200.0)
    e, x = tr.update(np.zeros((1, 1)), np.full((1, 1), 0.0), 11)          # one missed detection
    assert e[0, 0] == pytest.approx(0.5) and x[0, 0] == pytest.approx(200.0, abs=1.0)
    _, x = tr.update(np.ones((1, 1)), np.full((1, 1), 50.0), 30)          # long gap: reset
    assert x[0, 0] == pytest.approx(50.0) and tr.last_index == 30


def test_track_predictions_runs_per_sequence_in_frame_order():
    from tac_ufld.data.types import FrameRecord

    recs = [FrameRecord(dataset="elas", sequence=s, frame_id=f, image_path="", image_size=(640, 480),
                        lanes=(None, None), slot_known=(True, True))
            for s, f in [("B", 2), ("A", 1), ("B", 1), ("A", 0)]]
    assert sequence_order(recs) == [[3, 1], [2, 0]]
    exist = np.ones((4, 2, 2), dtype=np.float32)
    x = np.array([10.0, 50.0, 12.0, 48.0], dtype=np.float32)[:, None, None] * np.ones((4, 2, 2), np.float32)
    preds = Predictions(exist, x, np.full(4, np.nan, np.float32))
    out = track_predictions(preds, recs, KalmanParams(alpha=0.0))
    np.testing.assert_allclose(out.x[[3, 2]], x[[3, 2]])  # first frame of each sequence passes through
    assert abs(out.x[1, 0, 0] - 50.0) < abs(50.0 - 48.0) + 2.0 and out.exist.shape == exist.shape


# ------------------------------------------------------------------ ablations


def test_history_ablation_grid_shares_split_and_reuses_the_baseline(tmp_path):
    from tac_ufld.ablation import arm_configs, load_spec, reuse_single_frame

    spec = load_spec(PROJECT_ROOT / "configs" / "ablations" / "history.yaml")
    assert {"s2_t3", "s15_t5", "s1_t2"} <= set(spec.arms)
    configs = arm_configs(spec, str(tmp_path))
    splits = {str(c.data.split) for c in configs.values()}
    assert len(splits) == 1
    assert all(c.data.split.purge_frames >= c.data.temporal_step * (c.data.num_frames - 1) for c in configs.values())
    ref = configs[spec.reference_arm]
    ck = ref.output_root() / "seed_1" / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "lite_baseline.pt").write_bytes(b"x")
    (ref.output_root() / "seed_1" / "history_lite_baseline.csv").write_text("epoch\n1\n", encoding="utf-8")
    assert reuse_single_frame(configs, spec.reference_arm, "s10_t5") == ["lite_baseline seed 1"]
    assert (configs["s10_t5"].output_root() / "seed_1" / "checkpoints" / "lite_baseline.pt").exists()


def test_ablation_allow_list_is_explicit(tmp_path):
    from tac_ufld.ablation import arm_configs, load_spec

    spec = tmp_path / "a.yaml"
    spec.write_text("name: x\nbase_config: configs/elas_pilot.yaml\nreference_arm: a\n"
                    "arms:\n  a: {}\n  b: {data.temporal_step: 5}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="incomparable"):
        load_spec(spec)
    spec.write_text("name: x\nbase_config: configs/elas_pilot.yaml\nreference_arm: a\nallow: [data.scenes]\n"
                    "arms:\n  a: {}\n  b: {data.scenes: [BR_S01, ROD_S03]}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="held-out test scene"):
        arm_configs(load_spec(spec), str(tmp_path))
    aug = load_spec(PROJECT_ROOT / "configs" / "ablations" / "augmentation.yaml")
    assert {"backbone_lr_0p1", "freeze_stem_layer1", "current_frame_degradation", "small_head", "more_scenes"} <= set(aug.arms)
    arm_configs(aug, str(tmp_path))


def test_each_run_log_only_receives_its_own_run(tmp_path):
    import logging

    from tac_ufld.utils import setup_logging

    first, second = tmp_path / "a" / "run.log", tmp_path / "b" / "run.log"
    setup_logging(first).info("first run")
    setup_logging(second).info("second run")
    for h in logging.getLogger().handlers:
        h.flush()
    assert "second run" not in first.read_text(encoding="utf-8")
    assert "second run" in second.read_text(encoding="utf-8")
    setup_logging(tmp_path / "c" / "run.log")  # release the file handle of the test's last log
