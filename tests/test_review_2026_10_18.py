"""Regression tests for the 2026-10-18 review: anchor-aligned lite head and the ego-lane / lane-change module."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from tac_ufld.data.targets import make_row_anchors
from tac_ufld.ego import LaneChangeDetector, LaneChangeParams, ego_lane_state, event_metrics
from tac_ufld.models.lite import AnchorRowPool, LanePixelHead, LightweightBackbone

H, W, A, G, L = 384, 512, 18, 100, 2


def _rows_seen(net: nn.Module, anchor: int) -> tuple[int, int]:
    x = torch.randn(1, 3, H, W, requires_grad=True)
    net(x)[0, :, anchor, :].sum().backward()
    rows = x.grad.abs().sum(dim=(0, 1, 3)).nonzero().flatten()
    return int(rows.min()), int(rows.max())


def _net(anchor_fracs):
    torch.manual_seed(0)
    head = LanePixelHead(128, L, A, G, dropout=0.0, anchor_fracs=anchor_fracs)
    nn.init.normal_(head.cls_head.weight, std=0.01)  # the zero init would give zero gradients
    return nn.Sequential(LightweightBackbone(3, 128, 0.0), head).eval()


def test_aligned_head_reads_each_anchor_row():
    anchors = make_row_anchors(H, A, (0.59, 0.995))
    net = _net(((anchors + 0.5) / H).tolist())
    for a in (0, 9, 17):
        lo, hi = _rows_seen(net, a)
        assert lo <= anchors[a] <= hi, (a, anchors[a], lo, hi)


def test_uniform_head_is_the_old_adaptive_pool():
    head = LanePixelHead(128, L, A, G, anchor_fracs=None)
    assert isinstance(head.pool, nn.AdaptiveAvgPool2d)
    # the defect it reproduces: the top anchor cannot see its own row
    anchors = make_row_anchors(H, A, (0.59, 0.995))
    lo, hi = _rows_seen(_net(None), 0)
    assert hi < anchors[0]


def test_anchor_row_pool_samples_exact_rows():
    x = torch.arange(24, dtype=torch.float32).view(1, 1, 24, 1).expand(1, 1, 24, 32).contiguous()
    pool = AnchorRowPool([(5 + 0.5) / 24, (10.5 + 0.5) / 24], griding_num=32)
    out = pool(x)
    assert out.shape == (1, 1, 2, 32)
    assert torch.allclose(out[0, 0, :, 0], torch.tensor([5.0, 10.5]))


def test_checkpoint_without_key_loads_uniform_head():
    from tac_ufld.config import ExperimentConfig
    from tac_ufld.inference.loading import config_from_checkpoint

    cfg = config_from_checkpoint({"config": {"model": {"variants": ["lite_baseline"]}}})
    assert cfg.model.lite_anchor_rows == "uniform"
    assert ExperimentConfig().model.lite_anchor_rows == "aligned"


def test_anchor_error_split_and_position_error():
    from tac_ufld.metrics import anchor_counts

    G, W_ = 100, 512
    target = np.array([[10, G], [20, G], [G, 30]])           # (A=3, L=2); G = no lane
    x_t = np.array([[10, 0], [20, 0], [0, 30]], dtype=float) * (W_ - 1) / (G - 1)
    exist = np.array([[0.9, 0.9], [0.9, 0.1], [0.1, 0.1]])   # 1 false alarm (0,1), 1 miss (2,1)
    x_p = x_t.copy()
    x_p[1, 0] += 50.0                                        # wrong position at (1,0), 50 model px off
    c = anchor_counts(exist, x_p, target, 0.5, 1, W_, G, x_target=x_t, px_scale=2.0,
                      far_rows=np.array([True, False, False]))
    assert (c.fp_exist, c.fn_exist, c.wrong, c.both) == (1, 1, 1, 2)
    m = c.as_metrics()
    assert m["anchor_false_alarm_rate"] == 1 / (1 + c.tn)
    assert m["anchor_miss_rate"] == 1 / 3 and m["anchor_wrong_position_rate"] == 0.5
    assert m["anchor_mae_far_px"] == 0.0 and m["anchor_mae_near_px"] == 100.0  # 50 model px x scale 2


def test_elas_native_metrics_are_perfect_on_ground_truth():
    from tac_ufld.data.types import FrameRecord
    from tac_ufld.evaluation.native import NATIVE_PROTOCOLS

    lanes = (_lane(80.0, 280.0), _lane(560.0, 360.0))
    rec = FrameRecord(dataset="elas", sequence="S", frame_id=0, image_path=None, image_size=(640, 480),
                      lanes=lanes, slot_known=(True, True), valid_y_range=(340.0, 480.0), meta={"laneChange": 0.0})
    p = NATIVE_PROTOCOLS["elas"]
    out = p.finish(p.frame_counts(rec, list(lanes)))
    assert out["native_elas_center_coverage"] == 1.0 and out["native_elas_center_err_pct"] == 0.0
    shifted = [lane + np.array([24.0, 0.0], dtype=np.float32) for lane in lanes]  # centre 24 px off, width 480
    assert abs(p.finish(p.frame_counts(rec, shifted))["native_elas_center_err_pct"] - 5.0) < 1e-6
    lc = p.sequence_metrics([rec], [list(lanes)])
    assert lc["native_elas_lc_gt_events"] == 0 and lc["native_elas_lc_false_alarms"] == 0


def test_persistent_degradation_repeats_the_same_corruption_on_every_frame():
    import random

    from tac_ufld.config import AugmentationConfig
    from tac_ufld.data.transforms import degrade_current_frame

    frames = torch.rand(3, 3, 48, 64)
    cfg = AugmentationConfig()
    for op in ("occlude", "noise", "shadow", "glare"):
        out = degrade_current_frame(frames, cfg, op, random.Random(7), torch.Generator().manual_seed(7), scope="clip")
        deltas = out - frames
        assert not torch.allclose(out[-1], frames[-1]), op
        same_input = degrade_current_frame(frames[:1].expand(3, -1, -1, -1).contiguous(), cfg, op, random.Random(7),
                                           torch.Generator().manual_seed(7), scope="clip")
        assert torch.allclose(same_input[0], same_input[2]), op       # identical parameters per frame
        cur = degrade_current_frame(frames, cfg, op, random.Random(7), torch.Generator().manual_seed(7))
        assert torch.equal(cur[:-1], frames[:-1]), op                  # "current" leaves the history clean
        assert deltas[0].abs().sum() > 0, op


def test_training_persistent_share_degrades_the_whole_clip():
    import random

    from tac_ufld.config import AugmentationConfig
    from tac_ufld.data.transforms import PhotometricAugmenter

    frames = torch.rand(3, 3, 32, 48)
    base = dict(enabled=True, prob=0.0, current_frame_prob=1.0, current_frame_ops=["darken"])
    random.seed(0)
    only_current = PhotometricAugmenter(AugmentationConfig(**base))(frames)
    assert torch.equal(only_current[:-1], frames[:-1]) and not torch.allclose(only_current[-1], frames[-1])
    random.seed(0)
    whole = PhotometricAugmenter(AugmentationConfig(**base, current_frame_persistent_share=1.0))(frames)
    assert all(not torch.allclose(whole[i], frames[i]) for i in range(3))
    assert torch.allclose(whole[0] / frames[0].clamp_min(1e-6), whole[2] / frames[2].clamp_min(1e-6), atol=1e-4)


def test_position_variance_matches_the_categorical_formula():
    from tac_ufld.decoding import decode_logits, position_variance

    logits = torch.randn(2, 11, 3, 2)  # G = 10 cells + no-lane
    var = position_variance(logits)
    p = torch.softmax(logits[:, :-1], dim=1)
    i = torch.arange(10, dtype=torch.float32).view(1, -1, 1, 1)
    mean = decode_logits(logits)[1]
    assert torch.allclose(var, (p * (i - mean.unsqueeze(1)) ** 2).sum(1), atol=1e-5)
    sharp = torch.full((1, 11, 1, 1), -20.0)
    sharp[0, 4] = 20.0
    assert position_variance(sharp).item() < 1e-6


def test_adaptive_kalman_trusts_uncertain_frames_less():
    """Same jump in the measurement: with a large predictive variance (an occluded frame) the adaptive filter
    moves less toward it than with a small one; the constant-R filter cannot tell the two apart."""
    from tac_ufld.evaluation.tracking import KalmanParams, LaneKalmanTracker

    def run(var):
        t = LaneKalmanTracker(KalmanParams(q=1.0, r=4.0, adaptive_r=True, gate_sigma=1e6))
        for k in range(10):
            t.update(np.ones((1, 1)), np.full((1, 1), 100.0), k, np.full((1, 1), 4.0))
        return float(t.update(np.ones((1, 1)), np.full((1, 1), 130.0), 10, np.full((1, 1), var))[1][0, 0])

    sure, unsure = run(4.0), run(400.0)
    assert 100.0 < unsure < sure < 130.0


def test_distillation_loss_and_training_order():
    from tac_ufld.config import ExperimentConfig
    from tac_ufld.losses import distillation_loss
    from tac_ufld.models.registry import training_order

    z = torch.randn(2, 11, 4, 2)
    valid = torch.ones(2, 4, 2, dtype=torch.bool)
    assert distillation_loss(z, z, valid, 2.0).abs().item() < 1e-6          # same distribution: no loss
    other = torch.randn(2, 11, 4, 2)
    assert distillation_loss(z, other, valid, 2.0).item() > 0
    assert distillation_loss(z, other, torch.zeros_like(valid), 2.0).item() == 0.0   # nothing annotated
    order = training_order(["lite_tsm_kd", "lite_baseline_kd"], ExperimentConfig())
    for teacher_dep in ("ufld_baseline", "ufld_v07", "lite_baseline"):
        assert order.index(teacher_dep) < order.index("lite_tsm_kd")
    assert order.index("ufld_v07") < order.index("lite_baseline_kd")


def test_distillation_trains_with_a_teacher(tmp_path):
    """One real optimisation step of a single-frame student with a temporal teacher on a 3-frame clip."""
    from tac_ufld.config import ExperimentConfig
    from tac_ufld.losses import Hyperparams, VariantLoss
    from tac_ufld.models.registry import VARIANTS, build_model

    cfg = ExperimentConfig()
    d = cfg.data
    student = build_model("lite_baseline_kd", cfg, pretrained=False)
    teacher = build_model("ufld_v07", cfg, pretrained=False).eval()
    images = torch.randn(1, d.num_frames, 3, d.img_h, d.img_w)
    batch = {"images": images, "cls": torch.full((1, d.num_row_anchors, d.num_lanes), d.griding_num),
             "valid": torch.ones(1, d.num_row_anchors, d.num_lanes, dtype=torch.bool),
             "exist": torch.zeros(1, d.num_row_anchors, d.num_lanes), "x": torch.zeros(1, d.num_row_anchors, d.num_lanes)}
    out = student(images)
    with torch.no_grad():
        out["teacher_logits"] = teacher(images)["logits"]
    loss_fn = VariantLoss(VARIANTS["lite_baseline_kd"], cfg.train.loss, Hyperparams(4e-4, 5e-4, 1e-4, 0.0, 0.0),
                          d.img_w, distill_weight=1.0, distill_temperature=2.0)
    total, logs = loss_fn(student, out, batch)
    assert "kd" in logs and logs["kd"] > 0 and torch.isfinite(total)
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())


def test_macs_count_attention():
    from tac_ufld.evaluation.efficiency import count_macs

    mha = nn.MultiheadAttention(16, 2, batch_first=True)

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.mha = mha

        def forward(self, x):
            return self.mha(x, x, x)[0]

    lq = 10
    expected = 4 * lq * 16 * 16 + 2 * lq * lq * 16
    assert count_macs(Wrap(), torch.randn(1, lq, 16)) == expected


def test_lane_keeping_warns_and_steers_back():
    from tac_ufld.ego import EgoLaneTracker, lane_keeping

    tracker = EgoLaneTracker(320.0, 480.0, 340.0)
    width, out, signalled = 480.0, None, None
    for c in np.linspace(320, 140, 40):  # lane centre moves left in the image = the car drifts RIGHT
        lanes = [_lane(c - width / 2, c - width / 2 + 200), _lane(c + width / 2, c + width / 2 - 200)]
        s, _ = tracker.update(lanes)
        out = lane_keeping(s, tracker.detector)
        signalled = lane_keeping(s, tracker.detector, turn_signal=1)
        if out.ldw:
            break
    assert out is not None and out.ldw == 1 and out.steer < 0 and out.tlc_s < 5
    assert signalled.ldw == 0 and signalled.steer == 0.0 and not signalled.active  # intended: no warning


def test_elas_discovers_frames_directly_in_images(tmp_path):
    """GRI_S02 is unpacked as <scene>/images/lane_*.png (no images/images): it must not be dropped."""
    from tac_ufld.data.elas import discover_scenes

    for scene, sub in (("A_S01", "images/images"), ("B_S02", "images")):
        d = tmp_path / scene
        (d / sub).mkdir(parents=True)
        (d / "config.xml").write_text("<config/>", encoding="utf-8")
        (d / "groundtruth.xml").write_text("<groundtruth/>", encoding="utf-8")
        (d / sub / "lane_0.png").write_bytes(b"")
    idx = discover_scenes(tmp_path)
    assert idx["A_S01"].images == tmp_path / "A_S01" / "images" / "images"
    assert idx["B_S02"].images == tmp_path / "B_S02" / "images"


def test_lane_change_from_an_offset_flip():
    """A lane model re-assigns the crossing line early: the offset jumps +0.3 -> -0.3 with no boundary passing
    under the camera. That is a change to the RIGHT, not a fast drift to the left."""
    from tac_ufld.ego import EgoLaneTracker

    tracker = EgoLaneTracker(320.0, 480.0, 340.0)
    width, decisions = 480.0, []
    centres = np.r_[np.full(30, 320.0), np.linspace(320, 180, 45), np.full(3, 180.0)]
    for c in centres:  # drift right until the car is 0.29 lane widths right of the centre
        decisions.append(tracker.update([_lane(c - width / 2, c - width / 2 + 200),
                                         _lane(c + width / 2, c + width / 2 - 200)])[1])
    for c in np.linspace(320 + 150, 320 + 20, 60):  # re-assigned: new lane centre right of the car, then centred
        decisions.append(tracker.update([_lane(c - width / 2, c - width / 2 + 200),
                                         _lane(c + width / 2, c + width / 2 - 200)])[1])
    assert -1 not in decisions and 1 in decisions


def _toward(x_bottom: float, vp=(320.0, 300.0), y_bottom=480.0, y_far=340.0) -> np.ndarray:
    """A straight boundary through (x_bottom, y_bottom) that vanishes at ``vp`` (perspective-consistent)."""
    t = (y_bottom - y_far) / (y_bottom - vp[1])
    return np.array([[x_bottom + (vp[0] - x_bottom) * t, y_far], [x_bottom, y_bottom]], dtype=np.float32)


def test_heading_cue_is_yaw_not_offset():
    """Car aligned with the road but off-centre: the lane still vanishes on the camera axis, heading 0.
    Car yawed left: the lane vanishes right of the axis, heading > 0, and LKA steers right."""
    from tac_ufld.ego import ego_lane_state

    off_centre = ego_lane_state([_toward(60.0), _toward(520.0)], 480.0, 340.0, 320.0)  # centre 290: car right
    assert off_centre.offset > 0 and abs(off_centre.heading_px) < 1e-3
    yawed = ego_lane_state([_toward(80.0, vp=(380, 300)), _toward(560.0, vp=(380, 300))], 480.0, 340.0, 320.0)
    assert abs(yawed.offset) < 1e-6 and yawed.heading_px == pytest.approx(60.0, abs=1e-3)


def test_ldw_uses_the_wheel_and_turn_signal_suppresses_both_sides():
    from tac_ufld.ego import LaneChangeDetector, LaneKeepingParams, ego_lane_state, lane_keeping

    det = LaneChangeDetector(320.0)
    p = LaneKeepingParams()
    # centred car parked 0.22 lane widths right of the centre, not moving: the wheel is 0.02 from the line
    s = ego_lane_state([_toward(320 - 240 - 0.22 * 480), _toward(320 + 240 - 0.22 * 480)], 480.0, 340.0, 320.0)
    for _ in range(12):
        det.update(s)
    out = lane_keeping(s, det, params=p)
    assert out.ldw == 1, "the wheel is within warn_margin of the right line: warn although the centre is 0.28 away"
    for signal in (1, -1):  # after the crossing the offset flips while the signal still shows the old side
        assert lane_keeping(s, det, turn_signal=signal).ldw == 0


def _lane(x_bottom: float, x_top: float) -> np.ndarray:
    return np.array([[x_top, 340.0], [x_bottom, 480.0]], dtype=np.float32)


def test_lane_change_to_the_right_is_detected():
    """The car drifts right, the right boundary passes under the camera, the car centres in the new lane."""
    det = LaneChangeDetector(camera_x=320.0, params=LaneChangeParams())
    width, pred, gt = 480.0, [], []
    centres = np.r_[np.full(60, 320.0), np.linspace(320, 320 - width, 120), np.full(90, 320 - width)]
    for t, c in enumerate(centres):  # lane centre in the image moves LEFT = the car moves RIGHT
        left, right = c - width / 2, c + width / 2
        lanes = [_lane(left, left + 200), _lane(right, right - 200)]
        if right < 320:  # crossed: the old right boundary is now the new ego-left boundary
            lanes = [_lane(right, right - 200), _lane(right + width, right + width - 200)]
        s = ego_lane_state(lanes, 480.0, 340.0, 320.0, width)
        pred.append(det.update(s))
        gt.append(1 if 60 <= t < 200 else 0)
    m = event_metrics(np.array(pred), np.array(gt), 29.97)
    assert m["detected"] == 1 and m["direction_correct"] == 1 and m["false_alarms"] == 0
    assert set(pred) <= {0, 1}
