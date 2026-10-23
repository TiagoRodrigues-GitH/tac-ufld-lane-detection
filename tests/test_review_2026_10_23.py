"""Review 2026-10-23: Optuna's own split (hpo_split)."""

from __future__ import annotations

import numpy as np
import pytest

from tac_ufld.data.splits import hpo_split, split_group
from tac_ufld.data.types import FrameRecord


def _records(n_groups: int, per_group: int) -> list[FrameRecord]:
    lane = np.array([[1.0, 1.0], [2.0, 8.0]], dtype=np.float32)
    return [FrameRecord("t", f"G{g:02d}", i, None, (10, 10), (lane, None), (True, True))
            for g in range(n_groups) for i in range(per_group)]


def test_hpo_split_whole_groups_disjoint_and_capped():
    train = _records(20, 50)
    a_train, a_val = hpo_split(train, 0.1, 300, 60, "2026:hpo")
    assert len(a_train) == 300 and len(a_val) == 60
    assert not {split_group(r) for r in a_train} & {split_group(r) for r in a_val}   # whole groups
    assert {r.key for r in a_val} <= {r.key for r in train}                          # only from train
    b_train, b_val = hpo_split(train, 0.1, 300, 60, "2026:hpo")                       # deterministic
    assert [r.key for r in a_val] == [r.key for r in b_val]
    assert [r.key for r in a_train] == [r.key for r in b_train]


def test_hpo_split_keeps_a_training_group_and_needs_two():
    train, val = hpo_split(_records(2, 10), 0.9, None, None, "k")
    assert len({split_group(r) for r in train}) == 1 and len({split_group(r) for r in val}) == 1
    with pytest.raises(ValueError):
        hpo_split(_records(1, 10), 0.1, None, None, "k")


def test_spline_lane_mask_follows_a_curve_better_than_straight_segments():
    from tac_ufld.metrics import lane_mask, spline_points

    ys = np.array([300.0, 400.0, 500.0, 590.0])
    xs = 800 + 0.004 * (ys - 300) ** 2                      # a curved lane, sparse points
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    dense = spline_points(pts)
    assert len(dense) > len(pts) and np.allclose(dense[0], pts[0], atol=1e-3) and np.allclose(dense[-1], pts[-1], atol=1e-3)
    y_mid = 450.0                                           # true x on the curve vs both renderings
    x_true = 800 + 0.004 * (y_mid - 300) ** 2
    lin, spl = lane_mask(pts, 1640, 590, 3, "linear"), lane_mask(pts, 1640, 590, 3, "spline")
    x_lin, x_spl = np.flatnonzero(lin[int(y_mid)]).mean(), np.flatnonzero(spl[int(y_mid)]).mean()
    assert abs(x_spl - x_true) < abs(x_lin - x_true)
    assert lane_mask(pts[:1], 1640, 590, 30, "spline").sum() == 0            # one point: nothing drawn
    assert spline_points(np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)).shape == (1, 2)


def test_config_new_fields_validate():
    from tac_ufld.config import ConfigError, load_config, PROJECT_ROOT

    cfg = load_config(PROJECT_ROOT / "configs" / "culane.yaml")
    assert cfg.train.finetune_epochs == 15 and cfg.hpo.inherit["lite_tsm_static"] == "lite_tsm"
    assert cfg.evaluation.lane_interpolation == "auto" and cfg.evaluation.robustness_max_frames == 5000
    for override in ({"train.finetune_epochs": 0}, {"evaluation.lane_interpolation": "cubic"},
                     {"hpo.val_fraction": 1.0}):
        with pytest.raises(ConfigError):
            load_config(PROJECT_ROOT / "configs" / "culane.yaml", override)


def test_multistep_milestones_scale_with_the_run_budget():
    import torch

    from tac_ufld.config import load_config, PROJECT_ROOT
    from tac_ufld.training.trainer import Trainer

    cfg = load_config(PROJECT_ROOT / "configs" / "culane.yaml")   # multi-step at 25 and 38 of 50 epochs
    cfg.train.warmup_iters = 0                                     # isolate the decay from the warm-up
    model = torch.nn.Linear(2, 2)

    class _Stub:  # only what Trainer._scheduler reads
        pass
    stub = _Stub()
    stub.cfg, stub.train_loader = cfg, [0] * 10                    # 10 iterations per epoch
    for epochs, decay_epochs in ((50, (25, 38)), (10, (5, 7.6))):
        stub.epochs = epochs
        opt = torch.optim.SGD(model.parameters(), lr=1.0)
        sched = Trainer._scheduler(stub, opt)
        lrs = []
        for _ in range(epochs * 10):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        first_decay = next(i for i, v in enumerate(lrs) if v < 0.5) / 10
        assert abs(first_decay - decay_epochs[0]) <= 0.1


def test_consecutive_run_uses_the_stored_stride():
    from tac_ufld.visualization.overlays import longest_consecutive_run

    lane = np.array([[1.0, 1.0], [2.0, 8.0]], dtype=np.float32)
    culane = [FrameRecord("culane", "driver_23_30frame/clip.MP4", f, None, (10, 10), (lane,), (True,))
              for f in range(0, 30 * 10, 30)]                       # every 30th video frame, as CULane stores
    assert longest_consecutive_run(culane) == list(range(10))
    gap = culane[:4] + culane[5:]                                      # a missing stored frame breaks the run
    assert longest_consecutive_run(gap, min_length=2) == list(range(4, 9))


def test_model_grid_images_and_video(tmp_path):
    import cv2
    import pandas as pd

    from tac_ufld.visualization.overlays import save_grid_video, save_model_grids

    lane = np.array([[300.0, 589.0], [700.0, 300.0]], dtype=np.float32)
    records = []
    for k, fid in enumerate(range(0, 300, 30)):                       # 10 stored frames, CULane stride
        path = tmp_path / "img" / f"{fid:05d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.full((590, 1640, 3), 40 + k, dtype=np.uint8))
        records.append(FrameRecord("culane", "driver_23_30frame/c.MP4", fid, str(path), (1640, 590), (lane,),
                                   (True,), tags={"normal" if k % 2 else "night": True}))
    per_frame = pd.DataFrame({"tp_iou50": [1] * 10, "fp_iou50": [0] * 10, "fn_iou50": [0] * 10})
    worse = per_frame.assign(tp_iou50=[0] * 10, fn_iou50=[1] * 10)
    lanes = {"UFLD v1": [[lane]] * 10, "lite TSM": [[None]] * 10}
    frames = {"UFLD v1": per_frame, "lite TSM": worse}
    assert save_model_grids(records, lanes, frames, "iou50", tmp_path / "grids", per_group=2) == 4
    grid = cv2.imread(str(next((tmp_path / "grids" / "night").glob("*.jpg"))))
    assert grid.shape[1] == 3 * 480                                     # GT + 2 models, 3 columns
    video = save_grid_video(records, lanes, frames, "iou50", tmp_path / "v" / "model_comparison.mp4", fps=2)
    assert video is not None and video.exists() and video.stat().st_size > 1000
    cap = cv2.VideoCapture(str(video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    cap.release()


def test_culane_records_tagged_by_stored_stride(tmp_path):
    import sys
    from pathlib import Path

    from tac_ufld.config import PROJECT_ROOT, load_config
    from tac_ufld.data import build_adapter
    from tac_ufld.data.culane import stored_stride_tag

    assert stored_stride_tag("/driver_23_30frame/a.MP4/00000.jpg") == "stride_1s"
    assert stored_stride_tag("driver_193_90frame/a.MP4/00090.jpg") == "stride_3s"
    assert stored_stride_tag("other/a.jpg") is None
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset_fixtures import make_culane

    root = make_culane(tmp_path / "culane")
    cfg = load_config(PROJECT_ROOT / "configs" / "culane.yaml", {"data.root": str(root)})
    splits = build_adapter(cfg).official_splits()
    train_tags = {k for r in splits["train"] for k in (r.tags or {})}
    assert train_tags == {"stride_1s", "stride_3s"}
    test = splits["test"]
    assert all(r.tags.get("stride_1s") for r in test)                   # driver_37_30frame
    assert {k for r in test for k in r.tags} >= {"normal", "night"}     # categories kept
