"""CULane / TuSimple / OpenLane adapters on synthetic copies of the real layouts,
validation carving, dataset-native metrics and the real-data validator."""

from __future__ import annotations

import numpy as np
import pytest

from tac_ufld.config import PROJECT_ROOT, SplitConfig, load_config
from tac_ufld.data.culane import CULaneAdapter, assign_slots
from tac_ufld.data.openlane import OpenLaneAdapter
from tac_ufld.data.splits import DataSplits, carve_validation, check_no_leakage, official_split_report, split_group
from tac_ufld.data.tusimple import TuSimpleAdapter, official_slots
from tac_ufld.evaluation.native import iou_f1_all_lanes, native_metrics, sample_at_rows, tusimple_bench
from tests.dataset_fixtures import H_SAMPLES, make_culane, make_openlane, make_tusimple


@pytest.fixture(scope="session")
def culane_root(tmp_path_factory):
    return make_culane(tmp_path_factory.mktemp("culane"))


@pytest.fixture(scope="session")
def tusimple_root(tmp_path_factory):
    return make_tusimple(tmp_path_factory.mktemp("tusimple"))


@pytest.fixture(scope="session")
def openlane_root(tmp_path_factory):
    return make_openlane(tmp_path_factory.mktemp("openlane"))


def _mean_x(lane):
    return float(lane[:, 0].mean())


# ------------------------------------------------------------------ CULane


def test_culane_official_lists_and_categories(culane_root):
    splits = CULaneAdapter(culane_root).official_splits()
    assert {k: len(v) for k, v in splits.items()} == {"train": 9, "val": 3, "test": 3}
    test = splits["test"]
    assert {r.meta["category"] for r in test} == {"normal", "night"}
    assert all(r.image_size == (1640, 590) for r in test)
    assert test[0].sequence == "driver_37_30frame/05181432_0203.MP4" and test[0].frame_id == 0


def test_culane_slots_follow_seg_label_not_geometry(culane_root):
    adapter = CULaneAdapter(culane_root)
    rec = next(r for r in adapter.official_splits()["train"]
               if r.sequence.endswith("05151649_0422.MP4") and r.frame_id == 90)
    # lane-change frame: the ego-right lane (label 3) starts left of the image centre
    assert rec.lanes[1] is not None and rec.lanes[2] is not None and rec.lanes[0] is None
    assert _mean_x(rec.lanes[1]) < _mean_x(rec.lanes[2])
    geometric = assign_slots(rec.meta["eval_lanes"], 1640)
    assert geometric[2] is None, "the geometric fallback would have put both lanes on the left"
    assert adapter.stats["train"]["flag_mismatch"] == 0
    assert adapter.stats["train"]["slots_from_seg_label"] == 9


def test_culane_test_split_uses_geometric_fallback_and_keeps_all_lanes(culane_root):
    adapter = CULaneAdapter(culane_root)
    rec = adapter.official_splits()["test"][0]
    assert adapter.stats["test"]["slots_geometric"] == 3
    assert len(rec.meta["eval_lanes"]) == 4 and all(lane is not None for lane in rec.lanes)
    xs = [_mean_x(lane) for lane in rec.lanes]
    assert xs == sorted(xs)


def test_culane_frame_stride_and_temporal_step(culane_root):
    adapter = CULaneAdapter(culane_root)
    assert set(adapter.stride_report().values()) == {30, 90}
    rec = next(r for r in adapter.official_splits()["train"] if "90frame" in r.sequence and r.frame_id == 180)
    assert adapter.frame_path(rec.sequence, 180 - 90) is not None   # step 90 exists in 90-frame drivers
    assert adapter.frame_path(rec.sequence, 180 - 30) is None       # step 30 does not (old config value)


def test_culane_config_uses_supervisor_temporal_step():
    cfg = load_config(PROJECT_ROOT / "configs" / "culane.yaml")
    assert cfg.data.temporal_step == 90 and cfg.requires_confirmation


# ---------------------------------------------------------------- TuSimple


def test_tusimple_official_slot_rule():
    def lane(bx):
        ys = np.asarray(H_SAMPLES[8:], np.float32)
        t = (ys - 200) / 519
        return np.stack([640 + (bx - 640) * t, ys], axis=1)

    far_left, ego_left, ego_right, far_right, extra = lane(-150), lane(350), lane(900), lane(1450), lane(-700)
    short = np.array([[600, 680], [602, 690], [604, 700], [606, 710]], np.float32)
    slots, unplaced = official_slots([ego_right, far_left, short, ego_left, far_right, extra])
    assert slots[1] is ego_left and slots[0] is far_left and slots[2] is ego_right and slots[3] is far_right
    assert unplaced == 2  # the short lane and the third left lane


def test_tusimple_parsing_and_temporal_context(tusimple_root):
    adapter = TuSimpleAdapter(tusimple_root)
    splits = adapter.official_splits()
    assert len(splits["train"]) == 12 and len(splits["test"]) == 12 and splits["val"] == []
    rec = splits["train"][0]
    assert rec.frame_id == 20 and rec.sequence.startswith("train/")
    assert rec.valid_y_range == (160.0, 710.0)
    assert rec.meta["drive"] in {"0313-1", "0531"} and "split_order" in rec.meta
    assert adapter.frame_path(rec.sequence, 18) is not None and adapter.frame_path(rec.sequence, 16) is not None
    assert adapter.frame_path(rec.sequence, 12) is None
    xs = [_mean_x(lane) for lane in rec.lanes if lane is not None]
    assert xs == sorted(xs)
    assert adapter.stats["train"]["lanes_not_in_slots"] == 4  # 2 drives x (extra left + short lane)


def test_tusimple_validation_is_carved_from_train_without_leakage(tusimple_root):
    splits = TuSimpleAdapter(tusimple_root).official_splits()
    cfg = SplitConfig(split_seed=1, block_size=2, purge_frames=1, val_fraction=0.34, val_strategy="blocks")
    train, val = carve_validation(splits["train"], cfg, min_gap=1)
    assert train and val and not ({r.key for r in train} & {r.key for r in val})
    check_no_leakage(DataSplits(train, val, splits["test"]), min_gap=1)
    for drive in {r.meta["drive"] for r in val}:
        v = {r.meta["split_order"] for r in val if r.meta["drive"] == drive}
        t = {r.meta["split_order"] for r in train if r.meta["drive"] == drive}
        assert min(abs(a - b) for a in v for b in t) > 1  # purge gap in clip order
    report = official_split_report(DataSplits(train, val, splits["test"]))
    assert report["shared_groups_train_test"]["count"] >= 1  # official test shares drive 0531


def test_tusimple_native_bench_matches_official_rules():
    gt = [[-2, 100, 110, 120, 130], [-2, 400, 400, 400, 400]]
    ys = [100, 110, 120, 130, 140]
    assert tusimple_bench([list(map(float, g)) for g in gt], gt, ys) == (1.0, 0.0, 0.0)
    shifted = [[-2, 130, 140, 150, 160], [-2, 400, 400, 400, 400]]  # 30 px off > 20 / cos(angle)
    acc, fp, fn = tusimple_bench(shifted, gt, ys)
    assert fn == 0.5 and fp == 0.5 and acc == pytest.approx((0.2 + 1.0) / 2)
    assert tusimple_bench([[1.0] * 5] * 5, gt, ys) == (0.0, 0.0, 1.0)  # > len(gt) + 2 predictions


def test_tusimple_native_metrics_perfect_prediction(tusimple_root):
    recs = TuSimpleAdapter(tusimple_root).official_splits()["test"]
    pred = [list(r.meta["eval_lanes"]) for r in recs]
    m = native_metrics(recs, pred)
    assert m["native_tusimple_accuracy"] == pytest.approx(1.0)
    assert m["native_tusimple_fp"] == 0 and m["native_tusimple_fn"] == 0 and m["native_tusimple_f1"] == 1.0
    assert sample_at_rows(pred[0][0], [100.0, 700.0])[0] == -2


# ---------------------------------------------------------------- OpenLane


def test_openlane_attribute_slots_visibility_and_metadata(openlane_root):
    adapter = OpenLaneAdapter(openlane_root)
    splits = adapter.official_splits()
    assert len(splits["train"]) == 20 and len(splits["test"]) == 5
    rec = splits["train"][2]
    assert rec.frame_id == 2 and rec.sequence == "training/segment-100"
    assert all(lane is not None for lane in rec.lanes)
    assert len(rec.meta["eval_lanes"]) == 6  # 4 slots + attribute-0 lane + duplicate
    xs = [_mean_x(lane) for lane in rec.lanes]
    assert xs == sorted(xs)
    assert rec.lanes[1][:, 1].max() < 1160  # invisible points of the ego-left lane removed
    assert rec.meta["camera"]["intrinsic"][0][0] == 2000 and rec.meta["timestamp"] > 0
    assert adapter.stats["train"]["duplicate_attribute"] == 4
    assert adapter.frame_path(rec.sequence, rec.frame_id - 1) == splits["train"][1].image_path
    tagged = [r for r in splits["test"] if r.tags]
    assert len(tagged) == 2 and tagged[0].tags == {"night": True}


def test_openlane_validation_holds_out_whole_segments(openlane_root):
    splits = OpenLaneAdapter(openlane_root).official_splits()
    cfg = SplitConfig(split_seed=3, val_fraction=0.25, val_strategy="sequences")
    train, val = carve_validation(splits["train"], cfg, min_gap=2)
    assert {split_group(r) for r in train}.isdisjoint({split_group(r) for r in val})
    assert len({split_group(r) for r in val}) == 1
    assert carve_validation(splits["train"], cfg, 2)[1] == val  # deterministic


def test_openlane_native_metric_counts_all_lanes(openlane_root):
    recs = OpenLaneAdapter(openlane_root).official_splits()["test"]
    slot_only = [list(r.lanes) for r in recs]
    m = iou_f1_all_lanes(recs, slot_only, line_width=30, prefix="native_openlane")
    total_gt = sum(len(r.meta["eval_lanes"]) for r in recs)
    assert m["native_openlane_precision_iou50"] == 1.0
    # attribute-0 lanes (and duplicates) are annotated but cannot be predicted by a 4-slot model
    assert m["native_openlane_recall_iou50"] == pytest.approx(4 * len(recs) / total_gt)
    assert total_gt > 4 * len(recs)


# ------------------------------------------------------------ validator


@pytest.mark.parametrize("name,maker", [("culane", make_culane), ("tusimple", make_tusimple),
                                        ("openlane", make_openlane)])
def test_validate_dataset_passes_on_fixtures(name, maker, tmp_path):
    from tac_ufld.data.validation import validate_dataset

    root = maker(tmp_path / name)
    cfg = load_config(PROJECT_ROOT / "configs" / f"{name}.yaml", {"data.root": str(root)})
    report = validate_dataset(cfg, tmp_path / "out", n_overlays=2)
    assert report["ok"], report["problems"]
    assert len(list((tmp_path / "out" / "overlays").glob("*.png"))) == 2
    if name == "culane":
        assert report["temporal_step_multiple_of_all_strides"]
