"""OpenLane-V: RVLD's evaluation protocol and the adapter, on a synthetic copy of the on-disk layout."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from tac_ufld.data.openlane_v import BAND_SIZE, CROP_TOP, OpenLaneVAdapter, lane_slots
from tac_ufld.evaluation import rvld_protocol
from tac_ufld.evaluation.native import native_metrics
from tests.dataset_fixtures import _write_jpg


def straight_lane(bottom_x: float, top_x: float = 960.0, top_y: float = 717.0, n: int = 270) -> np.ndarray:
    ys = np.linspace(top_y, 1279.0, n, dtype=np.float32)
    xs = top_x + (bottom_x - top_x) * (ys - top_y) / (1279.0 - top_y)
    return np.stack([xs, ys], axis=1)


GT = [straight_lane(200), straight_lane(700), straight_lane(1250), straight_lane(1750)]


def counts_metrics(pred, gt):
    return rvld_protocol.finish(rvld_protocol.frame_counts(pred, gt))


# ------------------------------------------------------------------ protocol


def test_perfect_prediction_scores_one():
    m = counts_metrics(GT, GT)
    assert m["native_openlanev_f1_iou50"] == pytest.approx(1.0)
    assert m["native_openlanev_f1_iou80"] == pytest.approx(1.0)
    assert m["native_openlanev_miou"] == pytest.approx(1.0)


def test_lane_shifted_by_more_than_its_width_is_wrong():
    shifted = [lane + np.array([120.0, 0.0], dtype=np.float32) for lane in GT]  # 60 px at 960x640 > 30 px
    m = counts_metrics(shifted, GT)
    assert m["native_openlanev_tp_iou50"] == 0
    assert m["native_openlanev_fp_iou50"] == 4 and m["native_openlanev_fn_iou50"] == 4


def test_small_shift_passes_05_but_not_08():
    shifted = [lane + np.array([16.0, 0.0], dtype=np.float32) for lane in GT]  # 8 px at 960x640, IoU ~ 0.58
    m = counts_metrics(shifted, GT)
    assert m["native_openlanev_f1_iou50"] == pytest.approx(1.0)
    assert m["native_openlanev_f1_iou80"] == pytest.approx(0.0)


def test_empty_cases_follow_rvld():
    none = rvld_protocol.frame_counts([], GT)        # no prediction: FN only, nothing in mIoU
    assert none[2] == 4 and none[-1] == 0
    extra = rvld_protocol.frame_counts(GT[:2], [])   # no annotation: FP, counted as IoU 0 in mIoU
    assert extra[1] == 2 and extra[-1] == 2 and extra[-2] == 0


def test_flicker_and_missing_rates():
    good, bad = GT[:2], [lane + np.array([200.0, 0.0], dtype=np.float32) for lane in GT[:2]]
    video = [(good, GT[:2]), (bad, GT[:2]), (bad, GT[:2]), (good, GT[:2])]  # ok -> wrong -> wrong -> ok
    ns, nj, nm = rvld_protocol.temporal_counts([video], 0.5)
    assert (ns, nj, nm) == (0, 4, 2)                                     # 2 lanes x (flick, miss, flick)
    m = rvld_protocol.temporal_metrics([video])
    assert m["native_openlanev_flicker_rate_iou50"] == pytest.approx(4 / 6)
    assert m["native_openlanev_missing_rate_iou50"] == pytest.approx(2 / 6)


# ------------------------------------------------------------------ adapter


def test_lane_slots_by_side_of_the_centre():
    band = [l.copy() for l in GT]
    slots, unassigned = lane_slots([band[2], band[0], band[3], band[1]])
    assert [s[-1, 0] for s in slots] == pytest.approx([200, 700, 1250, 1750])  # x at the bottom, left to right
    assert unassigned == 0
    slots, unassigned = lane_slots([band[1], band[2], straight_lane(-300)])  # three lanes, two on the left
    assert slots[3] is None and slots[0] is not None and unassigned == 0
    slots, unassigned = lane_slots([straight_lane(-900), straight_lane(-300), band[0], band[2]])
    assert unassigned == 1                                                  # a third lane on the left


def make_openlane_v(root: Path, n_frames: int = 6) -> Path:
    """Two segments (one per split) with a cached 64x38 band per frame and RVLD-style labels."""
    lists = {"training": [], "validation": []}
    for split, segment in [("training", "segment-1_with_camera_labels"), ("validation", "segment-2_with_camera_labels")]:
        for i in range(n_frames):
            ts = str(1_000_000 + 100_000 * i)
            _write_jpg(root / "OpenLane-V" / "images_64x38" / split / segment / f"{ts}.jpg", (64, 38))
            label = root / "OpenLane-V" / "label" / split / segment / f"{ts}.pickle"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_bytes(pickle.dumps({"lanes": [l.copy() for l in GT[:3]]}))
            if i % 2 == 0:  # only every other frame is annotated; the others are history frames
                lists[split].append(f"{segment}/{ts}")
    (root / "OpenLane-V" / "list").mkdir(parents=True)
    for split, names in lists.items():
        (root / "OpenLane-V" / "list" / f"datalist_{split}.pickle").write_bytes(pickle.dumps(names))
    return root


def test_adapter_records_band_coordinates_and_history(tmp_path):
    adapter = OpenLaneVAdapter(make_openlane_v(tmp_path), img_size=(64, 38))
    splits = adapter.official_splits()
    assert len(splits["train"]) == 3 and len(splits["test"]) == 3 and splits["val"] == []
    rec = splits["test"][1]
    assert rec.image_size == BAND_SIZE and rec.frame_id == 2
    assert rec.lanes[1][:, 1].min() == pytest.approx(717.0 - CROP_TOP, abs=0.01)
    assert len(rec.meta["eval_lanes"]) == 3 and rec.meta["crop_top"] == CROP_TOP
    assert adapter.frame_path(rec.sequence, rec.frame_id - 1).name == "1100000.jpg"  # unannotated history frame
    assert adapter.frame_path(rec.sequence, -1) is None


def test_native_metrics_undo_the_crop(tmp_path):
    adapter = OpenLaneVAdapter(make_openlane_v(tmp_path), img_size=(64, 38))
    test = adapter.official_splits()["test"]
    m = native_metrics(test, [list(r.lanes) for r in test])  # the slot lanes, in band coordinates
    assert m["native_openlanev_f1_iou50"] == pytest.approx(1.0)
    assert m["native_openlanev_flicker_rate_iou50"] == pytest.approx(0.0)


def test_missing_cache_is_an_error(tmp_path):
    make_openlane_v(tmp_path)
    with pytest.raises(FileNotFoundError, match="build_openlane_v_cache"):
        OpenLaneVAdapter(tmp_path, img_size=(640, 384))
