"""CIPO (closest in-path object): reader, targets, head, geometry, metrics and the train/eval script.

The fixture is a tiny OpenLane tree whose lanes, 3-D lane points and boxes all come from ONE flat-ground pinhole
camera (fx = fy = 2000, cx = 960, cy = 640, 2 m high), so distances and in-path fractions have exact answers."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tac_ufld.cipo import (CameraModel, CipoRule, average_precision, average_precisions, box_iou,
                           cipo_frame_metrics, cipo_stability, distance_errors, ego_span, in_path_fraction,
                           lane_ground_distance, path_geometry, project_lane_ground, select_cipo)
from tac_ufld.config import PROJECT_ROOT, GeometricAugConfig, load_config
from tac_ufld.data.openlane_cipo import (CipoDataset, CipoSource, build_frames, cap_frames, encode_targets,
                                         parse_cipo, read_split_manifest, warp_boxes)
from tac_ufld.evaluation.efficiency import count_macs
from tac_ufld.models.cipo_head import CipoHead, FeatureTap, HeadProbe, cipo_loss, decode, focal_heatmap_loss
from tac_ufld.models.registry import build_model

CAM = CameraModel(fx=2000.0, fy=2000.0, cx=960.0, cy=640.0, height=2.0)
K = [[2000.0, 0.0, 960.0], [0.0, 2000.0, 640.0], [0.0, 0.0, 1.0]]
E = [[1.0, 0.0, 0.0, 1.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]]


def project(z: np.ndarray, y: np.ndarray, up: float = -2.0) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame point (forward z, left y, up) -> pixel (u, v)."""
    return CAM.cx - CAM.fx * y / z, CAM.cy - CAM.fy * up / z


def lane(lateral: float, attribute: int) -> dict:
    z = np.linspace(4.0, 80.0, 40)
    y = np.full_like(z, lateral)
    u, v = project(z, y)
    keep = (u >= 0) & (u < 1920) & (v < 1280)
    return {"attribute": attribute, "category": 1, "track_id": attribute,
            "uv": [u[keep].tolist(), v[keep].tolist()], "visibility": [1.0] * int(keep.sum()),
            "xyz": [z[keep].tolist(), y[keep].tolist(), [-2.0] * int(keep.sum())]}


def car(z: float, lateral: float, level: int, kind: int = 1, track: str = "t", width: float = 1.8,
        height: float = 1.5) -> dict:
    """Box of an object standing on the road at (z, lateral)."""
    u, v = project(np.array([z]), np.array([lateral]))
    w, h = width * CAM.fx / z, height * CAM.fy / z
    return {"x": float(u[0] - w / 2), "y": float(v[0] - h), "width": float(w), "height": float(h),
            "id": level, "trackid": track, "type": kind}


def frame_objects(i: int) -> list[dict]:
    """Even frames: a level-1 car 20 m ahead in the ego lane. Odd frames: that car is 70 m ahead (level 3, beyond
    the reaction distance), so there is no level 1. Always: a level-3 car in the left lane, a level-4 pedestrian."""
    objs = [car(15.0, 3.5, 3, track="left"), car(30.0, -8.0, 4, kind=2, track="ped", width=0.6, height=1.7)]
    objs.append(car(20.0, 0.0, 1, track="lead") if i % 2 == 0 else car(70.0, 0.0, 3, track="lead"))
    return objs


def make_cipo_openlane(root: Path, frames: int = 6, zip_cipo: bool = False) -> dict[str, str]:
    """images + lane3d_1000 + cipo for 3 training and 1 validation segment; returns sequence -> split."""
    segments = {"training/segment-a": "train", "training/segment-b": "train", "training/segment-c": "val",
                "validation/segment-z": "test"}
    lanes = [lane(5.25, 1), lane(1.75, 2), lane(-1.75, 3), lane(-5.25, 4)]
    cipo_files = {}
    for s_i, sequence in enumerate(segments):
        split_dir, segment = sequence.split("/")
        for f in range(frames):
            ts = 1_500_000_000_000_000 + s_i * 10_000_000 + f * 100_000
            img = np.full((1280, 1920, 3), 80, dtype=np.uint8)
            objs = frame_objects(f)
            for o in objs:
                cv2.rectangle(img, (int(o["x"]), int(o["y"])), (int(o["x"] + o["width"]), int(o["y"] + o["height"])),
                              (230, 40, 40) if o["type"] == 1 else (40, 230, 40), -1)
            path = root / "images" / split_dir / segment / f"{ts}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not (sequence == "training/segment-b" and f == 5):     # one CIPO frame without its image
                cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])[1].tofile(str(path))
            label = root / "lane3d_1000" / split_dir / segment / f"{ts}.json"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(json.dumps({"file_path": f"{split_dir}/{segment}/{ts}.jpg", "intrinsic": K,
                                         "extrinsic": E, "lane_lines": lanes}), encoding="utf-8")
            cipo_files[f"cipo/{split_dir}/{segment}/{ts}.jpg.json"] = json.dumps(
                {"raw_file_path": f"{split_dir}/{segment}/{ts}.jpg", "result": objs})
    if zip_cipo:
        with zipfile.ZipFile(root / "cipo.zip", "w") as z:
            for name, text in cipo_files.items():
                z.writestr(name, text)
            z.writestr("__MACOSX/cipo/._junk.json", "not json")
    else:
        for name, text in cipo_files.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(text, encoding="utf-8")
    return segments


def write_manifest(path: Path, sequences: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "dataset", "sequence", "frame_id", "image_path"])
        for seq, split in sequences.items():
            w.writerow([split, "openlane", seq, 0, "x.jpg"])
    return path


@pytest.fixture
def cipo_root(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "OpenLane"
    return root, make_cipo_openlane(root)


# -- reader ------------------------------------------------------------------------------------------------------------

def test_parse_cipo_accepts_every_variant_of_the_release():
    item = {"result": [{"x": 1, "y": 2, "width": 3, "height": 4, "id": 1, "trackid": "a", "type": 1},
                       {"x": 1, "y": 2, "width": 3, "height": 4, "attribute": 4, "trackid": "b", "type": 2},
                       {"x": 1, "y": 2, "width": 3, "height": 4, "id": "2", "type": 4},
                       {"x": 1, "y": 2, "width": 0, "height": 4, "id": 1, "type": 1},       # empty box
                       {"x": 1, "y": 2, "width": 3, "height": 4, "id": 7, "type": 1},       # unknown level
                       {"x": 1, "y": 2, "width": 3, "height": 4, "type": 1}]}               # no level
    boxes = parse_cipo(item)
    assert [(b.level, b.type, b.track_id) for b in boxes] == [(1, 1, "a"), (4, 2, "b"), (2, 4, "")]
    assert boxes[0].xyxy == (1.0, 2.0, 4.0, 6.0)
    assert parse_cipo({"results": item["result"][:1]})[0].level == 1     # the README's key
    assert parse_cipo({"raw_file_path": "x"}) == []


def test_source_reads_the_folder_and_the_zip_identically(tmp_path):
    folder_root, zip_root = tmp_path / "a", tmp_path / "b"
    make_cipo_openlane(folder_root, frames=2)
    make_cipo_openlane(zip_root, frames=2, zip_cipo=True)
    from_dir, from_zip = CipoSource(folder_root), CipoSource(zip_root / "cipo.zip")
    stamps_dir = from_dir.timestamps("training", "segment-a")
    stamps_zip = from_zip.timestamps("training", "segment-a")
    assert sorted(stamps_dir) == sorted(stamps_zip) and len(stamps_dir) == 2
    ts = min(stamps_dir)
    assert from_dir.load("training", "segment-a", ts) == from_zip.load("training", "segment-a", ts)
    assert from_zip.load("training", "segment-a", 123) is None
    assert from_zip.timestamps("training", "segment-missing") == {}
    with pytest.raises(FileNotFoundError):
        CipoSource(tmp_path / "nothing")


def test_build_frames_joins_images_lanes_camera_and_split(cipo_root, tmp_path):
    root, sequences = cipo_root
    manifest = read_split_manifest(write_manifest(tmp_path / "m.csv", sequences))
    assert manifest == sequences
    splits, stats = build_frames(root, CipoSource(root), manifest)
    assert {k: len(v) for k, v in splits.items()} == {"train": 11, "val": 6, "test": 6}
    assert stats["missing_image"] == 1
    test = splits["test"][0]
    assert test.camera == CameraModel(2000.0, 2000.0, 960.0, 640.0, 2.0)
    assert test.lanes is not None and all(test.lanes[i] is not None for i in range(4))
    assert test.lane_ground[0] is not None and test.lane_ground[1] is not None
    assert splits["train"][0].lanes is None           # training reads boxes only
    assert test.boxes_xyxy(levels=(1,)).shape == (1, 4)
    assert test.level1_track() == "lead" and splits["test"][1].level1_track() is None
    assert [f.frame_index for f in splits["test"]] == list(range(6))
    # caps are applied before reading, as contiguous blocks per sequence
    capped, _ = build_frames(root, CipoSource(root), manifest, {"train": 4, "val": None, "test": 3})
    assert len(capped["train"]) == 4 and len(capped["test"]) == 3
    idx = [f.frame_index for f in capped["test"]]
    assert idx == list(range(idx[0], idx[0] + 3))


def test_cap_frames_spreads_contiguous_blocks_over_sequences():
    class F:
        def __init__(self, s, i):
            self.sequence, self.i = s, i
    items = [F(s, i) for s in "ab" for i in range(10)]
    capped = cap_frames(items, 6)
    assert [(f.sequence, f.i) for f in capped] == [("a", 3), ("a", 4), ("a", 5), ("b", 3), ("b", 4), ("b", 5)]
    assert cap_frames(items, None) is items


def test_split_manifest_rejects_a_sequence_in_two_splits(tmp_path):
    path = write_manifest(tmp_path / "m.csv", {"training/s": "train"})
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["val", "openlane", "training/s", 1, "y.jpg"])
    with pytest.raises(ValueError, match="both"):
        read_split_manifest(path)


# -- targets, head, loss, decoding -------------------------------------------------------------------------------------

def test_targets_decode_back_to_the_boxes():
    boxes = np.array([[10.0, 20.0, 50.0, 44.0], [100.0, 30.0, 116.0, 70.0], [60.0, 60.0, 64.0, 63.0]])
    levels = np.array([1, 3, 4])
    t = encode_targets(boxes, levels, out_w=16, out_h=10, stride=8)
    assert t["mask"].sum() == 3 and t["heat"].max() == 1.0
    assert t["heat"][0].max() == 1.0 and t["heat"][1].max() == 0.0       # level 2 has no object
    # a perfect head: high logits at the peaks only, exact offsets and sizes at the centres
    heat = np.where(t["heat"] >= 1.0, 10.0, -10.0).astype(np.float32)
    offset = np.zeros((2, 10, 16), np.float32)
    log_size = np.zeros((2, 10, 16), np.float32)
    for k in range(3):
        iy, ix = divmod(int(t["ind"][k]), 16)
        offset[:, iy, ix] = t["offset"][k]
        log_size[:, iy, ix] = t["log_size"][k]
    out = {"heat": torch.from_numpy(heat)[None], "offset": torch.from_numpy(offset)[None],
           "log_size": torch.from_numpy(log_size)[None]}
    det = decode(out, stride=8, top_k=10, min_score=0.5)[0]
    assert len(det["boxes"]) == 3
    by_x = lambda b: b[np.argsort(b[:, 0])]  # noqa: E731
    np.testing.assert_allclose(by_x(det["boxes"]), by_x(boxes), atol=1e-3)
    assert sorted(det["levels"].tolist()) == [1, 3, 4]
    # the loss of the perfect head is ~0 on offset and size
    batch = {k: torch.from_numpy(v)[None] for k, v in t.items()}
    _, parts = cipo_loss(out, batch)
    assert parts["offset"] < 1e-6 and parts["size"] < 1e-6


def test_ignore_regions_and_focal_loss():
    t = encode_targets(np.zeros((0, 4)), np.zeros(0), out_w=8, out_h=8, stride=8,
                       ignore=np.array([[0.0, 0.0, 32.0, 32.0]]))
    assert t["weight"][:4, :4].sum() == 0 and t["weight"][4:, 4:].min() == 1
    target = torch.zeros(1, 4, 8, 8)
    target[0, 0, 3, 3] = 1.0
    good = torch.full((1, 4, 8, 8), -8.0)
    good[0, 0, 3, 3] = 8.0
    bad = -good
    assert focal_heatmap_loss(good, target) < 0.01 < focal_heatmap_loss(bad, target)
    # an ignored false alarm costs nothing
    alarm = good.clone()
    alarm[0, 1, 0, 0] = 8.0
    weight = torch.ones(1, 8, 8)
    weight[0, 0, 0] = 0.0
    assert focal_heatmap_loss(alarm, target, weight) == pytest.approx(float(focal_heatmap_loss(good, target, weight)))
    assert focal_heatmap_loss(alarm, target) > focal_heatmap_loss(good, target)


def _tiny_cfg(root: Path, tmp: Path):
    return load_config(PROJECT_ROOT / "configs" / "openlane_smoke.yaml", {
        "output_dir": str(tmp / "results"), "device": "cpu", "data.root": str(root),
        "data.img_h": 64, "data.img_w": 96, "data.griding_num": 10, "data.num_row_anchors": 8})


@pytest.mark.parametrize("variant,channels,strides", [("ufld_baseline", (128, 256, 512), (8, 16, 32)),
                                                      ("lite_baseline", (96, 128), (8, 16))])
def test_feature_tap_one_pass_gives_lanes_and_features(variant, channels, strides, tmp_path):
    model = build_model(variant, _tiny_cfg(tmp_path, tmp_path), pretrained=False).eval()
    tap = FeatureTap(model)
    assert tap.channels == channels and tap.strides == strides
    x = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        out, feats = tap(x)
        reference = model(x)["logits"]
    assert torch.allclose(out["logits"], reference)                      # the lane output is unchanged
    assert [tuple(f.shape) for f in feats] == [(2, c, 64 // s, 96 // s) for c, s in zip(channels, strides)]
    head = CipoHead(tap.channels)
    det = head(feats)
    assert det["heat"].shape == (2, 4, 8, 12) and det["offset"].shape == (2, 2, 8, 12)
    assert torch.sigmoid(det["heat"]).mean() < 0.2                      # focal prior: few false alarms at start
    tap.remove()
    tap._captured = ["untouched"] * len(tap._captured)
    with torch.no_grad():
        model(x)
    assert tap._captured == ["untouched"] * len(tap._captured)          # hooks removed


def test_feature_tap_refuses_temporal_models(tmp_path):
    model = build_model("ufld_v03", _tiny_cfg(tmp_path, tmp_path), pretrained=False)
    with pytest.raises(ValueError, match="single-frame"):
        FeatureTap(model)


def test_head_is_small_next_to_the_backbone():
    head = CipoHead((128, 256, 512))
    params = sum(p.numel() for p in head.parameters())
    macs = count_macs(HeadProbe(head, (128, 256, 512), (8, 16, 32)), torch.zeros(1, 3, 320, 480))
    assert params < 100_000 and macs < 0.1e9                            # docs/CIPO.md: 0.073 M / 0.07 GMACs


def test_dataset_items_and_box_warping(cipo_root, tmp_path):
    root, sequences = cipo_root
    splits, _ = build_frames(root, CipoSource(root), sequences)
    ds = CipoDataset(splits["test"], img_w=96, img_h=64, stride=8)
    item = ds[0]
    assert item["image"].shape == (3, 64, 96) and item["heat"].shape == (4, 8, 12)
    assert item["mask"].sum() == 3                                       # lead car, left car, pedestrian
    flip = np.array([[-1.0, 0, 95.0], [0, 1.0, 0], [0, 0, 1.0]])
    warped, keep = warp_boxes(np.array([[10.0, 5.0, 30.0, 25.0], [90.0, 0.0, 120.0, 10.0]]), flip, 96, 64)
    np.testing.assert_allclose(warped[0], [65.0, 5.0, 85.0, 25.0])
    assert keep.tolist() == [True, False]                                # second box: 5 of its 30 px stay inside
    aug = CipoDataset(splits["test"], 96, 64, 8, augment=GeometricAugConfig(enabled=True, prob=1.0, hflip_prob=1.0))
    assert aug[0]["mask"].sum() == 3
    with pytest.raises(ValueError):
        CipoDataset(splits["test"], img_w=100, img_h=64, stride=8)


# -- geometry ----------------------------------------------------------------------------------------------------------

def test_camera_flat_ground_and_lane_ground_reference():
    _, v = project(np.array([10.0, 20.0, 40.0]), np.zeros(3))
    np.testing.assert_allclose(CAM.ground_distance(v), [10.0, 20.0, 40.0])
    assert CAM.ground_distance(600.0) == math.inf                       # above the horizon
    assert CAM.ground_width_px(840.0, 3.5) == pytest.approx(3.5 * 2000 / 20)
    cam = CameraModel.from_openlane(K, E)
    assert cam == CAM
    ego = lane(1.75, 2)
    ground = project_lane_ground(np.asarray(ego["xyz"]), ego["visibility"], cam)
    assert lane_ground_distance(840.0, [ground]) == pytest.approx(20.0, rel=0.02)
    assert np.isnan(lane_ground_distance(1279.0 + 500, [ground]))       # outside the labelled rows
    hill = np.asarray(ego["xyz"]).copy()
    hill[2] += 0.02 * hill[0]                                            # road rising ahead
    g_hill = project_lane_ground(hill, None, cam)
    # a rising road shows a point higher in the image: the flat-ground proxy overestimates (25 m vs 20 m here)
    assert lane_ground_distance(800.0, [g_hill]) == pytest.approx(20.0, rel=0.02)
    assert float(cam.ground_distance(800.0)) == pytest.approx(25.0)


def test_in_path_geometry():
    left = np.stack(project(np.linspace(4, 80, 30), np.full(30, 1.75)), axis=1)
    right = np.stack(project(np.linspace(4, 80, 30), np.full(30, -1.75)), axis=1)
    lead = car(20.0, 0.0, 1)
    box = [lead["x"], lead["y"], lead["x"] + lead["width"], lead["y"] + lead["height"]]
    assert in_path_fraction(box, left, right, CAM) == pytest.approx(1.0)
    straddle = car(20.0, -1.75, 2)                                       # centred on the right boundary
    sbox = [straddle["x"], straddle["y"], straddle["x"] + straddle["width"], straddle["y"] + straddle["height"]]
    assert in_path_fraction(sbox, left, right, CAM) == pytest.approx(0.5, abs=0.01)
    adj = car(15.0, 3.5, 3)
    abox = [adj["x"], adj["y"], adj["x"] + adj["width"], adj["y"] + adj["height"]]
    assert in_path_fraction(abox, left, right, CAM) == 0.0
    # one boundary missing: a 3.5 m lane from the visible one gives the same span on this road
    xl, xr, src = ego_span(left, None, 840.0, CAM)
    assert src == "left+width" and (xl, xr) == pytest.approx(ego_span(left, right, 840.0, CAM)[:2], abs=0.5)
    assert ego_span(None, right, 840.0, CAM)[2] == "right+width"
    xl, xr, src = ego_span(None, None, 840.0, CAM)
    assert src == "virtual" and (xl + xr) / 2 == pytest.approx(960.0)
    assert ego_span(left, right, 600.0, CAM)[2] == "none"
    assert in_path_fraction([900, 500, 1000, 600], left, right, CAM) == 0.0   # bottom above the horizon


def test_select_cipo_rules():
    left = np.stack(project(np.linspace(4, 80, 30), np.full(30, 1.75)), axis=1)
    right = np.stack(project(np.linspace(4, 80, 30), np.full(30, -1.75)), axis=1)
    objs = [car(20.0, 0.0, 1), car(40.0, 0.2, 3), car(12.0, 3.5, 3), car(90.0, 0.0, 4)]
    boxes = np.array([[o["x"], o["y"], o["x"] + o["width"], o["y"] + o["height"]] for o in objs])
    scores = np.array([0.9, 0.8, 0.95, 0.7])
    l1 = np.array([0.2, 0.7, 0.1, 0.05])
    assert select_cipo(boxes, scores, l1, left, right, CAM, CipoRule("geometric")) == 0      # nearest in path
    assert select_cipo(boxes, scores, l1, left, right, CAM, CipoRule("geometric", score_threshold=0.92)) is None
    assert select_cipo(boxes[[1, 3]], scores[[1, 3]], l1[[1, 3]], left, right, CAM,
                       CipoRule("geometric", max_distance=30.0)) is None                        # too far
    assert select_cipo(boxes, scores, l1, left, right, CAM, CipoRule("learned", level1_threshold=0.5)) == 1
    assert select_cipo(boxes, scores, l1, left, right, CAM, CipoRule("learned", level1_threshold=0.8)) is None
    assert select_cipo(boxes, scores, l1, left, right, CAM, CipoRule("hybrid", level1_threshold=0.5)) == 1
    geometry = path_geometry(boxes, left, right, CAM)
    assert select_cipo(boxes, scores, l1, left, right, CAM, CipoRule("geometric"), geometry=geometry) == 0
    assert select_cipo(np.zeros((0, 4)), np.zeros(0), np.zeros(0), left, right, CAM, CipoRule()) is None
    with pytest.raises(ValueError):
        CipoRule("nearest")


# -- metrics -----------------------------------------------------------------------------------------------------------

def test_average_precision_hand_cases():
    gt = {"a": np.array([[0, 0, 10, 10]]), "b": np.array([[0, 0, 10, 10]]), "c": np.zeros((0, 4))}
    perfect = [("a", 0.9, np.array([0, 0, 10, 10])), ("b", 0.8, np.array([0, 0, 10, 10]))]
    assert average_precision(perfect, gt) == pytest.approx(1.0)
    # a false positive ranked first: precision 0, 1/2, 2/3 at recall 0, 0.5, 1 -> envelope 2/3 everywhere
    with_fp = [("c", 0.95, np.array([0, 0, 10, 10]))] + perfect
    assert average_precision(with_fp, gt) == pytest.approx(2 / 3)
    # a duplicate counts as a false positive after the match
    dup = perfect + [("a", 0.7, np.array([0, 0, 10, 10]))]
    assert average_precision(dup, gt) == pytest.approx(1.0)
    half = [("a", 0.9, np.array([0, 0, 10, 10]))]
    assert average_precision(half, gt) == pytest.approx(51 / 101)        # recall 0.5: points 0..0.5 have P = 1
    assert average_precision([], gt) == 0.0
    assert np.isnan(average_precision(perfect, {"a": np.zeros((0, 4))}))
    # several thresholds at once: IoU 10/12 = 0.83 is a hit at 0.5 and 0.8, a miss at 0.9
    loose = [("a", 0.9, np.array([0, 0, 10, 10])), ("b", 0.8, np.array([0, 0, 10, 12]))]
    np.testing.assert_allclose(average_precisions(loose, gt, [0.5, 0.8, 0.9]), [1.0, 1.0, 51 / 101])
    assert average_precision(loose, gt, 0.9) == pytest.approx(51 / 101)
    assert box_iou(np.array([[0, 0, 10, 10]]), np.array([[5, 0, 15, 10]]))[0, 0] == pytest.approx(1 / 3)


def test_cipo_frame_metrics_and_stability():
    box = np.array([0.0, 0.0, 10.0, 10.0])
    far = np.array([50.0, 50.0, 60.0, 60.0])
    gt = [box[None], np.zeros((0, 4)), box[None], box[None], np.zeros((0, 4))]
    pred = [box, None, None, far, far]
    m = cipo_frame_metrics(pred, gt)
    assert m["cipo_accuracy"] == pytest.approx(2 / 5)
    assert m["false_cipo_rate"] == pytest.approx(2 / 5) and m["missed_cipo_rate"] == pytest.approx(2 / 5)
    assert m["cipo_precision"] == pytest.approx(1 / 3) and m["cipo_recall"] == pytest.approx(1 / 3)
    s = cipo_stability(["s"] * 5 + ["t"], [0, 1, 2, 3, 4, 0], ["a", None, "a", "b", "b", "a"],
                       ["A", "A", "A", "A", "A", "A"], fps=10.0)
    assert s["consecutive_pairs"] == 4.0                                 # the sequence change is not a pair
    assert s["cipo_flicker_per_min"] == pytest.approx(2 / (4 / 600))
    assert s["cipo_switches_per_min"] == pytest.approx(1 / (4 / 600))
    d = distance_errors(np.array([10.0, 21.0, np.nan, 100.0]), np.array([10.0, 20.0, 5.0, 90.0]))
    assert d["n"] == 3 and d["within_10pct"] == pytest.approx(2 / 3)
    assert d["n_0_20m"] == 1 and d["abs_rel_80m_plus"] == pytest.approx(10 / 90)


# -- the script --------------------------------------------------------------------------------------------------------

def _load_script():
    spec = importlib.util.spec_from_file_location("train_cipo", PROJECT_ROOT / "scripts" / "train_cipo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_cipo_script_end_to_end(tmp_path):
    root = tmp_path / "OpenLane"
    sequences = make_cipo_openlane(root, zip_cipo=True)
    cfg = _tiny_cfg(root, tmp_path)
    model = build_model("ufld_baseline", cfg, pretrained=False)
    ckpt = tmp_path / "run" / "seed_1" / "checkpoints" / "ufld_baseline.pt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"config": cfg.to_dict(), "variant": "ufld_baseline", "state_dict": model.state_dict()}, ckpt)
    write_manifest(tmp_path / "run" / "data" / "split_manifest.csv", sequences)   # found next to the run
    out = tmp_path / "cipo_out"
    metrics = _load_script().main(["--lane-ckpt", str(ckpt), "--cipo", str(root / "cipo.zip"),
                                   "--openlane-root", str(root), "--out", str(out), "--epochs", "2",
                                   "--batch-size", "4", "--workers", "0", "--device", "cpu",
                                   "--latency-runs", "0", "--max-test-frames", "4"])
    assert metrics["frames"] == {"train": 11, "val": 6, "test": 4}
    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert set(saved["decision_test"]) == {"learned", "geometric", "hybrid"}
    assert 0.0 <= saved["detection_test"]["ap50_level1"] <= 1.0
    # with ground-truth lanes and labelled boxes the geometry finds every level-1 car (they sit in the ego lane)
    gt_lanes = saved["in_path_test"]["ground_truth_lanes"]
    assert gt_lanes["rule_validity"] == 1.0 and gt_lanes["span_sources"] == {"lanes": 2}
    assert gt_lanes["oracle_boxes_selection"]["cipo_accuracy"] == 1.0   # 70 m car rejected by the tuned distance
    assert saved["label_distance_test"]["abs_rel"] < 0.02                # flat road: proxy = reference
    assert saved["cost"]["head_params_millions"] < 0.1
    assert (out / "head.pt").exists() and (out / "history.csv").exists()
