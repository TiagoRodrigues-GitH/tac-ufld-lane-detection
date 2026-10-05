"""The CARLA collector's geometry and file layout, tested without the CARLA client installed."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import carla_collect as cc  # importable without the `carla` client

CAMERA = cc.CameraModel()
AT_ORIGIN = np.eye(4)  # camera at the world origin looking along +x (Unreal axes)


def test_intrinsics_for_90_degree_fov():
    k = CAMERA.intrinsics()
    assert k[0, 0] == pytest.approx(640.0) and k[1, 1] == pytest.approx(640.0)
    assert (k[0, 2], k[1, 2]) == (640.0, 360.0)


def test_project_ahead_right_up_and_behind():
    uv = CAMERA.project(np.array([[10.0, 0, 0], [10.0, 2.0, 0], [10.0, 0, 1.0], [-5.0, 0, 0]]), AT_ORIGIN)
    np.testing.assert_allclose(uv[0], [640, 360])
    np.testing.assert_allclose(uv[1], [640 + 640 * 2 / 10, 360])   # +y (right) -> larger u
    np.testing.assert_allclose(uv[2], [640, 360 - 640 * 1 / 10])   # +z (up) -> smaller v
    assert np.isnan(uv[3]).all()                                    # behind the camera


def test_sample_rows_vertical_line_and_gaps():
    near_to_far = np.array([[100.0, 700.0], [100.0, 160.0]])
    xs = cc.sample_rows(near_to_far, CAMERA.width)
    assert xs[:-1] == [100] * (len(cc.H_SAMPLES) - 1)  # rows 160..700
    assert xs[-1] == -2                                # row 710 is below the line
    off_image = np.array([[1500.0, 700.0], [1500.0, 160.0]])
    assert set(cc.sample_rows(off_image, CAMERA.width)) == {-2}
    with_nan = np.array([[np.nan, np.nan], [100.0, 700.0], [100.0, 160.0]])
    assert cc.sample_rows(with_nan, CAMERA.width)[0] == 100


def test_occluded_fraction():
    tags = np.zeros((CAMERA.height, CAMERA.width), dtype=np.uint8)
    lanes = [[100] * len(cc.H_SAMPLES)]
    assert cc.occluded_fraction(tags, lanes) == 0.0
    tags[cc.H_SAMPLES[0], 100] = 14  # a car over the first labelled point
    assert cc.occluded_fraction(tags, lanes) == pytest.approx(1 / len(cc.H_SAMPLES))
    assert cc.occluded_fraction(tags, [[-2] * len(cc.H_SAMPLES)]) == 0.0


def test_usable_lanes_needs_enough_points():
    short = [-2] * (len(cc.H_SAMPLES) - 4) + [1, 2, 3, 4]
    assert cc.usable_lanes([short, [5] * len(cc.H_SAMPLES)]) == [[5] * len(cc.H_SAMPLES)]


def test_clip_writer_layout_and_numbering(tmp_path):
    writer = cc.ClipWriter(tmp_path, "train", drive="Town04-s1")
    first = writer.new_clip()
    raw = writer.keep(first, [[1] * len(cc.H_SAMPLES)], {"town": "Town04"})
    assert raw == "clips/Town04-s1/00000/20.jpg"
    rejected = writer.new_clip()
    writer.discard(rejected)
    assert not rejected.exists()
    label = json.loads(writer.label_path.read_text(encoding="utf-8").splitlines()[0])
    assert label["raw_file"] == raw and label["h_samples"] == list(cc.H_SAMPLES)
    meta = json.loads((tmp_path / "train_meta.jsonl").read_text(encoding="utf-8"))
    assert meta == {"raw_file": raw, "town": "Town04"}
    again = cc.ClipWriter(tmp_path, "train", drive="Town04-s1")  # a second run continues the numbering
    assert again.new_clip().name == "00001"


def test_test_split_label_file(tmp_path):
    writer = cc.ClipWriter(tmp_path, "test", drive="Town05-s1")
    assert writer.label_path == tmp_path / "test_label.json"
    assert writer.split_dir == tmp_path / "test_set"


def test_parse_args_defaults():
    args = cc.parse_args(["--out", "x"])
    assert (args.town, args.clips, args.split, args.traffic) == ("Town04", 20, "train", 40)
