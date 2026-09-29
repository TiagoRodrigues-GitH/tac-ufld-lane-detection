"""Small synthetic copies of the CULane, TuSimple and OpenLane on-disk layouts.

They reproduce the real formats (folder structure, list files, JSON fields,
image sizes, frame-id conventions) with a handful of frames, so every adapter
can be tested offline. They do not replace a check on the real datasets
(``python -m tac_ufld validate-dataset``).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _write_jpg(path: Path, size: tuple[int, int], lanes: list[np.ndarray] | None = None) -> None:
    w, h = size
    img = np.full((h, w, 3), 70, dtype=np.uint8)
    for lane in lanes or []:
        pts = np.round(lane).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (255, 255, 255), max(2, w // 200))
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    buf.tofile(str(path))


def _converging_lanes(bottom_xs, size, vanish_y, top_y, step, drift=0.0):
    """Straight lanes from the bottom row to ``top_y``, converging to (w/2, vanish_y)."""
    w, h = size
    ys = np.arange(h - 1, top_y - 1, -step, dtype=np.float32)  # bottom -> top, as in CULane files
    lanes = []
    for bx in bottom_xs:
        t = (ys - vanish_y) / (h - 1 - vanish_y)
        xs = w / 2 + drift + (bx - w / 2) * t
        lanes.append(np.stack([xs, ys], axis=1))
    return lanes


# ---------------------------------------------------------------- CULane

CULANE_SIZE = (1640, 590)


def make_culane(root: Path) -> Path:
    """Two train drivers (30- and 90-frame stride), val, one test driver,
    seg labels, list files with existence flags and test categories."""
    bottoms = [250.0, 650.0, 1000.0, 1400.0]
    clips = {
        ("driver_23_30frame", "05151649_0422.MP4"): [0, 30, 60, 90, 120],
        ("driver_161_90frame", "06030819_0755.MP4"): [0, 90, 180, 270],
        ("driver_23_30frame", "05151700_0100.MP4"): [0, 30, 60],   # validation clip
        ("driver_37_30frame", "05181432_0203.MP4"): [0, 30, 60],   # test clip
    }
    train_gt, val_gt, test, normal, night = [], [], [], [], []
    for (driver, clip), ids in clips.items():
        for k, fid in enumerate(ids):
            rel = f"{driver}/{clip}/{fid:05d}.jpg"
            lanes = _converging_lanes(bottoms, CULANE_SIZE, 250, 270, 10, drift=5.0 * k)
            present = [0, 1, 2, 3]
            if clip == "05151649_0422.MP4" and fid == 60:
                present = [1, 2]          # only the ego lanes
            if clip == "05151649_0422.MP4" and fid == 90:
                # lane change: the car drifts right, every lane shifts left, and the
                # ego-right lane (label 3) starts left of the image centre
                lanes = [lane - np.array([220.0, 0.0], dtype=np.float32) for lane in lanes]
                present = [1, 2]
            kept = [lanes[i] for i in present]
            _write_jpg(root / rel, CULANE_SIZE, kept)
            (root / rel).with_suffix(".lines.txt").write_text(
                "\n".join(" ".join(f"{x:.2f} {y:.0f}" for x, y in lane) for lane in kept) + "\n", encoding="utf-8")
            flags = " ".join("1" if i in present else "0" for i in range(4))
            seg_rel = f"laneseg_label_w16/{driver}/{clip}/{fid:05d}.png"
            if driver == "driver_37_30frame":
                test.append(f"/{rel}")
                (night if fid == 60 else normal).append(f"/{rel}")
                continue
            label = np.zeros((CULANE_SIZE[1], CULANE_SIZE[0]), dtype=np.uint8)
            for i in present:
                pts = np.round(lanes[i]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(label, [pts], False, i + 1, 16)
            (root / seg_rel).parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".png", label)
            buf.tofile(str(root / seg_rel))
            line = f"/{rel} /{seg_rel} {flags}"
            (val_gt if clip == "05151700_0100.MP4" else train_gt).append(line)
    lists = root / "list"
    (lists / "test_split").mkdir(parents=True, exist_ok=True)
    (lists / "train_gt.txt").write_text("\n".join(train_gt) + "\n", encoding="utf-8")
    (lists / "val_gt.txt").write_text("\n".join(val_gt) + "\n", encoding="utf-8")
    (lists / "test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    (lists / "test_split" / "test0_normal.txt").write_text("\n".join(normal) + "\n", encoding="utf-8")
    (lists / "test_split" / "test8_night.txt").write_text("\n".join(night) + "\n", encoding="utf-8")
    return root


# -------------------------------------------------------------- TuSimple

TUSIMPLE_SIZE = (1280, 720)
H_SAMPLES = list(range(160, 720, 10))


def _tusimple_lane(bottom_x: float, start_y: int = 240, vanish=(640.0, 200.0)) -> list[float]:
    xs = []
    for y in H_SAMPLES:
        if y < start_y:
            xs.append(-2)
            continue
        t = (y - vanish[1]) / (719 - vanish[1])
        x = vanish[0] + (bottom_x - vanish[0]) * t
        xs.append(round(x, 1) if 0 <= x < TUSIMPLE_SIZE[0] else -2)  # outside the image = -2, as in the files
    return xs


def make_tusimple(root: Path, clips_per_drive: int = 6) -> Path:
    """train_set with two drives (0313-1, 0531), test_set sharing drive 0531
    (as in the real benchmark) plus drive 0530, and test_label.json."""
    def clip_lanes(k: int, extra_left: bool = False, short: bool = False) -> list[list[float]]:
        lanes = [_tusimple_lane(bx + 3 * k) for bx in (-150.0, 350.0, 900.0, 1450.0)]
        if extra_left:
            lanes.append(_tusimple_lane(-700.0))
        if short:  # < 90 px long: discarded by the official rule
            lanes.append([-2] * (len(H_SAMPLES) - 4) + [600.0, 602.0, 604.0, 606.0])
        return lanes

    def write_split(split_dir: Path, drives: list[str], json_path: Path, start_clip: int) -> None:
        lines = []
        for d_i, drive in enumerate(drives):
            for c in range(clips_per_drive):
                clip = str(start_clip + 20 * c)
                lanes = clip_lanes(c, extra_left=(c == 1), short=(c == 2))
                for f in range(14, 21):
                    pts = [np.array([[x, y] for x, y in zip(l, H_SAMPLES) if x >= 0], np.float32) for l in lanes]
                    _write_jpg(split_dir / "clips" / drive / clip / f"{f}.jpg", TUSIMPLE_SIZE,
                               [p for p in pts if len(p) >= 2])
                lines.append(json.dumps({"lanes": lanes, "h_samples": H_SAMPLES,
                                         "raw_file": f"clips/{drive}/{clip}/20.jpg"}))
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_split(root / "train_set", ["0313-1"], root / "train_set" / "label_data_0313.json", 6040)
    write_split(root / "train_set", ["0531"], root / "train_set" / "label_data_0531.json", 1492626000)
    write_split(root / "test_set", ["0530", "0531"], root / "test_label.json", 1492638000)
    return root


# -------------------------------------------------------------- OpenLane

OPENLANE_SIZE = (1920, 1280)


def _openlane_lane(bottom_x: float, attribute: int, visible_from: int = 0) -> dict:
    vs = np.arange(1279, 700, -40, dtype=np.float32)
    t = (vs - 650.0) / (1279 - 650.0)
    us = 960 + (bottom_x - 960) * t
    vis = np.ones(len(vs), dtype=np.float32)
    vis[:visible_from] = 0.0
    return {"category": 1, "attribute": attribute, "track_id": attribute + 10,
            "uv": [us.round(2).tolist(), vs.tolist()], "visibility": vis.tolist(),
            "xyz": [np.zeros(len(vs)).tolist()] * 3}


def make_openlane(root: Path, frames_per_segment: int = 5) -> Path:
    segments = {"training": ["segment-100", "segment-200", "segment-300", "segment-400"],
                "validation": ["segment-900"]}
    ts0 = 1507234567000000
    for split, segs in segments.items():
        for s_i, seg in enumerate(segs):
            for f in range(frames_per_segment):
                ts = ts0 + s_i * 10_000_000 + f * 100_000
                lanes = [_openlane_lane(bx + 4 * f, a) for bx, a in ((-600, 1), (500, 2), (1400, 3), (2500, 4))]
                lanes.append(_openlane_lane(-1800, 0))                 # outside the 4 ego-relative slots
                lanes[1]["visibility"][:3] = [0.0, 0.0, 0.0]            # partly occluded ego-left lane
                if f == 2:
                    lanes.append(_openlane_lane(520, 2, visible_from=6))  # duplicate attribute, shorter
                pts = [np.array(l["uv"], np.float32).T for l in lanes]
                _write_jpg(root / "images" / split / seg / f"{ts}.jpg", OPENLANE_SIZE, pts)
                ann = {"file_path": f"{split}/{seg}/{ts}.jpg", "intrinsic": [[2000, 0, 960], [0, 2000, 640], [0, 0, 1]],
                       "extrinsic": [[1, 0, 0, 1.5], [0, 1, 0, 0], [0, 0, 1, 2.1], [0, 0, 0, 1]], "lane_lines": lanes}
                path = root / "lane3d_1000" / split / seg / f"{ts}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(ann), encoding="utf-8")
                if split == "validation" and f < 2:
                    night = root / "lane3d_1000" / "test" / "night_case" / seg / f"{ts}.json"
                    night.parent.mkdir(parents=True, exist_ok=True)
                    night.write_text(json.dumps(ann), encoding="utf-8")
    return root
