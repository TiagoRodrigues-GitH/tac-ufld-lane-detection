"""Shared fixtures: synthetic ELAS scenes and helpers."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from tac_ufld.config import PROJECT_ROOT, load_config
from tac_ufld.data.elas import ELAS_ROW_FRACTIONS

REAL_ELAS = (Path(os.environ["ELAS_ROOT"]) if os.environ.get("ELAS_ROOT")
             else PROJECT_ROOT / "../../datasets/dataset_elas_v1").resolve()


def lane_x(y: np.ndarray, side: str, width: int, height: int, drift: float = 0.0) -> np.ndarray:
    """Straight synthetic lanes converging to a vanishing point at 40 % height."""
    vy = 0.4 * height
    slope = 0.75 * width / height  # keeps both lanes inside the frame down to the bottom row
    offset = -1 if side == "left" else 1
    return width / 2 + drift + offset * slope * (y - vy)


def make_elas_scene(root: Path, name: str, n_frames: int = 60, size: tuple[int, int] = (160, 120),
                    roi: tuple[int, int] = (70, 50), drop_right: set[int] | None = None,
                    single_point_left: set[int] | None = None) -> Path:
    """Write ``root/name/{config.xml, groundtruth.xml, images/images/lane_<i>.png}``."""
    w, h = size
    roi_y, roi_h = roi
    scene = root / name
    images = scene / "images" / "images"
    images.mkdir(parents=True, exist_ok=True)
    (scene / "config.xml").write_text(
        f'<config><dataset id="{name}" name="{name}" path="{name}/images/">'
        f'<frame_sequence start="0" end="{n_frames - 1}"></frame_sequence>'
        f'<frame_size width="{w}" height="{h}"></frame_size>'
        f'<region_of_interest x="0" y="{roi_y}" width="{w}" height="{roi_h}"></region_of_interest>'
        f'<ipm_points top_left="0" top_right="0" bottom_right="0" bottom_left="0"></ipm_points>'
        f"</dataset></config>", encoding="utf-8")
    rows = roi_y + np.asarray(ELAS_ROW_FRACTIONS) * roi_h
    frames = []
    for i in range(n_frames):
        drift = 4.0 * np.sin(i / 7.0)
        sides = []
        for side in ("left", "right"):
            xs = lane_x(rows, side, w, h, drift)
            if side == "right" and drop_right and i in drop_right:
                xs = np.full(4, np.nan)
            if side == "left" and single_point_left and i in single_point_left:
                xs = np.array([xs[0], np.nan, np.nan, np.nan])
            pts = "".join(f"<p{k + 1}>{'nan' if np.isnan(x) else f'{x:.1f}'}</p{k + 1}>" for k, x in enumerate(xs))
            sides.append(f"<{side}>{pts}</{side}>")
        frames.append(f'<frame id="{i}" laneChange="0"><position>{"".join(sides)}</position></frame>')
        img = np.full((h, w, 3), 90, dtype=np.uint8)
        yy = np.arange(int(0.45 * h), h)
        for side in ("left", "right"):
            if side == "right" and drop_right and i in drop_right:
                continue
            xx = lane_x(yy.astype(float), side, w, h, drift)
            pts = np.stack([xx, yy], axis=1).round().astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, (255, 255, 255), 2)
        ok, buf = cv2.imencode(".png", img)
        buf.tofile(str(images / f"lane_{i}.png"))
    (scene / "groundtruth.xml").write_text(
        f'<groundtruth><frames n="{n_frames}">{"".join(frames)}</frames></groundtruth>', encoding="utf-8")
    return scene


@pytest.fixture
def synthetic_elas(tmp_path: Path) -> Path:
    root = tmp_path / "elas"
    for name, n in (("SYN_A", 130), ("SYN_B", 130), ("SYN_C", 70)):
        make_elas_scene(root, name, n_frames=n, drop_right={5, 6}, single_point_left={9})
    return root


@pytest.fixture
def tiny_config(synthetic_elas: Path, tmp_path: Path):
    """Smoke config shrunk to synthetic data, CPU, tiny images."""
    return load_config(PROJECT_ROOT / "configs" / "elas_smoke.yaml", {
        "name": "pytest_run", "output_dir": str(tmp_path / "results"), "device": "cpu",
        "data.root": str(synthetic_elas), "data.scenes": ["SYN_A", "SYN_B", "SYN_C"],
        "data.scene_tags": {"SYN_A": {"rainy": True}, "SYN_B": {}, "SYN_C": {"occlusion": True}},
        "data.img_h": 64, "data.img_w": 96, "data.griding_num": 24, "data.num_row_anchors": 8,
        "data.row_anchor_range": [0.55, 0.99],
        "data.split.test_scenes": ["SYN_C"], "data.split.block_size": 30,
        "data.split.max_train_frames": None, "data.split.max_val_frames": None,
        "data.split.max_test_frames": None, "data.split.max_seen_test_frames": None,
    })
