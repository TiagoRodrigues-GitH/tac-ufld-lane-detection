"""ELAS dataset adapter (Ego-Lane Analysis System, Berriel et al.).

Ground-truth geometry (the audit's finding C1)
----------------------------------------------
``groundtruth.xml`` stores, per frame and per ego-lane side, four x
coordinates ``p1..p4`` (p1 = farthest, p4 = nearest). The y coordinates are
implicit: they sit at fractions **0, 1/4, 1/2 and 1** of the scene's region
of interest (ROI) height, measured from the top of the ROI.

The previous script assumed evenly spaced rows (0, 1/3, 2/3, 1), which
misplaces p2/p3 by up to ~20 px. Evidence (all 22 scenes, ~31k lane
sides): fitting a straight line through the four points gives a median
maximum residual of 1.0-4.6 px with (0, 1/4, 1/2, 1) versus 15.8-22.5 px
with thirds, and the gap ratio (p3->p4)/(p2->p3) is ~2.0 in every scene.
``tests/test_elas.py`` keeps this as a regression test on the real data.

Missing values
--------------
Coordinates are ``nan`` (or negative) when a point is not annotated. A side
with >= 2 valid points is a lane; a side with no valid point is "absent";
a side with exactly one valid point cannot be drawn and is marked
"unknown" (ignored by losses and metrics instead of being trained as
"no lane").

Slots: 0 = ego-left, 1 = ego-right, taken from the XML element names (the
old code sorted by mean x, which put a right-only lane in the left slot).
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from tac_ufld.data.base import LaneDatasetAdapter
from tac_ufld.data.types import FrameRecord

LOGGER = logging.getLogger(__name__)

ELAS_ROW_FRACTIONS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
ELAS_SIDES: tuple[str, ...] = ("left", "right")
_POINT_TAGS = ("p1", "p2", "p3", "p4")
_FRAME_ATTRS = (
    "intersection", "laneCenter", "laneChange", "laneLeft", "laneRight",
    "lmtLeft", "lmtRight", "roadSigns",
)
_IMAGE_NAME_PATTERNS = (
    "lane_{i}.png", "lane_{i}.jpg", "{i}.png", "{i}.jpg", "{i}.jpeg",
    "{i:04d}.png", "{i:04d}.jpg", "{i:05d}.png", "{i:05d}.jpg",
    "{i:06d}.png", "{i:06d}.jpg", "frame_{i}.png", "frame_{i:04d}.png",
    "frame_{i:05d}.png",
)


@dataclass(frozen=True)
class ElasSceneConfig:
    scene: str
    frame_start: int
    frame_end: int
    frame_w: int
    frame_h: int
    roi_x: int
    roi_y: int
    roi_w: int
    roi_h: int

    def point_rows_y(self) -> np.ndarray:
        """Image rows (original pixels) of p1..p4."""
        return self.roi_y + np.asarray(ELAS_ROW_FRACTIONS, dtype=np.float32) * self.roi_h

    @property
    def valid_y_range(self) -> tuple[float, float]:
        return float(self.roi_y), float(self.roi_y + self.roi_h)


@dataclass(frozen=True)
class ElasFrameAnnotation:
    lanes: tuple[np.ndarray | None, ...]
    known: tuple[bool, ...]
    attrs: dict


@dataclass(frozen=True)
class _ScenePaths:
    config: Path
    groundtruth: Path
    images: Path | None


def parse_scene_config(path: str | Path) -> ElasSceneConfig:
    ds = ET.parse(path).getroot().find("dataset")
    if ds is None:
        raise ValueError(f"{path}: missing <dataset> element")
    seq, size, roi = ds.find("frame_sequence"), ds.find("frame_size"), ds.find("region_of_interest")
    return ElasSceneConfig(
        scene=ds.get("id") or Path(path).parent.name,
        frame_start=int(seq.get("start")), frame_end=int(seq.get("end")),
        frame_w=int(size.get("width")), frame_h=int(size.get("height")),
        roi_x=int(roi.get("x")), roi_y=int(roi.get("y")),
        roi_w=int(roi.get("width")), roi_h=int(roi.get("height")),
    )


def _parse_float(text: str | None) -> float:
    try:
        return float(text) if text is not None else float("nan")
    except ValueError:
        return float("nan")


def parse_side(side_el: ET.Element | None, rows_y: np.ndarray) -> tuple[np.ndarray | None, bool]:
    """Return (lane points or None, slot_known) for one side element."""
    if side_el is None:
        return None, True
    points = []
    for tag, y in zip(_POINT_TAGS, rows_y):
        node = side_el.find(tag)
        x = _parse_float(node.text if node is not None else None)
        if np.isfinite(x) and x >= 0:
            points.append((x, float(y)))
    if len(points) >= 2:
        return np.asarray(points, dtype=np.float32), True
    if len(points) == 1:
        return None, False
    return None, True


def parse_groundtruth(path: str | Path, scene_cfg: ElasSceneConfig) -> dict[int, ElasFrameAnnotation]:
    rows_y = scene_cfg.point_rows_y()
    frames_el = ET.parse(path).getroot().find("frames")
    if frames_el is None:
        raise ValueError(f"{path}: missing <frames> element")
    out: dict[int, ElasFrameAnnotation] = {}
    for frame_el in frames_el.findall("frame"):
        fid = int(frame_el.get("id"))
        pos = frame_el.find("position")
        lanes, known = [], []
        for side in ELAS_SIDES:
            lane, is_known = parse_side(pos.find(side) if pos is not None else None, rows_y)
            lanes.append(lane)
            known.append(is_known)
        attrs = {k: _parse_float(frame_el.get(k)) for k in _FRAME_ATTRS if frame_el.get(k) is not None}
        out[fid] = ElasFrameAnnotation(tuple(lanes), tuple(known), attrs)
    return out


def discover_scenes(root: Path) -> dict[str, _ScenePaths]:
    """Index scenes below ``root``. Annotations and ``images/images`` may live
    in different export folders as long as they share the scene folder name."""
    configs: dict[str, Path] = {}
    gts: dict[str, Path] = {}
    images: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        if "config.xml" in filenames:
            configs.setdefault(d.name, d / "config.xml")
        if "groundtruth.xml" in filenames:
            gts.setdefault(d.name, d / "groundtruth.xml")
        if d.name == "images" and d.parent.name == "images":
            images.setdefault(d.parent.parent.name, d)
            dirnames[:] = []  # do not descend into thousands of image files
    return {
        name: _ScenePaths(configs[name], gts[name], images.get(name))
        for name in configs
        if name in gts
    }


class ElasAdapter(LaneDatasetAdapter):
    name = "elas"
    num_lanes = 2
    flip_permutation = (1, 0)  # ego-left <-> ego-right
    protocol = {
        "split": "no official split: held-out test scenes + seed-independent 60-frame blocks "
                 "with purge gap on the other scenes (data.split)",
        "test": "held-out scenes (primary) and seen-scene blocks (secondary)",
        "primary_metric": "lane_f1_iou50: CULane-style lane F1, 30 px width scaled to the image "
                          "(12 px at 640), Hungarian matching, IoU >= 0.5, within the scene ROI",
        "native_metric": "none (ELAS has no official lane-detection benchmark)",
        "annotation": "p1..p4 at 0, 1/4, 1/2, 1 of the ROI height; slots 0 = ego-left, 1 = ego-right",
    }

    def __init__(
        self,
        root: str | Path,
        scenes: list[str] | None = None,
        scene_tags: dict[str, dict[str, bool]] | None = None,
        include_frames_without_lanes: bool = False,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"ELAS root not found: {self.root}. Set data.root in the config "
                f"or the TAC_UFLD_DATA_ROOT environment variable."
            )
        index = discover_scenes(self.root)
        requested = list(scenes) if scenes else sorted(index)
        self.missing_scenes = [s for s in requested if s not in index or index[s].images is None]
        self._index = {s: index[s] for s in requested if s not in self.missing_scenes}
        if self.missing_scenes:
            LOGGER.warning("ELAS scenes missing annotations or images/images: %s", self.missing_scenes)
        if not self._index:
            raise FileNotFoundError(f"No usable ELAS scene found below {self.root}")
        self.scene_tags = scene_tags or {}
        self.include_frames_without_lanes = include_frames_without_lanes
        self._scene_cfg: dict[str, ElasSceneConfig] = {}
        self._pattern: dict[str, str] = {}
        self.stats: dict[str, dict[str, int]] = {}

    # -- interface -----------------------------------------------------------

    def sequences(self) -> list[str]:
        return list(self._index)

    def scene_config(self, scene: str) -> ElasSceneConfig:
        if scene not in self._scene_cfg:
            self._scene_cfg[scene] = parse_scene_config(self._index[scene].config)
        return self._scene_cfg[scene]

    def frame_path(self, sequence: str, frame_id: int) -> Path | None:
        if frame_id < 0 or sequence not in self._index:
            return None
        root = self._index[sequence].images
        cached = self._pattern.get(sequence)
        if cached is not None:
            candidate = root / cached.format(i=frame_id)
            if candidate.exists():
                return candidate
        for pattern in _IMAGE_NAME_PATTERNS:
            candidate = root / pattern.format(i=frame_id)
            if candidate.exists():
                self._pattern[sequence] = pattern
                return candidate
        return None

    def load_sequence(self, sequence: str) -> list[FrameRecord]:
        scene_cfg = self.scene_config(sequence)
        annotations = parse_groundtruth(self._index[sequence].groundtruth, scene_cfg)
        records: list[FrameRecord] = []
        stats = {"annotated": len(annotations), "no_lane": 0, "missing_image": 0,
                 "unknown_slots": 0, "used": 0}
        for fid in sorted(annotations):
            ann = annotations[fid]
            stats["unknown_slots"] += sum(not k for k in ann.known)
            if not any(lane is not None for lane in ann.lanes) and not self.include_frames_without_lanes:
                stats["no_lane"] += 1
                continue
            path = self.frame_path(sequence, fid)
            if path is None:
                stats["missing_image"] += 1
                continue
            records.append(FrameRecord(
                dataset=self.name, sequence=sequence, frame_id=fid, image_path=path,
                image_size=(scene_cfg.frame_w, scene_cfg.frame_h),
                lanes=ann.lanes, slot_known=ann.known,
                valid_y_range=scene_cfg.valid_y_range,
                tags=self.scene_tags.get(sequence),
                meta={"roi_y": scene_cfg.roi_y, "roi_h": scene_cfg.roi_h, **ann.attrs},
            ))
        stats["used"] = len(records)
        self.stats[sequence] = stats
        if records:
            self._check_image_size(records[0], scene_cfg)
        return records

    @staticmethod
    def _check_image_size(record: FrameRecord, scene_cfg: ElasSceneConfig) -> None:
        with Image.open(record.image_path) as img:
            if img.size != (scene_cfg.frame_w, scene_cfg.frame_h):
                raise ValueError(
                    f"{record.sequence}: image size {img.size} differs from config.xml "
                    f"frame_size {(scene_cfg.frame_w, scene_cfg.frame_h)}"
                )
