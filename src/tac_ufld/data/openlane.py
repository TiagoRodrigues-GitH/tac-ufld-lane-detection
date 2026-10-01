"""OpenLane (v1, 2D lanes) adapter.

On-disk layout (OpenLane release)::

    images/training/segment-<id>/<timestamp>.jpg
    images/validation/segment-<id>/<timestamp>.jpg
    lane3d_1000/training/segment-<id>/<timestamp>.json   (or lane3d_300/...)
    lane3d_1000/validation/segment-<id>/<timestamp>.json
    lane3d_1000/test/<scenario>_case/segment-<id>/<timestamp>.json   (optional scenario subsets)

Each annotation JSON holds ``file_path``, the camera ``intrinsic`` (3x3) and
``extrinsic`` (4x4), and ``lane_lines``; every lane has ``uv`` (2xN image
points), ``xyz`` (3xN camera-frame points), ``visibility`` (N), ``category``
(lane-marking type), ``attribute`` (1 = left-left, 2 = left, 3 = right,
4 = right-right relative to the ego vehicle, 0 = none of these) and
``track_id``. Images are 1920x1280 (Waymo front camera), 10 Hz.

Conventions:

* **Sequence / frame id**: the segment and the index of the frame in the
  segment's time-ordered frame list (timestamps are kept in ``meta``), so
  history frames are simply ``frame_id - k * temporal_step``.
* **Lane slots** (4): directly from ``attribute`` (1..4 -> slot 0..3). If
  two lanes share an attribute, the one with more visible points is kept.
  Lanes with attribute 0 are not assigned to a slot but are kept in
  ``meta["eval_lanes"]`` for the native evaluation.
* **Points**: ``uv`` points with visibility <= 0.5 or outside the image are
  dropped (2D protocol); lanes need >= 2 remaining points.
* **Camera metadata**: ``meta["camera"]`` keeps ``intrinsic`` / ``extrinsic``;
  3D coordinates are not used by this 2D pipeline.
* **Split** (dataset-specific): OpenLane's ``validation`` set is the test set
  (as in the literature). Validation for model selection is carved from the
  training segments by holding out whole segments
  (``data.split.val_strategy: sequences``).
* **Native metric**: OpenLane's 2D evaluation is CULane-style F1 (IoU 0.5).
  ``native_openlane_f1_iou50`` uses every annotated lane, 30 px lines at the
  native 1920x1280 resolution; this matches our reading of the OpenLane
  2D evaluation settings but was not cross-checked against the official code.

Status: tested on synthetic files that reproduce this layout. No OpenLane copy
was available during development; run ``python -m tac_ufld validate-dataset
--dataset openlane`` on the real data before reporting results.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from tac_ufld.data.base import LaneDatasetAdapter
from tac_ufld.data.types import FrameRecord

LOGGER = logging.getLogger(__name__)

OPENLANE_IMAGE_SIZE = (1920, 1280)
_SPLITS = {"train": "training", "test": "validation"}


def lane_uv(lane: dict, image_size: tuple[int, int], min_visibility: float = 0.5) -> np.ndarray | None:
    """Visible (u, v) points of one OpenLane lane inside the image, sorted by v."""
    uv = np.asarray(lane.get("uv", []), dtype=np.float32)
    if uv.ndim != 2 or uv.shape[0] != 2 or uv.shape[1] < 2:
        return None
    pts = uv.T
    vis = np.asarray(lane.get("visibility", np.ones(len(pts))), dtype=np.float32)
    if len(vis) != len(pts):
        vis = np.ones(len(pts), dtype=np.float32)
    w, h = image_size
    keep = (vis > min_visibility) & np.isfinite(pts).all(axis=1)
    keep &= (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    pts = pts[keep]
    if len(pts) < 2:
        return None
    return pts[np.argsort(pts[:, 1])]


def attribute_slots(lanes: list[tuple[int, np.ndarray]], num_lanes: int = 4
                    ) -> tuple[list[np.ndarray | None], int]:
    """(attribute, points) pairs -> slots; duplicates keep the longest lane.
    Returns (slots, number of duplicate lanes dropped)."""
    slots: list[np.ndarray | None] = [None] * num_lanes
    duplicates = 0
    for attribute, pts in lanes:
        if not 1 <= attribute <= num_lanes:
            continue
        current = slots[attribute - 1]
        if current is None:
            slots[attribute - 1] = pts
        else:
            duplicates += 1
            if len(pts) > len(current):
                slots[attribute - 1] = pts
    return slots, duplicates


def _timestamp(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 0


class OpenLaneAdapter(LaneDatasetAdapter):
    name = "openlane"
    num_lanes = 4
    flip_permutation = (3, 2, 1, 0)
    protocol = {
        "split": "official training segments for train/val (whole segments held out for val); "
                 "official validation set = test",
        "test": "OpenLane validation segments (scenario tags from lane3d_*/test/<case> when present)",
        "primary_metric": "lane_f1_iou50 (internal, attribute slot lanes, 30 px scaled to 1920 -> 35 px)",
        "native_metric": "native_openlane_f1_iou50: every annotated lane, 30 px at 1920x1280, IoU >= 0.5 "
                         "(our reading of OpenLane 2D eval; not cross-checked with the official code)",
        "annotation": "slots from lane attribute 1..4 (left-left, left, right, right-right)",
    }

    def __init__(self, root: str | Path, segments: list[str] | None = None,
                 annotation_dir: str | None = None, skip_unattributed_frames: bool = False) -> None:
        self.root = Path(root)
        self.skip_unattributed_frames = skip_unattributed_frames
        if not self.root.is_dir():
            raise FileNotFoundError(f"OpenLane root not found: {self.root}")
        self.ann_root = self._find_annotations(annotation_dir)
        self.image_root = self._find_images()
        self.segments = set(segments) if segments else None
        self._frames: dict[str, list[tuple[int, Path]]] = {}   # sequence -> [(timestamp, image)]
        self._splits: dict[str, list[FrameRecord]] | None = None
        self.stats: dict[str, dict[str, int]] = {}
        self._scenarios = self._scenario_tags()

    # -- discovery ---------------------------------------------------------------

    def _find_annotations(self, annotation_dir: str | None) -> Path:
        names = [annotation_dir] if annotation_dir else ["lane3d_1000", "lane3d_300"]
        for name in names:
            for candidate in [self.root / name, *self.root.glob(f"*/{name}")]:
                if (candidate / "training").is_dir() or (candidate / "validation").is_dir():
                    return candidate
        raise FileNotFoundError(f"no lane3d_1000/ or lane3d_300/ with training/ or validation/ below {self.root}")

    def _find_images(self) -> Path:
        for candidate in [self.root / "images", *self.root.glob("*/images")]:
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(f"no images/ folder below {self.root}")

    def _scenario_tags(self) -> dict[str, list[str]]:
        """``lane3d_*/test/<scenario>_case/segment-*/<ts>.json`` -> scenario names."""
        out: dict[str, list[str]] = {}
        test_dir = self.ann_root / "test"
        if not test_dir.is_dir():
            return out
        for case in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            tag = case.name.removesuffix("_case")
            for js in case.rglob("*.json"):
                out.setdefault(f"{js.parent.name}/{js.stem}", []).append(tag)
        return out

    def _segment_frames(self, split_dir: str, segment: str) -> list[tuple[int, Path]]:
        key = f"{split_dir}/{segment}"
        if key not in self._frames:
            folder = self.image_root / split_dir / segment
            frames = sorted(((_timestamp(p), p) for p in folder.glob("*.jpg")), key=lambda t: t[0])
            self._frames[key] = frames
        return self._frames[key]

    # -- parsing -------------------------------------------------------------------

    def _load_split(self, split: str, split_dir: str) -> list[FrameRecord]:
        stats: Counter = Counter()
        records: list[FrameRecord] = []
        ann_split = self.ann_root / split_dir
        if not ann_split.is_dir():
            LOGGER.warning("OpenLane: %s missing", ann_split)
            return records
        checked_size = False
        for seg_dir in sorted(p for p in ann_split.iterdir() if p.is_dir()):
            segment = seg_dir.name
            if self.segments is not None and segment not in self.segments:
                continue
            frames = self._segment_frames(split_dir, segment)
            index_of = {ts: i for i, (ts, _) in enumerate(frames)}
            sequence = f"{split_dir}/{segment}"
            for js in sorted(seg_dir.glob("*.json"), key=_timestamp):
                ts = _timestamp(js)
                if ts not in index_of:
                    stats["missing_image"] += 1
                    continue
                image = frames[index_of[ts]][1]
                if not checked_size:
                    with Image.open(image) as img:
                        if img.size != OPENLANE_IMAGE_SIZE:
                            raise ValueError(f"OpenLane image {image} is {img.size}, expected {OPENLANE_IMAGE_SIZE}")
                    checked_size = True
                item = json.loads(js.read_text(encoding="utf-8"))
                parsed, all_lanes = [], []
                for lane in item.get("lane_lines", []):
                    pts = lane_uv(lane, OPENLANE_IMAGE_SIZE)
                    if pts is None:
                        stats["lanes_invisible"] += 1
                        continue
                    all_lanes.append(pts)
                    parsed.append((int(lane.get("attribute", 0)), pts))
                slots, duplicates = attribute_slots(parsed, self.num_lanes)
                if self.skip_unattributed_frames and all_lanes and all(s is None for s in slots):
                    stats["skipped_unattributed"] += 1   # lanes visible, none with an ego-relative attribute
                    continue
                stats["lanes"] += len(all_lanes)
                stats["lanes_attribute0"] += sum(a == 0 for a, _ in parsed)
                stats["duplicate_attribute"] += duplicates
                tags = self._scenarios.get(f"{segment}/{ts}")
                stats["used"] += 1
                records.append(FrameRecord(
                    dataset=self.name, sequence=sequence, frame_id=index_of[ts], image_path=image,
                    image_size=OPENLANE_IMAGE_SIZE, lanes=tuple(slots), slot_known=(True,) * self.num_lanes,
                    valid_y_range=None, tags={t: True for t in tags} if tags else None,
                    meta={"split": split, "segment": segment, "timestamp": ts,
                          "camera": {"intrinsic": item.get("intrinsic"), "extrinsic": item.get("extrinsic")},
                          "eval_lanes": all_lanes,
                          "categories": [int(l.get("category", -1)) for l in item.get("lane_lines", [])],
                          "track_ids": [int(l.get("track_id", -1)) for l in item.get("lane_lines", [])]},
                ))
        self.stats[split] = dict(stats)
        return records

    def official_splits(self) -> dict[str, list[FrameRecord]]:
        if self._splits is None:
            out = {split: self._load_split(split, d) for split, d in _SPLITS.items()}
            out["val"] = []  # carved from train (whole segments) by the experiment
            self._splits = out
        return self._splits

    # -- interface -------------------------------------------------------------------

    def sequences(self) -> list[str]:
        return sorted({r.sequence for recs in self.official_splits().values() for r in recs})

    def load_sequence(self, sequence: str) -> list[FrameRecord]:
        return [r for recs in self.official_splits().values() for r in recs if r.sequence == sequence]

    def frame_path(self, sequence: str, frame_id: int) -> Path | None:
        if frame_id < 0:
            return None
        split_dir, _, segment = sequence.partition("/")
        frames = self._segment_frames(split_dir, segment)
        return frames[frame_id][1] if frame_id < len(frames) else None
