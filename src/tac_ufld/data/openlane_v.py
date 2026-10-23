"""OpenLane-V adapter (video lane detection benchmark of RVLD, Jin et al., ICCV 2023).

OpenLane-V re-annotates 588 OpenLane videos: up to four lanes per frame (the ego lane's two boundaries and
the neighbouring ones, "ego and alternative lanes", as in CULane), with invisible parts filled in so that
lanes are temporally consistent. Train 68,143 frames / 450 segments, test 23,021 frames / 138 segments
(OpenLane's validation videos). Labels and lists come from https://github.com/dongkwonjin/RVLD.

On-disk layout::

    <root>/images/{training,validation}/<segment>/<timestamp>.jpg            (OpenLane images, 1920x1280)
    <root>/OpenLane-V/label/{training,validation}/<segment>/<timestamp>.pickle   {"lanes": [(270, 2) arrays]}
    <root>/OpenLane-V/list/datalist_{training,validation}.pickle             ["<segment>/<timestamp>", ...]
    <root>/OpenLane-V/images_<W>x<H>/{training,validation}/<segment>/<timestamp>.jpg   (scripts/build_openlane_v_cache.py)

Conventions:

* **Input**: as RVLD, the top ``CROP_TOP`` = 480 rows are removed and the 1920x800 band is the image. The
  band is read from the cache built by ``scripts/build_openlane_v_cache.py`` at the network size; there is no
  fallback to the original images (they would not be cropped). Records therefore live in band coordinates
  (``image_size = (1920, 800)``, ``y_band = y - 480``).
* **Lane slots** (4, left to right): lanes are ordered by their x at the lowest annotated row; lanes left of
  the image centre fill slot 1 (nearest) and 0, lanes right of it slot 2 (nearest) and 3. A fifth lane or a
  third lane on one side is not assigned (counted in ``stats``) but stays in the evaluation.
* **Evaluation**: ``meta["eval_lanes"]`` holds every annotated lane in ORIGINAL 1920x1280 pixels and
  ``meta["crop_top"]`` the offset; ``tac_ufld.evaluation.rvld_protocol`` scores them exactly as RVLD does.
* **Sequence / frame id**: the segment and the frame's index in the segment's time-ordered list of cached
  frames, so history frames are ``frame_id - k * temporal_step`` (10 Hz).
* **Split**: OpenLane-V's training list for train/val (whole segments held out for validation,
  ``data.split.val_strategy: sequences``), its validation list as the test set, as in RVLD.
"""

from __future__ import annotations

import logging
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from tac_ufld.data.base import LaneDatasetAdapter
from tac_ufld.data.types import FrameRecord

LOGGER = logging.getLogger(__name__)

ORIGINAL_SIZE = (1920, 1280)
CROP_TOP = 480
BAND_SIZE = (ORIGINAL_SIZE[0], ORIGINAL_SIZE[1] - CROP_TOP)
_SPLITS = {"train": "training", "test": "validation"}


def lane_slots(lanes: list[np.ndarray], num_lanes: int = 4, centre_x: float = ORIGINAL_SIZE[0] / 2
               ) -> tuple[list[np.ndarray | None], int]:
    """Lanes -> 4 slots (left-left, left, right, right-right) by their x at the lowest point.
    Returns (slots, number of lanes left without a slot)."""
    half = num_lanes // 2
    bottom_x = [float(lane[np.argmax(lane[:, 1]), 0]) for lane in lanes]
    left = sorted((x, i) for i, x in enumerate(bottom_x) if x < centre_x)[::-1]   # nearest first
    right = sorted((x, i) for i, x in enumerate(bottom_x) if x >= centre_x)        # nearest first
    slots: list[np.ndarray | None] = [None] * num_lanes
    for rank, (_, i) in enumerate(left[:half]):
        slots[half - 1 - rank] = lanes[i]
    for rank, (_, i) in enumerate(right[:half]):
        slots[half + rank] = lanes[i]
    return slots, max(0, len(left) - half) + max(0, len(right) - half)


def to_band(lane: np.ndarray) -> np.ndarray | None:
    """Full-frame (x, y) points -> band coordinates, keeping the points inside the band, sorted by y."""
    pts = np.asarray(lane, dtype=np.float32)[:, :2].copy()
    pts[:, 1] -= CROP_TOP
    keep = np.isfinite(pts).all(axis=1) & (pts[:, 1] >= 0) & (pts[:, 1] <= BAND_SIZE[1] - 1)
    pts = pts[keep]
    if len(pts) < 2:
        return None
    return pts[np.argsort(pts[:, 1])]


class OpenLaneVAdapter(LaneDatasetAdapter):
    name = "openlane_v"
    num_lanes = 4
    flip_permutation = (3, 2, 1, 0)
    protocol = {
        "split": "OpenLane-V training list for train/val (whole segments held out for val); "
                 "OpenLane-V validation list = test (RVLD protocol)",
        "test": "23,021 frames of 138 OpenLane validation videos",
        "input": "top 480 rows removed, 1920x800 band at the network size (RVLD: 640x384)",
        "primary_metric": "native_openlanev_f1_iou50 (RVLD/LaneATT: 960x640, 30 px, spline, IoU > 0.5); "
                          "also F1 at IoU 0.8, mIoU, flickering and missing rates",
        "annotation": "up to 4 ego and alternative lanes; slots by side of the image centre at the lowest point",
    }

    def __init__(self, root: str | Path, img_size: tuple[int, int] = (640, 384),
                 segments: list[str] | None = None) -> None:
        self.root = Path(root)
        self.label_root = self.root / "OpenLane-V" / "label"
        self.list_root = self.root / "OpenLane-V" / "list"
        self.cache_root = self.root / "OpenLane-V" / f"images_{img_size[0]}x{img_size[1]}"
        for path, hint in [(self.label_root, "OpenLane-V labels (RVLD repository)"),
                           (self.list_root, "OpenLane-V lists (RVLD repository)"),
                           (self.cache_root, "python scripts/build_openlane_v_cache.py <root> "
                                             f"--size {img_size[0]}x{img_size[1]}")]:
            if not path.is_dir():
                raise FileNotFoundError(f"OpenLane-V: {path} not found ({hint})")
        self.segments = set(segments) if segments else None
        self._frames: dict[str, list[Path]] = {}
        self._index: dict[str, dict[str, int]] = {}
        self._splits: dict[str, list[FrameRecord]] | None = None
        self.stats: dict[str, dict[str, int]] = {}

    def _segment_frames(self, split_dir: str, segment: str) -> list[Path]:
        key = f"{split_dir}/{segment}"
        if key not in self._frames:
            folder = self.cache_root / split_dir / segment
            self._frames[key] = sorted(folder.glob("*.jpg"), key=lambda p: int(p.stem))
        return self._frames[key]

    def _datalist(self, split_dir: str) -> list[str]:
        with (self.list_root / f"datalist_{split_dir}.pickle").open("rb") as f:
            return list(pickle.load(f))

    def _labels(self, split_dir: str, names: list[str]) -> dict[str, list[np.ndarray]]:
        """Lanes of every listed frame. Reading tens of thousands of small pickles is slow on a SATA disk (over
        20 min on 2026-10-05), so they are gathered once into ``OpenLane-V/labels_<split>.pickle``."""
        index = self.label_root.parent / f"labels_{split_dir}.pickle"
        if index.exists():
            with index.open("rb") as f:
                labels = pickle.load(f)
            if set(names) <= labels.keys():
                return labels
        labels = {}
        for name in names:
            segment, ts = name.split("/")
            with (self.label_root / split_dir / segment / f"{ts}.pickle").open("rb") as f:
                labels[name] = [np.asarray(l, dtype=np.float32) for l in pickle.load(f)["lanes"]]
        tmp = index.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(index)
        return labels

    def _load_split(self, split: str, split_dir: str) -> list[FrameRecord]:
        stats: Counter = Counter()
        records: list[FrameRecord] = []
        names = self._datalist(split_dir)
        labels = self._labels(split_dir, names)
        for name in names:
            segment, ts = name.split("/")
            if self.segments is not None and segment not in self.segments:
                continue
            frames = self._segment_frames(split_dir, segment)
            key = f"{split_dir}/{segment}"
            if key not in self._index:
                self._index[key] = {p.stem: i for i, p in enumerate(frames)}
            index_of = self._index[key]
            if ts not in index_of:
                stats["missing_image"] += 1
                continue
            full = [lane for lane in labels[name] if len(lane) >= 2]
            band = [b for b in (to_band(l) for l in full) if b is not None]
            slots, unassigned = lane_slots(band, self.num_lanes)
            stats["lanes"] += len(full)
            stats["lanes_without_slot"] += unassigned
            stats["used"] += 1
            records.append(FrameRecord(
                dataset=self.name, sequence=f"{split_dir}/{segment}", frame_id=index_of[ts],
                image_path=frames[index_of[ts]], image_size=BAND_SIZE, lanes=tuple(slots),
                slot_known=(True,) * self.num_lanes, valid_y_range=None, tags=None,
                meta={"split": split, "segment": segment, "timestamp": int(ts), "crop_top": CROP_TOP,
                      "eval_lanes": full},
            ))
        if stats["missing_image"]:
            LOGGER.warning("OpenLane-V %s: %d listed frames have no cached image (download or cache incomplete)",
                           split_dir, stats["missing_image"])
        self.stats[split] = dict(stats)
        return records

    def official_splits(self) -> dict[str, list[FrameRecord]]:
        if self._splits is None:
            out = {split: self._load_split(split, d) for split, d in _SPLITS.items()}
            out["val"] = []  # carved from train (whole segments) by the experiment
            self._splits = out
        return self._splits

    def sequences(self) -> list[str]:
        return sorted({r.sequence for recs in self.official_splits().values() for r in recs})

    def load_sequence(self, sequence: str) -> list[FrameRecord]:
        return [r for recs in self.official_splits().values() for r in recs if r.sequence == sequence]

    def frame_path(self, sequence: str, frame_id: int) -> Path | None:
        if frame_id < 0:
            return None
        split_dir, _, segment = sequence.partition("/")
        frames = self._segment_frames(split_dir, segment)
        return frames[frame_id] if frame_id < len(frames) else None
