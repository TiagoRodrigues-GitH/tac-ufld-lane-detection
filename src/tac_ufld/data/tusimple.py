"""TuSimple lane-detection benchmark adapter.

On-disk layout (TuSimple release)::

    train_set/label_data_0313.json, label_data_0531.json, label_data_0601.json
    train_set/clips/<drive>/<clip>/1.jpg .. 20.jpg
    test_set/clips/<drive>/<clip>/1.jpg .. 20.jpg
    test_label.json                      (test ground truth, released separately;
                                          may also sit inside test_set/)

Each JSON line is ``{"lanes": [[x, ...], ...], "h_samples": [y, ...],
"raw_file": "clips/<drive>/<clip>/20.jpg"}``: lane ``i`` has x = ``lanes[i][k]``
at y = ``h_samples[k]``, and x = -2 where the lane is absent. Only the 20th
frame of each 1-second, 20-frame clip is annotated; frames 1..19 are
available as temporal context. Images are 1280x720.

Conventions:

* **Sequence / frame id**: the clip folder (``<split>/clips/<drive>/<clip>``)
  and the frame number (20), so history frames are 19, 18, ...
* **Lane slots** (4): the official UFLD rule (``scripts/convert_tusimple.py``):
  lanes whose end-to-end length is < 90 px are discarded; the rest are split
  by the sign of their slope angle (left lanes have negative dy/dx in image
  coordinates); on each side the two lanes closest to the centre (steepest)
  fill slots 1, 0 (left) and 2, 3 (right).
* **Valid band**: rows outside ``[min(h_samples), max(h_samples)]`` are not
  annotated and are ignored in training and evaluation.
* **Split** (dataset-specific): the official ``test_label.json`` is the test
  set. TuSimple has no official validation set, so validation is carved from
  the training clips in temporal blocks of consecutive clip numbers within
  each drive, with a purge gap (``data.split.val_strategy: blocks``).
  Clips of the official test set come from some of the same drives as the
  training clips: this is the official protocol, not a scene-disjoint test.
* **Native metric**: the official ``LaneEval.bench`` (accuracy, FP, FN) is in
  ``tac_ufld.evaluation.native``.

Status: tested on synthetic files that reproduce this layout. No TuSimple copy
was available during development; run ``python -m tac_ufld validate-dataset
--dataset tusimple`` on the real data before reporting results.
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

TUSIMPLE_IMAGE_SIZE = (1280, 720)
MIN_LANE_LENGTH_PX = 90.0  # official convert_tusimple.py: shorter lanes are skipped


def lane_points(xs: list[float], h_samples: list[float]) -> np.ndarray | None:
    """(N, 2) points of one TuSimple lane (x = -2 or negative -> missing)."""
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(h_samples, dtype=np.float32)
    keep = np.isfinite(x) & (x >= 0)
    if keep.sum() < 2:
        return None
    return np.stack([x[keep], y[keep]], axis=1)


def slope_angle(lane: np.ndarray) -> float | None:
    """Official ``calc_k``: arctan of the slope of y = f(x); ``None`` (the
    official -10 sentinel) for lanes shorter than 90 px end to end."""
    length = float(np.hypot(*(lane[0] - lane[-1])))
    if length < MIN_LANE_LENGTH_PX:
        return None
    if np.ptp(lane[:, 0]) < 1e-6:  # perfectly vertical: slope is infinite
        return float(np.pi / 2) if lane[-1, 1] > lane[0, 1] else float(-np.pi / 2)
    slope = np.polyfit(lane[:, 0], lane[:, 1], 1)[0]
    return float(np.arctan(slope))


def official_slots(lanes: list[np.ndarray]) -> tuple[list[np.ndarray | None], int]:
    """Assign up to 4 lanes to slots (0 far-left, 1 ego-left, 2 ego-right,
    3 far-right) with the official UFLD TuSimple rule. Returns (slots, number
    of annotated lanes not placed in a slot)."""
    angles = [slope_angle(lane) for lane in lanes]
    neg = sorted((a, i) for i, a in enumerate(angles) if a is not None and a < 0)
    pos = sorted((a, i) for i, a in enumerate(angles) if a is not None and a > 0)
    slots: list[np.ndarray | None] = [None] * 4
    # left: most negative angle (steepest) = ego-left (slot 1), next = slot 0
    if len(neg) >= 1:
        slots[1] = lanes[neg[0][1]]
    if len(neg) >= 2:
        slots[0] = lanes[neg[1][1]]
    # right: largest angle (steepest) = ego-right (slot 2), next = slot 3
    if len(pos) >= 1:
        slots[2] = lanes[pos[-1][1]]
    if len(pos) >= 2:
        slots[3] = lanes[pos[-2][1]]
    placed = sum(s is not None for s in slots)
    return slots, len(lanes) - placed


def _clip_parts(raw_file: str) -> tuple[str, str, int]:
    """``clips/0313-1/6040/20.jpg`` -> ("0313-1", "6040", 20)."""
    parts = Path(raw_file.replace("\\", "/")).parts
    frame = int(Path(parts[-1]).stem)
    return parts[-3], parts[-2], frame


def _numeric(text: str) -> int | None:
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


class TuSimpleAdapter(LaneDatasetAdapter):
    name = "tusimple"
    num_lanes = 4
    flip_permutation = (3, 2, 1, 0)
    protocol = {
        "split": "official train_set (label_data_*.json) for train/val; validation carved from "
                 "training clips (temporal blocks of consecutive clips per drive + purge gap); "
                 "official test_label.json = test",
        "test": "official test set (not scene-disjoint: shares drives with training, as in the benchmark)",
        "primary_metric": "lane_f1_iou50 (internal, slot lanes, width 30 px scaled to 1280 -> 23 px)",
        "native_metric": "native_tusimple_accuracy / _fp / _fn: official LaneEval.bench (20 px / cos(angle), "
                         "85 % of points, all annotated lanes, predictions sampled at h_samples); "
                         "native_tusimple_f1 = F1 of (1 - FP, 1 - FN) as reported by later papers",
        "annotation": "slots by the official convert_tusimple.py slope rule; lanes < 90 px not placed",
    }

    def __init__(self, root: str | Path, drives: list[str] | None = None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"TuSimple root not found: {self.root}")
        self.drives = set(drives) if drives else None
        self.train_files = sorted(self.root.rglob("label_data_*.json"))
        self.test_files = sorted(self.root.rglob("test_label.json"))
        if not self.train_files:
            raise FileNotFoundError(f"no label_data_*.json below {self.root}")
        if not self.test_files:
            LOGGER.warning("TuSimple: test_label.json not found below %s; the test split will be empty", self.root)
        self._clip_dirs: dict[str, Path] = {}
        self._splits: dict[str, list[FrameRecord]] | None = None
        self.stats: dict[str, dict[str, int]] = {}
        self._size_checked = False

    # -- parsing ---------------------------------------------------------------

    def _load_file(self, path: Path, split: str, stats: Counter) -> list[FrameRecord]:
        base = path.parent
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            raw = item["raw_file"]
            drive, clip, frame = _clip_parts(raw)
            if self.drives is not None and drive not in self.drives:
                continue
            image = self._resolve(base, raw)
            if image is None:
                stats["missing_image"] += 1
                continue
            h_samples = [float(v) for v in item["h_samples"]]
            all_lanes = [lane_points(xs, h_samples) for xs in item["lanes"]]
            lanes = [lane for lane in all_lanes if lane is not None]
            slots, unplaced = official_slots(lanes)
            stats["lanes"] += len(lanes)
            stats["lanes_not_in_slots"] += unplaced
            sequence = f"{split}/{drive}/{clip}"
            self._clip_dirs[sequence] = image.parent
            if not self._size_checked:
                self._check_size(image)
            stats["used"] += 1
            records.append(FrameRecord(
                dataset=self.name, sequence=sequence, frame_id=frame, image_path=image,
                image_size=TUSIMPLE_IMAGE_SIZE, lanes=tuple(slots), slot_known=(True,) * self.num_lanes,
                valid_y_range=(min(h_samples), max(h_samples)), tags=None,
                meta={"split": split, "drive": drive, "clip": clip, "raw_file": raw,
                      "h_samples": h_samples, "gt_lanes_x": [list(map(float, xs)) for xs in item["lanes"]],
                      "eval_lanes": lanes, "split_group": drive},
            ))
        return records

    def _resolve(self, base: Path, raw: str) -> Path | None:
        """``raw_file`` is relative to the split folder; test_label.json is
        usually downloaded separately and placed at the root."""
        for folder in (base, self.root, self.root / "train_set", self.root / "test_set", base.parent):
            candidate = folder / raw
            if candidate.exists():
                return candidate
        return None

    def _check_size(self, image: Path) -> None:
        with Image.open(image) as img:
            if img.size != TUSIMPLE_IMAGE_SIZE:
                raise ValueError(f"TuSimple image {image} is {img.size}, expected {TUSIMPLE_IMAGE_SIZE}")
        self._size_checked = True

    @staticmethod
    def _add_split_order(records: list[FrameRecord]) -> None:
        """Order clips within each drive by clip number, so validation blocks
        and purge gaps are formed from consecutive clips."""
        by_drive: dict[str, list[FrameRecord]] = {}
        for r in records:
            by_drive.setdefault(r.meta["drive"], []).append(r)
        for recs in by_drive.values():
            recs.sort(key=lambda r: (_numeric(r.meta["clip"]) or 0, r.meta["clip"]))
            for rank, r in enumerate(recs):
                r.meta["split_order"] = rank

    def official_splits(self) -> dict[str, list[FrameRecord]]:
        if self._splits is None:
            out = {}
            for split, files in (("train", self.train_files), ("test", self.test_files)):
                stats: Counter = Counter()
                recs = [r for f in files for r in self._load_file(f, split, stats)]
                self._add_split_order(recs)
                out[split] = sorted(recs, key=lambda r: (r.sequence, r.frame_id))
                self.stats[split] = dict(stats)
            out["val"] = []  # carved from train by the experiment (val_strategy)
            self._splits = out
        return self._splits

    # -- interface ---------------------------------------------------------------

    def sequences(self) -> list[str]:
        return sorted({r.sequence for recs in self.official_splits().values() for r in recs})

    def load_sequence(self, sequence: str) -> list[FrameRecord]:
        return [r for recs in self.official_splits().values() for r in recs if r.sequence == sequence]

    def frame_path(self, sequence: str, frame_id: int) -> Path | None:
        if frame_id < 1:
            return None
        if sequence not in self._clip_dirs:
            self.official_splits()
        folder = self._clip_dirs.get(sequence)
        if folder is None:
            return None
        path = folder / f"{frame_id}.jpg"
        return path if path.exists() else None
