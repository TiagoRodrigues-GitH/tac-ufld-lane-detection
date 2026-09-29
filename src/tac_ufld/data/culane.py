"""CULane adapter.

Ported from the supervisor's notebook (``locate_culane_list_root``,
``build_culane_file_index``, ``read_lines_txt``, ``load_culane_subset``).

On-disk layout (CULane release):

* ``driver_23_30frame``, ``driver_161_90frame``, ``driver_182_30frame``
  (train + val) and ``driver_37_30frame``, ``driver_100_30frame``,
  ``driver_193_90frame`` (test); inside, ``<clip>.MP4/<frame>.jpg`` with a
  ``<frame>.lines.txt`` per image (one lane per line, ``x y x y ...``);
* ``laneseg_label_w16/`` - segmentation labels (lane ids 1..4) for train/val;
* ``list/train_gt.txt`` / ``list/val_gt.txt`` - ``image seg_label e1 e2 e3 e4``
  (e_i = existence of slot i); ``list/test.txt`` - image only;
  ``list/test_split/test<k>_<category>.txt`` - the nine test categories.

Validation against that layout (see docs/DATASETS.md):

1. **Lane slots.** The official UFLD code takes a lane's slot from the
   segmentation label at the lane's middle point (UFLD v2
   ``scripts/cache_dataset.py``). This adapter does the same whenever the
   ``laneseg_label_w16`` image exists, and cross-checks the result with the
   list's existence flags. Without a label (the test split has none) it
   falls back to the geometric rule (side of the image centre at the lane's
   lowest point; closest lanes fill the ego slots). Evaluation always uses
   every annotated lane (``meta["eval_lanes"]``), as the official evaluator.
2. **Frame stride.** Clip folders store every 30th (``*_30frame``) or every
   90th (``*_90frame``) video frame, and frame ids are video-frame numbers.
   A ``temporal_step`` of 30 therefore finds no history in the 90-frame
   drivers. The supervisor's notebook used 90 (3 s at 30 fps), which exists
   in both folder types; ``configs/culane.yaml`` now uses 90.
   ``stride_report`` measures the stride of every clip.

Status: tested on synthetic files that reproduce the layout above. No CULane
copy was available during development; run ``python -m tac_ufld
validate-dataset --dataset culane`` on the real data before reporting.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from tac_ufld.data.base import LaneDatasetAdapter
from tac_ufld.data.types import FrameRecord

LOGGER = logging.getLogger(__name__)

CULANE_IMAGE_SIZE = (1640, 590)
_LISTS = {"train": "train_gt.txt", "val": "val_gt.txt", "test": "test.txt"}


def read_lines_txt(path: str | Path) -> list[np.ndarray]:
    """Parse a CULane ``.lines.txt`` file: one lane per line, ``x y x y ...``."""
    path = Path(path)
    if not path.exists():
        return []
    lanes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        nums = [float(v) for v in line.split()]
        if len(nums) < 4:
            continue
        pts = np.asarray(nums[: len(nums) // 2 * 2], dtype=np.float32).reshape(-1, 2)
        pts = pts[np.isfinite(pts).all(axis=1) & (pts[:, 0] >= 0) & (pts[:, 1] >= 0)]
        if len(pts) >= 2:
            lanes.append(pts)
    return lanes


def assign_slots(lanes: list[np.ndarray], image_w: int, num_lanes: int = 4) -> list[np.ndarray | None]:
    """Geometric fallback: place lanes in fixed slots by the side of the image
    centre at which each lane's lowest point lies; closest-to-centre lanes
    fill the ego slots."""
    left, right = [], []
    for lane in lanes:
        bottom_x = float(lane[np.argmax(lane[:, 1]), 0])
        (left if bottom_x < image_w / 2 else right).append((bottom_x, lane))
    left.sort(key=lambda item: -item[0])
    right.sort(key=lambda item: item[0])
    half = num_lanes // 2
    slots: list[np.ndarray | None] = [None] * num_lanes
    for i, (_, lane) in enumerate(left[:half]):
        slots[half - 1 - i] = lane
    for i, (_, lane) in enumerate(right[: num_lanes - half]):
        slots[half + i] = lane
    return slots


def seg_label_slots(lanes: list[np.ndarray], label: np.ndarray, num_lanes: int = 4
                    ) -> tuple[list[np.ndarray | None], int]:
    """Official rule (UFLD v2 ``cache_dataset.py``): the slot of a lane is the
    segmentation-label value at the lane's middle point (1..4 -> slot 0..3).
    Lanes whose middle point has label 0 are skipped, as in the official code.
    Returns (slots, number of skipped lanes)."""
    slots: list[np.ndarray | None] = [None] * num_lanes
    skipped = 0
    h, w = label.shape[:2]
    for lane in lanes:
        mid = len(lane) // 2
        x, y = int(lane[mid, 0]), int(lane[mid, 1])
        yy, xx = min(max(y - 1, 0), h - 1), min(max(x - 1, 0), w - 1)
        order = int(label[yy, xx])
        if 1 <= order <= num_lanes and slots[order - 1] is None:
            slots[order - 1] = lane
        else:
            skipped += 1
    return slots, skipped


def locate_list_root(base: Path) -> Path:
    for candidate in [base, *[p for p in base.rglob("*") if p.is_dir()]]:
        if (candidate / "list" / "train_gt.txt").exists() and (candidate / "list" / "val_gt.txt").exists():
            return candidate
    raise FileNotFoundError(f"No list/train_gt.txt + list/val_gt.txt below {base}")


def build_image_index(root: Path) -> dict[str, Path]:
    """Map ``driver_xx/clip/00000.jpg`` (lower-case) to its path, ignoring
    wrapper folders created by archive extraction."""
    index = {}
    for path in root.rglob("*.jpg"):
        parts = list(path.parts)
        drivers = [i for i, p in enumerate(parts) if p.lower().startswith("driver_")]
        if drivers:
            index["/".join(parts[drivers[-1]:]).lower()] = path
    return index


class CULaneAdapter(LaneDatasetAdapter):
    name = "culane"
    num_lanes = 4
    flip_permutation = (3, 2, 1, 0)
    protocol = {
        "split": "official lists: list/train_gt.txt (train), list/val_gt.txt (val), list/test.txt (test)",
        "test": "official test list (drivers 37, 100, 193); per-category tags from list/test_split",
        "primary_metric": "lane_f1_iou50 (internal: polylines of the slot lanes, width 30 px at 1640)",
        "native_metric": "native_culane_f1_iou50: every annotated lane, 30 px lines at 1640x590, "
                         "IoU >= 0.5, Hungarian matching. The official evaluator draws cubic-spline "
                         "interpolated lanes; this implementation draws polylines (close, not identical). "
                         "Category 'cross' has no lanes: only FP is meaningful there.",
        "annotation": "slots from laneseg_label_w16 lane ids (official); geometric fallback without labels",
    }

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"CULane root not found: {self.root}")
        self.list_root = locate_list_root(self.root)
        self._index = build_image_index(self.root)
        self._splits: dict[str, list[FrameRecord]] | None = None
        self.stats: dict[str, dict[str, int]] = {}
        LOGGER.info("CULane: %d images indexed", len(self._index))

    def _categories(self) -> dict[str, str]:
        out = {}
        split_dir = self.list_root / "list" / "test_split"
        for path in sorted(split_dir.glob("test*_*.txt")) if split_dir.exists() else []:
            category = path.stem.split("_", 1)[1]
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out[line.split()[0].lstrip("/").lower()] = category
        return out

    def _seg_label(self, rel: str | None) -> np.ndarray | None:
        if not rel:
            return None
        path = self.list_root / rel.lstrip("/")
        if not path.exists():
            return None
        with Image.open(path) as img:
            return np.asarray(img)

    def _load_list(self, split: str, file_name: str, categories: dict[str, str]) -> list[FrameRecord]:
        path = self.list_root / "list" / file_name
        stats = Counter()
        if not path.exists():
            LOGGER.warning("CULane list missing: %s", path)
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            rel = parts[0].lstrip("/").replace("\\", "/")
            image = self._index.get(rel.lower())
            if image is None:
                stats["missing_image"] += 1
                continue
            gt = image.with_suffix(".lines.txt")
            if not gt.exists():
                stats["missing_lines_txt"] += 1
                continue
            lanes = read_lines_txt(gt)
            flags = [int(v) for v in parts[2:2 + self.num_lanes]] if len(parts) >= 2 + self.num_lanes else None
            label = self._seg_label(parts[1]) if len(parts) > 1 else None
            if label is not None:
                slots, skipped = seg_label_slots(lanes, label, self.num_lanes)
                stats["slots_from_seg_label"] += 1
                stats["lanes_skipped_label0"] += skipped
            else:
                slots = assign_slots(lanes, CULANE_IMAGE_SIZE[0], self.num_lanes)
                stats["slots_geometric"] += 1
            if flags is not None:
                stats["flag_mismatch"] += int(any((s is not None) != bool(f) for s, f in zip(slots, flags)))
            category = categories.get(rel.lower())
            stats["used"] += 1
            records.append(FrameRecord(
                dataset=self.name, sequence=str(Path(rel).parent).replace("\\", "/"),
                frame_id=int(image.stem), image_path=image, image_size=CULANE_IMAGE_SIZE,
                lanes=tuple(slots), slot_known=(True,) * self.num_lanes,
                valid_y_range=None, tags={category: True} if category else None,
                meta={"split": split, "eval_lanes": lanes, "existence_flags": flags,
                      "category": category},
            ))
        self.stats[split] = dict(stats)
        if stats["flag_mismatch"]:
            LOGGER.warning("CULane %s: %d frames where slots disagree with the list existence flags",
                           split, stats["flag_mismatch"])
        return sorted(records, key=lambda r: (r.sequence, r.frame_id))

    def official_splits(self) -> dict[str, list[FrameRecord]]:
        if self._splits is None:
            categories = self._categories()
            self._splits = {split: self._load_list(split, name, categories) for split, name in _LISTS.items()}
        return self._splits

    def sequences(self) -> list[str]:
        return sorted({r.sequence for recs in self.official_splits().values() for r in recs})

    def load_sequence(self, sequence: str) -> list[FrameRecord]:
        return [r for recs in self.official_splits().values() for r in recs if r.sequence == sequence]

    def frame_path(self, sequence: str, frame_id: int) -> Path | None:
        if frame_id < 0:
            return None
        return self._index.get(f"{sequence}/{frame_id:05d}.jpg".lower())

    def stride_report(self) -> dict[str, int]:
        """Stored-frame stride of every clip (median id difference): 30 for
        ``*_30frame`` drivers, 90 for ``*_90frame`` drivers."""
        by_seq: dict[str, list[int]] = {}
        for key in self._index:
            seq, _, name = key.rpartition("/")
            stem = name.split(".")[0]
            if stem.isdigit():
                by_seq.setdefault(seq, []).append(int(stem))
        out = {}
        for seq, ids in by_seq.items():
            ids.sort()
            if len(ids) > 1:
                out[seq] = int(np.median(np.diff(ids)))
        return out
