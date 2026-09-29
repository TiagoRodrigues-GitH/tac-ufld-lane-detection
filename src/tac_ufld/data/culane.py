"""CULane adapter.

Ported from the supervisor's notebook (``locate_culane_list_root``,
``build_culane_file_index``, ``read_lines_txt``, ``load_culane_subset``).
Changes: lanes go to fixed slots by side of the image centre (CULane's
own convention: 0 = far-left, 1 = ego-left, 2 = ego-right, 3 = far-right)
instead of "sort by mean x and keep the first four", and the official
``test.txt`` list is exposed so a true test split exists.

Status: unit-tested on synthetic files only (no CULane copy on the
development machine). Validate on real data before reporting results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

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
    """Place lanes in fixed slots by the side of the image centre at which
    each lane's lowest point lies; closest-to-centre lanes fill ego slots."""
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

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"CULane root not found: {self.root}")
        self.list_root = locate_list_root(self.root)
        self._index = build_image_index(self.root)
        self._splits: dict[str, list[FrameRecord]] | None = None
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

    def _load_list(self, file_name: str, categories: dict[str, str]) -> list[FrameRecord]:
        path = self.list_root / "list" / file_name
        if not path.exists():
            LOGGER.warning("CULane list missing: %s", path)
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rel = line.split()[0].lstrip("/").replace("\\", "/")
            image = self._index.get(rel.lower())
            if image is None:
                continue
            gt = image.with_suffix(".lines.txt")
            if not gt.exists():
                continue
            slots = assign_slots(read_lines_txt(gt), CULANE_IMAGE_SIZE[0], self.num_lanes)
            category = categories.get(rel.lower())
            records.append(FrameRecord(
                dataset=self.name, sequence=str(Path(rel).parent).replace("\\", "/"),
                frame_id=int(image.stem), image_path=image, image_size=CULANE_IMAGE_SIZE,
                lanes=tuple(slots), slot_known=(True,) * self.num_lanes,
                valid_y_range=None, tags={category: True} if category else None,
            ))
        return sorted(records, key=lambda r: (r.sequence, r.frame_id))

    def official_splits(self) -> dict[str, list[FrameRecord]]:
        if self._splits is None:
            categories = self._categories()
            self._splits = {split: self._load_list(name, categories) for split, name in _LISTS.items()}
        return self._splits

    def sequences(self) -> list[str]:
        return sorted({r.sequence for recs in self.official_splits().values() for r in recs})

    def load_sequence(self, sequence: str) -> list[FrameRecord]:
        return [r for recs in self.official_splits().values() for r in recs if r.sequence == sequence]

    def frame_path(self, sequence: str, frame_id: int) -> Path | None:
        if frame_id < 0:
            return None
        return self._index.get(f"{sequence}/{frame_id:05d}.jpg".lower())
