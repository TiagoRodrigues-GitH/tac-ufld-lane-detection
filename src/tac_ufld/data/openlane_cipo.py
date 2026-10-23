"""OpenLane CIPO (closest in-path object) boxes: reader, frame index and training dataset.

Added 2026-10-19 (design: ``docs/CIPO.md``). The annotations come from ``cipo.zip`` of the OpenLane release
(Chen et al. 2022), one JSON per image::

    cipo/{training,validation}/segment-<id>/<timestamp>.jpg.json
    {"raw_file_path": "training/segment-<id>/<timestamp>.jpg",
     "result": [{"x", "y", "width", "height"   (top-left corner and size, pixels of the 1920x1280 image),
                 "id": 1..4                      (importance level; 1 = the CIPO),
                 "trackid": str, "type": 0..4    (unknown, vehicle, pedestrian, sign, cyclist)}, ...]}

What the files really contain (measured 2026-10-19 on the release, sha256 056dfd40...0c77): the list key is
``result`` (the OpenLane README says ``results``), the level is an int (README: str), and about 0.1 % of the boxes
carry the level under ``attribute`` instead of ``id``. ``parse_cipo`` accepts every variant.

``CipoSource`` reads the extracted folder or ``cipo.zip`` itself (no need to unpack 198k small files).
``build_frames`` joins the boxes with the images on disk and, when present, the lane3d_1000 label of the same
frame (ego lanes for the in-path test, camera intrinsics, 3-D lane points for the distance reference).
``CipoDataset`` yields network-size images and CenterNet-style targets (Zhou et al. 2019, "Objects as Points"):
one Gaussian peak per object on the heatmap channel of its importance level, plus the sub-cell centre offset and
the log box size at the peak cell.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tac_ufld.cipo import CameraModel, project_lane_ground
from tac_ufld.config import GeometricAugConfig
from tac_ufld.data.geometric import GeometricAugmenter, apply_h, warp_frames
from tac_ufld.data.openlane import OPENLANE_IMAGE_SIZE, attribute_slots, lane_uv
from tac_ufld.data.transforms import load_frame, normalize

LOGGER = logging.getLogger(__name__)

NUM_LEVELS = 4
OBJECT_TYPES = {0: "unknown", 1: "vehicle", 2: "pedestrian", 3: "sign", 4: "cyclist"}
PATH_TYPES = (1, 2, 4)        # objects that can be in the path; unknown (0) and sign (3) become ignore regions
OPENLANE_FLIP_PERMUTATION = (3, 2, 1, 0)


@dataclass(frozen=True)
class CipoBox:
    x: float            # top-left corner, original pixels
    y: float
    w: float
    h: float
    level: int          # importance level 1..4 (1 = the CIPO)
    type: int           # OBJECT_TYPES
    track_id: str = ""

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.x + self.w, self.y + self.h


def parse_cipo(item: dict) -> list[CipoBox]:
    """One CIPO JSON -> boxes. Boxes with no level in 1..4 or a non-positive size are dropped."""
    out = []
    for b in item.get("result", item.get("results")) or []:
        level = b.get("id", b.get("attribute"))
        try:
            level, w, h = int(level), float(b["width"]), float(b["height"])
            x, y, kind = float(b["x"]), float(b["y"]), int(b.get("type", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if not 1 <= level <= NUM_LEVELS or w <= 0 or h <= 0:
            continue
        out.append(CipoBox(x, y, w, h, level, kind, str(b.get("trackid", ""))))
    return out


def _timestamp(name: str) -> int | None:
    try:
        return int(name.split(".", 1)[0])
    except ValueError:
        return None


class CipoSource:
    """CIPO annotations from the extracted ``cipo/`` folder (or an OpenLane root holding it) or from ``cipo.zip``.

    The zip handle is reopened in every process, so DataLoader workers can read from it."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        self._zip_path: Path | None = None
        self._zip: zipfile.ZipFile | None = None
        self._zip_pid: int | None = None
        self._index: dict[tuple[str, str], dict[int, str]] = {}
        if path.is_file() and path.suffix.lower() == ".zip":
            self._zip_path = path
            for name in self._handle().namelist():
                parts = name.split("/")
                if len(parts) != 4 or parts[0] != "cipo" or not parts[3].endswith(".json"):
                    continue    # also skips the __MACOSX/ entries of the release zip
                ts = _timestamp(parts[3])
                if ts is not None:
                    self._index.setdefault((parts[1], parts[2]), {})[ts] = name
            self.root = path
        elif (path / "cipo").is_dir():
            self.root = path / "cipo"
        elif path.is_dir():
            self.root = path
        else:
            raise FileNotFoundError(f"no CIPO annotations at {path} (expected cipo/ or cipo.zip)")

    def _handle(self) -> zipfile.ZipFile:
        if self._zip is None or self._zip_pid != os.getpid():
            self._zip = zipfile.ZipFile(self._zip_path)
            self._zip_pid = os.getpid()
        return self._zip

    def __getstate__(self) -> dict:   # DataLoader workers (spawn on Windows) reopen the zip themselves
        state = dict(self.__dict__)
        state["_zip"], state["_zip_pid"] = None, None
        return state

    def timestamps(self, split_dir: str, segment: str) -> dict[int, str]:
        """timestamp -> member (zip) or file path (folder) for one segment."""
        key = (split_dir, segment)
        if key not in self._index and self._zip_path is None:
            folder = self.root / split_dir / segment
            found = {}
            if folder.is_dir():
                for p in folder.iterdir():
                    ts = _timestamp(p.name)
                    if ts is not None and p.name.endswith(".json"):
                        found[ts] = str(p)
            self._index[key] = found
        return self._index.get(key, {})

    def read(self, ref: str) -> list[CipoBox]:
        raw = self._handle().read(ref) if self._zip_path is not None else Path(ref).read_bytes()
        return parse_cipo(json.loads(raw))

    def load(self, split_dir: str, segment: str, ts: int) -> list[CipoBox] | None:
        ref = self.timestamps(split_dir, segment).get(ts)
        return None if ref is None else self.read(ref)


@dataclass
class CipoFrame:
    split: str                       # train / val / test
    sequence: str                    # "<split_dir>/<segment>"
    frame_index: int                 # position in the segment's time-ordered list of CIPO frames
    timestamp: int
    image_path: Path
    boxes: tuple[CipoBox, ...]
    lanes: tuple[np.ndarray | None, ...] | None = None              # attribute slots 0..3, original px
    lane_ground: tuple[np.ndarray | None, np.ndarray | None] | None = None   # ego-left/right (v, forward m)
    camera: CameraModel = field(default_factory=CameraModel)
    image_size: tuple[int, int] = OPENLANE_IMAGE_SIZE
    pose: np.ndarray | None = None   # 4x4 vehicle-to-world (lane3d label), for the ego speed

    @property
    def key(self) -> str:
        return f"{self.sequence}/{self.timestamp}"

    def boxes_xyxy(self, levels: tuple[int, ...] | None = None, types: tuple[int, ...] | None = PATH_TYPES
                   ) -> np.ndarray:
        sel = [b.xyxy for b in self.boxes if (levels is None or b.level in levels) and
               (types is None or b.type in types)]
        return np.asarray(sel, dtype=np.float64).reshape(-1, 4)

    def level1_track(self) -> str | None:
        ids = [b.track_id for b in self.boxes if b.level == 1]
        return ids[0] if ids else None


def read_lane_label(path: Path) -> tuple[tuple[np.ndarray | None, ...], tuple[np.ndarray | None, np.ndarray | None],
                                         CameraModel, np.ndarray | None]:
    """lane3d JSON -> (4 attribute slots in pixels, ego boundaries' (v, forward) ground points, camera, pose)."""
    item = json.loads(path.read_text(encoding="utf-8"))
    pose = np.asarray(item["pose"], dtype=np.float64) if item.get("pose") is not None else None
    camera = CameraModel()
    if item.get("intrinsic") is not None and item.get("extrinsic") is not None:
        camera = CameraModel.from_openlane(item["intrinsic"], item["extrinsic"])
    parsed, ground = [], {}
    for lane in item.get("lane_lines", []):
        attribute = int(lane.get("attribute", 0))
        pts = lane_uv(lane, OPENLANE_IMAGE_SIZE)
        if pts is not None:
            parsed.append((attribute, pts))
        if attribute in (2, 3) and "xyz" in lane:
            g = project_lane_ground(np.asarray(lane["xyz"], dtype=np.float64), lane.get("visibility"), camera)
            if g is not None and (attribute not in ground or len(g) > len(ground[attribute])):
                ground[attribute] = g
    slots, _ = attribute_slots(parsed, 4)
    return tuple(slots), (ground.get(2), ground.get(3)), camera, pose


def read_split_manifest(path: str | Path) -> dict[str, str]:
    """``data/split_manifest.csv`` of a lane run -> sequence ("training/segment-...") -> split (train/val/test)."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            prev = out.setdefault(row["sequence"], row["split"])
            if prev != row["split"]:
                raise ValueError(f"{row['sequence']} is in both {prev} and {row['split']} in {path}")
    return out


@dataclass(frozen=True)
class _FrameRef:
    split: str
    sequence: str
    frame_index: int
    timestamp: int
    ref: str
    image: Path


def cap_frames(frames: list, max_frames: int | None) -> list:
    """At most ``max_frames`` items (anything with ``.sequence``, in time order per sequence), spread over the
    sequences: one CONTIGUOUS block from the middle of each sequence (consecutive frames keep the stability
    metric meaningful), an equal share per sequence."""
    if max_frames is None or len(frames) <= max_frames:
        return frames
    by_seq: dict[str, list] = defaultdict(list)
    for f in frames:
        by_seq[f.sequence].append(f)
    per_seq = max(1, max_frames // len(by_seq))
    out = []
    for seq in sorted(by_seq):
        items = by_seq[seq]
        start = max(0, (len(items) - per_seq) // 2)
        out.extend(items[start:start + per_seq])
    return out[:max_frames]


def build_frames(openlane_root: str | Path, source: CipoSource, sequences: dict[str, str],
                 max_frames: dict[str, int | None] | None = None, lane_splits: tuple[str, ...] = ("val", "test"),
                 lane_dir: str = "lane3d_1000") -> tuple[dict[str, list[CipoFrame]], dict[str, int]]:
    """CIPO frames of the given sequences (sequence -> split) whose image is on disk, per split in time order.

    ``max_frames`` caps each split (``cap_frames``) BEFORE any JSON is read. Lane labels (about 100 kB each) are
    read only for ``lane_splits``: training needs the boxes only."""
    root = Path(openlane_root)
    stats: Counter = Counter()
    refs: dict[str, list[_FrameRef]] = defaultdict(list)
    for sequence, split in sorted(sequences.items()):
        split_dir, _, segment = sequence.partition("/")
        stamps = source.timestamps(split_dir, segment)
        if not stamps:
            stats["sequences_without_cipo"] += 1
            continue
        folder = root / "images" / split_dir / segment
        on_disk = set(os.listdir(folder)) if folder.is_dir() else set()   # one listing, not one stat per frame
        for index, ts in enumerate(sorted(stamps)):
            if f"{ts}.jpg" not in on_disk:
                stats["missing_image"] += 1
                continue
            refs[split].append(_FrameRef(split, sequence, index, ts, stamps[ts], folder / f"{ts}.jpg"))
    out: dict[str, list[CipoFrame]] = {}
    for split, items in refs.items():
        stats[f"available_{split}"] = len(items)
        items = cap_frames(items, (max_frames or {}).get(split))
        frames = []
        for r in items:
            boxes = tuple(source.read(r.ref))
            lanes = ground = pose = None
            camera = CameraModel()
            if split in lane_splits:
                split_dir, _, segment = r.sequence.partition("/")
                label = root / lane_dir / split_dir / segment / f"{r.timestamp}.json"
                if label.exists():
                    lanes, ground, camera, pose = read_lane_label(label)
                    stats[f"lane_label_{split}"] += 1
            frames.append(CipoFrame(split, r.sequence, r.frame_index, r.timestamp, r.image, boxes, lanes, ground,
                                    camera, pose=pose))
            stats["boxes"] += len(boxes)
            stats["level1_boxes"] += sum(b.level == 1 for b in boxes)
        stats[f"frames_{split}"] = len(frames)
        out[split] = frames
    return out, dict(stats)


# -- targets ---------------------------------------------------------------------------------------------------------

def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """CornerNet / CenterNet radius: a corner shift by this much keeps IoU >= ``min_overlap`` (smallest of three
    cases)."""
    a1, b1 = 1.0, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2 = 4.0, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, radius: int) -> None:
    """Max-merge a Gaussian peak (1.0 at (cx, cy), sigma = diameter / 6) into ``heatmap`` (H, W) in place."""
    d = 2 * radius + 1
    sigma = d / 6.0
    ys, xs = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))
    h, w = heatmap.shape
    left, right = min(cx, radius), min(w - cx, radius + 1)
    top, bottom = min(cy, radius), min(h - cy, radius + 1)
    if right <= 0 or bottom <= 0 or left < 0 or top < 0:
        return
    region = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    patch = g[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(region, patch, out=region)


def encode_targets(boxes: np.ndarray, levels: np.ndarray, out_w: int, out_h: int, stride: int,
                   ignore: np.ndarray | None = None, num_levels: int = NUM_LEVELS, max_objects: int = 256
                   ) -> dict[str, np.ndarray]:
    """Model-pixel xyxy boxes and their levels -> CenterNet targets at the head's resolution.

    ``heat`` (L, h, w) Gaussian peaks; ``weight`` (h, w) = 0 inside ``ignore`` boxes (their negatives are not
    penalised); ``ind`` / ``mask`` / ``offset`` / ``log_size`` (max_objects, ...) at each object's centre cell.
    ``offset`` is the centre's position inside its cell in [0, 1); ``log_size`` is log(w, h) in cells."""
    heat = np.zeros((num_levels, out_h, out_w), dtype=np.float32)
    weight = np.ones((out_h, out_w), dtype=np.float32)
    ind = np.zeros(max_objects, dtype=np.int64)
    mask = np.zeros(max_objects, dtype=np.float32)
    offset = np.zeros((max_objects, 2), dtype=np.float32)
    log_size = np.zeros((max_objects, 2), dtype=np.float32)
    if ignore is not None:
        for x1, y1, x2, y2 in np.asarray(ignore, dtype=np.float64).reshape(-1, 4) / stride:
            weight[max(0, int(y1)):max(0, int(math.ceil(y2))), max(0, int(x1)):max(0, int(math.ceil(x2)))] = 0.0
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    k = 0
    for (x1, y1, x2, y2), level in zip(boxes, levels):
        w, h = (x2 - x1) / stride, (y2 - y1) / stride
        if w <= 0 or h <= 0:
            continue
        cx, cy = (x1 + x2) / 2 / stride, (y1 + y2) / 2 / stride
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < out_w and 0 <= iy < out_h):
            continue
        radius = max(0, int(gaussian_radius(math.ceil(h), math.ceil(w))))
        draw_gaussian(heat[int(level) - 1], ix, iy, radius)
        weight[iy, ix] = 1.0    # an object's own centre is never ignored
        if k < max_objects:
            ind[k], mask[k] = iy * out_w + ix, 1.0
            offset[k] = cx - ix, cy - iy
            log_size[k] = math.log(max(w, 1e-3)), math.log(max(h, 1e-3))
            k += 1
    return {"heat": heat, "weight": weight, "ind": ind, "mask": mask, "offset": offset, "log_size": log_size}


def warp_boxes(boxes: np.ndarray, h: np.ndarray, img_w: int, img_h: int, min_kept: float = 0.3
               ) -> tuple[np.ndarray, np.ndarray]:
    """xyxy boxes through a homography: axis-aligned box of the four warped corners, clipped to the image.
    Returns (boxes, keep); a box keeps only if >= ``min_kept`` of its warped area stays inside the image."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return boxes, np.zeros(0, dtype=bool)
    corners = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], axis=1)
    warped = apply_h(h, corners.reshape(-1, 2)).reshape(-1, 4, 2)
    full = np.concatenate([warped.min(axis=1), warped.max(axis=1)], axis=1)
    clipped = full.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, img_w)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, img_h)
    area = lambda b: np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)  # noqa: E731
    keep = area(clipped) >= min_kept * np.maximum(area(full), 1e-9)
    keep &= area(clipped) > 0
    return clipped, keep


class CipoDataset(Dataset):
    """Network-size image + CenterNet targets for one CIPO frame. Path objects (vehicle, pedestrian, cyclist) are
    targets on the channel of their level; unknown objects and signs are ignore regions."""

    def __init__(self, frames: list[CipoFrame], img_w: int, img_h: int, stride: int = 8,
                 normalize_mode: str = "imagenet", augment: GeometricAugConfig | None = None,
                 max_objects: int = 256, jpeg_draft: bool = True) -> None:
        if img_w % stride or img_h % stride:
            raise ValueError(f"input {img_w}x{img_h} is not a multiple of the head stride {stride}")
        self.frames, self.img_w, self.img_h, self.stride = frames, img_w, img_h, stride
        self.normalize_mode, self.max_objects, self.jpeg_draft = normalize_mode, max_objects, jpeg_draft
        self.augmenter = (GeometricAugmenter(augment, OPENLANE_FLIP_PERMUTATION)
                          if augment is not None and augment.enabled else None)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        frame = self.frames[idx]
        image = load_frame(str(frame.image_path), self.img_w, self.img_h, draft=self.jpeg_draft)
        ow, oh = frame.image_size
        scale = np.array([self.img_w / ow, self.img_h / oh] * 2)
        path = [b for b in frame.boxes if b.type in PATH_TYPES]
        other = [b for b in frame.boxes if b.type not in PATH_TYPES]
        boxes = np.asarray([b.xyxy for b in path], dtype=np.float64).reshape(-1, 4) * scale
        levels = np.asarray([b.level for b in path], dtype=np.int64)
        ignore = np.asarray([b.xyxy for b in other], dtype=np.float64).reshape(-1, 4) * scale
        if self.augmenter is not None:
            sampled = self.augmenter.sample(self.img_w, self.img_h)
            if sampled is not None:
                h, _ = sampled
                image = warp_frames(image[None], h)[0]
                boxes, keep = warp_boxes(boxes, h, self.img_w, self.img_h)
                boxes, levels = boxes[keep], levels[keep]
                ignore, keep_i = warp_boxes(ignore, h, self.img_w, self.img_h)
                ignore = ignore[keep_i]
        targets = encode_targets(boxes, levels, self.img_w // self.stride, self.img_h // self.stride, self.stride,
                                 ignore=ignore, max_objects=self.max_objects)
        out = {k: torch.from_numpy(v) for k, v in targets.items()}
        out["image"] = normalize(image, self.normalize_mode)
        out["index"] = torch.tensor(idx)
        return out


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
