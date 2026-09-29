"""Geometric augmentation with consistent lane labels.

A sampled transform is ONE 3x3 homography ``H`` in model-pixel coordinates
(the resized image). It is applied to every frame of a clip (so the motion
between frames is preserved) and to the lane polylines, and the row-anchor
targets are re-encoded from the transformed lanes:

* translation, isotropic scaling and small rotations about the image centre,
  random crop + resize back (a zoom with offset), perspective jitter of the
  four corners, optional horizontal flip;
* homographies map straight segments to straight segments, so transforming
  the annotated polyline vertices is exact;
* a lane that leaves the image at a row becomes "no lane" there, exactly
  like the unaugmented encoder treats lanes outside the image;
* a row is only labelled "no lane" if the visible content of that row lies
  inside the annotated band of the source image (ELAS ROI). Rows that show
  unannotated content (e.g. above the ROI after a rotation) are ignored;
* a lane whose transformed points are no longer ordered in y (possible for
  near-horizontal lanes under rotation/perspective) is ignored in that sample;
* a horizontal flip mirrors left and right, so lane slots are permuted with
  the dataset's ``flip_permutation`` (ELAS: ego-left <-> ego-right).
  Datasets without a valid permutation cannot use flipping.

With ``H = I`` the encoder returns exactly ``encode_targets`` (tested).
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch

from tac_ufld.config import GeometricAugConfig
from tac_ufld.data.targets import IGNORE_INDEX, LaneTargets, interpolate_x, x_to_bin
from tac_ufld.data.types import FrameRecord

_BORDERS = {"constant": cv2.BORDER_CONSTANT, "replicate": cv2.BORDER_REPLICATE, "reflect": cv2.BORDER_REFLECT_101}
_ROW_SAMPLES = 64


def _translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


def _scale(s: float) -> np.ndarray:
    return np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)


def _rotate(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(N, 2) points through a homography."""
    homog = np.concatenate([pts.astype(np.float64), np.ones((len(pts), 1))], axis=1) @ h.T
    return (homog[:, :2] / homog[:, 2:3]).astype(np.float64)


class GeometricAugmenter:
    """Samples a homography per clip using Python's ``random`` (seeded per
    DataLoader worker), so runs are reproducible for a fixed seed."""

    def __init__(self, cfg: GeometricAugConfig, flip_permutation: tuple[int, ...] | None) -> None:
        if cfg.hflip_prob > 0 and flip_permutation is None:
            from tac_ufld.config import ConfigError

            raise ConfigError("geometric.hflip_prob > 0 but this dataset defines no left/right slot "
                              "permutation, so flipped labels would be wrong")
        self.cfg = cfg
        self.flip_permutation = flip_permutation

    def sample(self, width: int, height: int) -> tuple[np.ndarray, bool] | None:
        """Return (H, flipped) or None when no transform is applied."""
        cfg = self.cfg
        if not cfg.enabled or random.random() >= cfg.prob:
            return None
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        s = random.uniform(*cfg.scale)
        deg = random.uniform(-cfg.rotate_deg, cfg.rotate_deg)
        tx = random.uniform(-cfg.translate_x, cfg.translate_x) * width
        ty = random.uniform(-cfg.translate_y, cfg.translate_y) * height
        h = _translate(cx + tx, cy + ty) @ _rotate(deg) @ _scale(s) @ _translate(-cx, -cy)
        crop = random.uniform(*cfg.crop_scale)
        if crop < 1.0:  # crop a (crop*W, crop*H) window and resize it back
            x0 = random.uniform(0, (1 - crop) * (width - 1))
            y0 = random.uniform(0, (1 - crop) * (height - 1))
            h = _scale(1.0 / crop) @ _translate(-x0, -y0) @ h
        if cfg.perspective > 0:
            src = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
            jitter = np.array([[random.uniform(-1, 1) * cfg.perspective * width,
                                random.uniform(-1, 1) * cfg.perspective * height] for _ in range(4)],
                              dtype=np.float32)
            h = cv2.getPerspectiveTransform(src, src + jitter).astype(np.float64) @ h
        flipped = cfg.hflip_prob > 0 and random.random() < cfg.hflip_prob
        if flipped:
            h = np.array([[-1, 0, width - 1], [0, 1, 0], [0, 0, 1]], dtype=np.float64) @ h
        return h, flipped


def warp_frames(frames: torch.Tensor, h: np.ndarray, border: str = "constant") -> torch.Tensor:
    """(T, C, H, W) float frames -> the same frames warped by ``h`` (bilinear)."""
    t, c, height, width = frames.shape
    out = []
    for frame in frames:
        arr = frame.permute(1, 2, 0).contiguous().numpy()
        warped = cv2.warpPerspective(arr, h, (width, height), flags=cv2.INTER_LINEAR,
                                     borderMode=_BORDERS[border], borderValue=0)
        if warped.ndim == 2:
            warped = warped[..., None]
        out.append(torch.from_numpy(np.ascontiguousarray(warped.transpose(2, 0, 1))))
    return torch.stack(out)


def encode_targets_warped(
    record: FrameRecord, h: np.ndarray, img_w: int, img_h: int, row_anchors: np.ndarray,
    griding_num: int, flipped: bool = False, flip_permutation: tuple[int, ...] | None = None,
    min_row_valid: float = 1.0,
) -> LaneTargets:
    """Row-anchor targets of ``record`` after the model-space homography ``h``."""
    orig_w, orig_h = record.image_size
    sx, sy = img_w / orig_w, img_h / orig_h
    h_inv = np.linalg.inv(h)
    num_anchors, num_lanes = len(row_anchors), record.num_slots

    lanes, known = list(record.lanes), list(record.slot_known)
    if flipped:
        if flip_permutation is None:
            raise ValueError("flipped sample without a slot permutation")
        lanes = [record.lanes[p] for p in flip_permutation]
        known = [record.slot_known[p] for p in flip_permutation]

    def inside_source(pts_aug: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(visible in the source image, inside the annotated band)."""
        src = apply_h(h_inv, pts_aug)
        # half-open [0, W) x [0, H), the same convention as encode_targets (x_orig < orig_w)
        visible = (src[:, 0] >= -1e-6) & (src[:, 0] < img_w) & (src[:, 1] >= -1e-6) & (src[:, 1] < img_h)
        if record.valid_y_range is None:
            return visible, visible
        y_orig = src[:, 1] / sy
        y0, y1 = record.valid_y_range
        return visible, visible & (y_orig >= y0 - 1e-3) & (y_orig <= y1 + 1e-3)

    # A row can carry "no lane" labels only if all its visible content is annotated.
    xs = np.linspace(0, img_w - 1, _ROW_SAMPLES)
    row_valid = np.zeros(num_anchors, dtype=bool)
    for a, y in enumerate(row_anchors):
        visible, annotated = inside_source(np.stack([xs, np.full_like(xs, y)], axis=1))
        n_visible = visible.sum()
        row_valid[a] = n_visible == 0 or annotated.sum() / n_visible >= min_row_valid - 1e-9

    cls = np.full((num_anchors, num_lanes), IGNORE_INDEX, dtype=np.int64)
    exist = np.zeros((num_anchors, num_lanes), dtype=np.float32)
    x = np.zeros((num_anchors, num_lanes), dtype=np.float32)
    for slot, lane in enumerate(lanes):
        if not known[slot]:
            continue
        cls[row_valid, slot] = griding_num
        if lane is None:
            continue
        pts = lane[np.argsort(lane[:, 1])].astype(np.float64) * [sx, sy]
        aug = apply_h(h, pts)
        dy = np.diff(aug[:, 1])
        if not (np.all(dy > 0) or np.all(dy < 0)):
            cls[:, slot] = IGNORE_INDEX  # lane folded over itself: label undefined
            continue
        x_at = interpolate_x(aug.astype(np.float32), row_anchors.astype(np.float32))
        finite = np.isfinite(x_at)
        present = np.zeros(num_anchors, dtype=bool)
        if finite.any():
            idx = np.flatnonzero(finite)
            _, annotated = inside_source(np.stack([x_at[idx], row_anchors[idx]], axis=1))
            present[idx] = annotated & (x_at[idx] >= 0) & (x_at[idx] < img_w)
        x_model = np.clip(np.nan_to_num(x_at), 0, img_w - 1)
        cls[present, slot] = x_to_bin(x_model[present], img_w, griding_num)
        exist[present, slot] = 1.0
        x[present, slot] = x_model[present]
    return LaneTargets(cls=cls, exist=exist, x=x, valid=cls != IGNORE_INDEX)


def transform_lanes(record: FrameRecord, h: np.ndarray, img_w: int, img_h: int) -> list[np.ndarray | None]:
    """Lane polylines in augmented model pixels (for visual checks and tests)."""
    orig_w, orig_h = record.image_size
    scale = np.array([img_w / orig_w, img_h / orig_h])
    return [None if lane is None else apply_h(h, lane.astype(np.float64) * scale) for lane in record.lanes]
