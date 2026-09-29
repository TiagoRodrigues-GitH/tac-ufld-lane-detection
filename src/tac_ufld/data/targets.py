"""Row-anchor target encoding (UFLD formulation).

For every (row anchor, lane slot) the target is one of ``griding_num + 1``
classes: a horizontal cell index ``0..G-1`` or ``G`` = "no lane", exactly as
in Ultra-Fast-Lane-Detection. ``IGNORE_INDEX`` marks cells without a usable
annotation (outside the annotated ROI, or an "unknown" slot) so the loss and
the metrics skip them instead of learning "no lane" there.

Layout follows the official code: targets are (A, L) = (anchors, lanes);
logits are (B, G + 1, A, L).

Cell encoding is symmetric with decoding: ``bin = round(x / (W-1) * (G-1))``
and ``x = bin * (W-1) / (G-1)`` (model pixels), i.e. cell centres at
``linspace(0, W-1, G)`` as in the official ``col_sample``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tac_ufld.data.types import FrameRecord

IGNORE_INDEX = -100


def make_row_anchors(img_h: int, num_anchors: int, range_frac: tuple[float, float] | list[float]) -> np.ndarray:
    """Row anchors in model pixels, ordered top (far) to bottom (near)."""
    lo, hi = range_frac
    return np.linspace(lo * (img_h - 1), hi * (img_h - 1), num_anchors).astype(np.float32)


def x_to_bin(x_model: np.ndarray | float, img_w: int, griding_num: int) -> np.ndarray:
    return np.clip(np.round(np.asarray(x_model) / (img_w - 1) * (griding_num - 1)), 0, griding_num - 1).astype(np.int64)


def bin_to_x(bins: np.ndarray | float, img_w: int, griding_num: int) -> np.ndarray:
    return np.asarray(bins, dtype=np.float32) * (img_w - 1) / (griding_num - 1)


def interpolate_x(lane: np.ndarray, y_query: np.ndarray) -> np.ndarray:
    """x of a polyline at each query row; NaN outside the lane's vertical extent."""
    pts = lane[np.argsort(lane[:, 1])]
    ys, xs = pts[:, 1], pts[:, 0]
    out = np.interp(y_query, ys, xs)
    out[(y_query < ys[0]) | (y_query > ys[-1])] = np.nan
    return out.astype(np.float32)


@dataclass
class LaneTargets:
    cls: np.ndarray    # (A, L) int64: 0..G-1 cell, G no lane, IGNORE_INDEX ignored
    exist: np.ndarray  # (A, L) float32 1 where a lane is present
    x: np.ndarray      # (A, L) float32 x in model pixels (0 where no lane)
    valid: np.ndarray  # (A, L) bool   cls != IGNORE_INDEX


def encode_targets(
    record: FrameRecord, img_w: int, img_h: int, row_anchors: np.ndarray, griding_num: int
) -> LaneTargets:
    orig_w, orig_h = record.image_size
    num_anchors, num_lanes = len(row_anchors), record.num_slots
    y_orig = row_anchors * (orig_h / img_h)
    if record.valid_y_range is None:
        in_range = np.ones(num_anchors, dtype=bool)
    else:
        y0, y1 = record.valid_y_range
        in_range = (y_orig >= y0 - 1e-3) & (y_orig <= y1 + 1e-3)

    cls = np.full((num_anchors, num_lanes), IGNORE_INDEX, dtype=np.int64)
    exist = np.zeros((num_anchors, num_lanes), dtype=np.float32)
    x = np.zeros((num_anchors, num_lanes), dtype=np.float32)

    for slot, lane in enumerate(record.lanes):
        if not record.slot_known[slot]:
            continue
        cls[in_range, slot] = griding_num
        if lane is None:
            continue
        x_orig = interpolate_x(lane, y_orig)
        present = in_range & np.isfinite(x_orig) & (x_orig >= 0) & (x_orig < orig_w)
        x_model = np.clip(np.nan_to_num(x_orig) * (img_w / orig_w), 0, img_w - 1)
        cls[present, slot] = x_to_bin(x_model[present], img_w, griding_num)
        exist[present, slot] = 1.0
        x[present, slot] = x_model[present]
    return LaneTargets(cls=cls, exist=exist, x=x, valid=cls != IGNORE_INDEX)
