"""Evaluation metrics.

* Anchor level (audit C2 fixed): a (row anchor, lane) cell where both GT
  and prediction say "lane" but the predicted cell is more than
  ``tol_bins`` away from the GT cell is BOTH a false positive (wrong
  location claimed) and a false negative (true location missed), exactly as
  in the supervisor's ``compute_anchor_metrics``. The previous ELAS script
  dropped these cells entirely, which inflated precision and recall.
* Lane level (CULane protocol): each lane is rasterised as a polyline of
  width ``30 * image_width / 1640`` px (CULane's 30 px, scaled to the image
  width), lanes are matched one-to-one with the Hungarian algorithm on
  mask IoU, and a match counts as TP if IoU >= threshold. The primary
  threshold is 0.5 (CULane); others are secondary.
* Pixel level: micro-averaged precision/recall/F1 of the union of drawn
  prediction and GT masks (custom, not comparable with published numbers).
* Temporal: mean |dx| of predicted lane position between consecutive frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from tac_ufld.data.targets import IGNORE_INDEX, x_to_bin


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def f_beta(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


# ---------------------------------------------------------------- anchor level


@dataclass
class AnchorCounts:
    """Additive per-cell counts (sums over frames, so frames can be scored in worker processes).

    ``tp/fp/fn/tn`` follow the convention above (a wrong location is both FP and FN). The split counts
    separate the two error types of a row-anchor detector (added 2026-10-18): ``fp_exist`` = lane claimed where
    there is none (false alarm), ``fn_exist`` = lane missed, ``wrong`` = lane found at the wrong cell, ``both`` =
    cells where GT and prediction both say "lane". ``err_*`` are sums of |x_pred - x_gt| in original pixels over
    the ``both`` cells, split into the far quarter of the annotated band and the rest (near), as in ELAS's
    near/far lane-estimation error (Berriel et al. 2017)."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    fp_exist: int = 0
    fn_exist: int = 0
    wrong: int = 0
    both: int = 0
    err_near: float = 0.0
    n_near: int = 0
    err_far: float = 0.0
    n_far: int = 0

    def __iadd__(self, other: "AnchorCounts") -> "AnchorCounts":
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self

    def as_metrics(self, prefix: str = "anchor_") -> dict[str, float]:
        p, r, f1 = prf(self.tp, self.fp, self.fn)
        total = self.tp + self.fp + self.fn + self.tn
        nan = float("nan")
        n_err = self.n_near + self.n_far
        return {f"{prefix}precision": p, f"{prefix}recall": r, f"{prefix}f1": f1,
                f"{prefix}accuracy": (self.tp + self.tn) / total if total else 0.0,
                f"{prefix}tp": self.tp, f"{prefix}fp": self.fp, f"{prefix}fn": self.fn, f"{prefix}tn": self.tn,
                # existence errors as rates of their own population (lower is better)
                f"{prefix}false_alarm_rate": self.fp_exist / (self.fp_exist + self.tn) if self.fp_exist + self.tn else nan,
                # its population: on ELAS almost every cell holds a lane, so few negatives (report it with the rate)
                f"{prefix}gt_negative_cells": self.fp_exist + self.tn,
                f"{prefix}miss_rate": self.fn_exist / (self.fn_exist + self.both) if self.fn_exist + self.both else nan,
                f"{prefix}wrong_position_rate": self.wrong / self.both if self.both else nan,
                # lateral position error where both say "lane", original pixels (lower is better)
                f"{prefix}mae_px": (self.err_near + self.err_far) / n_err if n_err else nan,
                f"{prefix}mae_near_px": self.err_near / self.n_near if self.n_near else nan,
                f"{prefix}mae_far_px": self.err_far / self.n_far if self.n_far else nan}


def anchor_counts(exist_prob: np.ndarray, x_model: np.ndarray, target_cls: np.ndarray,
                  threshold: float, tol_bins: int, img_w: int, griding_num: int,
                  x_target: np.ndarray | None = None, px_scale: float = 1.0,
                  far_rows: np.ndarray | None = None) -> AnchorCounts:
    """Arrays may be (A, L) or (B, A, L). Cells with IGNORE_INDEX are skipped.

    ``x_target`` (model px, same shape) enables the position errors, scaled to original pixels by ``px_scale``;
    ``far_rows`` (A,) bool marks the anchors of the far quarter of the annotated band."""
    valid = target_cls != IGNORE_INDEX
    gt_pos = valid & (target_cls < griding_num)
    pred_pos = valid & (exist_prob >= threshold)
    close = np.abs(x_to_bin(x_model, img_w, griding_num) - target_cls) <= tol_bins
    both = pred_pos & gt_pos
    wrong = both & ~close
    counts = AnchorCounts(
        tp=int((both & close).sum()),
        fp=int((pred_pos & ~gt_pos).sum() + wrong.sum()),
        fn=int((~pred_pos & gt_pos).sum() + wrong.sum()),
        tn=int((valid & ~pred_pos & ~gt_pos).sum()),
        fp_exist=int((pred_pos & ~gt_pos).sum()), fn_exist=int((~pred_pos & gt_pos).sum()),
        wrong=int(wrong.sum()), both=int(both.sum()),
    )
    if x_target is not None:
        err = np.abs(x_model - x_target) * px_scale
        far = np.zeros(target_cls.shape, dtype=bool)
        if far_rows is not None:  # (A,) -> (..., A, L)
            far = far | np.asarray(far_rows, dtype=bool).reshape(-1, 1)
        counts.err_near, counts.n_near = float(err[both & ~far].sum()), int((both & ~far).sum())
        counts.err_far, counts.n_far = float(err[both & far].sum()), int((both & far).sum())
    return counts


# ------------------------------------------------------------------ lane level


def culane_line_width(image_w: int, ref_width: float = 30.0, ref_image_w: float = 1640.0) -> int:
    return max(1, int(round(ref_width * image_w / ref_image_w)))


def spline_points(pts: np.ndarray, per_segment: int = 50) -> np.ndarray:
    """Dense points on the interpolating cubic spline through ``pts`` (sorted by y), as the CULane evaluation
    does before drawing a lane (official tool ``splineInterpTimes``; the Python port of LaneATT,
    ``culane_metric.interp``: ``splprep(s=0, k=min(3, n-1))`` sampled ``per_segment`` times per segment).
    Falls back to the input points when the spline cannot be fitted (fewer than 2 distinct points)."""
    from scipy.interpolate import splev, splprep

    keep = np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]  # splprep fails on repeated points
    pts = pts[keep]
    if len(pts) < 2:
        return pts
    try:
        tck, u = splprep([pts[:, 0], pts[:, 1]], s=0, k=min(3, len(pts) - 1))
        dense = np.asarray(splev(np.linspace(0.0, 1.0, (len(u) - 1) * per_segment + 1), tck)).T
    except (ValueError, TypeError):
        return pts
    return dense.astype(np.float32)


def lane_mask(points: np.ndarray, width: int, height: int, line_width: int, interp: str = "linear") -> np.ndarray:
    """Lane rasterised as a ``line_width`` polyline. ``interp="spline"``: the points are first replaced by
    their interpolating cubic spline, as in the official CULane evaluation (used for CULane, 2026-10-23);
    ``"linear"``: straight segments between the given points (every other dataset, as before)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.asarray(points, dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) >= 2:
        pts = pts[np.argsort(pts[:, 1], kind="stable")]
        if interp == "spline":
            pts = spline_points(pts)
        cv2.polylines(mask, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, 1,
                      thickness=line_width, lineType=cv2.LINE_8)
    return mask.astype(bool)


def iou_matrix(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float64)
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            union = np.logical_or(pm, gm).sum()
            out[i, j] = np.logical_and(pm, gm).sum() / union if union else 0.0
    return out


@dataclass
class LaneMatch:
    tp: int
    fp: int
    fn: int
    matches: list[tuple[int, int, float]]  # (pred index, gt index, IoU) accepted


def match_lanes(ious: np.ndarray, threshold: float) -> LaneMatch:
    """Hungarian assignment maximising IoU; pairs below ``threshold`` are rejected."""
    n_pred, n_gt = ious.shape
    matches: list[tuple[int, int, float]] = []
    if n_pred and n_gt:
        rows, cols = linear_sum_assignment(-ious)
        matches = [(int(r), int(c), float(ious[r, c])) for r, c in zip(rows, cols) if ious[r, c] >= threshold]
    tp = len(matches)
    return LaneMatch(tp=tp, fp=n_pred - tp, fn=n_gt - tp, matches=matches)


def pixel_counts(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray], shape: tuple[int, int]) -> tuple[int, int, int]:
    pred = np.zeros(shape, dtype=bool)
    gt = np.zeros(shape, dtype=bool)
    for m in pred_masks:
        pred |= m
    for m in gt_masks:
        gt |= m
    return int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())


# -------------------------------------------------------------- temporal level


def temporal_jitter(frames: list[tuple[str, int, np.ndarray, np.ndarray]], threshold: float) -> dict[str, float]:
    """``frames`` = (sequence, frame_id, exist (A, L), x_orig (A, L)). Mean and
    p95 of |x_t - x_{t-1}| (original px) over cells present in both of two
    consecutive frames (frame_id difference 1) of the same sequence."""
    by_seq: dict[str, list] = {}
    for item in frames:
        by_seq.setdefault(item[0], []).append(item)
    diffs = []
    for items in by_seq.values():
        items.sort(key=lambda it: it[1])
        for prev, cur in zip(items, items[1:]):
            if cur[1] - prev[1] != 1:
                continue
            both = (prev[2] >= threshold) & (cur[2] >= threshold)
            diffs.extend(np.abs(cur[3][both] - prev[3][both]).tolist())
    if not diffs:
        return {"jitter_px": float("nan"), "jitter_p95_px": float("nan"), "jitter_pairs": 0}
    arr = np.asarray(diffs)
    return {"jitter_px": float(arr.mean()), "jitter_p95_px": float(np.percentile(arr, 95)),
            "jitter_pairs": int(arr.size)}
