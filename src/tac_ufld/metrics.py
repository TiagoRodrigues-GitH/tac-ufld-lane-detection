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
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def __iadd__(self, other: "AnchorCounts") -> "AnchorCounts":
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn
        return self

    def as_metrics(self, prefix: str = "anchor_") -> dict[str, float]:
        p, r, f1 = prf(self.tp, self.fp, self.fn)
        total = self.tp + self.fp + self.fn + self.tn
        return {f"{prefix}precision": p, f"{prefix}recall": r, f"{prefix}f1": f1,
                f"{prefix}accuracy": (self.tp + self.tn) / total if total else 0.0,
                f"{prefix}tp": self.tp, f"{prefix}fp": self.fp, f"{prefix}fn": self.fn, f"{prefix}tn": self.tn}


def anchor_counts(exist_prob: np.ndarray, x_model: np.ndarray, target_cls: np.ndarray,
                  threshold: float, tol_bins: int, img_w: int, griding_num: int) -> AnchorCounts:
    """Arrays may be (A, L) or (B, A, L). Cells with IGNORE_INDEX are skipped."""
    valid = target_cls != IGNORE_INDEX
    gt_pos = valid & (target_cls < griding_num)
    pred_pos = valid & (exist_prob >= threshold)
    close = np.abs(x_to_bin(x_model, img_w, griding_num) - target_cls) <= tol_bins
    both = pred_pos & gt_pos
    wrong = both & ~close
    return AnchorCounts(
        tp=int((both & close).sum()),
        fp=int((pred_pos & ~gt_pos).sum() + wrong.sum()),
        fn=int((~pred_pos & gt_pos).sum() + wrong.sum()),
        tn=int((valid & ~pred_pos & ~gt_pos).sum()),
    )


# ------------------------------------------------------------------ lane level


def culane_line_width(image_w: int, ref_width: float = 30.0, ref_image_w: float = 1640.0) -> int:
    return max(1, int(round(ref_width * image_w / ref_image_w)))


def lane_mask(points: np.ndarray, width: int, height: int, line_width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.asarray(points, dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) >= 2:
        pts = pts[np.argsort(pts[:, 1])]
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
