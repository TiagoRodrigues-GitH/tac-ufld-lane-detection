"""OpenLane-V evaluation protocol of RVLD (Jin et al., ICCV 2023), ported from the authors' code.

Sources: ``Modeling/OpenLane-V/PLD/code/evaluation/evaluate_iou_laneatt.py`` and ``evaluate_temporal.py`` in
https://github.com/dongkwonjin/RVLD, whose IoU code comes from LaneATT (Tabelini et al., MIT licence). Kept
identical so that our numbers sit in the same table as RVLD, LaneTCA, OMR, CLRNet and GANet on OpenLane-V:

* lanes in 1920x1280 pixels are rescaled to 960x640 (``x / 1919 * 959``, ``y / 1279 * 639``) and lanes with
  fewer than two points are dropped;
* every lane is interpolated with a parametric spline (``scipy.interpolate.splprep``, ``s=0``, five samples
  per input segment) and drawn as a 30 px polyline (``cv2.line`` between consecutive points);
* the IoU of the binary masks is matched one-to-one with the Hungarian algorithm on ``1 - IoU``; a prediction
  is correct if its matched IoU is ABOVE the threshold (0.5 and 0.8); F1 comes from TP/FP/FN summed over frames;
* mIoU is the mean matched IoU over every predicted lane (an unmatched prediction counts as 0);
* video metrics: annotated lanes of consecutive frames of a video are matched (IoU > 0.5); for every matched
  pair, the detection is *stable* if it is correct in both frames, *flickering* if correct in exactly one and
  *missing* if correct in neither. R_F and R_M are the flickering and missing shares (RVLD Table 3).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.optimize import linear_sum_assignment

ORIGINAL_SIZE = (1920, 1280)
EVAL_SIZE = (960, 640)
LINE_WIDTH = 30
THRESHOLDS = (0.5, 0.8)
PREFIX = "native_openlanev"
# per-frame count vector: TP, FP, FN for each threshold, then the sum of matched IoU and the number of predictions
_N = 3 * len(THRESHOLDS) + 2


def to_eval_scale(lane: np.ndarray) -> np.ndarray:
    pts = np.asarray(lane, dtype=np.float64)[:, :2].copy()
    pts[:, 0] = pts[:, 0] / (ORIGINAL_SIZE[0] - 1) * (EVAL_SIZE[0] - 1)
    pts[:, 1] = pts[:, 1] / (ORIGINAL_SIZE[1] - 1) * (EVAL_SIZE[1] - 1)
    return pts


def interp(points: list[tuple[float, float]], n: int = 5) -> np.ndarray:
    """LaneATT's ``interp``: parametric spline through the points, ``n`` samples per segment."""
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))
    u = np.linspace(0.0, 1.0, num=(len(u) - 1) * n + 1)
    return np.array(splev(u, tck)).T


def _prepare(lanes: list[np.ndarray]) -> list[np.ndarray]:
    """Rescale, drop lanes with < 2 points, remove repeated points (``dict.fromkeys``) and interpolate."""
    out = []
    for lane in lanes:
        if lane is None or len(lane) < 2:
            continue
        pts = list(dict.fromkeys(map(tuple, to_eval_scale(lane).tolist())))
        if len(pts) >= 2:
            out.append(interp(pts))
    return out


def _mask(lane: np.ndarray) -> np.ndarray:
    img = np.zeros((EVAL_SIZE[1], EVAL_SIZE[0]), dtype=np.uint8)
    pts = lane.astype(np.int32)
    for p1, p2 in zip(pts[:-1], pts[1:]):
        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color=255, thickness=LINE_WIDTH)
    return img > 0


def cross_iou(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
    """Discrete IoU between every pair of (prepared) lanes; one channel instead of LaneATT's three gives the
    same ratio."""
    ma, mb = [_mask(lane) for lane in a], [_mask(lane) for lane in b]
    ious = np.zeros((len(ma), len(mb)))
    for i, x in enumerate(ma):
        for j, y in enumerate(mb):
            ious[i, j] = (x & y).sum() / ((x | y).sum() + 1e-10)
    return ious


def frame_counts(pred: list[np.ndarray], gt: list[np.ndarray]) -> np.ndarray:
    """Count vector of one frame (lanes in 1920x1280 pixels): TP, FP, FN per threshold, matched-IoU sum and
    number of predictions."""
    p, g = _prepare(pred), _prepare(gt)
    out = np.zeros(_N, dtype=np.float64)
    if not p:
        for k in range(len(THRESHOLDS)):
            out[3 * k + 2] = len(g)
        return out
    if not g:
        for k in range(len(THRESHOLDS)):
            out[3 * k + 1] = len(p)
        out[-1] = len(p)
        return out
    ious = cross_iou(p, g)
    rows, cols = linear_sum_assignment(1 - ious)
    matched = ious[rows, cols]
    for k, t in enumerate(THRESHOLDS):
        tp = int((matched > t).sum())
        out[3 * k:3 * k + 3] = (tp, len(p) - tp, len(g) - tp)
    out[-2] = float(matched.sum())
    out[-1] = len(p)
    return out


def finish(counts: np.ndarray, prefix: str = PREFIX) -> dict[str, float]:
    """Metrics from the count vectors summed over frames."""
    metrics: dict[str, float] = {}
    for k, t in enumerate(THRESHOLDS):
        tp, fp, fn = counts[3 * k:3 * k + 3]
        precision = tp / (tp + fp) if tp else 0.0
        recall = tp / (tp + fn) if tp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
        tag = f"iou{int(round(t * 100)):02d}"
        metrics.update({f"{prefix}_f1_{tag}": f1, f"{prefix}_precision_{tag}": precision,
                        f"{prefix}_recall_{tag}": recall, f"{prefix}_tp_{tag}": int(tp),
                        f"{prefix}_fp_{tag}": int(fp), f"{prefix}_fn_{tag}": int(fn)})
    metrics[f"{prefix}_miou"] = counts[-2] / counts[-1] if counts[-1] else 0.0
    return metrics


def _gt_to_pred_iou(pred: list[np.ndarray], gt: list[np.ndarray]) -> np.ndarray:
    """IoU of the prediction matched to each annotated lane (0 when none); GT x pred Hungarian, as RVLD's
    ``culane_metric2``."""
    best = np.zeros(len(gt))
    if pred and gt:
        ious = cross_iou(gt, pred)
        rows, cols = linear_sum_assignment(1 - ious)
        best[rows] = ious[rows, cols]
    return best


GT_MATCH_THRESHOLD = 0.5  # RVLD matches annotated lanes of consecutive frames at 0.5 for every threshold


def temporal_counts(sequences: list[list[tuple[list[np.ndarray], list[np.ndarray]]]],
                    threshold: float = 0.5) -> tuple[int, int, int]:
    """(stable, flickering, missing) over consecutive frames of each video; each frame is (pred, gt) in
    1920x1280 pixels, frames in time order. ``threshold`` decides whether a detection is correct."""
    ns = nj = nm = 0
    for frames in sequences:
        prev_gt = prev_best = None
        for pred, gt in frames:
            p, g = _prepare(pred), _prepare(gt)
            best = _gt_to_pred_iou(p, g)
            if prev_gt is not None and g and prev_gt:
                ious = cross_iou(g, prev_gt)
                rows, cols = linear_sum_assignment(1 - ious)
                keep = ious[rows, cols] > GT_MATCH_THRESHOLD
                for cur, prev in zip(rows[keep], cols[keep]):
                    now_ok, before_ok = best[cur] > threshold, prev_best[prev] > threshold
                    if now_ok and before_ok:
                        ns += 1
                    elif now_ok or before_ok:
                        nj += 1
                    else:
                        nm += 1
            prev_gt, prev_best = g, best
    return ns, nj, nm


def temporal_metrics(sequences: list[list[tuple[list[np.ndarray], list[np.ndarray]]]],
                     prefix: str = PREFIX) -> dict[str, float]:
    out: dict[str, float] = {}
    for t in THRESHOLDS:
        ns, nj, nm = temporal_counts(sequences, t)
        total = ns + nj + nm
        tag = f"iou{int(round(t * 100)):02d}"
        out.update({f"{prefix}_flicker_rate_{tag}": nj / total if total else 0.0,
                    f"{prefix}_missing_rate_{tag}": nm / total if total else 0.0,
                    f"{prefix}_stable_rate_{tag}": ns / total if total else 0.0})
    return out
