"""Dataset-native evaluation protocols, reported separately from the internal
metrics (all ``native_*`` keys). They use every annotated lane of a frame
(``meta["eval_lanes"]``), not only the lanes that fit the model's slots.

* CULane  - lane F1 at IoU 0.5, 30 px lines at 1640x590, one-to-one matching.
  The official C++ evaluator draws cubic-spline-interpolated lanes; here the
  annotated polyline is drawn, so numbers are close but not identical.
* TuSimple - the official ``LaneEval.bench`` (tusimple-benchmark
  ``evaluate/lane.py``) ported line by line: per GT lane, the best
  prediction's share of points within ``20 px / cos(angle)``; a GT lane is
  matched if that share >= 0.85; accuracy, FP and FN rates per image,
  averaged over images. ``native_tusimple_f1`` is the F1 of (1 - FP, 1 - FN).
* OpenLane - CULane-style F1 at IoU 0.5 with 30 px lines at 1920x1280.
* ELAS - no official benchmark; nothing is added.
"""

from __future__ import annotations

import numpy as np

from tac_ufld.data.types import FrameRecord
from tac_ufld.metrics import iou_matrix, lane_mask, match_lanes, prf

TUSIMPLE_PIXEL_THRESH = 20.0
TUSIMPLE_PT_THRESH = 0.85


# ------------------------------------------------------------------ TuSimple


def _tusimple_angle(xs: np.ndarray, y_samples: np.ndarray) -> float:
    xs_valid, ys_valid = xs[xs >= 0], y_samples[xs >= 0]
    if len(xs_valid) > 1:
        k = np.polyfit(ys_valid, xs_valid, 1)[0]  # LinearRegression(ys -> xs) slope
        return float(np.arctan(k))
    return 0.0


def _tusimple_line_accuracy(pred: np.ndarray, gt: np.ndarray, thresh: float) -> float:
    pred = np.where(pred >= 0, pred, -100.0)
    gt = np.where(gt >= 0, gt, -100.0)
    return float(np.sum(np.abs(pred - gt) < thresh) / len(gt))


def tusimple_bench(pred: list[list[float]], gt: list[list[float]], y_samples: list[float],
                   running_time: float = 0.0) -> tuple[float, float, float]:
    """Official per-image score: (accuracy, FP rate, FN rate)."""
    if any(len(p) != len(y_samples) for p in pred):
        raise ValueError("every predicted lane needs one x per h_sample")
    if running_time > 200 or len(gt) + 2 < len(pred):
        return 0.0, 0.0, 1.0
    ys = np.asarray(y_samples, dtype=np.float64)
    angles = [_tusimple_angle(np.asarray(x, dtype=np.float64), ys) for x in gt]
    threshs = [TUSIMPLE_PIXEL_THRESH / np.cos(a) for a in angles]
    line_accs, fn, matched = [], 0.0, 0.0
    for x_gt, thresh in zip(gt, threshs):
        accs = [_tusimple_line_accuracy(np.asarray(x_p, float), np.asarray(x_gt, float), thresh) for x_p in pred]
        max_acc = max(accs) if accs else 0.0
        if max_acc < TUSIMPLE_PT_THRESH:
            fn += 1
        else:
            matched += 1
        line_accs.append(max_acc)
    fp = len(pred) - matched
    if len(gt) > 4 and fn > 0:
        fn -= 1
    s = sum(line_accs)
    if len(gt) > 4:
        s -= min(line_accs)
    return (s / max(min(4.0, len(gt)), 1.0), fp / len(pred) if len(pred) > 0 else 0.0,
            fn / max(min(len(gt), 4.0), 1.0))


def sample_at_rows(lane: np.ndarray, ys: list[float]) -> list[float]:
    """x of a polyline at each row, -2 outside its vertical extent (TuSimple format)."""
    pts = lane[np.argsort(lane[:, 1])]
    y = np.asarray(ys, dtype=np.float64)
    x = np.interp(y, pts[:, 1], pts[:, 0])
    out = np.where((y >= pts[0, 1]) & (y <= pts[-1, 1]), x, -2.0)
    return out.tolist()


def tusimple_metrics(records: list[FrameRecord], pred_lanes: list[list]) -> dict[str, float]:
    acc = fp = fn = 0.0
    n = 0
    for rec, lanes in zip(records, pred_lanes):
        ys = rec.meta.get("h_samples")
        gt = rec.meta.get("gt_lanes_x")
        if ys is None or gt is None:
            continue
        pred = [sample_at_rows(lane, ys) for lane in lanes if lane is not None]
        a, p, f = tusimple_bench(pred, gt, ys)
        acc, fp, fn, n = acc + a, fp + p, fn + f, n + 1
    if n == 0:
        return {}
    acc, fp, fn = acc / n, fp / n, fn / n
    return {"native_tusimple_accuracy": acc, "native_tusimple_fp": fp, "native_tusimple_fn": fn,
            "native_tusimple_f1": _f1_from_rates(fp, fn), "native_frames": n}


def _f1_from_rates(fp: float, fn: float) -> float:
    precision, recall = 1.0 - fp, 1.0 - fn
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


# ------------------------------------------------------ CULane-style (IoU F1)


def all_lanes_counts(rec: FrameRecord, lanes: list, line_width: int, threshold: float = 0.5) -> tuple[int, int, int]:
    """TP, FP, FN of one frame against every annotated lane (one term of ``iou_f1_all_lanes``)."""
    gt = rec.meta.get("eval_lanes")
    if gt is None:
        gt = rec.present_lanes()
    w, h = rec.image_size
    pred = [lane for lane in lanes if lane is not None]
    pm = [lane_mask(lane, w, h, line_width) for lane in pred]
    gm = [lane_mask(lane, w, h, line_width) for lane in gt]
    m = match_lanes(iou_matrix(pm, gm), threshold)
    return m.tp, m.fp, m.fn


def all_lanes_metrics(tp: int, fp: int, fn: int, threshold: float, prefix: str) -> dict[str, float]:
    p, r, f1 = prf(tp, fp, fn)
    tag = f"iou{int(round(threshold * 100)):02d}"
    return {f"{prefix}_f1_{tag}": f1, f"{prefix}_precision_{tag}": p, f"{prefix}_recall_{tag}": r,
            f"{prefix}_tp_{tag}": tp, f"{prefix}_fp_{tag}": fp, f"{prefix}_fn_{tag}": fn}


def iou_f1_all_lanes(records: list[FrameRecord], pred_lanes: list[list], line_width: int,
                     threshold: float = 0.5, prefix: str = "native") -> dict[str, float]:
    tp = fp = fn = 0
    for rec, lanes in zip(records, pred_lanes):
        t, f, n = all_lanes_counts(rec, lanes, line_width, threshold)
        tp, fp, fn = tp + t, fp + f, fn + n
    return all_lanes_metrics(tp, fp, fn, threshold, prefix)


# Datasets whose native metric is the all-lanes IoU F1 (a sum over frames, so it can be scored in parallel)
ALL_LANES_NATIVE = {"culane": "native_culane", "openlane": "native_openlane"}


def native_metrics(records: list[FrameRecord], pred_lanes: list[list]) -> dict[str, float]:
    """Dispatch on the dataset of the records (one dataset per evaluation)."""
    if not records:
        return {}
    dataset = records[0].dataset
    if dataset == "tusimple":
        return tusimple_metrics(records, pred_lanes)
    if dataset in ALL_LANES_NATIVE:
        return iou_f1_all_lanes(records, pred_lanes, line_width=30, prefix=ALL_LANES_NATIVE[dataset])
    return {}
