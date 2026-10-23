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
* OpenLane-V - RVLD's protocol (``rvld_protocol``): F1 at IoU 0.5 and 0.8 and mIoU at 960x640, plus the
  flickering and missing rates over consecutive frames.
* ELAS - no official lane-detection benchmark. Added 2026-10-18: the dataset paper's ego-lane metrics
  (Berriel et al. 2017): lane-centre deviation in px and in % of the lane width at the bottom of the ROI, with the
  share of frames where both boundaries were predicted (``native_elas_center_*``), and lane-change event detection
  from the predicted lanes against the ``laneChange`` label (``native_elas_lc_*``, see ``tac_ufld.ego``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tac_ufld.data.types import FrameRecord
from tac_ufld.evaluation import rvld_protocol
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


def all_lanes_counts(rec: FrameRecord, lanes: list, line_width: int, threshold: float = 0.5,
                     interp: str = "linear") -> tuple[int, int, int]:
    """TP, FP, FN of one frame against every annotated lane (one term of ``iou_f1_all_lanes``).
    ``interp="spline"``: lanes drawn along their interpolating cubic spline (official CULane tool)."""
    gt = rec.meta.get("eval_lanes")
    if gt is None:
        gt = rec.present_lanes()
    w, h = rec.image_size
    pred = [lane for lane in lanes if lane is not None]
    pm = [lane_mask(lane, w, h, line_width, interp) for lane in pred]
    gm = [lane_mask(lane, w, h, line_width, interp) for lane in gt]
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


# ------------------------------------------------------------ OpenLane-V (RVLD)


def _band_to_frame(lanes: list, crop_top: float) -> list[np.ndarray]:
    out = []
    for lane in lanes:
        if lane is not None:
            pts = np.asarray(lane, dtype=np.float64).copy()
            pts[:, 1] += crop_top
            out.append(pts)
    return out


def openlane_v_counts(rec: FrameRecord, lanes: list) -> np.ndarray:
    """RVLD per-frame counts; predictions come in band coordinates, annotations in full-frame pixels."""
    return rvld_protocol.frame_counts(_band_to_frame(lanes, rec.meta.get("crop_top", 0)), rec.meta["eval_lanes"])


def openlane_v_video_metrics(records: list[FrameRecord], pred_lanes: list[list]) -> dict[str, float]:
    """RVLD's flickering / missing rates over consecutive test frames of each segment."""
    videos: dict[str, list[tuple[int, list, list]]] = {}
    for rec, lanes in zip(records, pred_lanes):
        videos.setdefault(rec.sequence, []).append(
            (rec.frame_id, _band_to_frame(lanes, rec.meta.get("crop_top", 0)), rec.meta["eval_lanes"]))
    sequences = [[(p, g) for _, p, g in sorted(frames, key=lambda f: f[0])] for frames in videos.values()]
    return rvld_protocol.temporal_metrics(sequences)


# ------------------------------------------------------------------ ELAS (Berriel et al. 2017)


def elas_center_counts(rec: FrameRecord, lanes: list) -> np.ndarray:
    """Lane-centre deviation of ELAS ("difference of the lane centre predicted by our system and the lane centre
    of the ground truth", reported as % of the lane width), at the bottom row of the ROI.

    Vector: [frames with both GT boundaries, of those with both predicted, sum |dc| px, sum |dc| / GT width]."""
    from tac_ufld.ego import ego_lane_state

    if rec.valid_y_range is None:
        return np.zeros(4)
    y0, y1 = rec.valid_y_range
    cam = rec.image_size[0] / 2.0
    gt = ego_lane_state(list(rec.lanes), y1, y0, cam)
    if not np.isfinite(gt.center_x) or not (np.isfinite(gt.left_x) and np.isfinite(gt.right_x)):
        return np.zeros(4)
    pr = ego_lane_state(list(lanes), y1, y0, cam)
    # both boundaries predicted AND in the right order (extrapolated to the bottom row they may cross)
    if not (np.isfinite(pr.left_x) and np.isfinite(pr.right_x) and np.isfinite(pr.center_x)):
        return np.array([1.0, 0.0, 0.0, 0.0])
    dc = abs(pr.center_x - gt.center_x)
    return np.array([1.0, 1.0, dc, dc / gt.width])


def elas_center_finish(total: np.ndarray) -> dict[str, float]:
    n_gt, n, err_px, err_frac = (float(v) for v in total)
    nan = float("nan")
    return {"native_elas_center_coverage": n / n_gt if n_gt else nan,
            "native_elas_center_err_px": err_px / n if n else nan,
            "native_elas_center_err_pct": 100.0 * err_frac / n if n else nan}


def elas_lane_change_metrics(records: list[FrameRecord], pred_lanes: list[list]) -> dict[str, float]:
    """Lane-change events from the PREDICTED lanes (``tac_ufld.ego``) against the ELAS ``laneChange`` label.

    Frames of each scene are replayed in order at 29.97 FPS; a gap of more than 15 frames (split blocks)
    restarts the tracker. With ground-truth lanes the same code finds 8/8 events (scripts/ego_lane_check.py).
    Only sequences that contain at least one event count toward the delay; false alarms count everywhere."""
    from tac_ufld.ego import EgoLaneTracker, event_metrics

    by_seq: dict[str, list[tuple[int, FrameRecord, list]]] = {}
    for rec, lanes in zip(records, pred_lanes):
        if rec.valid_y_range is not None and "laneChange" in rec.meta:
            by_seq.setdefault(rec.sequence, []).append((rec.frame_id, rec, lanes))
    if not by_seq:
        return {}
    pred_all, gt_all = [], []
    for items in by_seq.values():
        items.sort(key=lambda it: it[0])
        first = items[0][1]
        tracker = EgoLaneTracker(first.image_size[0] / 2.0, first.valid_y_range[1], first.valid_y_range[0])
        prev_id = None
        for fid, rec, lanes in items:
            if prev_id is not None and fid - prev_id > 15:
                tracker.reset()
                pred_all.append(0)
                gt_all.append(0)  # keeps events of different blocks apart
            _, decision = tracker.update(list(lanes))
            label = rec.meta.get("laneChange", 0.0)
            pred_all.append(decision)
            gt_all.append(int(label) if np.isfinite(label) else 0)
            prev_id = fid
        pred_all.append(0)
        gt_all.append(0)
    m = event_metrics(np.asarray(pred_all), np.asarray(gt_all), fps=29.97)
    return {f"native_elas_lc_{k}": float(v) for k, v in m.items()}


# ------------------------------------------------------------------ dispatch


@dataclass(frozen=True)
class NativeProtocol:
    """How a dataset's native metric is computed.

    ``frame_counts`` returns an additive count vector per frame (so frames can be scored in worker processes)
    and ``finish`` turns the summed vector into metrics; ``sequence_metrics`` needs the whole evaluation set
    at once (TuSimple's per-image averages, video metrics) and runs in the main process.
    """

    frame_counts: Callable[[FrameRecord, list], np.ndarray] | None = None
    finish: Callable[[np.ndarray], dict[str, float]] | None = None
    sequence_metrics: Callable[[list[FrameRecord], list[list]], dict[str, float]] | None = None


def _all_lanes_protocol(prefix: str, interp: str = "linear") -> NativeProtocol:
    def counts(rec: FrameRecord, lanes: list) -> np.ndarray:
        return np.asarray(all_lanes_counts(rec, lanes, line_width=30, interp=interp), dtype=np.int64)

    def finish(total: np.ndarray) -> dict[str, float]:
        return all_lanes_metrics(int(total[0]), int(total[1]), int(total[2]), 0.5, prefix)

    return NativeProtocol(frame_counts=counts, finish=finish)


NATIVE_PROTOCOLS: dict[str, NativeProtocol] = {
    "culane": _all_lanes_protocol("native_culane", interp="spline"),  # official tool: spline, 30 px at 1640x590
    "openlane": _all_lanes_protocol("native_openlane"),
    "openlane_v": NativeProtocol(frame_counts=openlane_v_counts, finish=rvld_protocol.finish,
                                 sequence_metrics=openlane_v_video_metrics),
    "tusimple": NativeProtocol(sequence_metrics=tusimple_metrics),
    "elas": NativeProtocol(frame_counts=elas_center_counts, finish=elas_center_finish,
                           sequence_metrics=elas_lane_change_metrics),
}


def native_metrics(records: list[FrameRecord], pred_lanes: list[list]) -> dict[str, float]:
    """Every native metric of the records' dataset (one dataset per evaluation), in this process."""
    if not records or records[0].dataset not in NATIVE_PROTOCOLS:
        return {}
    protocol = NATIVE_PROTOCOLS[records[0].dataset]
    metrics: dict[str, float] = {}
    if protocol.frame_counts is not None:
        total = sum(protocol.frame_counts(rec, lanes) for rec, lanes in zip(records, pred_lanes))
        metrics.update(protocol.finish(total))
    if protocol.sequence_metrics is not None:
        metrics.update(protocol.sequence_metrics(records, pred_lanes))
    return metrics
