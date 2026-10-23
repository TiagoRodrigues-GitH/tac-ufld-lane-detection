"""Closest in-path object (CIPO): ego-lane geometry, selection rules, distance proxies and metrics.

Added 2026-10-19 (design: ``docs/CIPO.md``). The detector itself is ``models/cipo_head.py``; this module holds
everything that is not learned, so the same code scores ground-truth boxes (upper bounds) and predictions.

Everything is in ORIGINAL image pixels (OpenLane: 1920x1280). Boxes are ``(N, 4)`` arrays ``x1, y1, x2, y2``;
lanes are the (N, 2) ``(x, y)`` polylines that the lane models output or the labels hold.

* ``CameraModel``          pinhole camera above a flat road: distance of an image row, lane width in pixels.
* ``ego_span``             x interval of the ego lane at an image row, from the ego boundaries, or from one
                           boundary plus a nominal lane width, or a "virtual" lane centred on the camera.
* ``in_path_fraction``     share of a box's width inside the ego lane at its contact row (the box's bottom).
                           OpenLane's level 1 asks for "over 50 % of the body in the ego lane".
* ``select_cipo``          the CIPO among the detections of one frame (rules ``learned``, ``geometric``,
                           ``hybrid``; ``docs/CIPO.md`` section 5); ``path_geometry`` precomputes what it needs.
* ``lane_ground_distance`` reference distance at an image row from the 3-D points of the ground-truth ego
                           boundaries (no lidar needed; includes road slope).
* ``average_precision``    COCO-style 101-point interpolated AP at one IoU threshold.
* ``cipo_frame_metrics``   per-frame CIPO decision against the level-1 label (accuracy, false / missed CIPO).
* ``cipo_stability``       CIPO flicker and identity switches per minute on consecutive frames.

Conventions: OpenLane's ego boundaries are lane slots 1 and 2 (attributes 2 = left, 3 = right), not 0 and 1 as in
ELAS; ``OPENLANE_EGO_SLOTS`` says so once. The camera sits on the car's centre line (column ``cx``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tac_ufld.ego import x_at

OPENLANE_EGO_SLOTS = (1, 2)
FPS_OPENLANE = 10.0


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera at ``height`` metres above a flat road, optical axis parallel to it.

    A road point at forward distance Z appears at row ``v = cy + fy * height / Z``. This is how OpenLane's lane
    ``xyz`` (x forward, y left, z up, camera origin) projects to its ``uv`` (measured 2026-10-19: median error
    2.9 px with the extrinsic rotation ignored). Defaults: the Waymo front camera as OpenLane labels it."""

    fx: float = 2074.0
    fy: float = 2074.0
    cx: float = 960.0
    cy: float = 640.0
    height: float = 2.116

    @classmethod
    def from_openlane(cls, intrinsic: Sequence[Sequence[float]], extrinsic: Sequence[Sequence[float]]
                      ) -> CameraModel:
        k, e = np.asarray(intrinsic, dtype=np.float64), np.asarray(extrinsic, dtype=np.float64)
        return cls(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), float(e[2, 3]))

    def ground_distance(self, v: np.ndarray | float) -> np.ndarray:
        """Forward distance (m) of the road point seen at row ``v``; inf at or above the horizon."""
        dv = np.asarray(v, dtype=np.float64) - self.cy
        with np.errstate(divide="ignore"):
            return np.where(dv > 1e-6, self.fy * self.height / np.maximum(dv, 1e-6), np.inf)

    def ground_width_px(self, v: float, metres: float) -> float:
        """Width in pixels, at row ``v``, of a road segment ``metres`` wide (0 at or above the horizon)."""
        return max(0.0, metres * self.fx * (float(v) - self.cy) / (self.fy * self.height))


def ego_span(left: np.ndarray | None, right: np.ndarray | None, v: float, camera: CameraModel,
             lane_width_m: float = 3.5) -> tuple[float, float, str]:
    """(x_left, x_right, source) of the ego lane at row ``v``.

    ``source``: ``lanes`` (both boundaries), ``left+width`` / ``right+width`` (one boundary plus a lane of
    ``lane_width_m`` on a flat road), ``virtual`` (no usable boundary: a lane centred on the camera, the
    "virtual ego-lane lines" of OpenLane's level 2), or ``none`` (row at or above the horizon)."""
    width = camera.ground_width_px(v, lane_width_m)
    if width <= 0:
        return float("nan"), float("nan"), "none"
    xl, xr = x_at(left, v), x_at(right, v)
    if np.isfinite(xl) and np.isfinite(xr):
        if xr > xl:
            return xl, xr, "lanes"
        return camera.cx - width / 2, camera.cx + width / 2, "virtual"   # crossed boundaries: unusable
    if np.isfinite(xl):
        return xl, xl + width, "left+width"
    if np.isfinite(xr):
        return xr - width, xr, "right+width"
    return camera.cx - width / 2, camera.cx + width / 2, "virtual"


def in_path_fraction(box: Sequence[float], left: np.ndarray | None, right: np.ndarray | None,
                     camera: CameraModel, lane_width_m: float = 3.5) -> float:
    """Share of the box's width inside the ego lane, measured at the box's bottom row (where it meets the road)."""
    x1, _, x2, y2 = (float(c) for c in box)
    xl, xr, source = ego_span(left, right, y2, camera, lane_width_m)
    if source == "none" or x2 <= x1:
        return 0.0
    return max(0.0, min(x2, xr) - max(x1, xl)) / (x2 - x1)


def ego_boundaries(lanes: Sequence[np.ndarray | None] | None,
                   slots: tuple[int, int] = OPENLANE_EGO_SLOTS) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(ego-left, ego-right) polylines out of a lane-slot tuple (None when absent)."""
    if lanes is None:
        return None, None
    get = (lambda i: lanes[i] if i < len(lanes) else None)
    return get(slots[0]), get(slots[1])


@dataclass(frozen=True)
class CipoRule:
    """How the CIPO of a frame is chosen among its detections (``docs/CIPO.md`` section 5).

    * ``learned``:   the detection with the highest level-1 score, if >= ``level1_threshold``; no geometry.
    * ``geometric``: detections with score >= ``score_threshold``, in path (>= ``min_in_path``) and nearer than
                     ``max_distance`` (flat-ground distance of the bottom row: the "reaction distance");
                     the CIPO is the nearest one (largest bottom row). The learned level is not used.
    * ``hybrid``:    the geometric filter, then the highest level-1 score, if >= ``level1_threshold``."""

    mode: str = "geometric"
    score_threshold: float = 0.3
    level1_threshold: float = 0.3
    min_in_path: float = 0.5
    max_distance: float = float("inf")
    lane_width_m: float = 3.5

    def __post_init__(self) -> None:
        if self.mode not in ("learned", "geometric", "hybrid"):
            raise ValueError(f"unknown CIPO rule {self.mode!r}")


def path_geometry(boxes: np.ndarray, left: np.ndarray | None, right: np.ndarray | None, camera: CameraModel,
                  lane_width_m: float = 3.5) -> tuple[np.ndarray, np.ndarray]:
    """(in-path fraction, flat-ground distance of the bottom row) of every box: what ``select_cipo`` needs,
    computed once per frame when many rules are compared."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    frac = np.array([in_path_fraction(b, left, right, camera, lane_width_m) for b in boxes], dtype=np.float64)
    return frac, camera.ground_distance(boxes[:, 3]) if len(boxes) else np.zeros(0)


def select_cipo(boxes: np.ndarray, scores: np.ndarray, level1_scores: np.ndarray | None,
                left: np.ndarray | None, right: np.ndarray | None, camera: CameraModel, rule: CipoRule,
                geometry: tuple[np.ndarray, np.ndarray] | None = None) -> int | None:
    """Index of the CIPO among ``boxes`` (N, 4), or None. ``scores``: detection confidence (any level);
    ``level1_scores``: the level-1 ("this is the CIPO") confidence of each detection (needed by learned/hybrid);
    ``geometry``: ``path_geometry`` of the boxes if already computed (same ``lane_width_m``)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return None
    scores = np.asarray(scores, dtype=np.float64)
    if rule.mode == "learned":
        l1 = np.asarray(level1_scores, dtype=np.float64)
        best = int(np.argmax(l1))
        return best if l1[best] >= rule.level1_threshold else None
    frac, dist = geometry if geometry is not None else path_geometry(boxes, left, right, camera, rule.lane_width_m)
    keep = (scores >= rule.score_threshold) & (dist <= rule.max_distance) & (frac >= rule.min_in_path)
    candidates = np.flatnonzero(keep)
    if len(candidates) == 0:
        return None
    if rule.mode == "geometric":
        return int(candidates[np.argmax(boxes[candidates, 3])])
    l1 = np.asarray(level1_scores, dtype=np.float64)[candidates]
    best = int(np.argmax(l1))
    return int(candidates[best]) if l1[best] >= rule.level1_threshold else None


# -- distance ------------------------------------------------------------------------------------------------------

def project_lane_ground(xyz: np.ndarray, visibility: np.ndarray | None, camera: CameraModel,
                        min_visibility: float = 0.5, min_forward: float = 1.0) -> np.ndarray | None:
    """OpenLane lane ``xyz`` (3, N) -> (M, 2) ``(v, forward distance)`` of its visible points, sorted by v."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[0] != 3 or xyz.shape[1] < 2:
        return None
    keep = np.isfinite(xyz).all(axis=0) & (xyz[0] > min_forward)
    if visibility is not None and len(visibility) == xyz.shape[1]:
        keep &= np.asarray(visibility, dtype=np.float64) > min_visibility
    if keep.sum() < 2:
        return None
    x, z = xyz[0, keep], xyz[2, keep]
    v = camera.cy - camera.fy * z / x
    out = np.stack([v, x], axis=1)
    return out[np.argsort(out[:, 0])]


def lane_ground_distance(v: float, ground: Sequence[np.ndarray | None]) -> float:
    """Forward distance at row ``v`` read off the ego boundaries' 3-D points (mean over the boundaries that span
    the row; no extrapolation); nan when no boundary spans it."""
    found = []
    for pts in ground:
        if pts is None or len(pts) < 2 or not pts[0, 0] <= v <= pts[-1, 0]:
            continue
        found.append(float(np.interp(v, pts[:, 0], pts[:, 1])))
    return float(np.mean(found)) if found else float("nan")


def distance_errors(estimate: np.ndarray, reference: np.ndarray,
                    bins: Sequence[float] = (0.0, 20.0, 40.0, 80.0, np.inf)) -> dict[str, float]:
    """abs-rel error and share within 10 %, overall and by reference-range bin (m)."""
    est, ref = np.asarray(estimate, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    ok = np.isfinite(est) & np.isfinite(ref) & (ref > 0)
    out: dict[str, float] = {"n": float(ok.sum())}
    if not ok.any():
        return out
    rel = np.abs(est[ok] - ref[ok]) / ref[ok]
    out.update(abs_rel=float(np.mean(rel)), abs_rel_median=float(np.median(rel)),
               within_10pct=float(np.mean(rel < 0.1)))
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (ref[ok] >= lo) & (ref[ok] < hi)
        tag = f"{lo:g}_{hi:g}m" if np.isfinite(hi) else f"{lo:g}m_plus"
        out[f"n_{tag}"] = float(sel.sum())
        if sel.any():
            out[f"abs_rel_{tag}"] = float(np.mean(rel[sel]))
    return out


# -- detection metrics -------------------------------------------------------------------------------------------

def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, M) IoU of xyxy boxes."""
    a, b = np.asarray(a, dtype=np.float64).reshape(-1, 4), np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def average_precisions(detections: Sequence[tuple[object, float, np.ndarray]],
                       ground_truth: dict[object, np.ndarray], thresholds: Sequence[float]) -> np.ndarray:
    """COCO-style AP at several IoU thresholds: greedy matching by descending score, 101-point interpolated
    precision.

    ``detections``: (frame key, score, xyxy box); ``ground_truth``: frame key -> (M, 4) boxes (frames without
    objects may be omitted). nan when there is no ground-truth box at all. Matching is done per frame (it only
    depends on the order of that frame's detections), with each frame's IoU matrix computed once."""
    thresholds = list(thresholds)
    gts = {k: np.asarray(g, dtype=np.float64).reshape(-1, 4) for k, g in ground_truth.items()}
    n_gt = sum(len(g) for g in gts.values())
    if n_gt == 0:
        return np.full(len(thresholds), np.nan)
    if len(detections) == 0:
        return np.zeros(len(thresholds))
    scores = np.array([float(d[1]) for d in detections])
    order = np.lexsort((np.arange(len(scores)), -scores))     # descending score, ties by input order
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    by_frame: dict[object, list[int]] = {}
    for i in order:
        by_frame.setdefault(detections[i][0], []).append(int(i))
    tp = np.zeros((len(thresholds), len(scores)))
    for key, idx in by_frame.items():
        g = gts.get(key)
        if g is None or len(g) == 0:
            continue
        iou = box_iou(np.asarray([detections[i][2] for i in idx], dtype=np.float64), g)
        for t, thr in enumerate(thresholds):
            free = np.ones(len(g), dtype=bool)
            for r, i in enumerate(idx):
                row = np.where(free, iou[r], -1.0)
                j = int(row.argmax())
                if row[j] >= thr:
                    free[j] = False
                    tp[t, rank[i]] = 1.0
    samples = np.linspace(0, 1, 101)
    out = np.zeros(len(thresholds))
    for t in range(len(thresholds)):
        ctp = np.cumsum(tp[t])
        recall = ctp / n_gt
        precision = ctp / np.arange(1, len(ctp) + 1)
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        idx = np.searchsorted(recall, samples, side="left")
        out[t] = np.mean(np.where(idx < len(envelope), envelope[np.minimum(idx, len(envelope) - 1)], 0.0))
    return out


def average_precision(detections: Sequence[tuple[object, float, np.ndarray]],
                      ground_truth: dict[object, np.ndarray], iou_threshold: float = 0.5) -> float:
    """COCO-style AP at one IoU threshold (``average_precisions``)."""
    return float(average_precisions(detections, ground_truth, [iou_threshold])[0])


def cipo_frame_metrics(selected: Sequence[np.ndarray | None], ground_truth: Sequence[np.ndarray],
                       iou_threshold: float = 0.5) -> dict[str, float]:
    """Per-frame CIPO decision (one box or None) against the level-1 boxes of the frame ((M, 4), M usually 0/1).

    * correct: no CIPO predicted and none labelled, or the predicted box overlaps a level-1 box (IoU >= thr);
    * false CIPO: a CIPO is predicted but no level-1 box matches it (phantom braking risk);
    * missed CIPO: a level-1 box is labelled but no matching CIPO is predicted."""
    n = len(selected)
    tp = fp = fn = correct = 0
    for pred, gt in zip(selected, ground_truth):
        gt = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
        hit = pred is not None and len(gt) > 0 and box_iou(np.asarray(pred)[None], gt).max() >= iou_threshold
        if pred is None and len(gt) == 0:
            correct += 1
        elif hit:
            correct += 1
            tp += 1
        else:
            fp += pred is not None
            fn += len(gt) > 0
    n_pred = sum(p is not None for p in selected)
    n_gt = sum(len(np.asarray(g).reshape(-1, 4)) > 0 for g in ground_truth)
    precision = tp / n_pred if n_pred else float("nan")
    recall = tp / n_gt if n_gt else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if n_pred and n_gt and precision + recall > 0 else float("nan")
    return {"n_frames": float(n), "n_frames_with_cipo": float(n_gt), "n_frames_predicted": float(n_pred),
            "cipo_accuracy": correct / n if n else float("nan"), "cipo_precision": precision,
            "cipo_recall": recall, "cipo_f1": f1, "false_cipo_rate": fp / n if n else float("nan"),
            "missed_cipo_rate": fn / n if n else float("nan")}


def cipo_stability(sequences: Sequence[str], frame_index: Sequence[int], predicted_id: Sequence[str | None],
                   gt_id: Sequence[str | None], fps: float = FPS_OPENLANE) -> dict[str, float]:
    """Flicker and identity switches of the selected CIPO over consecutive frames (same sequence, index + 1).

    ``predicted_id``: the track id of the ground-truth box the predicted CIPO matches (``"?"`` if it matches
    none, None if no CIPO is predicted); ``gt_id``: the level-1 track id (None if no level 1).
    * flicker: the predicted CIPO appears or disappears while the labelled one does not;
    * switch: the predicted CIPO changes identity while the labelled level-1 identity stays the same."""
    flicker = switches = pairs = 0
    for i in range(1, len(sequences)):
        if sequences[i] != sequences[i - 1] or frame_index[i] != frame_index[i - 1] + 1:
            continue
        pairs += 1
        gt_same = gt_id[i] == gt_id[i - 1]
        p0, p1 = predicted_id[i - 1], predicted_id[i]
        if (p0 is None) != (p1 is None) and gt_same:
            flicker += 1
        elif p0 is not None and p1 is not None and p0 != p1 and gt_same:
            switches += 1
    minutes = pairs / fps / 60.0
    return {"consecutive_pairs": float(pairs),
            "cipo_flicker_per_min": flicker / minutes if minutes else float("nan"),
            "cipo_switches_per_min": switches / minutes if minutes else float("nan")}
