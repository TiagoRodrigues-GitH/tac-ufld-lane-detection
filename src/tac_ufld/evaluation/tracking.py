"""Output-level temporal filtering: a causal Kalman tracker on the lane points.

This is the classical, almost free way to use time in lane detection: run a
single-frame detector and filter its output over the frames of the stream.
Every row-anchor cell (anchor ``a``, lane slot ``l``) carries a
constant-velocity Kalman filter on its x position, and the existence
probability is smoothed by an exponential moving average. A cell that
reappears far from its prediction (a lane change, a detection jump) is
re-initialised instead of being dragged by the filter.

It is causal (each output uses only the current and past frames), so it is a
deployable reference: a learned temporal model only earns its extra cost if
it beats "single-frame model + this tracker". Its parameters are tuned on
validation only (``tune_kalman``), like the post-processing.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from tac_ufld.data.types import FrameRecord
from tac_ufld.evaluation.predictor import Predictions


@dataclass(frozen=True)
class KalmanParams:
    q: float = 1.0          # process noise: acceleration variance (px^2 / frame^4, model pixels)
    r: float = 16.0         # measurement noise variance (px^2): ~4 px detector noise
    alpha: float = 0.5      # existence smoothing: e_t = alpha * e_(t-1) + (1 - alpha) * measured
    gate_px: float = 40.0   # innovation above this re-initialises the cell
    max_gap: int = 10       # frames without an update after which a cell / the stream is reset
    threshold: float = 0.5  # existence probability that counts as a measurement

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class LaneKalmanTracker:
    """State of one stream: arrays of shape (A, L) (row anchors x lane slots)."""

    def __init__(self, params: KalmanParams) -> None:
        self.p = params
        self.reset()

    def reset(self) -> None:
        self.last_index: int | None = None
        self.x = self.v = self.p00 = self.p01 = self.p11 = self.e = self.age = None
        self.valid = None

    def _init_cells(self, mask: np.ndarray, x: np.ndarray) -> None:
        r = self.p.r
        self.x[mask] = x[mask]
        self.v[mask] = 0.0
        self.p00[mask], self.p01[mask], self.p11[mask] = r, 0.0, r
        self.age[mask] = 0
        self.valid[mask] = True

    def update(self, exist: np.ndarray, x: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Filter one frame: returns the smoothed (exist, x). ``index`` is the
        frame index (gaps are allowed and handled as a longer prediction)."""
        p = self.p
        exist = np.asarray(exist, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        if self.last_index is None or index <= self.last_index or index - self.last_index > p.max_gap:
            self.reset()
        measured = exist >= p.threshold
        if self.last_index is None:
            shape = exist.shape
            self.x, self.v = np.zeros(shape), np.zeros(shape)
            self.p00, self.p01, self.p11 = np.zeros(shape), np.zeros(shape), np.zeros(shape)
            self.age, self.valid = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=bool)
            self.e = exist.copy()
            self._init_cells(measured, x)
            self.last_index = index
            return exist.astype(np.float32), x.astype(np.float32)
        dt = float(index - self.last_index)
        self.last_index = index
        # predict (constant velocity, white acceleration noise)
        q = p.q
        self.x = self.x + self.v * dt
        p00 = self.p00 + 2 * dt * self.p01 + dt * dt * self.p11 + q * dt ** 4 / 4
        p01 = self.p01 + dt * self.p11 + q * dt ** 3 / 2
        p11 = self.p11 + q * dt ** 2
        self.p00, self.p01, self.p11 = p00, p01, p11
        self.e = p.alpha * self.e + (1.0 - p.alpha) * exist
        self.age += int(dt)
        # measurement update
        innovation = x - self.x
        reinit = measured & (~self.valid | (np.abs(innovation) > p.gate_px))
        upd = measured & ~reinit
        s = self.p00 + p.r
        k0, k1 = self.p00 / s, self.p01 / s
        self.x = np.where(upd, self.x + k0 * innovation, self.x)
        self.v = np.where(upd, self.v + k1 * innovation, self.v)
        n00, n01, n11 = (1 - k0) * self.p00, (1 - k0) * self.p01, self.p11 - k1 * self.p01
        self.p00 = np.where(upd, n00, self.p00)
        self.p01 = np.where(upd, n01, self.p01)
        self.p11 = np.where(upd, n11, self.p11)
        self.age[upd] = 0
        self._init_cells(reinit, x)
        self.valid &= self.age <= p.max_gap
        x_out = np.where(self.valid, self.x, x)
        return self.e.astype(np.float32), x_out.astype(np.float32)


def sequence_order(records: list[FrameRecord]) -> list[list[int]]:
    """Indices of ``records`` grouped by sequence, each group in frame order."""
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        groups.setdefault(r.sequence, []).append(i)
    return [sorted(idx, key=lambda i: records[i].frame_id) for _, idx in sorted(groups.items())]


def track_predictions(preds: Predictions, records: list[FrameRecord], params: KalmanParams) -> Predictions:
    """Apply one tracker per sequence, frames in order (causal)."""
    exist, x = preds.exist.copy(), preds.x.copy()
    for idx in sequence_order(records):
        tracker = LaneKalmanTracker(params)
        for i in idx:
            exist[i], x[i] = tracker.update(preds.exist[i], preds.x[i], records[i].frame_id)
    return Predictions(exist=exist, x=x, current_weight=preds.current_weight, loss=preds.loss)


def tune_kalman(evaluator, preds: Predictions, records: list[FrameRecord], postprocess,
                grid: dict[str, list[float]], metric: str) -> tuple[KalmanParams, pd.DataFrame]:
    """Grid search on (validation) predictions with the model's tuned
    post-processing; the measurement threshold follows the post-processing."""
    base = KalmanParams(threshold=postprocess.threshold)
    keys = list(grid)
    rows, best, best_score = [], base, -np.inf
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = replace(base, **{k: (int(v) if k == "max_gap" else float(v)) for k, v in zip(keys, combo)})
        result = evaluator.evaluate(track_predictions(preds, records, params), records, postprocess, lane_only=True)
        score = result.metrics[metric]
        rows.append({**params.as_dict(), metric: score})
        if score > best_score:
            best, best_score = params, score
    return best, pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)
