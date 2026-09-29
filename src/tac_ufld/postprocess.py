"""Row-anchor predictions -> lane polylines in original image pixels.

``poly_degree = 0`` reproduces the official UFLD output (one point per row
anchor where the lane exists). ``poly_degree >= 1`` fits x = f(y) through
those points and resamples it (the refinement step of both original scripts).
Predictions are restricted to the annotated band (ELAS ROI) because the
ground truth is undefined outside it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from tac_ufld.config import PostprocessConfig


@dataclass(frozen=True)
class PostprocessParams:
    threshold: float = 0.5
    min_points: int = 3
    poly_degree: int = 0
    duplicate_distance: float = 10.0
    y_step: float = 2.0

    @classmethod
    def from_config(cls, cfg: PostprocessConfig) -> "PostprocessParams":
        return cls(**asdict(cfg))

    def replace(self, **kwargs) -> "PostprocessParams":
        return PostprocessParams(**{**asdict(self), **kwargs})

    def as_dict(self) -> dict:
        return asdict(self)


def lanes_from_prediction(
    exist: np.ndarray,
    x_model: np.ndarray,
    row_anchors: np.ndarray,
    params: PostprocessParams,
    model_size: tuple[int, int],
    orig_size: tuple[int, int],
    valid_y_range: tuple[float, float] | None = None,
    slot_known: tuple[bool, ...] | None = None,
) -> list[np.ndarray | None]:
    """Return one polyline (N, 2) in original pixels or None per lane slot.
    Slots whose annotation is unknown (``slot_known[i] is False``) return
    None so they are neither rewarded nor penalised."""
    (mw, mh), (ow, oh) = model_size, orig_size
    y_orig = row_anchors * (oh / mh)
    in_range = np.ones_like(y_orig, dtype=bool)
    if valid_y_range is not None:
        in_range = (y_orig >= valid_y_range[0] - 1e-3) & (y_orig <= valid_y_range[1] + 1e-3)
    lanes: list[np.ndarray | None] = []
    for slot in range(exist.shape[1]):
        if slot_known is not None and not slot_known[slot]:
            lanes.append(None)
            continue
        mask = (exist[:, slot] >= params.threshold) & in_range
        if mask.sum() < max(params.min_points, 2):
            lanes.append(None)
            continue
        xs, ys = x_model[mask, slot] * (ow / mw), y_orig[mask]
        degree = int(params.poly_degree)
        if degree > 0 and len(ys) > degree:
            coeff = np.polyfit(ys, xs, degree)
            ys = np.arange(ys.min(), ys.max() + 1e-6, params.y_step, dtype=np.float64)
            xs = np.polyval(coeff, ys)
        keep = (xs >= 0) & (xs < ow)
        lanes.append(np.stack([xs[keep], ys[keep]], axis=1).astype(np.float32) if keep.sum() >= 2 else None)
    return suppress_duplicates(lanes, params.duplicate_distance)


def suppress_duplicates(lanes: list[np.ndarray | None], distance: float) -> list[np.ndarray | None]:
    """If two slots predict (almost) the same line, keep the longer one."""
    present = [i for i, lane in enumerate(lanes) if lane is not None]
    out = list(lanes)
    for a_pos, a in enumerate(present):
        for b in present[a_pos + 1:]:
            la, lb = out[a], out[b]
            if la is None or lb is None:
                continue
            if abs(float(la[:, 0].mean()) - float(lb[:, 0].mean())) < distance:
                span_a = np.ptp(la[:, 1])
                span_b = np.ptp(lb[:, 1])
                out[b if span_a >= span_b else a] = None
    return out
