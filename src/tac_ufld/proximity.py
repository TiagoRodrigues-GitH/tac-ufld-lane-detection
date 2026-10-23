"""Range, closing speed, time to collision and headway of the objects ahead: the predictor behind a forward-
collision (FCW) or proximity warning.

Added 2026-10-19 (``docs/CIPO.md`` section 10). Works on the 2-D boxes of the CIPO detector (or the labels) in
ORIGINAL image pixels, the ego-lane polylines and the per-frame camera (``tac_ufld.cipo.CameraModel``).

* ``ground_range``          range of the road point at image row ``v``, with camera pitch theta (looking down > 0):
                            d = h / tan(theta + atan((v - c_y) / f_y)); theta = 0 gives d = f_y h / (v - c_y).
* ``horizon_from_lanes``    horizon row, hence pitch, from the vanishing point of the ego-lane boundaries (near parts).
* ``ground_range_sigma``    first-order error: sigma_d = (h + d^2/h) * sqrt((dalpha/dv sigma_v)^2 + sigma_theta^2),
                            i.e. about d^2/(f_y h) per pixel and d^2/h per radian of pitch: it grows with d^2.
* ``size_range``            d = f W / w_px from a size prior (vehicle width, pedestrian / cyclist height); its error
                            grows with d (prior spread) and d^2 (pixel error), more slowly than the contact model.
* ``fuse_ranges``           inverse-variance combination of the two.
* ``RangeKalman``           constant-velocity Kalman filter on (range, range rate), white-noise acceleration
                            discretised as in Saerkkae (2013), "Bayesian Filtering and Smoothing", ch. 4.
* ``CipoRangeTracker``      one filter per tracked CIPO: box association by IoU, innovation gating, reset on a new
                            object, prediction through missed frames.
* ``time_to_collision``, ``time_headway``, ``FcwMonitor``  TTC = d / (-d_dot) while closing, THW = d / v_ego, and a
                            debounced warning rule (thresholds: ``FcwParams``).
* ``lateral_position``, ``adjacent_close``  which lane a box is in and its lateral gap (m) to the ego lane and to the
                            ego car's side: a "vehicle close beside" flag for the forward-adjacent zone.
* ``ego_speed``             ego speed from consecutive OpenLane poses.

Assumptions are parameters, never hidden constants. Where a number is a design choice and not a published
requirement, its docstring says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tac_ufld.cipo import CameraModel, box_iou, ego_span, in_path_fraction

OPENLANE_TIME_UNIT_S = 1e-8     # OpenLane frame timestamps advance ~1e7 units per 0.1 s frame (measured 2026-10-19)


# -- monocular range ---------------------------------------------------------------------------------------------------

def pitch_from_horizon(v_horizon: float, camera: CameraModel) -> float:
    """Camera pitch (rad, looking down > 0) relative to the road plane whose horizon is at row ``v_horizon``."""
    return math.atan((camera.cy - v_horizon) / camera.fy)


def ground_range(v: float | np.ndarray, camera: CameraModel, pitch: float = 0.0) -> np.ndarray:
    """Forward range (m) of the road point at row ``v`` for a camera ``camera.height`` above a flat road, pitched
    down by ``pitch`` (rad) relative to it; inf at or above the horizon."""
    angle = pitch + np.arctan((np.asarray(v, dtype=np.float64) - camera.cy) / camera.fy)
    with np.errstate(divide="ignore"):
        return np.where(angle > 1e-6, camera.height / np.tan(np.maximum(angle, 1e-6)), np.inf)


def ground_range_sigma(v: float, camera: CameraModel, pitch: float = 0.0, sigma_v: float = 2.0,
                       sigma_pitch: float = math.radians(0.2)) -> float:
    """First-order standard deviation (m) of ``ground_range`` for a row error ``sigma_v`` (px) and a pitch error
    ``sigma_pitch`` (rad). The defaults are assumptions to be replaced by measured values (``docs/CIPO.md``)."""
    d = float(ground_range(v, camera, pitch))
    if not math.isfinite(d):
        return math.inf
    dv = float(v) - camera.cy
    dalpha_dv = camera.fy / (camera.fy ** 2 + dv ** 2)
    dd_dangle = camera.height + d * d / camera.height          # |d/dphi (h cot phi)| = h (1 + cot^2 phi)
    return dd_dangle * math.hypot(dalpha_dv * sigma_v, sigma_pitch)


def _fit_line(lane: np.ndarray, near_fraction: float) -> tuple[float, float] | None:
    """x = a y + b over the nearest ``near_fraction`` of the lane's rows."""
    pts = lane[np.argsort(lane[:, 1])].astype(np.float64)
    y0, y1 = pts[0, 1], pts[-1, 1]
    near = pts[pts[:, 1] >= y1 - near_fraction * (y1 - y0)]
    if len(near) < 2 or near[-1, 1] - near[0, 1] < 1e-6:
        return None
    a, b = np.polyfit(near[:, 1], near[:, 0], 1)
    return float(a), float(b)


def horizon_from_lanes(left: np.ndarray | None, right: np.ndarray | None, camera: CameraModel,
                       near_fraction: float = 0.5, max_pitch: float = math.radians(8.0)) -> float:
    """Horizon row from the vanishing point of the two ego boundaries (lines fitted to their near parts).

    Parallel lane lines on a planar road meet on the horizon of that plane, so this row gives the camera pitch
    relative to the road just ahead (it absorbs vehicle pitch and gentle slopes). nan if a boundary is missing,
    the lines do not converge upward, or the implied pitch exceeds ``max_pitch``."""
    if left is None or right is None or len(left) < 2 or len(right) < 2:
        return float("nan")
    fl, fr = _fit_line(left, near_fraction), _fit_line(right, near_fraction)
    if fl is None or fr is None or not fl[0] < fr[0]:      # left: x grows upward (a < 0); right: x shrinks
        return float("nan")
    v = (fr[1] - fl[1]) / (fl[0] - fr[0])
    if abs(pitch_from_horizon(v, camera)) > max_pitch:
        return float("nan")
    return float(v)


@dataclass(frozen=True)
class SizePrior:
    """Real size of an object class, seen in the image as ``dimension`` (``width`` or ``height``)."""

    dimension: str
    metres: float
    sigma: float


# Design assumptions (not published values): a passenger car seen from behind is about 1.8 m wide, an adult
# pedestrian about 1.7 m tall, a cyclist on the bicycle about 1.75 m tall. Width is unreliable for vehicles seen at an
# angle (their side adds to the box width) and for trucks; ``docs/CIPO.md`` measures the bias on OpenLane.
SIZE_PRIORS = {1: SizePrior("width", 1.8, 0.15), 2: SizePrior("height", 1.7, 0.10), 4: SizePrior("height", 1.75, 0.15)}


def size_range(box: Sequence[float], camera: CameraModel, prior: SizePrior, sigma_px: float = 2.0
               ) -> tuple[float, float]:
    """(range, sigma) from the box size: d = f W / w_px. sigma^2 = (d sigma_W / W)^2 + (d^2 sigma_px / (f W))^2."""
    x1, y1, x2, y2 = (float(c) for c in box)
    pixels, focal = (x2 - x1, camera.fx) if prior.dimension == "width" else (y2 - y1, camera.fy)
    if pixels <= 0:
        return math.inf, math.inf
    d = focal * prior.metres / pixels
    return d, math.hypot(d * prior.sigma / prior.metres, d * d * sigma_px / (focal * prior.metres))


def fuse_ranges(estimates: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Inverse-variance mean of (range, sigma) pairs; non-finite pairs are skipped. (inf, inf) if none is left."""
    ok = [(d, s) for d, s in estimates if math.isfinite(d) and math.isfinite(s) and s > 0]
    if not ok:
        return math.inf, math.inf
    w = np.array([1.0 / s ** 2 for _, s in ok])
    d = float(np.sum(w * np.array([d for d, _ in ok])) / np.sum(w))
    return d, float(1.0 / math.sqrt(np.sum(w)))


# -- tracking --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RangeKalmanParams:
    """``accel_psd``: spectral density q (m^2/s^3) of the white-noise relative acceleration; q = 1 lets the
    range rate drift about 1 m/s in 1 s. A tuning assumption: set it from data (CARLA ground truth)."""

    accel_psd: float = 2.0
    init_rate_sigma: float = 5.0      # m/s, prior spread of the closing speed of a new object
    gate_chi2: float = 9.0            # innovation gate (1 dof): 9 = 3 sigma; beyond it the object is treated as new


class RangeKalman:
    """Constant-velocity Kalman filter, state (range d, range rate d_dot). Discretised Wiener velocity model
    (Saerkkae 2013, ch. 4): F = [[1, dt], [0, 1]], Q = q [[dt^3/3, dt^2/2], [dt^2/2, dt]]; measurement z = d with a
    per-frame variance (the range models above give it)."""

    def __init__(self, params: RangeKalmanParams | None = None) -> None:
        self.params = params or RangeKalmanParams()
        self.x = np.zeros(2)
        self.P = np.eye(2)
        self.initialized = False

    def reset(self) -> None:
        self.initialized = False

    def start(self, z: float, r_var: float) -> None:
        self.x = np.array([z, 0.0])
        self.P = np.diag([r_var, self.params.init_rate_sigma ** 2])
        self.initialized = True

    def predict(self, dt: float) -> None:
        q = self.params.accel_psd
        f = np.array([[1.0, dt], [0.0, 1.0]])
        qm = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + qm

    def innovation(self, z: float, r_var: float) -> tuple[float, float]:
        return z - self.x[0], self.P[0, 0] + r_var

    def update(self, z: float, r_var: float) -> None:
        y, s = self.innovation(z, r_var)
        k = self.P[:, 0] / s
        self.x = self.x + k * y
        self.P = self.P - np.outer(k, self.P[0, :])
        self.P = 0.5 * (self.P + self.P.T)

    @property
    def range(self) -> float:
        return float(self.x[0])

    @property
    def range_rate(self) -> float:
        return float(self.x[1])

    @property
    def sigmas(self) -> tuple[float, float]:
        return float(math.sqrt(max(self.P[0, 0], 0.0))), float(math.sqrt(max(self.P[1, 1], 0.0)))


@dataclass(frozen=True)
class TrackState:
    range: float              # m, filtered (inf when there is no object)
    range_rate: float         # m/s, < 0 while closing
    range_sigma: float
    rate_sigma: float
    age: int                  # frames since the track started
    new_track: bool           # this frame started a new object
    measured: bool            # this frame had a measurement (else: prediction only)


NO_TRACK = TrackState(math.inf, 0.0, math.inf, math.inf, 0, False, False)


class CipoRangeTracker:
    """Range and closing speed of the CIPO over a stream (one instance per camera stream).

    Each frame gives the CIPO box (or None) and its range measurement. The box continues the current track if it
    overlaps the previous one (IoU >= ``iou_gate``) and its range passes the filter's innovation gate; otherwise a
    new track starts (another object became the CIPO). Without a box the filter predicts for up to ``max_missed``
    frames, then the track ends."""

    def __init__(self, params: RangeKalmanParams | None = None, iou_gate: float = 0.3, max_missed: int = 5) -> None:
        self.kf = RangeKalman(params)
        self.iou_gate, self.max_missed = iou_gate, max_missed
        self.reset()

    def reset(self) -> None:
        self.kf.reset()
        self.box: np.ndarray | None = None
        self.age = 0
        self.missed = 0

    def update(self, box: Sequence[float] | None, range_m: float, range_sigma: float, dt: float) -> TrackState:
        if box is None or not math.isfinite(range_m) or not math.isfinite(range_sigma):
            if not self.kf.initialized:
                return NO_TRACK
            self.missed += 1
            if self.missed > self.max_missed:
                self.reset()
                return NO_TRACK
            self.kf.predict(dt)
            self.age += 1
            return self._state(new=False, measured=False)
        box = np.asarray(box, dtype=np.float64)
        r_var = range_sigma ** 2
        same = False
        if self.kf.initialized and self.box is not None and box_iou(box[None], self.box[None])[0, 0] >= self.iou_gate:
            self.kf.predict(dt)
            y, s = self.kf.innovation(range_m, r_var)
            same = y * y / s <= self.kf.params.gate_chi2
        if same:
            self.kf.update(range_m, r_var)
            self.age += 1
        else:
            self.kf.start(range_m, r_var)
            self.age = 1
        self.box, self.missed = box, 0
        return self._state(new=not same, measured=True)

    def _state(self, new: bool, measured: bool) -> TrackState:
        rs, vs = self.kf.sigmas
        return TrackState(self.kf.range, self.kf.range_rate, rs, vs, self.age, new, measured)


# -- warnings --------------------------------------------------------------------------------------------------------

def time_to_collision(range_m: float, range_rate: float, min_closing_speed: float = 0.0) -> float:
    """TTC = d / (-d_dot) while closing faster than ``min_closing_speed`` (m/s); inf otherwise. Constant-velocity
    TTC: it ignores the lead vehicle's deceleration, so it is optimistic when the lead brakes."""
    closing = -range_rate
    if not math.isfinite(range_m) or closing <= max(min_closing_speed, 1e-9):
        return math.inf
    return range_m / closing


def time_headway(range_m: float, ego_speed: float, min_speed: float = 0.5) -> float:
    """THW = d / v_ego (s); inf below ``min_speed`` m/s (standing still)."""
    if not math.isfinite(range_m) or ego_speed < min_speed:
        return math.inf
    return range_m / ego_speed


@dataclass(frozen=True)
class FcwParams:
    """Warning thresholds.

    ``ttc_s``: NHTSA's NCAP forward-collision-warning confirmation test passes a car only if the alert comes at a
    TTC of at least 2.1 s (stopped lead vehicle, ego at 72.4 km/h), 2.4 s (decelerating lead vehicle) and 2.0 s
    (lead vehicle at 32.2 km/h, ego at 72.4 km/h) (NHTSA confirmation test reports, e.g. Dynamic Research, 2012). A
    threshold of 2.4 s is therefore the least that can pass all three; the pipeline latency (detection, filter
    convergence, ``confirm_frames``) and the constant-velocity TTC's optimism toward a braking lead come on top, so
    calibrate the margin in simulation. ISO 15623 defines the FCW performance requirements (not reproduced here).

    ``thw_s``: headway warning threshold. No verified published value was found, so it is off (None) by default.

    ``min_closing_speed``, ``confirm_frames``, ``max_rate_sigma``: noise guards, design assumptions."""

    ttc_s: float = 2.4
    thw_s: float | None = None
    min_closing_speed: float = 0.5
    confirm_frames: int = 2
    max_rate_sigma: float = 3.0


@dataclass(frozen=True)
class FcwOutput:
    ttc: float
    thw: float
    warning: bool
    reason: str               # "" | "ttc" | "thw" | "ttc+thw"


class FcwMonitor:
    """Streaming FCW rule on a ``TrackState``: warn when TTC (or THW, if enabled) stays under its threshold for
    ``confirm_frames`` consecutive frames. TTC is used only once the closing speed is known well enough
    (``rate_sigma <= max_rate_sigma``)."""

    def __init__(self, params: FcwParams | None = None) -> None:
        self.params = params or FcwParams()
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def update(self, state: TrackState, ego_speed: float) -> FcwOutput:
        p = self.params
        ttc = (time_to_collision(state.range, state.range_rate, p.min_closing_speed)
               if state.rate_sigma <= p.max_rate_sigma else math.inf)
        thw = time_headway(state.range, ego_speed)
        reasons = [name for name, hit in (("ttc", ttc < p.ttc_s), ("thw", p.thw_s is not None and thw < p.thw_s))
                   if hit]
        self.count = self.count + 1 if reasons else 0
        return FcwOutput(ttc, thw, self.count >= p.confirm_frames, "+".join(reasons))


# -- vehicles beside ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LateralInfo:
    lane: str                 # "ego" | "left" | "right" | "unknown" (contact row at or above the horizon)
    range: float              # m, forward range of the contact row (flat road, given pitch)
    lateral: float            # m, lateral position of the box centre (left > 0), at that range
    gap_to_lane: float        # m, nearest box edge to the ego-lane boundary on its side (< 0: overlaps; nan: ego)
    gap_to_ego: float         # m, from the nearest box edge to the ego car's side (ego_half_width from the centre)


def lateral_position(box: Sequence[float], left: np.ndarray | None, right: np.ndarray | None,
                     camera: CameraModel, pitch: float = 0.0, ego_half_width: float = 0.95,
                     lane_width_m: float = 3.5) -> LateralInfo:
    """Lane and lateral gaps of an object box, from its contact row and the ego-lane boundaries.

    Image columns become metres at the contact row's range: X = (u - c_x) d / f_x (flat road, small pitch).
    ``ego_half_width`` (assumption: a passenger car about 1.9 m wide, camera on its centre line)."""
    x1, _, x2, v = (float(c) for c in box)
    d = float(ground_range(v, camera, pitch))
    xl, xr, source = ego_span(left, right, v, camera, lane_width_m)
    if source == "none" or not math.isfinite(d):
        return LateralInfo("unknown", math.inf, math.nan, math.nan, math.nan)
    metres = d / camera.fx
    lateral = (camera.cx - 0.5 * (x1 + x2)) * metres                 # left of the camera > 0
    if in_path_fraction(box, left, right, camera, lane_width_m) >= 0.5:
        lane = "ego"
    else:
        lane = "left" if 0.5 * (x1 + x2) < 0.5 * (xl + xr) else "right"
    if lane == "right":
        gap_lane, edge = (x1 - xr) * metres, x1
    else:
        gap_lane, edge = (xl - x2) * metres, x2
    gap_ego = abs((edge - camera.cx) * metres) - ego_half_width
    if lane == "ego" or (x1 <= camera.cx <= x2):
        gap_ego = -ego_half_width                                    # straight ahead: no lateral clearance
    return LateralInfo(lane, d, lateral, gap_lane if lane != "ego" else math.nan, gap_ego)


@dataclass(frozen=True)
class AdjacentParams:
    """Forward-adjacent proximity flag. A front camera sees only the zone AHEAD of the ego car's front; the blind
    spot beside and behind the driver (the zones of ISO 17387 lane-change decision aids) needs side or rear sensors.
    Both thresholds are design assumptions, to calibrate in CARLA."""

    max_range: float = 15.0          # m
    max_gap_to_ego: float = 1.0      # m


def adjacent_close(info: LateralInfo, params: AdjacentParams | None = None) -> bool:
    p = params or AdjacentParams()
    return (info.lane in ("left", "right") and info.range <= p.max_range and math.isfinite(info.gap_to_ego)
            and info.gap_to_ego <= p.max_gap_to_ego)


# -- ego motion -------------------------------------------------------------------------------------------------------

def ego_speed(pose_prev: np.ndarray, pose: np.ndarray, dt: float) -> float:
    """Speed (m/s) from two 4x4 vehicle-to-world poses ``dt`` seconds apart (OpenLane ``pose``)."""
    if dt <= 0:
        return math.nan
    step = np.asarray(pose, dtype=np.float64)[:3, 3] - np.asarray(pose_prev, dtype=np.float64)[:3, 3]
    return float(np.linalg.norm(step) / dt)
