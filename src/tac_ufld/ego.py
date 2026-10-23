"""Ego-lane analysis on top of a lane detector: where the car is in its lane and whether it is changing lanes.

Added 2026-10-18 (review). Works on the per-slot polylines that any model of this project outputs
(``postprocess.lanes_from_prediction``: slot 0 = ego-left boundary, slot 1 = ego-right boundary, original image
pixels) or on the ground truth, so the same code measures the upper bound (perfect lanes) and the real system.

* ``ego_lane_state``    one frame -> the lane the car is following: its centre and width at the bottom of the
                        road band, the car's lateral offset in lane widths (0 = centred, +0.5 = on the right
                        boundary) and a heading cue (how far the lane centre moves between the near and far rows).
* ``LaneChangeDetector`` a stream of states -> per-frame manoeuvre (-1 = moving to the LEFT lane, +1 = to the
                        RIGHT lane, 0 = keeping the lane), as in the ELAS ``laneChange`` label. A manoeuvre starts
                        when the car drifts toward a boundary (offset and lateral speed beyond thresholds, i.e. a
                        short time to line crossing) or when a boundary passes under the camera; it ends once the car
                        is centred again (in the new lane, or back in the old one = an aborted drift).
* ``lane_keeping``      lane departure warning (time to line crossing, ISO 17361 idea) and a lane keeping assist
                        steering request, suppressed during a detected lane change.
* ``event_metrics``     per-frame decisions vs. labels -> event recall, direction accuracy, false alarms per
                        minute, detection delay and frame-level F1.

Conventions: the camera is assumed to sit on the car's centre line, so the car is at image column ``camera_x``
(ELAS: "the car position is expected to be in the center of the image", Berriel et al. 2017). Direction signs
follow ELAS (verified on the data: ``laneChange = +1`` episodes end with one more lane on the car's left).

Limits: a lane change is a lateral manoeuvre; turning at an intersection usually loses the lane markings, so it is
not detected here (it needs ego-motion: IMU/CAN yaw rate, visual odometry, or a dataset with turn labels).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class EgoLaneState:
    left_x: float          # x of the ego-left boundary at the reference row (nan if not seen)
    right_x: float         # x of the ego-right boundary at the reference row (nan if not seen)
    center_x: float        # lane centre at the reference row (nan if no boundary)
    width: float           # lane width at the reference row, px (measured, or remembered if one side is missing)
    offset: float          # (camera_x - center_x) / width: 0 = centred, > 0 = car right of the centre
    heading_px: float      # vanishing point of the lane minus camera_x (px): > 0 = the lane runs to the RIGHT of
                           # the car's axis (the car points left of the road), so it must steer right
    boundaries: tuple[float, ...] = ()   # x of every visible boundary at the reference row (for crossings)


def vanishing_x(xl: float, xr: float, fl: float, fr: float, y_ref: float, y_far: float) -> float:
    """x where the two boundaries, as straight lines through their reference- and far-row points, meet.

    For a forward camera on the car's axis, ``vanishing_x - camera_x`` measures the yaw between car and lane
    (zero when the car points along the lane, wherever it is laterally). The difference of the lane centres at two
    rows does NOT: by perspective it mostly measures the lateral offset (fixed 2026-10-19, CARLA session review)."""
    if not all(np.isfinite(v) for v in (xl, xr, fl, fr)) or y_far == y_ref:
        return float("nan")
    sl, sr = (fl - xl) / (y_far - y_ref), (fr - xr) / (y_far - y_ref)
    if abs(sl - sr) < 1e-9:
        return float("nan")
    return float(xl + sl * (xr - xl) / (sl - sr))


def x_at(lane: np.ndarray | None, y: float) -> float:
    """x of a polyline at row ``y``; linear extrapolation from the two nearest points beyond its extent."""
    if lane is None or len(lane) < 2:
        return float("nan")
    pts = lane[np.argsort(lane[:, 1])]
    ys, xs = pts[:, 1].astype(np.float64), pts[:, 0].astype(np.float64)
    if ys[0] <= y <= ys[-1]:
        return float(np.interp(y, ys, xs))
    i, j = (-2, -1) if y > ys[-1] else (0, 1)
    if ys[j] == ys[i]:
        return float(xs[j])
    return float(xs[j] + (y - ys[j]) * (xs[j] - xs[i]) / (ys[j] - ys[i]))


def ego_lane_state(lanes: list[np.ndarray | None], y_ref: float, y_far: float, camera_x: float,
                   width_memory: float | None = None) -> EgoLaneState:
    """Ego-lane geometry of one frame from the ego-left (slot 0) and ego-right (slot 1) boundaries."""
    left = lanes[0] if len(lanes) > 0 else None
    right = lanes[1] if len(lanes) > 1 else None
    xl, xr = x_at(left, y_ref), x_at(right, y_ref)
    if np.isfinite(xl) and np.isfinite(xr) and xr > xl:
        width = xr - xl
        center = 0.5 * (xl + xr)
    elif width_memory and np.isfinite(xl):
        width, center = width_memory, xl + 0.5 * width_memory
    elif width_memory and np.isfinite(xr):
        width, center = width_memory, xr - 0.5 * width_memory
    else:
        width, center = width_memory or float("nan"), float("nan")
    fl, fr = x_at(left, y_far), x_at(right, y_far)
    vp = vanishing_x(xl, xr, fl, fr, y_ref, y_far)
    heading = vp - camera_x if np.isfinite(vp) else float("nan")
    offset = (camera_x - center) / width if np.isfinite(center) and width and np.isfinite(width) else float("nan")
    seen = tuple(float(x) for x in (xl, xr) if np.isfinite(x))
    return EgoLaneState(xl, xr, center, float(width), float(offset), float(heading), seen)


@dataclass
class LaneChangeParams:
    fps: float = 29.97              # ELAS frame rate
    smooth: int = 5                 # moving-average window of the offset, frames
    speed_window: int = 10          # frames over which the lateral speed is measured
    start_offset: float = 0.18      # |offset| (lane widths) beyond which a drift can start a manoeuvre
    start_speed: float = 0.10       # lateral speed toward the boundary, lane widths per second
    max_tlc_s: float = 2.5          # ... and time to line crossing below this, seconds
    confirm: int = 5                # consecutive frames the drift condition must hold
    end_offset: float = 0.15        # |offset| under which the car counts as centred again
    end_confirm: int = 10           # consecutive centred frames that end a manoeuvre
    crossing_jump: float = 60.0     # px: max motion of a boundary between frames to call it the same line
    timeout_s: float = 8.0          # a manoeuvre longer than this is closed
    # Offset flip (added 2026-10-19 after the ROD_S01 preview): a lane model re-assigns the crossing line from
    # one slot to the other a few frames early and abruptly, so no boundary "passes under the camera" and the
    # offset jumps from +a to -b. A sign flip of at least ``flip_jump`` lane widths, both sides beyond
    # ``flip_min`` and confirmed by the median of the last ``flip_confirm`` frames, is a crossing toward the side
    # the car was approaching (offset +0.3 -> -0.3: the car crossed its RIGHT boundary).
    flip_min: float = 0.12
    flip_jump: float = 0.35
    flip_confirm: int = 3


@dataclass
class LaneChangeDetector:
    """Streaming lane-change / lateral-drift detector (one instance per video stream)."""

    camera_x: float
    params: LaneChangeParams = field(default_factory=LaneChangeParams)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.offset, self.speed = float("nan"), 0.0
        self._offsets: deque[float] = deque(maxlen=self.params.smooth)
        self._smoothed: deque[float] = deque(maxlen=self.params.speed_window + 1)
        self._prev_boundaries: tuple[float, ...] = ()
        self._raw: deque[float] = deque(maxlen=self.params.flip_confirm)
        self.state, self._drift, self._centred, self._age = 0, 0, 0, 0
        self.crossed = False

    def _offset_flip(self, raw: float) -> int:
        """+1 / -1 when the offset flipped sign by a lane re-assignment (see ``LaneChangeParams.flip_*``)."""
        p = self.params
        self._raw.append(raw)
        if len(self._raw) < self._raw.maxlen or len(self._smoothed) < self._raw.maxlen:
            return 0
        ref = self._smoothed[-self._raw.maxlen]   # smoothed offset before the last ``flip_confirm`` frames
        recent = float(np.median(self._raw))
        if (ref * recent < 0 and abs(ref) >= p.flip_min and abs(recent) >= p.flip_min
                and abs(recent - ref) >= p.flip_jump):
            return 1 if ref > 0 else -1
        return 0

    def _crossing(self, boundaries: tuple[float, ...]) -> int:
        """+1 if a boundary passed under the camera from right to left (car moved RIGHT), -1 for the opposite."""
        out = 0
        for x in boundaries:
            if not self._prev_boundaries:
                break
            prev = min(self._prev_boundaries, key=lambda p: abs(p - x))
            if abs(prev - x) <= self.params.crossing_jump and (prev - self.camera_x) * (x - self.camera_x) < 0:
                out = 1 if prev > self.camera_x else -1
        if boundaries:
            self._prev_boundaries = boundaries
        return out

    def update(self, s: EgoLaneState) -> int:
        p = self.params
        crossing = self._crossing(s.boundaries)
        if not crossing and np.isfinite(s.offset):
            crossing = self._offset_flip(s.offset)
        if crossing and self._offsets:
            # the reference lane changes at the crossing: restart the offset history in the new lane
            self._offsets.clear()
            self._smoothed.clear()
            self._raw.clear()
            self._drift = 0
        if np.isfinite(s.offset):
            self._offsets.append(s.offset)
        if self._offsets:
            self._smoothed.append(float(np.mean(self._offsets)))
        off = self._smoothed[-1] if self._smoothed else float("nan")
        speed = 0.0
        if len(self._smoothed) > 1:
            speed = (self._smoothed[-1] - self._smoothed[0]) * p.fps / (len(self._smoothed) - 1)
        self.offset, self.speed = off, speed  # smoothed offset (lane widths) and lateral speed (widths / s)

        if self.state == 0:
            if crossing:
                self.state, self.crossed, self._age = crossing, True, 0
            elif np.isfinite(off):
                side = 1 if off > 0 else -1
                toward = side * speed
                tlc = (0.5 - abs(off)) / toward if toward > 1e-6 else np.inf
                drifting = abs(off) >= p.start_offset and toward >= p.start_speed and tlc <= p.max_tlc_s
                self._drift = self._drift + 1 if drifting else 0
                if self._drift >= p.confirm:
                    self.state, self.crossed, self._age, self._drift = side, False, 0, 0
        else:
            self._age += 1
            if crossing:
                self.crossed = True
                if crossing != self.state:      # crossed the other way: the car went back
                    self.state = crossing
            centred = np.isfinite(off) and abs(off) <= p.end_offset
            self._centred = self._centred + 1 if centred else 0
            if self._centred >= p.end_confirm or self._age > p.timeout_s * p.fps:
                self.state, self._centred, self._age = 0, 0, 0
        return self.state


class EgoLaneTracker:
    """One video stream: lanes per frame -> (``EgoLaneState``, manoeuvre decision). Remembers the lane width
    (median of the last 30 frames with both boundaries) for frames where one boundary is not seen."""

    def __init__(self, camera_x: float, y_ref: float, y_far: float, params: LaneChangeParams | None = None) -> None:
        self.y_ref, self.y_far = y_ref, y_far
        self.detector = LaneChangeDetector(camera_x, params or LaneChangeParams())
        self._widths: deque[float] = deque(maxlen=30)

    def reset(self) -> None:
        self.detector.reset()
        self._widths.clear()

    def update(self, lanes: list[np.ndarray | None]) -> tuple[EgoLaneState, int]:
        memory = float(np.median(self._widths)) if self._widths else None
        s = ego_lane_state(lanes, self.y_ref, self.y_far, self.detector.camera_x, memory)
        if np.isfinite(s.left_x) and np.isfinite(s.right_x) and s.right_x > s.left_x:
            self._widths.append(s.right_x - s.left_x)
        return s, self.detector.update(s)


@dataclass
class LaneKeepingParams:
    # The WHEEL, not the car's centre, crosses the line first: at |offset| = 0.5 - half_width (lane widths).
    # 0.26 = 0.9 m half-width of a car in a 3.5 m lane; set it from the vehicle and the measured lane width.
    half_width: float = 0.26
    warn_tlc_s: float = 1.0       # LDW: warn when the WHEEL will reach the line within this time (seconds) ...
    warn_margin: float = 0.05     # ... or is closer than this to it (lane widths, ~0.18 m at 3.5 m)
    k_offset: float = 1.0         # LKA steering gain on the lateral offset (lane widths)
    k_heading: float = 0.5        # ... and on the heading cue (vanishing point minus camera_x, in lane widths)
    max_steer: float = 0.3        # |steer| limit, normalised like CARLA's VehicleControl.steer (-1 left, +1 right)


@dataclass(frozen=True)
class LaneKeepingOutput:
    tlc_s: float          # time until the wheel reaches the line on the side the car moves to (inf if not moving
                          # toward it, 0 if already over it)
    ldw: int              # lane departure warning: -1 drifting out on the left, +1 on the right, 0 none
    steer: float          # LKA steering request (normalised, see ``LaneKeepingParams.max_steer``); 0 if unknown
    active: bool          # False while the turn signal is on (an intended manoeuvre) or no lane is seen


def lane_keeping(state: EgoLaneState, detector: LaneChangeDetector, turn_signal: int = 0,
                 params: LaneKeepingParams | None = None) -> LaneKeepingOutput:
    """Lane departure warning (LDW) and lane keeping assist (LKA) from the ego-lane state of one frame.

    LDW follows the time-to-line-crossing idea of ISO 17361, measured to the WHEEL: the wheel reaches the line at
    ``|offset| = 0.5 - half_width``; warn when that happens within ``warn_tlc_s`` at the current lateral speed, or
    the wheel is within ``warn_margin`` of the line. LKA is a proportional law on the lateral offset plus the
    heading cue (a Stanley-type controller in normalised units): ``steer = -k_offset * offset + k_heading *
    heading / width``, where heading is the lane's vanishing point minus the camera axis (``vanishing_x``). The
    gains are placeholders to be tuned in closed loop (CARLA); a metric controller needs the camera calibration
    (ELAS ``config.xml`` gives IPM points; CARLA gives the intrinsics).

    Suppression comes ONLY from ``turn_signal`` (-1 left, +1 right; CAN in a car, the ego vehicle's light state in
    CARLA), and an active signal suppresses BOTH sides: after the crossing the offset changes sign while the
    signal still points to the old side, so a side-specific rule would re-arm LDW and steer the car back. An
    unintended drift looks exactly like the start of a lane change, so the vision-based ``LaneChangeDetector``
    never silences the warning; its state is reported separately for logging. (Revised 2026-10-19 after the
    CARLA session's review: heading cue, wheel-based crossing, both-side suppression.)"""
    p = params or LaneKeepingParams()
    off, speed = detector.offset, detector.speed
    if not np.isfinite(off) or not np.isfinite(state.width) or state.width <= 0:
        return LaneKeepingOutput(float("inf"), 0, 0.0, False)
    side = 1 if off > 0 else -1
    toward = side * speed
    to_line = 0.5 - p.half_width - abs(off)            # lane widths left before the wheel touches the line
    tlc = 0.0 if to_line <= 0 else (to_line / toward if toward > 1e-6 else float("inf"))
    intended = turn_signal != 0
    ldw = side if not intended and (tlc <= p.warn_tlc_s or to_line <= p.warn_margin) else 0
    heading = state.heading_px / state.width if np.isfinite(state.heading_px) else 0.0
    steer = 0.0 if intended else float(np.clip(-p.k_offset * off + p.k_heading * heading, -p.max_steer, p.max_steer))
    return LaneKeepingOutput(float(tlc), int(ldw), steer, not intended)


def segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Runs of equal non-zero labels: (start, end_inclusive, value)."""
    out, i = [], 0
    while i < len(labels):
        if labels[i] != 0:
            j = i
            while j + 1 < len(labels) and labels[j + 1] == labels[i]:
                j += 1
            out.append((i, j, int(labels[i])))
            i = j + 1
        else:
            i += 1
    return out


def event_metrics(pred: np.ndarray, gt: np.ndarray, fps: float) -> dict[str, float]:
    """Event- and frame-level scores of per-frame manoeuvre decisions (-1/0/+1) against labels."""
    gt_events, pred_events = segments(gt), segments(pred)
    hit, right_dir, delays = 0, 0, []
    for s, e, v in gt_events:
        overlap = [(ps, pv) for ps, pe, pv in pred_events if ps <= e and pe >= s]
        if overlap:
            hit += 1
            right_dir += any(pv == v for _, pv in overlap)
            delays.append(max(min(ps for ps, _ in overlap) - s, 0) / fps)
    false_alarms = sum(1 for ps, pe, _ in pred_events if not any(ps <= e and pe >= s for s, e, _ in gt_events))
    minutes = len(gt) / fps / 60.0
    f1 = {}
    for v, name in ((-1, "left"), (1, "right")):
        tp = int(((pred == v) & (gt == v)).sum())
        fp, fn = int(((pred == v) & (gt != v)).sum()), int(((pred != v) & (gt == v)).sum())
        f1[f"frame_f1_{name}"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return {"gt_events": len(gt_events), "detected": hit, "direction_correct": right_dir,
            "event_recall": hit / len(gt_events) if gt_events else float("nan"),
            "false_alarms": false_alarms, "false_alarms_per_min": false_alarms / minutes if minutes else float("nan"),
            "median_delay_s": float(np.median(delays)) if delays else float("nan"), **f1}
