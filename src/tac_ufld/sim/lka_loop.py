"""Closed-loop lane keeping in CARLA: camera -> lane detector -> ``tac_ufld.ego`` (ego lane, LDW, LKA) -> steering.

Each simulation frame (synchronous mode, ``RunConfig.fps``):

1. the front camera image goes to the detector (``detector_bridge``: the trained model, possibly in another Python)
   or, in ``oracle`` mode, the ego-lane boundaries of the CARLA road map are projected into the image instead
   (perfect perception: the upper bound of the controller, and the way to tune its gains);
2. ``EgoLaneTracker`` + ``lane_keeping`` turn the lanes into an LDW decision and an LKA steering request; the turn
   signal is read back from the ego vehicle's light state, as a car reads it from CAN;
3. a scripted driver (``ScenarioDirector``) decides who steers: the LKA, or the "driver" for a scripted drift or an
   intended lane change; a PI controller holds a constant speed;
4. ground truth from the road map (lateral offset from the lane centre, lane width, heading error) is logged next
   to the system's outputs (``lka_metrics.COLUMNS``) and scored by ``lka_metrics.summarize``.

Pure parts (camera geometry, driver laws, route shape, the scenario state machine) need neither CARLA nor torch and
are unit-tested in ``tests/test_carla_lka.py``. The CARLA part runs in the CARLA client venv (Python 3.12):
``carla`` is imported lazily. Conventions follow CARLA (Unreal): x forward, y right, z up, yaw grows to the right;
offsets are > 0 when the car is right of the lane centre and steering > 0 turns right, as in ``tac_ufld.ego``.
"""

from __future__ import annotations

import math
import queue
import random
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tac_ufld.ego import EgoLaneTracker, LaneChangeParams, LaneKeepingParams, ego_lane_state, lane_keeping
from tac_ufld.sim import lka_metrics

if TYPE_CHECKING:  # the CARLA client is needed to drive, not to import this module or test its pure parts
    import carla

ROUTE_STEP_M = 5.0
SENSOR_TIMEOUT_S = 60.0
LEFT_BLINKER, RIGHT_BLINKER = 32, 16      # carla.VehicleLightState bits
LEAD_BLUEPRINTS = ("vehicle.carlamotors.carlacola", "vehicle.tesla.cybertruck", "vehicle.audi.etron")


# --------------------------------------------------------------------------- camera geometry (no CARLA)

@dataclass(frozen=True)
class CameraRig:
    """Front camera on the car's axis. The defaults reproduce the ELAS view the models were trained on (640x480,
    flat-road horizon near 60 % of the height, ego lane ~520 px wide at the bottom row, no hood in view)."""
    width: int = 640
    height: int = 480
    fov_deg: float = 70.0
    x_m: float = 2.0            # ahead of the vehicle origin (CARLA: centre of the car at ground level)
    z_m: float = 1.3            # above the vehicle origin
    horizon_frac: float = 0.6   # image row of the flat-ground horizon / height

    @property
    def focal(self) -> float:
        return self.width / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))

    @property
    def pitch_deg(self) -> float:
        """CARLA pitch (> 0 looks up) that puts the flat-ground horizon at ``horizon_frac``."""
        return math.degrees(math.atan((self.horizon_frac - 0.5) * self.height / self.focal))

    def intrinsics(self) -> np.ndarray:
        f = self.focal
        return np.array([[f, 0.0, self.width / 2.0], [0.0, f, self.height / 2.0], [0.0, 0.0, 1.0]])

    def project(self, points: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
        """World points (N, 3) -> pixels (N, 2); NaN for points less than 0.5 m in front of the camera."""
        cam = world_to_camera @ np.c_[points, np.ones(len(points))].T
        xyz = np.stack([cam[1], -cam[2], cam[0]])  # Unreal camera axes -> x right, y down, z forward
        uv = (self.intrinsics() @ xyz)[:2] / np.where(xyz[2] > 0.5, xyz[2], np.nan)
        return uv.T

    def ground_row(self, distance_m: float) -> float:
        """Image row of flat ground ``distance_m`` ahead of the camera."""
        below = math.atan2(self.z_m, distance_m) + math.radians(self.pitch_deg)
        return self.height / 2.0 + self.focal * math.tan(below)

    def ground_distance(self, row: float) -> float:
        """Inverse of ``ground_row``: metres ahead of the camera seen at ``row`` (inf at or above the horizon)."""
        below = math.atan((row - self.height / 2.0) / self.focal) - math.radians(self.pitch_deg)
        return self.z_m / math.tan(below) if below > 0 else float("inf")


def image_polyline(uv: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Projected points inside the image as an (N, 2) float32 polyline, None if fewer than 2."""
    ok = np.isfinite(uv).all(axis=1)
    ok[ok] &= (uv[ok, 0] >= 0) & (uv[ok, 0] < width) & (uv[ok, 1] >= 0) & (uv[ok, 1] < height)
    return uv[ok].astype(np.float32) if ok.sum() >= 2 else None


# --------------------------------------------------------------------------- vehicle geometry and driver laws

def signed_offset(point: tuple[float, float], centre: tuple[float, float], right: tuple[float, float]) -> float:
    """Lateral distance of ``point`` from the lane centre along the lane's right vector (> 0 = right)."""
    return (point[0] - centre[0]) * right[0] + (point[1] - centre[1]) * right[1]


def wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def to_vehicle_frame(dx: float, dy: float, yaw_deg: float) -> tuple[float, float]:
    """World displacement -> (forward, right) metres in the vehicle frame."""
    yaw = math.radians(yaw_deg)
    return math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy


def pure_pursuit_steer(forward: float, right: float, wheelbase_m: float, max_steer_deg: float) -> float:
    """Normalised steering (CARLA ``VehicleControl.steer``) that arcs through a target point in the vehicle frame."""
    lookahead = math.hypot(forward, right)
    if lookahead < 1e-3:
        return 0.0
    alpha = math.atan2(right, forward)
    delta = math.atan2(2.0 * wheelbase_m * math.sin(alpha), lookahead)
    return float(np.clip(delta / math.radians(max_steer_deg), -1.0, 1.0))


def heading_hold_steer(heading_deg: float, side: int, v_lat: float, speed: float, gain: float = 1.0,
                       feedforward: float = 0.0) -> float:
    """Scripted inattentive drift: hold the heading that gives a lateral speed ``v_lat`` (m/s) toward ``side``
    (Euro NCAP LSS tests use 0.2-0.5 m/s, then release). ``gain``: normalised steer per radian of heading error;
    ``feedforward``: the steer that follows the road's curvature (``curvature_steer``), without which the heading
    error on a 180 m curve is larger than the drift heading itself."""
    target = side * math.asin(min(v_lat / max(speed, 1.0), 0.5))
    return float(np.clip(feedforward + gain * (target - math.radians(heading_deg)), -1.0, 1.0))


def curvature_steer(curvature: float, wheelbase_m: float, max_steer_deg: float) -> float:
    """Normalised steer of a kinematic bicycle on a path of signed ``curvature`` (1/m, > 0 turns right)."""
    return float(np.clip(math.atan(wheelbase_m * curvature) / math.radians(max_steer_deg), -1.0, 1.0))


def turn_signal_from_lights(flags: int, left_bit: int = LEFT_BLINKER, right_bit: int = RIGHT_BLINKER) -> int:
    """-1 left blinker, +1 right blinker, 0 none or both (hazard lights are not a manoeuvre)."""
    left, right = bool(flags & left_bit), bool(flags & right_bit)
    return -1 if left and not right else 1 if right and not left else 0


@dataclass
class SpeedController:
    """PI cruise control on throttle with anti-windup; brakes only when clearly too fast."""
    kp: float = 0.35
    ki: float = 0.08
    brake_gain: float = 0.3
    integral: float = 5.0          # starts near the throttle that holds highway speed (ki * integral = 0.4)

    def update(self, target: float, speed: float, dt: float) -> tuple[float, float]:
        err = target - speed
        u = self.kp * err + self.ki * self.integral
        if (0.0 < u < 1.0) or (u >= 1.0 and err < 0) or (u <= 0.0 and err > 0):
            self.integral += err * dt
        brake = float(np.clip(-self.brake_gain * err, 0.0, 1.0)) if err < -1.0 else 0.0
        return float(np.clip(u, 0.0, 1.0)), brake


@dataclass
class SteeringDisturbance:
    """Slow random steering offset (Ornstein-Uhlenbeck: crosswind, road crown, tyre pull): stationary standard
    deviation ``sigma`` (normalised steer), correlation time ``tau_s``. Seeded, so the oracle and every detector
    see the same disturbance; without it a car spawned centred on a straight road never leaves the centre."""
    sigma: float = 0.003
    tau_s: float = 2.0
    seed: int = 1
    value: float = 0.0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def step(self, dt: float) -> float:
        a = math.exp(-dt / self.tau_s)
        self.value = a * self.value + self.sigma * math.sqrt(1.0 - a * a) * float(self._rng.standard_normal())
        return self.value


def route_curvature(xy: np.ndarray, step_m: float = ROUTE_STEP_M, window: int = 4) -> np.ndarray:
    """|curvature| (1/m) along a polyline sampled every ``step_m``: heading change over ``window`` segments."""
    seg = np.diff(np.asarray(xy, dtype=np.float64), axis=0)
    heading = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    if len(heading) <= window:
        return np.zeros(0)
    return np.abs(heading[window:] - heading[:-window]) / (window * step_m)


def route_turn_deg(xy: np.ndarray) -> float:
    """Total turning along a polyline (degrees, both directions added)."""
    seg = np.diff(np.asarray(xy, dtype=np.float64), axis=0)
    return float(np.degrees(np.abs(np.diff(np.unwrap(np.arctan2(seg[:, 1], seg[:, 0])))).sum()))


def route_score(xy: np.ndarray, road: str, warmup_m: float, step_m: float = ROUTE_STEP_M,
                min_radius_m: float = 100.0) -> float:
    """Lower is better; inf = unusable. ``straight``: the largest curvature. ``curve``: minus the total turn, for
    routes that start on a straight stretch (the warm-up) and never curve tighter than ``min_radius_m``."""
    kappa = route_curvature(xy, step_m)
    if kappa.size == 0:
        return math.inf
    if road == "straight":
        return float(kappa.max())
    head = kappa[: max(int(warmup_m / step_m), 1)]
    if kappa.max() > 1.0 / min_radius_m or head.max() > 1.0 / 400.0:
        return math.inf
    return -route_turn_deg(xy)


# --------------------------------------------------------------------------- scenarios (no CARLA)

@dataclass(frozen=True)
class Event:
    kind: str           # "drift_ldw" (LDW test) | "drift_lka" (LKA test) | "lane_change" (intended, signalled)
    t_s: float          # start, seconds after the warm-up
    side: int = 1       # preferred side; flipped when there is no same-direction lane on that side
    v_lat: float = 0.4  # lateral speed of a drift, m/s


@dataclass(frozen=True)
class Scenario:
    name: str
    town: str
    road: str                    # "straight" | "curve": how the start point is chosen
    duration_s: float
    events: tuple[Event, ...] = ()
    lead_vehicle: bool = False
    neighbours: bool = False     # start in a lane with same-direction lanes on both sides
    map_layers: str = "all"      # "minimal": the _Opt map without buildings, props, foliage (roads always load)
    description: str = ""


SCENARIOS = {
    "straight": Scenario("straight", "Town04", "straight", 40.0,
                         description="highway, LKA only"),
    # full Town06 crashed the server on this 16 GB PC when switching maps; Town06_Opt without layers loads in 15 s
    "curves": Scenario("curves", "Town06", "curve", 40.0, map_layers="minimal",
                       description="highway curves (radius >= 100 m), LKA only"),
    "lead": Scenario("lead", "Town04", "straight", 40.0, lead_vehicle=True,
                     description="truck 15 m ahead in the same lane, hiding the far lane markings"),
    "drift": Scenario("drift", "Town04", "straight", 64.0, neighbours=True,
                      events=(Event("drift_ldw", 4.0, 1), Event("drift_ldw", 19.0, -1),
                              Event("drift_lka", 34.0, 1), Event("drift_lka", 49.0, -1)),
                      description="scripted unintended drifts at 0.4 m/s: two with the LKA off (LDW test), "
                                  "two released toward the line with the LKA on (LKA test)"),
    "lane_change": Scenario("lane_change", "Town04", "straight", 36.0, neighbours=True,
                            events=(Event("lane_change", 5.0, -1), Event("lane_change", 20.0, 1)),
                            description="intended lane changes with the turn signal on (left, then right)"),
}


@dataclass(frozen=True)
class GroundTruth:
    offset_m: float          # vehicle origin from the nearest lane centre, > 0 = right
    lane_width_m: float
    heading_deg: float       # vehicle yaw minus lane yaw, > 0 = pointing right of the lane
    half_width_m: float
    speed_mps: float

    def margin(self, side: int) -> float:
        """Metres between the body edge on ``side`` and the boundary on that side (< 0 = over the line)."""
        return 0.5 * self.lane_width_m - (side * self.offset_m + self.half_width_m)

    @property
    def on_line(self) -> int:
        return int(lka_metrics.on_line(np.array([self.offset_m]), np.array([self.lane_width_m]),
                                       np.array([self.half_width_m]))[0])


@dataclass(frozen=True)
class Command:
    phase: str
    lka_enabled: bool        # the LKA request is applied (it may still be suppressed by the turn signal)
    driver: str = "none"     # scripted driver: "none" | "follow" (lane centre) | "drift" | "change" (target lane)
    side: int = 0
    turn_signal: int = 0
    v_lat: float = 0.0


KEEP = Command("keep", True)


@dataclass(frozen=True)
class DirectorParams:
    align_s: float = 2.0                # driver centres the car before a drift (repeatable start, as in Euro NCAP)
    hold_after_crossing_s: float = 0.5  # LDW drift: keep drifting this long after the body reaches the line
    drift_timeout_s: float = 12.0
    recover_s: float = 5.0              # LKA back on after an LDW drift
    release_margin_m: float = 0.5       # LKA drift: driver lets go when the body edge is this close to the line
    release_s: float = 6.0              # LKA alone after the release
    signal_lead_s: float = 1.0          # turn signal on this long before the lane change starts
    centred_m: float = 0.4              # lane change done when centred in the new lane ...
    centred_hold_s: float = 0.5         # ... for this long
    lc_timeout_s: float = 10.0


class ScenarioDirector:
    """Who steers when: the scripted driver's state machine over a scenario's events (t < 0 is the warm-up)."""

    def __init__(self, events: tuple[Event, ...], params: DirectorParams | None = None) -> None:
        self.p = params or DirectorParams()
        self._queue = sorted(events, key=lambda e: e.t_s)
        self._event: Event | None = None
        self._side, self._stage, self._t0 = 0, "", 0.0
        self._crossed_at: float | None = None
        self._centred_since: float | None = None
        self._switched = False
        self._prev_offset: float | None = None

    def _centre_crossing(self, gt: GroundTruth) -> bool:
        """The nearest lane changed (the offset jumps by about a lane width when the car's centre crosses)."""
        prev, self._prev_offset = self._prev_offset, gt.offset_m
        return prev is not None and abs(gt.offset_m - prev) > 0.5 * gt.lane_width_m

    def _enter(self, t: float, stage: str) -> None:
        self._stage, self._t0 = stage, t

    def step(self, t: float, gt: GroundTruth, has_lane: Callable[[int], bool]) -> Command:
        crossed_centre = self._centre_crossing(gt)
        if t < 0:
            return Command("warmup", False, "follow")
        if self._event is None:
            if not self._queue or t < self._queue[0].t_s:
                return KEEP
            ev = self._queue.pop(0)
            side = ev.side or 1
            if not has_lane(side) and has_lane(-side):
                side = -side
            self._event, self._side = ev, side
            self._crossed_at, self._centred_since, self._switched = None, None, False
            self._enter(t, "signal" if ev.kind == "lane_change" else "align")
        if self._event.kind == "lane_change":
            return self._lane_change(t, gt, crossed_centre)
        return self._drift(t, gt)

    def _drift(self, t: float, gt: GroundTruth) -> Command:
        ev, side, p = self._event, self._side, self.p
        if self._stage == "align":
            if t - self._t0 < p.align_s:
                return Command("align", False, "follow")
            self._enter(t, "drift")
        if self._stage == "drift":
            if ev.kind == "drift_ldw":
                if gt.on_line and self._crossed_at is None:
                    self._crossed_at = t
                done = self._crossed_at is not None and t - self._crossed_at >= p.hold_after_crossing_s
                phase = "drift_ldw"
            else:
                done, phase = gt.margin(side) <= p.release_margin_m, "drift_lka_setup"
            if not done and t - self._t0 < p.drift_timeout_s:
                return Command(phase, False, "drift", side, 0, ev.v_lat)
            self._enter(t, "after")
        ldw_test = ev.kind == "drift_ldw"
        if t - self._t0 < (p.recover_s if ldw_test else p.release_s):
            return Command("recover" if ldw_test else "drift_lka", True)
        self._event = None
        return KEEP

    def _lane_change(self, t: float, gt: GroundTruth, crossed_centre: bool) -> Command:
        side, p = self._side, self.p
        if self._stage == "signal":
            if t - self._t0 < p.signal_lead_s:
                return Command("lc_signal", True, "none", side, side)
            self._enter(t, "change")
        self._switched |= crossed_centre
        if self._switched and abs(gt.offset_m) < p.centred_m:
            self._centred_since = t if self._centred_since is None else self._centred_since
        else:
            self._centred_since = None
        centred = self._centred_since is not None and t - self._centred_since >= p.centred_hold_s
        if centred or t - self._t0 >= p.lc_timeout_s:
            self._event = None
            return KEEP
        return Command("lane_change", True, "change", side, side)


@dataclass
class RunConfig:
    fps: float = 30.0                   # ELAS frame rate: the temporal models' history spacing as in training
    speed_kmh: float = 60.0
    warmup_s: float = 3.0               # ground-truth driver while the detector and the tracker fill their history
    rig: CameraRig = field(default_factory=CameraRig)
    y_ref_frac: float = 0.95            # EgoLaneTracker rows (fractions of the height)
    y_far_frac: float = 0.72
    keeping: LaneKeepingParams = field(default_factory=LaneKeepingParams)
    vehicle_half_width: bool = True     # set keeping.half_width from the car and the lane width at the start
    vehicle: str = "vehicle.lincoln.mkz_2020"
    weather: str = "ClearNoon"
    seed: int = 1
    town: str | None = None             # override the scenario's town
    duration_s: float | None = None     # override the scenario's duration
    map_layers: str | None = None       # override the scenario's "all" | "minimal"
    lead_gap_m: float = 15.0
    disturbance: float = 0.003          # SteeringDisturbance sigma (0 = none)
    shadow: bool = False                # the detector only rides along (logged: lanes, offset, LDW); ground-truth
                                        # lanes steer, so a failing detector cannot end the run
    snapshot_every: int = 0             # save an overlay JPEG every N frames (0 = never)
    video: bool = False                 # overlay MP4 of the whole run


# --------------------------------------------------------------------------- CARLA

def load_carla() -> Any:
    try:
        import carla
    except ImportError as exc:
        raise SystemExit("CARLA client not installed: use the Python 3.12 venv (C:\\CARLA\\venv312) with "
                         "`carla==0.9.16`, see docs/CARLA.md") from exc
    return carla


@contextmanager
def synchronous_world(client: carla.Client, town: str, fps: float, no_rendering: bool,
                      map_layers: str = "all") -> Iterator[carla.World]:
    """Load the town in synchronous mode; restore the previous settings on exit, even after errors."""
    carla_mod = load_carla()
    name = f"{town}_Opt" if map_layers == "minimal" else town
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != name:
        if map_layers == "minimal":
            world = client.load_world(name, map_layers=carla_mod.MapLayer.NONE)
        else:
            world = client.load_world(name)
    settings = world.get_settings()
    original = (settings.synchronous_mode, settings.fixed_delta_seconds, settings.no_rendering_mode)
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 1.0 / fps
    settings.no_rendering_mode = no_rendering
    world.apply_settings(settings)
    try:
        yield world
    finally:
        restore = world.get_settings()
        restore.synchronous_mode, restore.fixed_delta_seconds, restore.no_rendering_mode = original
        world.apply_settings(restore)


def same_direction_lane(wp: carla.Waypoint, side: int) -> carla.Waypoint | None:
    """The neighbouring driving lane on ``side`` (-1 left, +1 right) if traffic there goes the same way."""
    carla_mod = load_carla()
    lane = wp.get_left_lane() if side < 0 else wp.get_right_lane()
    ok = lane is not None and lane.lane_type == carla_mod.LaneType.Driving and lane.lane_id * wp.lane_id > 0
    return lane if ok else None


def walk_lane(wp: carla.Waypoint, length_m: float, step_m: float = ROUTE_STEP_M) -> np.ndarray | None:
    """xy of the lane centre every ``step_m`` from ``wp``; None if the lane ends earlier."""
    points, cur = [], wp
    for _ in range(int(length_m / step_m) + 1):
        loc = cur.transform.location
        points.append((loc.x, loc.y))
        ahead = cur.next(step_m)
        if not ahead:
            return None
        cur = ahead[0]
    return np.asarray(points)


def pick_start(world_map: carla.Map, scenario: Scenario, horizon_m: float, warmup_m: float, rng: random.Random,
               candidates: int = 800) -> tuple[carla.Waypoint, float]:
    """Best start among ``candidates`` random lane points (seeded): the straightest route for ``straight``, the
    most turning route that starts straight for ``curve``. Returns (waypoint, score)."""
    carla_mod = load_carla()
    points = [w for w in world_map.generate_waypoints(10.0)
              if w.lane_type == carla_mod.LaneType.Driving and not w.is_junction]
    rng.shuffle(points)
    best, best_score = None, math.inf
    for wp in points[:candidates]:
        if scenario.neighbours and not (same_direction_lane(wp, -1) and same_direction_lane(wp, 1)):
            continue
        xy = walk_lane(wp, horizon_m)
        score = route_score(xy, scenario.road, warmup_m) if xy is not None else math.inf
        if score < best_score:
            best, best_score = wp, score
    if best is None:
        raise RuntimeError(f"no usable {scenario.road} start for {scenario.name} in {world_map.name}")
    return best, best_score


def lane_curvature(wp: carla.Waypoint, ahead_m: float = 5.0) -> float:
    """Signed curvature of ``wp``'s lane (1/m, > 0 turns right) from the yaw change over ``ahead_m``."""
    nxt = wp.next(ahead_m)
    if not nxt:
        return 0.0
    return math.radians(wrap_deg(nxt[0].transform.rotation.yaw - wp.transform.rotation.yaw)) / ahead_m


def lane_edges(wp: carla.Waypoint, length_m: float = 60.0, step_m: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """3D points of the left and right edges of ``wp``'s lane, from ``wp`` forward."""
    left, right, cur = [], [], wp
    for _ in range(int(length_m / step_m)):
        t = cur.transform
        r, half = t.get_right_vector(), cur.lane_width / 2.0
        c = t.location
        left.append((c.x - r.x * half, c.y - r.y * half, c.z - r.z * half))
        right.append((c.x + r.x * half, c.y + r.y * half, c.z + r.z * half))
        ahead = cur.next(step_m)
        if not ahead:
            break
        cur = ahead[0]
    return np.asarray(left), np.asarray(right)


class LaneFollower:
    """Target points along one lane of the road map (scripted driver and lead vehicle)."""

    def __init__(self, waypoint: carla.Waypoint) -> None:
        self.wp = waypoint

    def target(self, location: carla.Location, lookahead_m: float) -> carla.Location:
        for _ in range(100):  # advance to the lane point nearest to the vehicle
            ahead = self.wp.next(1.0)
            if not ahead:
                break
            nearest = min(ahead, key=lambda w: w.transform.location.distance(location))
            if nearest.transform.location.distance(location) >= self.wp.transform.location.distance(location):
                break
            self.wp = nearest
        ahead = self.wp.next(lookahead_m)
        return (ahead[0] if ahead else self.wp).transform.location


class Rig:
    """Ego car (camera, collision sensor) and an optional lead vehicle; destroys everything it spawned on exit."""

    def __init__(self, client: carla.Client, world: carla.World, cfg: RunConfig, start: carla.Waypoint,
                 camera: bool, lead: bool) -> None:
        self._client, self._world, self._cfg = client, world, cfg
        self._actors: list[carla.Actor] = []
        self._frames: queue.Queue | None = None
        self.camera = None
        self.lead = None
        self.collisions: list[str] = []
        try:
            self._build(start, camera, lead)
        except BaseException:
            self.close()
            raise

    def _spawn(self, blueprint: carla.ActorBlueprint, transform: carla.Transform, **kw) -> carla.Actor:
        actor = self._world.spawn_actor(blueprint, transform, **kw)
        self._actors.append(actor)
        return actor

    def _build(self, start: carla.Waypoint, camera: bool, lead: bool) -> None:
        carla_mod, cfg, rig = load_carla(), self._cfg, self._cfg.rig
        library = self._world.get_blueprint_library()
        lift = carla_mod.Location(z=0.3)
        self.ego = self._spawn(library.find(cfg.vehicle), carla_mod.Transform(start.transform.location + lift,
                                                                              start.transform.rotation))
        self.mount = carla_mod.Transform(carla_mod.Location(x=rig.x_m, z=rig.z_m), carla_mod.Rotation(pitch=rig.pitch_deg))
        self.mount_matrix = np.array(self.mount.get_matrix())
        if camera:
            bp = library.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(rig.width))
            bp.set_attribute("image_size_y", str(rig.height))
            bp.set_attribute("fov", str(rig.fov_deg))
            self.camera = self._spawn(bp, self.mount, attach_to=self.ego)
            self._frames = queue.Queue()
            self.camera.listen(self._frames.put)
        collision = self._spawn(library.find("sensor.other.collision"), carla_mod.Transform(), attach_to=self.ego)
        collision.listen(lambda event: self.collisions.append(event.other_actor.type_id))
        if lead:
            ahead = start.next(cfg.lead_gap_m)
            names = [n for n in LEAD_BLUEPRINTS if library.filter(n)]
            self.lead = self._spawn(library.find(names[0]), carla_mod.Transform(ahead[0].transform.location + lift,
                                                                                ahead[0].transform.rotation))
            self.lead_follower = LaneFollower(ahead[0])
            self.lead_speed = SpeedController()
        self._world.tick()
        speed = cfg.speed_kmh / 3.6
        for actor in (self.ego, self.lead):
            if actor is not None:
                actor.set_target_velocity(actor.get_transform().get_forward_vector() * speed)
        physics = self.ego.get_physics_control()
        wheels = physics.wheels
        self.max_steer_deg = float(wheels[0].max_steer_angle) or 70.0
        front = np.mean([[w.position.x, w.position.y] for w in wheels[:2]], axis=0)
        rear = np.mean([[w.position.x, w.position.y] for w in wheels[2:]], axis=0)
        wheelbase = float(np.linalg.norm(front - rear)) / 100.0  # wheel positions are in centimetres
        self.wheelbase = wheelbase if 1.5 < wheelbase < 5.0 else 2.9
        self.half_width = float(self.ego.bounding_box.extent.y)

    def tick(self) -> np.ndarray | None:
        """Advance one frame; the camera image of that exact frame as RGB, if there is a camera."""
        frame = self._world.tick()
        if self._frames is None:
            return None
        image = self._frames.get(timeout=SENSOR_TIMEOUT_S)
        while image.frame < frame:
            image = self._frames.get(timeout=SENSOR_TIMEOUT_S)
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        return np.ascontiguousarray(bgra[:, :, 2::-1])

    def world_to_camera(self, vehicle_transform: carla.Transform) -> np.ndarray:
        return np.linalg.inv(np.array(vehicle_transform.get_matrix()) @ self.mount_matrix)

    def set_turn_signal(self, signal: int) -> None:
        carla_mod = load_carla()
        bits = LEFT_BLINKER if signal < 0 else RIGHT_BLINKER if signal > 0 else 0
        self.ego.set_light_state(carla_mod.VehicleLightState(bits))

    def drive_lead(self, ego_location: carla.Location, speed_mps: float, dt: float) -> float:
        """Lead vehicle follows its lane and holds the gap; returns the gap (m)."""
        carla_mod, cfg = load_carla(), self._cfg
        tf = self.lead.get_transform()
        gap = tf.location.distance(ego_location)
        vel = self.lead.get_velocity()
        target = float(np.clip(speed_mps + 0.5 * (cfg.lead_gap_m - gap), 0.5 * speed_mps, 1.5 * speed_mps))
        throttle, brake = self.lead_speed.update(target, math.hypot(vel.x, vel.y), dt)
        aim = self.lead_follower.target(tf.location, max(8.0, 0.8 * speed_mps))
        fwd, right = to_vehicle_frame(aim.x - tf.location.x, aim.y - tf.location.y, tf.rotation.yaw)
        steer = pure_pursuit_steer(fwd, right, 3.5, 70.0)
        self.lead.apply_control(carla_mod.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        return gap

    def close(self) -> None:
        carla_mod = load_carla()
        for actor in self._actors:
            if actor.type_id.startswith("sensor.") and actor.is_alive:
                actor.stop()
        self._client.apply_batch_sync([carla_mod.command.DestroyActor(a) for a in self._actors], False)
        self._actors = []

    def __enter__(self) -> Rig:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def draw_overlay(rgb: np.ndarray, lanes: list, gt_lanes: list, rows: tuple[float, float], lines: list[str]):
    """BGR frame: ground-truth ego lane (green), predicted ego-left / ego-right (red / blue), the tracker rows."""
    import cv2

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for lane in gt_lanes:
        if lane is not None:
            cv2.polylines(img, [lane.round().astype(np.int32).reshape(-1, 1, 2)], False, (0, 200, 0), 1, cv2.LINE_AA)
    for lane, colour in zip(lanes, ((0, 0, 255), (255, 80, 0))):
        if lane is not None:
            cv2.polylines(img, [lane.round().astype(np.int32).reshape(-1, 1, 2)], False, colour, 2, cv2.LINE_AA)
    w = img.shape[1]
    for y in rows:
        cv2.line(img, (w // 2 - 12, int(y)), (w // 2 + 12, int(y)), (0, 255, 255), 1)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class _OverlaySink:
    """Snapshots (JPEG every N frames) and an optional MP4; OpenCV cannot write to non-ASCII Windows paths, so files
    are encoded in memory or written to the temp folder first."""

    def __init__(self, out_dir: Path, tag: str, fps: float, size: tuple[int, int], every: int, video: bool) -> None:
        import cv2

        self.out_dir, self.tag, self.every = out_dir, tag, every
        self._writer, self._tmp = None, None
        if video:
            self._tmp = Path(tempfile.mkdtemp(prefix="carla_lka_"))
            self._writer = cv2.VideoWriter(str(self._tmp / "run.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if every:
            (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    def write(self, index: int, bgr: np.ndarray) -> None:
        import cv2

        if self._writer is not None:
            self._writer.write(bgr)
        if self.every and index % self.every == 0:
            ok, buf = cv2.imencode(".jpg", bgr)
            if ok:
                buf.tofile(str(self.out_dir / "frames" / f"{self.tag}_{index:05d}.jpg"))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            shutil.move(str(self._tmp / "run.mp4"), str(self.out_dir / f"{self.tag}.mp4"))
            shutil.rmtree(self._tmp, ignore_errors=True)


def run_scenario(client: carla.Client, scenario: Scenario, cfg: RunConfig, detector, out_dir: Path, tag: str,
                 log: Callable[[str], None] = print) -> tuple[list[dict], dict]:
    """Drive one scenario; returns the per-frame rows (``lka_metrics.COLUMNS``) and run information.

    ``detector``: an object with ``reset()`` and ``infer(rgb, frame_index) -> LaneOutput`` (``detector_bridge``),
    or None for the oracle (ground-truth lanes projected into the image)."""
    town = cfg.town or scenario.town
    duration = cfg.duration_s or scenario.duration_s
    camera = detector is not None or cfg.video or cfg.snapshot_every > 0
    speed = cfg.speed_kmh / 3.6
    layers = cfg.map_layers or scenario.map_layers
    with synchronous_world(client, town, cfg.fps, no_rendering=not camera, map_layers=layers) as world:
        world.set_weather(getattr(load_carla().WeatherParameters, cfg.weather))
        world_map = world.get_map()
        start, score = pick_start(world_map, scenario, speed * (duration + cfg.warmup_s), speed * cfg.warmup_s,
                                  random.Random(cfg.seed))
        loc = start.transform.location
        info = {"scenario": scenario.name, "tag": tag, "town": town, "map_layers": layers, "duration_s": duration,
                "start": {"x": round(loc.x, 1), "y": round(loc.y, 1), "road_id": start.road_id,
                          "lane_id": start.lane_id, "route_score": score}}
        log(f"[{tag}] {town}: start road {start.road_id} lane {start.lane_id} at ({loc.x:.0f}, {loc.y:.0f}), "
            f"route score {score:.4g}")
        with Rig(client, world, cfg, start, camera, scenario.lead_vehicle) as rig:
            rows, drive_info = _drive(world_map, rig, scenario, cfg, detector, duration, out_dir, tag, log)
    return rows, {**info, **drive_info}


def _drive(world_map: carla.Map, rig: Rig, scenario: Scenario, cfg: RunConfig, detector, duration: float,
           out_dir: Path, tag: str, log: Callable[[str], None]) -> tuple[list[dict], dict]:
    carla_mod = load_carla()
    cam, dt, v_target = cfg.rig, 1.0 / cfg.fps, cfg.speed_kmh / 3.6
    cx, y_ref, y_far = cam.width / 2.0, cfg.y_ref_frac * cam.height, cfg.y_far_frac * cam.height
    tracker = EgoLaneTracker(cx, y_ref, y_far, LaneChangeParams(fps=cfg.fps))
    steering_tracker = EgoLaneTracker(cx, y_ref, y_far, LaneChangeParams(fps=cfg.fps)) if cfg.shadow else tracker
    start_wp = world_map.get_waypoint(rig.ego.get_location())
    keeping = cfg.keeping
    if cfg.vehicle_half_width:
        keeping = replace(keeping, half_width=rig.half_width / start_wp.lane_width)
    director, speed_ctl = ScenarioDirector(scenario.events), SpeedController()
    noise = SteeringDisturbance(sigma=cfg.disturbance, seed=cfg.seed)
    if detector is not None:
        detector.reset()
    sink = None
    if cfg.video or cfg.snapshot_every:
        sink = _OverlaySink(out_dir, tag, cfg.fps, (cam.width, cam.height), cfg.snapshot_every, cfg.video)
    rows: list[dict] = []
    distance, prev_loc, off_road, signal_cmd = 0.0, None, 0, 0
    follower, prev_driver, terminated = None, None, ""
    n_frames = int(round((cfg.warmup_s + duration) * cfg.fps))
    try:
        for i in range(n_frames):
            rgb = rig.tick()
            t = i * dt - cfg.warmup_s
            tf = rig.ego.get_transform()
            loc, vel = tf.location, rig.ego.get_velocity()
            speed = math.hypot(vel.x, vel.y)
            distance += loc.distance(prev_loc) if prev_loc is not None else 0.0
            prev_loc = loc
            wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=carla_mod.LaneType.Driving)
            centre, right = wp.transform.location, wp.transform.get_right_vector()
            curvature = lane_curvature(wp)
            gt = GroundTruth(signed_offset((loc.x, loc.y), (centre.x, centre.y), (right.x, right.y)), wp.lane_width,
                             wrap_deg(tf.rotation.yaw - wp.transform.rotation.yaw), rig.half_width, speed)
            w2c = rig.world_to_camera(tf)
            gt_lanes = [image_polyline(cam.project(edge, w2c), cam.width, cam.height) if len(edge) > 1 else None
                        for edge in lane_edges(wp)]
            gt_state = ego_lane_state(gt_lanes, y_ref, y_far, cx)

            if detector is None:
                lanes, det_ms, fallbacks = gt_lanes, float("nan"), 0
            else:
                out = detector.infer(rgb, frame_index=i)
                lanes, det_ms, fallbacks = (list(out.lanes) + [None, None])[:2], out.latency_ms, out.fallbacks
            state, lc_state = tracker.update(lanes)
            signal = turn_signal_from_lights(int(rig.ego.get_light_state()))
            lk = lane_keeping(state, tracker.detector, signal, keeping)
            if steering_tracker is not tracker:   # shadow: logged decisions from the detector, steering from GT
                gt_track_state, _ = steering_tracker.update(gt_lanes)
                steer_request = lane_keeping(gt_track_state, steering_tracker.detector, signal, keeping).steer
            else:
                steer_request = lk.steer

            cmd = director.step(t, gt, lambda side: same_direction_lane(wp, side) is not None)
            if cmd.turn_signal != signal_cmd:   # read back through the light state from the next frame on
                rig.set_turn_signal(cmd.turn_signal)
                signal_cmd = cmd.turn_signal
            if cmd.driver != prev_driver and cmd.driver in ("follow", "change"):
                target_lane = same_direction_lane(wp, cmd.side) if cmd.driver == "change" else None
                follower = LaneFollower(target_lane or wp)
            prev_driver = cmd.driver
            driver_steer = 0.0
            if cmd.driver in ("follow", "change"):
                aim = follower.target(loc, max(8.0, (1.6 if cmd.driver == "change" else 0.8) * speed))
                fwd, rgt = to_vehicle_frame(aim.x - loc.x, aim.y - loc.y, tf.rotation.yaw)
                driver_steer = pure_pursuit_steer(fwd, rgt, rig.wheelbase, rig.max_steer_deg)
            elif cmd.driver == "drift":
                driver_steer = heading_hold_steer(gt.heading_deg, cmd.side, cmd.v_lat, speed, feedforward=curvature_steer(
                    curvature, rig.wheelbase, rig.max_steer_deg))
            lka_steer = steer_request if cmd.lka_enabled else 0.0
            disturbance = noise.step(dt)
            steer = float(np.clip(lka_steer + driver_steer + disturbance, -1.0, 1.0))
            throttle, brake = speed_ctl.update(v_target, speed, dt)
            rig.ego.apply_control(carla_mod.VehicleControl(throttle=throttle, steer=steer, brake=brake))
            lead_gap = rig.drive_lead(loc, v_target, dt) if rig.lead is not None else float("nan")

            row = {"frame": i, "t_s": t, "phase": cmd.phase, "lka_enabled": int(cmd.lka_enabled),
                   "driver": cmd.driver, "turn_signal": signal, "speed_mps": speed, "distance_m": distance,
                   "gt_offset_m": gt.offset_m, "gt_lane_width_m": gt.lane_width_m, "gt_heading_deg": gt.heading_deg,
                   "gt_on_line": gt.on_line, "gt_curvature": curvature, "gt_img_offset": gt_state.offset, "half_width_m": gt.half_width_m,
                   "pred_offset": tracker.detector.offset, "pred_offset_raw": state.offset,
                   "pred_heading": state.heading_px / state.width if state.width else float("nan"),
                   "pred_width_px": state.width, "lanes_seen": sum(lane is not None for lane in lanes),
                   "lc_state": lc_state, "tlc_s": min(lk.tlc_s, 99.0), "ldw": lk.ldw, "lka_active": int(lk.active),
                   "lka_steer": steer_request, "driver_steer": driver_steer, "disturbance": disturbance, "steer": steer,
                   "throttle": throttle,
                   "brake": brake, "lead_gap_m": lead_gap, "det_ms": det_ms, "fallbacks": fallbacks}
            rows.append(row)
            if sink is not None and rgb is not None:
                text = [f"{tag}  t {t:5.1f} s  {cmd.phase}  signal {signal:+d}  lc {lc_state:+d}",
                        f"offset pred {state.offset:+.2f}  gt {gt_state.offset:+.2f} lw  ({gt.offset_m:+.2f} m)",
                        f"steer {steer:+.3f} (lka {lka_steer:+.3f})  ldw {lk.ldw:+d}  tlc {min(lk.tlc_s, 99):.1f} s"]
                sink.write(i, draw_overlay(rgb, lanes, gt_lanes, (y_ref, y_far), text))
            if rig.collisions:
                terminated = f"collision with {rig.collisions[0]} at t={t:.1f} s"
                break
            off_road = off_road + 1 if abs(gt.offset_m) > gt.lane_width_m else 0
            if off_road > cfg.fps:
                terminated = f"left the road at t={t:.1f} s"
                break
    finally:
        if sink is not None:
            sink.close()
    if terminated:
        log(f"[{tag}] stopped: {terminated}")
    info = {"frames": len(rows), "terminated": terminated, "shadow": cfg.shadow, "wheelbase_m": rig.wheelbase,
            "max_steer_deg": rig.max_steer_deg, "half_width_m": rig.half_width,
            "lane_width_m": start_wp.lane_width, "keeping": asdict(keeping),
            "y_ref_px": y_ref, "y_far_px": y_far, "y_ref_ahead_m": cam.ground_distance(y_ref),
            "y_far_ahead_m": cam.ground_distance(y_far), "camera": asdict(cam), "camera_pitch_deg": cam.pitch_deg}
    return rows, info
