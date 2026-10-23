"""CARLA closed-loop pieces that need no simulator: geometry, driver laws, scenario logic, metrics, the detector
socket protocol, and a kinematic closed loop through the real ego-lane code."""
import math
import socket
import threading

import numpy as np
import pytest

from tac_ufld.ego import EgoLaneTracker, LaneChangeParams, LaneKeepingParams, lane_keeping
from tac_ufld.sim import detector_bridge as db
from tac_ufld.sim import lka_loop as ll
from tac_ufld.sim import lka_metrics as lm

RIG = ll.CameraRig()


def carla_matrix(x=0.0, y=0.0, z=0.0, pitch=0.0, yaw=0.0, roll=0.0):
    """carla.Transform.get_matrix() (LibCarla geom/Transform.h), angles in degrees."""
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    return np.array([[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
                     [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
                     [sp, -cp * sr, cp * cr, z],
                     [0.0, 0.0, 0.0, 1.0]])


def world_to_camera(x, y, yaw, rig=RIG):
    return np.linalg.inv(carla_matrix(x, y, 0.0, yaw=yaw) @ carla_matrix(rig.x_m, 0.0, rig.z_m, pitch=rig.pitch_deg))


# --------------------------------------------------------------------------- geometry

def test_camera_horizon_rows_and_ground_distance():
    assert RIG.focal == pytest.approx(320 / math.tan(math.radians(35)))
    assert RIG.ground_row(1e9) == pytest.approx(0.6 * 480, abs=1e-3)       # horizon at 60 % of the height
    for d in (4.0, 10.0, 30.0):
        assert RIG.ground_distance(RIG.ground_row(d)) == pytest.approx(d)
    assert RIG.ground_distance(0.5 * 480) == math.inf                      # above the horizon
    lane_px = (RIG.ground_row(RIG.ground_distance(479)) - 0.6 * 480) * 3.5 / RIG.z_m
    assert 480 < lane_px < 560                                             # ELAS: ~520 px at the bottom row


def test_projection_matches_ground_row_and_sides():
    w2c = world_to_camera(10.0, 5.0, 0.0)
    d = 12.0
    ahead = np.array([[10.0 + RIG.x_m + d, 5.0, 0.0], [10.0 + RIG.x_m + d, 6.75, 0.0], [0.0, 5.0, 0.0]])
    uv = RIG.project(ahead, w2c)
    np.testing.assert_allclose(uv[0], [320.0, RIG.ground_row(d)], atol=1e-6)
    assert uv[1, 0] > 320 and uv[1, 1] == pytest.approx(uv[0, 1])        # +y (right) -> right in the image
    assert np.isnan(uv[2]).all()                                           # behind the camera
    turned = RIG.project(ahead[:1], world_to_camera(10.0, 5.0, 10.0))      # car yawed right: point appears left
    assert turned[0, 0] < 320


def test_image_polyline_keeps_points_inside():
    uv = np.array([[np.nan, np.nan], [-5.0, 400.0], [100.0, 470.0], [120.0, 400.0], [700.0, 300.0]])
    np.testing.assert_allclose(ll.image_polyline(uv, 640, 480), [[100, 470], [120, 400]])
    assert ll.image_polyline(uv[:3], 640, 480) is None


def test_vehicle_frame_and_driver_laws():
    fwd, right = ll.to_vehicle_frame(0.0, 10.0, 90.0)   # facing +y: a point at +y is straight ahead
    assert fwd == pytest.approx(10.0) and right == pytest.approx(0.0, abs=1e-9)
    assert ll.pure_pursuit_steer(10.0, 1.0, 2.9, 70.0) > 0 > ll.pure_pursuit_steer(10.0, -1.0, 2.9, 70.0)
    assert ll.pure_pursuit_steer(10.0, 0.0, 2.9, 70.0) == 0.0
    assert ll.heading_hold_steer(0.0, 1, 0.4, 16.7) > 0                     # start drifting right
    held = math.degrees(math.asin(0.4 / 16.7))
    assert ll.heading_hold_steer(held, 1, 0.4, 16.7) == pytest.approx(0.0, abs=1e-12)
    ff = ll.curvature_steer(1 / 180, 2.9, 70.0)                            # right-hand R 180 m curve
    assert ff == pytest.approx(math.atan(2.9 / 180) / math.radians(70.0)) and ff > 0
    assert ll.heading_hold_steer(held, 1, 0.4, 16.7, feedforward=ff) == pytest.approx(ff)
    assert ll.signed_offset((1.0, 2.0), (0.0, 0.0), (0.0, 1.0)) == pytest.approx(2.0)
    assert ll.wrap_deg(350.0) == pytest.approx(-10.0) and ll.wrap_deg(-190.0) == pytest.approx(170.0)


def test_turn_signal_from_light_bits():
    assert ll.turn_signal_from_lights(ll.LEFT_BLINKER | 1) == -1
    assert ll.turn_signal_from_lights(ll.RIGHT_BLINKER) == 1
    assert ll.turn_signal_from_lights(ll.LEFT_BLINKER | ll.RIGHT_BLINKER) == 0   # hazard lights
    assert ll.turn_signal_from_lights(0) == 0


def test_speed_controller_holds_speed():
    ctl, v = ll.SpeedController(), 10.0
    for _ in range(600):   # plant: acceleration = 4 * throttle - drag
        throttle, brake = ctl.update(16.7, v, 1 / 30)
        v += (4.0 * throttle - 6.0 * brake - 0.1 * v) / 30
    assert v == pytest.approx(16.7, abs=0.2)


def test_steering_disturbance_is_seeded_and_scaled():
    a, b = ll.SteeringDisturbance(0.01, 2.0, seed=3), ll.SteeringDisturbance(0.01, 2.0, seed=3)
    xs = np.array([a.step(1 / 30) for _ in range(30 * 600)])
    assert xs[:50].tolist() == [b.step(1 / 30) for _ in range(50)]
    assert xs.std() == pytest.approx(0.01, rel=0.15)
    assert ll.SteeringDisturbance(0.0).step(0.1) == 0.0


def test_route_shape():
    s = np.arange(0, 400, ll.ROUTE_STEP_M)
    straight = np.c_[s, np.zeros_like(s)]
    radius = 150.0
    ang = s / radius
    arc = np.c_[radius * np.sin(ang), radius * (1 - np.cos(ang))]
    assert ll.route_curvature(straight).max() == pytest.approx(0.0, abs=1e-12)
    assert np.median(ll.route_curvature(arc)) == pytest.approx(1 / radius, rel=0.02)
    assert ll.route_turn_deg(arc) == pytest.approx(math.degrees(ang[-1]), rel=0.03)
    assert ll.route_score(straight, "straight", 50) < ll.route_score(arc, "straight", 50)
    lead_in = np.r_[np.c_[-s[::-1][:-1] - 5, np.zeros(len(s) - 1)], arc]   # straight 395 m, then the arc
    assert ll.route_score(lead_in, "curve", 50) < 0                            # usable, turning
    assert ll.route_score(arc, "curve", 50) == math.inf                        # no straight warm-up stretch
    tight = np.c_[60 * np.sin(s / 60), 60 * (1 - np.cos(s / 60))]
    assert ll.route_score(tight, "curve", 50) == math.inf                      # radius below 100 m


# --------------------------------------------------------------------------- scenario director

def gt_at(offset, width=3.5, half=1.0, heading=0.0):
    return ll.GroundTruth(offset, width, heading, half, 16.7)


def test_director_ldw_drift_sequence():
    d = ll.ScenarioDirector((ll.Event("drift_ldw", 1.0, 1),))
    both = lambda side: True  # noqa: E731
    assert d.step(-0.5, gt_at(0.0), both).phase == "warmup"
    assert d.step(0.5, gt_at(0.0), both) == ll.KEEP
    assert d.step(1.0, gt_at(0.0), both).phase == "align"
    cmd = d.step(3.1, gt_at(0.1), both)
    assert (cmd.phase, cmd.driver, cmd.side, cmd.lka_enabled) == ("drift_ldw", "drift", 1, False)
    assert d.step(5.0, gt_at(0.8), both).phase == "drift_ldw"       # body on the line from 0.75 m
    assert d.step(5.4, gt_at(0.9), both).phase == "drift_ldw"       # still within the 0.5 s hold
    assert d.step(5.6, gt_at(0.95), both).phase == "recover"
    assert d.step(10.5, gt_at(0.0), both).phase == "recover"         # 5 s of LKA recovery
    assert d.step(10.7, gt_at(0.0), both) == ll.KEEP


def test_director_lka_release_and_side_flip():
    d = ll.ScenarioDirector((ll.Event("drift_lka", 0.0, 1),))
    only_left = lambda side: side < 0  # noqa: E731
    d.step(0.0, gt_at(0.0), only_left)
    cmd = d.step(2.1, gt_at(0.0), only_left)
    assert cmd.phase == "drift_lka_setup" and cmd.side == -1         # no lane on the right: drift left
    assert d.step(3.0, gt_at(-0.2), only_left).phase == "drift_lka_setup"   # margin 0.55 m
    assert d.step(3.1, gt_at(-0.3), only_left).phase == "drift_lka"         # margin 0.45 m: released
    assert d.step(9.2, gt_at(0.0), only_left) == ll.KEEP


def test_director_lane_change_signals_and_ends_centred():
    d = ll.ScenarioDirector((ll.Event("lane_change", 0.0, -1),))
    both = lambda side: True  # noqa: E731
    cmd = d.step(0.0, gt_at(0.0), both)
    assert (cmd.phase, cmd.turn_signal, cmd.driver) == ("lc_signal", -1, "none")
    cmd = d.step(1.0, gt_at(-0.1), both)
    assert (cmd.phase, cmd.driver, cmd.side, cmd.turn_signal) == ("lane_change", "change", -1, -1)
    assert d.step(2.0, gt_at(-1.7), both).phase == "lane_change"
    assert d.step(2.1, gt_at(1.7), both).phase == "lane_change"      # centre crossed: new lane
    assert d.step(2.5, gt_at(0.3), both).phase == "lane_change"      # centred, holding
    assert d.step(3.0, gt_at(0.2), both) == ll.KEEP


# --------------------------------------------------------------------------- kinematic closed loop

def kinematic_loop(y0, yaw0, seconds, keeping, lka=True, fps=30.0, v=16.7, wheelbase=2.9, curvature=0.0):
    """Bicycle model on a flat lane (centre y = 0, or an arc of ``curvature``), seen through ``CameraRig``,
    tracked by ``EgoLaneTracker`` and steered by ``lane_keeping``. Returns per-frame (offset, steer, ldw)."""
    tracker = EgoLaneTracker(RIG.width / 2, 0.95 * RIG.height, 0.72 * RIG.height, LaneChangeParams(fps=fps))
    x, y, yaw = 0.0, y0, yaw0
    out = []
    for _ in range(int(seconds * fps)):
        s = np.arange(x, x + 60.0, 1.0)
        centre = 0.5 * curvature * s ** 2
        w2c = world_to_camera(x, y, yaw)
        lanes = [ll.image_polyline(RIG.project(np.c_[s, centre + side * 1.75, np.zeros_like(s)], w2c), 640, 480)
                 for side in (-1, 1)]
        state, _ = tracker.update(lanes)
        lk = lane_keeping(state, tracker.detector, 0, keeping)
        steer = lk.steer if lka else 0.0
        offset = y - 0.5 * curvature * x ** 2
        out.append((offset, steer, lk.ldw))
        delta = steer * math.radians(70.0)
        x += v * math.cos(math.radians(yaw)) / fps
        y += v * math.sin(math.radians(yaw)) / fps
        yaw += math.degrees(v * math.tan(delta) / wheelbase) / fps
    return np.array(out)


def test_kinematic_lka_centres_the_car():
    keeping = LaneKeepingParams(k_offset=0.6, k_heading=0.5, half_width=1.0 / 3.5)
    run = kinematic_loop(y0=0.6, yaw0=0.0, seconds=8.0, keeping=keeping)
    assert run[0, 1] < 0                                  # right of centre: steers left
    assert abs(run[-60:, 0]).max() < 0.05                 # centred after 6 s
    assert (abs(run[:, 0]) + 1.0 < 1.75).all()            # never on the line


def test_kinematic_lka_holds_a_curve():
    keeping = LaneKeepingParams(k_offset=0.6, k_heading=0.5, half_width=1.0 / 3.5)
    run = kinematic_loop(y0=0.0, yaw0=0.0, seconds=8.0, keeping=keeping, curvature=1 / 300)
    assert run[-30:, 1].mean() > 0                        # right-hand curve: steady right steer
    assert abs(run[:, 0]).max() < 0.5


def test_kinematic_ldw_warns_before_the_wheel_crosses():
    keeping = LaneKeepingParams(half_width=1.0 / 3.5)
    heading = math.degrees(math.asin(0.4 / 16.7))         # 0.4 m/s toward the right line, LKA off
    run = kinematic_loop(y0=0.0, yaw0=heading, seconds=5.0, keeping=keeping, lka=False)
    cross = int(np.argmax(run[:, 0] + 1.0 > 1.75))
    warn = int(np.argmax(run[:, 2] == 1))
    assert cross > 0 and 0 < warn < cross
    assert (cross - warn) / 30.0 > 0.5                    # more than half a second of warning


def test_events_merge_same_side_gaps_only():
    sig = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, -1, -1, 0])
    assert lm.events(sig) == [(1, 2, 1), (4, 4, 1), (8, 8, 1), (9, 10, -1)]
    assert lm.events(sig, merge=2) == [(1, 4, 1), (8, 8, 1), (9, 10, -1)]


def test_crossing_is_one_event_through_a_lane_change():
    # moving right: the nearest lane switches when the centre crosses, the offset flips sign
    off = np.array([0.0, 0.5, 0.9, 1.5, -1.6, -0.9, -0.5, 0.0])
    width, half = np.full(8, 3.5), np.full(8, 1.0)
    assert lm.on_line(off, width, half).tolist() == [0, 0, 1, 1, -1, -1, 0, 0]
    assert lm.crossing_events(off, width, half) == [(2, 5, 1)]
    np.testing.assert_allclose(lm.lane_margin(off, width, half)[:3], [0.75, 0.25, -0.15])


def _log(n, **cols):
    base = {k: np.zeros(n) for k in lm.COLUMNS if k not in lm.TEXT_COLUMNS}
    base.update(phase=np.array(["keep"] * n, dtype=object), driver=np.array(["none"] * n, dtype=object),
                t_s=np.arange(n) / 10.0, distance_m=np.arange(n) * 2.0, gt_lane_width_m=np.full(n, 3.5),
                half_width_m=np.full(n, 1.0), lka_enabled=np.ones(n), speed_mps=np.full(n, 20.0))
    base.update(cols)
    return base


def test_summary_ldw_lead_time_and_rates():
    n = 100  # 10 s at 10 fps, 198 m
    off = np.zeros(n)
    off[40:60] = np.linspace(0.0, 1.2, 20)      # drift right; body on the line from 0.75 m
    phase = np.array(["keep"] * n, dtype=object)
    phase[40:60] = "drift_ldw"
    ldw = np.zeros(n)
    ldw[45:58] = 1
    m = lm.summarize(_log(n, gt_offset_m=off, phase=phase, ldw=ldw, lka_enabled=(phase == "keep").astype(float)),
                     fps=10.0)
    onset = int(np.argmax(np.abs(off) + 1.0 > 1.75))
    assert m["crossings"] == 1 and m["crossings_lka"] == 0          # the drift had the LKA off
    assert m["ldw_drift_crossings"] == 1 and m["ldw_missed"] == 0
    assert m["ldw_lead_median_s"] == pytest.approx((onset - 45) / 10.0)
    assert m["distance_km"] == pytest.approx(0.198)
    assert m["ldw_per_km"] == pytest.approx(1 / 0.198)
    assert np.isnan(m["time_to_first_departure_s"])


def test_summary_lka_failure_intended_change_and_warmup():
    n = 60
    t = np.arange(n) / 10.0 - 1.0                  # first 10 frames are warm-up
    off = np.zeros(n)
    off[5:8] = 1.0                                 # crossing during warm-up: ignored
    off[30:35] = 0.9                               # LKA-controlled crossing at t = 2.0 s
    off[45:50] = -0.9                              # intended: turn signal left
    signal = np.zeros(n)
    signal[44:52] = -1
    phase = np.array(["keep"] * n, dtype=object)
    phase[44:52] = "lane_change"
    m = lm.summarize(_log(n, t_s=t, gt_offset_m=off, turn_signal=signal, phase=phase), fps=10.0)
    assert m["crossings"] == 2 and m["crossings_intended"] == 1 and m["crossings_lka"] == 1
    assert m["time_to_first_departure_s"] == pytest.approx(2.0)
    assert m["lane_changes"] == 1


def test_log_roundtrip(tmp_path):
    rows = [{"frame": 0, "t_s": -0.1, "phase": "warmup", "gt_offset_m": float("nan")},
            {"frame": 1, "t_s": 0.0, "phase": "keep", "gt_offset_m": 0.25}]
    lm.write_log(tmp_path / "log.csv", rows)
    log = lm.read_log(tmp_path / "log.csv")
    assert log["phase"].tolist() == ["warmup", "keep"]
    assert np.isnan(log["gt_offset_m"][0]) and log["gt_offset_m"][1] == 0.25


class _FakeHandler:
    def __init__(self):
        self.resets = 0

    def info(self):
        return {"variant": "fake", "num_lanes": 2}

    def reset(self):
        self.resets += 1

    def infer(self, rgb, frame_index, timestamp=None):
        x = float(rgb[..., 0].mean())
        return db.LaneOutput([np.array([[x, 10.0], [x, 20.0]]), None], [0.9, None], 1.5, frame_index % 2)


def test_bridge_roundtrip_over_a_socket():
    handler, ready = _FakeHandler(), threading.Event()
    port = {}

    def on_ready(p):
        port["p"] = p
        ready.set()

    server = threading.Thread(target=db.serve, args=(handler, "127.0.0.1", 0, on_ready), daemon=True)
    server.start()
    assert ready.wait(10)
    client = db.RemoteLaneDetector("127.0.0.1", port["p"], connect_timeout_s=10)
    assert client.info()["variant"] == "fake"
    frame = np.full((4, 6, 3), 7, dtype=np.uint8)
    out = client.infer(frame, frame_index=3)
    np.testing.assert_allclose(out.lanes[0], [[7, 10], [7, 20]])
    assert out.lanes[1] is None and out.confidence == [0.9, None] and out.fallbacks == 1
    client.reset()
    assert handler.resets == 1
    with pytest.raises(RuntimeError, match="does not match"):
        client._call({"cmd": "infer", "shape": [4, 6, 3], "frame_index": 0}, b"\x00" * 5)
    db.send_message(client.sock, {"cmd": "shutdown"})
    assert db.recv_message(client.sock)[0]["ok"]
    client.sock.close()
    server.join(10)
    assert not server.is_alive()


def test_framing_rejects_oversized_header():
    a, b = socket.socketpair()
    with a, b:
        a.sendall(db.LENGTH.pack(db.MAX_HEADER_BYTES + 1))
        with pytest.raises(ValueError):
            db.recv_message(b)
