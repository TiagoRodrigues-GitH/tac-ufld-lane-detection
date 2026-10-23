"""Range, error model, horizon from lanes, size prior, Kalman tracking, TTC / THW warnings and side vehicles.

A synthetic camera (fx = fy = 2000, cx = 960, cy = 640, 2 m above a flat road), optionally pitched down, gives
every expected value in closed form."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tac_ufld.cipo import CameraModel
from tac_ufld.proximity import (SIZE_PRIORS, AdjacentParams, CipoRangeTracker, FcwMonitor, FcwParams, RangeKalman,
                                RangeKalmanParams, TrackState, adjacent_close, ego_speed, fuse_ranges, ground_range,
                                ground_range_sigma, horizon_from_lanes, lateral_position, pitch_from_horizon,
                                size_range, time_headway, time_to_collision)

CAM = CameraModel(fx=2000.0, fy=2000.0, cx=960.0, cy=640.0, height=2.0)


def project(z, y, pitch: float = 0.0, up=0.0):
    """Point at forward z, left y, ``up`` metres above the road -> (u, v) for a camera pitched down by ``pitch``."""
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y, dtype=np.float64)
    down = CAM.height - np.asarray(up, dtype=np.float64)
    forward = z * math.cos(pitch) + down * math.sin(pitch)
    below = down * math.cos(pitch) - z * math.sin(pitch)
    return CAM.cx - CAM.fx * y / forward, CAM.cy + CAM.fy * below / forward


def lane(lateral: float, pitch: float = 0.0) -> np.ndarray:
    z = np.linspace(6.0, 80.0, 40)
    u, v = project(z, np.full_like(z, lateral), pitch)
    return np.stack([u, v], axis=1)


def car_box(z: float, lateral: float, width: float = 1.8, height: float = 1.5, pitch: float = 0.0) -> list[float]:
    (ul, ur), (vb, vt) = project([z, z], [lateral + width / 2, lateral - width / 2], pitch)[0], \
        project([z, z], [lateral, lateral], pitch, up=[0.0, height])[1]
    return [float(ul), float(vt), float(ur), float(vb)]


# -- range models ------------------------------------------------------------------------------------------------------

def test_ground_range_with_and_without_pitch():
    z = np.array([10.0, 30.0, 60.0])
    _, v = project(z, np.zeros(3))
    np.testing.assert_allclose(ground_range(v, CAM), z)
    np.testing.assert_allclose(ground_range(v, CAM), CAM.ground_distance(v))
    pitch = math.radians(1.0)
    _, vp = project(z, np.zeros(3), pitch)
    np.testing.assert_allclose(ground_range(vp, CAM, pitch), z, rtol=1e-9)
    biased = ground_range(vp, CAM)                       # pitch ignored: the error grows with range
    rel = np.abs(biased - z) / z
    assert rel[0] < rel[1] < rel[2] and rel[2] > 0.3
    assert ground_range(CAM.cy - 10, CAM) == math.inf


def test_range_error_model_matches_finite_differences_and_grows_with_d_squared():
    for d in (10.0, 40.0):
        v = float(project([d], [0.0])[1][0])
        numeric_v = abs(float(ground_range(v + 0.5, CAM) - ground_range(v - 0.5, CAM)))
        assert ground_range_sigma(v, CAM, sigma_v=1.0, sigma_pitch=0.0) == pytest.approx(numeric_v, rel=0.01)
        eps = 1e-5
        numeric_p = abs(float(ground_range(v, CAM, eps) - ground_range(v, CAM, -eps))) / (2 * eps)
        assert ground_range_sigma(v, CAM, sigma_v=0.0, sigma_pitch=1.0) == pytest.approx(numeric_p, rel=0.01)
        # first-order textbook forms: d^2 / (f_y h) per pixel, d^2 / h per radian
        assert ground_range_sigma(v, CAM, sigma_v=1.0, sigma_pitch=0.0) == pytest.approx(
            d * d / (CAM.fy * CAM.height), rel=0.03)
    s20 = ground_range_sigma(float(project([20.0], [0.0])[1][0]), CAM)
    s40 = ground_range_sigma(float(project([40.0], [0.0])[1][0]), CAM)
    assert s40 / s20 == pytest.approx(4.0, rel=0.05)


def test_horizon_from_lanes_recovers_the_pitch():
    for deg in (-1.5, 0.0, 2.0):
        pitch = math.radians(deg)
        v_h = horizon_from_lanes(lane(1.75, pitch), lane(-1.75, pitch), CAM)
        assert v_h == pytest.approx(CAM.cy - CAM.fy * math.tan(pitch), abs=0.05)
        assert pitch_from_horizon(v_h, CAM) == pytest.approx(pitch, abs=1e-5)
        z = 35.0
        v = float(project([z], [0.0], pitch)[1][0])
        assert float(ground_range(v, CAM, pitch_from_horizon(v_h, CAM))) == pytest.approx(z, rel=1e-3)
    assert math.isnan(horizon_from_lanes(lane(1.75), None, CAM))
    assert math.isnan(horizon_from_lanes(lane(-1.75), lane(1.75), CAM))     # swapped: lines diverge upward


def test_size_prior_range_and_fusion():
    box = car_box(20.0, 0.0)
    d, s = size_range(box, CAM, SIZE_PRIORS[1])
    assert d == pytest.approx(20.0, rel=1e-6)
    assert s == pytest.approx(math.hypot(20 * 0.15 / 1.8, 400 * 2.0 / (2000 * 1.8)))
    ped = car_box(12.0, 2.0, width=0.5, height=1.7)
    assert size_range(ped, CAM, SIZE_PRIORS[2])[0] == pytest.approx(12.0, rel=1e-6)
    assert size_range([0, 0, 0, 10], CAM, SIZE_PRIORS[1]) == (math.inf, math.inf)
    d, s = fuse_ranges([(20.0, 2.0), (22.0, 2.0), (math.inf, 1.0)])
    assert d == pytest.approx(21.0) and s == pytest.approx(2.0 / math.sqrt(2))
    d, s = fuse_ranges([(20.0, 1.0), (30.0, 3.0)])
    assert d == pytest.approx((20 / 1 + 30 / 9) / (1 + 1 / 9))
    assert fuse_ranges([]) == (math.inf, math.inf)


# -- tracking ----------------------------------------------------------------------------------------------------------

def test_kalman_recovers_a_constant_closing_speed():
    rng = np.random.default_rng(0)
    kf, dt = RangeKalman(RangeKalmanParams(accel_psd=0.5)), 0.1
    t = np.arange(60) * dt
    truth = 60.0 - 5.0 * t
    z = truth + rng.normal(0.0, 1.0, len(t))
    rates = []
    for k, zk in enumerate(z):
        if k == 0:
            kf.start(zk, 1.0)
        else:
            kf.predict(dt)
            kf.update(zk, 1.0)
        rates.append(kf.range_rate)
    assert kf.range_rate == pytest.approx(-5.0, abs=1.0)
    assert kf.range == pytest.approx(truth[-1], abs=1.0)
    raw = np.diff(z) / dt                                # frame differences: std about sqrt(2) * 1 m / 0.1 s
    assert np.std(rates[-20:]) < 0.1 * np.std(raw)
    assert kf.sigmas[1] < 2.0


def test_tracker_continues_resets_and_coasts():
    tr = CipoRangeTracker(RangeKalmanParams(), iou_gate=0.3, max_missed=2)
    a = [900.0, 700.0, 1020.0, 840.0]
    s1 = tr.update(a, 20.0, 1.0, 0.1)
    assert s1.new_track and s1.age == 1
    s2 = tr.update([902.0, 701.0, 1022.0, 841.0], 19.6, 1.0, 0.1)
    assert not s2.new_track and s2.age == 2
    s3 = tr.update([100.0, 700.0, 220.0, 840.0], 19.0, 1.0, 0.1)    # a different object: no overlap
    assert s3.new_track
    s4 = tr.update([101.0, 700.0, 221.0, 840.0], 45.0, 1.0, 0.1)    # same place, impossible range jump
    assert s4.new_track and s4.range == pytest.approx(45.0)
    coast = [tr.update(None, math.nan, math.nan, 0.1) for _ in range(3)]
    assert not coast[0].measured and math.isfinite(coast[1].range)
    assert coast[2].range == math.inf                                # max_missed exceeded: track ended
    assert tr.update(None, math.nan, math.nan, 0.1).range == math.inf


def test_ttc_thw_and_the_warning_rule():
    assert time_to_collision(20.0, -5.0) == pytest.approx(4.0)
    assert time_to_collision(20.0, 1.0) == math.inf and time_to_collision(20.0, -0.3, 0.5) == math.inf
    assert time_headway(20.0, 10.0) == pytest.approx(2.0) and time_headway(20.0, 0.1) == math.inf
    mon = FcwMonitor(FcwParams(ttc_s=2.4, confirm_frames=2))
    closing = TrackState(10.0, -5.0, 0.5, 0.5, 10, False, True)        # TTC 2.0 s
    first, second = mon.update(closing, 15.0), mon.update(closing, 15.0)
    assert first.reason == "ttc" and not first.warning and second.warning
    unsure = TrackState(10.0, -5.0, 0.5, 4.0, 2, False, True)          # closing speed not known yet
    assert mon.update(unsure, 15.0).ttc == math.inf
    far = TrackState(40.0, -5.0, 1.0, 0.5, 10, False, True)            # TTC 8 s, THW 1.0 s at 40 m/s
    assert not FcwMonitor().update(far, 40.0).warning                  # THW is off by default
    thw = FcwMonitor(FcwParams(thw_s=1.2, confirm_frames=1)).update(far, 40.0)
    assert thw.warning and thw.reason == "thw" and thw.thw == pytest.approx(1.0)


def test_side_vehicles_lane_and_gaps():
    left, right = lane(1.75), lane(-1.75)
    info = lateral_position(car_box(15.0, 3.5), left, right, CAM)
    assert info.lane == "left" and info.range == pytest.approx(15.0)
    assert info.lateral == pytest.approx(3.5, abs=1e-6)
    assert info.gap_to_lane == pytest.approx(3.5 - 0.9 - 1.75, abs=1e-6)
    assert info.gap_to_ego == pytest.approx(3.5 - 0.9 - 0.95, abs=1e-6)
    assert not adjacent_close(info) and adjacent_close(info, AdjacentParams(max_gap_to_ego=2.0))
    r = lateral_position(car_box(10.0, -2.8), left, right, CAM)
    assert r.lane == "right" and r.lateral == pytest.approx(-2.8) and r.gap_to_lane == pytest.approx(2.8 - 0.9 - 1.75)
    assert adjacent_close(r)                                           # 0.95 m from the ego side, 10 m ahead
    assert not adjacent_close(r, AdjacentParams(max_range=8.0))
    cutting_in = lateral_position(car_box(12.0, 2.2), left, right, CAM)
    assert cutting_in.lane == "left" and cutting_in.gap_to_lane < 0     # already over the left boundary
    ahead = lateral_position(car_box(25.0, 0.0), left, right, CAM)
    assert ahead.lane == "ego" and math.isnan(ahead.gap_to_lane)
    assert lateral_position([900, 500, 1000, 600], left, right, CAM).lane == "unknown"


def test_ego_speed_from_poses():
    p0, p1 = np.eye(4), np.eye(4)
    p1[:3, 3] = [0.6, 0.8, 0.0]
    assert ego_speed(p0, p1, 0.1) == pytest.approx(10.0)
    assert math.isnan(ego_speed(p0, p1, 0.0))


def test_range_check_script_on_the_synthetic_openlane(tmp_path):
    import importlib.util

    from tac_ufld.config import PROJECT_ROOT
    from tests.test_cipo import make_cipo_openlane, write_manifest

    root = tmp_path / "OpenLane"
    manifest = write_manifest(tmp_path / "m.csv", make_cipo_openlane(root))
    spec = importlib.util.spec_from_file_location("cipo_range_check", PROJECT_ROOT / "scripts" / "cipo_range_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    m = module.main(["--openlane-root", str(root), "--manifest", str(manifest), "--out", str(tmp_path / "out")])
    assert m["frames"] == 6 and m["objects"] == 18                    # 3 objects per frame, all on labelled rows
    for est in ("flat", "pitched", "size", "fused"):                  # flat road, exact sizes: every model is right
        assert m["range"]["all"][est]["abs_rel"] < 0.01, est
    assert m["range"]["ego_lane"]["objects"] == 6 and m["range"]["adjacent_lanes"]["objects"] == 12
    assert abs(m["range"]["lane_pitch_deg"]["median"]) < 0.01
    assert m["tracking"]["closing_speed_kalman"]["n"] == 0            # level 1 only on every other frame
    assert (tmp_path / "out" / "range_rows.csv").exists()
