"""How good are the monocular range models, and what does the Kalman filter add? Labelled boxes, no detector.

Runs on the OpenLane CIPO labels of one split (default: the test segments of a lane run's split manifest) and
writes ``range_check.json`` + ``range_rows.csv`` (docs/CIPO.md section 10):

* range per labelled object (vehicles, pedestrians, cyclists, every importance level) from
  - ``flat``:    the box's contact row on a flat road, camera pitch 0 (d = f_y h / (v - c_y));
  - ``pitched``: the same with the pitch from the vanishing point of the ground-truth ego-lane boundaries;
  - ``size``:    the class size prior (vehicle width 1.8 m, pedestrian 1.7 m / cyclist 1.75 m tall);
  - ``fused``:   inverse-variance mean of ``pitched`` and ``size`` with their first-order error models;
  against the REFERENCE range: the forward coordinate of the ground-truth ego-lane 3-D points at the contact row
  (objects whose contact row the lanes do not cover, or whose box is cut by the image bottom, are skipped);
* coverage of the error model: share of objects with |pitched - reference| <= 2 sigma (95 % if calibrated);
* level-1 tracks (same ``trackid``, consecutive frames): closing speed from frame differences of the fused range
  vs the constant-velocity Kalman filter, both against the slope of the reference range over +-0.5 s;
* ego speed from the poses, time headway of the level-1 object, and how often the FCW rule would fire.

    python scripts/cipo_range_check.py --manifest results/openlane/data/split_manifest.csv ^
        --out results/cipo/range_check
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_ufld.cipo import distance_errors, ego_boundaries, lane_ground_distance  # noqa: E402
from tac_ufld.config import PROJECT_ROOT  # noqa: E402
from tac_ufld.data.openlane_cipo import PATH_TYPES, CipoSource, build_frames, read_split_manifest  # noqa: E402
from tac_ufld.proximity import (OPENLANE_TIME_UNIT_S, SIZE_PRIORS, CipoRangeTracker, FcwMonitor,  # noqa: E402
                                FcwParams, RangeKalmanParams, ego_speed, fuse_ranges, ground_range,
                                ground_range_sigma, horizon_from_lanes, lateral_position, pitch_from_horizon,
                                size_range, time_headway)

LOGGER = logging.getLogger("cipo_range_check")
ESTIMATORS = ("flat", "pitched", "size", "fused")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--openlane-root", type=Path, default=(PROJECT_ROOT / "../../datasets/OpenLane").resolve())
    ap.add_argument("--cipo", type=Path, default=None, help="cipo/ folder or cipo.zip (default <root>/cipo)")
    ap.add_argument("--manifest", type=Path, required=True, help="split_manifest.csv of a lane run")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--sigma-v", type=float, default=2.0, help="px, contact-row error assumed by the error model")
    ap.add_argument("--sigma-pitch-deg", type=float, default=0.2, help="deg, pitch error assumed by the error model")
    ap.add_argument("--accel-psd", type=float, default=2.0, help="Kalman white-noise acceleration density, m^2/s^3")
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args(argv)


def object_rows(frames, sigma_v: float, sigma_pitch: float) -> list[dict]:
    rows = []
    for f in frames:
        if f.lanes is None or f.lane_ground is None:
            continue
        left, right = ego_boundaries(f.lanes)
        v_h = horizon_from_lanes(left, right, f.camera)
        pitch = pitch_from_horizon(v_h, f.camera) if math.isfinite(v_h) else math.nan
        for b in f.boxes:
            if b.type not in PATH_TYPES:
                continue
            x1, y1, x2, y2 = b.xyxy
            if y2 >= f.image_size[1] - 2:
                continue
            ref = lane_ground_distance(y2, f.lane_ground)
            if not math.isfinite(ref):
                continue
            cam = f.camera
            pitched = float(ground_range(y2, cam, pitch)) if math.isfinite(pitch) else math.nan
            s_pitched = ground_range_sigma(y2, cam, pitch, sigma_v, sigma_pitch) if math.isfinite(pitch) else math.inf
            size_d, size_s = size_range(b.xyxy, cam, SIZE_PRIORS[b.type])
            fused, fused_s = fuse_ranges([(pitched, s_pitched), (size_d, size_s)])
            info = lateral_position(b.xyxy, left, right, cam, pitch if math.isfinite(pitch) else 0.0)
            rows.append({"sequence": f.sequence, "frame_index": f.frame_index, "timestamp": f.timestamp,
                         "track": b.track_id, "level": b.level, "type": b.type, "lane": info.lane,
                         "pitch_deg": math.degrees(pitch) if math.isfinite(pitch) else math.nan,
                         "reference": ref, "flat": float(ground_range(y2, cam)), "pitched": pitched,
                         "pitched_sigma": s_pitched, "size": size_d, "size_sigma": size_s, "fused": fused,
                         "fused_sigma": fused_s, "box": (x1, y1, x2, y2), "lateral": info.lateral,
                         "gap_to_ego": info.gap_to_ego})
    return rows


def _signed_bias(est: np.ndarray, ref: np.ndarray) -> float:
    ok = np.isfinite(est) & np.isfinite(ref) & (ref > 0)
    return float(np.median((est[ok] - ref[ok]) / ref[ok])) if ok.any() else math.nan


def range_metrics(rows: list[dict]) -> dict:
    out: dict = {}
    groups = {"all": rows, "level1": [r for r in rows if r["level"] == 1],
              "vehicles": [r for r in rows if r["type"] == 1], "pedestrians": [r for r in rows if r["type"] == 2],
              "ego_lane": [r for r in rows if r["lane"] == "ego"],
              "adjacent_lanes": [r for r in rows if r["lane"] in ("left", "right")]}
    for name, items in groups.items():
        ref = np.array([r["reference"] for r in items], dtype=np.float64)
        g = {"objects": len(items)}
        for est in ESTIMATORS:
            e = np.array([r[est] for r in items], dtype=np.float64)
            g[est] = {**distance_errors(e, ref), "median_signed_rel": _signed_bias(e, ref)}
        out[name] = g
    ok = [r for r in rows if math.isfinite(r["pitched"]) and math.isfinite(r["pitched_sigma"])]
    if ok:
        z = np.array([abs(r["pitched"] - r["reference"]) / r["pitched_sigma"] for r in ok])
        out["error_model_coverage_2sigma"] = float(np.mean(z <= 2.0))
        out["error_model_objects"] = len(ok)
    pitches = np.array([r["pitch_deg"] for r in rows], dtype=np.float64)
    pitches = pitches[np.isfinite(pitches)]
    if len(pitches):
        out["lane_pitch_deg"] = {"median": float(np.median(pitches)), "p05": float(np.percentile(pitches, 5)),
                                 "p95": float(np.percentile(pitches, 95))}
    return out


def _reference_rate(ts: np.ndarray, ref: np.ndarray, i: int, half_window_s: float = 0.5) -> float:
    sel = np.abs(ts - ts[i]) <= half_window_s + 1e-6
    if sel.sum() < 5:
        return math.nan
    return float(np.polyfit(ts[sel] - ts[i], ref[sel], 1)[0])


def tracking_metrics(rows: list[dict], frames, accel_psd: float) -> dict:
    """Level-1 objects followed over consecutive frames: frame-difference vs Kalman closing speed, TTC, THW."""
    pose = {(f.sequence, f.frame_index): (f.timestamp, f.pose) for f in frames}
    tracks: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["level"] == 1 and math.isfinite(r["fused"]):
            tracks[(r["sequence"], r["track"])].append(r)
    diff_err, kf_err, warnings, ttc_low, thw_values, frames_used = [], [], 0, 0, [], 0
    for items in tracks.values():
        items.sort(key=lambda r: r["frame_index"])
        runs, run = [], [items[0]]
        for r in items[1:]:
            if r["frame_index"] == run[-1]["frame_index"] + 1:
                run.append(r)
            else:
                runs.append(run)
                run = [r]
        runs.append(run)
        for run in runs:
            if len(run) < 10:
                continue
            ts = np.array([r["timestamp"] for r in run], dtype=np.float64) * OPENLANE_TIME_UNIT_S
            ref = np.array([r["reference"] for r in run])
            tracker, fcw = CipoRangeTracker(RangeKalmanParams(accel_psd=accel_psd)), FcwMonitor(FcwParams())
            for i, r in enumerate(run):
                dt = ts[i] - ts[i - 1] if i else 0.1
                state = tracker.update(r["box"], r["fused"], r["fused_sigma"], dt)
                prev = pose.get((r["sequence"], r["frame_index"] - 1))
                cur = pose.get((r["sequence"], r["frame_index"]))
                v_ego = math.nan
                if prev and cur and prev[1] is not None and cur[1] is not None:
                    v_ego = ego_speed(prev[1], cur[1], (cur[0] - prev[0]) * OPENLANE_TIME_UNIT_S)
                out = fcw.update(state, v_ego if math.isfinite(v_ego) else 0.0)
                warnings += out.warning
                ttc_low += out.ttc < FcwParams().ttc_s
                if math.isfinite(v_ego) and v_ego > 0.5:
                    thw_values.append(time_headway(r["reference"], v_ego))
                frames_used += 1
                truth = _reference_rate(ts, ref, i)
                if i < 5 or not math.isfinite(truth) or state.new_track:
                    continue
                kf_err.append(state.range_rate - truth)
                diff_err.append((r["fused"] - run[i - 1]["fused"]) / dt - truth)
    thw = np.array(thw_values)

    def summary(err: list[float]) -> dict:
        e = np.array(err)
        return {"n": int(len(e)), "mae_mps": float(np.mean(np.abs(e))) if len(e) else math.nan,
                "rmse_mps": float(np.sqrt(np.mean(e ** 2))) if len(e) else math.nan}

    return {"level1_track_frames": frames_used, "closing_speed_frame_difference": summary(diff_err),
            "closing_speed_kalman": summary(kf_err), "fcw_warning_frames": int(warnings),
            "ttc_below_threshold_frames": int(ttc_low), "fcw_ttc_threshold_s": FcwParams().ttc_s,
            "thw_s": {"n": int(len(thw)), "median": float(np.median(thw)) if len(thw) else math.nan,
                      "share_below_1s": float(np.mean(thw < 1.0)) if len(thw) else math.nan,
                      "share_below_2s": float(np.mean(thw < 2.0)) if len(thw) else math.nan}}


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    args.out.mkdir(parents=True, exist_ok=True)
    source = CipoSource(args.cipo if args.cipo is not None else args.openlane_root / "cipo")
    sequences = {s: sp for s, sp in read_split_manifest(args.manifest).items() if sp == args.split}
    t0 = time.perf_counter()
    splits, stats = build_frames(args.openlane_root, source, sequences, {args.split: args.max_frames},
                                 lane_splits=(args.split,))
    frames = splits.get(args.split, [])
    LOGGER.info("%d frames of %d sequences read in %.0f s", len(frames), len(sequences), time.perf_counter() - t0)
    rows = object_rows(frames, args.sigma_v, math.radians(args.sigma_pitch_deg))
    metrics = {"split": args.split, "frames": len(frames), "objects": len(rows), "data_stats": stats,
               "assumed_sigma_v_px": args.sigma_v, "assumed_sigma_pitch_deg": args.sigma_pitch_deg,
               "size_priors": {k: v.__dict__ for k, v in SIZE_PRIORS.items()},
               "range": range_metrics(rows), "tracking": tracking_metrics(rows, frames, args.accel_psd)}
    (args.out / "range_check.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if rows:
        with open(args.out / "range_rows.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "box"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    for est in ESTIMATORS:
        m = metrics["range"]["all"][est]
        LOGGER.info("%-8s abs-rel %.3f  median %.3f  within 10%% %.3f  bias %+.3f  (n %d)", est,
                    m.get("abs_rel", math.nan), m.get("abs_rel_median", math.nan), m.get("within_10pct", math.nan),
                    m["median_signed_rel"], m["n"])
    t = metrics["tracking"]
    LOGGER.info("closing speed MAE: frame difference %.2f m/s, Kalman %.2f m/s (n %d)",
                t["closing_speed_frame_difference"]["mae_mps"], t["closing_speed_kalman"]["mae_mps"],
                t["closing_speed_kalman"]["n"])
    LOGGER.info("wrote %s", args.out / "range_check.json")
    return metrics


if __name__ == "__main__":
    main()
