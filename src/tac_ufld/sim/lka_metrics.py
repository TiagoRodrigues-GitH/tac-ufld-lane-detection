"""Closed-loop lane-keeping metrics from a per-frame log (pure NumPy: no CARLA, no torch).

The log has one row per simulation frame with the columns of ``COLUMNS`` (written by ``tac_ufld.sim.lka_loop``).
Ground truth (``gt_*``) comes from the simulator's road map; the system under test only sees the camera.

Definitions (ISO 17361 / Euro NCAP Lane Support Systems in spirit, adapted to what the simulator gives):

* line crossing     the vehicle body overlaps a lane boundary: ``|gt_offset_m| + half_width_m > lane_width / 2``.
                    One event per contiguous overlap (gaps shorter than ``merge_s`` merged); its side is the sign
                    of the offset at the onset (+1 right line, -1 left line). A full lane change is one event.
* intended          the turn signal was on toward that side at the onset.
* LKA-controlled    frames where the scenario lets the LKA steer (``lka_enabled``); a crossing that starts there is
                    an LKA failure.
* LDW event         contiguous run of the same non-zero ``ldw`` (gaps shorter than ``merge_s`` merged).
* LDW lead time     for each crossing during a driver-induced drift with the LKA off (phase ``drift_ldw``): crossing
                    onset minus the onset of the LDW event on that side that is active somewhere in
                    [onset - ``lookback_s``, onset + ``late_s``]; negative = warned after the crossing, none = missed.
* time to first departure   time (after warm-up) of the first crossing that starts in an LKA-controlled frame.
* LKA excursion     in the phase ``drift_lka`` (driver releases the wheel while drifting toward a line, Euro NCAP
                    LKA style): max of ``|offset| + half_width - lane_width / 2``; > 0 = metres beyond the line.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

COLUMNS = (
    "frame", "t_s", "phase", "lka_enabled", "driver", "turn_signal", "speed_mps", "distance_m",
    "gt_offset_m", "gt_lane_width_m", "gt_heading_deg", "gt_on_line", "gt_curvature", "gt_img_offset",
    "half_width_m",
    "pred_offset", "pred_offset_raw", "pred_heading", "pred_width_px", "lanes_seen", "lc_state",
    "tlc_s", "ldw", "lka_active", "lka_steer", "driver_steer", "disturbance", "steer", "throttle", "brake",
    "lead_gap_m", "det_ms", "fallbacks",
)
TEXT_COLUMNS = ("phase", "driver")
LC_PHASES = ("lc_signal", "lane_change")


# --------------------------------------------------------------------------- log files

def write_log(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in COLUMNS})


def _fmt(value) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6g}"
    return str(value)


def read_log(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return columns_from_rows(rows)


def columns_from_rows(rows: list[dict]) -> dict[str, np.ndarray]:
    out = {}
    for k in COLUMNS:
        values = [r.get(k, "") for r in rows]
        if k in TEXT_COLUMNS:
            out[k] = np.asarray([str(v) for v in values], dtype=object)
        else:
            out[k] = np.asarray([float(v) if v not in ("", None) else np.nan for v in values], dtype=np.float64)
    return out


# --------------------------------------------------------------------------- events

def events(signal: np.ndarray, merge: int = 0) -> list[tuple[int, int, int]]:
    """Runs of the same non-zero value as (start, end_inclusive, value); same-value runs separated by fewer than
    ``merge`` zero frames are joined (a warning that flickers off for one frame is one warning)."""
    sig = np.nan_to_num(np.asarray(signal, dtype=np.float64)).astype(int)
    out: list[tuple[int, int, int]] = []
    i, n = 0, len(sig)
    while i < n:
        if sig[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and sig[j + 1] == sig[i]:
            j += 1
        if out and out[-1][2] == sig[i] and i - out[-1][1] - 1 < merge:  # only zeros since the previous run
            out[-1] = (out[-1][0], j, out[-1][2])
        else:
            out.append((i, j, int(sig[i])))
        i = j + 1
    return out


def on_line(offset_m: np.ndarray, lane_width_m: np.ndarray, half_width_m: np.ndarray) -> np.ndarray:
    """Per frame: +1 / -1 when the body overlaps the right / left boundary of its nearest lane, else 0."""
    off = np.asarray(offset_m, dtype=np.float64)
    over = np.abs(off) + np.asarray(half_width_m, dtype=np.float64) > 0.5 * np.asarray(lane_width_m, dtype=np.float64)
    return np.where(over & np.isfinite(off), np.sign(off), 0).astype(int)


def crossing_events(offset_m: np.ndarray, lane_width_m: np.ndarray, half_width_m: np.ndarray,
                    merge: int = 0) -> list[tuple[int, int, int]]:
    """Line crossings as (start, end_inclusive, side at onset). The side may flip inside one overlap (the nearest
    lane switches when the centre crosses), so events are runs of "on a line", not of one side."""
    side = on_line(offset_m, lane_width_m, half_width_m)
    out = []
    for s, e, _ in events(np.abs(side), merge):
        out.append((s, e, int(side[s])))
    return out


def lane_margin(offset_m: np.ndarray, lane_width_m: np.ndarray, half_width_m: np.ndarray) -> np.ndarray:
    """Metres between the body edge and the nearest boundary (< 0 = beyond the line)."""
    return 0.5 * np.asarray(lane_width_m) - (np.abs(np.asarray(offset_m)) + np.asarray(half_width_m))


# --------------------------------------------------------------------------- summary

def _per_km(count: float, km: float) -> float:
    return count / km if km > 0 else float("nan")


def _nanstat(fn, values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(fn(values)) if values.size else float("nan")


def summarize(log: dict[str, np.ndarray], fps: float, merge_s: float = 0.3, lookback_s: float = 4.0,
              late_s: float = 1.0) -> dict[str, float]:
    """Metrics of one closed-loop run (frames with ``t_s < 0`` are warm-up and are ignored)."""
    keep_rows = np.asarray(log["t_s"]) >= 0
    log = {k: np.asarray(v)[keep_rows] for k, v in log.items()}
    n = len(log["t_s"])
    if n == 0:
        return {"frames": 0}
    merge = max(int(round(merge_s * fps)), 0)
    phase = log["phase"]
    lka_on = log["lka_enabled"] > 0.5
    signal = np.nan_to_num(log["turn_signal"]).astype(int)
    off, width, half = log["gt_offset_m"], log["gt_lane_width_m"], log["half_width_m"]
    km = (np.nanmax(log["distance_m"]) - np.nanmin(log["distance_m"])) / 1000.0
    m: dict[str, float] = {"frames": n, "duration_s": n / fps, "distance_km": km,
                           "mean_speed_kmh": 3.6 * _nanstat(np.mean, log["speed_mps"])}

    keep = phase == "keep"
    m["lat_rms_m"] = _nanstat(lambda v: np.sqrt(np.mean(v ** 2)), off[keep])
    m["lat_p95_m"] = _nanstat(lambda v: np.percentile(np.abs(v), 95), off[keep])
    m["lat_max_m"] = _nanstat(lambda v: np.max(np.abs(v)), off[keep])

    crossings = crossing_events(off, width, half, merge)
    intended = [signal[s] != 0 and signal[s] == side for s, _, side in crossings]
    by_lka = [lka_on[s] and not it for (s, _, _), it in zip(crossings, intended)]
    m["crossings"] = len(crossings)
    m["crossings_per_km"] = _per_km(len(crossings), km)
    m["crossings_intended"] = int(sum(intended))
    m["crossings_lka"] = int(sum(by_lka))
    m["crossings_lka_per_km"] = _per_km(sum(by_lka), km)
    first = [s for (s, _, _), bad in zip(crossings, by_lka) if bad]
    m["time_to_first_departure_s"] = float(log["t_s"][first[0]]) if first else float("nan")

    warnings = events(log["ldw"], merge)
    m["ldw_events"] = len(warnings)
    m["ldw_per_km"] = _per_km(len(warnings), km)
    m["ldw_in_keep"] = sum(1 for s, _, _ in warnings if phase[s] == "keep")
    m["ldw_in_lane_change"] = sum(1 for s, _, _ in warnings if phase[s] in LC_PHASES)

    leads, missed = [], 0
    look, late = int(round(lookback_s * fps)), int(round(late_s * fps))
    for s, _, side in crossings:
        if phase[s] != "drift_ldw":
            continue
        onsets = [ws for ws, we, wv in warnings if wv == side and we >= s - look and ws <= s + late]
        if not onsets:
            missed += 1
            continue
        before = [o for o in onsets if o <= s]
        leads.append((s - (max(before) if before else min(onsets))) / fps)  # latest warning that started in time
    m["ldw_drift_crossings"] = len(leads) + missed
    m["ldw_missed"] = missed
    m["ldw_lead_median_s"] = _nanstat(np.median, leads)
    m["ldw_lead_min_s"] = _nanstat(np.min, leads)

    released = phase == "drift_lka"
    margin = lane_margin(off, width, half)
    m["lka_max_excursion_m"] = _nanstat(lambda v: np.max(-v), margin[released])

    lc = np.isin(phase, LC_PHASES)
    m["lka_suppressed_frac_lc"] = _nanstat(np.mean, (log["lka_active"][lc] < 0.5).astype(float)) if lc.any() \
        else float("nan")
    lc_runs = events(np.where(phase == "lane_change", signal, 0))
    m["lane_changes"] = len(lc_runs)
    m["lane_changes_detected"] = sum(1 for s, e, side in lc_runs if np.any(log["lc_state"][s:e + 1] == side))

    seen = log["lanes_seen"]
    m["both_lanes_frac"] = _nanstat(np.mean, (seen >= 2).astype(float))
    m["offset_mae_lw"] = _nanstat(np.mean, np.abs(log["pred_offset_raw"] - log["gt_img_offset"]))
    m["det_ms_mean"] = _nanstat(np.mean, log["det_ms"])
    return m
