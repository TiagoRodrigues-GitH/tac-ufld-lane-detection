"""Real-data validation of a dataset adapter (``python -m tac_ufld
validate-dataset``). Run it once on every new copy of a dataset before
training; it never modifies the data.

Checks, per split:

* frame / sequence counts, adapter statistics (missing images, skipped lanes);
* image sizes of a sample of files against the adapter's declared size;
* lane coordinates inside the image; points per lane;
* slot occupancy and lane ORDER: for every pair of present slots i < j the
  lane in slot i must lie left of the lane in slot j at their lowest shared row;
* availability of the temporal history frames for the configured
  ``temporal_step`` / ``num_frames``;
* unique frame keys;
* overlays (slot colours + encoded row-anchor targets) for visual inspection.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from tac_ufld.config import ExperimentConfig
from tac_ufld.data import build_adapter
from tac_ufld.data.targets import encode_targets, make_row_anchors
from tac_ufld.data.types import FrameRecord

SLOT_COLOURS = [(255, 64, 64), (255, 200, 0), (0, 200, 255), (160, 64, 255), (0, 255, 0), (255, 0, 255)]


def _order_violation(rec: FrameRecord) -> bool:
    present = [(i, lane) for i, lane in enumerate(rec.lanes) if lane is not None]
    for a in range(len(present)):
        for b in range(a + 1, len(present)):
            (_, la), (_, lb) = present[a], present[b]
            lo = max(la[:, 1].min(), lb[:, 1].min())
            hi = min(la[:, 1].max(), lb[:, 1].max())
            if hi < lo:
                continue  # no shared rows
            y = hi  # lowest shared row (closest to the car)
            xa = np.interp(y, *la[np.argsort(la[:, 1])][:, ::-1].T)
            xb = np.interp(y, *lb[np.argsort(lb[:, 1])][:, ::-1].T)
            if xa >= xb:
                return True
    return False


def _split_stats(records: list[FrameRecord], adapter, cfg: ExperimentConfig, seq_start: dict[str, int]) -> dict:
    d = cfg.data
    n_slots = max((r.num_slots for r in records), default=0)
    present = np.zeros(n_slots, dtype=np.int64)
    unknown = np.zeros(n_slots, dtype=np.int64)
    points, out_of_bounds, order_bad = [], 0, 0
    for r in records:
        w, h = r.image_size
        for i, (lane, known) in enumerate(zip(r.lanes, r.slot_known)):
            if not known:
                unknown[i] += 1
            if lane is None:
                continue
            present[i] += 1
            points.append(len(lane))
            out_of_bounds += int(((lane[:, 0] < 0) | (lane[:, 0] >= w) | (lane[:, 1] < 0) | (lane[:, 1] >= h)).any())
        order_bad += int(_order_violation(r))
    rng = random.Random(0)
    # History before the first frame of a sequence is expected to be missing;
    # a missing frame INSIDE a sequence means temporal_step does not match
    # the stored frame ids.
    history_total = history_missing = before_start = 0
    for r in rng.sample(records, min(300, len(records))):
        for k in range(1, d.num_frames):
            fid = r.frame_id - k * d.temporal_step
            if fid < seq_start.get(r.sequence, 0):
                before_start += 1
                continue
            history_total += 1
            history_missing += int(adapter.frame_path(r.sequence, fid) is None)
    size_bad = []
    for r in rng.sample(records, min(20, len(records))):
        with Image.open(r.image_path) as img:
            if img.size != tuple(r.image_size):
                size_bad.append({"image": str(r.image_path), "found": img.size, "declared": r.image_size})
    n = max(len(records), 1)
    return {
        "frames": len(records),
        "sequences": len({r.sequence for r in records}),
        "slot_present_rate": (present / n).round(4).tolist(),
        "slot_unknown_rate": (unknown / n).round(4).tolist(),
        "frames_without_lanes": int(sum(not r.has_any_lane() for r in records)),
        "points_per_lane": {"min": int(min(points, default=0)), "median": float(np.median(points)) if points else 0,
                            "max": int(max(points, default=0))},
        "lanes_out_of_bounds": out_of_bounds,
        "lane_order_violations": order_bad,
        "lane_order_violation_rate": round(order_bad / n, 5),
        "history_missing_rate": round(history_missing / history_total, 4) if history_total else 0.0,
        "history_before_sequence_start": before_start,
        "image_size_mismatches": size_bad,
    }


def _overlay(rec: FrameRecord, cfg: ExperimentConfig, anchors: np.ndarray, path: Path) -> None:
    import cv2

    from tac_ufld.visualization.overlays import imread_bgr, imwrite, put_label

    d = cfg.data
    img = imread_bgr(rec.image_path)
    for lane in rec.meta.get("eval_lanes", []) or []:
        pts = np.round(lane[np.argsort(lane[:, 1])]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (200, 200, 200), 1, cv2.LINE_AA)
    for i, lane in enumerate(rec.lanes):
        if lane is None:
            continue
        colour = SLOT_COLOURS[i % len(SLOT_COLOURS)][::-1]
        pts = np.round(lane[np.argsort(lane[:, 1])]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, colour, 3, cv2.LINE_AA)
    t = encode_targets(rec, d.img_w, d.img_h, anchors, d.griding_num)
    ow, oh = rec.image_size
    for a, s in zip(*np.nonzero(t.exist > 0.5)):
        cv2.circle(img, (int(t.x[a, s] * ow / d.img_w), int(anchors[a] * oh / d.img_h)), 3, (255, 255, 0), -1)
    put_label(img, f"{rec.key}  slots: 0 red, 1 yellow, 2 blue, 3 purple; grey = all annotated lanes")
    imwrite(path, img)


def validate_dataset(cfg: ExperimentConfig, out_dir: Path, n_overlays: int = 24) -> dict:
    adapter = build_adapter(cfg)
    official = adapter.official_splits()
    splits = {k: v for k, v in official.items() if v} if official else {"all": [r for recs in adapter.load_all().values() for r in recs]}
    report: dict = {"dataset": adapter.name, "root": str(cfg.data_root()), "protocol": adapter.protocol,
                    "adapter_stats": getattr(adapter, "stats", {}), "splits": {}}
    keys: set[str] = set()
    duplicates = 0
    seq_start: dict[str, int] = {}
    for recs in splits.values():
        for r in recs:
            seq_start[r.sequence] = min(seq_start.get(r.sequence, r.frame_id), r.frame_id)
    for name, recs in splits.items():
        report["splits"][name] = _split_stats(recs, adapter, cfg, seq_start)
        for r in recs:
            duplicates += int(r.key in keys)
            keys.add(r.key)
    report["duplicate_keys"] = duplicates
    if hasattr(adapter, "stride_report"):
        strides = adapter.stride_report()
        values, counts = np.unique(list(strides.values()), return_counts=True) if strides else ([], [])
        report["frame_stride_histogram"] = {int(v): int(c) for v, c in zip(values, counts)}
        report["temporal_step_multiple_of_all_strides"] = bool(
            strides and all(cfg.data.temporal_step % s == 0 for s in set(strides.values())))
    anchors = make_row_anchors(cfg.data.img_h, cfg.data.num_row_anchors, cfg.data.row_anchor_range)
    all_recs = [r for recs in splits.values() for r in recs]
    rng = random.Random(0)
    for rec in rng.sample(all_recs, min(n_overlays, len(all_recs))):
        _overlay(rec, cfg, anchors, out_dir / "overlays" / f"{rec.key.replace('/', '_')}.png")
    problems = []
    for name, s in report["splits"].items():
        if s["image_size_mismatches"]:
            problems.append(f"{name}: image sizes differ from the declared size")
        if s["lanes_out_of_bounds"]:
            problems.append(f"{name}: {s['lanes_out_of_bounds']} lanes with points outside the image")
        if s["lane_order_violation_rate"] > 0.01:
            problems.append(f"{name}: lane slots out of left-to-right order in "
                            f"{100 * s['lane_order_violation_rate']:.1f}% of frames")
        if cfg.data.num_frames > 1 and s["history_missing_rate"] > 0.05:
            problems.append(f"{name}: {100 * s['history_missing_rate']:.0f}% of history frames inside the "
                            f"sequences are missing for temporal_step={cfg.data.temporal_step}")
    if duplicates:
        problems.append(f"{duplicates} duplicate frame keys")
    report["problems"] = problems
    report["ok"] = not problems
    lines = [f"{adapter.name}: {'OK' if report['ok'] else 'PROBLEMS FOUND'}"]
    for name, s in report["splits"].items():
        lines.append(f"  {name:<6} frames={s['frames']:<7} sequences={s['sequences']:<6} "
                     f"slot presence={s['slot_present_rate']} order violations={s['lane_order_violations']} "
                     f"history missing={100 * s['history_missing_rate']:.1f}%")
    lines += [f"  problem: {p}" for p in problems]
    report["summary"] = "\n".join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
