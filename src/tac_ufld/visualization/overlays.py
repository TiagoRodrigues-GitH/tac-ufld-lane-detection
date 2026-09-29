"""Image overlays: label checks, original/GT/prediction panels, error galleries,
temporal videos. All I/O is Unicode-safe (``cv2.imread`` fails on paths such
as ``Residência`` on Windows, audit note)."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tac_ufld.data.targets import LaneTargets
from tac_ufld.data.types import FrameRecord

GT_COLOR = (0, 200, 0)        # green (BGR)
PRED_COLOR = (0, 0, 255)      # red
REF_COLOR = (255, 140, 0)     # blue-ish: reference / baseline
ANCHOR_COLOR = (255, 255, 0)  # cyan: encoded targets
ROI_COLOR = (200, 200, 200)


def imread_bgr(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot decode image {path}")
    return img


def imwrite(path: str | Path, img: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise IOError(f"cannot encode {path}")
    buf.tofile(str(path))
    return path


def draw_lanes(img: np.ndarray, lanes, color, thickness: int = 2, points: bool = False) -> np.ndarray:
    for lane in lanes:
        if lane is None or len(lane) < 2:
            continue
        pts = np.round(lane[np.argsort(lane[:, 1])]).astype(np.int32)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)
        if points:
            for x, y in pts:
                cv2.circle(img, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
    return img


def put_label(img: np.ndarray, text: str, y: int = 22, scale: float = 0.55) -> np.ndarray:
    """White text on a dark box (readable on sky and asphalt alike)."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x0, y0, x1, y1 = 4, max(0, y - th - 5), min(img.shape[1], 12 + tw), min(img.shape[0], y + base + 3)
    img[y0:y1, x0:x1] = (0.35 * img[y0:y1, x0:x1]).astype(img.dtype)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_roi(img: np.ndarray, record: FrameRecord) -> np.ndarray:
    if record.valid_y_range is not None:
        for y in record.valid_y_range:
            cv2.line(img, (0, int(y)), (img.shape[1] - 1, int(y)), ROI_COLOR, 1, cv2.LINE_AA)
    return img


def comparison_panel(record: FrameRecord, pred_lanes, title: str = "") -> np.ndarray:
    """Original | ground truth | prediction, side by side."""
    img = imread_bgr(record.image_path)
    gt = draw_lanes(draw_roi(img.copy(), record), record.present_lanes(), GT_COLOR, 3)
    pred = draw_lanes(draw_roi(img.copy(), record), pred_lanes, PRED_COLOR, 3)
    put_label(img, f"original  {record.key}")
    put_label(gt, "ground truth")
    put_label(pred, f"prediction  {title}")
    return cv2.hconcat([img, gt, pred])


def save_label_check(records: list[FrameRecord], targets: list[LaneTargets], row_anchors: np.ndarray,
                     model_size: tuple[int, int], out_dir: Path) -> list[Path]:
    """GT polyline + annotated points (green) and the ENCODED training targets
    decoded back to image space (cyan dots). If the dots do not sit on the
    painted markings, labels are wrong - check before any training run."""
    saved = []
    mw, mh = model_size
    for rec, tgt in zip(records, targets):
        img = draw_roi(imread_bgr(rec.image_path), rec)
        draw_lanes(img, rec.present_lanes(), GT_COLOR, 2, points=True)
        ow, oh = rec.image_size
        for a, slot in zip(*np.nonzero(tgt.exist > 0.5)):
            x, y = tgt.x[a, slot] * ow / mw, row_anchors[a] * oh / mh
            cv2.circle(img, (int(round(x)), int(round(y))), 3, ANCHOR_COLOR, -1, cv2.LINE_AA)
        put_label(img, f"{rec.key}  green=GT points  cyan=encoded targets  grey=ROI")
        saved.append(imwrite(out_dir / f"{rec.sequence}_{rec.frame_id}.png", img))
    return saved


def save_examples(records: list[FrameRecord], lanes: list, per_frame: pd.DataFrame, tag: str,
                  label: str, out_dir: Path, n: int) -> None:
    """Worst frames (errors) and best frames (successes) by per-frame lane F1."""
    tp, fp, fn = per_frame[f"tp_{tag}"], per_frame[f"fp_{tag}"], per_frame[f"fn_{tag}"]
    f1 = (2 * tp / (2 * tp + fp + fn).clip(lower=1)).to_numpy()
    order = np.argsort(f1, kind="stable")
    buckets = {"errors": [i for i in order if fp.iloc[i] + fn.iloc[i] > 0][:n],
               "successes": [i for i in order[::-1] if fp.iloc[i] + fn.iloc[i] == 0 and tp.iloc[i] > 0][:n]}
    for bucket, idxs in buckets.items():
        for i in idxs:
            rec = records[i]
            panel = comparison_panel(rec, lanes[i], f"{label} TP={tp.iloc[i]} FP={fp.iloc[i]} FN={fn.iloc[i]}")
            imwrite(out_dir / bucket / f"{rec.sequence}_{rec.frame_id}.png", panel)


def longest_consecutive_run(records: list[FrameRecord], min_length: int = 8) -> list[int]:
    """Indices of the longest run of consecutive frame ids within one sequence."""
    by_seq: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        by_seq.setdefault(rec.sequence, []).append(i)
    best: list[int] = []
    for idxs in by_seq.values():
        idxs.sort(key=lambda i: records[i].frame_id)
        run = idxs[:1]
        for prev, cur in zip(idxs, idxs[1:]):
            run = run + [cur] if records[cur].frame_id - records[prev].frame_id == 1 else [cur]
            if len(run) > len(best):
                best = list(run)
    return best if len(best) >= min_length else []


SLOT_COLORS = [(64, 64, 255), (0, 200, 255), (255, 200, 0), (255, 64, 160)]  # BGR per lane slot


def draw_prediction(rgb: np.ndarray, result, title: str = "", gt_lanes=None) -> np.ndarray:
    """BGR image with the streaming result: one colour per slot, the lane's
    mean existence probability, history use and latency."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if gt_lanes:
        draw_lanes(img, gt_lanes, GT_COLOR, 2)
    for slot, (lane, conf) in enumerate(zip(result.lanes, result.lane_confidence)):
        if lane is None:
            continue
        colour = SLOT_COLORS[slot % len(SLOT_COLORS)]
        draw_lanes(img, [lane], colour, 3)
        top = lane[np.argmin(lane[:, 1])]
        if conf is not None:
            cv2.putText(img, f"{conf:.2f}", (int(top[0]) - 14, int(top[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colour, 2, cv2.LINE_AA)
    lat = result.latency_ms
    band = img[:52].copy()
    band[:] = (20, 20, 20)
    img[:52] = cv2.addWeighted(img[:52], 0.35, band, 0.65, 0)
    lines = [f"{title}   frame {result.frame_index}   history {result.history_indices}"
             f"{'   fallback x' + str(result.fallbacks) if result.fallbacks else ''}",
             f"preprocess {lat.get('preprocess_ms', 0):.1f} ms   model {lat.get('model_ms', 0):.1f} ms   "
             f"postprocess {lat.get('postprocess_ms', 0):.1f} ms"]
    for i, text in enumerate(lines):
        cv2.putText(img, text, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class VideoSink:
    """Incremental MP4 writer (one frame in memory at a time). OpenCV cannot
    write to non-ASCII Windows paths, so it writes to the temp folder and
    moves the file on ``close``."""

    def __init__(self, path: str | Path, fps: float = 10.0) -> None:
        self.path = Path(path)
        self.fps = fps
        self.tmp = Path(tempfile.gettempdir()) / f"tac_ufld_sink_{os.getpid()}_{id(self)}.mp4"
        self.writer = None
        self.frames = 0

    def write(self, bgr: np.ndarray) -> None:
        if self.writer is None:
            h, w = bgr.shape[:2]
            self.writer = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not self.writer.isOpened():
                raise IOError("cannot open an MP4 writer")
        self.writer.write(bgr)
        self.frames += 1

    def close(self) -> Path | None:
        if self.writer is None:
            return None
        self.writer.release()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.tmp), str(self.path))
        return self.path


def save_temporal_video(records: list[FrameRecord], lanes_by_label: dict[str, list], out_path: Path,
                        fps: int = 10, max_frames: int = 150) -> Path | None:
    """GT (green) plus one colour per model over the longest consecutive run."""
    run = longest_consecutive_run(records)[:max_frames]
    if not run:
        return None
    colours = [PRED_COLOR, REF_COLOR, (255, 0, 255), (0, 255, 255), (128, 0, 255), (0, 128, 255)]
    writer = None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # cv2.VideoWriter cannot open non-ASCII paths on Windows: write to the
    # system temp dir, then move.
    tmp = Path(tempfile.gettempdir()) / f"tac_ufld_{os.getpid()}.mp4"
    try:
        for n, i in enumerate(run):
            rec = records[i]
            img = draw_lanes(draw_roi(imread_bgr(rec.image_path), rec), rec.present_lanes(), GT_COLOR, 3)
            for colour, (label, lanes) in zip(colours, lanes_by_label.items()):
                draw_lanes(img, lanes[i], colour, 2)
            put_label(img, f"{rec.key} ({n + 1}/{len(run)})")
            put_label(img, "green=GT  " + "  ".join(f"{c}={l}" for c, l in
                      zip(["red", "blue", "magenta", "yellow", "purple", "orange"], lanes_by_label)),
                      y=img.shape[0] - 10, scale=0.4)
            if writer is None:
                writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
                if not writer.isOpened():
                    return None
            writer.write(img)
    finally:
        if writer is not None:
            writer.release()
    if tmp.exists():
        shutil.move(str(tmp), str(out_path))
        return out_path
    return None
