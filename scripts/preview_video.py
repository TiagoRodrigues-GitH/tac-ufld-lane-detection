"""Side-by-side preview video: the original frames, the ground truth and every trained model, in one grid.

Panels: the camera frame as the models receive it; the ground truth (the annotated ego-lane boundaries, the ego
lane and the annotated lane-change label); then one panel per model with its ego-lane boundaries (red = left,
blue = right), the ego lane the car is following (green fill), the ground truth (thin white) and the outputs of
``tac_ufld.ego``: lateral offset, lane-change decision vs. the label, lane departure warning (LDW) and the lane
keeping assist (LKA) steering request. Each model panel also shows this frame's lane TP/FP/FN against the ground
truth and the running lane F1 of the segment, with the evaluator's rule (CULane-width lane masks inside the
annotated band, Hungarian matching at IoU >= 0.5). Models run frame by frame through ``StreamingLaneDetector``
(the deployment path, history features cached, the post-processing tuned on validation), so the panel also shows
the per-frame model latency.

Works on any dataset in the ELAS layout: ELAS itself and the CARLA scenes written by ``carla_collect_elas.py``
(``--root C:/CARLA/lane_data/carla_elas``)::

    python scripts/preview_video.py --device cpu --columns 4 --panel-width 480 \
        --checkpoint results/elas_pilot_v2/seed_1/checkpoints/ufld_baseline.pt --name "UFLD v1" \
        --checkpoint results/review_lite_tsm_pilot/seed_1/checkpoints/lite_tsm.pt --name "lite TSM" \
        --segment BR_S02:0:450 --segment ROD_S01:20:700 --out results/review_preview/elas_all_models_gt.mp4

The preview F1 is a sanity check on the frames shown, not the reported metric (that is computed by the
evaluator on the whole test split). Labels are bilingual (PT / EN). A CSV with the per-segment counts is written
next to the video. The output is re-encoded to H.264 with ffmpeg when it is on PATH.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_ufld.data.elas import discover_scenes, parse_groundtruth, parse_scene_config  # noqa: E402
from tac_ufld.ego import EgoLaneTracker, lane_keeping  # noqa: E402
from tac_ufld.inference.loading import load_model  # noqa: E402
from tac_ufld.inference.streaming import StreamingLaneDetector  # noqa: E402
from tac_ufld.metrics import culane_line_width, iou_matrix, lane_mask, match_lanes, prf  # noqa: E402

RED, BLUE, GREEN, WHITE, YELLOW, BLACK = (40, 40, 230), (230, 120, 30), (60, 200, 60), (240, 240, 240), (0, 215, 255), (0, 0, 0)
SIDE = {-1: "ESQ/LEFT", 0: "-", 1: "DIR/RIGHT"}
IOU = 0.5
HOLD = 2  # display only: frames an unlabelled frame may borrow the last label from (see render_segment)


def _text(img, s, y, x=8, scale=0.5, colour=WHITE, thickness=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def _strip(img, y0, y1):
    """Darken a full-width strip so the text on it stays legible after video compression."""
    roi = img[y0:y1]
    img[y0:y1] = (roi * 0.35).astype(img.dtype)


def _polyline(img, lane, colour, thickness):
    if lane is not None and len(lane) >= 2:
        pts = np.round(lane[np.argsort(lane[:, 1])]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, colour, thickness, cv2.LINE_AA)


def _ego_fill(img, left, right):
    if left is None or right is None or len(left) < 2 or len(right) < 2:
        return
    poly = np.concatenate([left[np.argsort(left[:, 1])], right[np.argsort(-right[:, 1])]]).astype(np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], GREEN)
    cv2.addWeighted(overlay, 0.30, img, 0.70, 0, dst=img)


def _band(panel, cfg):
    band = panel.copy()
    cv2.rectangle(band, (0, cfg.roi_y), (cfg.frame_w - 1, cfg.roi_y + cfg.roi_h - 1), (90, 90, 90), -1)
    cv2.addWeighted(band, 0.15, panel, 0.85, 0, dst=panel)


def _gt_lanes(ann_frame):
    """Annotated lanes of known slots, as the evaluator uses them."""
    if ann_frame is None:
        return []
    return [lane for lane, known in zip(ann_frame.lanes, ann_frame.known) if lane is not None and known]


def lane_counts(pred, gt, cfg, threshold=IOU):
    """(TP, FP, FN) of one frame with the evaluator's rule: lane masks 30 px wide at 1640 px image width
    (CULane), cropped to the annotated band plus one line width, Hungarian matching at IoU >= threshold."""
    lw = culane_line_width(cfg.frame_w)
    y0 = max(0, int(cfg.roi_y) - lw)
    y1 = min(cfg.frame_h, int(cfg.roi_y + cfg.roi_h) + lw + 1)

    def masks(lanes):
        out = []
        for lane in lanes:
            shifted = np.asarray(lane, dtype=np.float32).copy()
            shifted[:, 1] -= y0
            out.append(lane_mask(shifted, cfg.frame_w, y1 - y0, lw))
        return out

    m = match_lanes(iou_matrix(masks(pred), masks(gt)), threshold)
    return m.tp, m.fp, m.fn


def _f1(c):
    return prf(*c)[2] if sum(c) else float("nan")


def _model_panel(bgr, cfg, ann_frame, res, state, decision, lka, label, name, frame_counts, seg_counts, device):
    panel = bgr.copy()
    _band(panel, cfg)
    _ego_fill(panel, res.lanes[0], res.lanes[1] if len(res.lanes) > 1 else None)
    if ann_frame is not None:
        for lane in ann_frame.lanes:
            _polyline(panel, lane, WHITE, 1)
    _polyline(panel, res.lanes[0], RED, 3)
    if len(res.lanes) > 1:
        _polyline(panel, res.lanes[1], BLUE, 3)
    cx = int(cfg.frame_w / 2)
    cv2.line(panel, (cx, cfg.frame_h - 1), (cx, cfg.frame_h - 25), YELLOW, 2)
    _strip(panel, 0, 118)
    _text(panel, f"{name}  |  {res.latency_ms.get('model_ms', float('nan')):.1f} ms {device.upper()}", 20, colour=YELLOW)
    off = state.offset
    _text(panel, f"desvio/offset: {off:+.2f} faixa/lane" if np.isfinite(off) else "desvio/offset: -", 42)
    _text(panel, f"troca de faixa/lane change: {SIDE[decision]}   (GT: {SIDE[label]})", 64,
          colour=YELLOW if decision else WHITE)
    _text(panel, f"LDW: {SIDE[lka.ldw]}   LKA steer: {lka.steer:+.2f}", 86, colour=(70, 70, 255) if lka.ldw else WHITE,
          thickness=2 if lka.ldw else 1)
    tp, fp, fn = frame_counts
    f1 = _f1(seg_counts)
    _text(panel, f"TP {tp} FP {fp} FN {fn}  |  F1 trecho/clip: " + (f"{f1:.3f}" if np.isfinite(f1) else "-"), 108,
          colour=WHITE if fp == 0 and fn == 0 else (80, 160, 255))
    return panel


def _gt_panel(bgr, cfg, ann_frame, state, decision, label, source, held_from=None):
    panel = bgr.copy()
    _band(panel, cfg)
    lanes = list(ann_frame.lanes) if ann_frame is not None else []
    left = lanes[0] if len(lanes) > 0 else None
    right = lanes[1] if len(lanes) > 1 else None
    _ego_fill(panel, left, right)
    _polyline(panel, left, RED, 3)
    _polyline(panel, right, BLUE, 3)
    for lane in lanes[2:]:
        _polyline(panel, lane, WHITE, 2)
    cx = int(cfg.frame_w / 2)
    cv2.line(panel, (cx, cfg.frame_h - 1), (cx, cfg.frame_h - 25), YELLOW, 2)
    _strip(panel, 0, 96)
    _text(panel, f"Verdade de campo / Ground truth ({source})", 20, colour=GREEN)
    off = state.offset
    _text(panel, f"desvio/offset: {off:+.2f} faixa/lane" if np.isfinite(off) else "desvio/offset: -", 42)
    _text(panel, f"rotulo/label troca de faixa/lane change: {SIDE[label]}", 64, colour=YELLOW if label else WHITE)
    _text(panel, f"detector nas faixas GT / on GT lanes: {SIDE[decision]}", 86, colour=YELLOW if decision else WHITE)
    if ann_frame is None:
        _text(panel, "sem anotacao / not annotated", cfg.frame_h - 30, colour=YELLOW)
    elif held_from is not None:
        _text(panel, f"rotulo do quadro / label of frame {held_from} (nao pontuado / not scored)", cfg.frame_h - 30,
              scale=0.42, colour=YELLOW)
    return panel


def _fit(panel, width):
    if not width or panel.shape[1] == width:
        return panel
    h = int(round(panel.shape[0] * width / panel.shape[1]))
    return cv2.resize(panel, (width, h), interpolation=cv2.INTER_AREA)


def render_segment(models, paths, start, end, writer_state, fps, columns, panel_width, source, device="cpu"):
    """``models`` = [(name, detector)]; the original frame, the ground truth and one panel per model, in a grid
    of ``columns`` columns. Returns (frames written, {name: model latencies}, {name: [TP, FP, FN]})."""
    cfg = parse_scene_config(paths.config)
    ann = parse_groundtruth(paths.groundtruth, cfg)

    def tracker():
        return EgoLaneTracker(cfg.frame_w / 2.0, cfg.roi_y + cfg.roi_h, cfg.roi_y)

    trackers = [tracker() for _ in models]
    gt_tracker = tracker()
    for _, det in models:
        det.reset_stream(cfg.scene)
    latency = {name: [] for name, _ in models}
    counts = {name: [0, 0, 0] for name, _ in models}
    frames, held = 0, None
    for fid in range(max(start, cfg.frame_start), min(end, cfg.frame_end) + 1):
        path = next((paths.images / f"lane_{fid}{ext}" for ext in (".png", ".jpg")
                     if (paths.images / f"lane_{fid}{ext}").exists()), None)  # ELAS png, CARLA jpg
        if path is None:
            continue
        bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        a = ann.get(fid)
        if a is not None:
            held = (fid, a)
        # Datasets labelled every k-th frame (CARLA: k = 3) would make the ground truth blink; for display only,
        # an unlabelled frame shows the last label if it is at most HOLD frames old. Scoring uses labelled frames.
        shown, held_from = a, None
        if a is None and held is not None and fid - held[0] <= HOLD:
            shown, held_from = held[1], held[0]
        raw = shown.attrs.get("laneChange", 0) if shown is not None else 0
        label = int(raw) if np.isfinite(raw) else 0
        gt = _gt_lanes(a)
        left = bgr.copy()
        _strip(left, 0, 52)
        _strip(left, cfg.frame_h - 22, cfg.frame_h)
        _text(left, "Original (entrada) / Original (input)", 20)
        _text(left, f"{cfg.scene}  frame {fid}", 42)
        _text(left, "vermelho/red = esq/left, azul/blue = dir/right, branco/white = GT, verde/green = faixa ego",
              cfg.frame_h - 8, scale=0.36)
        gt_state, gt_decision = gt_tracker.update(list(shown.lanes) if shown is not None else [None, None])
        panels = [left, _gt_panel(bgr, cfg, shown, gt_state, gt_decision, label, source, held_from)]
        for (name, det), trk in zip(models, trackers):
            res = det.infer_frame(rgb, stream_id=cfg.scene, frame_index=fid, sequence_id=cfg.scene)
            state, decision = trk.update(res.lanes)
            fc = (0, 0, 0)
            if a is not None and any(lane is not None for lane in a.lanes):  # as the evaluator: unlabelled
                # frames and frames without any annotated lane are shown but not scored
                fc = lane_counts([lane for lane in res.lanes if lane is not None], gt, cfg)
                counts[name] = [c + d for c, d in zip(counts[name], fc)]
            panels.append(_model_panel(bgr, cfg, shown, res, state, decision, lane_keeping(state, trk.detector),
                                       label, name, fc, counts[name], device))
            latency[name].append(res.latency_ms.get("model_ms", float("nan")))
        panels = [_fit(p, panel_width) for p in panels]
        while len(panels) % columns:
            panels.append(np.zeros_like(panels[0]))
        rows = [np.concatenate(panels[i:i + columns], axis=1) for i in range(0, len(panels), columns)]
        frame = np.concatenate(rows, axis=0)
        if writer_state["writer"] is None:
            h, w = frame.shape[:2]
            writer_state["writer"] = cv2.VideoWriter(str(writer_state["tmp"]), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer_state["writer"].write(frame)
        frames += 1
    return frames, latency, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", action="append", required=True, help="repeatable: one panel per model")
    ap.add_argument("--name", action="append", default=None, help="labels, in --checkpoint order")
    ap.add_argument("--root", default="../../datasets/dataset_elas_v1", help="dataset in the ELAS layout")
    ap.add_argument("--source", default=None, help="ground-truth label on its panel (default: ELAS or CARLA)")
    ap.add_argument("--segment", action="append", required=True, help="SCENE:first_frame:last_frame (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=float, default=29.97)
    ap.add_argument("--columns", type=int, default=2, help="panels per row of the grid")
    ap.add_argument("--panel-width", type=int, default=0, help="resize every panel to this width (0 = native)")
    args = ap.parse_args()

    import torch

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    loaded = [load_model(c, device) for c in args.checkpoint]
    names = args.name or [m.variant for m in loaded]
    if len(names) != len(loaded):
        ap.error("give one --name per --checkpoint, or none")
    source = args.source or ("CARLA" if "carla" in str(args.root).lower() else "ELAS")
    index = discover_scenes(Path(args.root))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.gettempdir()) / "tac_ufld_preview_raw.mp4"  # cv2 cannot write non-ASCII paths
    writer_state = {"writer": None, "tmp": tmp}
    rows = []
    total = {n: [0, 0, 0] for n in names}
    for seg in args.segment:
        scene, start, end = seg.split(":")
        cfg = parse_scene_config(index[scene].config)
        band = (float(cfg.roi_y), float(cfg.roi_y + cfg.roi_h))
        models = [(n, StreamingLaneDetector(m, valid_y_range=band)) for n, m in zip(names, loaded)]
        frames, latency, counts = render_segment(models, index[scene], int(start), int(end), writer_state, args.fps,
                                                 args.columns, args.panel_width, source, device)
        for n in names:
            total[n] = [c + d for c, d in zip(total[n], counts[n])]
            rows.append({"segment": seg, "model": n, "frames": frames, "tp": counts[n][0], "fp": counts[n][1],
                         "fn": counts[n][2], "lane_f1_iou50": round(_f1(counts[n]), 4),
                         "median_model_ms": round(float(np.nanmedian(latency[n][5:] or [np.nan])), 2),
                         "device": device})
        f1s = ", ".join(f"{n} F1 {_f1(counts[n]):.3f}" for n in names)
        print(f"{scene} [{start}-{end}]: {frames} frames; {f1s}")
    for n in names:
        rows.append({"segment": "all", "model": n, "frames": "", "tp": total[n][0], "fp": total[n][1],
                     "fn": total[n][2], "lane_f1_iou50": round(_f1(total[n]), 4), "median_model_ms": "",
                     "device": device})
    writer_state["writer"].release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:  # H.264 + yuv420p plays in browsers and on phones; mp4v does not
        h264 = Path(tempfile.gettempdir()) / "tac_ufld_preview_h264.mp4"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "27", "-preset", "medium", "-movflags", "+faststart", str(h264)], check=True)
        shutil.move(str(h264), str(out))
        tmp.unlink(missing_ok=True)
    else:
        shutil.move(str(tmp), str(out))
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MB) and {out.with_suffix('.csv').name}")


if __name__ == "__main__":
    main()
