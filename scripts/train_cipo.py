"""Train and evaluate the CIPO head on a FROZEN lane model (docs/CIPO.md, option F).

The lane model's backbone and lane head are not changed (its lane results stay valid); only the small detection
head is trained, on the OpenLane CIPO boxes of the lane run's own train segments. Validation segments select the
epoch and tune the CIPO decision rules; the test segments (OpenLane's official validation set) are scored once.

    python scripts/train_cipo.py --lane-ckpt results/openlane/seed_1/checkpoints/ufld_baseline.pt ^
        --cipo <OpenLane>/cipo --out results/cipo/ufld_baseline_s1 [--max-train-frames 300 --epochs 2]

``--cipo`` takes the extracted ``cipo/`` folder or ``cipo.zip``. The split comes from the lane run's
``data/split_manifest.csv`` (next to its ``seed_*`` folders), so the frozen backbone never saw a validation or test
segment. Output: ``metrics.json`` (everything below), ``history.csv``, ``head.pt``.

Metrics (test split; ``docs/CIPO.md`` section 7):
* detection: COCO-style AP@0.5 per importance level and class-agnostic, AP@[.5:.95] for level 1;
* CIPO decision per rule (learned / geometric / hybrid, thresholds tuned on validation): accuracy, precision,
  recall, false- and missed-CIPO rates, flicker and identity switches per minute;
* in-path decomposition: rule validity (ground-truth level-1 boxes vs ground-truth / predicted ego lanes) and
  selection with oracle boxes (all labelled objects) + ground-truth / predicted lanes;
* distance: flat-ground proxy vs the lane-ground reference, for labelled level-1 boxes and predicted CIPOs;
* cost: head parameters and MACs, lane-model MACs, GPU latency of the lane model alone and with the head.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_ufld.cipo import (CipoRule, average_precision, average_precisions, box_iou,  # noqa: E402
                          cipo_frame_metrics, cipo_stability, distance_errors, ego_boundaries, ego_span,
                          lane_ground_distance, path_geometry, select_cipo)
from tac_ufld.config import PROJECT_ROOT, GeometricAugConfig  # noqa: E402
from tac_ufld.data.openlane_cipo import (NUM_LEVELS, PATH_TYPES, CipoDataset, CipoFrame,  # noqa: E402
                                         CipoSource, build_frames, read_split_manifest, seed_worker)
from tac_ufld.decoding import decode_logits  # noqa: E402
from tac_ufld.evaluation.efficiency import count_macs, count_parameters  # noqa: E402
from tac_ufld.inference.loading import load_model  # noqa: E402
from tac_ufld.models.cipo_head import CipoHead, FeatureTap, HeadProbe, cipo_loss, decode  # noqa: E402
from tac_ufld.postprocess import lanes_from_prediction  # noqa: E402

LOGGER = logging.getLogger("train_cipo")
SCORE_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
DISTANCE_GRID = (20.0, 30.0, 45.0, 60.0, 80.0, math.inf)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lane-ckpt", type=Path, required=True, help="trained single-frame lane model (.pt)")
    ap.add_argument("--cipo", type=Path, default=None, help="cipo/ folder or cipo.zip (default <root>/cipo)")
    ap.add_argument("--openlane-root", type=Path, default=None, help="OpenLane folder with images/ and lane3d_1000/")
    ap.add_argument("--split-manifest", type=Path, default=None, help="default: <lane run>/data/split_manifest.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-train-frames", type=int, default=None)
    ap.add_argument("--max-val-frames", type=int, default=None)
    ap.add_argument("--max-test-frames", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=64, help="head channels")
    ap.add_argument("--augment", action="store_true", help="geometric augmentation (flip, shift, zoom)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--latency-runs", type=int, default=50, help="0 skips the latency measurement")
    return ap.parse_args(argv)


def _device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _openlane_root(args: argparse.Namespace, cfg) -> Path:
    if args.openlane_root is not None:
        return args.openlane_root
    root = os.path.expandvars(str(cfg.data.root))
    if "$" not in root and Path(root).is_dir():
        return Path(root)
    return (PROJECT_ROOT / "../../datasets/OpenLane").resolve()


def _loader(frames: list[CipoFrame], cfg, args, train: bool, stride: int, persistent: bool = True
            ) -> torch.utils.data.DataLoader:
    augment = None
    if train and args.augment:
        augment = GeometricAugConfig(enabled=True, prob=0.7, translate_x=0.05, translate_y=0.03, scale=[0.9, 1.1],
                                     crop_scale=[0.85, 1.0], hflip_prob=0.5)
    ds = CipoDataset(frames, cfg.data.img_w, cfg.data.img_h, stride, cfg.data.normalize, augment,
                     jpeg_draft=bool(getattr(cfg.data, "jpeg_draft", True)))
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=train, num_workers=args.workers,
                                       pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                                       generator=gen, persistent_workers=persistent and args.workers > 0,
                                       drop_last=train and len(ds) > args.batch_size)


# -- inference ------------------------------------------------------------------------------------------------------

@torch.no_grad()
def predict(tap: FeatureTap, head: CipoHead, loader, frames: list[CipoFrame], loaded, device: str, amp: bool,
            stride: int) -> list[dict]:
    """Detections (original pixels) and predicted lanes for every frame of ``loader``."""
    head.eval()
    cfg = loaded.cfg
    mw, mh = cfg.data.img_w, cfg.data.img_h
    anchors = loaded.row_anchors
    out: list[dict | None] = [None] * len(frames)
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp and device.startswith("cuda")):
            lane_out, feats = tap(x)
            det = head(feats)
        dets = decode(det, stride)
        exist, bins = decode_logits(lane_out["logits"].float())
        grid = lane_out["logits"].shape[1] - 1
        x_model = (bins * (mw - 1) / (grid - 1)).cpu().numpy()
        exist = exist.cpu().numpy()
        for j, idx in enumerate(batch["index"].tolist()):
            frame = frames[idx]
            ow, oh = frame.image_size
            d = dets[j]
            d["boxes"] = d["boxes"] * np.array([ow / mw, oh / mh, ow / mw, oh / mh])
            d["lanes"] = lanes_from_prediction(exist[j], x_model[j], anchors, loaded.postprocess, (mw, mh), (ow, oh))
            out[idx] = d
    return out


# -- metrics --------------------------------------------------------------------------------------------------------

def detection_metrics(frames: list[CipoFrame], preds: list[dict], full: bool = True) -> dict[str, float]:
    """AP@0.5 for level 1 and class-agnostic; ``full`` adds levels 2-4, AP@0.75 and AP@[.5:.95] for level 1."""
    m: dict[str, float] = {}
    coco = np.round(np.arange(0.5, 0.96, 0.05), 2)
    for level in range(1, NUM_LEVELS + 1 if full else 2):
        gt = {f.key: f.boxes_xyxy(levels=(level,)) for f in frames}
        dets = [(f.key, float(s), b) for f, p in zip(frames, preds)
                for b, s in zip(p["boxes"], p["level_scores"][:, level - 1])]
        if level == 1 and full:
            ap = average_precisions(dets, gt, coco)
            m["ap50_level1"], m["ap75_level1"], m["ap_coco_level1"] = float(ap[0]), float(ap[5]), float(np.mean(ap))
        else:
            m[f"ap50_level{level}"] = average_precision(dets, gt, 0.5)
    gt_all = {f.key: f.boxes_xyxy() for f in frames}
    dets_all = [(f.key, float(s), b) for f, p in zip(frames, preds) for b, s in zip(p["boxes"], p["scores"])]
    m["ap50_any_level"] = average_precision(dets_all, gt_all, 0.5)
    return m


def _geometry(frames: list[CipoFrame], boxes: list[np.ndarray], lanes: list) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for f, b, ln in zip(frames, boxes, lanes):
        left, right = ego_boundaries(ln)
        out.append(path_geometry(b, left, right, f.camera))
    return out


def _select_all(frames, boxes, scores, l1, geometry, lanes, rule: CipoRule) -> list[int | None]:
    sel = []
    for f, b, s, q, g, ln in zip(frames, boxes, scores, l1, geometry, lanes):
        left, right = ego_boundaries(ln)
        sel.append(select_cipo(b, s, q, left, right, f.camera, rule, geometry=g))
    return sel


def _decisions(frames, boxes, selection) -> list[np.ndarray | None]:
    return [None if i is None else b[i] for b, i in zip(boxes, selection)]


def tune_rule(mode: str, frames, boxes, scores, l1, geometry, lanes) -> tuple[CipoRule, float]:
    """Best rule of one mode on these (validation) frames, by CIPO frame accuracy."""
    gt = [f.boxes_xyxy(levels=(1,), types=None) for f in frames]
    if mode == "learned":
        grid = [CipoRule("learned", level1_threshold=t) for t in SCORE_GRID]
    elif mode == "geometric":
        grid = [CipoRule("geometric", score_threshold=s, max_distance=d) for s, d in product(SCORE_GRID, DISTANCE_GRID)]
    elif mode == "oracle":   # labelled boxes (score 1): only the reaction distance is tuned
        grid = [CipoRule("geometric", score_threshold=0.5, max_distance=d) for d in DISTANCE_GRID]
    else:
        grid = [CipoRule("hybrid", score_threshold=s, level1_threshold=t, max_distance=d)
                for s, t, d in product(SCORE_GRID, SCORE_GRID, DISTANCE_GRID)]
    best, best_acc = grid[0], -1.0
    for rule in grid:
        acc = cipo_frame_metrics(_decisions(frames, boxes, _select_all(frames, boxes, scores, l1, geometry, lanes,
                                                                       rule)), gt)["cipo_accuracy"]
        if acc > best_acc:
            best, best_acc = rule, acc
    return best, best_acc


def _matched_track(frame: CipoFrame, box: np.ndarray | None) -> str | None:
    if box is None:
        return None
    gts = [b for b in frame.boxes if b.type in PATH_TYPES]
    if not gts:
        return "?"
    iou = box_iou(np.asarray(box)[None], np.asarray([b.xyxy for b in gts]))[0]
    j = int(np.argmax(iou))
    return gts[j].track_id if iou[j] >= 0.5 else "?"


def decision_metrics(frames, preds, lanes_by_source: dict[str, list], rules: dict[str, CipoRule]) -> dict:
    gt = [f.boxes_xyxy(levels=(1,), types=None) for f in frames]
    boxes = [p["boxes"] for p in preds]
    scores = [p["scores"] for p in preds]
    l1 = [p["level_scores"][:, 0] for p in preds]
    lanes = lanes_by_source["predicted"]
    geometry = _geometry(frames, boxes, lanes)
    out = {}
    for mode, rule in rules.items():
        if mode == "oracle":
            continue
        decided = _decisions(frames, boxes, _select_all(frames, boxes, scores, l1, geometry, lanes, rule))
        m = cipo_frame_metrics(decided, gt)
        m.update(cipo_stability([f.sequence for f in frames], [f.frame_index for f in frames],
                                [_matched_track(f, b) for f, b in zip(frames, decided)],
                                [f.level1_track() for f in frames]))
        m["distance"] = predicted_distance(frames, decided)
        out[mode] = m
    return out


def in_path_metrics(frames: list[CipoFrame], lanes_by_source: dict[str, list], oracle_rule: CipoRule) -> dict:
    """Rule validity and oracle-box selection with ground-truth and predicted ego lanes."""
    out: dict = {}
    gt_l1 = [f.boxes_xyxy(levels=(1,), types=None) for f in frames]
    has_gt_lanes = [f.lanes is not None and all(b is not None for b in ego_boundaries(f.lanes)) for f in frames]
    for source, lanes in lanes_by_source.items():
        fractions, spans = [], {}
        for f, ln, ok in zip(frames, lanes, has_gt_lanes):
            if not ok:
                continue
            left, right = ego_boundaries(ln)
            for box in f.boxes_xyxy(levels=(1,), types=None):
                frac, _ = path_geometry(box[None], left, right, f.camera)
                fractions.append(float(frac[0]))
                kind = ego_span(left, right, box[3], f.camera)[2]
                spans[kind] = spans.get(kind, 0) + 1
        fr = np.asarray(fractions)
        oracle_boxes = [f.boxes_xyxy() for f in frames]
        ones = [np.ones(len(b)) for b in oracle_boxes]
        geometry = _geometry(frames, oracle_boxes, lanes)
        decided = _decisions(frames, oracle_boxes,
                             _select_all(frames, oracle_boxes, ones, ones, geometry, lanes, oracle_rule))
        out[f"{source}_lanes"] = {
            "level1_boxes_scored": float(len(fr)),
            "rule_validity": float(np.mean(fr >= 0.5)) if len(fr) else float("nan"),
            "median_in_path_fraction": float(np.median(fr)) if len(fr) else float("nan"),
            "span_sources": spans,
            "oracle_boxes_selection": cipo_frame_metrics(decided, gt_l1),
        }
    return out


def _distance_pairs(frames: list[CipoFrame], boxes: list[np.ndarray | None], reference_boxes: list[np.ndarray | None]
                    ) -> tuple[list[float], list[float]]:
    est, ref = [], []
    for f, box, rbox in zip(frames, boxes, reference_boxes):
        if box is None or rbox is None or f.lane_ground is None:
            continue
        if rbox[3] >= f.image_size[1] - 2:   # cut by the image bottom: contact row not visible
            continue
        r = lane_ground_distance(float(rbox[3]), f.lane_ground)
        if np.isfinite(r):
            est.append(float(f.camera.ground_distance(box[3])))
            ref.append(r)
    return est, ref


def label_distance(frames: list[CipoFrame]) -> dict[str, float]:
    """Flat-ground proxy on the labelled level-1 boxes vs the lane-ground reference (validates the proxy)."""
    boxes = [f.boxes_xyxy(levels=(1,), types=None) for f in frames]
    first = [b[0] if len(b) else None for b in boxes]
    return distance_errors(*_distance_pairs(frames, first, first))


def predicted_distance(frames: list[CipoFrame], decided: list[np.ndarray | None]) -> dict[str, float]:
    """Flat-ground distance of each predicted CIPO that matches a level-1 box vs the reference at THAT box."""
    matched, refs = [], []
    for f, box in zip(frames, decided):
        gt = f.boxes_xyxy(levels=(1,), types=None)
        if box is None or len(gt) == 0:
            matched.append(None)
            refs.append(None)
            continue
        iou = box_iou(np.asarray(box)[None], gt)[0]
        j = int(np.argmax(iou))
        matched.append(box if iou[j] >= 0.5 else None)
        refs.append(gt[j] if iou[j] >= 0.5 else None)
    return distance_errors(*_distance_pairs(frames, matched, refs))


@torch.no_grad()
def cost(tap: FeatureTap, head: CipoHead, loaded, device: str, runs: int, stride: int) -> dict[str, float]:
    cfg = loaded.cfg
    h, w = cfg.data.img_h, cfg.data.img_w
    probe = HeadProbe(head, tap.channels, tap.strides).eval()
    out = {"head_params_millions": count_parameters(head) / 1e6,
           "head_gmacs": count_macs(probe, torch.zeros(1, 3, h, w, device=device)) / 1e9,
           "lane_model_params_millions": count_parameters(loaded.model) / 1e6,
           "lane_model_gmacs": count_macs(loaded.model, torch.zeros(1, 3, h, w, device=device)) / 1e9,
           "input": f"{w}x{h}", "head_stride": stride}
    if runs > 0 and device.startswith("cuda"):
        x = torch.randn(1, 3, h, w, device=device)

        def timed(fn) -> float:
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / runs * 1000.0

        out["lane_model_ms"] = timed(lambda: loaded.model(x))
        out["lane_model_plus_head_ms"] = timed(lambda: head(tap(x)[1]))
        out["latency_device"] = torch.cuda.get_device_name(0)
        out["latency_note"] = "batch 1, FP32, no TensorRT; includes Python overhead"
    return out


# -- main -----------------------------------------------------------------------------------------------------------

def train(tap, head, train_loader, val_loader, val_frames, loaded, args, device, stride) -> list[dict]:
    amp = not args.no_amp and device.startswith("cuda")
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = max(1, args.epochs * len(train_loader))
    warm = min(200, steps // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, best_score, best_state = [], -1.0, None
    for epoch in range(1, args.epochs + 1):
        head.train()
        t0, sums, n = time.perf_counter(), {"loss": 0.0, "heat": 0.0, "offset": 0.0, "size": 0.0}, 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=amp):
                _, feats = tap(batch["image"])
            with torch.autocast(device_type="cuda", enabled=amp):
                out = head([f.detach() for f in feats])
            loss, parts = cipo_loss(out, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums["loss"] += float(loss.detach())
            for k, v in parts.items():
                sums[k] += v
            n += 1
        preds = predict(tap, head, val_loader, val_frames, loaded, device, amp, stride)
        det = detection_metrics(val_frames, preds, full=False)
        score = det["ap50_level1"] if np.isfinite(det["ap50_level1"]) else det["ap50_any_level"]
        row = {"epoch": epoch, **{f"train_{k}": v / max(n, 1) for k, v in sums.items()},
               **{f"val_{k}": v for k, v in det.items()}, "seconds": time.perf_counter() - t0}
        improved = score > best_score
        if improved:
            best_score, best_state = score, {k: v.detach().clone() for k, v in head.state_dict().items()}
        LOGGER.info("epoch %d/%d loss %.4f (heat %.4f off %.4f size %.4f) val AP50 L1 %.3f any %.3f %s %.0fs",
                    epoch, args.epochs, row["train_loss"], row["train_heat"], row["train_offset"], row["train_size"],
                    det["ap50_level1"], det["ap50_any_level"], "BEST" if improved else "", row["seconds"])
        history.append(row)
    if best_state is not None:
        head.load_state_dict(best_state)
    return history


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, CipoRule):
        return o.__dict__
    raise TypeError(type(o))


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    loaded = load_model(args.lane_ckpt, device=device)
    cfg = loaded.cfg
    if cfg.data.preprocessing.mode != "rgb":
        raise SystemExit(f"lane model uses preprocessing '{cfg.data.preprocessing.mode}'; the CIPO dataset feeds RGB")
    for p in loaded.model.parameters():
        p.requires_grad_(False)
    loaded.model.eval()
    tap = FeatureTap(loaded.model)
    stride = tap.strides[0]
    head = CipoHead(tap.channels, width=args.width).to(device)

    root = _openlane_root(args, cfg)
    source = CipoSource(args.cipo if args.cipo is not None else root / "cipo")
    manifest = args.split_manifest or args.lane_ckpt.resolve().parents[2] / "data" / "split_manifest.csv"
    sequences = read_split_manifest(manifest)
    caps = {"train": args.max_train_frames, "val": args.max_val_frames, "test": args.max_test_frames}
    t0 = time.perf_counter()
    splits, stats = build_frames(root, source, sequences, caps)
    LOGGER.info("frames %s (read in %.0f s): %s", {k: len(v) for k, v in splits.items()},
                time.perf_counter() - t0, stats)
    for split in ("train", "val", "test"):
        if not splits.get(split):
            raise SystemExit(f"no CIPO frames for split '{split}' (manifest {manifest})")

    loaders = {s: _loader(splits[s], cfg, args, s == "train", stride, persistent=s != "test")
               for s in ("train", "val", "test")}
    history = train(tap, head, loaders["train"], loaders["val"], splits["val"], loaded, args, device, stride)
    amp = not args.no_amp and device.startswith("cuda")

    preds = {s: predict(tap, head, loaders[s], splits[s], loaded, device, amp, stride) for s in ("val", "test")}
    lanes = {s: {"ground_truth": [f.lanes for f in splits[s]], "predicted": [p["lanes"] for p in preds[s]]}
             for s in ("val", "test")}

    # Decision rules: tuned on validation (predicted lanes; the oracle rule on ground-truth lanes and boxes).
    val, vp = splits["val"], preds["val"]
    vb, vs, vl1 = [p["boxes"] for p in vp], [p["scores"] for p in vp], [p["level_scores"][:, 0] for p in vp]
    vgeo = _geometry(val, vb, lanes["val"]["predicted"])
    rules, val_acc = {}, {}
    for mode in ("learned", "geometric", "hybrid"):
        rules[mode], val_acc[mode] = tune_rule(mode, val, vb, vs, vl1, vgeo, lanes["val"]["predicted"])
    ob = [f.boxes_xyxy() for f in val]
    ones = [np.ones(len(b)) for b in ob]
    rules["oracle"], val_acc["oracle"] = tune_rule("oracle", val, ob, ones, ones,
                                                   _geometry(val, ob, lanes["val"]["ground_truth"]),
                                                   lanes["val"]["ground_truth"])

    test = splits["test"]
    metrics = {
        "lane_checkpoint": str(args.lane_ckpt), "variant": loaded.variant, "backbone": tap.kind,
        "frames": {k: len(v) for k, v in splits.items()}, "data_stats": stats, "args": vars(args),
        "detection_test": detection_metrics(test, preds["test"]),
        "detection_val": detection_metrics(val, vp),
        "rules": rules, "rule_val_accuracy": val_acc,
        "decision_test": decision_metrics(test, preds["test"], lanes["test"], rules),
        "in_path_test": in_path_metrics(test, lanes["test"], rules["oracle"]),
        "label_distance_test": label_distance(test),
        "cost": cost(tap, head, loaded, device, args.latency_runs, stride),
        "level1_frames_test": float(np.mean([len(f.boxes_xyxy(levels=(1,), types=None)) > 0 for f in test])),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
    with open(args.out / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    torch.save({"state_dict": head.state_dict(), "channels": tap.channels, "strides": tap.strides,
                "width": args.width, "lane_checkpoint": str(args.lane_ckpt), "rules": {k: v.__dict__ for k, v in
                                                                                        rules.items()}},
               args.out / "head.pt")
    d = metrics["detection_test"]
    LOGGER.info("TEST AP50 level1 %.3f | AP50 any %.3f | AP[.5:.95] level1 %.3f", d["ap50_level1"],
                d["ap50_any_level"], d["ap_coco_level1"])
    for mode, m in metrics["decision_test"].items():
        LOGGER.info("TEST CIPO %-9s acc %.3f P %.3f R %.3f false %.3f missed %.3f", mode, m["cipo_accuracy"],
                    m["cipo_precision"], m["cipo_recall"], m["false_cipo_rate"], m["missed_cipo_rate"])
    LOGGER.info("wrote %s", args.out / "metrics.json")
    tap.remove()
    return metrics


if __name__ == "__main__":
    main()
