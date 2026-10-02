"""Evaluate decoded predictions against FrameRecords."""

from __future__ import annotations

import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tac_ufld.config import ExperimentConfig
from tac_ufld.data.targets import LaneTargets, encode_targets
from tac_ufld.data.types import FrameRecord
from tac_ufld.evaluation.native import ALL_LANES_NATIVE, all_lanes_counts, all_lanes_metrics, native_metrics
from tac_ufld.evaluation.predictor import Predictions
from tac_ufld.metrics import (
    AnchorCounts, anchor_counts, culane_line_width, f_beta, iou_matrix, lane_mask,
    match_lanes, pixel_counts, prf, temporal_jitter,
)
from tac_ufld.postprocess import PostprocessParams, lanes_from_prediction


def iou_tag(threshold: float) -> str:
    return f"iou{int(round(threshold * 100)):02d}"


@dataclass
class EvalResult:
    metrics: dict[str, float]
    per_frame: pd.DataFrame
    pred_lanes: list[list[np.ndarray | None]]


class Evaluator:
    """Stateless apart from caches of per-record targets."""

    def __init__(self, cfg: ExperimentConfig, row_anchors: np.ndarray) -> None:
        self.cfg = cfg
        self.anchors = row_anchors
        d = cfg.data
        self.model_size = (d.img_w, d.img_h)
        self.griding_num = d.griding_num
        self.thresholds = list(cfg.evaluation.iou_thresholds)
        cell_px = (d.img_w - 1) / (d.griding_num - 1)
        self.tol_bins = int(round(cfg.evaluation.anchor_tolerance_px / cell_px))
        self._targets: dict[str, LaneTargets] = {}

    @property
    def primary_tag(self) -> str:
        return iou_tag(self.thresholds[0])

    def _target(self, rec: FrameRecord) -> LaneTargets:
        if rec.key not in self._targets:
            d = self.cfg.data
            self._targets[rec.key] = encode_targets(rec, d.img_w, d.img_h, self.anchors, d.griding_num)
        return self._targets[rec.key]

    def _masks(self, lanes: list[np.ndarray], rec: FrameRecord, line_width: int) -> list[np.ndarray]:
        """Masks cropped to the annotated band (identical IoU, less work)."""
        w, h = rec.image_size
        y0, y1 = (0, h) if rec.valid_y_range is None else (
            max(0, int(np.floor(rec.valid_y_range[0])) - line_width),
            min(h, int(np.ceil(rec.valid_y_range[1])) + line_width + 1),
        )
        out = []
        for lane in lanes:
            shifted = lane.copy()
            shifted[:, 1] -= y0
            out.append(lane_mask(shifted, w, y1 - y0, line_width))
        return out

    def _score_frame(self, exist: np.ndarray, x: np.ndarray, weight: float, rec: FrameRecord,
                     params: PostprocessParams, lane_only: bool, native_all_lanes: bool) -> tuple:
        """Everything ``evaluate`` needs from one frame; frames are independent, so this also runs in worker
        processes (``evaluation.workers``). Returns (row, lanes, lane counts per IoU threshold, pixel counts,
        anchor counts, jitter item, all-lanes native counts)."""
        ev = self.cfg.evaluation
        w, h = rec.image_size
        lw = culane_line_width(w, ev.culane_line_width, ev.culane_image_width)
        lanes = lanes_from_prediction(exist, x, self.anchors, params, self.model_size,
                                      rec.image_size, rec.valid_y_range, rec.slot_known)
        pred = [lane for lane in lanes if lane is not None]
        gt = [lane for lane, known in zip(rec.lanes, rec.slot_known) if lane is not None and known]
        pm, gm = self._masks(pred, rec, lw), self._masks(gt, rec, lw)
        ious = iou_matrix(pm, gm)
        row = {"sequence": rec.sequence, "frame_id": rec.frame_id, "n_gt": len(gt), "n_pred": len(pred)}
        counts = {}
        for t in self.thresholds:
            m = match_lanes(ious, t)
            counts[t] = (m.tp, m.fp, m.fn)
            tag = iou_tag(t)
            row.update({f"tp_{tag}": m.tp, f"fp_{tag}": m.fp, f"fn_{tag}": m.fn})
        pix, anchors, jitter, native = (0, 0, 0), AnchorCounts(), None, None
        if not lane_only:
            shape = pm[0].shape if pm else (gm[0].shape if gm else (1, 1))
            pix = pixel_counts(pm, gm, shape)
            _, _, row["pixel_f1"] = prf(*pix)
            anchors = anchor_counts(exist, x, self._target(rec).cls, params.threshold,
                                    self.tol_bins, self.model_size[0], self.griding_num)
            jitter = (rec.sequence, rec.frame_id, exist, x * (w / self.model_size[0]))
            if native_all_lanes:
                native = all_lanes_counts(rec, lanes, line_width=30)
        row["current_weight"] = float(weight)
        return row, lanes, counts, pix, anchors, jitter, native

    def _score_parallel(self, preds: Predictions, records: list[FrameRecord], params: PostprocessParams,
                        lane_only: bool, native_all_lanes: bool) -> list[tuple]:
        """Frames split into chunks over ``evaluation.workers`` spawned processes; the pool lives for one call
        (no idle workers holding memory while the next model trains)."""
        workers = self.cfg.evaluation.workers
        bounds = np.linspace(0, len(records), min(len(records), 4 * workers) + 1).astype(int)
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker, initargs=(self.cfg, self.anchors)) as pool:
            futures = [pool.submit(_score_chunk, preds.exist[a:b], preds.x[a:b], preds.current_weight[a:b],
                                   records[a:b], params, lane_only, native_all_lanes)
                       for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
            return [frame for future in futures for frame in future.result()]

    def evaluate(self, preds: Predictions, records: list[FrameRecord], params: PostprocessParams,
                 lane_only: bool = False) -> EvalResult:
        if len(records) != len(preds.exist):
            raise ValueError(f"{len(records)} records but {len(preds.exist)} predictions")
        dataset = records[0].dataset if records else None
        parallel = self.cfg.evaluation.workers > 1 and len(records) >= 8 * self.cfg.evaluation.workers
        native_in_frames = parallel and not lane_only and dataset in ALL_LANES_NATIVE
        if parallel:
            frames = self._score_parallel(preds, records, params, lane_only, native_in_frames)
        else:
            frames = [self._score_frame(preds.exist[i], preds.x[i], preds.current_weight[i], rec, params,
                                        lane_only, False) for i, rec in enumerate(records)]
        lane_tot = {t: np.zeros(3, dtype=np.int64) for t in self.thresholds}
        pix = np.zeros(3, dtype=np.int64)
        native = np.zeros(3, dtype=np.int64)
        anchors = AnchorCounts()
        rows, all_lanes, jitter_items = [], [], []
        for row, lanes, counts, frame_pix, frame_anchors, jitter, frame_native in frames:
            rows.append(row)
            all_lanes.append(lanes)
            for t in self.thresholds:
                lane_tot[t] += counts[t]
            if not lane_only:
                pix += frame_pix
                anchors += frame_anchors
                jitter_items.append(jitter)
            if frame_native is not None:
                native += frame_native

        metrics: dict[str, float] = {"n_frames": len(records),
                                     "n_gt_lanes": int(sum(r["n_gt"] for r in rows)),
                                     "n_pred_lanes": int(sum(r["n_pred"] for r in rows)),
                                     "empty_prediction_frames": int(sum(r["n_pred"] == 0 for r in rows))}
        for t, (tp, fp, fn) in lane_tot.items():
            p, r, f1 = prf(tp, fp, fn)
            tag = iou_tag(t)
            metrics.update({f"lane_precision_{tag}": p, f"lane_recall_{tag}": r, f"lane_f1_{tag}": f1,
                            f"lane_f2_{tag}": f_beta(p, r, 2.0),
                            f"lane_tp_{tag}": int(tp), f"lane_fp_{tag}": int(fp), f"lane_fn_{tag}": int(fn)})
        per_frame = pd.DataFrame(rows)
        metrics.update(self._condition_metrics(records, per_frame))
        if not lane_only:
            p, r, f1 = prf(*pix)
            metrics.update({"pixel_precision": p, "pixel_recall": r, "pixel_f1": f1})
            metrics.update(anchors.as_metrics())
            metrics.update(temporal_jitter(jitter_items, params.threshold))
            if native_in_frames:
                metrics.update(all_lanes_metrics(int(native[0]), int(native[1]), int(native[2]), 0.5,
                                                 ALL_LANES_NATIVE[dataset]))
            else:
                metrics.update(native_metrics(records, all_lanes))
        if preds.loss is not None:
            metrics["focal_loss"] = preds.loss
        weights = preds.current_weight[np.isfinite(preds.current_weight)]
        metrics["mean_current_frame_weight"] = float(weights.mean()) if weights.size else float("nan")
        return EvalResult(metrics, per_frame, all_lanes)

    def _condition_metrics(self, records: list[FrameRecord], per_frame: pd.DataFrame) -> dict[str, float]:
        """Lane F1 (primary IoU) per scene condition tag. Scenes without tag
        metadata are excluded; ``{}`` tags count as 'nominal'."""
        tag = self.primary_tag
        groups: dict[str, list[int]] = {}
        for i, rec in enumerate(records):
            if rec.tags is None:
                continue
            active = [name for name, on in rec.tags.items() if on] or ["nominal"]
            for name in active:
                groups.setdefault(name, []).append(i)
        out = {}
        for name, idx in sorted(groups.items()):
            sub = per_frame.iloc[idx]
            _, _, f1 = prf(sub[f"tp_{tag}"].sum(), sub[f"fp_{tag}"].sum(), sub[f"fn_{tag}"].sum())
            out[f"condition_f1_{tag}_{name}"] = f1
            out[f"condition_frames_{name}"] = len(idx)
        return out

    def sweep(self, preds: Predictions, records: list[FrameRecord], grid: dict[str, list[float]],
              base: PostprocessParams) -> tuple[PostprocessParams, pd.DataFrame]:
        """Grid-search post-processing on (validation) predictions; the best
        combination maximises ``evaluation.selection_metric``."""
        metric = self.cfg.evaluation.selection_metric
        keys = list(grid)
        rows, best, best_score = [], base, -np.inf
        for combo in itertools.product(*(grid[k] for k in keys)):
            params = base.replace(**{k: (int(v) if k in ("min_points", "poly_degree") else float(v))
                                     for k, v in zip(keys, combo)})
            result = self.evaluate(preds, records, params, lane_only=True)
            score = result.metrics[metric]
            rows.append({**params.as_dict(), **{k: v for k, v in result.metrics.items() if k.startswith("lane_")}})
            if score > best_score:
                best, best_score = params, score
        return best, pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ worker processes (evaluation.workers > 1)
_WORKER: Evaluator | None = None


def _init_worker(cfg: ExperimentConfig, row_anchors: np.ndarray) -> None:
    global _WORKER
    _WORKER = Evaluator(cfg, row_anchors)


def _score_chunk(exist: np.ndarray, x: np.ndarray, weight: np.ndarray, records: list[FrameRecord],
                 params: PostprocessParams, lane_only: bool, native_all_lanes: bool) -> list[tuple]:
    assert _WORKER is not None, "worker not initialised"
    return [_WORKER._score_frame(exist[i], x[i], weight[i], rec, params, lane_only, native_all_lanes)
            for i, rec in enumerate(records)]
