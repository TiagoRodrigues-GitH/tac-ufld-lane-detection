"""Evaluate decoded predictions against FrameRecords."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tac_ufld.config import ExperimentConfig
from tac_ufld.data.targets import LaneTargets, encode_targets
from tac_ufld.data.types import FrameRecord
from tac_ufld.evaluation.native import native_metrics
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

    def evaluate(self, preds: Predictions, records: list[FrameRecord], params: PostprocessParams,
                 lane_only: bool = False) -> EvalResult:
        if len(records) != len(preds.exist):
            raise ValueError(f"{len(records)} records but {len(preds.exist)} predictions")
        ev = self.cfg.evaluation
        lane_tot = {t: np.zeros(3, dtype=np.int64) for t in self.thresholds}
        pix = np.zeros(3, dtype=np.int64)
        anchors = AnchorCounts()
        rows, all_lanes, jitter_items = [], [], []
        for i, rec in enumerate(records):
            w, h = rec.image_size
            lw = culane_line_width(w, ev.culane_line_width, ev.culane_image_width)
            lanes = lanes_from_prediction(preds.exist[i], preds.x[i], self.anchors, params, self.model_size,
                                          rec.image_size, rec.valid_y_range, rec.slot_known)
            all_lanes.append(lanes)
            pred = [lane for lane in lanes if lane is not None]
            gt = [lane for lane, known in zip(rec.lanes, rec.slot_known) if lane is not None and known]
            pm, gm = self._masks(pred, rec, lw), self._masks(gt, rec, lw)
            ious = iou_matrix(pm, gm)
            row = {"sequence": rec.sequence, "frame_id": rec.frame_id, "n_gt": len(gt), "n_pred": len(pred)}
            for t in self.thresholds:
                m = match_lanes(ious, t)
                lane_tot[t] += (m.tp, m.fp, m.fn)
                tag = iou_tag(t)
                row.update({f"tp_{tag}": m.tp, f"fp_{tag}": m.fp, f"fn_{tag}": m.fn})
            if not lane_only:
                shape = pm[0].shape if pm else (gm[0].shape if gm else (1, 1))
                p_tp, p_fp, p_fn = pixel_counts(pm, gm, shape)
                pix += (p_tp, p_fp, p_fn)
                _, _, row["pixel_f1"] = prf(p_tp, p_fp, p_fn)
                anchors += anchor_counts(preds.exist[i], preds.x[i], self._target(rec).cls, params.threshold,
                                         self.tol_bins, self.model_size[0], self.griding_num)
                jitter_items.append((rec.sequence, rec.frame_id, preds.exist[i],
                                     preds.x[i] * (w / self.model_size[0])))
            row["current_weight"] = float(preds.current_weight[i])
            rows.append(row)

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
