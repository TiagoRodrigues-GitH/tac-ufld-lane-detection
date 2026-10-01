"""Robustness to a degraded current frame (``python -m tac_ufld robustness``).

Clean test frames are where a single-frame model has everything it needs,
so they cannot show what temporal information is for. This evaluation
re-runs every trained checkpoint of a finished run on the held-out test
scenes with the CURRENT frame of each sample degraded (occluding boxes,
strong blur, darkening or noise; one op per frame, chosen and parameterised
by a fixed per-frame seed) while the history frames stay clean. Every model
sees exactly the same corruptions; the capacity-control models see the
degraded current frame in every position, like a single-frame model.

Reported per model and seed: lane F1 on all degraded frames, per
degradation op, and with the validation-tuned Kalman tracker, next to the
clean-frame F1 of the run. The validation-tuned post-processing of each
checkpoint is used; nothing is re-tuned on test.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from tac_ufld.config import AugmentationConfig, load_config
from tac_ufld.data import build_adapter
from tac_ufld.data.dataset import TemporalLaneDataset
from tac_ufld.data.splits import build_splits
from tac_ufld.evaluation.evaluator import Evaluator, iou_tag
from tac_ufld.evaluation.predictor import predict
from tac_ufld.evaluation.tracking import KalmanParams, track_predictions
from tac_ufld.inference.loading import load_model
from tac_ufld.metrics import prf

LOGGER = logging.getLogger(__name__)


def _f1(frame: pd.DataFrame, tag: str) -> float:
    return prf(frame[f"tp_{tag}"].sum(), frame[f"fp_{tag}"].sum(), frame[f"fn_{tag}"].sum())[2]


def evaluate_run(run: Path, device: str, seed: int = 2026, batch_size: int = 8, workers: int = 2) -> pd.DataFrame:
    run = Path(run)
    cfg = load_config(run / "config_resolved.yaml")
    adapter = build_adapter(cfg)
    splits, _ = build_splits(cfg, adapter)
    degradation = AugmentationConfig()  # the same corruptions for every run and model
    clean = pd.read_csv(run / "all_results.csv")
    clean = clean[(clean["split"] == "test") & (clean["protocol"] == "tuned") & (clean["input"] == "full")]
    rows = []
    for ckpt in sorted(p for p in run.glob("seed_*/checkpoints/*.pt") if not p.name.endswith(".last.pt")):
        loaded = load_model(ckpt, device)
        data = dataclasses.replace(loaded.cfg.data, augmentation=degradation)
        ds = TemporalLaneDataset(splits.test, adapter, data, loaded.row_anchors, loaded.num_frames,
                                 static_history=bool(loaded.card.get("static_history")), eval_degradation_seed=seed)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
        preds = predict(loaded.model, loader, device, data.img_w, amp=device.startswith("cuda"))
        evaluator = Evaluator(loaded.cfg, loaded.row_anchors)
        tag = iou_tag(loaded.cfg.evaluation.iou_thresholds[0])
        seed_id = int(ckpt.parent.parent.name.removeprefix("seed_"))
        c = clean[(clean["seed"] == seed_id) & (clean["variant"] == loaded.variant)]
        base = {"seed": seed_id, "variant": loaded.variant,
                "clean_lane_f1": float(c[f"lane_f1_{tag}"].iloc[0]) if len(c) else float("nan")}
        inputs = [("degraded", preds)]
        if loaded.kalman:
            inputs.append(("degraded+kalman", track_predictions(preds, ds.records, KalmanParams(**loaded.kalman))))
        for name, p in inputs:
            per_frame = evaluator.evaluate(p, ds.records, loaded.postprocess, lane_only=True).per_frame
            per_frame["op"] = ds.eval_ops
            overall = _f1(per_frame, tag)
            rows.append({**base, "input": name, "op": "all", "lane_f1": overall, "frames": len(per_frame)})
            for op, sub in per_frame.groupby("op"):
                rows.append({**base, "input": name, "op": op, "lane_f1": _f1(sub, tag), "frames": len(sub)})
            LOGGER.info("[robustness] %s seed %d %s: clean %.3f, degraded %.3f", loaded.variant, seed_id, name,
                        base["clean_lane_f1"], overall)
    df = pd.DataFrame(rows)
    out = run / "report"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "robustness.csv", index=False)
    summary = (df.groupby(["variant", "input", "op"], sort=False)
               .agg(lane_f1=("lane_f1", "mean"), lane_f1_std=("lane_f1", "std"), clean_lane_f1=("clean_lane_f1", "mean"),
                    n_seeds=("seed", "nunique"), frames=("frames", "first")).reset_index())
    summary["drop_vs_clean"] = summary["lane_f1"] - summary["clean_lane_f1"]
    summary.to_csv(out / "robustness_summary.csv", index=False)
    _paired(df, cfg).to_csv(out / "robustness_paired_tests.csv", index=False)
    return summary


def _paired(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Same-seed paired tests on occluded-frame and all-degraded lane F1: every
    temporal model against its single-frame reference, its equal-training
    control and its capacity control, plus the declared primary pairs (one
    Holm family on occluded-frame F1)."""
    from tac_ufld.evaluation.stats import paired_comparisons
    from tac_ufld.models.registry import resolve_spec

    deg = df[df["input"] == "degraded"]
    wide = deg.pivot_table(index=["variant", "seed"], columns="op", values="lane_f1").reset_index()
    wide = wide.rename(columns={"occlude": "occluded_lane_f1", "all": "degraded_lane_f1"})
    present = set(wide["variant"])
    pairs = []
    for v in present:
        spec = resolve_spec(v, cfg)
        for ref in (spec.reference, spec.budget_reference, spec.static_reference):
            if spec.temporal and ref in present and (v, ref) not in pairs:
                pairs.append((v, ref))
    primary = [tuple(p) for p in cfg.evaluation.primary_pairs if all(v in present for v in p)]
    pairs += [p for p in primary if p not in pairs]
    metrics = {m: True for m in ("occluded_lane_f1", "degraded_lane_f1") if m in wide}
    return paired_comparisons(wide, sorted(pairs), metrics, primary_pairs=primary, primary_metric="occluded_lane_f1")
