"""End-to-end experiment orchestration.

Order of operations (the test split is only read in step 5):

1. data     - parse the dataset, build the seed-independent split, check for
              leakage, save manifests, render label-check overlays and the
              ELAS geometry check.
2. hpo      - Optuna per variant on (a subset of) train, scored on val.
3. training - for every seed, train variants in warm-start order; each
              temporal variant starts from the SAME seed's reference model.
4. tuning   - per model and seed, sweep post-processing on validation.
5. testing  - held-out-scene test and seen-scene test, with the validation-
              tuned and with one common fixed post-processing; temporal
              ablation (history replaced by the current frame).
6. report   - efficiency, multi-seed aggregates, paired tests, figures,
              CSV / Excel / Markdown.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tac_ufld import __version__
from tac_ufld.config import PROJECT_ROOT, ExperimentConfig, expand_env, save_config
from tac_ufld.data import build_adapter
from tac_ufld.data.dataset import TemporalLaneDataset
from tac_ufld.data.splits import DataSplits, build_splits, cap_records
from tac_ufld.data.targets import encode_targets, make_row_anchors
from tac_ufld.data.types import FrameRecord
from tac_ufld.evaluation.efficiency import measure_efficiency
from tac_ufld.evaluation.evaluator import Evaluator, iou_tag
from tac_ufld.evaluation.predictor import predict
from tac_ufld.evaluation.recurrent_eval import carry_predictions
from tac_ufld.evaluation.tracking import track_predictions, tune_kalman
from tac_ufld.evaluation.stats import aggregate, paired_comparisons
from tac_ufld.inference.card import model_card
from tac_ufld.losses import Hyperparams
from tac_ufld.models.registry import (
    SHORT_LABELS, VARIANTS, build_model, init_from_checkpoint, resolve_spec, training_order, warm_start,
)
from tac_ufld.postprocess import PostprocessParams
from tac_ufld.reporting import (
    config_diff, export_lines_txt, markdown_table, protocol_section, summary_table, write_excel,
)
from tac_ufld.training.hpo import active_space, run_study
from tac_ufld.training.trainer import Trainer, TrainResult, load_checkpoint
from tac_ufld.utils import environment_info, read_json, seed_worker, set_seed, setup_logging, write_json
from tac_ufld.visualization import overlays, plots

LOGGER = logging.getLogger(__name__)
EVAL_SPLITS = ("test", "seen_test")


class FullRunNotConfirmed(RuntimeError):
    """A configuration marked ``requires_confirmation`` was started without confirmation."""


class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig, variants: list[str] | None = None,
                 seeds: list[int] | None = None, resume: bool = False, confirmed: bool = False) -> None:
        self.cfg = cfg
        self.confirmed = confirmed
        self.out = cfg.output_root()
        self.out.mkdir(parents=True, exist_ok=True)
        setup_logging(self.out / "run.log")
        self.device = cfg.resolve_device()
        self.variants = training_order(variants or cfg.model.variants, cfg)
        self.seeds = list(seeds or cfg.train.seeds)
        self.resume = resume
        d = cfg.data
        self.anchors = make_row_anchors(d.img_h, d.num_row_anchors, d.row_anchor_range)
        self.evaluator = Evaluator(cfg, self.anchors)
        self.labels = {k: VARIANTS[k].display_name for k in VARIANTS}
        self.best_params: dict[str, dict] = {}
        self.adapter = None
        self.splits: DataSplits | None = None
        self._datasets: dict[tuple, TemporalLaneDataset] = {}

    # ================================================================== data

    def prepare_data(self) -> DataSplits:
        cfg = self.cfg
        self.adapter = build_adapter(cfg)
        s = cfg.data.split
        splits, split_info = build_splits(cfg, self.adapter)
        for name in ("train", "val", "test"):
            if not getattr(splits, name):
                raise RuntimeError(f"split '{name}' is empty - check data.scenes / split.test_scenes")
        data_dir = self.out / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        splits.manifest().to_csv(data_dir / "split_manifest.csv", index=False)
        splits.summary().to_csv(data_dir / "split_summary.csv", index=False)
        write_json(data_dir / "split_report.json", split_info)
        if getattr(self.adapter, "stats", None):
            pd.DataFrame(self.adapter.stats).T.rename_axis("sequence").to_csv(data_dir / "dataset_stats.csv")
        LOGGER.info("splits: %s (min gap %d frames, held-out test scenes %s)",
                    {k: len(v) for k, v in splits.as_dict().items()}, cfg.min_split_gap(), s.test_scenes)
        self.splits = splits
        return splits

    def dataset(self, split: str, num_frames: int, augment: bool = False,
                records: list[FrameRecord] | None = None) -> TemporalLaneDataset:
        key = (split, num_frames, augment, None if records is None else len(records))
        if key not in self._datasets:
            recs = records if records is not None else self.splits.as_dict()[split]
            self._datasets[key] = TemporalLaneDataset(recs, self.adapter, self.cfg.data, self.anchors,
                                                      num_frames, augment=augment)
        return self._datasets[key]

    def loader(self, dataset: TemporalLaneDataset, shuffle: bool, seed: int = 0) -> DataLoader:
        workers = self.cfg.data.num_workers
        return DataLoader(
            dataset, batch_size=self.cfg.train.batch_size, shuffle=shuffle, num_workers=workers,
            pin_memory=self.device.startswith("cuda"), worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed), persistent_workers=workers > 0,
            drop_last=shuffle and len(dataset) > self.cfg.train.batch_size,
        )

    def check_labels(self, per_sequence: int = 2, max_sequences: int = 40) -> pd.DataFrame:
        """Label-check overlays + straight-line residual of the 4 ELAS points.
        Overlays are drawn for at most ``max_sequences`` evenly spaced
        sequences (CULane/TuSimple have thousands of clips)."""
        d = self.cfg.data
        by_seq: dict[str, list[FrameRecord]] = {}
        for recs in self.splits.as_dict().values():
            for r in recs:
                by_seq.setdefault(r.sequence, []).append(r)
        names = sorted(by_seq)
        overlay_seqs = {names[int(i)] for i in np.linspace(0, len(names) - 1, min(max_sequences, len(names)))}
        chosen, rows = [], []
        for seq, recs in sorted(by_seq.items()):
            recs.sort(key=lambda r: r.frame_id)
            if seq in overlay_seqs:
                chosen += [recs[int(i)] for i in np.linspace(0, len(recs) - 1, min(per_sequence, len(recs)))]
            residuals = []
            for r in recs:
                for lane in r.present_lanes():
                    if len(lane) == 4:
                        coef = np.polyfit(lane[:, 1], lane[:, 0], 1)
                        residuals.append(np.abs(np.polyval(coef, lane[:, 1]) - lane[:, 0]).max())
            if residuals:
                rows.append({"sequence": seq, "lanes_checked": len(residuals),
                             "median_line_residual_px": float(np.median(residuals)),
                             "p90_line_residual_px": float(np.percentile(residuals, 90))})
        targets = [encode_targets(r, d.img_w, d.img_h, self.anchors, d.griding_num) for r in chosen]
        overlays.save_label_check(chosen, targets, self.anchors, (d.img_w, d.img_h), self.out / "data" / "label_check")
        geometry = pd.DataFrame(rows)
        geometry.to_csv(self.out / "data" / "label_geometry_check.csv", index=False)
        if not geometry.empty and (geometry["median_line_residual_px"] > 6).any():
            LOGGER.warning("label geometry check: some scenes have median residual > 6 px:\n%s", geometry)
        return geometry

    # ============================================================ parameters

    def base_hyperparams(self) -> Hyperparams:
        t = self.cfg.train
        return Hyperparams(lr=t.lr, lr_fusion=t.lr_fusion, weight_decay=t.weight_decay,
                           lambda_temporal=t.lambda_temporal, lambda_coord=t.lambda_coord,
                           lr_backbone_mult=t.lr_backbone_mult)

    def hyperparams(self, variant: str) -> Hyperparams:
        return replace(self.base_hyperparams(), **self.best_params.get(variant, {}))

    # ============================================================== training

    def _build(self, variant: str, warm_from: Path | None) -> torch.nn.Module:
        spec = resolve_spec(variant, self.cfg)
        model = build_model(variant, self.cfg)
        if spec.warm_start:
            if warm_from is None or not Path(warm_from).exists():
                raise FileNotFoundError(f"{variant} needs a trained '{spec.warm_start}' checkpoint")
            payload = torch.load(warm_from, map_location="cpu", weights_only=False)
            loaded = warm_start(model, payload["state_dict"])
            LOGGER.info("[%s] warm start from %s (%s)", variant, Path(warm_from).name, ", ".join(loaded))
        elif self.cfg.model.init_checkpoint:
            path = Path(expand_env(self.cfg.model.init_checkpoint, "model.init_checkpoint"))
            path = path if path.is_absolute() else PROJECT_ROOT / path
            info = init_from_checkpoint(model, str(path), self.cfg.model.init_scope)
            LOGGER.info("[%s] initialised from %s: %d tensors loaded, %d skipped (shape mismatch)",
                        variant, path.name, len(info["loaded"]), len(info["skipped"]))
        return model.to(self.device)

    def _frames(self, variant: str) -> int:
        return self.cfg.data.num_frames if resolve_spec(variant, self.cfg).temporal else 1

    def train_variant(self, variant: str, seed: int, hp: Hyperparams, epochs: int, patience: int,
                      warm_from: Path | None, checkpoint: Path | None, train_records=None,
                      trial=None, writer=None, tag: str = "", resume: bool = False
                      ) -> tuple[torch.nn.Module, TrainResult]:
        set_seed(seed, self.cfg.train.deterministic)
        spec = resolve_spec(variant, self.cfg)
        frames = self._frames(variant)
        model = self._build(variant, warm_from)
        train_ds = self.dataset("train", frames, augment=True, records=train_records)
        val_ds = self.dataset("val", frames)
        trainer = Trainer(
            model, spec, self.cfg, hp, self.device,
            train_loader=self.loader(train_ds, shuffle=True, seed=seed),
            val_loader=self.loader(val_ds, shuffle=False), val_records=val_ds.records,
            evaluator=self.evaluator, epochs=epochs, patience=patience,
            writer=writer, trial=trial, tag=tag or f"{variant} s{seed}",
        )
        # The checkpoint is self-describing: the resolved config rebuilds the
        # exact architecture and the card fixes the input/output conventions
        # used by streaming inference, ONNX export and the UI.
        meta = {"seed": seed, "config_hash": self.cfg.config_hash(), "version": __version__,
                "config": self.cfg.to_dict(), "card": model_card(self.cfg, variant)}
        result = trainer.fit(checkpoint, extra_meta=meta, resume=resume)
        return model, result

    # =================================================================== HPO

    def run_hpo(self) -> dict[str, dict]:
        cfg, h = self.cfg, self.cfg.hpo
        hpo_dir = self.out / "hpo"
        if h.load_best_params:
            self.best_params = read_json(Path(h.load_best_params))
            LOGGER.info("loaded best hyper-parameters from %s", h.load_best_params)
            return self.best_params
        if (hpo_dir / "best_params.json").exists() and self.resume:
            self.best_params = read_json(hpo_dir / "best_params.json")
            LOGGER.info("resume: reusing %s", hpo_dir / "best_params.json")
            return self.best_params
        if not h.enabled:
            return self.best_params
        targets = [v for v in self.variants if h.variants is None or v in h.variants]
        subset = cap_records(self.splits.train, h.max_train_frames, f"{cfg.data.split.split_seed}:hpo")
        references: dict[str, Path] = {}
        for variant in self.variants:
            spec = resolve_spec(variant, cfg)
            space = active_space(h.search_space, variant, spec.temporal)
            study_file = hpo_dir / f"{variant}_best.json"
            if variant in targets and space and self.resume and study_file.exists():
                self.best_params[variant] = read_json(study_file)["params"]
                LOGGER.info("resume: reusing finished HPO study %s", study_file.name)
            elif variant in targets and space:
                def objective(hp: Hyperparams, trial, _v=variant, _spec=spec) -> float:
                    model, result = self.train_variant(
                        _v, h.seed, hp, h.epochs, patience=0, warm_from=references.get(_spec.warm_start),
                        checkpoint=None, train_records=subset, trial=trial, tag=f"hpo {_v} t{trial.number}")
                    del model
                    self._free()
                    return result.best_score

                result = run_study(variant, space, self.base_hyperparams(), objective, h.n_trials,
                                   h.sampler_seed, hpo_dir)
                self.best_params[variant] = result["params"]
            dependants = [v for v in targets if resolve_spec(v, cfg).warm_start == variant]
            if dependants:  # train this variant once to warm-start the dependants' trials
                ref_path = hpo_dir / "references" / f"{variant}.pt"
                done = ref_path.with_suffix(".done")
                if not (self.resume and ref_path.exists() and done.exists()):
                    model, _ = self.train_variant(variant, h.seed, self.hyperparams(variant), h.epochs, 0,
                                                  references.get(spec.warm_start), ref_path, train_records=subset,
                                                  tag=f"hpo reference {variant}")
                    done.write_text("complete\n", encoding="utf-8")
                    ref_path.with_name(ref_path.stem + ".last.pt").unlink(missing_ok=True)
                    del model
                    self._free()
                references[variant] = ref_path
        write_json(hpo_dir / "best_params.json", self.best_params)
        return self.best_params

    # ================================================================= seeds

    def run_seed(self, seed: int, visualize: bool) -> pd.DataFrame:
        cfg = self.cfg
        seed_dir = self.out / f"seed_{seed}"
        results_path = seed_dir / "results.csv"
        if self.resume and results_path.exists():
            LOGGER.info("resume: seed %d already evaluated", seed)
            return pd.read_csv(results_path)
        writer = None
        if cfg.evaluation.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(self.out / "tensorboard" / f"seed_{seed}"))
        checkpoints: dict[str, Path] = {}
        try:
            for variant in self.variants:
                ckpt = seed_dir / "checkpoints" / f"{variant}.pt"
                history_csv = seed_dir / f"history_{variant}.csv"
                checkpoints[variant] = ckpt
                # The best checkpoint is written during training, so it alone
                # does not prove that training finished: the history CSV does.
                if self.resume and ckpt.exists() and history_csv.exists():
                    LOGGER.info("resume: %s seed %d already trained", variant, seed)
                    continue
                spec = resolve_spec(variant, cfg)
                model, result = self.train_variant(
                    variant, seed, self.hyperparams(variant), cfg.train.epochs, cfg.train.early_stopping_patience,
                    checkpoints.get(spec.warm_start), ckpt, writer=writer, resume=self.resume)
                result.history.to_csv(history_csv, index=False)
                ckpt.with_name(ckpt.stem + ".last.pt").unlink(missing_ok=True)  # large; training is complete
                plots.plot_history(result.history, self.labels[variant], cfg.evaluation.selection_metric,
                                   seed_dir / "plots" / f"history_{variant}")
                del model
                self._free()
        finally:
            if writer is not None:
                writer.close()
        results = self._evaluate_seed(seed, checkpoints, seed_dir, visualize)
        results.to_csv(results_path, index=False)
        return results

    def _evaluate_seed(self, seed: int, checkpoints: dict[str, Path], seed_dir: Path, visualize: bool) -> pd.DataFrame:
        cfg = self.cfg
        common = PostprocessParams.from_config(cfg.evaluation.common_postprocess)
        (seed_dir / "postprocess").mkdir(parents=True, exist_ok=True)
        (seed_dir / "per_frame").mkdir(parents=True, exist_ok=True)
        rows, test_lanes = [], {}
        for variant in self.variants:
            spec = resolve_spec(variant, cfg)
            frames = self._frames(variant)
            model = build_model(variant, cfg, pretrained=False).to(self.device)
            load_checkpoint(checkpoints[variant], model, self.device)
            val_ds = self.dataset("val", frames)
            val_preds = predict(model, self.loader(val_ds, False), self.device, cfg.data.img_w, cfg.train.amp)
            tuned, sweep = self.evaluator.sweep(val_preds, val_ds.records, cfg.evaluation.postprocess_grid, common)
            sweep.to_csv(seed_dir / "postprocess" / f"{variant}_val_sweep.csv", index=False)
            kalman = None
            if cfg.evaluation.kalman:  # output-level tracker, tuned on validation like the post-processing
                kalman, kf_sweep = tune_kalman(self.evaluator, val_preds, val_ds.records, tuned,
                                               cfg.evaluation.kalman_grid, cfg.evaluation.selection_metric)
                kf_sweep.to_csv(seed_dir / "postprocess" / f"{variant}_kalman_val_sweep.csv", index=False)
            write_json(seed_dir / "postprocess" / f"{variant}_tuned.json",
                       {"params": tuned.as_dict(), "selected_on": "val", "metric": cfg.evaluation.selection_metric,
                        "kalman": kalman.as_dict() if kalman else None})
            protocols = {"tuned": tuned, "common": common}
            for split in EVAL_SPLITS:
                if not self.splits.as_dict()[split]:
                    continue
                ds = self.dataset(split, frames)
                preds = predict(model, self.loader(ds, False), self.device, cfg.data.img_w, cfg.train.amp)
                variants_of_input = [("full", preds)]
                if spec.temporal:
                    variants_of_input.append(("static_history", predict(
                        model, self.loader(ds, False), self.device, cfg.data.img_w, cfg.train.amp,
                        static_history=True)))
                if kalman is not None:
                    variants_of_input.append(("kalman", track_predictions(preds, ds.records, kalman)))
                if split == "test" and cfg.evaluation.carry_state_eval and getattr(model, "recurrent", False):
                    variants_of_input.append(("carry", carry_predictions(
                        model, ds, self.adapter, self.device, cfg.data.img_w, cfg.train.amp, cfg.data.temporal_step)))
                for input_mode, p in variants_of_input:
                    for protocol, params in protocols.items():
                        if input_mode != "full" and protocol != "tuned":
                            continue
                        res = self.evaluator.evaluate(p, ds.records, params)
                        kf_cols = {f"kf_{k}": v for k, v in kalman.as_dict().items()} \
                            if input_mode == "kalman" else {}
                        rows.append({"seed": seed, "variant": variant, "split": split, "protocol": protocol,
                                     "input": input_mode, **res.metrics,
                                     **{f"pp_{k}": v for k, v in params.as_dict().items()}, **kf_cols})
                        if protocol == "tuned" and input_mode == "full":
                            res.per_frame.to_csv(seed_dir / "per_frame" / f"{variant}_{split}.csv", index=False)
                            if split == "test":
                                test_lanes[variant] = res.pred_lanes
                            if visualize:
                                export_lines_txt(ds.records, res.pred_lanes,
                                                 seed_dir / "predictions" / split / variant)
                                overlays.save_examples(ds.records, res.pred_lanes, res.per_frame,
                                                       self.evaluator.primary_tag, self.labels[variant],
                                                       seed_dir / "visuals" / split / variant,
                                                       cfg.evaluation.n_visual_examples)
            del model
            self._free()
        if visualize and test_lanes:
            records = self.dataset("test", 1).records
            overlays.save_temporal_video(records, {self.labels[v]: l for v, l in test_lanes.items()},
                                         seed_dir / "visuals" / "test_sequence.mp4")
        return pd.DataFrame(rows)

    # ================================================================ report

    def measure_efficiency(self) -> pd.DataFrame:
        d = self.cfg.data
        rows = []
        for variant in self.variants:
            model = build_model(variant, self.cfg, pretrained=False).to(self.device)
            rows.append({"variant": variant, **measure_efficiency(
                model, self._frames(variant), d.img_h, d.img_w, self.device, self.cfg.evaluation.latency_runs,
                in_channels=self.cfg.in_channels)})
            del model
            self._free()
        return pd.DataFrame(rows)

    def report(self, results: pd.DataFrame, efficiency: pd.DataFrame) -> Path:
        cfg, tag = self.cfg, self.evaluator.primary_tag
        rep = self.out / "report"
        rep.mkdir(parents=True, exist_ok=True)
        main_metrics = [f"lane_f1_{tag}", f"lane_precision_{tag}", f"lane_recall_{tag}", f"lane_f2_{tag}"]
        main_metrics += [f"lane_f1_{iou_tag(t)}" for t in cfg.evaluation.iou_thresholds[1:]]
        main_metrics += ["pixel_f1", "anchor_f1", "jitter_px", "empty_prediction_frames",
                         f"lane_fp_{tag}", f"lane_fn_{tag}"]
        pairs = [(v, resolve_spec(v, cfg).reference) for v in self.variants
                 if resolve_spec(v, cfg).reference in self.variants]
        # equal-training controls: temporal vs the baseline given the same extra
        # training, and that control vs the plain baseline (does more training help?)
        pairs += [(v, resolve_spec(v, cfg).budget_reference) for v in self.variants
                  if resolve_spec(v, cfg).budget_reference in self.variants]
        pairs += [(v, resolve_spec(v, cfg).warm_start) for v in self.variants
                  if not resolve_spec(v, cfg).temporal and resolve_spec(v, cfg).warm_start in self.variants]
        sheets, md = {}, [f"# Experiment report: {cfg.name}\n",
                          f"Config hash `{cfg.config_hash()}`, package {__version__}, seeds {self.seeds}, "
                          f"device {self.device}.\n",
                          f"Primary metric: `{cfg.evaluation.selection_metric}` (CULane-style lane F1, width "
                          "scaled to the image), test split, validation-tuned post-processing.\n"]
        split_info = read_json(self.out / "data" / "split_report.json") \
            if (self.out / "data" / "split_report.json").exists() else {}
        changes = config_diff(cfg)
        write_json(rep / "config_changes.json", changes)
        md += protocol_section(cfg, split_info, self.best_params, self.variants, changes)
        native_cols = sorted(c for c in results.columns if c.startswith("native_") and c != "native_frames")
        for split in EVAL_SPLITS:
            for protocol in ("tuned", "common"):
                sub = results[(results["split"] == split) & (results["protocol"] == protocol)
                              & (results["input"] == "full")]
                if sub.empty:
                    continue
                agg = aggregate(sub, main_metrics)
                name = f"{split}_{protocol}"
                agg.to_csv(rep / f"aggregate_{name}.csv", index=False)
                sheets[f"agg_{name}"] = agg
                md += [f"\n## {split} split, {protocol} post-processing (mean ± std over seeds)\n",
                       markdown_table(summary_table(agg, main_metrics[:6] + ["jitter_px"], self.labels))]
                if native_cols:
                    native = aggregate(sub, native_cols)
                    native.to_csv(rep / f"aggregate_native_{name}.csv", index=False)
                    sheets[f"native_{name}"] = native
                    md += [f"\n### Dataset-native metrics ({split}, {protocol}); not comparable across datasets\n",
                           markdown_table(summary_table(native, native_cols, self.labels))]
                if split == "test" and protocol == "tuned":
                    plots.plot_metric_bars(agg, main_metrics[:6], self.labels, rep / "test_metrics",
                                           "Held-out test scenes (validation-tuned post-processing)")
                    plots.plot_condition_heatmap(sub, self.labels, rep / "test_conditions")
                    plots.plot_error_counts(sub, tag, self.labels, rep / "test_error_counts")
                    plots.plot_efficiency(efficiency, sub, f"lane_f1_{tag}", self.labels, rep / "efficiency")
                    higher = {f"lane_f1_{tag}": True, f"lane_f2_{tag}": True, "pixel_f1": True, "jitter_px": False}
                    paired = paired_comparisons(sub, pairs, higher)
                    paired.to_csv(rep / "paired_tests_test_tuned.csv", index=False)
                    sheets["paired_tests"] = paired
                    md += ["\n## Paired comparisons vs same-family single-frame reference (test, tuned)\n",
                           "Exact two-sided Wilcoxon over seeds, Holm-adjusted. `underpowered` = "
                           "the seed count cannot reach p < 0.05. Rows paired with `*_baseline_ct` compare with "
                           "the baseline that received the same extra training (equal-training control).\n",
                           markdown_table(paired.round(4)) if not paired.empty else "_(no pairs)_\n"]
        md += self._temporal_sections(results, rep, sheets)
        ablation = self._ablation_table(results)
        if not ablation.empty:
            ablation.to_csv(rep / "temporal_ablation.csv", index=False)
            sheets["temporal_ablation"] = ablation
            md += ["\n## Temporal ablation (history frames replaced by the current frame, test, tuned)\n",
                   markdown_table(ablation)]
        efficiency.to_csv(rep / "efficiency.csv", index=False)
        sheets["efficiency"] = efficiency
        md += ["\n## Efficiency (batch 1)\n", markdown_table(efficiency)]
        sheets["all_results"] = results
        write_excel(rep / "results.xlsx", sheets)
        path = rep / "REPORT.md"
        path.write_text("\n".join(md), encoding="utf-8")
        LOGGER.info("report written to %s", rep)
        return path

    def _temporal_sections(self, results: pd.DataFrame, rep: Path, sheets: dict) -> list[str]:
        """Where temporal information should help: per-condition F1 and
        jitter, the output-level Kalman reference, and carried-state results."""
        tag = self.evaluator.primary_tag
        metric = f"lane_f1_{tag}"
        sub = results[(results["split"] == "test") & (results["protocol"] == "tuned")]
        full = sub[sub["input"] == "full"]
        md: list[str] = []
        cond = sorted(c for c in full.columns if c.startswith(f"condition_f1_{tag}_") and full[c].notna().any())
        if cond:
            agg = aggregate(full, cond + ["jitter_px"])
            agg.to_csv(rep / "test_conditions.csv", index=False)
            sheets["test_conditions"] = agg
            names = [c.removeprefix(f"condition_f1_{tag}_") for c in cond]
            frames = {n: int(full[f"condition_frames_{n}"].max()) for n in names if f"condition_frames_{n}" in full}
            md += ["\n## Where temporal information should help (held-out test, tuned)\n",
                   f"Lane F1 per scene condition (frames per condition: {frames}); a condition covers whole "
                   "scenes, so with three held-out scenes these are scene-level, not frame-level, subsets. "
                   "`jitter_px` = mean frame-to-frame change of the predicted lane x (original pixels): lower is "
                   "steadier (it also includes real lane motion).\n",
                   markdown_table(summary_table(agg, cond + ["jitter_px"], self.labels))]
        kf = sub[sub["input"] == "kalman"]
        if not kf.empty:
            combined = pd.concat([full, kf.assign(variant=kf["variant"] + "+kf")], ignore_index=True)
            labels = {**self.labels, **{f"{v}+kf": f"{self.labels.get(v, v)} + Kalman" for v in kf["variant"].unique()}}
            order = [n for v in self.variants for n in (v, f"{v}+kf") if n in set(combined["variant"])]
            agg = aggregate(combined, [metric, "jitter_px"])
            agg["variant"] = pd.Categorical(agg["variant"], order, ordered=True)
            agg = agg.sort_values(["variant", "metric"])
            agg["variant"] = agg["variant"].astype(str)
            agg.to_csv(rep / "kalman_reference.csv", index=False)
            sheets["kalman_reference"] = agg
            kf_pairs = []
            present = set(combined["variant"])
            for v in self.variants:
                spec = resolve_spec(v, self.cfg)
                if spec.temporal and f"{spec.reference}+kf" in present:
                    kf_pairs.append((v, f"{spec.reference}+kf"))
                elif not spec.temporal and f"{v}+kf" in present:
                    kf_pairs.append((f"{v}+kf", v))
            paired = paired_comparisons(combined, kf_pairs, {metric: True, "jitter_px": False})
            paired.to_csv(rep / "paired_tests_kalman.csv", index=False)
            sheets["paired_kalman"] = paired
            md += ["\n## Output-level Kalman tracker (the cheap temporal reference)\n",
                   "Every model's predictions filtered by a causal constant-velocity Kalman filter per lane point, "
                   "parameters tuned on validation. A learned temporal model earns its cost only if it beats its "
                   "single-frame baseline + this tracker (pairs `temporal vs baseline+kf` below).\n",
                   markdown_table(summary_table(agg, [metric, "jitter_px"], labels)),
                   markdown_table(paired.round(4)) if not paired.empty else "_(no pairs)_\n"]
        carry = sub[sub["input"] == "carry"]
        if not carry.empty:
            win = full.set_index(["variant", "seed"])[metric]
            car = carry.set_index(["variant", "seed"])[metric]
            delta = (car - win.reindex(car.index)).groupby(level=0).agg(["mean", "std", "count"]).reset_index()
            delta.columns = ["variant", f"mean_gain_carry_vs_window_{metric}", "std", "n_seeds"]
            delta.to_csv(rep / "carry_state.csv", index=False)
            sheets["carry_state"] = delta
            md += ["\n## Recurrent models with a carried state (held-out test)\n",
                   "`window` (main result) runs the ConvGRU from zero over the training clip; `carry` keeps one "
                   "state per stream across the whole scene (longer memory, one state tensor). Difference in "
                   "lane F1, carry minus window:\n", markdown_table(delta)]
        return md

    def _ablation_table(self, results: pd.DataFrame) -> pd.DataFrame:
        metric = f"lane_f1_{self.evaluator.primary_tag}"
        sub = results[(results["split"] == "test") & (results["protocol"] == "tuned")]
        full = sub[sub["input"] == "full"].set_index(["variant", "seed"])[metric]
        static = sub[sub["input"] == "static_history"].set_index(["variant", "seed"])[metric]
        if static.empty:
            return pd.DataFrame()
        delta = (full.reindex(static.index) - static).groupby(level=0).agg(["mean", "std", "count"])
        return delta.rename(columns={"mean": f"mean_gain_from_history_{metric}", "std": "std",
                                     "count": "n_seeds"}).reset_index()

    # =================================================================== run

    def run(self) -> Path:
        cfg = self.cfg
        if cfg.requires_confirmation and not self.confirmed:
            raise FullRunNotConfirmed(
                f"'{cfg.name}' is marked requires_confirmation (a long, full experiment: "
                f"{len(self.variants)} variants x {len(self.seeds)} seeds x up to {cfg.train.epochs} epochs"
                f"{', HPO ' + str(cfg.hpo.n_trials) + ' trials/variant' if cfg.hpo.enabled else ''}). "
                f"Re-run with --confirm to start it.")
        save_config(cfg, self.out / "config_resolved.yaml")
        write_json(self.out / "environment.json", {**environment_info(), "config_hash": cfg.config_hash(),
                                                  "version": __version__, "variants": self.variants,
                                                  "seeds": self.seeds, "device": self.device})
        if len(self.seeds) < 6:
            LOGGER.warning("%d seed(s): paired Wilcoxon tests cannot reach p < 0.05 with fewer than 6 seeds",
                           len(self.seeds))
        self.prepare_data()
        self.check_labels()
        self.run_hpo()
        all_results = []
        for i, seed in enumerate(self.seeds):
            LOGGER.info("=========== seed %d (%d/%d) ===========", seed, i + 1, len(self.seeds))
            all_results.append(self.run_seed(seed, visualize=i < cfg.evaluation.visualize_seeds))
            pd.concat(all_results, ignore_index=True).to_csv(self.out / "all_results.csv", index=False)
        results = pd.concat(all_results, ignore_index=True)
        return self.report(results, self.measure_efficiency())

    def _free(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
