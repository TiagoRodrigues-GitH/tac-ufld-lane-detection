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
from tac_ufld.config import ExperimentConfig, save_config
from tac_ufld.data import build_adapter
from tac_ufld.data.dataset import TemporalLaneDataset
from tac_ufld.data.splits import DataSplits, cap_records, check_no_leakage, split_scenes_and_blocks
from tac_ufld.data.targets import encode_targets, make_row_anchors
from tac_ufld.data.types import FrameRecord
from tac_ufld.evaluation.efficiency import measure_efficiency
from tac_ufld.evaluation.evaluator import Evaluator, iou_tag
from tac_ufld.evaluation.predictor import predict
from tac_ufld.evaluation.stats import aggregate, paired_comparisons
from tac_ufld.losses import Hyperparams
from tac_ufld.models.registry import VARIANTS, build_model, resolve_spec, training_order, warm_start
from tac_ufld.postprocess import PostprocessParams
from tac_ufld.reporting import export_lines_txt, markdown_table, summary_table, write_excel
from tac_ufld.training.hpo import active_space, run_study
from tac_ufld.training.trainer import Trainer, TrainResult, load_checkpoint
from tac_ufld.utils import environment_info, read_json, seed_worker, set_seed, setup_logging, write_json
from tac_ufld.visualization import overlays, plots

LOGGER = logging.getLogger(__name__)
EVAL_SPLITS = ("test", "seen_test")


class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig, variants: list[str] | None = None,
                 seeds: list[int] | None = None, resume: bool = False) -> None:
        self.cfg = cfg
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
        official = self.adapter.official_splits()
        s = cfg.data.split
        if official:
            splits = DataSplits(
                train=cap_records(official.get("train", []), s.max_train_frames, f"{s.split_seed}:cap:train"),
                val=cap_records(official.get("val", []), s.max_val_frames, f"{s.split_seed}:cap:val"),
                test=cap_records(official.get("test", []), s.max_test_frames, f"{s.split_seed}:cap:test"),
            )
            check_no_leakage(splits)
        else:
            splits = split_scenes_and_blocks(self.adapter.load_all(), s, cfg.min_split_gap())
        for name in ("train", "val", "test"):
            if not getattr(splits, name):
                raise RuntimeError(f"split '{name}' is empty - check data.scenes / split.test_scenes")
        data_dir = self.out / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        splits.manifest().to_csv(data_dir / "split_manifest.csv", index=False)
        splits.summary().to_csv(data_dir / "split_summary.csv", index=False)
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

    def check_labels(self, per_sequence: int = 2) -> pd.DataFrame:
        """Label-check overlays + straight-line residual of the 4 ELAS points."""
        d = self.cfg.data
        by_seq: dict[str, list[FrameRecord]] = {}
        for recs in self.splits.as_dict().values():
            for r in recs:
                by_seq.setdefault(r.sequence, []).append(r)
        chosen, rows = [], []
        for seq, recs in sorted(by_seq.items()):
            recs.sort(key=lambda r: r.frame_id)
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
                           lambda_temporal=t.lambda_temporal, lambda_coord=t.lambda_coord)

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
        return model.to(self.device)

    def _frames(self, variant: str) -> int:
        return self.cfg.data.num_frames if resolve_spec(variant, self.cfg).temporal else 1

    def train_variant(self, variant: str, seed: int, hp: Hyperparams, epochs: int, patience: int,
                      warm_from: Path | None, checkpoint: Path | None, train_records=None,
                      trial=None, writer=None, tag: str = "") -> tuple[torch.nn.Module, TrainResult]:
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
        result = trainer.fit(checkpoint, extra_meta={"seed": seed, "config_hash": self.cfg.config_hash(),
                                                     "version": __version__})
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
            if variant in targets and space:
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
                model, _ = self.train_variant(variant, h.seed, self.hyperparams(variant), h.epochs, 0,
                                              references.get(spec.warm_start), ref_path, train_records=subset,
                                              tag=f"hpo reference {variant}")
                references[variant] = ref_path
                del model
                self._free()
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
                checkpoints[variant] = ckpt
                if self.resume and ckpt.exists():
                    LOGGER.info("resume: %s seed %d already trained", variant, seed)
                    continue
                spec = resolve_spec(variant, cfg)
                model, result = self.train_variant(
                    variant, seed, self.hyperparams(variant), cfg.train.epochs, cfg.train.early_stopping_patience,
                    checkpoints.get(spec.warm_start), ckpt, writer=writer)
                result.history.to_csv(seed_dir / f"history_{variant}.csv", index=False)
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
                for input_mode, p in variants_of_input:
                    for protocol, params in protocols.items():
                        if input_mode != "full" and protocol != "tuned":
                            continue
                        res = self.evaluator.evaluate(p, ds.records, params)
                        rows.append({"seed": seed, "variant": variant, "split": split, "protocol": protocol,
                                     "input": input_mode, **res.metrics,
                                     **{f"pp_{k}": v for k, v in params.as_dict().items()}})
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
                model, self._frames(variant), d.img_h, d.img_w, self.device, self.cfg.evaluation.latency_runs)})
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
        sheets, md = {}, [f"# Experiment report: {cfg.name}\n",
                          f"Config hash `{cfg.config_hash()}`, package {__version__}, seeds {self.seeds}, "
                          f"device {self.device}.\n",
                          "Primary metric: lane F1 at IoU 0.5 (CULane protocol, width scaled to image), "
                          "held-out test scenes, validation-tuned post-processing.\n"]
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
                           "the seed count cannot reach p < 0.05.\n",
                           markdown_table(paired.round(4)) if not paired.empty else "_(no pairs)_\n"]
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
