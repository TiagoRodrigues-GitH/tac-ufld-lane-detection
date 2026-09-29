"""Ablation runner (preprocessing, augmentation, regularisation).

An ablation spec (``configs/ablations/*.yaml``) names a base experiment config
and a set of *arms*, each a small dict of dotted overrides. Fairness is
enforced, not assumed: every arm must keep the base config's split, seeds,
epochs, model family, HPO setting and selection metric, so arms differ only
in what is being ablated. Every arm is a normal experiment (own folder,
resumable); the runner then compares each arm with the reference arm on the
same seeds (paired Wilcoxon + Holm) and writes ``ABLATION.md``.

No arm is declared better unless the paired test says so; with fewer than
6 seeds the comparison is flagged ``underpowered``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from tac_ufld.config import PROJECT_ROOT, ConfigError, ExperimentConfig, load_config
from tac_ufld.evaluation.stats import aggregate, paired_comparisons
from tac_ufld.reporting import markdown_table

LOGGER = logging.getLogger(__name__)

# Overrides that would make arms incomparable.
FORBIDDEN_PREFIXES = ("data.split", "data.dataset", "data.root", "data.scenes", "train.seeds", "train.epochs",
                      "train.early_stopping_patience", "hpo", "model.variants", "evaluation.selection_metric",
                      "evaluation.iou_thresholds", "name", "output_dir", "data.img_h", "data.img_w",
                      "data.num_row_anchors", "data.griding_num", "data.num_frames", "data.temporal_step")


@dataclass
class AblationSpec:
    name: str
    base_config: str
    arms: dict[str, dict]
    reference_arm: str
    variants: list[str] | None = None
    seeds: list[int] | None = None
    hpo: bool = False
    description: str = ""


def load_spec(path: str | Path) -> AblationSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = set(raw) - set(AblationSpec.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"unknown ablation key(s) {sorted(unknown)}")
    spec = AblationSpec(**raw)
    if spec.reference_arm not in spec.arms:
        raise ConfigError(f"reference_arm '{spec.reference_arm}' is not one of the arms {list(spec.arms)}")
    for arm, overrides in spec.arms.items():
        for key in overrides or {}:
            if key.startswith(FORBIDDEN_PREFIXES):
                raise ConfigError(f"arm '{arm}' overrides '{key}', which would make the arms incomparable")
    return spec


def arm_configs(spec: AblationSpec, output_dir: str | None = None) -> dict[str, ExperimentConfig]:
    base_path = Path(spec.base_config)
    base_path = base_path if base_path.is_absolute() else PROJECT_ROOT / base_path
    common = {"hpo.enabled": spec.hpo}
    if spec.variants:
        common["model.variants"] = spec.variants
    if spec.seeds:
        common["train.seeds"] = spec.seeds
    if output_dir:
        common["output_dir"] = output_dir
    configs = {}
    for arm, overrides in spec.arms.items():
        cfg = load_config(base_path, {**common, **(overrides or {}), "name": f"{spec.name}__{arm}"})
        configs[arm] = cfg
    ref = configs[spec.reference_arm]
    for arm, cfg in configs.items():  # the fairness contract, checked on the resolved configs
        for attr in ("split", "num_frames", "temporal_step", "img_h", "img_w"):
            if getattr(cfg.data, attr) != getattr(ref.data, attr):
                raise ConfigError(f"arm '{arm}' changes data.{attr}")
        if (cfg.train.seeds, cfg.train.epochs, cfg.model.variants) != (ref.train.seeds, ref.train.epochs,
                                                                       ref.model.variants):
            raise ConfigError(f"arm '{arm}' changes seeds, epochs or variants")
    return configs


def summarize(spec: AblationSpec, configs: dict[str, ExperimentConfig], out_dir: Path) -> Path:
    frames = []
    for arm, cfg in configs.items():
        path = cfg.output_root() / "all_results.csv"
        if not path.exists():
            LOGGER.warning("arm %s has no results yet (%s)", arm, path)
            continue
        df = pd.read_csv(path)
        df = df[(df["split"] == "test") & (df["protocol"] == "tuned") & (df["input"] == "full")].copy()
        df["arm"] = arm
        df["in_channels"] = cfg.in_channels
        frames.append(df)
    if not frames:
        raise RuntimeError("no arm has results")
    results = pd.concat(frames, ignore_index=True)
    metric = configs[spec.reference_arm].evaluation.selection_metric
    results["arm_variant"] = results["arm"] + "/" + results["variant"]
    agg = aggregate(results.rename(columns={"arm_variant": "variant", "variant": "model"}),
                    [metric, "lane_precision_iou50", "lane_recall_iou50", "pixel_f1", "jitter_px"])
    pairs = [(f"{arm}/{v}", f"{spec.reference_arm}/{v}") for arm in configs if arm != spec.reference_arm
             for v in results["variant"].unique()]
    paired = paired_comparisons(results.rename(columns={"arm_variant": "variant", "variant": "model"}), pairs,
                                {metric: True, "jitter_px": False})
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "ablation_results.csv", index=False)
    agg.to_csv(out_dir / "ablation_aggregate.csv", index=False)
    paired.to_csv(out_dir / "ablation_paired_tests.csv", index=False)
    arch = {arm: cfg.in_channels for arm, cfg in configs.items() if cfg.in_channels != 3}
    md = [f"# Ablation: {spec.name}\n", spec.description + "\n" if spec.description else "",
          f"Base config `{spec.base_config}`, reference arm `{spec.reference_arm}`, metric `{metric}` on the test "
          "split with validation-tuned post-processing. Arms share split, seeds, epochs, model family and HPO "
          "setting (enforced).\n",
          f"Arms that change the model input (first convolution adapted, separate configurations): {arch or 'none'}.\n",
          "\n## Mean over seeds\n", markdown_table(agg[agg["metric"] == metric].round(4)),
          "\n## Paired comparison with the reference arm (same seeds)\n",
          "No arm is better or worse unless `significant` is True; `underpowered` means the number of seeds cannot "
          "reach p < 0.05.\n", markdown_table(paired.round(4)) if not paired.empty else "_(no pairs)_\n"]
    path = out_dir / "ABLATION.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def run_ablation(spec_path: str | Path, resume: bool = True, only: list[str] | None = None,
                 output_dir: str | None = None) -> Path:
    from tac_ufld.experiment import ExperimentRunner

    spec = load_spec(spec_path)
    configs = arm_configs(spec, output_dir)
    for arm, cfg in configs.items():
        if only and arm not in only:
            continue
        LOGGER.info("=========== ablation %s: arm %s ===========", spec.name, arm)
        ExperimentRunner(cfg, resume=resume, confirmed=True).run()
    ref = configs[spec.reference_arm]
    return summarize(spec, configs, ref.output_root().parent / spec.name)
