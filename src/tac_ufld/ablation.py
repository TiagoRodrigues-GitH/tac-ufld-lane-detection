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

Spec options added in v0.4:

* ``common``: overrides applied to EVERY arm, the reference included (e.g. a
  larger purge gap shared by all arms of a history ablation, so every arm
  uses the same split);
* ``allow``: guarded keys a spec may vary on purpose (``data.num_frames``,
  ``data.temporal_step`` for the history ablation, ``data.scenes`` to add
  training scenes). The held-out test scenes can never change, so the
  test-set comparison stays like for like;
* ``grid``: arms generated as the product of value lists (added to ``arms``);
* ``reuse_single_frame``: arms that differ from the reference only in
  ``data.num_frames`` / ``data.temporal_step`` reuse the reference arm's
  single-frame checkpoints (their training data is identical), so the
  baseline is trained once and every temporal arm starts from the same
  weights.
"""

from __future__ import annotations

import itertools
import logging
import shutil
from dataclasses import dataclass, field
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


ALLOWABLE = ("data.num_frames", "data.temporal_step", "data.scenes")
GRID_ABBREV = {"data.temporal_step": "s", "data.num_frames": "t"}
REUSE_KEYS = ("data.num_frames", "data.temporal_step")


@dataclass
class AblationSpec:
    name: str
    base_config: str
    reference_arm: str
    arms: dict[str, dict] = field(default_factory=dict)
    variants: list[str] | None = None
    seeds: list[int] | None = None
    hpo: bool = False
    description: str = ""
    common: dict | None = None
    allow: list[str] = field(default_factory=list)
    grid: dict[str, list] | None = None
    reuse_single_frame: bool = False


def grid_arms(grid: dict[str, list]) -> dict[str, dict]:
    """``{"data.temporal_step": [2, 5], "data.num_frames": [3]}`` -> arms ``s2_t3``, ``s5_t3``."""
    keys = list(grid)
    arms = {}
    for values in itertools.product(*(grid[k] for k in keys)):
        name = "_".join(f"{GRID_ABBREV.get(k, k.rsplit('.', 1)[-1])}{v}" for k, v in zip(keys, values))
        arms[name] = dict(zip(keys, values))
    return arms


def load_spec(path: str | Path) -> AblationSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = set(raw) - set(AblationSpec.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"unknown ablation key(s) {sorted(unknown)}")
    spec = AblationSpec(**raw)
    bad_allow = [k for k in spec.allow if k not in ALLOWABLE]
    if bad_allow:
        raise ConfigError(f"allow: {bad_allow} cannot be varied (allowable: {ALLOWABLE})")
    if spec.grid:
        generated = grid_arms(spec.grid)
        clash = set(generated) & set(spec.arms)
        if clash:
            raise ConfigError(f"grid arms {sorted(clash)} are also listed under arms")
        spec.arms = {**spec.arms, **generated}
    if spec.reference_arm not in spec.arms:
        raise ConfigError(f"reference_arm '{spec.reference_arm}' is not one of the arms {list(spec.arms)}")
    for key in spec.common or {}:
        if key.startswith(("name", "output_dir", "model.variants", "train.seeds")):
            raise ConfigError(f"common override '{key}' is set by the spec itself (variants/seeds) or the runner")
    for arm, overrides in spec.arms.items():
        for key in overrides or {}:
            if key.startswith(FORBIDDEN_PREFIXES) and key not in spec.allow:
                raise ConfigError(f"arm '{arm}' overrides '{key}', which would make the arms incomparable "
                                  f"(list it under 'allow' only if varying it is the point of the ablation)")
    return spec


def arm_configs(spec: AblationSpec, output_dir: str | None = None) -> dict[str, ExperimentConfig]:
    base_path = Path(spec.base_config)
    base_path = base_path if base_path.is_absolute() else PROJECT_ROOT / base_path
    common = {"hpo.enabled": spec.hpo, **(spec.common or {})}
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
            if f"data.{attr}" in spec.allow:
                continue
            if getattr(cfg.data, attr) != getattr(ref.data, attr):
                raise ConfigError(f"arm '{arm}' changes data.{attr}")
        if cfg.data.split.test_scenes != ref.data.split.test_scenes:
            raise ConfigError(f"arm '{arm}' changes the held-out test scenes")
        if not set(ref.data.split.test_scenes) <= set(cfg.data.scenes):
            raise ConfigError(f"arm '{arm}' drops a held-out test scene from data.scenes")
        if (cfg.train.seeds, cfg.train.epochs, cfg.model.variants) != (ref.train.seeds, ref.train.epochs,
                                                                       ref.model.variants):
            raise ConfigError(f"arm '{arm}' changes seeds, epochs or variants")
    return configs


def _only_history_differs(cfg: ExperimentConfig, ref: ExperimentConfig) -> bool:
    """True when two arms differ only in the temporal context (and their names)."""
    a, b = cfg.to_dict(), ref.to_dict()
    for d in (a, b):
        d.pop("name")
        for key in REUSE_KEYS:
            d["data"].pop(key.split(".")[1])
    return a == b


def reuse_single_frame(configs: dict[str, ExperimentConfig], reference: str, arm: str) -> list[str]:
    """Copy the reference arm's finished single-frame checkpoints (and their
    training histories) into ``arm``, which then skips training them."""
    from tac_ufld.models.registry import resolve_spec

    ref, cfg = configs[reference], configs[arm]
    if arm == reference or not _only_history_differs(cfg, ref):
        return []
    copied = []
    for seed in cfg.train.seeds:
        for variant in cfg.model.variants:
            if resolve_spec(variant, cfg).temporal:
                continue
            src_dir, dst_dir = ref.output_root() / f"seed_{seed}", cfg.output_root() / f"seed_{seed}"
            ckpt, hist = src_dir / "checkpoints" / f"{variant}.pt", src_dir / f"history_{variant}.csv"
            if not (ckpt.exists() and hist.exists()):
                continue
            (dst_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            for src, dst in ((ckpt, dst_dir / "checkpoints" / ckpt.name), (hist, dst_dir / hist.name)):
                if not dst.exists():
                    shutil.copyfile(src, dst)
            copied.append(f"{variant} seed {seed}")
    return copied


def validation_table(configs: dict[str, ExperimentConfig], metric: str, reference: str) -> pd.DataFrame:
    """Best validation score and epoch of every arm / variant / seed, read
    from the training histories (what a recipe choice may legitimately use)."""
    rows = []
    for arm, cfg in configs.items():
        for path in sorted(cfg.output_root().glob("seed_*/history_*.csv")):
            hist = pd.read_csv(path)
            col = f"val_{metric}"
            if col not in hist or hist.empty:
                continue
            best = hist.loc[hist[col].idxmax()]
            rows.append({"arm": arm, "variant": path.stem.removeprefix("history_"),
                         "seed": int(path.parent.name.removeprefix("seed_")), "best_val": float(best[col]),
                         "best_epoch": int(best["epoch"]), "epochs_run": int(hist["epoch"].max()),
                         "scenes_changed": sorted(cfg.data.scenes) != sorted(configs[reference].data.scenes)})
    return pd.DataFrame(rows)


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
    validation = validation_table(configs, metric, spec.reference_arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(out_dir / "ablation_validation.csv", index=False)
    results.to_csv(out_dir / "ablation_results.csv", index=False)
    agg.to_csv(out_dir / "ablation_aggregate.csv", index=False)
    paired.to_csv(out_dir / "ablation_paired_tests.csv", index=False)
    arch = {arm: cfg.in_channels for arm, cfg in configs.items() if cfg.in_channels != 3}
    md = [f"# Ablation: {spec.name}\n", spec.description + "\n" if spec.description else "",
          f"Base config `{spec.base_config}`, reference arm `{spec.reference_arm}`, metric `{metric}` on the test "
          "split with validation-tuned post-processing. Arms share split, seeds, epochs, model family and HPO "
          "setting (enforced).\n",
          f"Arms that change the model input (first convolution adapted, separate configurations): {arch or 'none'}.\n",
          "\n## Validation (use this to choose a recipe; the test split is for reporting only)\n",
          "Best validation score per seed (the epoch the checkpoint was selected at). Arms that change the "
          "training scenes also change the validation set, so their validation scores are not comparable.\n",
          markdown_table(validation.groupby(["arm", "variant"], sort=False).agg(
              best_val_mean=("best_val", "mean"), best_val_std=("best_val", "std"),
              best_epochs=("best_epoch", lambda e: ",".join(map(str, e)))).reset_index().round(4))
          if not validation.empty else "_(no histories)_\n",
          "\n## Held-out test, mean over seeds\n", markdown_table(agg[agg["metric"] == metric].round(4)),
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
    order = [spec.reference_arm] + [a for a in configs if a != spec.reference_arm]  # reference first
    for arm in order:
        if only and arm not in only:
            continue
        cfg = configs[arm]
        LOGGER.info("=========== ablation %s: arm %s ===========", spec.name, arm)
        if spec.reuse_single_frame:
            copied = reuse_single_frame(configs, spec.reference_arm, arm)
            if copied:
                LOGGER.info("arm %s reuses the reference arm's single-frame checkpoints: %s", arm, copied)
        ExperimentRunner(cfg, resume=True if spec.reuse_single_frame else resume, confirmed=True).run()
    ref = configs[spec.reference_arm]
    return summarize(spec, configs, ref.output_root().parent / spec.name)
