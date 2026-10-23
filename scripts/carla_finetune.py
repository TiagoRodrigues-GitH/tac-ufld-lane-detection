"""Fine-tune ELAS-trained lane models on CARLA lane data, then score them before and after (docs/CARLA.md).

For each variant (``configs/carla_finetune.yaml``):

1. the configuration is the checkpoint's own (architecture, preprocessing, row anchors, augmentation), with the
   ``experiment`` overrides of the YAML (epochs, learning rate, output folder) and the CARLA data root;
2. splits are explicit, never carved from a test town: train = the CARLA training towns + every n-th frame of a
   few ELAS training scenes; validation = the held-out CARLA town(s) (early stopping, post-processing tuning);
   ``test`` = the checkpoint's held-out ELAS test scenes (does fine-tuning make the model forget ELAS?);
   ``seen_test`` = the CARLA validation town again, scored with the validation-tuned post-processing (reported as
   ``carla_val``: it is selection data, not a test). The closed-loop towns (Town04, Town06) are refused;
3. ``before``: the original checkpoint, scored on these splits (its post-processing re-tuned on CARLA validation);
4. fine-tuning starts from the original checkpoint itself (``ExperimentRunner.train_variant`` with ``warm_from`` =
   the checkpoint: identical state-dict keys, a strict full load), so a temporal model keeps its trained fusion;
5. ``after``: the fine-tuned checkpoint, scored the same way. ``<variant>_tuned.json`` lands next to the
   checkpoint, where ``tac_ufld.inference.loading`` (and so the closed loop) finds it.

Outputs: ``results/carla_finetune_<variant>/{before,}/seed_<n>/...``, ``.../carla_finetune.csv`` (before and after
rows) and the split manifest. Project venv (torch)::

    python scripts/carla_finetune.py --config configs/carla_finetune.yaml --variants lite_tsm lite_v05
"""
from __future__ import annotations

import argparse
import copy
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac_ufld.config import PROJECT_ROOT, ExperimentConfig, _build, expand_env  # noqa: E402
from tac_ufld.data.base import LaneDatasetAdapter  # noqa: E402
from tac_ufld.data.elas import ElasAdapter  # noqa: E402
from tac_ufld.data.splits import DataSplits  # noqa: E402
from tac_ufld.experiment import ExperimentRunner  # noqa: E402

LOGGER = logging.getLogger("carla_finetune")
TEST_TOWNS = ("Town04", "Town06")


class CombinedAdapter(LaneDatasetAdapter):
    """Sequences from several ELAS-layout roots (CARLA scenes and real ELAS scenes): records and history frames are
    read from the root each sequence comes from."""
    name = "elas"
    num_lanes = 2
    flip_permutation = (1, 0)

    def __init__(self, adapters: list[ElasAdapter]) -> None:
        self.adapters = adapters
        self.protocol = {**adapters[0].protocol,
                         "split": "explicit (scripts/carla_finetune.py): CARLA training towns + ELAS mix / "
                                  "held-out CARLA town / held-out ELAS scenes"}
        self._owner = {seq: a for a in adapters for seq in a.sequences()}
        self.stats: dict = {}

    def sequences(self) -> list[str]:
        return sorted(self._owner)

    def load_sequence(self, sequence: str):
        records = self._owner[sequence].load_sequence(sequence)
        self.stats[sequence] = self._owner[sequence].stats.get(sequence, {})
        return records

    def frame_path(self, sequence: str, frame_id: int):
        owner = self._owner.get(sequence)
        return owner.frame_path(sequence, frame_id) if owner is not None else None


def _path(text: str, key: str) -> Path:
    path = Path(expand_env(text, key))
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def town_scenes(root: Path, towns: list[str]) -> list[str]:
    """Finished CARLA scene folders (``CARLA_<town>_s<seed>_<k>`` with its groundtruth.xml) of ``towns``; an entry
    ``Town05_s9`` selects one collection seed of a town only."""
    for town in towns:
        if town.split("_")[0] in TEST_TOWNS:
            raise SystemExit(f"{town} is a closed-loop test town: it must not be used for fine-tuning")
    prefixes = [f"CARLA_{t}_" if re.search(r"_s\d+$", t) else f"CARLA_{t}_s" for t in towns]
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "groundtruth.xml").exists()
                  and any(p.name.startswith(prefix) for prefix in prefixes))


def build_runner(spec: dict, variant: str, seed: int, evaluate_only: bool = False) -> tuple[ExperimentRunner, Path]:
    paths = {**spec.get("reference_checkpoints", {}), **spec["checkpoints"]}
    checkpoint = _path(paths[variant], f"checkpoints.{variant}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    carla_root = _path(spec["carla"]["root"], "carla.root")
    elas_root = _path(spec["elas"]["root"], "elas.root")
    train_scenes = town_scenes(carla_root, spec["carla"]["train_towns"])
    val_scenes = town_scenes(carla_root, spec["carla"]["val_towns"])
    test_towns = spec["carla"].get("test_towns") or []
    test_scenes = town_scenes(carla_root, test_towns) if test_towns else []
    if not train_scenes or not val_scenes or (test_towns and not test_scenes):
        raise SystemExit(f"no CARLA scenes for the train/val/test towns below {carla_root}")
    shared = set(test_scenes) & set(train_scenes + val_scenes)
    if shared:
        raise SystemExit(f"test scenes also used for training or validation: {sorted(shared)}")

    raw = _merge(payload["config"], spec["experiment"])
    raw["name"] = f"carla_finetune_{variant}"
    raw["description"] = f"{variant} fine-tuned on CARLA ({', '.join(spec['carla']['train_towns'])}), " \
                         f"validated on {', '.join(spec['carla']['val_towns'])}"
    raw["data"] = {**raw["data"], "root": str(carla_root), "scenes": train_scenes + val_scenes + test_scenes}
    cfg = _build(ExperimentConfig, raw)
    cfg.validate()
    elas_test = list(cfg.data.split.test_scenes)
    mix = [s for s in spec["elas"]["mix_scenes"] if s not in elas_test]

    runner = ExperimentRunner(cfg, variants=[variant], seeds=[seed])
    runner.variants = [variant]   # no warm-start source: the checkpoint itself is the starting point
    carla = ElasAdapter(carla_root, scenes=train_scenes + val_scenes + test_scenes)
    elas = ElasAdapter(elas_root, scenes=mix + elas_test, scene_tags=cfg.data.scene_tags)
    runner.adapter = CombinedAdapter([carla, elas])
    every = int(spec["elas"]["frame_every"])
    train = [r for s in train_scenes for r in runner.adapter.load_sequence(s)]
    train += [r for s in mix for r in runner.adapter.load_sequence(s)[::every]]
    val = [r for s in val_scenes for r in runner.adapter.load_sequence(s)]
    carla_test = [r for s in test_scenes for r in runner.adapter.load_sequence(s)]
    # ELAS test: the forgetting check of a training run (already scored there when only evaluating)
    test = [] if evaluate_only else [r for s in elas_test for r in runner.adapter.load_sequence(s)]
    # seen_test = the CARLA test towns; without them it falls back to the validation frames (selection data:
    # reported as carla_val, an optimistic number)
    runner.splits = DataSplits(train=train, val=val, test=test, seen_test=carla_test or val)
    data_dir = runner.out / ("heldout" if evaluate_only else "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    runner.splits.summary().to_csv(data_dir / "split_summary.csv", index=False)
    LOGGER.info("%s: train %d (CARLA %d scenes + ELAS %s every %d), val %d (%s), CARLA test %d (%s), ELAS test %d",
                variant, len(train), len(train_scenes), mix, every, len(val), spec["carla"]["val_towns"],
                len(carla_test), test_towns or "none: validation frames instead", len(test))
    return runner, checkpoint


def _carla_split_name(spec: dict) -> str:
    return "carla_test" if spec["carla"].get("test_towns") else "carla_val"


def evaluate_only(spec: dict, variant: str, seed: int) -> pd.DataFrame:
    """Score the original and the fine-tuned checkpoint on the CARLA test towns (``carla.test_towns``), with
    post-processing tuned on the validation towns; no training. Writes ``heldout/`` and
    ``carla_finetune_heldout.csv``."""
    if not spec["carla"].get("test_towns"):
        raise SystemExit("--evaluate-only needs carla.test_towns in the config (scenes nobody selected on)")
    runner, original = build_runner(spec, variant, seed, evaluate_only=True)
    tuned = runner.out / f"seed_{seed}" / "checkpoints" / f"{variant}.pt"
    stages = [("before", original), ("after", tuned)]
    if variant in spec.get("reference_checkpoints", {}):
        stages = [("reference", original)]          # scored as it is, never fine-tuned
    elif not tuned.exists():
        raise SystemExit(f"no fine-tuned checkpoint at {tuned}: run the fine-tuning first")
    rows = [runner._evaluate_seed(seed, {variant: ckpt}, runner.out / "heldout" / stage / f"seed_{seed}",
                                  visualize=False).assign(stage=stage, checkpoint=str(ckpt))
            for stage, ckpt in stages]
    out = pd.concat(rows, ignore_index=True)
    out["split"] = out["split"].replace({"seen_test": "carla_test"})
    out.to_csv(runner.out / "carla_finetune_heldout.csv", index=False)
    return out


def finetune(spec: dict, variant: str, seed: int, skip_before: bool) -> pd.DataFrame:
    runner, original = build_runner(spec, variant, seed)
    rows = []
    if not skip_before:
        before_dir = runner.out / "before" / f"seed_{seed}"
        rows.append(runner._evaluate_seed(seed, {variant: original}, before_dir, visualize=False)
                    .assign(stage="before", checkpoint=str(original)))
    seed_dir = runner.out / f"seed_{seed}"
    ckpt = seed_dir / "checkpoints" / f"{variant}.pt"
    cfg = runner.cfg
    model, result = runner.train_variant(variant, seed, runner.hyperparams(variant), cfg.train.epochs,
                                         cfg.train.early_stopping_patience, warm_from=original, checkpoint=ckpt)
    result.history.to_csv(seed_dir / f"history_{variant}.csv", index=False)
    ckpt.with_name(ckpt.stem + ".last.pt").unlink(missing_ok=True)
    del model
    runner._free()
    rows.append(runner._evaluate_seed(seed, {variant: ckpt}, seed_dir, visualize=False)
                .assign(stage="after", checkpoint=str(ckpt)))
    out = pd.concat(rows, ignore_index=True)
    out["split"] = out["split"].replace({"seen_test": _carla_split_name(spec), "test": "elas_test"})
    out.to_csv(runner.out / "carla_finetune.csv", index=False)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/carla_finetune.yaml")
    ap.add_argument("--variants", nargs="+", default=None, help="default: every checkpoint in the config")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-before", action="store_true", help="do not score the original checkpoint")
    ap.add_argument("--dry-run", action="store_true", help="build the splits and encode a few samples, no training")
    ap.add_argument("--evaluate-only", action="store_true",
                    help="no training: score the original and fine-tuned checkpoints on carla.test_towns")
    args = ap.parse_args(argv)
    spec = yaml.safe_load(_path(args.config, "--config").read_text(encoding="utf-8"))
    default = list(spec["checkpoints"]) + (list(spec.get("reference_checkpoints", {})) if args.evaluate_only else [])
    for variant in args.variants or default:
        if args.evaluate_only:
            res = evaluate_only(spec, variant, args.seed)
            keep = res[(res["protocol"] == "tuned") & (res["input"] == "full")]
            print(keep[["stage", "variant", "split", "lane_f1_iou50", "anchor_mae_px", "n_frames"]].to_string(
                index=False), flush=True)
            continue
        if args.dry_run:
            runner, _ = build_runner(spec, variant, args.seed)
            ds = runner.dataset("train", runner._frames(variant), augment=True)
            for i in (0, len(ds) // 2, len(ds) - 1):
                sample = ds[i]
                print(variant, i, {k: tuple(v.shape) for k, v in sample.items() if hasattr(v, "shape")})
            runner.check_labels(per_sequence=1, max_sequences=8)   # out/data/label_check: targets drawn
            continue
        res = finetune(spec, variant, args.seed, args.skip_before)
        keep = res[(res["protocol"] == "tuned") & (res["input"] == "full")]
        print(keep[["stage", "variant", "split", "lane_f1_iou50", "anchor_mae_px", "n_frames"]].to_string(index=False),
              flush=True)


if __name__ == "__main__":
    main()
