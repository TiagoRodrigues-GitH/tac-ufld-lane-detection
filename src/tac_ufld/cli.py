"""Command-line interface.

Experiments (ELAS is the default dataset)::

    python -m tac_ufld run --config configs/elas_smoke.yaml             # plumbing check, ~10 min
    python -m tac_ufld run --config configs/elas_pilot.yaml             # GPU pilot, ~1-2 h
    python -m tac_ufld run --config configs/elas.yaml --confirm         # full experiment (days)
    python -m tac_ufld run --dataset culane --confirm                   # registry config of a dataset
    python -m tac_ufld run --all-datasets --smoke                       # every ENABLED dataset
    python -m tac_ufld check-data --config configs/elas.yaml

Datasets::

    python -m tac_ufld datasets [--check]
    python -m tac_ufld validate-dataset --dataset tusimple [--root D:/TuSimple]

Everything else (ablation, sanity, streaming, export, benchmark, UI, site,
package, doctor): ``python -m tac_ufld <command> --help``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from tac_ufld.config import DATA_ROOT_ENV, PROJECT_ROOT, ConfigError, ExperimentConfig, load_config

LOGGER = logging.getLogger("tac_ufld")
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "elas.yaml"


# ---------------------------------------------------------------- arguments


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="experiment YAML (default: the dataset's registry config, else configs/elas.yaml)")
    p.add_argument("--dataset", help="dataset from the registry (must be enabled)")
    p.add_argument("--registry", help="dataset registry YAML (default: configs/datasets.yaml)")
    p.add_argument("--data-root", help="override the dataset root")
    p.add_argument("--name", help="override experiment name (output folder)")
    p.add_argument("--device", help="cpu | cuda | auto")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tac_ufld", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="experiment: HPO, training, evaluation, report")
    _common(run)
    run.add_argument("--all-datasets", action="store_true", help="run every enabled dataset of the registry")
    run.add_argument("--smoke", action="store_true", help="with --all-datasets: use each dataset's smoke config")
    run.add_argument("--seeds", type=int, nargs="+")
    run.add_argument("--variants", nargs="+")
    run.add_argument("--epochs", type=int, help="override train.epochs")
    run.add_argument("--no-hpo", action="store_true", help="skip Optuna, use config defaults")
    run.add_argument("--resume", action="store_true", help="continue an interrupted run in the same output folder")
    run.add_argument("--confirm", action="store_true", help="required for configs marked requires_confirmation")

    check = sub.add_parser("check-data", help="parse data, splits, leakage + label-geometry checks, overlays")
    _common(check)

    ds = sub.add_parser("datasets", help="list the dataset registry (only enabled datasets are scanned)")
    ds.add_argument("--registry")
    ds.add_argument("--check", action="store_true", help="resolve roots and parse every enabled dataset")

    val = sub.add_parser("validate-dataset", help="real-data check of an adapter: counts, ranges, slots, overlays")
    val.add_argument("--dataset", required=True)
    val.add_argument("--root", help="dataset root (allowed even if the dataset is disabled in the registry)")
    val.add_argument("--registry")
    val.add_argument("--config", help="experiment YAML for image size / anchors (default: registry config)")
    val.add_argument("--overlays", type=int, default=24, help="number of overlay images to write")
    val.add_argument("--out", help="output folder (default: results/validate_<dataset>)")

    from tac_ufld import commands  # noqa: E402  (extra sub-commands live in their own module)

    commands.register(sub)
    return p


# ------------------------------------------------------------ config logic


def _registry(path: str | None):
    from tac_ufld.data.registry import DEFAULT_REGISTRY, load_registry

    if path is None and not DEFAULT_REGISTRY.exists():
        return None
    return load_registry(path)


def resolve_config(args, config_path: str | Path | None = None, dataset: str | None = None) -> ExperimentConfig:
    """Config + registry rules -> a validated config whose data root exists.

    * the dataset (``--dataset`` or the config's ``data.dataset``) must be
      enabled in the registry, otherwise the run is rejected;
    * a config for a different dataset than ``--dataset`` is rejected;
    * data root precedence: ``--data-root`` > ``TAC_UFLD_DATA_ROOT`` (legacy,
      ELAS) > the registry root (``${VAR}`` expanded, must exist) >
      ``data.root`` of the config (only without a registry).
    """
    from tac_ufld.data.registry import require_enabled

    registry = _registry(getattr(args, "registry", None))
    dataset = dataset or getattr(args, "dataset", None)
    entry = None
    if dataset:
        if registry is None:
            raise ConfigError("--dataset needs the dataset registry (configs/datasets.yaml)")
        entry = require_enabled(registry, dataset)
    path = Path(config_path or getattr(args, "config", None) or (entry.config_path() if entry else DEFAULT_CONFIG))
    overrides = {}
    for attr, key in (("name", "name"), ("device", "device"), ("epochs", "train.epochs")):
        if getattr(args, attr, None):
            overrides[key] = getattr(args, attr)
    if getattr(args, "no_hpo", False):
        overrides["hpo.enabled"] = False
    cfg = load_config(path, overrides)
    name = cfg.data.dataset.lower()
    if entry is not None and entry.name != name:
        raise ConfigError(f"--dataset {entry.name} but {path.name} is a '{name}' config")
    if registry is not None and name in registry:
        entry = entry or require_enabled(registry, name)
    if getattr(args, "data_root", None):
        cfg.data.root = args.data_root
    elif entry is not None and not os.environ.get(DATA_ROOT_ENV):
        cfg.data.root = str(entry.resolve_root())
    if not cfg.data_root().is_dir():
        raise ConfigError(f"data root for '{name}' does not exist: {cfg.data_root()}")
    return cfg


# ---------------------------------------------------------------- commands


def cmd_run(args) -> int:
    from tac_ufld.experiment import ExperimentRunner, FullRunNotConfirmed

    if args.all_datasets:
        from tac_ufld.data.registry import select_datasets

        if args.config or args.dataset or args.data_root:
            raise ConfigError("--all-datasets takes each dataset's config and root from the registry")
        if os.environ.get(DATA_ROOT_ENV):
            raise ConfigError(f"unset {DATA_ROOT_ENV} for multi-dataset runs (it would apply to every dataset)")
        entries = select_datasets(_registry(args.registry), None)  # fails early on any missing root
        configs = [resolve_config(args, e.config_path(smoke=args.smoke), e.name) for e in entries]
    else:
        configs = [resolve_config(args)]
    for cfg in configs:
        runner = ExperimentRunner(cfg, variants=args.variants, seeds=args.seeds, resume=args.resume,
                                  confirmed=args.confirm)
        try:
            report = runner.run()
        except FullRunNotConfirmed as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        LOGGER.info("done: %s", report)
    return 0


def cmd_check_data(args) -> int:
    from tac_ufld.experiment import ExperimentRunner

    cfg = resolve_config(args)
    runner = ExperimentRunner(cfg)
    splits = runner.prepare_data()
    geometry = runner.check_labels(per_sequence=3)
    print(splits.summary().pivot_table(index="sequence", columns="split", values="frames", fill_value=0))
    if not geometry.empty:
        print(geometry.to_string(index=False))
    print(f"\nOutputs: {runner.out / 'data'}")
    return 0


def cmd_datasets(args) -> int:
    from tac_ufld.data.registry import DatasetRootError, load_registry

    registry = load_registry(args.registry)
    print(f"{'dataset':<10} {'enabled':<8} {'root status':<60} config")
    for entry in registry.values():
        if not entry.enabled:
            status = "disabled (not scanned)"
        else:
            try:
                status = f"ok: {entry.resolve_root()}"
            except DatasetRootError as exc:
                status = f"ERROR: {exc}"
        print(f"{entry.name:<10} {str(entry.enabled):<8} {status:<60} {entry.config or '-'}")
    if args.check:
        from tac_ufld.data import build_adapter

        for entry in registry.values():
            if not entry.enabled:
                continue
            cfg = load_config(entry.config_path())
            cfg.data.root = str(entry.resolve_root())
            adapter = build_adapter(cfg)
            official = adapter.official_splits()
            counts = ({k: len(v) for k, v in official.items()} if official
                      else {s: len(r) for s, r in adapter.load_all().items()})
            print(f"{entry.name}: {counts}")
    return 0


def cmd_validate_dataset(args) -> int:
    from tac_ufld.data.validation import validate_dataset

    registry = _registry(args.registry)
    name = args.dataset.lower()
    if registry is None or name not in registry:
        raise ConfigError(f"dataset '{args.dataset}' is not in the registry")
    entry = registry[name]
    if args.root:
        root = Path(args.root)
    else:
        if not entry.enabled:
            raise ConfigError(f"dataset '{name}' is disabled; enable it in the registry or pass --root")
        root = entry.resolve_root()
    cfg = load_config(args.config or entry.config_path())
    cfg.data.root = str(root)
    out = Path(args.out) if args.out else PROJECT_ROOT / "results" / f"validate_{name}"
    report = validate_dataset(cfg, out, n_overlays=args.overlays)
    print(report["summary"])
    print(f"\nFull report: {out / 'validation_report.json'}  overlays: {out / 'overlays'}")
    return 0 if report["ok"] else 1


COMMANDS = {"run": cmd_run, "check-data": cmd_check_data, "datasets": cmd_datasets,
            "validate-dataset": cmd_validate_dataset}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from tac_ufld import commands

    handler = COMMANDS.get(args.command) or commands.HANDLERS[args.command]
    try:
        return handler(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
