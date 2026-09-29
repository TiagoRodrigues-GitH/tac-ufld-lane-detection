"""Command-line interface.

    python -m tac_ufld check-data --config configs/elas.yaml
    python -m tac_ufld run        --config configs/elas.yaml [--seeds 1 2] [--variants ufld_baseline ufld_v02] [--resume]
    python -m tac_ufld run        --config configs/elas_smoke.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys

from tac_ufld.config import PROJECT_ROOT, load_config


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tac_ufld", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_text in (("run", "full experiment: HPO, training, evaluation, report"),
                            ("check-data", "parse data, build splits, leakage + label-geometry checks, overlays")):
        s = sub.add_parser(name, help=help_text)
        s.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "elas.yaml"))
        s.add_argument("--data-root", help="override data.root")
        s.add_argument("--name", help="override experiment name (output folder)")
        s.add_argument("--device", help="cpu | cuda | auto")
    run = sub.choices["run"]
    run.add_argument("--seeds", type=int, nargs="+")
    run.add_argument("--variants", nargs="+")
    run.add_argument("--epochs", type=int, help="override train.epochs")
    run.add_argument("--no-hpo", action="store_true", help="skip Optuna, use config defaults")
    run.add_argument("--resume", action="store_true", help="reuse finished checkpoints/seeds in the output folder")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    overrides = {}
    if args.data_root:
        overrides["data.root"] = args.data_root
    if args.name:
        overrides["name"] = args.name
    if args.device:
        overrides["device"] = args.device
    if getattr(args, "epochs", None):
        overrides["train.epochs"] = args.epochs
    if getattr(args, "no_hpo", False):
        overrides["hpo.enabled"] = False
    cfg = load_config(args.config, overrides)

    from tac_ufld.experiment import ExperimentRunner

    if args.command == "check-data":
        runner = ExperimentRunner(cfg)
        splits = runner.prepare_data()
        geometry = runner.check_labels(per_sequence=3)
        print(splits.summary().pivot_table(index="sequence", columns="split", values="frames", fill_value=0))
        if not geometry.empty:
            print(geometry.to_string(index=False))
        print(f"\nOutputs: {runner.out / 'data'}")
        return 0
    runner = ExperimentRunner(cfg, variants=args.variants, seeds=args.seeds, resume=args.resume)
    report = runner.run()
    logging.getLogger("tac_ufld").info("done: %s", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
