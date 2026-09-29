"""Optuna hyper-parameter optimisation (audit finding C6 fixed).

* The TPE sampler is seeded (``hpo.sampler_seed``); each study is created
  fresh in memory, so trials from older code or configs can never mix in.
* The objective is the validation selection metric; the test split is never
  touched. Every variant, including the baselines, gets the same budget.
* Search spaces come from the config (``hpo.search_space``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import optuna

from tac_ufld.config import SearchParam
from tac_ufld.losses import Hyperparams
from tac_ufld.utils import write_json

LOGGER = logging.getLogger(__name__)


def active_space(space: dict[str, SearchParam], variant: str, temporal: bool) -> dict[str, SearchParam]:
    """Parameters that apply to ``variant`` (fusion/temporal knobs only for temporal models)."""
    temporal_only = {"lr_fusion", "lambda_temporal"}
    return {name: p for name, p in space.items()
            if p.applies_to(variant) and (temporal or name not in temporal_only)}


def suggest(trial: optuna.Trial, space: dict[str, SearchParam], base: Hyperparams) -> Hyperparams:
    values = {name: trial.suggest_float(name, p.low, p.high, log=p.log) for name, p in space.items()}
    return replace(base, **values)


def run_study(
    variant: str,
    space: dict[str, SearchParam],
    base: Hyperparams,
    objective: Callable[[Hyperparams, optuna.Trial], float],
    n_trials: int,
    sampler_seed: int,
    out_dir: Path,
) -> dict:
    """Run one study; returns ``{"params": {...}, "value": float|None}`` and
    writes trials CSV + best params JSON to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=variant, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=max(1, min(5, n_trials // 3)), n_warmup_steps=2),
    )

    def _objective(trial: optuna.Trial) -> float:
        hp = suggest(trial, space, base)
        score = objective(hp, trial)
        LOGGER.info("[hpo %s] trial %d -> %.4f  %s", variant, trial.number, score, trial.params)
        return score

    study.optimize(_objective, n_trials=n_trials, gc_after_trial=True, catch=(FloatingPointError,))
    study.trials_dataframe().to_csv(out_dir / f"{variant}_trials.csv", index=False)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        LOGGER.warning("[hpo %s] no completed trial; keeping config defaults", variant)
        result = {"params": {}, "value": None}
    else:
        result = {"params": dict(study.best_trial.params), "value": float(study.best_value),
                  "best_trial": study.best_trial.number, "n_trials": len(study.trials)}
    write_json(out_dir / f"{variant}_best.json", result)
    return result
