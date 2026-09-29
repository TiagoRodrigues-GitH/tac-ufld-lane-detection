from __future__ import annotations

import pandas as pd
import pytest

from tac_ufld.config import PROJECT_ROOT, ConfigError, load_config
from tac_ufld.evaluation.stats import aggregate, holm, min_attainable_p, paired_comparisons


@pytest.mark.parametrize("name", ["elas.yaml", "elas_smoke.yaml", "culane.yaml"])
def test_shipped_configs_validate(name):
    cfg = load_config(PROJECT_ROOT / "configs" / name)
    assert cfg.config_hash()


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError):
        load_config(PROJECT_ROOT / "configs" / "elas_smoke.yaml", {"train.epochz": 3})


def test_full_config_has_enough_seeds_for_significance():
    cfg = load_config(PROJECT_ROOT / "configs" / "elas.yaml")
    assert min_attainable_p(len(cfg.train.seeds)) < 0.05
    assert min_attainable_p(4) == 0.125


def test_holm_adjustment():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_paired_comparisons_and_aggregate():
    rows = []
    for seed in range(1, 7):
        rows.append({"variant": "base", "seed": seed, "f1": 0.50 + 0.01 * seed})
        rows.append({"variant": "temp", "seed": seed, "f1": 0.55 + 0.01 * seed + 0.001 * seed})
    df = pd.DataFrame(rows)
    paired = paired_comparisons(df, [("temp", "base")], {"f1": True})
    assert paired.loc[0, "wilcoxon_p"] == pytest.approx(0.03125)
    assert bool(paired.loc[0, "significant"]) and not bool(paired.loc[0, "underpowered"])
    agg = aggregate(df, ["f1"])
    assert set(agg["variant"]) == {"base", "temp"} and (agg["n"] == 6).all()
