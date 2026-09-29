"""End-to-end run on synthetic ELAS scenes (CPU, tiny images, 1 epoch)."""

from __future__ import annotations

import pandas as pd

from tac_ufld.experiment import ExperimentRunner


def test_full_pipeline_on_synthetic_data(tiny_config):
    tiny_config.train.seeds = [1]
    tiny_config.train.epochs = 1
    tiny_config.hpo.n_trials = 1
    tiny_config.hpo.variants = ["ufld_v02"]
    tiny_config.evaluation.postprocess_grid = {"threshold": [0.3, 0.5], "min_points": [2], "poly_degree": [0, 1]}
    runner = ExperimentRunner(tiny_config, variants=["ufld_baseline", "ufld_v02", "ufld_v04",
                                                     "lite_baseline", "lite_v05"])
    report = runner.run()
    out = runner.out
    assert report.exists()
    results = pd.read_csv(out / "all_results.csv")
    assert set(results["variant"]) == {"ufld_baseline", "ufld_v02", "ufld_v04", "lite_baseline", "lite_v05"}
    assert set(results["split"]) == {"test", "seen_test"}
    # held-out scene only in test
    manifest = pd.read_csv(out / "data" / "split_manifest.csv")
    assert set(manifest.loc[manifest["split"] == "test", "sequence"]) == {"SYN_C"}
    assert "SYN_C" not in set(manifest.loc[manifest["split"] != "test", "sequence"])
    # label geometry check on synthetic straight lanes is ~0 px
    geometry = pd.read_csv(out / "data" / "label_geometry_check.csv")
    assert (geometry["median_line_residual_px"] < 0.5).all()
    assert (out / "hpo" / "best_params.json").exists()
    assert (out / "report" / "results.xlsx").exists()
    assert any((out / "seed_1" / "visuals").rglob("*.png"))
    static = results[results["input"] == "static_history"]
    assert set(static["variant"]) == {"ufld_v02", "ufld_v04", "lite_v05"}
