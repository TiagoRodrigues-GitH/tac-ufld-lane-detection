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


def test_v04_pipeline_with_new_models_controls_and_temporal_references(tiny_config):
    """New fusions, equal-training controls, current-frame degradation, backbone
    lr / freezing, the Kalman reference and carried-state evaluation, end to end."""
    import json

    cfg = tiny_config
    cfg.name = "tiny_v04"
    cfg.train.seeds = [1]
    cfg.train.epochs = 1
    cfg.train.lr_backbone_mult = 0.5
    cfg.train.freeze_backbone_stages = 1
    cfg.hpo.enabled = False
    cfg.data.augmentation.current_frame_prob = 0.5
    cfg.evaluation.postprocess_grid = {"threshold": [0.5], "min_points": [2], "poly_degree": [0]}
    cfg.evaluation.kalman_grid = {"q": [1.0], "alpha": [0.0, 0.5]}
    variants = ["ufld_baseline", "ufld_baseline_ct", "ufld_v06", "ufld_v07", "lite_baseline", "lite_v06"]
    runner = ExperimentRunner(cfg, variants=variants)
    report = runner.run()
    results = pd.read_csv(runner.out / "all_results.csv")
    assert set(results["variant"]) == set(variants)
    kalman = results[results["input"] == "kalman"]
    assert set(kalman["variant"]) == set(variants) and "kf_q" in kalman
    carry = results[results["input"] == "carry"]
    assert set(carry["variant"]) == {"ufld_v07", "lite_v06"} and set(carry["split"]) == {"test"}
    paired = pd.read_csv(runner.out / "report" / "paired_tests_test_tuned.csv")
    pairs = set(zip(paired["variant"], paired["reference"]))
    assert {("ufld_v06", "ufld_baseline"), ("ufld_v06", "ufld_baseline_ct"),
            ("ufld_baseline_ct", "ufld_baseline")} <= pairs
    text = report.read_text(encoding="utf-8")
    for section in ("Output-level Kalman tracker", "carried state", "Where temporal information should help"):
        assert section in text
    tuned = json.loads((runner.out / "seed_1" / "postprocess" / "ufld_v07_tuned.json").read_text(encoding="utf-8"))
    assert tuned["kalman"]["q"] == 1.0
