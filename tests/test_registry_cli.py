"""Dataset registry rules, CLI dataset selection and the full-run guard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tac_ufld import cli
from tac_ufld.config import PROJECT_ROOT, ConfigError, expand_env, load_config
from tac_ufld.data.registry import DatasetRootError, load_registry, select_datasets
from tests.dataset_fixtures import make_tusimple


def _write_registry(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_shipped_registry_has_only_elas_enabled():
    reg = load_registry()
    assert set(reg) == {"elas", "culane", "tusimple", "openlane"}
    assert [e.name for e in reg.values() if e.enabled] == ["elas"]


def test_env_expansion(monkeypatch):
    monkeypatch.setenv("TAC_TEST_ROOT", "/data/x")
    assert expand_env("${TAC_TEST_ROOT}/a") == "/data/x/a"
    monkeypatch.delenv("TAC_TEST_ROOT")
    assert expand_env("${TAC_TEST_ROOT:-fallback}") == "fallback"
    with pytest.raises(ConfigError, match="TAC_TEST_ROOT"):
        expand_env("${TAC_TEST_ROOT}")


def test_disabled_datasets_are_never_resolved(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE_ROOT", raising=False)
    reg = load_registry(_write_registry(tmp_path / "r.yaml", f"""
datasets:
  elas: {{enabled: true, root: "{tmp_path.as_posix()}"}}
  culane: {{enabled: false, root: "${{NOPE_ROOT}}"}}
"""))
    assert [e.name for e in select_datasets(reg)] == ["elas"]  # culane's unset variable is never read
    with pytest.raises(ConfigError, match="disabled"):
        select_datasets(reg, ["culane"])


def test_enabled_dataset_with_missing_root_is_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE_ROOT", raising=False)
    reg = load_registry(_write_registry(tmp_path / "r.yaml", """
datasets:
  culane: {enabled: true, root: "${NOPE_ROOT}"}
  tusimple: {enabled: true, root: "/definitely/not/here"}
"""))
    with pytest.raises(DatasetRootError, match="NOPE_ROOT"):
        select_datasets(reg, ["culane"])
    with pytest.raises(DatasetRootError, match="does not exist"):
        select_datasets(reg, ["tusimple"])


def test_invalid_registries_are_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown dataset"):
        load_registry(_write_registry(tmp_path / "a.yaml", "datasets:\n  kitti: {enabled: true}\n"))
    with pytest.raises(ConfigError, match="unknown key"):
        load_registry(_write_registry(tmp_path / "b.yaml", "datasets:\n  elas: {enabled: true, rooot: x}\n"))
    with pytest.raises(ConfigError, match="true or false"):
        load_registry(_write_registry(tmp_path / "c.yaml", "datasets:\n  elas: {enabled: 'yes'}\n"))
    with pytest.raises(ConfigError):
        load_config(PROJECT_ROOT / "configs" / "elas_smoke.yaml", {"data.dataset": "kitti"})


def test_cli_rejects_disabled_dataset_and_mismatched_config(tmp_path, capsys):
    reg = _write_registry(tmp_path / "r.yaml", f"""
datasets:
  elas: {{enabled: true, root: "{tmp_path.as_posix()}", config: configs/elas_smoke.yaml}}
  tusimple: {{enabled: false, root: "{tmp_path.as_posix()}", config: configs/tusimple.yaml}}
""")
    assert cli.main(["check-data", "--registry", str(reg), "--dataset", "tusimple"]) == 2
    assert "disabled" in capsys.readouterr().err
    assert cli.main(["check-data", "--registry", str(reg), "--config", "configs/tusimple.yaml"]) == 2
    assert cli.main(["check-data", "--registry", str(reg), "--dataset", "elas",
                     "--config", str(PROJECT_ROOT / "configs" / "tusimple.yaml")]) == 2
    assert "is a 'tusimple' config" in capsys.readouterr().err


def test_datasets_command_lists_without_scanning_disabled(capsys):
    assert cli.main(["datasets"]) == 0
    out = capsys.readouterr().out
    assert "disabled (not scanned)" in out and "elas" in out


def test_full_run_requires_confirmation(monkeypatch, capsys):
    """`run` on configs/elas.yaml must refuse to start without --confirm and
    must not train anything."""
    from tac_ufld import experiment

    called = []
    monkeypatch.setattr(experiment.ExperimentRunner, "prepare_data", lambda self: called.append(1))
    assert cli.main(["run", "--config", str(PROJECT_ROOT / "configs" / "elas.yaml"), "--name", "guard_test",
                     "--device", "cpu"]) == 2
    assert "--confirm" in capsys.readouterr().err and not called
    assert load_config(PROJECT_ROOT / "configs" / "elas.yaml").requires_confirmation
    for name in ("elas_smoke.yaml", "elas_pilot.yaml"):
        assert not load_config(PROJECT_ROOT / "configs" / name).requires_confirmation


def test_tusimple_end_to_end_with_carved_validation(tmp_path):
    """Official split + carved validation + native TuSimple metrics, one tiny epoch."""
    from tac_ufld.experiment import ExperimentRunner

    root = make_tusimple(tmp_path / "tusimple", clips_per_drive=8)
    cfg = load_config(PROJECT_ROOT / "configs" / "tusimple.yaml", {
        "name": "tusimple_e2e", "output_dir": str(tmp_path / "results"), "device": "cpu",
        "requires_confirmation": False, "data.root": str(root), "data.img_h": 64, "data.img_w": 96,
        "data.griding_num": 24, "data.num_row_anchors": 8, "data.num_workers": 0,
        "data.split.block_size": 3, "data.split.val_fraction": 0.3, "model.pretrained": False,
        "train.seeds": [1], "train.epochs": 1, "train.batch_size": 4,
        "evaluation.postprocess_grid": {"threshold": [0.5], "min_points": [2], "poly_degree": [0]},
        "evaluation.latency_runs": 2, "evaluation.n_visual_examples": 1, "evaluation.tensorboard": False,
    })
    runner = ExperimentRunner(cfg, variants=["ufld_baseline", "ufld_v02"])
    runner.run()
    results = pd.read_csv(runner.out / "all_results.csv")
    assert {"native_tusimple_accuracy", "native_tusimple_fp", "native_tusimple_fn"} <= set(results.columns)
    manifest = pd.read_csv(runner.out / "data" / "split_manifest.csv")
    assert set(manifest["split"]) == {"train", "val", "test"}
    assert (manifest.loc[manifest["split"] == "test", "sequence"].str.startswith("test/")).all()
    report = (runner.out / "report" / "REPORT.md").read_text(encoding="utf-8")
    assert "Dataset-native metrics" in report and "## Protocol" in report
