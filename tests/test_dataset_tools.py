"""The download checker and the per-dataset smoke runner (scripts/)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pandas as pd
import pytest

from tac_ufld.config import PROJECT_ROOT, load_config
from tests.conftest import make_elas_scene
from tests.dataset_fixtures import make_culane, make_openlane, make_tusimple

CHECK = PROJECT_ROOT / "scripts" / "check_dataset.py"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check(dataset: str, root) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CHECK), dataset, str(root)], capture_output=True, text=True)


def test_check_dataset_flags_missing_culane_drivers_and_explains_the_fix(tmp_path):
    out = _check("culane", make_culane(tmp_path / "culane"))
    assert out.returncode == 1 and "NOT READY" in out.stdout
    assert "[MISS]  driver_182_30frame/" in out.stdout and "extract driver_182_30frame.tar.gz" in out.stdout
    assert "[ OK ]  train_gt.txt: image + .lines.txt + seg label" in out.stdout
    assert "published: 88,880" in out.stdout  # counts are compared with the published ones


def test_check_dataset_tusimple_and_openlane_layouts(tmp_path):
    tusimple = _check("tusimple", make_tusimple(tmp_path / "tusimple"))
    assert tusimple.returncode == 1 and "[MISS]  train_set/label_data_0601.json" in tusimple.stdout
    assert "history frames 18, 16: 6 samples, every file present" in tusimple.stdout
    openlane = _check("openlane", make_openlane(tmp_path / "openlane"))
    assert openlane.returncode == 0 and "READY" in openlane.stdout
    assert "every annotated segment has its image folder" in openlane.stdout


def test_check_dataset_finds_data_one_level_down_and_reads_only(tmp_path):
    root = tmp_path / "download"
    make_openlane(root / "OpenLane")
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    out = _check("openlane", root)
    assert out.returncode == 0 and "use " in out.stdout and "as the dataset root" in out.stdout
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


def test_check_dataset_elas(tmp_path):
    for name in ("BR_S02", "VIX_S05", "VV_S03", "SYN_A"):
        make_elas_scene(tmp_path / "elas", name, n_frames=12)
    out = _check("elas", tmp_path / "elas")
    assert out.returncode == 0 and "4 scenes" in out.stdout


def test_smoke_configs_keep_the_full_configs_geometry():
    for name in ("culane", "tusimple", "openlane"):
        full = load_config(PROJECT_ROOT / "configs" / f"{name}.yaml", {"data.root": "."}).data
        smoke = load_config(PROJECT_ROOT / "configs" / f"{name}_smoke.yaml", {"data.root": "."}).data
        for key in ("dataset", "img_h", "img_w", "num_lanes", "num_row_anchors", "row_anchor_range",
                    "griding_num", "num_frames", "temporal_step", "augmentation"):
            assert getattr(smoke, key) == getattr(full, key), (name, key)


@pytest.mark.parametrize("dataset", ["culane", "tusimple", "openlane"])
def test_smoke_runs_every_stage_on_a_synthetic_copy(dataset, monkeypatch, tmp_path):
    """Training -> tuning -> evaluation with the internal AND the native metric,
    at the dataset's real geometry, on CPU."""
    smoke = _script("smoke_datasets")
    monkeypatch.setattr(smoke, "PROJECT_ROOT", tmp_path)  # outputs under tmp_path
    (tmp_path / "configs").mkdir()
    for f in (PROJECT_ROOT / "configs").glob(f"{dataset}_smoke.yaml"):
        (tmp_path / "configs" / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    row = smoke.smoke(dataset, force_mock=True, device="cpu", variants=["ufld_baseline", "ufld_v03"])
    assert row["ok"], row["problems"]
    assert row["data"] == "mock" and row["native metric"].startswith("native_")
    results = pd.read_csv(tmp_path / "results" / "smoke_datasets" / f"{dataset}_mock" / "all_results.csv")
    assert set(results["variant"]) == {"ufld_baseline", "ufld_v03"}
    log = (tmp_path / "results" / "smoke_datasets" / f"{dataset}_mock" / "run.log").read_text(encoding="utf-8")
    assert "[metrics internal]" in log and f"[metrics native {dataset}]" in log
