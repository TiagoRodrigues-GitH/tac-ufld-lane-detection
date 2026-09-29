"""Benchmark simulation, sanity checks, ablation fairness, CLI tools, doctor
and packaging."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from tac_ufld import cli
from tac_ufld.config import PROJECT_ROOT, ConfigError, load_config
from tac_ufld.evaluation.hardware import BudgetProfile, load_profile, simulate
from tests.test_streaming import _cfg, _checkpoint


# ----------------------------------------------------------------- simulation


def test_simulation_meets_budget_when_fast_enough():
    p = BudgetProfile(target_fps=30, max_latency_ms=50, frames=300, seed=1)
    r = simulate([10.0] * 50, p, num_frames_model=3, temporal_step=2)
    assert r["processed"] == 300 and r["skipped_busy"] == 0 and r["meets_fps"] and r["meets_latency_p95"]
    assert r["history_fallback_rate"] == 0.0


def test_simulation_overload_skips_frames_and_breaks_temporal_context():
    p = BudgetProfile(target_fps=30, max_latency_ms=50, frames=300, policy="latest", seed=1)
    r = simulate([80.0] * 50, p, num_frames_model=3, temporal_step=2)
    assert r["skipped_busy"] > 0 and not r["meets_fps"] and r["history_fallback_rate"] > 0.5
    fifo = simulate([80.0] * 50, BudgetProfile(target_fps=30, frames=100, policy="fifo"), 1, 1)
    assert fifo["skipped_busy"] == 0 and fifo["latency_p95_ms"] > 1000  # queue grows without bound


def test_simulation_drops_slowdown_and_determinism():
    p = BudgetProfile(target_fps=30, frames=600, drop_rate=0.1, seed=3)
    a, b = simulate([5.0, 6.0, 7.0], p), simulate([5.0, 6.0, 7.0], p)
    assert a == b and 30 < a["sensor_dropped"] < 100 and a["meets_fps"]
    slow = simulate([5.0] * 10, BudgetProfile(target_fps=30, frames=300, slowdown=10.0))
    assert slow["service_ms_mean_scaled"] == pytest.approx(50.0) and not slow["meets_fps"]


def test_profiles_load_and_validate():
    for name in ("desktop", "jetson_orin_nano_assumed", "jetson_agx_orin_assumed"):
        prof = load_profile(PROJECT_ROOT / "configs" / "deploy" / f"{name}.yaml")
        assert prof.target_fps > 0
    assert "assumed" in load_profile(PROJECT_ROOT / "configs" / "deploy" / "jetson_orin_nano_assumed.yaml").slowdown_source
    with pytest.raises(ConfigError):
        load_profile(None, drop_rate=1.5)


# -------------------------------------------------------------------- sanity


def test_sanity_checks_every_variant_without_training():
    from tac_ufld.sanity import run_sanity

    cfg = _cfg()
    rows = run_sanity(cfg, cfg.model.variants)
    assert len(rows) == 6 and all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]
    v05 = next(r for r in rows if r["variant"] == "lite_v05")
    assert v05["loaded"] == "backbone,head" and v05["grad_fusion_nonzero_frac"] > 0
    assert next(r for r in rows if r["variant"] == "ufld_v02")["loaded"] == "ufld"


# ------------------------------------------------------------------ ablation


def test_ablation_specs_are_fair_and_plan_only_by_default(capsys, tmp_path):
    from tac_ufld.ablation import arm_configs, load_spec

    for name in ("preprocessing", "augmentation"):
        spec = load_spec(PROJECT_ROOT / "configs" / "ablations" / f"{name}.yaml")
        configs = arm_configs(spec, str(tmp_path))
        seeds = {tuple(c.train.seeds) for c in configs.values()}
        assert len(seeds) == 1 and len({c.data.split.split_seed for c in configs.values()}) == 1
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nbase_config: configs/elas_pilot.yaml\nreference_arm: a\n"
                   "arms:\n  a: {}\n  b: {train.seeds: [9]}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="incomparable"):
        load_spec(bad)
    assert cli.main(["ablate", "--spec", str(PROJECT_ROOT / "configs" / "ablations" / "preprocessing.yaml")]) == 0
    out = capsys.readouterr().out
    assert "Plan only" in out and "in_channels=1" in out and "in_channels=4" in out


# ------------------------------------------------------------------ CLI tools


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """A fake finished run: results/<run>/seed_1/checkpoints/<variant>.pt."""
    root = tmp_path_factory.mktemp("results")
    ck_dir = root / "tiny_run" / "seed_1" / "checkpoints"
    ck_dir.mkdir(parents=True)
    cfg = _cfg()
    paths = {}
    for variant in ("ufld_baseline", "ufld_v02"):
        paths[variant] = _checkpoint(ck_dir, variant, cfg)
    video = root / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    rng = np.random.default_rng(0)
    for _ in range(8):
        writer.write(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    writer.release()
    return root, paths, video


def test_export_cli_with_synthetic_frames(tiny_run, tmp_path):
    pytest.importorskip("onnxruntime")
    root, paths, _ = tiny_run
    out = tmp_path / "deploy"
    assert cli.main(["export", "--checkpoint", str(paths["ufld_v02"]), "--out", str(out), "--synthetic",
                     "--check-frames", "4", "--int8", "--calib-frames", "4"]) == 0
    checks = json.loads((out / "numerical_checks.json").read_text(encoding="utf-8"))
    assert checks["onnxruntime_fp32"]["pass"] and "onnxruntime_int8" in checks
    assert (out / "encoder.onnx").exists() and (out / "int8" / "head.onnx").exists()


def test_stream_cli_writes_overlay_and_jsonl(tiny_run, tmp_path):
    _, paths, video = tiny_run
    out = tmp_path / "overlay.mp4"
    assert cli.main(["stream", "--checkpoint", str(paths["ufld_v02"]), "--video", str(video), "--out", str(out),
                     "--device", "cpu", "--every", "2"]) == 0
    assert out.exists() and out.stat().st_size > 0
    lines = out.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["frame_index"] for l in lines] == [0, 2, 4, 6]


def test_benchmark_cli_compares_modes(tiny_run, tmp_path):
    _, paths, _ = tiny_run
    out = tmp_path / "bench"
    assert cli.main(["benchmark", "--checkpoints", str(paths["ufld_baseline"]), str(paths["ufld_v02"]),
                     "--device", "cpu", "--measure-frames", "5", "--frames", "60", "--input-size", "120x160",
                     "--out", str(out)]) == 0
    report = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    modes = [(r["measured"]["variant"], r["measured"]["mode"]) for r in report["results"]]
    assert modes == [("ufld_baseline", "single"), ("ufld_v02", "cached"), ("ufld_v02", "recompute")]
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "Measured on this machine" in md and "NOT a device measurement" in md


def test_doctor_reports_environment():
    from tac_ufld.doctor import format_report, run_checks

    report = run_checks()
    assert report["torch"] and "datasets" in report and report["datasets"]["culane"] == "disabled"
    assert "PyTorch" in format_report(report)


@pytest.mark.skipif(not (PROJECT_ROOT / ".git").exists(), reason="not a git checkout (e.g. inside the Docker image)")
def test_package_excludes_data_checkpoints_and_caches(tmp_path, tiny_run):
    from tac_ufld.package import build_package

    run = tmp_path / "results" / "fake_run"
    (run / "seed_1" / "checkpoints").mkdir(parents=True)
    (run / "environment.json").write_text("{}", encoding="utf-8")        # matched by two patterns
    (run / "seed_1" / "checkpoints" / "m.pt").write_bytes(b"0")           # must never be packaged
    zip_path, report = build_package(tmp_path / "dist", [run])
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert len(names) == len(set(names))
    assert any(n.endswith("pilot_results/fake_run/environment.json") for n in names)
    root = names[0].split("/")[0]
    rel = [n[len(root) + 1:] for n in names]
    assert "MANIFEST.txt" in rel and "src/tac_ufld/cli.py" in rel and "configs/datasets.yaml" in rel
    assert "reference/tac_ufld_elas_pipeline_version_01_corrected.py" in rel
    assert not [n for n in rel if n.endswith((".pt", ".pth", ".onnx", ".engine"))]
    assert not [n for n in rel if n.startswith(("results/", "datasets/", ".venv/")) or "__pycache__" in n]
    assert "files" in report
