"""Smoke test of the whole pipeline on every dataset (a few batches each).

    python scripts/smoke_datasets.py                          # all four datasets
    python scripts/smoke_datasets.py --datasets culane tusimple
    python scripts/smoke_datasets.py --mock                   # force the synthetic copies
    python scripts/smoke_datasets.py --device cuda

For each dataset the REAL data is used when its root is set and exists
(ELAS_ROOT / CULANE_ROOT / TUSIMPLE_ROOT / OPENLANE_ROOT; ELAS also finds its
default folder), otherwise a small synthetic copy of the real on-disk layout
is written to a temporary folder (tests/dataset_fixtures.py) and deleted
afterwards. Every stage runs: split + leakage check, training (1 epoch, few
frames, no HPO), post-processing tuning on validation, evaluation with the
internal metrics AND the dataset's native metric, the Kalman reference,
latency, report. Numbers are NOT results; the script only proves the code
runs end to end with each dataset's real geometry (configs/<dataset>_smoke.yaml).

Output: results/smoke_datasets/<dataset>_<real|mock>/ and a summary table;
exit code 1 if any dataset failed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))           # for tests.dataset_fixtures
sys.path.insert(0, str(PROJECT_ROOT / "src"))   # when the package is not installed

import pandas as pd  # noqa: E402

from tac_ufld.config import load_config  # noqa: E402
from tac_ufld.experiment import ExperimentRunner  # noqa: E402

DATASETS = ["elas", "culane", "tusimple", "openlane"]
ROOT_ENV = {"elas": "ELAS_ROOT", "culane": "CULANE_ROOT", "tusimple": "TUSIMPLE_ROOT", "openlane": "OPENLANE_ROOT"}
NATIVE_KEY = {"culane": "native_culane_f1_iou50", "tusimple": "native_tusimple_accuracy",
              "openlane": "native_openlane_f1_iou50"}


def real_root(name: str) -> Path | None:
    value = os.environ.get(ROOT_ENV[name])
    if not value and name == "elas":
        value = str(PROJECT_ROOT / "../../datasets/dataset_elas_v1")
    return Path(value).resolve() if value and Path(value).is_dir() else None


def mock_root(name: str, tmp: Path) -> tuple[Path, dict]:
    """Synthetic copy of the dataset layout + config overrides it needs."""
    if name == "elas":
        from tests.conftest import make_elas_scene

        root = tmp / "elas"
        for scene, n in (("SYN_A", 130), ("SYN_B", 130), ("SYN_C", 70)):
            make_elas_scene(root, scene, n_frames=n, size=(320, 240), roi=(130, 110))
        return root, {"data.scenes": ["SYN_A", "SYN_B", "SYN_C"], "data.scene_tags": {},
                      "data.split.test_scenes": ["SYN_C"], "data.split.block_size": 30,
                      "data.split.max_seen_test_frames": 16}
    from tests import dataset_fixtures as fx

    if name == "tusimple":  # enough clips per drive for validation blocks plus the 4-clip purge gap
        return fx.make_tusimple(tmp / name, clips_per_drive=24), {"data.split.block_size": 4,
                                                                   "data.split.val_fraction": 0.2}
    maker = {"culane": fx.make_culane, "openlane": fx.make_openlane}[name]
    return maker(tmp / name), {}


def smoke(name: str, force_mock: bool, device: str, variants: list[str] | None) -> dict:
    t0 = time.time()
    tmp = None
    root = None if force_mock else real_root(name)
    source = "real" if root else "mock"
    overrides = {}
    try:
        if root is None:
            tmp = Path(tempfile.mkdtemp(prefix=f"smoke_{name}_"))
            root, overrides = mock_root(name, tmp)
        config = PROJECT_ROOT / "configs" / f"{name}_smoke.yaml"
        overrides.update({"name": f"{name}_{source}", "output_dir": str(PROJECT_ROOT / "results" / "smoke_datasets"),
                          "device": device, "data.root": str(root), "train.seeds": [1], "train.epochs": 1,
                          "hpo.enabled": False, "evaluation.latency_runs": 2, "evaluation.tensorboard": False})
        if name == "elas":  # the ELAS smoke config trains two seeds with HPO; one quick pass is enough here
            overrides.update({"data.split.max_train_frames": 32, "data.split.max_val_frames": 16,
                              "data.split.max_test_frames": 16, "data.num_workers": 0,
                              "evaluation.postprocess_grid": {"threshold": [0.5], "min_points": [2],
                                                              "poly_degree": [0]},
                              "evaluation.kalman_grid": {"q": [1.0], "alpha": [0.0]}})
        cfg = load_config(config, overrides)
        runner = ExperimentRunner(cfg, variants=variants, confirmed=True)
        runner.run()
        res = pd.read_csv(runner.out / "all_results.csv")
        test = res[(res["split"] == "test") & (res["protocol"] == "tuned") & (res["input"] == "full")]
        missing = [v for v in runner.variants if not (runner.out / "seed_1" / "checkpoints" / f"{v}.pt").exists()]
        native = NATIVE_KEY.get(name)
        problems = []
        if missing:
            problems.append(f"no checkpoint for {missing}")
        if set(test["variant"]) != set(runner.variants):
            problems.append("test rows missing for some variants")
        if native and native not in test.columns:
            problems.append(f"native metric {native} missing")
        return {"dataset": name, "data": source, "ok": not problems, "problems": "; ".join(problems),
                "variants": len(runner.variants),
                "train/val/test frames": "/".join(str(len(runner.splits.as_dict()[s])) for s in ("train", "val", "test")),
                "lane_f1_iou50 (mean)": round(float(test["lane_f1_iou50"].mean()), 3),
                "pixel_f1 (mean)": round(float(test["pixel_f1"].mean()), 3),
                "native metric": f"{native}={test[native].mean():.3f}" if native and native in test else "none (ELAS)",
                "seconds": round(time.time() - t0), "output": str(runner.out)}
    except Exception as exc:  # report every dataset, then fail at the end
        traceback.print_exc()
        return {"dataset": name, "data": source, "ok": False, "problems": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.time() - t0)}
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--mock", action="store_true", help="use the synthetic copies even when real data exists")
    p.add_argument("--device", default="cpu", help="cpu (default, leaves the GPU free), cuda or auto")
    p.add_argument("--variants", nargs="+", help="subset of variants (default: those of the smoke config)")
    args = p.parse_args()
    rows = [smoke(name, args.mock, args.device, args.variants) for name in args.datasets]
    table = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results" / "smoke_datasets"
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.csv"
    kept = pd.read_csv(summary) if summary.exists() else pd.DataFrame()
    if not kept.empty:  # keep the rows of datasets not re-run now
        kept = kept[~kept["dataset"].isin(table["dataset"])]
    pd.concat([kept, table], ignore_index=True).to_csv(summary, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\n" + table.drop(columns=["output"], errors="ignore").to_string(index=False))
    print(f"\nsummary of every dataset run so far: {summary}")
    return 0 if table["ok"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
