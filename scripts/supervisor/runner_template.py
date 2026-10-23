#!/usr/bin/env python3
"""TAC-UFLD: lightweight temporal lane detection vs. UFLD v1 on CULane, in ONE file.

The file carries the full TAC-UFLD code (packed at the bottom). On first use it unpacks the code next to itself
into ``tac_ufld_code/``. Every path is RELATIVE to the folder of this file.

FOLDER LAYOUT

    tac_ufld_run.py
    datasets/CULane/            driver_*/, laneseg_label_w16/, list/   (see "CULANE DATA" below)

STEPS (in this order, from the folder of this file)

    python tac_ufld_run.py setup                    # unpack, install packages, ImageNet weights, self-check
    python tac_ufld_run.py smoke --synthetic        # 5-15 min on generated CULane-format data: every stage runs
    python tac_ufld_run.py smoke                    # 10-20 min on a few real CULane frames
    python tac_ufld_run.py run --plan core          # the experiment (resumable: rerun the same command)
    python tac_ufld_run.py summary --plan core      # tables in the terminal + where the plots and videos are
    python tac_ufld_run.py pack --plan core         # zip of the results to send back (no checkpoints)

``python tac_ufld_run.py plan --plan core`` prints the exact configuration without running anything.

PLANS (docs/PLAN_CULANE_2026-10-23.md inside the code)

    core          6 models, 7 seeds. Baselines (UFLD v1, lite) 50 epochs; warm-started models (temporal and the
                  _ct / _static controls) a fixed 15-epoch fine-tuning budget. One 6 GB GPU (RTX 3050): seed 1
                  in about 2.5 days (Optuna ~10 h + training ~49 h), 7 seeds in about 15-16 days.
    full          10 models (adds UFLD v0.7 / v0.8 and their control, lite v0.6), 7 seeds, 50 epochs for every
                  model. ~230 GPU-hours per seed on an RTX 3050; ~25-35 h per seed on an A100 (seeds in parallel,
                  see SEVERAL GPUs). Needs >= 16 GB of GPU memory for batch 32.
    reproduction  UFLD v1 alone with exactly the article's recipe (1 seed, ~21 h on an RTX 3050): must reach the
                  published CULane F1 of 68.4 (+/- 0.5) before any comparison with UFLD's numbers.

PROTOCOL (the same for every plan)

* Official CULane lists: train 88,880 / val 9,675 / test 34,680 (9 categories). Optuna tunes on its OWN split,
  carved from train by whole clips (results/<run>/hpo/hpo_split.json proves zero overlap with val and test). The
  official val selects the checkpoint and the post-processing. The test set is read once, at the end.
* UFLD v1 (ResNet-18, Qin et al. ECCV 2020) keeps the article's recipe (no tuning): 288x800, 18 row anchors,
  200 cells, auxiliary segmentation loss, SGD 0.1, batch 32, warm-up 695 iterations, decay at epochs 25 and 38.
* Every model gets the same augmentation: UFLD's geometric one, photometric jitter, and current-frame
  degradation (30 % of the clips; a third of those on the whole window).
* Warm-started models score their starting weights first (epoch 0): none ends below the baseline it starts from.
* Metrics: CULane F1 at IoU 0.5 (lanes spline-interpolated, 30 px at 1640x590) in total and per category, FP on
  "cross", precision, recall, FP / FN per frame, position error, jitter, the Kalman-tracker reference, four
  robustness protocols, parameters, GMACs and latency. 7 paired seeds, exact Wilcoxon with Holm on the two
  declared primary comparisons (lite_tsm vs ufld_baseline, lite_tsm vs lite_tsm_static).
* Outputs per run (tac_ufld_code/results/<run>/): report/REPORT.md, CSV and Excel tables, plots (bars per metric,
  per-category heatmap, error counts, efficiency, training curves), seed_1/visuals/ (best and worst frames per
  model, model_comparison/<category>/*.jpg grids with one panel per model, model_comparison.mp4).

CULANE DATA (about 41 GB; Google Drive refuses scripted downloads when busy, so download in a browser)

    https://drive.google.com/drive/folders/1mSLgwVTiaUMAb4AVOWwlCD5JcWdrwpvu
    driver_23_30frame.tar.gz, driver_161_90frame.tar.gz, driver_182_30frame.tar.gz,   (train / val images)
    driver_37_30frame.tar.gz, driver_100_30frame.tar.gz, driver_193_90frame.tar.gz,   (test images)
    laneseg_label_w16.tar.gz, list.tar.gz, annotations_new.tar.gz, laneseg_label_w16_test.zip
    Extract all into datasets/CULane/ (needs ~85 GB free during extraction); annotations_new.tar.gz LAST,
    overwriting the old annotations. Then: python tac_ufld_run.py check

SEVERAL GPUs (optional): seed 1 runs the hyper-parameter search; afterwards the seeds are independent.

    python tac_ufld_run.py run --plan full --seeds 1            # first (it runs the search)
    CUDA_VISIBLE_DEVICES=1 python tac_ufld_run.py run --plan full --seeds 2 3     # then one process per GPU
    CUDA_VISIBLE_DEVICES=2 python tac_ufld_run.py run --plan full --seeds 4 5
    CUDA_VISIBLE_DEVICES=3 python tac_ufld_run.py run --plan full --seeds 6 7
    python tac_ufld_run.py run --plan full                      # last: report + robustness (finished seeds skip)

On a SLURM cluster use scripts/cluster/culane.slurm inside tac_ufld_code/ instead.

REQUIREMENTS: Python >= 3.10, an NVIDIA GPU, PyTorch with CUDA installed BEFORE setup
(https://pytorch.org/get-started/locally/), internet for setup only. ffmpeg on PATH makes the videos H.264.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE / "tac_ufld_code"

# Dataset folders, RELATIVE to this file. CULane is the experiment of the article; the other datasets are kept as
# options: uncomment one line (and pass --dataset <name>) to run the same pipeline on it with its own config.
DATASETS = {
    #            folder next to this file      plan -> config inside the code
    "culane":   ("datasets/CULane",          {"core": "configs/culane.yaml",
                                              "full": "configs/cluster/culane_full.yaml",
                                              "reproduction": "configs/culane_ufld_reproduction.yaml",
                                              "smoke": "configs/culane_smoke.yaml"}),
    # "elas":     ("datasets/dataset_elas_v1", {"core": "configs/elas_pilot_v3.yaml",
    #                                           "full": "configs/cluster/elas_full22.yaml",
    #                                           "smoke": "configs/elas_smoke.yaml"}),
    # "openlane": ("datasets/OpenLane",        {"core": "configs/openlane.yaml",
    #                                           "smoke": "configs/openlane_smoke.yaml"}),
    # "tusimple": ("datasets/TUSimple",        {"core": "configs/tusimple.yaml",
    #                                           "smoke": "configs/tusimple_smoke.yaml"}),
}
REQUIREMENTS = ["numpy>=1.24", "pandas>=2.0", "scipy>=1.10", "opencv-python-headless>=4.8", "Pillow>=10",
                "matplotlib>=3.7", "optuna>=3.4", "PyYAML>=6", "openpyxl>=3.1", "tensorboard>=2.14"]


# -- helpers ------------------------------------------------------------------------------------------------------------

def say(msg: str) -> None:
    print(f"[tac_ufld_run] {msg}", flush=True)


def code_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CODE / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONUTF8", "1")
    return env


def tac(*args: str) -> int:
    """``python -m tac_ufld <args>`` inside the unpacked code."""
    cmd = [sys.executable, "-m", "tac_ufld", *args]
    say("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=CODE, env=code_env())


def unpack(force: bool = False) -> None:
    if os.name == "nt" and len(str(HERE)) > 80:
        say(f"note: this folder's path is {len(str(HERE))} characters long. The code handles Windows' 260-character "
            "limit, but other tools (Explorer, zip) may not: a short folder such as C:\\tac is safer")
    digest = hashlib.sha256(PAYLOAD.encode("ascii")).hexdigest()[:16]
    stamp = CODE / ".payload"
    if CODE.is_dir() and stamp.exists() and stamp.read_text().strip() == digest and not force:
        return
    if CODE.is_dir():   # keep results/ of earlier runs, replace the code
        for item in CODE.iterdir():
            if item.name not in ("results",):
                shutil.rmtree(item) if item.is_dir() else item.unlink()
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:
        z.extractall(CODE)
    stamp.write_text(digest)
    say(f"code unpacked to {CODE} (bundle {BUNDLE_INFO})")


def gpu_memory_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:
        pass
    return 0.0


def data_root(dataset: str, override: str | None) -> Path:
    root = Path(override) if override else HERE / DATASETS[dataset][0]
    return root.resolve()


def deep_set(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def config_name(dataset: str, plan: str) -> str:
    import yaml

    return yaml.safe_load((CODE / DATASETS[dataset][1][plan]).read_text(encoding="utf-8"))["name"]


def build_config(dataset: str, plan: str, root: Path, workers: int, batch: int | None) -> Path:
    """The plan's config (models, seeds, budgets, HPO, primary comparisons all come from it) with this machine's
    data root and loader workers, written to tac_ufld_code/configs/generated/."""
    import yaml

    plans = DATASETS[dataset][1]
    if plan not in plans:
        sys.exit(f"[tac_ufld_run] plan '{plan}' is not defined for {dataset} (available: {', '.join(plans)})")
    base = plans[plan]
    cfg = yaml.safe_load((CODE / base).read_text(encoding="utf-8"))
    deep_set(cfg, "requires_confirmation", False)
    deep_set(cfg, "data.root", str(root))
    deep_set(cfg, "data.num_workers", 0 if plan == "smoke" else workers)
    gpu = gpu_memory_gb()
    if plan == "full" and 0 < gpu < 15.0:
        say(f"WARNING: the full plan trains 3-frame UFLD models with batch 32, which needs >= 16 GB of GPU memory; "
            f"this GPU has {gpu:.0f} GB. Use the core plan here, or --batch-size 16 (then report it: it changes "
            "UFLD's recipe).")
    if batch:
        deep_set(cfg, "train.batch_size", batch)
        say(f"batch size overridden to {batch} (report it: UFLD's recipe uses 32)")
    out = CODE / "configs" / "generated" / f"{cfg['name']}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = f"# generated by tac_ufld_run.py from {base} (bundle {BUNDLE_INFO}); GPU {gpu:.0f} GB\n"
    out.write_text(header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def check_data(dataset: str, root: Path) -> None:
    if not root.is_dir():
        sys.exit(f"[tac_ufld_run] dataset folder not found: {root}\n"
                 f"  put the {dataset} data in {HERE / DATASETS[dataset][0]} or pass --data-root")


def synthetic_root() -> Path:
    """Generated CULane-format data (a few 1640x590 frames with lanes, lists, masks and categories)."""
    root = HERE / "datasets" / "CULane_synthetic"
    if not (root / "list" / "train_gt.txt").exists():
        say(f"writing synthetic CULane-format data to {root}")
        rc = subprocess.call([sys.executable, "-c", "import sys; from pathlib import Path; "
                              "from tools.dataset_fixtures import make_culane; make_culane(Path(sys.argv[1]))",
                              str(root)], cwd=CODE, env=code_env())
        if rc:
            sys.exit("[tac_ufld_run] could not write the synthetic data")
    return root


# -- commands -----------------------------------------------------------------------------------------------------------

def cmd_setup(args) -> int:
    unpack(force=args.force)
    try:
        import torch
    except ImportError:
        sys.exit("[tac_ufld_run] PyTorch is not installed. Install it with CUDA first: "
                 "https://pytorch.org/get-started/locally/")
    say(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, "
        f"GPU memory {gpu_memory_gb():.1f} GB")
    if not args.no_install:
        rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q", *REQUIREMENTS])
        if rc:
            return rc
    say("downloading the ImageNet ResNet weights (once, into the torch hub cache)")
    rc = subprocess.call([sys.executable, "-c",
                          "import torch; from tac_ufld.models.resnet import IMAGENET_WEIGHTS as W; "
                          "[torch.hub.load_state_dict_from_url(u, map_location='cpu') for u in W.values()]"],
                         cwd=CODE, env=code_env())
    if rc:
        return rc
    tac("doctor")
    say("setup done. 'root does not exist' lines above only concern datasets not placed in datasets/ yet")
    return 0


def cmd_check(args) -> int:
    """Reads the dataset only: completeness against the published counts, then label overlays to look at."""
    unpack()
    root = data_root(args.dataset, args.data_root)
    check_data(args.dataset, root)
    rc = subprocess.call([sys.executable, "scripts/check_dataset.py", args.dataset, str(root)], cwd=CODE,
                         env=code_env())
    if rc:
        return rc
    return tac("validate-dataset", "--dataset", args.dataset, "--root", str(root),
               "--config", DATASETS[args.dataset][1]["core"])


def cmd_plan(args) -> int:
    unpack()
    root = synthetic_root() if getattr(args, "synthetic", False) else data_root(args.dataset, args.data_root)
    cfg = build_config(args.dataset, args.plan, root, args.workers, args.batch_size)
    say(f"config: {cfg}")
    print(cfg.read_text(encoding="utf-8"))
    return 0


def cmd_smoke(args) -> int:
    unpack()
    root = synthetic_root() if args.synthetic else data_root(args.dataset, args.data_root)
    check_data(args.dataset, root)
    cfg = build_config(args.dataset, "smoke", root, args.workers, args.batch_size)
    rc = tac("run", "--config", str(cfg), "--data-root", str(root))
    if rc:
        return rc
    run_dir = CODE / "results" / config_name(args.dataset, "smoke")
    rc = tac("robustness", "--runs", str(run_dir), "--workers", "0")
    if rc == 0:
        say("smoke test passed: every stage ran (its numbers are not results). "
            f"Look at {run_dir / 'report' / 'REPORT.md'} and {run_dir / 'seed_1' / 'visuals'}")
    return rc


def cmd_run(args) -> int:
    unpack()
    root = data_root(args.dataset, args.data_root)
    check_data(args.dataset, root)
    cfg = build_config(args.dataset, args.plan, root, args.workers, args.batch_size)
    extra = ["--seeds", *map(str, args.seeds)] if args.seeds else []
    rc = tac("run", "--config", str(cfg), "--data-root", str(root), "--resume", *extra)
    if rc:
        say("the run stopped; rerun the same command to resume where it stopped")
        return rc
    if args.seeds:
        say(f"seeds {args.seeds} done. When every seed is done, run the same command WITHOUT --seeds: it skips "
            "the finished seeds, writes the report and runs the robustness evaluation")
        return 0
    run_dir = CODE / "results" / config_name(args.dataset, args.plan)
    say("robustness: four corruption regimes on the held-out test (evaluation only)")
    rc = tac("robustness", "--runs", str(run_dir), "--workers", str(args.workers))
    if rc == 0:
        summarize(run_dir)
    return rc


def summarize(run_dir: Path) -> None:
    """Main tables in the terminal (mean over the finished seeds) and the paths of plots, grids and videos."""
    import pandas as pd

    report = run_dir / "report"
    agg = report / "aggregate_test_tuned.csv"
    if not agg.exists():
        say(f"no report yet in {report} (it is written when every seed of the run is done)")
        return
    d = pd.read_csv(agg)
    wide = d.pivot_table(index="variant", columns="metric", values="mean")
    cols = [c for c in ("lane_f1_iou50", "lane_precision_iou50", "lane_recall_iou50", "lane_fp_per_frame_iou50",
                        "lane_fn_per_frame_iou50", "frame_accuracy_iou50", "anchor_mae_px", "jitter_px")
            if c in wide.columns]
    n = int(d["n"].max()) if "n" in d else 0
    print(f"\nTest set, mean over {n} seed(s) (post-processing tuned on val):")
    print(wide[cols].round(4).to_string())
    native = report / "aggregate_native_test_tuned.csv"
    if native.exists():
        nd = pd.read_csv(native).pivot_table(index="variant", columns="metric", values="mean")
        keep = [c for c in nd.columns if c.startswith("native_culane_f1") or c.startswith("native_culane_fp")]
        if keep:
            print("\nCULane native metric (every annotated lane, spline, 30 px at 1640x590):")
            print(nd[keep].round(4).to_string())
    cond = report / "test_conditions.csv"
    if cond.exists():
        c = pd.read_csv(cond).pivot_table(index="variant", columns="metric", values="mean")
        f1 = [m for m in c.columns if m.startswith("condition_f1_iou50_")]
        if f1:
            t = c[f1].rename(columns=lambda m: m.removeprefix("condition_f1_iou50_"))
            print("\nF1 per test category:")
            print(t.round(3).to_string())
        fp = [m for m in c.columns if m == "condition_fp_iou50_cross"]
        if fp:
            print("\nFP on 'cross' (no lanes there; lower is better):")
            print(c[fp].round(0).to_string())
    eff = report / "efficiency.csv"
    if eff.exists():
        e = pd.read_csv(eff)
        print("\nCost (GMACs and latency of the whole 3-frame clip for temporal models; in streaming each frame "
              "costs one pass, e.g. lite_tsm = lite_baseline):")
        print(e[[c for c in ("variant", "params_millions", "gmacs", "latency_ms", "fps") if c in e]].round(3)
              .to_string(index=False))
    tests = report / "paired_tests_test_tuned.csv"
    if tests.exists():
        print(f"\nPaired tests (Wilcoxon, Holm on the primary pairs): {tests}")
    print(f"\nReport: {report / 'REPORT.md'}\nPlots: {report}/*.png\n"
          f"Model comparison: {run_dir / 'seed_1' / 'visuals' / 'model_comparison'} and model_comparison.mp4")


def cmd_summary(args) -> int:
    unpack()
    summarize(CODE / "results" / config_name(args.dataset, args.plan))
    return 0


def cmd_pack(args) -> int:
    unpack()
    name = config_name(args.dataset, args.plan)
    run_dir = CODE / "results" / name
    if not run_dir.is_dir():
        sys.exit(f"[tac_ufld_run] no results in {run_dir}")
    out = HERE / f"results_{name}.zip"
    skip_suffix = {".pt", ".pth", ".onnx", ".engine", ".db"}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in run_dir.rglob("*"):
            if p.is_file() and p.suffix not in skip_suffix and "tensorboard" not in p.parts:
                z.write(p, p.relative_to(CODE / "results"))
    say(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MB); please send this file back")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="TAC-UFLD experiments (see the file header)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="unpack the code, install packages, download weights, self-check")
    s.add_argument("--no-install", action="store_true", help="skip pip install (packages already present)")
    s.add_argument("--force", action="store_true", help="unpack the code again")
    cpus = max(2, min(8, (os.cpu_count() or 4) - 2))
    for name, help_ in (("check", "check the dataset and write label overlays"),
                        ("plan", "print the configuration only"), ("smoke", "quick test of every stage"),
                        ("run", "the experiment (resumable)"), ("summary", "print the result tables"),
                        ("pack", "zip the results to send back")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--dataset", default="culane", choices=sorted(DATASETS))
        if name in ("plan", "run", "summary", "pack"):
            p.add_argument("--plan", default="core", help="core | full | reproduction (CULane)")
        if name in ("check", "plan", "smoke", "run"):
            p.add_argument("--data-root", help="dataset folder (default: datasets/<name> next to this file)")
        if name in ("plan", "smoke", "run"):
            p.add_argument("--workers", type=int, default=cpus, help="data-loading processes")
            p.add_argument("--batch-size", type=int, help="override the batch size of the plan")
        if name in ("plan", "smoke"):
            p.add_argument("--synthetic", action="store_true", help="use generated CULane-format data")
        if name == "run":
            p.add_argument("--seeds", type=int, nargs="+", help="run only these seeds (e.g. one per GPU)")
    args = ap.parse_args()
    return {"setup": cmd_setup, "check": cmd_check, "plan": cmd_plan, "smoke": cmd_smoke, "run": cmd_run,
            "summary": cmd_summary, "pack": cmd_pack}[args.cmd](args)


BUNDLE_INFO = "__BUNDLE_INFO__"
PAYLOAD = (
__PAYLOAD__
)

if __name__ == "__main__":
    sys.exit(main())
