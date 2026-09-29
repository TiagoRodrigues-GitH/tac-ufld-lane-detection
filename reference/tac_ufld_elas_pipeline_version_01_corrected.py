# ============================================================
# TAC-UFLD v1.1 -- ELAS PIPELINE (bugfixed, ADAS/TRL-ready)
# Spatial lane-pixel head + warped residual temporal fusion
# ============================================================
#
# This file supersedes TAC-UFLD v1.0. Structure, dataset parsing, Optuna
# HPO, ablation framework, plotting, and multi-seed statistics are kept.
# Fixes applied in this revision:
#   * Sanity-check console output hardcoded shape "(2, 32)" regardless of
#     cfg.num_row_anchors -> now derived from cfg at print time.
#   * cfg.row_anchors comment described a [0.28*H, 0.98*H] span that did
#     not match the actual linspace(0.52*H, 0.99*H, ...) call -> comment
#     corrected to match the real anchor span.
#   * DATASET_ROOT was a single hardcoded relative path with no documented
#     alternative -> replaced by an explicit dataset-selection block (ELAS
#     active, CULane/TuSimple wired but commented, per dataset layout).
#   * No quantitative ADAS/TRL acceptance thresholds existed anywhere in
#     the file (only a latency budget) -> added ADAS_READINESS_TARGETS +
#     a readiness report printed/saved at the end of run_pipeline.
#   * No qualitative "expected vs detected" output existed for visual
#     inspection -> added save_scenario_comparison_images(), producing
#     original/GT/prediction overlay panels per scenario tag.
#   * RUN_OPTUNA_HPO / epoch / trial counts were left at "smoke==final"
#     values -> split into an explicit SMOKE-TEST profile (active by
#     default) and a FINAL-RUN profile (commented), per project request.
#   * from torch.utils.tensorboard import SummaryWriter crashed at import
#     time with "ImportError: TensorBoard logging requires TensorBoard
#     version 1.15 or above" whenever the `tensorboard` PyPI package was
#     missing or stale in the active environment -> wrapped in the same
#     try/except-and-pip-install pattern already used for optuna/scipy/
#     seaborn below, using --upgrade so a stale-but-present install
#     actually gets replaced instead of pip treating it as satisfied.
#
# ------------------------------------------------------------------
# ADAS EMBEDDED IMPLEMENTATION -- METRIC OBJECTIVES (TRL 3/4 gate)
# ------------------------------------------------------------------
# These are the acceptance thresholds this pipeline is evaluated against
# before the model is considered viable for TRL 3 (analytical/experimental
# proof-of-concept, this ELAS offline benchmark) and TRL 4 (validation in
# a lab-representative environment, e.g. HIL/vehicle logs). They are not
# hyperparameters -- do not tune to hit them, they define pass/fail.
#
#   Metric                          | TRL3 target | TRL4 target | Rationale
#   -------------------------------- | ----------- | ----------- | ---------
#   polyline_pixel_f1 (primary)      |   >= 0.75   |   >= 0.90   | pixel-level lane mask agreement, ~SCNN/UFLD-class benchmarks
#   iou_f1 (lane-level, IoU>=0.35)   |   >= 0.70   |   >= 0.85   | matches CULane-style whole-lane detection scoring
#   polyline_pixel_f2_safety         |   >= 0.75   |   >= 0.90   | recall-weighted: missed lane costs more than spurious one
#   lane_position_jitter_px (mean)   |   <= 8 px   |   <= 4 px   | frame-to-frame stability feeding a steering-assist controller
#   inference latency (ADAS_TARGET_FPS = 30 Hz)  | <= 33.3 ms  | <= 33.3 ms | real-time budget for the embedded target platform
#   false negative rate (missed lane)|   <= 15%    |   <=  8%    | FN is the unsafe failure mode for lane-keep assist
#
# All thresholds are printed against the achieved values by
# print_adas_readiness_report()/plot_adas_readiness() at the end of every
# run, and exported to adas_readiness_report.csv per seed. Numbers above
# are literature-informed starting points (SCNN, UFLD, LaneATT-class
# results on CULane/TuSimple) -- recalibrate them against your specific
# ECU/vehicle safety-case requirements before using them as a go/no-go
# gate in a real ADAS program.
#
# Run order:
#   1. python this_script.py --sanity        # verify coordinate pipeline
#   2. python this_script.py                 # smoke test (default config below)
#      then flip to the FINAL-RUN profile in section 0 for the paper run
# ============================================================

import os
import sys
import copy
import time
import random
import shutil
import subprocess
import textwrap
import argparse
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from PIL import Image as PILImage
import gc
import cv2
import pickle
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Graphviz PATH setup -- single source of truth, used by every
# architecture-diagram exporter below (see section 5d). Previously this
# search ran three separate times (once here at import time, once again
# inside generate_architecture_diagram(), and not at all inside
# export_arch_images()) with three slightly different implementations.
# Consolidated into one idempotent helper so it only searches the
# filesystem once per process and every diagram exporter shares the
# same result.
# ------------------------------------------------------------------
_GRAPHVIZ_CANDIDATES = [
    Path(r"C:\Program Files\Orange\Library\bin"),
    Path(r"C:\Program Files\Graphviz\bin"),
]


def _ensure_graphviz_on_path():
    """Make sure Graphviz's `dot` executable is discoverable on PATH
    before torchview (which shells out to `dot` to render PNG/SVG) is
    used. Runs the filesystem/PATH search only once per process."""
    if _ensure_graphviz_on_path.done:
        return
    _ensure_graphviz_on_path.done = True
    for candidate in _GRAPHVIZ_CANDIDATES:
        if (candidate / "dot.exe").exists() or (candidate / "dot").exists():
            os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
            print("[graphviz] using", candidate)
            return
    dot = shutil.which("dot")
    if dot:
        os.environ["PATH"] = str(Path(dot).parent) + os.pathsep + os.environ.get("PATH", "")
        print("[graphviz] using", Path(dot).parent)
    else:
        print("[graphviz] NOT FOUND -- torchview PNG/SVG export will be "
              "skipped (install Graphviz and ensure `dot` is on PATH)")


_ensure_graphviz_on_path.done = False
_ensure_graphviz_on_path()
# ------------------------------------------------------------------


def _pip_install(package):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=False)


try:
    import optuna
except ImportError:
    _pip_install("optuna")
    import optuna

try:
    import scipy
    import scipy.stats
except ImportError:
    _pip_install("scipy")
    import scipy
    import scipy.stats

try:
    import seaborn as sns
except ImportError:
    _pip_install("seaborn")
    import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")


# ------------------------------------------------------------------
# BUGFIX (this revision): a bare
#     from torch.utils.tensorboard import SummaryWriter
# crashes at import time with:
#     ImportError: TensorBoard logging requires TensorBoard version
#     1.15 or above
# whenever the `tensorboard` PyPI package is missing OR present-but-stale
# in the active environment/venv -- torch.utils.tensorboard does its own
# version probe on import and raises exactly this error in both cases.
# Fixed the same way optuna/scipy/seaborn are handled above: try the
# import, and if it fails, install and retry. Unlike the plain
# `_pip_install()` helper used above, this uses --upgrade specifically,
# because a bare `pip install tensorboard` is a no-op ("Requirement
# already satisfied") when some old/broken tensorboard is already
# installed -- which is precisely the case that produces this error --
# so without --upgrade the retry would fail again with the same message.
# ------------------------------------------------------------------
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "tensorboard"],
        check=False,
    )
    from torch.utils.tensorboard import SummaryWriter

# ------------------------------------------------------------------
# BUGFIX (prior revision, kept): the previous version created a single
# module-level `SummaryWriter(log_dir="results/tensorboard/tac_ufld")`
# here, at IMPORT time. Three separate problems with that:
#   1. It was a pure side effect of importing this module -- even
#      `import this_module` without running anything created a directory
#      on disk.
#   2. It was never actually used anywhere else in the file (no
#      `writer.add_scalar` / `add_image` / `add_graph` calls existed at
#      all) -- it was dead code that looked like instrumentation but
#      logged nothing.
#   3. Even if it had been used, every seed / model variant / Optuna
#      trial would have written into the exact same log_dir and
#      overwritten each other's scalars, which directly violates the
#      "different trials do not overwrite each other's logs" requirement.
# Fixed by removing the eager global writer and replacing it with
# `get_tb_writer(log_dir)` below, called with a unique per-run directory
# at the point each run actually starts (see run_pipeline / HPO code).
# ------------------------------------------------------------------

_active_tb_writers = []


def get_tb_writer(log_dir):
    """Create a SummaryWriter rooted at a unique log_dir and track it so
    close_all_tb_writers() can flush/close everything at process exit even
    if an exception skips an explicit close(). Never share one writer's
    log_dir across seeds/models/trials -- each call site below passes a
    directory that already encodes seed + model/variant (+ trial number
    for Optuna), so TensorBoard's run picker can tell them apart."""
    log_dir = str(log_dir)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    w = SummaryWriter(log_dir=log_dir)
    _active_tb_writers.append(w)
    return w


def close_tb_writer(writer):
    """Fecha um writer e remove a última referência dele de
    _active_tb_writers, para que o writer (e o grafo jit-traced que
    add_graph() gerou) possa realmente ser coletado pelo GC. Antes,
    get_tb_writer() empilhava todo writer nessa lista e nada nunca
    removia -- close_all_tb_writers() (a única função que faz .clear())
    não é chamada em lugar nenhum do __main__, então todo writer de
    seed/modelo/trial criado durante o run inteiro ficava vivo até o
    fim do processo."""
    try:
        writer.flush()
        writer.close()
    except Exception as exc:
        print(f"[tensorboard] warning: failed to close writer cleanly: {exc}")
    finally:
        if writer in _active_tb_writers:
            _active_tb_writers.remove(writer)


def close_all_tb_writers():
    for w in list(_active_tb_writers):
        close_tb_writer(w)


# ============================================================
# 0a. Disk-space guard (this revision)
# ============================================================
#
# BUGFIX: a full disk previously surfaced as
#     OSError: [Errno 28] No space left on device
# raised deep inside save_lanes_txt() / export_cache_predictions(), which
# crashed the ENTIRE multi-seed run after training, HPO, evaluation and
# every plot for that seed had already completed successfully -- the
# single least valuable place to lose the whole process. Two changes:
#   1. _check_disk_space() is called before RESULT_ROOT's per-seed work
#      starts and again right before the final export step, so a nearly-
#      full drive is reported as an early, actionable warning instead of
#      a crash discovered hours later.
#   2. save_lanes_txt() itself now catches OSError and skips that one
#      file (see its docstring below) instead of propagating.
# Neither of these FIXES a full disk -- only you can free space or point
# RESULT_ROOT at a drive with more room -- but together they turn "the
# whole run dies with no partial results" into "you get a warning and
# keep whatever results already fit."

def _check_disk_space(path, min_free_mb=500, label=""):
    """Best-effort free-space check for the drive containing `path`.
    Returns True if at least `min_free_mb` MB are free, else prints a
    warning and returns False. Never raises -- if the check itself fails
    (e.g. an exotic filesystem), it prints why and assumes space is OK
    rather than blocking the run over a diagnostic failure."""
    try:
        probe = Path(path)
        probe = probe if probe.exists() else probe.parent
        usage = shutil.disk_usage(str(probe))
        free_mb = usage.free / (1024 ** 2)
    except Exception as exc:
        print(f"[disk-space] could not check free space for {path}: {exc}")
        return True
    if free_mb < min_free_mb:
        print(
            f"[disk-space] WARNING{f' ({label})' if label else ''}: only "
            f"{free_mb:.0f} MB free on the drive containing {path} "
            f"(wanted >= {min_free_mb} MB). Exports/checkpoints/TensorBoard "
            f"logs below may fail with OSError [Errno 28]. Consider "
            f"deleting stale results/ subfolders from previous runs, or "
            f"pointing RESULT_ROOT at a drive with more room."
        )
        return False
    return True


# ============================================================
# 0b. Robust partial state_dict loading (this revision)
# ============================================================
#
# BUGFIX: plain `module.load_state_dict(source_state, strict=False)` only
# tolerates MISSING or UNEXPECTED keys -- a key that exists in both source
# and destination but has a different tensor SHAPE still raises a
# RuntimeError regardless of strict=False. That case is no longer
# hypothetical in this revision: the temporal models' current-frame
# backbone (LightweightBackboneTemporal, half-width -- see its docstring)
# is now architecturally different from the baseline's backbone
# (LightweightBackbone, full-width), so the old
#     temporal_model_v02.backbone.load_state_dict(baseline_model.backbone.state_dict())
# warm-start would crash immediately on the first shape mismatch. Every
# warm-start call site below now goes through this helper instead, which
# copies only the tensors that are both present AND shape-compatible and
# reports what it had to skip.

def load_partial_state_dict(module, source_state_dict, label=""):
    """Copy every tensor from source_state_dict into module whose key
    exists in both and whose shape matches; keys that are missing or
    shape-mismatched are skipped (reported, not raised)."""
    dest_state = module.state_dict()
    compatible = {}
    skipped = []
    for k, v in source_state_dict.items():
        if k in dest_state and dest_state[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append(k)
    dest_state.update(compatible)
    module.load_state_dict(dest_state)
    if skipped:
        preview = skipped[:5]
        more = " ..." if len(skipped) > 5 else ""
        print(f"[warm-start{f' ({label})' if label else ''}] "
              f"copied {len(compatible)}/{len(source_state_dict)} tensor(s); "
              f"skipped (shape mismatch or missing in destination): "
              f"{preview}{more}")
    return compatible.keys(), skipped


# ============================================================
# 0. PIPELINE SWITCHES
# ============================================================

N_TRAIN_SUBSET = 2000
N_VAL_SUBSET = 500
N_TEST_SUBSET = 500

# ------------------------------------------------------------------
# Dataset selection -- ELAS is the active dataset for this pipeline.
# CULane and TuSimple loaders are NOT implemented here (their annotation
# format differs from ELAS's XML groundtruth and needs its own parser);
# these paths are left commented as the wiring point for that future
# work, per the project request. Only ever one DATASET_ROOT is active.
# ------------------------------------------------------------------
DATASET_ROOT = Path("../datasets/dataset_elas_v1")   # ELAS (active)

# CULane -- https://xingangpan.github.io/projects/CULane.html
# DATASET_ROOT = Path("../datasets/CULane")

# TuSimple -- https://gts.ai/dataset-download/tusimple/
# DATASET_ROOT = Path("../datasets/TUSimple")

LEAKAGE_BLOCK_SIZE = 60
LEAKAGE_PURGE_FRAMES = 4

SCENE_PLAN = {
    "BR_S01": {"weight": 25, "tags": {"transition": True}},
    "BR_S02": {"weight": 25, "tags": {"transition": True}},
    "ROD_S03": {"weight": 20, "tags": {"rainy": True}},
    "VIX_S03": {"weight": 20, "tags": {"occlusion": True, "transition": True}},
    "VIX_S05": {"weight": 20, "tags": {"occlusion": True}},
    "VIX_S07": {"weight": 20, "tags": {"shaky": True, "transition": True}},
    "VIX_S04": {"weight": 20, "tags": {"transition": True}},
    "VV_S03": {"weight": 15, "tags": {"transition": True}},
    "GRI_S01": {"weight": 15, "tags": {"transition": True}},
    "VIX_S11": {"weight": 20, "tags": {}},
}

# ------------------------------------------------------------------
# SMOKE-TEST profile (active by default): fast end-to-end run to verify
# the whole pipeline (data -> train -> Optuna HPO -> postproc sweep ->
# eval -> plots -> comparison images) executes without crashing, before
# committing hours to the FINAL-RUN profile below. Not representative of
# reported accuracy -- see ADAS_READINESS_TARGETS above and the smoke
# warning printed in __main__.
N_OPTUNA_TRIALS = 6
SMOKE_EPOCHS = 50
OPTUNA_EPOCHS = 3
RUN_OPTUNA_HPO = True
SEEDS = [42,1,7,2026]

# FINAL-RUN profile for IEEE-grade / TRL3-4 results: uncomment and comment
# out the block above. Needs >=6 seeds for paired_wilcoxon_vs_baseline()
# and >=50 epochs so early stopping (patience=8) can actually trigger.
# N_OPTUNA_TRIALS = 20
# SMOKE_EPOCHS = 50
# OPTUNA_EPOCHS = 20
# RUN_OPTUNA_HPO = True
# SEEDS = [1, 2, 3, 4, 5, 6]

TRAIN_BASELINE_FIRST = True

STUDY_VERSION_TAG = "v1_spatial_warp"
RESULT_ROOT = Path(__file__).resolve().parent / "results" / "v1_elas"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Recommended-structure directories (task Section 9), adapted to this
# project's existing RESULT_ROOT/results/v1_elas layout rather than
# duplicating a second top-level "results/" tree, per the task's own
# "integrate into it instead of duplicating" instruction.
#   * ARCHITECTURE_ROOT is experiment-independent (task Section 2: not
#     tied to a seed or epoch) -- one shared results/v1_elas/architecture/
#     folder for every Torchview diagram, regardless of which seed
#     generated the model.
#   * TENSORBOARD_ROOT holds one subdirectory per (seed, model) plus
#     optuna/<variant>/trial_<n> for HPO runs (see get_tb_writer() call
#     sites) -- unique log_dirs throughout, never shared.
#   * Per-seed visualizations/videos live under each seed's own
#     result_root (seed_<n>/plots/visualizations, seed_<n>/videos/...)
#     since predictions genuinely differ by seed; this deliberately
#     departs from the task's flat example layout for that reason.
# ------------------------------------------------------------------
ARCHITECTURE_ROOT = RESULT_ROOT / "architecture"
TENSORBOARD_ROOT = RESULT_ROOT / "tensorboard"
ARCHITECTURE_ROOT.mkdir(parents=True, exist_ok=True)
TENSORBOARD_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "baseline": "Baseline",
    "v02": "TAC-UFLD v1.0 (warped)",
    "v03": "TAC-UFLD v1.0 (gated init)",
    "v04": "TAC-UFLD v1.0 (deep)",
}
BASELINE_LABEL = MODEL_LABELS["baseline"]
MODEL_ORDER = [MODEL_LABELS[k] for k in ("baseline", "v02", "v03", "v04")]
_palette_colors = sns.color_palette("colorblind", n_colors=len(MODEL_ORDER))
MODEL_PALETTE = dict(zip(MODEL_ORDER, _palette_colors))

POSTPROCESS_SWEEP_GRID = {
    "threshold": [0.25, 0.35, 0.45, 0.55],
    "min_points": [3, 4, 6],
    "poly_degree": [1, 2, 3],
    "y_step": [1, 2, 3],
}
POSTPROCESS_BASE_PARAMS = {"min_lane_length_fraction": 0.15, "duplicate_distance": 12}
POSTPROCESS_COMMON_FIXED_PARAMS = {
    "threshold": 0.35, "min_points": 4, "min_lane_length_fraction": 0.15,
    "duplicate_distance": 12, "y_step": 2, "poly_degree": 2,
}


# ============================================================
# 1. ELAS dataset -- discovery, parsing, splitting
# ============================================================

def discover_scene_paths(dataset_root, scene_name):
    scene_dirs = [p for p in Path(dataset_root).rglob("*")
                  if p.is_dir() and p.name == scene_name]
    config_path = next((p / "config.xml" for p in scene_dirs
                        if (p / "config.xml").exists()), None)
    gt_path = next((p / "groundtruth.xml" for p in scene_dirs
                    if (p / "groundtruth.xml").exists()), None)
    image_root = next((p / "images" / "images" for p in scene_dirs
                       if (p / "images" / "images").is_dir()), None)
    return config_path, gt_path, image_root


def parse_elas_config(config_xml_path):
    tree = ET.parse(config_xml_path)
    root = tree.getroot()
    ds = root.find("dataset")
    fs = ds.find("frame_sequence")
    frame_size = ds.find("frame_size")
    roi = ds.find("region_of_interest")
    ipm = ds.find("ipm_points")
    return {
        "id": ds.get("id"), "name": ds.get("name"), "path": ds.get("path"),
        "frame_start": int(fs.get("start")), "frame_end": int(fs.get("end")),
        "frame_w": int(frame_size.get("width")), "frame_h": int(frame_size.get("height")),
        "roi_x": int(roi.get("x")), "roi_y": int(roi.get("y")),
        "roi_w": int(roi.get("width")), "roi_h": int(roi.get("height")),
        "ipm_top_left": float(ipm.get("top_left")),
        "ipm_top_right": float(ipm.get("top_right")),
        "ipm_bottom_right": float(ipm.get("bottom_right")),
        "ipm_bottom_left": float(ipm.get("bottom_left")),
    }


def elas_point_rows_y(elas_cfg):
    y_bottom = elas_cfg["roi_y"] + elas_cfg["roi_h"]
    y_top = elas_cfg["roi_y"]
    # p1 is the FARTHEST point (top of the ROI) and p4 the NEAREST (bottom
    # of the ROI) -- confirmed empirically: lane width computed with the
    # old order grew with distance (physically impossible for a
    # perspective projection); with the linspace reversed it decreases,
    # as it should.
    return np.linspace(y_top, y_bottom, 4)


def parse_elas_groundtruth(gt_xml_path, elas_cfg):
    row_ys = elas_point_rows_y(elas_cfg)
    tree = ET.parse(gt_xml_path)
    root = tree.getroot()
    frames_el = root.find("frames")
    frames = {}
    for frame_el in frames_el.findall("frame"):
        frame_id = int(frame_el.get("id"))
        pos = frame_el.find("position")
        entry = {"left": None, "right": None}
        for side in ("left", "right"):
            side_el = pos.find(side) if pos is not None else None
            if side_el is None:
                continue
            xs = []
            for tag in ("p1", "p2", "p3", "p4"):
                node = side_el.find(tag)
                if node is None or node.text is None:
                    xs.append(None); continue
                val = float(node.text)
                xs.append(val if val >= 0 else None)
            pts = [(x, y) for x, y in zip(xs, row_ys) if x is not None]
            if len(pts) >= 2:
                entry[side] = np.array(pts, dtype=np.float32)
        frames[frame_id] = entry
    return frames


_IMAGE_NAME_PATTERNS = [
    "{i}.png", "{i}.jpg", "{i}.jpeg",
    "lane_{i}.png", "lane_{i}.jpg",
    "{i:04d}.png", "{i:04d}.jpg",
    "{i:05d}.png", "{i:05d}.jpg",
    "{i:06d}.png", "{i:06d}.jpg",
    "frame_{i}.png", "frame_{i:04d}.png", "frame_{i:05d}.png",
]
_resolved_image_pattern_by_root = {}


def resolve_image_path(images_root, frame_id):
    """BUGFIX: a single global `_resolved_image_pattern` cached whichever
    filename pattern the FIRST scene resolved and then applied it to every
    other scene too, even if that scene actually uses a different naming
    convention -- silently returning None (missing frame) for any scene
    whose pattern doesn't match the first one's. Cached per `images_root`
    instead, so each scene's own resolved pattern is remembered
    independently."""
    root_key = str(images_root)
    cached = _resolved_image_pattern_by_root.get(root_key)
    if cached is not None:
        candidate = images_root / cached.format(i=frame_id)
        if candidate.exists():
            return candidate
    for pattern in _IMAGE_NAME_PATTERNS:
        candidate = images_root / pattern.format(i=frame_id)
        if candidate.exists():
            _resolved_image_pattern_by_root[root_key] = pattern
            return candidate
    return None


def build_elas_rows(images_root, gt_frames, frame_ids, scene_name, scene_cfg, tags):
    rows = []
    for fid in frame_ids:
        img_path = resolve_image_path(images_root, fid)
        if img_path is None:
            continue
        gt = gt_frames.get(fid)
        if gt is None or (gt["left"] is None and gt["right"] is None):
            continue
        rows.append({
            "scene": scene_name, "frame_id": fid, "image_path": img_path,
            "scene_cfg": scene_cfg, "tags": tags,
        })
    return rows


def _allocate_quota(pool_sizes, weights, total):
    selected = {name: 0 for name in pool_sizes}
    total = min(total, sum(pool_sizes.values()))
    for _ in range(total):
        candidates = [name for name, size in pool_sizes.items()
                      if selected[name] < size]
        if not candidates:
            break
        name = max(candidates, key=lambda item: weights.get(item, 1) / (selected[item] + 1))
        selected[name] += 1
    return selected


def prepare_scene_splits(dataset_root, scene_plan, split_sizes, seed):
    rng = random.Random(seed)
    split_names = tuple(split_sizes)
    pools = {name: {split: [] for split in split_names} for name in scene_plan}
    scene_records = {}
    missing = []

    for scene_name, plan in scene_plan.items():
        config_path, gt_path, image_root = discover_scene_paths(
            dataset_root, scene_name
        )

        if config_path is None or gt_path is None or image_root is None:
            missing.append(scene_name)
            continue

        scene_cfg = parse_elas_config(config_path)
        gt_frames = parse_elas_groundtruth(gt_path, scene_cfg)
        frame_ids = sorted(gt_frames)

        rows = build_elas_rows(
            image_root,
            gt_frames,
            frame_ids,
            scene_name,
            scene_cfg,
            plan["tags"],
        )

        if not rows:
            missing.append(scene_name)
            continue

        blocks = [
            rows[i:i + LEAKAGE_BLOCK_SIZE]
            for i in range(0, len(rows), LEAKAGE_BLOCK_SIZE)
        ]

        assignment_order = list(range(len(blocks)))
        rng.shuffle(assignment_order)

        for block_index, block in enumerate(blocks):
            split = split_names[
                assignment_order[block_index] % len(split_names)
            ]

            start = LEAKAGE_PURGE_FRAMES if block_index > 0 else 0
            end = (
                -LEAKAGE_PURGE_FRAMES
                if block_index < len(blocks) - 1
                else None
            )

            pools[scene_name][split].extend(block[start:end])

        scene_records[scene_name] = {
            "config": config_path,
            "gt_frames": gt_frames,
            "image_root": image_root,
        }

    selected = {split: [] for split in split_names}

    for split, requested in split_sizes.items():
        available = {
            scene: len(pools[scene][split])
            for scene in scene_records
        }

        weights = {
            scene: scene_plan[scene]["weight"]
            for scene in scene_records
        }

        total_available = sum(available.values())

        if total_available < requested:
            print(
                f"[WARN] {split}: only {total_available} "
                f"leakage-safe rows available; requested {requested}. "
                f"Using all available rows."
            )
            requested = total_available

        quotas = _allocate_quota(
            available,
            weights,
            requested,
        )

        for scene, quota in quotas.items():
            candidates = list(pools[scene][split])
            rng.shuffle(candidates)
            selected[split].extend(candidates[:quota])

        rng.shuffle(selected[split])

    return selected, scene_records, missing


# ============================================================
# 2. Config  (aspect-correct, denser anchors, larger grid)
# ============================================================

class Config:
    pass


cfg = Config()

cfg.device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ELAS source is 640x480 (4:3). Keep aspect. 512x384 is half-res.
cfg.img_h = 384
cfg.img_w = 512

cfg.default_orig_w = 640
cfg.default_orig_h = 480

cfg.num_lanes = 2
cfg.num_frames = 3
cfg.elas_temporal_step = 2

# grid_size must be <= feature-map width. With img_w=512 and 4 stride-2
# blocks, feature width = 512/16 = 32. We pool the head to 64 wide (2x
# upsample) which is a good trade-off between resolution and cost.
cfg.grid_size = 64

# Dense row anchors across the whole ROI. ELAS ROI spans roughly
# y_orig in [180, 400] => model y in [144, 320] for img_h=384.
# linspace over [0.52*H, 0.99*H] covers that with margin (fixed comment:
# previously stated [0.28*H, 0.98*H], which did not match the call below).
cfg.row_anchors = np.linspace(0.52 * cfg.img_h, 0.99 * cfg.img_h, 48).astype(np.float32)
cfg.num_row_anchors = len(cfg.row_anchors)

cfg.batch_size = 16
cfg.num_workers = 4

cfg.epochs_baseline = SMOKE_EPOCHS
cfg.epochs_v02 = SMOKE_EPOCHS
cfg.epochs_v03 = SMOKE_EPOCHS
cfg.epochs_v04 = SMOKE_EPOCHS

cfg.lr = 5e-5
cfg.weight_decay = 5e-4

# ---- augmentation ----
cfg.augmentation_enabled = True
cfg.aug_prob = 0.80
cfg.aug_brightness = 0.18
cfg.aug_contrast = 0.18
cfg.aug_saturation = 0.15
cfg.aug_noise_std = 0.015
cfg.aug_blur_prob = 0.10
cfg.aug_erasing_prob = 0.15
cfg.aug_erasing_scale = (0.02, 0.08)
cfg.temporal_frame_dropout_prob = 0.10

cfg.backbone_dropout = 0.05
cfg.head_dropout1 = 0.15
cfg.head_dropout2 = 0.10
cfg.label_smoothing = 0.05
cfg.grad_clip_norm = 1.0

# ---- loss weights ----
cfg.lambda_exist = 1.0
cfg.lambda_fusion_prior = 0.02
cfg.lambda_temporal = 0.50
cfg.lambda_coord = 35.0     # direct x regression is the main signal
cfg.lambda_y = 0.0        # y offset is secondary
cfg.lambda_flow_smooth = 0.01
cfg.lambda_gate = 0.01     # kept for backwards-compat in v03 loop

# ---- UFLD-faithful row-classification loss weights (this revision) ----
# See LanePixelHead's and main_loss's docstrings: these two terms are the
# actual Ultra-Fast-Lane-Detection-style loss (full (grid_size+1)-way row
# classification + adjacent-row structural/similarity loss), added
# alongside the pre-existing regression-style terms above rather than
# replacing them. Starting points from the UFLD paper's own loss
# weighting -- recalibrate empirically, do not assume these transfer
# unchanged to ELAS's row-anchor density/grid_size.
cfg.lambda_ufld_ce = 1.0
cfg.lambda_structural = 0.10

cfg.exist_threshold = 0.35

cfg.early_stopping_patience = 8
cfg.early_stopping_min_delta = 1e-4
cfg.diag_gap_ratio = 1.5

# ---- visualization / experiment-tracking switches (this revision) ----
# All of these are read with getattr(cfg, name, default) at the call
# sites, so existing code that constructs a bare Config() and only sets
# the fields it cares about (e.g. inside Optuna trials, see
# train_and_eval_variant_trial's `cfg_trial = copy.deepcopy(cfg_base)`)
# keeps working without having to also set every new switch.
cfg.tensorboard_enabled = True
# Scalars are cheap -- log every epoch. Images/graph/video are not, so
# they get their own frequency knobs instead of also using this one.
cfg.tb_log_batch_scalars_every_n_batches = 50
cfg.visualize_every_n_epochs = 5          # 0 disables per-epoch image logging
cfg.max_visualization_samples = 6         # fixed validation subset size
cfg.visualization_seed = 12345            # fixed sampling, independent of cfg.seed
cfg.tb_log_graph = True                   # log computational graph once per model
cfg.temporal_video_fps = 8
cfg.temporal_video_max_frames = 60
cfg.error_inspection_max_examples = 12    # per FP/FN/temporal-help/temporal-hurt bucket

cfg.seed = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(cfg.seed)


# ============================================================
# 2a. ADAS embedded-deployment targets
# ============================================================
#
# These are not tunable hyperparameters -- they encode the actual
# deployment requirement the "beat the baseline" comparison exists to
# serve. A model that wins on polyline_pixel_f1 but blows the frame
# budget, or that is accurate-but-jittery frame to frame, is not a
# usable ADAS candidate. Both are checked and reported alongside the
# accuracy metrics rather than folded into model selection during
# training, so the person reviewing results sees the trade-off instead
# of having it hidden by a single scalar.

ADAS_TARGET_FPS = 30.0
ADAS_LATENCY_BUDGET_MS = 1000.0 / ADAS_TARGET_FPS

# F-beta weighting used for the "safety" score: beta > 1 weights recall
# (missed lane markings) more heavily than precision (spurious lane
# markings). A false negative silently disables lane keeping for that
# frame; a false positive is usually caught by consistency checks
# downstream. beta=2.0 is a starting point, not a validated constant --
# tune it against whatever downstream controller/consistency-filter
# behavior the ADAS stack actually has.
ADAS_SAFETY_FBETA = 2.0

# TRL3/TRL4 quantitative acceptance thresholds -- see header docstring for
# rationale/provenance. "higher_is_better=False" metrics (jitter) are
# checked as achieved <= target; all others as achieved >= target.
ADAS_READINESS_TARGETS = {
    "polyline_pixel_f1":       {"trl3": 0.75, "trl4": 0.90, "higher_is_better": True},
    "iou_f1":                  {"trl3": 0.70, "trl4": 0.85, "higher_is_better": True},
    "polyline_pixel_f2_safety": {"trl3": 0.75, "trl4": 0.90, "higher_is_better": True},
    "lane_position_jitter_px": {"trl3": 8.0,  "trl4": 4.0,  "higher_is_better": False},
}


def print_adas_readiness_report(results_df, out_csv_path=None):
    """Compare achieved per-model metrics against ADAS_READINESS_TARGETS
    and print/export a PASS/FAIL table for TRL3 and TRL4. This is the
    go/no-go table referenced by the header docstring; it does not affect
    training or model selection."""
    rows = []
    print("\n" + "=" * 60)
    print("ADAS READINESS REPORT (TRL3 / TRL4 gate)")
    print("=" * 60)
    for _, r in results_df.iterrows():
        model = r["model"]
        for metric, spec in ADAS_READINESS_TARGETS.items():
            if metric not in r or pd.isna(r[metric]):
                continue
            achieved = float(r[metric])
            hib = spec["higher_is_better"]
            trl3_pass = achieved >= spec["trl3"] if hib else achieved <= spec["trl3"]
            trl4_pass = achieved >= spec["trl4"] if hib else achieved <= spec["trl4"]
            rows.append({
                "model": model, "metric": metric, "achieved": achieved,
                "trl3_target": spec["trl3"], "trl3_pass": trl3_pass,
                "trl4_target": spec["trl4"], "trl4_pass": trl4_pass,
            })
            print(f"{model:28s} {metric:26s} achieved={achieved:8.4f}  "
                  f"TRL3[{'PASS' if trl3_pass else 'FAIL'}]  "
                  f"TRL4[{'PASS' if trl4_pass else 'FAIL'}]")
    report_df = pd.DataFrame(rows)
    if out_csv_path is not None and not report_df.empty:
        report_df.to_csv(out_csv_path, index=False)
    return report_df


# ============================================================
# 3. Image helpers + target encoding
# ============================================================

def load_image_tensor(image_path, cfg):
    image = PILImage.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    image_resized = image.resize((cfg.img_w, cfg.img_h), resample=PILImage.BILINEAR)
    arr = np.asarray(image_resized).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    tensor = torch.tensor(arr, dtype=torch.float32)
    return tensor, orig_w, orig_h


def interpolate_lane_x_at_y(lane_points, y_query):
    pts = np.asarray(lane_points, dtype=np.float32)
    if len(pts) < 2:
        return None
    pts = pts[np.argsort(pts[:, 1])]
    x, y = pts[:, 0], pts[:, 1]
    if y_query < y.min() or y_query > y.max():
        return None
    try:
        return float(np.interp(y_query, y, x))
    except Exception:
        return None


def lanes_to_targets(lanes, orig_w, orig_h, cfg):
    """Return per-(lane, anchor) targets in model pixels.
    cls_target   : (L, A)      int64   bin index (auxiliary CE)
    exist_target : (L, A)      float32 0/1
    coords_target: (L, A, 2)   float32 (x_model_px, y_model_px)

    BUGFIX (this revision): non-existing anchors used to default
    cls_target to 0 -- the SAME index as a real "leftmost bin" class. That
    was harmless for the pre-existing masked CE (main_loss's `ce`, which
    only ever reads cls_target where exist_target>0.5), but it is wrong
    for the new UFLD-faithful full-grid classification loss added in
    main_loss (`ufld_ce`), which trains on EVERY anchor -- including the
    non-existing ones -- against an explicit "no lane" class. Non-existing
    anchors now default to `cfg.grid_size`, i.e. the (grid_size+1)-th
    class LanePixelHead's cls_head reserves for "no lane at this anchor",
    matching Ultra-Fast-Lane-Detection's own row-classification
    formulation instead of overloading a real location bin as a stand-in
    for absence.
    """
    L = cfg.num_lanes
    A = cfg.num_row_anchors

    cls_target = np.full((L, A), cfg.grid_size, dtype=np.int64)
    exist_target = np.zeros((L, A), dtype=np.float32)
    coords_target = np.zeros((L, A, 2), dtype=np.float32)

    lane_infos = []
    for lane in lanes:
        if len(lane) < 2:
            continue
        mean_x = float(np.mean(lane[:, 0]))
        lane_infos.append((mean_x, lane))
    lane_infos = sorted(lane_infos, key=lambda x: x[0])
    selected = lane_infos[:L]

    for lane_id, (_, lane) in enumerate(selected):
        for a, y_model in enumerate(cfg.row_anchors):
            y_orig = y_model * (orig_h / cfg.img_h)
            x_orig = interpolate_lane_x_at_y(lane, y_orig)
            if x_orig is None or x_orig < 0 or x_orig >= orig_w:
                continue
            x_model = float(np.clip(x_orig * (cfg.img_w / orig_w), 0, cfg.img_w - 1))
            bin_id = int(round(x_model / (cfg.img_w - 1) * (cfg.grid_size - 1)))
            bin_id = int(np.clip(bin_id, 0, cfg.grid_size - 1))
            cls_target[lane_id, a] = bin_id
            exist_target[lane_id, a] = 1.0
            coords_target[lane_id, a, 0] = x_model
            coords_target[lane_id, a, 1] = y_model
    return cls_target, exist_target, coords_target


# ============================================================
# 4. ELAS temporal dataset
# ============================================================

_missing_frame_warned = set()


class ELASTemporalDataset(Dataset):
    def __init__(self, rows, scene_records, cfg, augment=False):
        self.rows = rows
        self.scene_records = scene_records
        self.cfg = cfg
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def get_temporal_paths(self, row):
        """BUGFIX: a missing historical frame silently fell back to the
        current frame with no record of it happening, which for a scene
        with many gaps could quietly degrade v02/v03/v04's temporal signal
        (feeding the same frame 2-3 times instead of real history) without
        ever showing up anywhere. Now reported once per (scene, frame_id)
        via the module-level `_missing_frame_warned` set, so repeated
        __getitem__ calls on the same sample (every epoch) don't spam the
        console."""
        frame_id = row["frame_id"]
        images_root = row["image_path"].parent
        step = self.cfg.elas_temporal_step
        frame_ids = [
            max(frame_id - step * (self.cfg.num_frames - 1 - i), 0)
            for i in range(self.cfg.num_frames)
        ]
        paths = []
        for fid in frame_ids:
            p = resolve_image_path(images_root, fid)
            if p is None:
                key = (row["scene"], fid)
                if key not in _missing_frame_warned:
                    _missing_frame_warned.add(key)
                    print(f"[WARN] could not resolve frame {fid} in scene "
                          f"{row['scene']} -- falling back to current frame "
                          f"(temporal signal degraded for this sample)")
                p = row["image_path"]
            paths.append(p)
        return paths

    def _augment_sequence(self, images):
        if not self.augment or not self.cfg.augmentation_enabled:
            return images
        if random.random() > self.cfg.aug_prob:
            return images

        brightness = 1.0 + random.uniform(-self.cfg.aug_brightness, self.cfg.aug_brightness)
        contrast = 1.0 + random.uniform(-self.cfg.aug_contrast, self.cfg.aug_contrast)
        saturation = 1.0 + random.uniform(-self.cfg.aug_saturation, self.cfg.aug_saturation)

        out = []
        for frame in images:
            mean = frame.mean(dim=(1, 2), keepdim=True)
            frame = (frame - mean) * contrast + mean
            gray = frame.mean(dim=0, keepdim=True)
            frame = gray + (frame - gray) * saturation
            frame = frame * brightness
            if self.cfg.aug_noise_std > 0:
                frame = frame + torch.randn_like(frame) * self.cfg.aug_noise_std
            out.append(frame.clamp(0.0, 1.0))
        images = torch.stack(out, dim=0)

        if random.random() < self.cfg.aug_erasing_prob:
            _, _, H, W = images.shape
            area = H * W
            target_area = random.uniform(*self.cfg.aug_erasing_scale) * area
            aspect = random.uniform(0.5, 2.0)
            eh = max(1, min(H, int((target_area * aspect) ** 0.5)))
            ew = max(1, min(W, int((target_area / aspect) ** 0.5)))
            y0 = random.randint(0, max(H - eh, 0))
            x0 = random.randint(0, max(W - ew, 0))
            images[:, :, y0:y0 + eh, x0:x0 + ew] = torch.rand(
                images.shape[0], images.shape[1], eh, ew, dtype=images.dtype
            )

        if self.cfg.temporal_frame_dropout_prob > 0 and random.random() < self.cfg.temporal_frame_dropout_prob:
            if images.shape[0] > 1:
                drop_idx = random.randrange(images.shape[0] - 1)
                images[drop_idx] = images[-1].clone()

        return images

    def __getitem__(self, idx):
        row = self.rows[idx]
        frame_id, image_path = row["frame_id"], row["image_path"]

        temporal_paths = self.get_temporal_paths(row)
        frames = []
        orig_w, orig_h = self.cfg.default_orig_w, self.cfg.default_orig_h

        for p in temporal_paths:
            img_tensor, ow, oh = load_image_tensor(p, self.cfg)
            frames.append(img_tensor)
            if str(p) == str(image_path):
                orig_w, orig_h = ow, oh

        images = torch.stack(frames, dim=0)
        images = self._augment_sequence(images)

        gt = self.scene_records[row["scene"]]["gt_frames"].get(
            frame_id, {"left": None, "right": None}
        )
        lanes = [lane for lane in (gt["left"], gt["right"]) if lane is not None]
        cls_target, exist_target, coords_target = lanes_to_targets(
            lanes, orig_w, orig_h, self.cfg
        )

        return {
            "images": images,
            "cls_target": torch.tensor(cls_target, dtype=torch.long),
            "exist_target": torch.tensor(exist_target, dtype=torch.float32),
            "coords_target": torch.tensor(coords_target, dtype=torch.float32),
            "frame_id": frame_id,
            "scene": row["scene"],
            "image_path": str(image_path),
            "orig_w": orig_w,
            "orig_h": orig_h,
        }


# ============================================================
# 5. Models
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConvBlock(nn.Module):
    """MobileNet-style depthwise-separable 3x3 block: one 3x3 conv per
    input channel (depthwise) followed by a 1x1 conv that mixes channels
    (pointwise). For the same (Cin, Cout) this costs roughly
    Cout/(9+Cout) times a standard ConvBlock's 3x3 dense convolution --
    used here specifically to keep TinyHistoryEncoder (below) cheap
    enough that adding temporal history frames does not undo the savings
    from LightweightBackboneTemporal's halved channel width. Not used in
    the baseline or in the temporal models' current-frame backbone, so it
    has no effect on the single-frame UFLD baseline's fidelity."""
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class LightweightBackbone(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128, dropout=0.05):
        super().__init__()
        self.out_channels = feature_dim
        self.net = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2, dropout=dropout),
            ConvBlock(32, 64, stride=2, dropout=dropout),
            ConvBlock(64, 96, stride=2, dropout=dropout),
            ConvBlock(96, feature_dim, stride=2, dropout=dropout),
        )

    def forward(self, x):
        return self.net(x)


class LightweightBackboneTemporal(nn.Module):
    """Half-width variant of LightweightBackbone (16/32/48/64 channels
    instead of 32/64/96/128), used ONLY as the CURRENT-frame encoder
    inside the temporal models (TACUFLDTemporalModel and its v03 gated
    subclass) -- never for the single-frame baseline, which keeps the
    full-width LightweightBackbone above so it remains a faithful,
    unmodified UFLD-style reference point.

    Rationale (this revision's "make v02/v03/v04 lighter than baseline"
    change): the single-frame baseline has to extract everything it
    knows about lane position from ONE image, so it needs the full-width
    encoder. A temporal model additionally sees T-1 historical frames
    (through TinyHistoryEncoder + ResidualTemporalFusion below), so it
    can recover comparable accuracy from a narrower current-frame
    encoder while still coming out lighter overall than the baseline --
    see compute_model_efficiency()/model_efficiency.csv to confirm this
    holds for your actual trained weights; the backbone's convolutions
    dominate total MACs by roughly 7:1 over the head at this
    img_h/img_w/grid_size, so halving every backbone channel count is the
    single highest-leverage lever for cutting a temporal model's cost,
    well above anything achievable by shrinking the fusion module alone.
    """
    def __init__(self, in_channels=3, dropout=0.05):
        super().__init__()
        self.out_channels = 64
        self.net = nn.Sequential(
            ConvBlock(in_channels, 16, stride=2, dropout=dropout),
            ConvBlock(16, 32, stride=2, dropout=dropout),
            ConvBlock(32, 48, stride=2, dropout=dropout),
            ConvBlock(48, self.out_channels, stride=2, dropout=dropout),
        )

    def forward(self, x):
        return self.net(x)


class TinyHistoryEncoder(nn.Module):
    """Very cheap per-frame feature extractor used ONLY for the T-1
    HISTORICAL frames inside the temporal models -- never for the current
    frame (which goes through LightweightBackboneTemporal) and never for
    the baseline. Four depthwise-separable blocks at a narrow width (mid)
    reach the same output shape as LightweightBackboneTemporal at roughly
    1/6-1/8 the MACs, so that even after adding TWO of these (num_frames=3
    means 2 history frames) plus the fusion module, a temporal model's
    total compute stays below LightweightBackbone alone (the baseline's
    entire backbone budget) -- see compute_model_efficiency() to verify
    against your actual cfg.num_frames / img_h / img_w."""
    def __init__(self, in_channels=3, out_channels=64, mid=12, dropout=0.05):
        super().__init__()
        self.out_channels = out_channels
        self.net = nn.Sequential(
            DepthwiseSeparableConvBlock(in_channels, mid, stride=2, dropout=dropout),
            DepthwiseSeparableConvBlock(mid, mid * 2, stride=2, dropout=dropout),
            DepthwiseSeparableConvBlock(mid * 2, mid * 2, stride=2, dropout=dropout),
            DepthwiseSeparableConvBlock(mid * 2, out_channels, stride=2, dropout=dropout),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
# 5a. Row-anchor classification head (UFLD-faithful)
# ------------------------------------------------------------------

class LanePixelHead(nn.Module):
    """Row-anchor classification head, faithful to the localization
    formulation in Ultra-Fast-Lane-Detection (Qin et al., ECCV 2020).

    BUGFIX / REDESIGN (this revision): the previous version decoded lane
    position from an independently-learned sigmoid regression head
    (`x_head`), with a separate `exist_head` for presence and `bin_head`
    used only as an auxiliary, positives-only classification loss. That
    combination has no counterpart in the original UFLD paper -- UFLD
    decodes BOTH existence and location from ONE (grid_size+1)-way
    softmax per (lane, row anchor): `grid_size` candidate x-location bins
    plus one explicit "no lane at this anchor" bin, with location decoded
    via an expectation (soft-argmax) over the location bins and existence
    decoded as 1 minus the "no lane" bin's probability. That is what this
    class now computes: `cls_head` replaces the previous `exist_head` +
    `x_head` pair, and everything downstream is DERIVED from its single
    softmax distribution.

    Output dict keys are UNCHANGED from the previous version
    (exist_logits, x_norm, y_off, bin_logits), so every consumer --
    decode_outputs, compute_anchor_metrics, evaluate_anchor,
    build_prediction_cache, evaluate_iou_cache, the TensorBoard logging,
    every plotting/visualization function -- keeps working exactly as
    before with no changes required there. Only HOW those tensors are
    computed changed. `grid_logits_full` ((B,L,A,grid_size+1)) is exposed
    as one additional key, read only by main_loss's new `ufld_ce` +
    `structural_loss` terms (see its docstring) and by nothing else.

    `y_off` is kept (a tiny, near-zero-initialised head) purely for
    backward compatibility with call sites that read `outputs["y_off"]`
    unconditionally (e.g. main_loss) -- classic UFLD fixes y at the row
    anchor itself and does not regress it, which is why cfg.lambda_y = 0
    already zeroes this term's contribution to the loss.
    """
    def __init__(self, feature_dim=128, num_lanes=2, num_row_anchors=32,
                 grid_size=64, mid=64, hidden=128, dropout=0.15):
        super().__init__()
        self.L = num_lanes
        self.A = num_row_anchors
        self.G = grid_size
        self.mid = mid

        self.pre = nn.Sequential(
            nn.Conv2d(feature_dim, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((num_row_anchors, grid_size))
        self.lane_proj = nn.Conv2d(mid, num_lanes * mid, 1)

        in_dim = mid * grid_size
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        # UFLD-style unified classification: grid_size location bins + 1
        # explicit "no lane" bin. Replaces the previous separate
        # exist_head + x_head pair.
        self.cls_head = nn.Linear(hidden, grid_size + 1)
        self.y_head = nn.Linear(hidden, 1)  # see class docstring

        nn.init.zeros_(self.y_head.weight)
        nn.init.constant_(self.y_head.bias, 0.0)
        # Bias the background ("no lane") logit above the location bins at
        # init, so the model starts conservative (predicting "no lane"
        # almost everywhere) -- the same intent the previous version's
        # `exist_head` bias=-1.0 initialisation had.
        nn.init.zeros_(self.cls_head.weight)
        with torch.no_grad():
            self.cls_head.bias.zero_()
            self.cls_head.bias[-1] = 1.0

    def forward(self, feat):
        B = feat.shape[0]
        h = self.pre(feat)                              # (B, mid, Hf, Wf)
        h = self.pool(h)                                # (B, mid, A, G)
        h = self.lane_proj(h)                           # (B, L*mid, A, G)
        h = h.reshape(B, self.L, self.mid, self.A, self.G)
        h = h.permute(0, 1, 3, 2, 4).contiguous()       # (B, L, A, mid, G)
        h = h.reshape(B * self.L * self.A, self.mid * self.G)

        z = self.mlp(h)                                 # (B*L*A, hidden)

        grid_logits = self.cls_head(z).reshape(B, self.L, self.A, self.G + 1)
        y_off = torch.tanh(self.y_head(z)).reshape(B, self.L, self.A)

        probs = F.softmax(grid_logits, dim=-1)
        exist_prob = (1.0 - probs[..., -1]).clamp(1e-4, 1 - 1e-4)
        exist_logits = torch.logit(exist_prob, eps=1e-4)

        bin_index = torch.arange(self.G, device=feat.device, dtype=probs.dtype)
        fg_probs = probs[..., :self.G]
        fg_probs_norm = fg_probs / fg_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        expected_bin = (fg_probs_norm * bin_index).sum(dim=-1)   # (B,L,A) in [0, G-1]
        x_norm = (expected_bin / max(self.G - 1, 1)).clamp(0, 1)

        return {
            "exist_logits": exist_logits,
            "x_norm":       x_norm,
            "y_off":        y_off,
            "bin_logits":   grid_logits[..., :self.G],
            "grid_logits_full": grid_logits,
        }


def _build_backbone(cfg):
    """Full-width backbone -- the single-frame UFLD baseline ONLY."""
    return LightweightBackbone(3, 128, dropout=cfg.backbone_dropout)


def _build_temporal_backbone(cfg):
    """Half-width current-frame backbone for the temporal models
    (v02/v03/v04). See LightweightBackboneTemporal's docstring."""
    return LightweightBackboneTemporal(3, dropout=cfg.backbone_dropout)


def _build_head(cfg, feature_dim=128):
    return LanePixelHead(
        feature_dim=feature_dim,
        num_lanes=cfg.num_lanes,
        num_row_anchors=cfg.num_row_anchors,
        grid_size=cfg.grid_size,
        mid=96,
        hidden=192,
        dropout=cfg.head_dropout1,
    )


class SingleFrameUFLDLikeModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = _build_backbone(cfg)
        self.head = _build_head(cfg, feature_dim=self.backbone.out_channels)

    def forward(self, x):
        # Accept either (B, C, H, W) or (B, T, C, H, W); take last frame.
        if x.dim() == 5:
            x = x[:, -1]
        return self.head(self.backbone(x))

# ------------------------------------------------------------------
# 5b. Temporal fusion with learnable flow warp
# ------------------------------------------------------------------

class LearnableFlowWarp(nn.Module):
    """Per-pixel 2D offset between (t-1) feature and (t) feature, used to
    warp (t-1) onto (t)."""
    def __init__(self, channels=128, max_disp=8, hidden=64):
        super().__init__()
        self.max_disp = max_disp
        self.flow_net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden // 2), nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, 2, 3, padding=1),
        )

    def forward(self, feat_prev, feat_cur):
        # feat_prev, feat_cur : (B, C, Hf, Wf)
        B, C, Hf, Wf = feat_prev.shape
        flow = torch.tanh(self.flow_net(torch.cat([feat_prev, feat_cur], dim=1)))
        flow = flow * self.max_disp                       # (B, 2, Hf, Wf)

        ys, xs = torch.meshgrid(
            torch.arange(Hf, device=feat_prev.device, dtype=feat_prev.dtype),
            torch.arange(Wf, device=feat_prev.device, dtype=feat_prev.dtype),
            indexing="ij",
        )
        base = torch.stack([xs, ys], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        sample_xy = base + flow                           # (B, 2, Hf, Wf)

        # normalise to [-1, 1]
        sample_x = sample_xy[:, 0] / max(Wf - 1, 1) * 2 - 1
        sample_y = sample_xy[:, 1] / max(Hf - 1, 1) * 2 - 1
        grid = torch.stack([sample_x, sample_y], dim=-1)  # (B, Hf, Wf, 2)

        warped = F.grid_sample(
            feat_prev, grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        )
        return warped, flow


class ResidualTemporalFusion(nn.Module):
    """fused = f_cur + gate * (hist_aggregate - f_cur)
    gate is per-pixel and starts near 0 (closed).
    hist_aggregate is the average of history frames warped to f_cur.
    """
    def __init__(self, channels=128, num_frames=3, hidden=64, max_disp=8):
        super().__init__()
        self.num_frames = num_frames
        self.channels = channels
        self.warp = LearnableFlowWarp(channels, max_disp=max_disp, hidden=hidden)

        # aggregator: (T-1) * C -> C
        self.history_conv = nn.Sequential(
            nn.Conv2d(channels * max(num_frames - 1, 1), channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )

        self.gate_net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.constant_(self.gate_net[-1].bias, -2.0)

    def forward(self, features):
        # features : (B, T, C, Hf, Wf), index 0 oldest, index -1 = current
        B, T, C, Hf, Wf = features.shape
        f_cur = features[:, -1]                            # (B, C, Hf, Wf)

        warped = []
        flows = []
        for k in range(T - 1):
            w_k, flow_k = self.warp(features[:, k], f_cur)
            warped.append(w_k)
            flows.append(flow_k)

        hist = self.history_conv(torch.cat(warped, dim=1))  # (B, C, Hf, Wf)

        gate = torch.sigmoid(
            self.gate_net(torch.cat([f_cur, hist], dim=1))
        )                                                   # (B, 1, Hf, Wf)

        fused = f_cur + gate * (hist - f_cur)               # (B, C, Hf, Wf)
        return fused, gate, flows


class TACUFLDTemporalModel(nn.Module):
    """Unified model for v02 / v03 / v04 (the baseline uses
    SingleFrameUFLDLikeModel instead). `gate_bias_init` controls whether
    the gate starts closed (default) or slightly open (v03 variant).

    REDESIGN (this revision, "make v02/v03/v04 lighter than the
    baseline"): the CURRENT frame goes through LightweightBackboneTemporal
    (half-width relative to the baseline's LightweightBackbone), while
    the T-1 HISTORICAL frames go through TinyHistoryEncoder (a much
    cheaper depthwise-separable encoder producing the same channel count).
    Previously every one of the T frames -- current and history alike --
    went through the SAME full-width backbone as the baseline, so a
    temporal model's total compute was ~T times the baseline's backbone
    cost alone (T=3 by default) BEFORE even adding the fusion module on
    top -- there was no way for a temporal variant to end up lighter than
    the baseline with that design. Asymmetric per-frame cost (expensive
    for the one frame that has to carry the most signal, cheap for the
    frames that only contribute motion context) is what makes "more
    temporal information, less total compute than a wider single-frame
    model" possible at all. See compute_model_efficiency() /
    model_efficiency.csv to confirm the actual numbers for your trained
    configuration -- the margin depends on cfg.num_frames, img_h, img_w
    and grid_size, not on this design choice alone.
    """
    def __init__(self, cfg, gate_bias_init=-2.0):
        super().__init__()
        self.num_frames = cfg.num_frames
        self.backbone = _build_temporal_backbone(cfg)
        self.history_encoder = TinyHistoryEncoder(
            in_channels=3, out_channels=self.backbone.out_channels,
        )
        self.temporal_fusion = ResidualTemporalFusion(
            channels=self.backbone.out_channels, num_frames=cfg.num_frames,
            hidden=32, max_disp=8,
        )
        # override gate bias for variant
        with torch.no_grad():
            self.temporal_fusion.gate_net[-1].bias.fill_(gate_bias_init)
        self.head = _build_head(cfg, feature_dim=self.backbone.out_channels)
        self.ablation_mode = None

    def _fuse_features(self, feat):
        """feat : (B, T, C, Hf, Wf) -> fused, gates, flows"""
        if self.ablation_mode == "last_frame":
            B, T, C, Hf, Wf = feat.shape
            fused = feat[:, -1]
            gate = torch.zeros(B, 1, Hf, Wf, device=feat.device, dtype=feat.dtype)
            flows = []
        elif self.ablation_mode == "uniform_fusion":
            B, T, C, Hf, Wf = feat.shape
            fused = feat.mean(dim=1)
            gate = torch.full((B, 1, Hf, Wf), 1.0 / T,
                              device=feat.device, dtype=feat.dtype)
            flows = []
        elif self.ablation_mode == "no_warp":
            # residual fusion without flow (identity warp = last frame)
            B, T, C, Hf, Wf = feat.shape
            f_cur = feat[:, -1]
            hist = self.temporal_fusion.history_conv(
                torch.cat([feat[:, k] for k in range(T - 1)], dim=1)
            )
            gate = torch.sigmoid(
                self.temporal_fusion.gate_net(torch.cat([f_cur, hist], dim=1))
            )
            fused = f_cur + gate * (hist - f_cur)
            flows = []
        else:
            fused, gate, flows = self.temporal_fusion(feat)
        return fused, gate, flows

    def forward(self, x):
        # x : (B, T, C, H, W), index -1 = current frame.
        B, T, C, H, W = x.shape
        cur = x[:, -1]
        feat_cur = self.backbone(cur)                    # (B, Cf, Hf, Wf) -- full-cost path, once
        Cf, Hf, Wf = feat_cur.shape[1:]

        if T > 1:
            hist = x[:, :-1].reshape(B * (T - 1), C, H, W)
            feat_hist = self.history_encoder(hist).reshape(B, T - 1, Cf, Hf, Wf)
            feat = torch.cat([feat_hist, feat_cur.unsqueeze(1)], dim=1)  # oldest..current
        else:
            feat = feat_cur.unsqueeze(1)

        fused, gate, flows = self._fuse_features(feat)
        out = self.head(fused)

        out["gate"] = gate
        out["flows"] = flows
        # legacy key expected by _evaluate_val_loss
        out["gates"] = gate.unsqueeze(1)  # (B, 1, 1, Hf, Wf) so .mean over dims works
        return out

    def forward_per_frame(self, x, frame_indices=None):
        """Used only by the auxiliary temporal_alignment_loss during
        training (never at inference/deployment, so it is NOT part of the
        efficiency comparison against the baseline). Runs the SAME
        current-frame backbone on each requested frame index -- fine for
        an alignment regularizer comparing two nearby frames processed
        identically, independent of the asymmetric current/history split
        used in forward()."""
        if frame_indices is None:
            frame_indices = range(x.shape[1])
        outputs = []
        for i in frame_indices:
            feat = self.backbone(x[:, i])
            outputs.append(self.head(feat))
        return outputs


# ------------------------------------------------------------------
# 5c. v03 alias: gated model = TACUFLDTemporalModel with different init
# ------------------------------------------------------------------

class TACUFLDGatedTemporalModel(TACUFLDTemporalModel):
    def __init__(self, cfg):
        # gate starts slightly open
        super().__init__(cfg, gate_bias_init=-0.5)


# ============================================================
# 5d. Torchview architecture visualization (Section 2 of the task)
# ============================================================
#
# Optional dependency: torchview is not imported at module load time so
# that the rest of the pipeline still runs (including training) on a
# machine that never installed it. `pip install torchview` to enable.

def _try_import_torchview():
    try:
        import torchview
        return torchview
    except ImportError:
        print(
            "[torchview] not installed -- architecture diagrams will be "
            "skipped. Install with:\n    pip install torchview\n"
            "(torchview itself only needs torch + graphviz; the 'graphviz' "
            "system package -- e.g. `apt-get install graphviz` / "
            "`brew install graphviz` -- is required for PNG/SVG export.)"
        )
        return None


def _dummy_input_for_model(model, cfg, temporal):
    """Build a dummy input matching the model's real forward() signature.
    Both SingleFrameUFLDLikeModel and TACUFLDTemporalModel accept a
    5D (B, T, C, H, W) tensor -- SingleFrameUFLDLikeModel.forward() takes
    the last frame internally (`if x.dim() == 5: x = x[:, -1]`), so a
    single dummy shape covers both without inspecting private attributes.
    `temporal` is accepted for symmetry with the rest of the codebase
    (build_prediction_cache, evaluate_anchor, ...) but is not needed to
    pick the shape, since both model families accept the 5D form."""
    return torch.randn(1, cfg.num_frames, 3, cfg.img_h, cfg.img_w, device=cfg.device)


def _wrap_for_graph_export(model):
    """Wrap a model so its dict output becomes a plain tuple of tensors.
    Both torch.jit tracing (SummaryWriter.add_graph) and torchview need a
    tensor/tuple-of-tensors output, not a dict -- and non-tensor entries
    (e.g. the "flows" list some ablation modes leave empty) are dropped
    by *type*, not by a hardcoded key name (the previous export_arch_images
    hardcoded `o["logits"], o["exist"], o["coords"]`, which don't exist
    anywhere in this model's actual output dict and would have raised
    KeyError the first time it ran). This works unmodified for
    SingleFrameUFLDLikeModel, TACUFLDTemporalModel and
    TACUFLDGatedTemporalModel alike."""
    class _TensorOnlyWrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(x)
            if isinstance(out, dict):
                return tuple(v for v in out.values() if torch.is_tensor(v))
            return out

    return _TensorOnlyWrap(model)


def export_model_architecture(model, cfg, temporal, label, out_dir,
                              ablation_mode=None, writer=None):
    """Generate the architecture diagram for `model` as an actual image
    (PNG + SVG via torchview) and, best-effort, a TensorBoard Graph.

    This replaces the previous generate_architecture_diagram() (a stub
    that set up the Graphviz PATH and then returned an empty list without
    ever calling torchview) and the duplicated export_arch_images()
    definitions (one of which hardcoded output dict keys that don't exist
    on this model and would have crashed on first use). One implementation:
      * builds its own dummy input from cfg via _dummy_input_for_model(),
        so callers don't need to hand it a real batch from a DataLoader;
      * imports torchview lazily via _try_import_torchview() and degrades
        gracefully (still produces the TensorBoard graph) if it is
        missing, instead of raising ImportError;
      * never hardcodes output dict key names -- see
        _wrap_for_graph_export().

    Returns {'torchview': [saved file paths], 'tensorboard': bool}.
    """
    _ensure_graphviz_on_path()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"torchview": [], "tensorboard": False}

    was_training = model.training
    prev_ablation = getattr(model, "ablation_mode", None)
    if hasattr(model, "ablation_mode"):
        model.ablation_mode = ablation_mode
    model.eval()

    dummy = _dummy_input_for_model(model, cfg, temporal)

    try:
        torchview = _try_import_torchview()
        if torchview is not None:
            try:
                torchview.draw_graph(
                    model,
                    input_size=tuple(dummy.shape),
                    device=cfg.device,
                    graph_name=label,
                    depth=4,
                    expand_nested=True,
                    save_graph=True,
                    directory=str(out_dir),
                    filename=f"{label}_architecture",
                )
                for ext in ("png", "svg"):
                    p = out_dir / f"{label}_architecture.{ext}"
                    if p.exists():
                        result["torchview"].append(str(p))
                print(f"[torchview] saved: {result['torchview']}")
            except Exception as exc:
                print(f"[torchview] failed for {label}: {exc}")

        tb_writer = writer
        close_after = False
        if tb_writer is None:
            tb_writer = get_tb_writer(out_dir / "tb" / label)
            close_after = True
        try:
            wrapped = _wrap_for_graph_export(model)
            with torch.no_grad():
                tb_writer.add_graph(wrapped, dummy)
            result["tensorboard"] = True
            print(f"[tensorboard] graph saved for {label} -> {out_dir / 'tb' / label}")
        except Exception as exc:
            print(f"[tensorboard] add_graph failed for {label}: {exc}")
        finally:
            if close_after:
                close_tb_writer(tb_writer)
    finally:
        if hasattr(model, "ablation_mode"):
            model.ablation_mode = prev_ablation
        model.train(was_training)

    return result


def _export_architecture_if_missing(model, cfg, temporal, label,
                                    out_dir=None, ablation_mode=None):
    """Architecture diagrams only depend on module structure, not on
    training seed or learned weights (see ARCHITECTURE_ROOT's docstring
    above: "experiment-independent"), so this generates them once and
    skips regenerating on every subsequent seed via a marker file."""
    out_dir = Path(out_dir) if out_dir is not None else ARCHITECTURE_ROOT
    marker = out_dir / f"{label}_architecture.done"
    if marker.exists():
        return None
    result = export_model_architecture(
        model, cfg, temporal, label, out_dir, ablation_mode=ablation_mode,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")
    return result


# ============================================================
# 6. Losses
# ============================================================

def main_loss(outputs, cls_target, exist_target, coords_target, cfg, mode="baseline"):
    """Unified loss for all variants.

    cls_target    : (B, L, A)      int64   auxiliary bin labels (background
                                            class = cfg.grid_size for
                                            non-existing anchors -- see
                                            lanes_to_targets)
    exist_target  : (B, L, A)      float32 in {0,1}
    coords_target : (B, L, A, 2)   float32 x,y in model pixels

    FIXED (prior revision): every mode -- including "baseline" -- now
    backpropagates into the coordinate signal. All four models share the
    exact same loss formula, so train_loss/validation_loss are directly
    comparable across the whole table.

    REDESIGN (this revision): LanePixelHead's `x_norm`/`exist_logits` are
    now DERIVED from a classification distribution (see its docstring)
    rather than produced by independent regression/existence heads, so
    the pre-existing `ce`/`exist_loss`/`x_loss`/`y_loss` terms below are
    UNCHANGED in shape/semantics and still compute exactly as before --
    their gradients now simply flow back through the softmax/expectation
    into `cls_head` instead of into a separate `x_head`/`exist_head`.
    Two NEW terms are added on top, not in place of the above, to make
    this the actual Ultra-Fast-Lane-Detection loss rather than only a
    same-shaped stand-in for it:
      * `ufld_ce`: cross-entropy over EVERY row anchor (not just the
        masked positive ones the pre-existing `ce` term uses) against the
        full (grid_size+1)-way distribution (`grid_logits_full`), with the
        background class as a real, supervised target for anchors with no
        lane -- exactly UFLD's row-classification loss.
      * `structural_loss`: L1 similarity between the classification
        distributions of ADJACENT row anchors, encouraging a lane's
        predicted location to vary smoothly down the image instead of
        jumping row to row -- UFLD's structural loss, and the ingredient
        that most differentiates it from plain independent per-row
        classification.
    Both are optional (`outputs.get("grid_logits_full")`) so this function
    still works unchanged against a head that does not expose that key.
    """
    exist_logits = outputs["exist_logits"]
    x_norm       = outputs["x_norm"]
    y_off        = outputs["y_off"]
    bin_logits   = outputs["bin_logits"]

    B, L, A = exist_logits.shape
    G = bin_logits.shape[-1]
    mask = exist_target > 0.5
    mask_f = mask.float()
    n_pos = mask_f.sum().clamp(min=1.0)

    # ---- auxiliary bin CE (masked to positives) ----
    bin_logits_flat = bin_logits.reshape(B * L * A, G)
    cls_flat        = cls_target.reshape(B * L * A)
    mask_flat       = mask.reshape(B * L * A)
    if mask_flat.any():
        ce = F.cross_entropy(
            bin_logits_flat[mask_flat], cls_flat[mask_flat],
            reduction="mean",
            label_smoothing=getattr(cfg, "label_smoothing", 0.05),
        )
    else:
        ce = bin_logits_flat.sum() * 0.0

    # ---- focal existence ----
    bce = F.binary_cross_entropy_with_logits(exist_logits, exist_target, reduction="none")
    p = torch.sigmoid(exist_logits)
    p_t = p * exist_target + (1 - p) * (1 - exist_target)
    focal_w = (1.0 - p_t).clamp(min=1e-4) ** 1.5
    exist_loss = (bce * focal_w).sum() / n_pos.clamp(min=1.0)

    # ---- direct x regression ----
    # fp32 under autocast to avoid fp16 underflow on a loss this small, and
    # beta=0.02 in NORMALISED [0,1] coordinates (~10px at img_w=512) keeps
    # the loss in its steeper, more informative near-L1 regime for the
    # error sizes we actually see, instead of the almost-flat quadratic
    # region that beta=1.0 gave for errors that are always << 1.
    with torch.amp.autocast("cuda", enabled=False):
        x_norm_fp32 = x_norm.float()
        x_target_fp32 = (coords_target[..., 0] / max(cfg.img_w - 1, 1)).float()
        if mask.any():
            x_loss = F.smooth_l1_loss(
                x_norm_fp32[mask], x_target_fp32[mask],
                reduction="mean", beta=0.01,
            )
        else:
            x_loss = x_norm_fp32.sum() * 0.0

    # ---- y offset regression (in pixels) ----
    # NOTE: cfg.lambda_y = 0.0 by design. coords_target[...,1] is always
    # exactly the row-anchor position (see lanes_to_targets), so there is
    # nothing for y_off to correct -- its zero-initialised head already
    # outputs the right answer. Still computed/logged in case row anchors
    # ever stop being fixed.
    if mask.any():
        anchors = torch.tensor(cfg.row_anchors, device=y_off.device, dtype=y_off.dtype)
        span = float(cfg.row_anchors[1] - cfg.row_anchors[0])
        anchors = anchors.view(1, 1, A)
        y_pred = anchors + y_off * span
        y_target = coords_target[..., 1]
        y_loss = F.smooth_l1_loss(
            y_pred[mask], y_target[mask], reduction="mean", beta=2.0,
        )
    else:
        y_loss = y_off.sum() * 0.0

    # ---- UFLD-faithful full-grid classification + structural loss ----
    # (this revision) -- see function docstring above.
    grid_logits_full = outputs.get("grid_logits_full")
    if grid_logits_full is not None:
        Bg, Lg, Ag, Gp1 = grid_logits_full.shape
        ufld_ce = F.cross_entropy(
            grid_logits_full.reshape(Bg * Lg * Ag, Gp1),
            cls_target.reshape(Bg * Lg * Ag),
            reduction="mean",
            label_smoothing=getattr(cfg, "label_smoothing", 0.05),
        )
        grid_probs = F.softmax(grid_logits_full, dim=-1)
        structural_loss = (grid_probs[:, :, 1:, :] - grid_probs[:, :, :-1, :]).abs().mean()
    else:
        ufld_ce = ce.new_zeros(())
        structural_loss = ce.new_zeros(())

    total = (
        0.5 * ce
        + cfg.lambda_exist * exist_loss
        + cfg.lambda_coord * x_loss
        + cfg.lambda_y * y_loss
        + getattr(cfg, "lambda_ufld_ce", 1.0) * ufld_ce
        + getattr(cfg, "lambda_structural", 0.1) * structural_loss
    )

    comparable = (ce + 0.5 * bce.mean()).detach()

    return total, {
        "ce_loss":         ce.detach(),
        "exist_loss":      exist_loss.detach(),
        "x_loss":          x_loss.detach(),
        "y_loss":          y_loss.detach(),
        "ufld_ce_loss":    ufld_ce.detach(),
        "structural_loss": structural_loss.detach(),
        "comparable_loss": comparable,
    }

def temporal_alignment_loss(out_cur, out_prev, exist_weight=0.5):
    cls_c = out_cur["bin_logits"]; cls_p = out_prev["bin_logits"]
    ex_c  = out_cur["exist_logits"]; ex_p = out_prev["exist_logits"]
    return F.smooth_l1_loss(cls_c, cls_p) + exist_weight * F.smooth_l1_loss(ex_c, ex_p)


def flow_smoothness_loss(flows):
    if not flows:
        # Device-matched to whatever loss this gets added to (previously a
        # bare CPU torch.tensor(0.0), which would raise a device-mismatch
        # error the moment it was combined with a CUDA loss tensor).
        return torch.tensor(0.0, device=cfg.device)
    total = 0.0
    for flow in flows:
        dx = flow[:, 0:1]
        dy = flow[:, 1:2]
        tv_x = (dx[:, :, :, 1:] - dx[:, :, :, :-1]).abs().mean()
        tv_y = (dy[:, :, 1:, :] - dy[:, :, :-1, :]).abs().mean()
        total = total + (tv_x + tv_y)
    return total / max(len(flows), 1)


def fusion_prior_loss(gate_mean, prior=0.4):
    """Encourage the mean gate to sit near `prior`, so the temporal branch
    is neither fully off nor overpowering."""
    return (gate_mean - prior).pow(2)


# ============================================================
# 7. Decoding and anchor-level evaluation
# ============================================================

@torch.no_grad()
def decode_outputs(outputs, cfg, exist_threshold=None):
    exist_logits = outputs["exist_logits"]
    x_norm       = outputs["x_norm"]
    y_off        = outputs.get("y_off")

    B, L, A = exist_logits.shape
    if exist_threshold is None:
        exist_threshold = cfg.exist_threshold

    pred_prob  = torch.sigmoid(exist_logits)
    pred_exist = (pred_prob >= exist_threshold).float()

    pred_x_px = x_norm * (cfg.img_w - 1)

    anchors = torch.tensor(cfg.row_anchors, device=x_norm.device, dtype=x_norm.dtype)
    anchors = anchors.view(1, 1, A)
    if y_off is not None:
        span = float(cfg.row_anchors[1] - cfg.row_anchors[0])
        pred_y_px = (anchors + y_off * span).clamp(0, cfg.img_h - 1)
    else:
        pred_y_px = anchors.expand(B, L, A)

    return pred_x_px, pred_y_px, pred_prob, pred_exist


@torch.no_grad()
def compute_anchor_metrics(outputs, cls_target, exist_target, cfg):
    """Anchor-level lenient F1: (lane, anchor) is correct if existence is on
    AND the decoded bin is within +-3 of target bin."""
    x_px, y_px, pred_prob, pred_exist = decode_outputs(outputs, cfg)
    pred_bins = (x_px / max(cfg.img_w - 1, 1) * (cfg.grid_size - 1)).round().long()
    pred_bins = pred_bins.clamp(0, cfg.grid_size - 1)

    pred_pos = pred_exist > 0.5
    true_pos = exist_target > 0.5
    bin_error = torch.abs(pred_bins - cls_target.long())
    correct = pred_pos & true_pos & (bin_error <= 1)

    tp = correct.sum().item()
    fp = (pred_pos & (~true_pos)).sum().item()
    fn = ((~pred_pos) & true_pos).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


@torch.no_grad()
def evaluate_anchor(model, loader, cfg, temporal=True):
    model.eval()
    total_tp = total_fp = total_fn = 0
    for batch in loader:
        images = batch["images"].to(cfg.device)
        cls_target = batch["cls_target"].to(cfg.device)
        exist_target = batch["exist_target"].to(cfg.device)
        outputs = model(images) if temporal else model(images[:, -1])
        m = compute_anchor_metrics(outputs, cls_target, exist_target, cfg)
        total_tp += m["tp"]; total_fp += m["fp"]; total_fn += m["fn"]

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    denom = max(total_tp + total_fp + total_fn, 1)
    return {
        "anchor_precision": precision,
        "anchor_recall": recall,
        "anchor_f1": f1,
        "anchor_false_positive_rate": total_fp / denom,
        "anchor_false_negative_rate": total_fn / denom,
        "anchor_true_positives": total_tp,
        "anchor_false_positives": total_fp,
        "anchor_false_negatives": total_fn,
    }


# ============================================================
# 7a. Overfitting diagnostics
# ============================================================

def diagnose_epoch(prev_row, row, cfg):
    flags = []
    metric_key = "polyline_pixel_f1" if "polyline_pixel_f1" in row else "anchor_f1"
    gap_ratio = getattr(cfg, "diag_gap_ratio", 1.5)

    if row["train_loss"] > 0 and row["validation_loss"] > gap_ratio * row["train_loss"]:
        flags.append(
            f"LARGE TRAIN/VALIDATION GAP: validation_loss={row['validation_loss']:.3f} "
            f"is >{gap_ratio}x train_loss={row['train_loss']:.3f}"
        )

    if prev_row is not None:
        train_delta = row["train_loss"] - prev_row["train_loss"]
        val_delta = row["validation_loss"] - prev_row["validation_loss"]
        metric_delta = row[metric_key] - prev_row.get(metric_key, row[metric_key])
        MEANINGFUL = 1e-3
        if train_delta < -MEANINGFUL and val_delta > MEANINGFUL:
            flags.append(
                f"OVERFITTING: train_loss down ({train_delta:+.3f}) but "
                f"validation_loss up ({val_delta:+.3f})"
            )
    return flags


# ============================================================
# 8. Generic training loop
# ============================================================

# ------------------------------------------------------------------
# 8-TB. TensorBoard instrumentation helpers (task Sections 3, 4, 6, 7)
# ------------------------------------------------------------------
#
# Design notes (also covered in the final "Implementation summary"
# deliverable, kept here as the code-adjacent version):
#   * Writers are created per (seed, model-key) in run_pipeline / the
#     Optuna objective with a unique log_dir -- see get_tb_writer() above
#     -- so different runs never share or overwrite each other's scalars.
#   * Batch-level scalars use a distinct tag namespace ("batch/...") from
#     epoch-level scalars ("epoch/...") so they are never plotted on the
#     same implied step scale by accident, and use a monotonically
#     increasing global batch counter (not the per-epoch batch_idx) as
#     their step, so TensorBoard's x-axis is continuous across epochs.
#   * Anchor-level and pixel-level metrics are logged under distinct tag
#     prefixes ("metrics/anchor_*" vs "metrics/pixel_*") specifically so a
#     case of "anchor_f1 high, pixel F1 ~0" (the exact bug this revision's
#     main_loss fix addresses -- see the comment on main_loss) is visible
#     as two clearly-labelled, non-overlapping curves rather than a single
#     ambiguous "f1" tag.
#   * The graph is logged once per model (guarded by `_graph_logged`),
#     not every epoch/trial -- tracing is comparatively expensive and the
#     architecture does not change during training.


def log_model_graph_to_tensorboard(writer, model, cfg, temporal, label,
                                   ablation_mode=None):
    """TensorBoard Graph (task Section 4). Traces the actual model via
    torch.jit-based tracing (what SummaryWriter.add_graph uses internally)
    with a dummy input shaped from the real forward() signature. Logged
    once per model, not per epoch/trial (see call sites). Dynamic control
    flow -- the ablation_mode branches inside
    TACUFLDTemporalModel._fuse_features, `if x.dim() == 5` in
    SingleFrameUFLDLikeModel.forward -- can make jit tracing warn or fail;
    that is handled here rather than crashing the training run, per the
    task's "handle unsupported tracing operations gracefully" requirement.
    """
    if writer is None:
        return False
    was_training = model.training
    prev_ablation = getattr(model, "ablation_mode", None)
    if hasattr(model, "ablation_mode"):
        model.ablation_mode = ablation_mode
    model.eval()
    ok = False
    try:
        dummy = _dummy_input_for_model(model, cfg, temporal)
        wrapped = _wrap_for_graph_export(model)
        with torch.no_grad():
            writer.add_graph(wrapped, dummy)
        ok = True
        print(f"[tensorboard] graph logged for {label}")
    except Exception as exc:
        print(f"[tensorboard] add_graph failed for {label}: {exc}")
    finally:
        if hasattr(model, "ablation_mode"):
            model.ablation_mode = prev_ablation
        model.train(was_training)
    return ok


def log_scalars_to_tensorboard(writer, row, epoch, prefix=""):
    """Epoch-level scalars (task Section 3.A/3.B/3.C). `row` is one row
    of the `history` list already built by train_model -- this function
    does not compute anything new, it only routes existing values to
    clearly-namespaced TensorBoard tags so existing CSV/plot code paths
    are untouched."""
    if writer is None:
        return
    p = f"{prefix}/" if prefix else ""

    loss_tags = {
        "Loss/train_total": "train_loss",
        "Loss/validation_total": "validation_loss",
        "Loss/train_cross_entropy": "train_cross_entropy_loss",
        "Loss/validation_cross_entropy": "validation_cross_entropy_loss",
        "Loss/train_existence": "train_existence_loss",
        "Loss/validation_existence": "validation_existence_loss",
        "Loss/train_x_regression": "train_x_loss",
        "Loss/validation_x_regression": "validation_x_loss",
        "Loss/train_y_regression": "train_y_loss",
        "Loss/train_temporal_consistency": "train_temporal_consistency_loss",
        "Loss/validation_temporal_consistency": "validation_temporal_consistency_loss",
        "Loss/train_temporal_consistency_weighted": "train_temporal_consistency_loss_weighted",
        "Loss/validation_temporal_consistency_weighted": "validation_temporal_consistency_loss_weighted",
    }
    for tag, key in loss_tags.items():
        if key in row and row[key] is not None and np.isfinite(row[key]):
            writer.add_scalar(f"{p}{tag}", float(row[key]), epoch)

    # Overfitting/generalization (task Section 3.C): train vs validation on
    # the SAME chart via add_scalars so the gap is visible at a glance.
    if "train_loss" in row and "validation_loss" in row:
        writer.add_scalars(f"{p}Loss/train_vs_validation", {
            "train": float(row["train_loss"]),
            "validation": float(row["validation_loss"]),
        }, epoch)
    if "generalization_gap_validation_minus_train" in row:
        writer.add_scalar(f"{p}Diagnostics/generalization_gap",
                          float(row["generalization_gap_validation_minus_train"]), epoch)

    # Anchor-level metrics -- distinct namespace from pixel-level (task
    # Section 3.B: "clearly distinguish anchor-level from pixel-level").
    anchor_tags = {
        "Metrics/anchor_f1": "anchor_f1",
        "Metrics/anchor_precision": "anchor_precision",
        "Metrics/anchor_recall": "anchor_recall",
        "Metrics/anchor_false_positive_rate": "anchor_false_positive_rate",
        "Metrics/anchor_false_negative_rate": "anchor_false_negative_rate",
    }
    for tag, key in anchor_tags.items():
        if key in row and row[key] is not None:
            writer.add_scalar(f"{p}{tag}", float(row[key]), epoch)

    if "polyline_pixel_f1" in row and row["polyline_pixel_f1"] is not None:
        writer.add_scalar(f"{p}Metrics/pixel_f1", float(row["polyline_pixel_f1"]), epoch)

    if "diagnostic_flags" in row and row["diagnostic_flags"]:
        writer.add_text(f"{p}Diagnostics/flags", row["diagnostic_flags"], epoch)


def log_learning_rates_to_tensorboard(writer, optimizer, global_step, prefix=""):
    if writer is None:
        return
    p = f"{prefix}/" if prefix else ""
    for i, group in enumerate(optimizer.param_groups):
        writer.add_scalar(f"{p}LearningRate/group_{i}", group["lr"], global_step)


# ------------------------------------------------------------------
# 8-VIZ. Fixed validation subset + prediction overlay rendering
# (shared by TensorBoard image logging, the FP/FN folders, and the
# temporal-consistency videos, so all three use exactly the same
# reproducible sample selection and drawing code).
# ------------------------------------------------------------------

def select_fixed_visualization_subset(dataset, cfg, n=None):
    """A FIXED (seeded independently of cfg.seed, so it doesn't shift if
    the training seed changes), reproducible subset of *validation*
    indices for all qualitative logging (task Section 8: "use a fixed
    validation subset ... reproducible sampling"). Returns a sorted list
    of dataset indices -- sorted so that, when the caller groups by scene,
    frame order within a scene stays chronological (needed for the
    temporal video / jitter-style logic; harmless for the plain image
    log)."""
    n = n or getattr(cfg, "max_visualization_samples", 6)
    n = min(n, len(dataset))
    rng = random.Random(getattr(cfg, "visualization_seed", 12345))
    idx = list(range(len(dataset)))
    rng.shuffle(idx)
    return sorted(idx[:n])


def _bgr_uint8_to_tb_tensor(image_bgr):
    """cv2 BGR uint8 HxWx3 -> torch float CHW RGB in [0,1], the layout
    SummaryWriter.add_image expects by default (CHW)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t


def render_prediction_overlay(item, cfg, params, gate_value=None, show_confidence=True):
    """One rendered panel for a single cached prediction: original image
    with ground truth (green) and decoded/post-processed prediction (red)
    overlaid, a legend, and the frame/sequence identifier -- the
    consistent color/line-width/legend/identifier contract the task asks
    for in Section 6. `item` is one entry of the list returned by
    build_prediction_cache(); `params` is a post-processing param dict as
    used elsewhere (postprocess_item / evaluate_iou_cache)."""
    image = cv2.imread(item["image_path"])
    if image is None:
        return None
    pred_lanes = postprocess_item(item, cfg, params)
    gt_lanes = item["gt_lanes"]

    panel = image.copy()
    panel = _draw_lanes(panel, gt_lanes, _COLOR_GT, thickness=3)
    panel = _draw_lanes(panel, pred_lanes, _COLOR_PRED, thickness=2)

    label = f"{item['scene']}  frame={item['frame_id']}"
    if show_confidence and item["pred_prob"].size:
        mean_conf = float(item["pred_prob"][item["pred_prob"] >= params["threshold"]].mean()
                          if (item["pred_prob"] >= params["threshold"]).any() else 0.0)
        label += f"  mean_conf={mean_conf:.2f}"
    if gate_value is not None:
        label += f"  gate={float(gate_value):.2f}"

    cv2.putText(panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
               (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
               (255, 255, 255), 1, cv2.LINE_AA)
    legend_y = panel.shape[0] - 14
    cv2.putText(panel, "green = ground truth   red = prediction",
               (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
               (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel, "green = ground truth   red = prediction",
               (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
               (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def log_validation_images_to_tensorboard(writer, model, cfg, val_dataset, epoch,
                                         temporal, tuned_params, mode):
    """Task Section 6, wired into the training loop (not a separate
    script, so it always reflects the real training state -- called from
    inside train_model on the SAME model object mid-training, in eval()
    mode with no_grad, then restored). Runs only every
    cfg.visualize_every_n_epochs epochs and only over the fixed subset
    from select_fixed_visualization_subset(), so the CPU-bound smoke-test
    setup described in the task's Section 8 is not overwhelmed by
    per-batch image logging.
    """
    every_n = getattr(cfg, "visualize_every_n_epochs", 0)
    if writer is None or every_n <= 0 or epoch % every_n != 0:
        return
    indices = select_fixed_visualization_subset(val_dataset, cfg)
    was_training = model.training
    model.eval()
    try:
        cache = []
        with torch.no_grad():
            for idx in indices:
                sample = val_dataset[idx]
                images = sample["images"].unsqueeze(0).to(cfg.device)
                outputs = model(images) if temporal else model(images[:, -1])
                x_px, y_px, pred_prob, _ = decode_outputs(outputs, cfg)
                row = val_dataset.rows[idx]
                gt = val_dataset.scene_records[row["scene"]]["gt_frames"][row["frame_id"]]
                gt_lanes = [lane for lane in (gt["left"], gt["right"]) if lane is not None]
                cache.append({
                    "image_path": str(sample["image_path"]),
                    "scene": row["scene"], "frame_id": row["frame_id"],
                    "gt_lanes": gt_lanes,
                    "width": int(sample["orig_w"]), "height": int(sample["orig_h"]),
                    "pred_prob": pred_prob[0].detach().cpu().numpy(),
                    "pred_x_px": x_px[0].detach().cpu().numpy(),
                    "pred_y_px": y_px[0].detach().cpu().numpy(),
                })
        for i, item in enumerate(cache):
            panel = render_prediction_overlay(item, cfg, tuned_params)
            if panel is not None:
                writer.add_image(f"validation_samples/{mode}/sample_{i}",
                                 _bgr_uint8_to_tb_tensor(panel), epoch)
    except Exception as exc:
        # Per task Section 8: "do not silently skip failed visualizations
        # ... record informative warnings while allowing the main training
        # pipeline to continue."
        print(f"[tensorboard] image logging failed at epoch {epoch} ({mode}): {exc}")
    finally:
        model.train(was_training)


def _hpo_val_f1(model, val_dataset, cfg, temporal, threshold):
    params = dict(POSTPROCESS_COMMON_FIXED_PARAMS)
    params["threshold"] = threshold
    cache = build_prediction_cache(
        model, val_dataset, cfg, temporal=temporal,
        ablation_mode=getattr(model, "ablation_mode", None),
    )
    metrics, _ = evaluate_iou_cache(cache, cfg, params, model_name="hpo_val", verbose=False)
    return metrics["polyline_pixel_f1"]


def _temporal_frame_names(num_frames):
    return [
        "frame_t" if k == 0 else f"frame_t_minus_{k}"
        for k in range(num_frames - 1, -1, -1)
    ]


def _evaluate_val_loss(model, val_loader, cfg, mode, temporal=True, lambda_temporal=0.0):
    model.eval()
    total_loss = total_main = total_ce = total_ex = 0.0
    total_x = total_temp = total_comparable = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(cfg.device, non_blocking=True)
            cls_target = batch["cls_target"].to(cfg.device, non_blocking=True)
            exist_target = batch["exist_target"].to(cfg.device, non_blocking=True)
            coords_target = batch["coords_target"].to(cfg.device, non_blocking=True)

            outputs = model(images) if temporal else model(images[:, -1])
            loss_main, logs = main_loss(
                outputs, cls_target, exist_target, coords_target, cfg, mode=mode,
            )

            if temporal and lambda_temporal > 0 and hasattr(model, "forward_per_frame"):
                n_frames = images.shape[1]
                pf = model.forward_per_frame(images, frame_indices=(n_frames - 2, n_frames - 1))
                temp_loss = temporal_alignment_loss(pf[-1], pf[-2], exist_weight=0.5)
            else:
                temp_loss = torch.tensor(0.0, device=cfg.device)

            loss = loss_main + lambda_temporal * temp_loss

            total_loss += float(loss.detach().cpu())
            total_main += float(loss_main.detach().cpu())
            total_ce += float(logs["ce_loss"].detach().cpu())
            total_ex += float(logs["exist_loss"].detach().cpu())
            total_x += float(logs["x_loss"].detach().cpu())
            total_comparable += float(logs["comparable_loss"].detach().cpu())
            total_temp += float(temp_loss.detach().cpu())

    n = max(len(val_loader), 1)
    return {
        "validation_loss": total_loss / n,
        "validation_main_loss": total_main / n,
        "validation_cross_entropy_loss": total_ce / n,
        "validation_existence_loss": total_ex / n,
        "validation_x_loss": total_x / n,
        "validation_temporal_consistency_loss": total_temp / n,
        "validation_comparable_loss": total_comparable / n,
        "fusion_weights": None,
        "fusion_kind": "residual gate (per-pixel, mean over batch and space)" if temporal else None,
    }


seed_holder = {"seed": None}


def print_epoch_report(mode, epoch, epochs, row, lambda_temporal, temporal,
                       fusion_weights, fusion_kind, selection_name,
                       best_metric, status, flags=None):
    """Streamlined per-epoch console report.

    Focused on the numbers that actually matter for comparing the four
    models against each other: the loss trajectory (train vs val -- now
    directly comparable across modes since main_loss no longer differs
    between "baseline" and the temporal variants), the primary quality
    metric (polyline_pixel_f1, the pixel-accurate metric used for model
    selection and for the final test comparison), the lenient anchor_f1 as
    a secondary sanity signal, and the generalization gap so overfitting is
    visible at a glance. The per-component loss breakdown (ce/exist/x/y)
    is still written to the CSV history every epoch for anyone who wants
    the detail; it's just not printed to console anymore to keep the
    per-epoch scan-and-compare workflow fast.
    """
    flags = flags or []
    def fmt(v, p=4):
        return f"{v:.{p}f}" if isinstance(v, float) else str(v)
    print()
    print(f"[{mode}] epoch {epoch:02d}/{epochs}")
    gap = row.get("generalization_gap_validation_minus_train")
    loss_parts = [
        f"train_loss={fmt(row['train_loss'])}",
        f"validation_loss={fmt(row['validation_loss'])}",
    ]
    if gap is not None:
        loss_parts.append(f"gap={fmt(gap)}")
    print("LOSS    " + "  ".join(loss_parts))
    metric_parts = [f"anchor_f1={fmt(row['anchor_f1'])}"]
    if "polyline_pixel_f1" in row:
        metric_parts.append(f"polyline_pixel_f1={fmt(row['polyline_pixel_f1'])}  <-- primary")
    print("METRIC  " + "  ".join(metric_parts))
    if flags:
        print("DIAG    " + " | ".join(flags))
    print(
        f"SELECT  selection_metric={selection_name}  "
        f"current_value={row[selection_name]:.4f}  "
        f"best_value_so_far={best_metric:.4f}  status={status}"
    )


def train_model(model, loader, val_loader, cfg, epochs, mode, temporal=True,
                lr_backbone=1e-4, lr_head=1e-4, lr_fusion=5e-4,
                lambda_temporal=0.0, save_path=None,
                trial=None, val_dataset=None, writer=None, tb_prefix=None):
    """`writer` (task Sections 3, 4, 6): an optional SummaryWriter, created
    by the caller with a unique per-(seed, model) log_dir via
    get_tb_writer() -- this function never creates or closes a writer
    itself, so callers that share one writer across baseline/v02/v03/v04
    (distinguished by `tb_prefix`) or use one writer per model are both
    supported. Passing writer=None (the default) fully disables all
    TensorBoard instrumentation and reproduces the exact previous
    behaviour of this function.
    `tb_prefix` defaults to `mode` (e.g. "baseline", "v02") so scalars
    from different models logged to the same writer land in different
    TensorBoard tag namespaces instead of overwriting each other's curves.
    """
    tb_prefix = mode if tb_prefix is None else tb_prefix
    if writer is not None and getattr(cfg, "tb_log_graph", True):
        log_model_graph_to_tensorboard(writer, model, cfg, temporal, label=mode)

    params = []
    if hasattr(model, "backbone"):
        params.append({"params": model.backbone.parameters(), "lr": lr_backbone})
    if hasattr(model, "history_encoder"):
        params.append({"params": model.history_encoder.parameters(), "lr": lr_backbone})
    if hasattr(model, "head"):
        params.append({"params": model.head.parameters(), "lr": lr_head})
    if hasattr(model, "temporal_fusion"):
        params.append({"params": model.temporal_fusion.parameters(), "lr": lr_fusion})

    optimizer = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    # Cosine LR decay: with a fixed LR the smoke-test runs converged fast
    # and then overfit (best_epoch=1/20 for v04!) with nothing to cool them
    # down. Annealing each param group's LR towards ~5% of its initial
    # value over the run gives the optimizer a chance to settle into a
    # minimum instead of bouncing around one once it gets close.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1),
        eta_min=min(g["lr"] for g in optimizer.param_groups) * 0.05,
    )
    use_amp = cfg.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    use_pixel_f1 = val_dataset is not None
    selection_metric_name = "polyline_pixel_f1" if use_pixel_f1 else "anchor_f1"

    best_metric = -float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0
    is_hpo_run = (trial is not None)
    patience = 1 if is_hpo_run else getattr(cfg, "early_stopping_patience", 8)
    min_delta = getattr(cfg, "early_stopping_min_delta", 0.0)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = total_main = total_ce = total_ex = 0.0
        total_temp = total_comparable = 0.0
        total_x = total_y = 0.0
        start = time.time()

        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["images"].to(cfg.device, non_blocking=True)
            cls_target = batch["cls_target"].to(cfg.device, non_blocking=True)
            exist_target = batch["exist_target"].to(cfg.device, non_blocking=True)
            coords_target = batch["coords_target"].to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images) if temporal else model(images[:, -1])
                loss_main, logs = main_loss(
                    outputs, cls_target, exist_target, coords_target, cfg, mode=mode,
                )

                # flow smoothness regulariser
                if temporal and outputs.get("flows"):
                    fl = flow_smoothness_loss(outputs["flows"])
                    loss_main = loss_main + getattr(cfg, "lambda_flow_smooth", 0.0) * fl
                    logs["flow_smooth_loss"] = fl.detach()
                else:
                    logs["flow_smooth_loss"] = torch.tensor(0.0)

                # temporal alignment loss
                if temporal and lambda_temporal > 0 and hasattr(model, "forward_per_frame"):
                    n_frames = images.shape[1]
                    pf = model.forward_per_frame(
                        images, frame_indices=(n_frames - 2, n_frames - 1)
                    )
                    temp_loss = temporal_alignment_loss(pf[-1], pf[-2], exist_weight=0.5)
                else:
                    temp_loss = torch.tensor(0.0, device=cfg.device)

                loss = loss_main + lambda_temporal * temp_loss

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if getattr(cfg, "grad_clip_norm", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if getattr(cfg, "grad_clip_norm", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_main += float(loss_main.detach().cpu())
            total_ce += float(logs["ce_loss"].detach().cpu())
            total_ex += float(logs["exist_loss"].detach().cpu())
            total_comparable += float(logs["comparable_loss"].detach().cpu())
            total_temp += float(temp_loss.detach().cpu())
            if "x_loss" in logs: total_x += float(logs["x_loss"])
            if "y_loss" in logs: total_y += float(logs["y_loss"])

            if batch_idx % 20 == 0 or batch_idx == len(loader):
                # "coord_x" is the metric to watch here: it's the loss that
                # decides whether the model's predicted lane position is
                # actually usable (see the main_loss fix above). "y=" and
                # the old duplicate "aux=" (== coord_x) were dropped since
                # they never carried extra information.
                print(
                    f"{mode} | epoch {epoch}/{epochs} | batch {batch_idx}/{len(loader)} | "
                    f"loss={total_loss / batch_idx:.4f} | ce={total_ce / batch_idx:.4f} | "
                    f"exist={total_ex / batch_idx:.4f} | coord_x={total_x / batch_idx:.4f} | "
                    f"temporal={total_temp / batch_idx:.4f} | elapsed={time.time() - start:.1f}s"
                )

            # ---- TensorBoard batch-level scalars (task Section 3.A) ----
            # Distinct "batch/" tag namespace from the epoch-level "epoch/"
            # tags written below, and a monotonically increasing global
            # step (not the per-epoch batch_idx) so curves are continuous
            # across epoch boundaries instead of resetting to 0 each epoch.
            n_bs = getattr(cfg, "tb_log_batch_scalars_every_n_batches", 0)
            if writer is not None and n_bs > 0 and batch_idx % n_bs == 0:
                global_step = (epoch - 1) * len(loader) + batch_idx
                writer.add_scalar(f"{tb_prefix}/batch/loss_total",
                                  total_loss / batch_idx, global_step)
                writer.add_scalar(f"{tb_prefix}/batch/loss_cross_entropy",
                                  total_ce / batch_idx, global_step)
                writer.add_scalar(f"{tb_prefix}/batch/loss_existence",
                                  total_ex / batch_idx, global_step)
                writer.add_scalar(f"{tb_prefix}/batch/loss_coord_x",
                                  total_x / batch_idx, global_step)
                if temporal:
                    writer.add_scalar(f"{tb_prefix}/batch/loss_temporal_consistency",
                                      total_temp / batch_idx, global_step)
                log_learning_rates_to_tensorboard(writer, optimizer, global_step,
                                                  prefix=f"{tb_prefix}/batch")

        n_train = max(len(loader), 1)
        val_metrics = evaluate_anchor(model, val_loader, cfg, temporal=temporal)
        val_loss_metrics = _evaluate_val_loss(
            model, val_loader, cfg, mode=mode, temporal=temporal,
            lambda_temporal=lambda_temporal,
        )
        fusion_weights = val_loss_metrics.pop("fusion_weights")
        fusion_kind = val_loss_metrics.pop("fusion_kind")

        avg_temp_train = total_temp / n_train
        row = {
            "epoch": epoch,
            "train_loss": total_loss / n_train,
            "train_main_loss": total_main / n_train,
            "train_cross_entropy_loss": total_ce / n_train,
            "train_existence_loss": total_ex / n_train,
            "train_comparable_loss": total_comparable / n_train,
            "train_x_loss": total_x / n_train,
            "train_y_loss": total_y / n_train,
            "lambda_temporal": lambda_temporal,
            "train_temporal_consistency_loss": avg_temp_train,
            "train_temporal_consistency_loss_weighted": lambda_temporal * avg_temp_train,
            **val_loss_metrics,
            "validation_temporal_consistency_loss_weighted":
                lambda_temporal * val_loss_metrics["validation_temporal_consistency_loss"],
            **val_metrics,
        }
        row["generalization_gap_validation_minus_train"] = (
            row["validation_loss"] - row["train_loss"]
        )
        row["comparable_generalization_gap_validation_minus_train"] = (
            row["validation_comparable_loss"] - row["train_comparable_loss"]
        )

        if val_dataset is not None:
            row["polyline_pixel_f1"] = _hpo_val_f1(
                model, val_dataset, cfg, temporal, threshold=cfg.exist_threshold,
            )

        selection_metric = row[selection_metric_name]
        prev_row = history[-1] if history else None
        flags = diagnose_epoch(prev_row, row, cfg)
        row["diagnostic_flags"] = " | ".join(flags) if flags else ""

        # ---- TensorBoard epoch-level scalars + validation images ----
        # (task Sections 3.A/3.B/3.C and 6). Epoch/batch progress itself
        # is the `epoch` step argument every add_scalar call above and
        # below already uses as its global step.
        if writer is not None:
            log_scalars_to_tensorboard(writer, row, epoch, prefix=tb_prefix)
            if val_dataset is not None:
                # No final-tuned post-processing params exist yet at this
                # point in the pipeline (sweep_postprocessing runs after
                # training, in run_pipeline) -- POSTPROCESS_COMMON_FIXED_PARAMS
                # is used here purely to render a human-checkable overlay
                # during training, not as an evaluation protocol. It has no
                # effect on polyline_pixel_f1 selection above, which goes
                # through the real evaluate_iou_cache/_hpo_val_f1 path.
                log_validation_images_to_tensorboard(
                    writer, model, cfg, val_dataset, epoch, temporal,
                    tuned_params=POSTPROCESS_COMMON_FIXED_PARAMS, mode=mode,
                )

        if trial is not None:
            trial.report(float(selection_metric), epoch)
            if trial.should_prune():
                history.append(row)
                raise optuna.TrialPruned()

        history.append(row)

        if trial is not None and row.get("anchor_f1", 1.0) < 0.35:
            raise optuna.TrialPruned()

        tie = abs(selection_metric - best_metric) <= min_delta
        improved = (selection_metric > best_metric + min_delta) or (
            tie and row["validation_loss"] < best_val_loss)
        if improved:
            best_val_loss = row["validation_loss"]
            best_metric = float(selection_metric)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            status = "BEST"
        else:
            patience_counter += 1
            status = f"no improvement ({patience_counter}/{patience})"

        print_epoch_report(
            mode, epoch, epochs, row, lambda_temporal, temporal,
            fusion_weights, fusion_kind, selection_metric_name,
            best_metric, status=status, flags=flags,
        )

        if improved and save_path is not None:
            torch.save(best_state, save_path)

        if patience > 0 and patience_counter >= patience and epoch < epochs:
            print(f"{mode} | early stopping at epoch {epoch} | best_epoch={best_epoch}")
            break

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    if not history_df.empty:
        history_df.attrs["best_epoch"] = best_epoch
        history_df.attrs["best_metric_name"] = selection_metric_name
        history_df.attrs["best_metric"] = best_metric
    return history_df


# ============================================================
# 8a. Plot helpers
# ============================================================

def _save_figure(fig, out_path, dpi=150):
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _best_epoch(history_df):
    if "best_epoch" in history_df.attrs:
        return history_df.attrs["best_epoch"]
    key = "polyline_pixel_f1" if "polyline_pixel_f1" in history_df.columns else "anchor_f1"
    return int(history_df.loc[history_df[key].idxmax(), "epoch"])


def _present(df, columns):
    return [c for c in columns if c in df.columns]


def plot_model_history(history_df, model_label, out_path):
    if history_df.empty:
        return
    best_epoch = _best_epoch(history_df)
    flagged = history_df.loc[history_df["diagnostic_flags"] != "", "epoch"] \
        if "diagnostic_flags" in history_df.columns else []
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    loss_long = history_df.melt(
        id_vars="epoch", value_vars=_present(history_df, ["train_loss", "validation_loss"]),
        var_name="series", value_name="loss",
    )
    sns.lineplot(data=loss_long, x="epoch", y="loss", hue="series", style="series",
                 markers=True, dashes=False, linewidth=2, ax=ax)
    ax.set_title(f"{model_label}: convergence")

    ax = axes[0, 1]
    component_cols = _present(history_df, [
        "train_cross_entropy_loss", "train_existence_loss", "train_x_loss",
        "train_y_loss", "train_temporal_consistency_loss",
    ])
    component_cols = [c for c in component_cols if history_df[c].abs().max() > 0]
    if component_cols:
        comp_long = history_df.melt(id_vars="epoch", value_vars=component_cols,
                                    var_name="series", value_name="loss")
        sns.lineplot(data=comp_long, x="epoch", y="loss", hue="series", marker="o", ax=ax)
        ax.legend(fontsize=7)
    ax.set_title(f"{model_label}: loss components")

    ax = axes[1, 0]
    metric_cols = _present(history_df, ["polyline_pixel_f1", "anchor_f1",
                                        "anchor_precision", "anchor_recall"])
    metric_long = history_df.melt(id_vars="epoch", value_vars=metric_cols,
                                  var_name="series", value_name="score")
    sns.lineplot(data=metric_long, x="epoch", y="score", hue="series", style="series",
                 markers=True, dashes=False, ax=ax)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{model_label}: validation metrics")

    ax = axes[1, 1]
    gap_col = "generalization_gap_validation_minus_train"
    if gap_col in history_df.columns:
        colors = ["#d55e00" if v > 0 else "#0072b2" for v in history_df[gap_col]]
        sns.barplot(data=history_df, x="epoch", y=gap_col, hue="epoch", palette=colors,
                    legend=False, ax=ax)
        ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"{model_label}: generalization gap")

    for a in (axes[0, 0], axes[0, 1], axes[1, 0]):
        a.axvline(best_epoch, color="green", linestyle=":", linewidth=2)
        for e in flagged:
            a.axvline(e, color="red", linestyle="--", alpha=0.25)
    _save_figure(fig, out_path)


def plot_all_models_convergence(histories, out_path):
    frames = []
    for key, hist in histories.items():
        if hist is None or hist.empty:
            continue
        label = MODEL_LABELS.get(key, key)
        d = hist.copy(); d["model"] = label
        frames.append(d)
    if not frames:
        return
    long = pd.concat(frames, ignore_index=True)
    panels = [
        ("train_comparable_loss", "train_comparable_loss"),
        ("validation_comparable_loss", "validation_comparable_loss"),
        ("anchor_f1", "anchor_f1"),
        ("polyline_pixel_f1", "polyline_pixel_f1"),
        ("train_x_loss", "train_x_loss (direct x regression)"),
        ("validation_loss", "validation_loss"),
    ]
    panels = [p for p in panels if p[0] in long.columns]
    n_cols = 3
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.3 * n_rows), squeeze=False)
    order = [m for m in MODEL_ORDER if m in long["model"].unique()]
    for ax, (col, title) in zip(axes.ravel(), panels):
        sns.lineplot(data=long, x="epoch", y=col, hue="model", hue_order=order,
                     palette=MODEL_PALETTE, marker="o", linewidth=2, ax=ax, legend=False)
        if col in ("polyline_pixel_f1", "anchor_f1"):
            ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=10)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=MODEL_PALETTE[m], marker="o", linewidth=2, label=m)
               for m in order]
    fig.legend(handles=handles, loc="lower center", ncol=len(order))
    _save_figure(fig, out_path)


def plot_fusion_weight_evolution(histories, out_path):
    keys = [k for k in ("v02", "v03", "v04")
            if k in histories and not histories[k].empty
            and any(c.startswith("fusion_weight_") for c in histories[k].columns)]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(5.8 * len(keys), 4.2),
                             sharey=True, squeeze=False)
    for ax, key in zip(axes[0], keys):
        hist = histories[key]
        cols = [c for c in hist.columns if c.startswith("fusion_weight_")]
        long = hist.melt(id_vars="epoch", value_vars=cols,
                         var_name="frame", value_name="fusion_weight")
        long["frame"] = long["frame"].str.replace("fusion_weight_", "", regex=False)
        sns.lineplot(data=long, x="epoch", y="fusion_weight", hue="frame",
                     marker="o", linewidth=2, ax=ax)
        ax.set_title(MODEL_LABELS.get(key, key))
        ax.set_ylim(0, 1)
    _save_figure(fig, out_path)


def plot_results_matrix(results_df, metrics, out_path):
    if results_df.empty:
        return
    data = results_df.set_index("model")[metrics].astype(float)
    fig, ax = plt.subplots(figsize=(2.0 * len(metrics) + 3, 0.8 * len(data) + 2.5))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.5, ax=ax)
    ax.set_title("Final results matrix")
    _save_figure(fig, out_path)


def plot_metric_comparison_bars(results_df, metrics, out_path, title):
    long = results_df.melt(id_vars="model", value_vars=metrics,
                           var_name="metric", value_name="value")
    order = [m for m in MODEL_ORDER if m in results_df["model"].values]
    fig, ax = plt.subplots(figsize=(1.9 * len(metrics) + 4, 5.5))
    sns.barplot(data=long, x="metric", y="value", hue="model", hue_order=order,
                palette=MODEL_PALETTE, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
    ax.set_ylim(0, min(1.12, max(1.0, long["value"].max() * 1.15)))
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save_figure(fig, out_path)


def plot_delta_vs_baseline(results_df, metrics, out_path):
    delta_cols = [f"delta_{m}_vs_baseline" for m in metrics
                  if f"delta_{m}_vs_baseline" in results_df.columns]
    others = results_df[results_df["model"] != BASELINE_LABEL]
    if not delta_cols or others.empty:
        return
    long = others.melt(id_vars="model", value_vars=delta_cols,
                       var_name="metric", value_name="difference_from_baseline")
    order = [m for m in MODEL_ORDER if m in others["model"].values]
    fig, ax = plt.subplots(figsize=(1.9 * len(delta_cols) + 4, 5))
    sns.barplot(data=long, x="metric", y="difference_from_baseline", hue="model",
                hue_order=order, palette=MODEL_PALETTE, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%+.3f", fontsize=7, padding=2)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Difference from Baseline")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save_figure(fig, out_path)


def plot_condition_heatmap(results_df, out_path):
    cols = [c for c in ("iou_f1_rainy", "iou_f1_occlusion", "iou_f1_shaky",
                        "iou_f1_transition") if c in results_df.columns]
    if not cols:
        return
    data = results_df.set_index("model")[cols].astype(float)
    data = data.reindex([m for m in MODEL_ORDER if m in data.index])
    fig, ax = plt.subplots(figsize=(2.0 * len(cols) + 3, 0.8 * len(data) + 2.8))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
                mask=data.isna(), linewidths=0.5, ax=ax)
    ax.set_title("iou_f1 per condition")
    _save_figure(fig, out_path)


def plot_ablation(ablation_df, out_path):
    if ablation_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=ablation_df, x="model", y="iou_f1", hue="ablation", ax=axes[0])
    for c in axes[0].containers:
        axes[0].bar_label(c, fmt="%.3f", fontsize=7, padding=2)
    axes[0].set_title("Ablation iou_f1")
    delta = ablation_df[ablation_df["ablation"] != "full"]
    if not delta.empty:
        sns.barplot(data=delta, x="model", y="delta_iou_f1_vs_full", hue="ablation", ax=axes[1])
        for c in axes[1].containers:
            axes[1].bar_label(c, fmt="%+.3f", fontsize=7, padding=2)
        axes[1].axhline(0, color="black", linewidth=1)
    _save_figure(fig, out_path)


def plot_efficiency_tradeoff(results_df, out_path):
    needed = {"latency_milliseconds", "iou_f1", "parameters_total_millions"}
    if not needed.issubset(results_df.columns):
        return
    fig, ax = plt.subplots(figsize=(8.5, 6))
    order = [m for m in MODEL_ORDER if m in results_df["model"].values]
    sns.scatterplot(data=results_df, x="latency_milliseconds", y="iou_f1",
                    size="parameters_total_millions", hue="model", hue_order=order,
                    palette=MODEL_PALETTE, sizes=(200, 900), edgecolor="black", ax=ax)
    for _, r in results_df.iterrows():
        ax.annotate(r["model"], (r["latency_milliseconds"], r["iou_f1"]),
                    textcoords="offset points", xytext=(8, 8), fontsize=8)
    ax.axvline(ADAS_LATENCY_BUDGET_MS, color="red", linestyle="--", linewidth=1.5,
               label=f"{ADAS_TARGET_FPS:.0f} FPS budget ({ADAS_LATENCY_BUDGET_MS:.1f} ms)")
    ax.legend(fontsize=8)
    _save_figure(fig, out_path)


def plot_confusion_grid(confusion_by_model, out_path):
    keys = [k for k in ("baseline", "v02", "v03", "v04") if k in confusion_by_model]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.8), squeeze=False)
    for ax, key in zip(axes[0], keys):
        c = confusion_by_model[key]
        tp, fp, fn = c["true_positives"], c["false_positives"], c["false_negatives"]
        matrix = np.array([[tp, fn], [fp, 0]], dtype=float)
        sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(MODEL_LABELS[key])
    _save_figure(fig, out_path)


def plot_postprocessing_sweep_heatmap(sweep_df, model_label, out_path):
    if sweep_df.empty:
        return
    pivot = sweep_df.pivot_table(index="threshold", columns="min_points",
                                 values="iou_f1", aggfunc="max")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set_title(f"{model_label}: val iou_f1 over postproc grid")
    _save_figure(fig, out_path)


def plot_adas_readiness(results_df, out_path):
    """Two-panel ADAS-readiness view, separate from the generic metric bars:
    (1) temporal-stability (jitter) per model -- lower is better, this is
        the axis the temporal-fusion variants exist to win on -- and
    (2) safety-weighted pixel F2 vs the plain pixel F1, so a model that
        trades precision for recall (fewer missed lanes) is visible even
        if its F1 looks unchanged.
    """
    needed_a = {"lane_position_jitter_px"}
    needed_b = {"polyline_pixel_f1", "polyline_pixel_f2_safety"}
    if not (needed_a.issubset(results_df.columns) or needed_b.issubset(results_df.columns)):
        return
    order = [m for m in MODEL_ORDER if m in results_df["model"].values]
    n_panels = int(needed_a.issubset(results_df.columns)) + int(needed_b.issubset(results_df.columns))
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5), squeeze=False)
    axes = axes[0]
    ax_idx = 0

    if needed_a.issubset(results_df.columns):
        ax = axes[ax_idx]; ax_idx += 1
        sns.barplot(data=results_df, x="model", y="lane_position_jitter_px",
                    hue="model", order=order, hue_order=order,
                    palette=MODEL_PALETTE, legend=False, ax=ax)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
        ax.set_title("Frame-to-frame lane jitter (lower = more stable)")
        ax.set_ylabel("mean |Δx| between consecutive frames (orig px)")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    if needed_b.issubset(results_df.columns):
        ax = axes[ax_idx]; ax_idx += 1
        long = results_df.melt(
            id_vars="model", value_vars=["polyline_pixel_f1", "polyline_pixel_f2_safety"],
            var_name="metric", value_name="score",
        )
        sns.barplot(data=long, x="model", y="score", hue="metric", order=order, ax=ax)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=7, padding=2)
        ax.set_title("Pixel F1 vs recall-weighted safety F2")
        ax.set_ylim(0, 1.02)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    _save_figure(fig, out_path)


def summarize_training_histories(histories):
    rows = []
    for key, hist in histories.items():
        if hist is None or hist.empty:
            continue
        best_epoch = _best_epoch(hist)
        best = hist.loc[hist["epoch"] == best_epoch].iloc[0]
        row = {
            "model": MODEL_LABELS.get(key, key),
            "epochs_trained": int(hist["epoch"].max()),
            "best_epoch": int(best_epoch),
            "selection_metric": hist.attrs.get("best_metric_name", "polyline_pixel_f1"),
        }
        for col in ("polyline_pixel_f1", "anchor_f1", "train_comparable_loss",
                    "validation_comparable_loss",
                    "train_x_loss", "train_loss", "validation_loss"):
            if col in best.index:
                row[f"{col}_at_best_epoch"] = float(best[col])
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# 9a. Optuna HPO
# ============================================================

def train_and_eval_variant_trial(
    trial, variant, train_loader, val_loader, val_dataset, cfg_base,
    baseline_ckpt_path, trial_histories_dir=None, tensorboard_root=None,
):
    cfg_trial = copy.deepcopy(cfg_base)
    cfg_trial.tb_log_graph = False 
    cfg_trial.lr = trial.suggest_float("lr", 1e-5, 2e-4, log=True)
    cfg_trial.weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True)
    cfg_trial.exist_threshold = trial.suggest_float("exist_threshold", 0.2, 0.5)
    lambda_temporal = trial.suggest_float("lambda_temporal", 0.05, 2.0, log=True)
    lr_fusion = trial.suggest_float("lr_fusion", 1e-4, 1e-3, log=True)
    cfg_trial.lambda_coord = trial.suggest_float("lambda_coord", 10.0, 50.0, log=True)

    if variant == "v02":
        model = TACUFLDTemporalModel(cfg_trial).to(cfg_trial.device)
        mode = "v02"
    elif variant == "v03":
        model = TACUFLDGatedTemporalModel(cfg_trial).to(cfg_trial.device)
        mode = "v03"
    elif variant == "v04":
        model = TACUFLDTemporalModel(cfg_trial).to(cfg_trial.device)
        mode = "v04"
    else:
        raise ValueError(variant)

    # BUGFIX (this revision): the baseline's backbone (LightweightBackbone,
    # full-width) and every temporal variant's current-frame backbone
    # (LightweightBackboneTemporal, half-width) are now different
    # architectures by design (see LightweightBackboneTemporal's
    # docstring), so there is no shape-compatible backbone warm-start here
    # anymore -- every trial's backbone/history_encoder train from
    # scratch. The head is still warm-startable: load_partial_state_dict
    # copies every head.* tensor whose shape matches (i.e. everything
    # except head.pre.0.weight, whose input-channel count differs) instead
    # of the previous `model.head.load_state_dict(head_state, strict=False)`,
    # which would have raised a RuntimeError on that one mismatched shape
    # even with strict=False (strict=False only tolerates missing/extra
    # KEYS, not a shared key with a different shape).
    if baseline_ckpt_path.exists():
        state = torch.load(baseline_ckpt_path, map_location=cfg_trial.device,
                           weights_only=True)
        head_state = {k[len("head."):]: v for k, v in state.items()
                      if k.startswith("head.")}
        if head_state:
            load_partial_state_dict(
                model.head, head_state,
                label=f"{variant} trial{trial.number} head",
            )

    # ---- per-trial TensorBoard writer (task Section 5) ----
    # Unique log_dir per (variant, trial.number) so TensorBoard's run
    # comparison view / HParams tab can list every trial side by side
    # without collisions, and so pruned trials keep whatever partial
    # history they logged instead of being silently overwritten by the
    # next trial.
    trial_writer = None
    if tensorboard_root is not None and getattr(cfg_trial, "tensorboard_enabled", True):
        trial_writer = get_tb_writer(
            Path(tensorboard_root) / "optuna" / variant / f"trial_{trial.number:04d}"
        )

    history = None
    best_val_f1 = float("nan")
    try:
        history = train_model(
            model, train_loader, val_loader, cfg_trial,
            epochs=OPTUNA_EPOCHS, mode=mode, temporal=True,
            lr_backbone=cfg_trial.lr * 0.1, lr_head=cfg_trial.lr,
            lr_fusion=lr_fusion, lambda_temporal=lambda_temporal,
            save_path=None, trial=trial, val_dataset=val_dataset,
            writer=trial_writer, tb_prefix=f"{variant}_trial{trial.number:04d}",
        )
        best_val_f1 = float(history["polyline_pixel_f1"].max())
        return best_val_f1
    finally:
        if trial_histories_dir is not None:
            trial_histories_dir.mkdir(parents=True, exist_ok=True)
            csv_path = trial_histories_dir / f"{variant}_trial{trial.number}.csv"
            if history is not None and not history.empty:
                history.to_csv(csv_path, index=False)
        if trial_writer is not None:
            try:
                trial_writer.add_hparams(
                    {
                        "lr": cfg_trial.lr,
                        "weight_decay": cfg_trial.weight_decay,
                        "exist_threshold": cfg_trial.exist_threshold,
                        "lambda_temporal": lambda_temporal,
                        "lr_fusion": lr_fusion,
                        "lambda_coord": cfg_trial.lambda_coord,
                        "variant": variant,
                    },
                    {"hparam/best_validation_polyline_pixel_f1": (
                        best_val_f1 if np.isfinite(best_val_f1) else 0.0)},
                )
            except Exception as exc:
                print(f"[tensorboard] add_hparams failed for trial {trial.number}: {exc}")
            close_tb_writer(trial_writer)   # <-- em vez de flush()+close()

        del model
        gc.collect()
        if cfg_trial.device == "cuda":
            torch.cuda.empty_cache()


def run_variant_hpo(variant, train_loader, val_loader, val_dataset, cfg_base,
                    baseline_ckpt_path, result_root, plots_root=None,
                    tensorboard_root=None):
    trial_histories_dir = Path(result_root) / f"trial_histories_{STUDY_VERSION_TAG}"
    study = optuna.create_study(
        direction="maximize",
        study_name=f"ufld_{variant}_hp_tuning_{STUDY_VERSION_TAG}",
        storage=f"sqlite:///{Path(result_root) / f'optuna_{variant}.db'}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(
        lambda trial: train_and_eval_variant_trial(
            trial, variant, train_loader, val_loader, val_dataset, cfg_base,
            baseline_ckpt_path, trial_histories_dir=trial_histories_dir,
            tensorboard_root=tensorboard_root,
        ),
        n_trials=N_OPTUNA_TRIALS,
    )
    return study


# ============================================================
# 10-13. IoU-compatible evaluation / post-processing / export
# ============================================================

@lru_cache(maxsize=None)
def get_original_image_size(image_path):
    with PILImage.open(image_path) as img:
        return img.size


def model_points_to_original(points_model, image_path, cfg):
    orig_w, orig_h = get_original_image_size(str(image_path))
    points_model = np.asarray(points_model, dtype=np.float32)
    points_orig = points_model.copy()
    points_orig[:, 0] = points_orig[:, 0] * (orig_w / cfg.img_w)
    points_orig[:, 1] = points_orig[:, 1] * (orig_h / cfg.img_h)
    return points_orig


IOU_MATCH_THRESHOLD = 0.35
PIXEL_LINE_WIDTH = 12   # reduced from 10 for tighter, more informative pixel F1 / atual (~10-17px) sem "inflar" artificialmente


def lane_to_mask(lane_points, width, height, line_width=PIXEL_LINE_WIDTH):
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.asarray(lane_points, dtype=np.float32)
    if len(pts) < 2:
        return mask
    valid = (np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
             & (pts[:, 0] >= 0) & (pts[:, 0] < width)
             & (pts[:, 1] >= 0) & (pts[:, 1] < height))
    pts = pts[valid]
    if len(pts) < 2:
        return mask
    pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts], isClosed=False, color=1,
                  thickness=line_width, lineType=cv2.LINE_8)
    return mask


def compute_lane_iou(pred_lane, gt_lane, width, height, line_width=PIXEL_LINE_WIDTH):
    pm = lane_to_mask(pred_lane, width, height, line_width)
    gm = lane_to_mask(gt_lane, width, height, line_width)
    inter = np.logical_and(pm, gm).sum()
    union = np.logical_or(pm, gm).sum()
    return float(inter / union) if union > 0 else 0.0


def match_lanes_by_iou(pred_lanes, gt_lanes, width, height,
                       iou_threshold=IOU_MATCH_THRESHOLD,
                       line_width=PIXEL_LINE_WIDTH):
    if len(pred_lanes) == 0 and len(gt_lanes) == 0:
        return 0, 0, 0, []
    if len(pred_lanes) == 0:
        return 0, 0, len(gt_lanes), []
    if len(gt_lanes) == 0:
        return 0, len(pred_lanes), 0, []

    candidates = []
    for i, p in enumerate(pred_lanes):
        for j, g in enumerate(gt_lanes):
            iou = compute_lane_iou(p, g, width, height, line_width=line_width)
            candidates.append((iou, i, j))
    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)




    matched_p, matched_g, matches = set(), set(), []
    for iou, i, j in candidates:
        if iou < iou_threshold:
            break
        if i in matched_p or j in matched_g:
            continue
        matched_p.add(i); matched_g.add(j)
        matches.append({"pred_idx": i, "gt_idx": j, "iou": float(iou)})

    tp = len(matches)
    return tp, len(pred_lanes) - tp, len(gt_lanes) - tp, matches


@torch.inference_mode()
def build_prediction_cache(model, dataset, cfg, temporal=True, ablation_mode=None):
    model.eval()
    previous_ablation = getattr(model, "ablation_mode", None)
    if hasattr(model, "ablation_mode"):
        model.ablation_mode = ablation_mode

    cache = []
    try:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            images = sample["images"].unsqueeze(0).to(cfg.device)
            amp_context = (torch.amp.autocast("cuda")
                           if cfg.device == "cuda" else nullcontext())
            with amp_context:
                if temporal:
                    outputs = model(images)
                elif hasattr(model, "forward_per_frame"):
                    outputs = model.forward_per_frame(images)[-1]
                else:
                    outputs = model(images[:, -1])

            x_px, y_px, pred_prob, _ = decode_outputs(outputs, cfg)
            x_px = x_px[0].detach().cpu().numpy()          # (L, A)
            y_px = y_px[0].detach().cpu().numpy()          # (L, A)
            pred_prob = pred_prob[0].detach().cpu().numpy()  # (L, A)

            row = dataset.rows[idx]
            gt = dataset.scene_records[row["scene"]]["gt_frames"][row["frame_id"]]
            gt_lanes = [lane for lane in (gt["left"], gt["right"]) if lane is not None]

            # gate_mean: additive field, only present for temporal models
            # whose outputs dict has a "gate" tensor (ResidualTemporalFusion
            # / its ablation variants). None for the single-frame baseline.
            # Consumed by the temporal-consistency video (task Section 7:
            # "show confidence scores, temporal weights or gating
            # information if these are actually available") -- never
            # invented for models that don't produce one.
            gate_mean = None
            if isinstance(outputs, dict) and "gate" in outputs and outputs["gate"] is not None:
                gate_mean = float(outputs["gate"][0].mean().detach().cpu())

            cache.append({
                "idx": idx,
                "image_path": str(sample["image_path"]),
                "scene": row["scene"],
                "frame_id": row["frame_id"],
                "tags": row["tags"],
                "scene_cfg": row.get("scene_cfg"),
                "gt_lanes": gt_lanes,
                "width": int(sample["orig_w"]),
                "height": int(sample["orig_h"]),
                "pred_prob": pred_prob,
                "pred_x_px": x_px,
                "pred_y_px": y_px,
                "gate_mean": gate_mean,
            })
    finally:
        if hasattr(model, "ablation_mode"):
            model.ablation_mode = previous_ablation
    return cache


# ============================================================
# 11. Geometric post-processing
# ============================================================

def refine_lane_polyfit(points_model, image_path, cfg, params,
                        y_model_min=None, y_model_max=None):
    points_model = np.asarray(points_model, dtype=np.float32)
    min_points = params["min_points"]
    y_step = params["y_step"]
    poly_degree = params["poly_degree"]

    if "min_lane_length_fraction" in params:
        if (y_model_min is not None) and (y_model_max is not None):
            roi_span = max(float(y_model_max) - float(y_model_min), 1.0)
        else:
            roi_span = float(cfg.img_h)
        min_lane_length = params["min_lane_length_fraction"] * roi_span
    else:
        min_lane_length = params.get("min_lane_length", 30)

    if len(points_model) < min_points:
        return None

    valid = (np.isfinite(points_model[:, 0]) & np.isfinite(points_model[:, 1])
             & (points_model[:, 0] >= 0) & (points_model[:, 0] < cfg.img_w)
             & (points_model[:, 1] >= 0) & (points_model[:, 1] < cfg.img_h))
    points_model = points_model[valid]
    if len(points_model) < min_points:
        return None

    if y_model_min is not None:
        points_model = points_model[points_model[:, 1] >= float(y_model_min)]
    if y_model_max is not None:
        points_model = points_model[points_model[:, 1] <= float(y_model_max)]
    if len(points_model) < min_points:
        return None

    points_model = points_model[np.argsort(points_model[:, 1])]
    x, y = points_model[:, 0], points_model[:, 1]
    y_min = float(y.min()) if y_model_min is None else max(float(y.min()), float(y_model_min))
    y_max = float(y.max()) if y_model_max is None else min(float(y.max()), float(y_model_max))
    if y_max - y_min < min_lane_length:
        return None

    unique_y, unique_idx = np.unique(y, return_index=True)
    unique_x = x[unique_idx]
    if len(unique_y) < min_points:
        return None

    degree = min(poly_degree, len(unique_y) - 1)
    try:
        coeff = np.polyfit(unique_y, unique_x, deg=degree)
    except Exception:
        return None

    y_dense = np.arange(y_min, y_max + 1, y_step, dtype=np.float32)
    x_dense = np.polyval(coeff, y_dense).astype(np.float32)
    refined_model = np.stack([x_dense, y_dense], axis=1)

    valid_model = ((refined_model[:, 0] >= 0) & (refined_model[:, 0] < cfg.img_w)
                   & (refined_model[:, 1] >= 0) & (refined_model[:, 1] < cfg.img_h))
    refined_model = refined_model[valid_model]
    if len(refined_model) < min_points:
        return None

    refined_orig = model_points_to_original(refined_model, image_path, cfg)
    orig_w, orig_h = get_original_image_size(str(image_path))
    valid_orig = ((refined_orig[:, 0] >= 0) & (refined_orig[:, 0] < orig_w)
                  & (refined_orig[:, 1] >= 0) & (refined_orig[:, 1] < orig_h))
    refined_orig = refined_orig[valid_orig]
    return refined_orig if len(refined_orig) >= min_points else None


def suppress_duplicate_lanes(lanes, duplicate_distance=15):
    if len(lanes) <= 1:
        return lanes
    infos = []
    for lane in lanes:
        if lane is None or len(lane) < 2:
            continue
        infos.append({
            "lane": lane,
            "mean_x": float(np.mean(lane[:, 0])),
            "length": float(lane[:, 1].max() - lane[:, 1].min()),
        })
    infos = sorted(infos, key=lambda d: d["length"], reverse=True)
    kept = []
    for info in infos:
        if any(abs(info["mean_x"] - k["mean_x"]) < duplicate_distance for k in kept):
            continue
        kept.append(info)
    kept = sorted(kept, key=lambda d: d["mean_x"])
    return [d["lane"] for d in kept]


def get_eval_roi_model(item, cfg):
    scene_cfg = item.get("scene_cfg")
    if not scene_cfg:
        return 0.0, float(cfg.img_h - 1)
    orig_h = float(item.get("height", 480))
    y0 = float(scene_cfg.get("roi_y", 0))
    y1 = y0 + float(scene_cfg.get("roi_h", orig_h))
    y0_model = np.clip(y0 * cfg.img_h / max(orig_h, 1.0), 0, cfg.img_h - 1)
    y1_model = np.clip(y1 * cfg.img_h / max(orig_h, 1.0), 0, cfg.img_h - 1)
    return float(min(y0_model, y1_model)), float(max(y0_model, y1_model))


def postprocess_item(item, cfg, params):
    threshold = params["threshold"]
    pred_prob = item["pred_prob"]
    x_px = item["pred_x_px"]
    y_px = item["pred_y_px"]

    L, A = pred_prob.shape
    lanes = []
    roi_y_min, roi_y_max = get_eval_roi_model(item, cfg)

    for lane_id in range(L):
        pts = []
        for a in range(A):
            if pred_prob[lane_id, a] >= threshold:
                x = float(x_px[lane_id, a])
                y = float(y_px[lane_id, a])
                if np.isfinite(x) and np.isfinite(y) and roi_y_min <= y <= roi_y_max:
                    pts.append([x, y])
        if len(pts) < params["min_points"]:
            continue
        refined = refine_lane_polyfit(
            np.asarray(pts, dtype=np.float32),
            item["image_path"], cfg, params,
            y_model_min=roi_y_min, y_model_max=roi_y_max,
        )
        if refined is not None:
            lanes.append(refined)
    return suppress_duplicate_lanes(lanes, duplicate_distance=params["duplicate_distance"])


def compute_pixel_f1(pred_lanes, gt_lanes, width, height, line_width=PIXEL_LINE_WIDTH):
    pred_mask = np.zeros((height, width), dtype=np.uint8)
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    for lane in pred_lanes:
        pred_mask |= lane_to_mask(lane, width, height, line_width)
    for lane in gt_lanes:
        gt_mask |= lane_to_mask(lane, width, height, line_width)
    tp = int(np.logical_and(pred_mask > 0, gt_mask > 0).sum())
    fp = int(np.logical_and(pred_mask > 0, gt_mask == 0).sum())
    fn = int(np.logical_and(pred_mask == 0, gt_mask > 0).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1, tp, fp, fn


def _fbeta_score(precision, recall, beta=2.0):
    """F-beta: weights recall `beta` times as heavily as precision.
    beta=1 reduces to the usual F1. Used for the ADAS "safety" score, where
    a missed lane (false negative) is a worse failure mode than a spurious
    one (false positive)."""
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / max(denom, 1e-12)


def evaluate_iou_cache(cache, cfg, params, model_name, verbose=True):
    total_tp = total_fp = total_fn = 0
    pixel_tp = pixel_fp = pixel_fn = 0
    total_gt = total_pred = 0
    empty_files = 0

    diag_lane_candidates = 0
    diag_refined_ok = 0

    per_item = []
    for item in cache:
        pred_lanes = postprocess_item(item, cfg, params)
        gt_lanes = item["gt_lanes"]
        pred_prob = item["pred_prob"]
        for lane_id in range(pred_prob.shape[0]):
            if int((pred_prob[lane_id] >= params["threshold"]).sum()) > 0:
                diag_lane_candidates += 1
        diag_refined_ok += len(pred_lanes)

        tp, fp, fn, _ = match_lanes_by_iou(
            pred_lanes, gt_lanes, item["width"], item["height"],
            iou_threshold=IOU_MATCH_THRESHOLD, line_width=PIXEL_LINE_WIDTH,
        )
        per_item.append((item, tp, fp, fn))
        total_tp += tp; total_fp += fp; total_fn += fn

        _, _, _, p_tp, p_fp, p_fn = compute_pixel_f1(
            pred_lanes, gt_lanes, item["width"], item["height"],
            line_width=PIXEL_LINE_WIDTH,
        )
        pixel_tp += p_tp; pixel_fp += p_fp; pixel_fn += p_fn
        total_gt += len(gt_lanes); total_pred += len(pred_lanes)
        if len(pred_lanes) == 0:
            empty_files += 1

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    detection_accuracy = total_tp / max(total_tp + total_fp + total_fn, 1)

    pixel_precision = pixel_tp / max(pixel_tp + pixel_fp, 1)
    pixel_recall = pixel_tp / max(pixel_tp + pixel_fn, 1)
    pixel_f1 = 2 * pixel_precision * pixel_recall / max(
        pixel_precision + pixel_recall, 1e-12)
    # ADAS "safety" score: same pixel-level TP/FP/FN as polyline_pixel_f1,
    # but weighted so a missed lane (FN) costs more than a spurious one
    # (FP). Report alongside polyline_pixel_f1 rather than replacing it --
    # picking a deployment operating point is a decision for whoever owns
    # the downstream controller/consistency-filter behaviour, not something
    # to bake silently into a single number.
    pixel_f2_safety = _fbeta_score(pixel_precision, pixel_recall, beta=ADAS_SAFETY_FBETA)

    condition_f1 = {}
    for category in ("rainy", "occlusion", "shaky", "transition"):
        selected = [(tp, fp, fn) for item, tp, fp, fn in per_item if item["tags"].get(category)]
        if not selected:
            condition_f1[f"iou_f1_{category}"] = float("nan"); continue
        c_tp = sum(s[0] for s in selected)
        c_fp = sum(s[1] for s in selected)
        c_fn = sum(s[2] for s in selected)
        cp = c_tp / max(c_tp + c_fp, 1)
        cr = c_tp / max(c_tp + c_fn, 1)
        condition_f1[f"iou_f1_{category}"] = 2 * cp * cr / max(cp + cr, 1e-8)

    metrics_result = {
        "model": model_name,
        "total_gt_lanes": total_gt, "total_pred_lanes": total_pred,
        "iou_precision": precision, "iou_recall": recall, "iou_f1": f1,
        "iou_detection_accuracy": detection_accuracy,
        "pixel_precision": pixel_precision, "pixel_recall": pixel_recall,
        "polyline_pixel_f1": pixel_f1,
        "polyline_pixel_f2_safety": pixel_f2_safety,
        "pixel_true_positives": pixel_tp,
        "pixel_false_positives": pixel_fp,
        "pixel_false_negatives": pixel_fn,
        "empty_prediction_files": empty_files,
        **condition_f1,
    }
    confusion_counts = {
        "true_positives": total_tp, "false_positives": total_fp, "false_negatives": total_fn,
    }
    if verbose:
        print(
            f"[diag] {model_name}: candidates={diag_lane_candidates}, "
            f"refined_ok={diag_refined_ok}, "
            f"pixel_TP={pixel_tp} pixel_FP={pixel_fp} pixel_FN={pixel_fn}, "
            f"polyline_pixel_f1={pixel_f1:.4f}, "
            f"polyline_pixel_f2_safety={pixel_f2_safety:.4f}"
        )
    return metrics_result, confusion_counts


def compute_temporal_jitter(cache, cfg, exist_threshold=None):
    """ADAS-relevant complement to the per-frame accuracy metrics above:
    the mean frame-to-frame change in predicted lane x-position, in
    original-image pixels, at (lane, anchor) positions where the model
    reports the lane present in both of two *consecutive* frames
    (frame_id delta == 1) of the same scene.

    Why this matters for the "temporal fusion should beat the single-frame
    baseline" objective: once the baseline's coordinate-regression bug is
    fixed, its per-frame accuracy can approach (or beat) the temporal
    variants', because per-frame accuracy was never something temporal
    fusion is uniquely positioned to improve. Frame-to-frame stability is:
    a lane detector feeding a steering-assist controller that jumps left
    and right between frames is unusable even at high per-frame F1. This
    is exactly the axis the warped/gated fusion models are supposed to buy
    in exchange for their ~2x latency and parameter cost, so it belongs in
    every comparison table alongside accuracy, not as an afterthought.

    Caveat: the train/val/test subsets are drawn from leakage-safe blocks
    that are shuffled across splits (see prepare_scene_splits), so only a
    fraction of frame pairs in `cache` are actually consecutive in the
    source video. `jitter_n_pairs` reports how many (lane, anchor)
    comparisons the estimate is based on -- treat a low count as a
    low-confidence estimate rather than silently trusting the mean, and
    consider building a small dedicated temporally-contiguous eval split
    if this needs to be reported with confidence.
    """
    if exist_threshold is None:
        exist_threshold = cfg.exist_threshold

    by_scene = {}
    for item in cache:
        by_scene.setdefault(item["scene"], []).append(item)

    diffs = []
    for items in by_scene.values():
        items = sorted(items, key=lambda it: it["frame_id"])
        for prev, cur in zip(items, items[1:]):
            if cur["frame_id"] - prev["frame_id"] != 1:
                continue
            scale = cur["width"] / max(cfg.img_w, 1)
            mask = (prev["pred_prob"] >= exist_threshold) & (cur["pred_prob"] >= exist_threshold)
            if not mask.any():
                continue
            dx = np.abs(cur["pred_x_px"][mask] - prev["pred_x_px"][mask]) * scale
            diffs.extend(dx.tolist())

    if not diffs:
        return {
            "lane_position_jitter_px": float("nan"),
            "lane_position_jitter_p95_px": float("nan"),
            "jitter_n_pairs": 0,
        }
    diffs = np.asarray(diffs, dtype=np.float64)
    return {
        "lane_position_jitter_px": float(diffs.mean()),
        "lane_position_jitter_p95_px": float(np.percentile(diffs, 95)),
        "jitter_n_pairs": int(len(diffs)),
    }


def sweep_postprocessing(model, val_dataset, cfg, temporal, param_grid, base_params=None):
    from itertools import product
    base = dict(base_params or {})
    cache = build_prediction_cache(model, val_dataset, cfg, temporal=temporal)
    keys = list(param_grid.keys())
    best = None
    rows = []
    for combo in product(*[param_grid[k] for k in keys]):
        params = dict(base)
        params.update(dict(zip(keys, combo)))
        metrics, _ = evaluate_iou_cache(cache, cfg, params, model_name="sweep", verbose=False)
        row = {**params,
               "iou_f1": metrics["iou_f1"],
               "polyline_pixel_f1": metrics["polyline_pixel_f1"],
               "pixel_precision": metrics["pixel_precision"],
               "pixel_recall": metrics["pixel_recall"],
               "iou_precision": metrics["iou_precision"],
               "iou_recall": metrics["iou_recall"]}
        rows.append(row)
        if best is None or row["iou_f1"] > best["iou_f1"]:
            best = row
    return best, pd.DataFrame(rows).sort_values("iou_f1", ascending=False)


def save_lanes_txt(lanes, out_path):
    """BUGFIX (this revision): a full disk previously surfaced here as an
    uncaught OSError [Errno 28] "No space left on device", killing the
    entire multi-seed run at the very last step (export) after training,
    HPO, evaluation and every plot for that seed had already succeeded.
    Exports are the least valuable artifact to lose at that point --
    report and skip this one file instead of propagating."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for lane in lanes:
                lane = lane[np.argsort(lane[:, 1])]
                values = []
                for x, y in lane:
                    values.append(f"{x:.2f}"); values.append(f"{y:.2f}")
                f.write(" ".join(values) + "\n")
    except OSError as exc:
        print(f"[export] failed to write {out_path}: {exc}. Skipping this "
              f"file (disk may be full -- see any disk-space warning above).")


def export_cache_predictions(cache, cfg, params, pred_root):
    pred_root = Path(pred_root)
    pred_root.mkdir(parents=True, exist_ok=True)
    _check_disk_space(pred_root, min_free_mb=200, label=f"export to {pred_root.name}")
    rows = []
    for item in cache:
        lanes = postprocess_item(item, cfg, params)
        stem = f"{item['scene']}_{item['frame_id']}"
        out_path = pred_root / f"{stem}.lines.txt"
        save_lanes_txt(lanes, out_path)
        rows.append({
            "idx": item["idx"], "scene": item["scene"], "frame_id": item["frame_id"],
            "prediction_path": str(out_path), "num_lanes_saved": len(lanes),
        })
    return pd.DataFrame(rows)


# ============================================================
# 14. Efficiency metrics
# ============================================================

try:
    import resource
except ImportError:
    resource = None


def _process_memory_mb():
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage / (1024.0 if os.name == "posix" else 1.0)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = PMC()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        get_memory.restype = wintypes.BOOL
        if get_memory(handle, ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / (1024.0 ** 2)
    return float("nan")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _count_macs_with_hooks(model, dummy_input):
    macs = [0]
    handles = []
    def conv_hook(module, inp, out):
        out_h, out_w = out.shape[-2], out.shape[-1]
        in_ch_per_group = module.in_channels // module.groups
        kh, kw = module.kernel_size
        macs[0] += out.shape[0] * module.out_channels * out_h * out_w * in_ch_per_group * kh * kw
    def linear_hook(module, inp, out):
        macs[0] += out.shape[0] * module.in_features * module.out_features
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
    try:
        with torch.no_grad():
            model(dummy_input)
    finally:
        for h in handles:
            h.remove()
    return macs[0]


def estimate_macs_flops(model, cfg, temporal, ablation_mode=None):
    was_training = model.training
    model.eval()
    prev = getattr(model, "ablation_mode", None)
    if hasattr(model, "ablation_mode"):
        model.ablation_mode = ablation_mode
    try:
        if temporal:
            dummy = torch.randn(1, cfg.num_frames, 3, cfg.img_h, cfg.img_w, device=cfg.device)
        else:
            dummy = torch.randn(1, 3, cfg.img_h, cfg.img_w, device=cfg.device)
        macs = _count_macs_with_hooks(model, dummy)
    finally:
        if hasattr(model, "ablation_mode"):
            model.ablation_mode = prev
        model.train(was_training)
    return macs, macs * 2


@torch.no_grad()
def measure_latency_fps_memory(model, cfg, temporal, n_runs=30, warmup=5):
    model.eval()
    device = cfg.device
    if temporal:
        dummy = torch.randn(1, cfg.num_frames, 3, cfg.img_h, cfg.img_w, device=device)
    else:
        dummy = torch.randn(1, 3, cfg.img_h, cfg.img_w, device=device)
    for _ in range(warmup):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(n_runs):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    latency_ms = (elapsed / n_runs) * 1000.0
    fps = 1000.0 / latency_ms if latency_ms > 0 else float("nan")
    if device == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory_mb = _process_memory_mb()
    return latency_ms, fps, peak_memory_mb


def compute_model_efficiency(model, cfg, temporal, label, ablation_mode=None):
    total_params, trainable = count_parameters(model)
    macs, flops = estimate_macs_flops(model, cfg, temporal, ablation_mode=ablation_mode)
    latency_ms, fps, peak_memory_mb = measure_latency_fps_memory(model, cfg, temporal)
    return {
        "model": label,
        "parameters_total": total_params,
        "parameters_trainable": trainable,
        "parameters_total_millions": total_params / 1e6,
        "giga_macs": macs / 1e9,
        "giga_flops": flops / 1e9,
        "latency_milliseconds": latency_ms,
        "frames_per_second": fps,
        "peak_memory_megabytes": peak_memory_mb,
        "meets_realtime_budget": bool(latency_ms <= ADAS_LATENCY_BUDGET_MS),
        "realtime_budget_milliseconds": ADAS_LATENCY_BUDGET_MS,
    }


def sanity_check_plot(dataset, out_path):
    sample = dataset[0]
    cls_target = sample["cls_target"].numpy()
    exist_target = sample["exist_target"].numpy()
    fig, ax = plt.subplots(figsize=(4, 6))
    for lane_id in range(cls_target.shape[0]):
        ys = np.arange(cls_target.shape[1])
        xs = np.where(exist_target[lane_id] > 0, cls_target[lane_id], np.nan)
        ax.plot(xs, ys, marker="o", label=f"lane {lane_id}")
    ax.invert_yaxis()
    ax.set_xlabel("grid bin"); ax.set_ylabel("row anchor index")
    ax.set_title(f"frame_id={sample['frame_id']}")
    ax.legend()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_sanity_check():
    """Standalone: verify the coordinate pipeline before any training.
    Prints target and model-coordinate ranges and produces a visualisation
    of GT lanes in model space overlayed with what the decoder would produce
    for a random output."""
    print("Running coordinate-pipeline sanity check ...")
    split_rows, scene_records, _ = prepare_scene_splits(
        DATASET_ROOT, SCENE_PLAN,
        {"train": 50, "val": 50, "test": 50}, seed=cfg.seed,
    )
    ds = ELASTemporalDataset(split_rows["test"], scene_records, cfg, augment=False)
    sample = ds[0]

    expect_imgs = (cfg.num_frames, 3, cfg.img_h, cfg.img_w)
    expect_cls = (cfg.num_lanes, cfg.num_row_anchors)
    print(f"images shape      : {tuple(sample['images'].shape)}   (expect {expect_imgs})")
    print(f"cls_target shape  : {tuple(sample['cls_target'].shape)}   (expect {expect_cls})")
    print(f"exist_target shape: {tuple(sample['exist_target'].shape)}")
    print(f"coords_target shp : {tuple(sample['coords_target'].shape)}   (expect {expect_cls + (2,)})")
    print(f"orig size         : {sample['orig_w']} x {sample['orig_h']}")

    ct = sample["coords_target"].numpy()
    et = sample["exist_target"].numpy()
    mask = et > 0.5
    if mask.any():
        xs = ct[..., 0][mask]
        ys = ct[..., 1][mask]
        print(f"x range in model px: [{xs.min():.1f}, {xs.max():.1f}]  (image width {cfg.img_w})")
        print(f"y range in model px: [{ys.min():.1f}, {ys.max():.1f}]  (image height {cfg.img_h})")
        if xs.min() < 0 or xs.max() >= cfg.img_w:
            print("[ERROR] x out of range in model space.")
        if ys.min() < 0 or ys.max() >= cfg.img_h:
            print("[ERROR] y out of range in model space.")
    else:
        print("[WARN] no positive anchors in sample 0")

    # Random-output decode test
    # Random-output decode test (single-frame model takes the last frame)
    model = SingleFrameUFLDLikeModel(cfg).to(cfg.device).eval()
    with torch.no_grad():
        last_frame = sample["images"][-1].unsqueeze(0).to(cfg.device)  # (1, 3, H, W)
        fake = model(last_frame)
    x_px, y_px, prob, _ = decode_outputs(fake, cfg)
    print(f"head output keys  : {list(fake.keys())}")
    print(f"x_px shape        : {tuple(x_px.shape)}")
    print(f"x_px range        : [{float(x_px.min()):.2f}, {float(x_px.max()):.2f}]")
    print(f"y_px range        : [{float(y_px.min()):.2f}, {float(y_px.max()):.2f}]")
    print("Sanity check done. If the ranges look sensible, run the full pipeline.")
    n_pos = int((et > 0.5).sum())
    print(f"positive anchors: {n_pos} / {et.size}   (per lane: "
          f"{[int((et[l] > 0.5).sum()) for l in range(et.shape[0])]})")
    print(f"anchors covered (y-extent per lane):")
    for l in range(et.shape[0]):
        ys = ct[l, et[l] > 0.5, 1]
        if len(ys) > 0:
            print(f"  lane {l}: y in [{ys.min():.1f}, {ys.max():.1f}]  "
                  f"({len(ys)} anchors)")
    # Add to run_sanity_check, after the per-lane print:
    counts = []
    for i in range(min(200, len(ds))):
        s = ds[i]
        et = s["exist_target"].numpy()
        counts.append((int((et[0] > 0.5).sum()), int((et[1] > 0.5).sum())))
    counts = np.array(counts)
    print(f"\nAcross 200 val samples:")
    print(f"  frames with lane 0 only: {(counts[:,0] > 0).sum() - ((counts[:,0] > 0) & (counts[:,1] > 0)).sum()}")
    print(f"  frames with lane 1 only: {(counts[:,1] > 0).sum() - ((counts[:,0] > 0) & (counts[:,1] > 0)).sum()}")
    print(f"  frames with both lanes : {((counts[:,0] > 0) & (counts[:,1] > 0)).sum()}")
    print(f"  frames with neither    : {((counts[:,0] == 0) & (counts[:,1] == 0)).sum()}")           
    # Coverage across the val set

    pos_per_frame = []
    for i in range(len(ds)):
        s = ds[i]
        et = s["exist_target"].numpy()
        pos_per_frame.append(int((et > 0.5).sum()))
    pos_per_frame = np.array(pos_per_frame)
    print(f"\npositive anchors per frame across val:")
    print(f"  min={pos_per_frame.min()}, max={pos_per_frame.max()}, "
      f"mean={pos_per_frame.mean():.1f}, median={np.median(pos_per_frame):.0f}")
    print(f"  frames with <20 positives: {(pos_per_frame < 20).sum()}/{len(pos_per_frame)}")
    print(f"  frames with >40 positives: {(pos_per_frame > 40).sum()}/{len(pos_per_frame)}")
# ============================================================
# 14a. Qualitative scenario comparison images (normal vs detected)
# ============================================================

_SCENARIO_TAG_ORDER = [
    ("rainy", "Rainy"), ("occlusion", "Occlusion"),
    ("shaky", "Shaky camera"), ("transition", "Lighting transition"),
    (None, "Clear / nominal"),
]
_COLOR_GT = (0, 220, 0)       # green  (BGR, cv2)
_COLOR_PRED = (0, 0, 255)     # red

def _draw_lanes(image_bgr, lanes, color, thickness=3):
    for lane in lanes:
        pts = np.asarray(lane, dtype=np.float32)
        if len(pts) < 2:
            continue
        pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image_bgr, [pts_i], isClosed=False, color=color,
                     thickness=thickness, lineType=cv2.LINE_AA)
    return image_bgr


def save_scenario_comparison_images(cache, cfg, params, model_label, out_dir,
                                    max_per_tag=1):
    """Save 'original | ground truth | detected' comparison panels for one
    representative frame per scenario tag (rainy / occlusion / shaky /
    transition / clear), so it is visually clear what quality level each
    polyline_pixel_f1 / iou_f1 value in the CSVs actually corresponds to.
    Returns the list of saved file paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    used_frames = set()
    for tag_key, tag_label in _SCENARIO_TAG_ORDER:
        candidates = [
            item for item in cache
            if (item["tags"].get(tag_key) if tag_key is not None
                else not any(item["tags"].values()))
            and item["image_path"] not in used_frames
        ]
        if not candidates:
            continue
        for item in candidates[:max_per_tag]:
            used_frames.add(item["image_path"])
            image = cv2.imread(item["image_path"])
            if image is None:
                continue
            pred_lanes = postprocess_item(item, cfg, params)
            gt_lanes = item["gt_lanes"]

            panel_original = image.copy()
            panel_gt = _draw_lanes(image.copy(), gt_lanes, _COLOR_GT)
            panel_pred = _draw_lanes(image.copy(), pred_lanes, _COLOR_PRED)

            for panel, caption in ((panel_original, "Original"),
                                   (panel_gt, "Ground truth"),
                                   (panel_pred, "Detected")):
                cv2.putText(panel, caption, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(panel, caption, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (0, 0, 0), 1, cv2.LINE_AA)

            combined = cv2.hconcat([panel_original, panel_gt, panel_pred])
            fname = f"{model_label.replace(' ', '_')}_{tag_label.replace(' ', '_')}_{item['scene']}_{item['frame_id']}.png"
            out_path = out_dir / fname
            cv2.imwrite(str(out_path), combined)
            saved.append(out_path)
    return saved


def save_before_after_images(cache, cfg, params, model_label, out_dir, n_samples=3):
    """(this revision) Simple two-panel deliverable, requested explicitly
    alongside the richer three-panel save_scenario_comparison_images
    above: for each model, the SAME street frame shown once WITHOUT any
    overlay and once WITH the model's decoded lane detections drawn on
    top (prediction only -- no ground truth), so results can be eyeballed
    at a glance without reading a CSV. Called once per model at the end
    of run_pipeline; writes to plots_root/before_after/."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for item in cache[:max(n_samples, 0)]:
        image = cv2.imread(item["image_path"])
        if image is None:
            continue
        pred_lanes = postprocess_item(item, cfg, params)
        panel_before = image.copy()
        panel_after = _draw_lanes(image.copy(), pred_lanes, _COLOR_PRED, thickness=3)
        for panel, caption in ((panel_before, "Without lane detection"),
                               (panel_after, "With lane detection")):
            cv2.putText(panel, caption, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, caption, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 0, 0), 1, cv2.LINE_AA)
        combined = cv2.hconcat([panel_before, panel_after])
        fname = (f"{model_label.replace(' ', '_')}_{item['scene']}_"
                 f"{item['frame_id']}_before_after.png")
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), combined)
        saved.append(out_path)
    return saved


# ============================================================
# 14b. False-positive / false-negative visual error inspection
# (task Section 6, "Error inspection")
# ============================================================
#
# Classification uses the SAME evaluation primitive the reported metrics
# come from -- match_lanes_by_iou() at IOU_MATCH_THRESHOLD, the identical
# function evaluate_iou_cache() calls -- rather than a new, separately-
# invented notion of "looks wrong". A predicted lane with no matching GT
# lane at that IoU threshold is exactly what the iou_f1/iou_precision
# numbers already count as a false positive; same for an unmatched GT
# lane and false negatives. This function only decides which frames are
# worth saving as pictures, it does not compute a different metric.

def save_error_inspection_images(cache, cfg, params, model_label, out_root,
                                 max_examples=None):
    """Save individual false-positive and false-negative examples (task
    Section 6). `cache` should be a VALIDATION-set prediction cache (per
    the task's "selected examples are drawn only from the validation set
    during training and model selection" -- pass a cache built from
    val_dataset, not test_dataset, when this is used for anything that
    could influence model/threshold selection; a test-set cache is fine
    only for the final qualitative report figures, not for tuning).
    Writes to `out_root/false_positives/` and `out_root/false_negatives/`.
    Returns {"false_positives": [...], "false_negatives": [...]} of saved
    paths."""
    max_examples = max_examples or 12
    out_root = Path(out_root)
    fp_dir = out_root / "false_positives"
    fn_dir = out_root / "false_negatives"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    saved = {"false_positives": [], "false_negatives": []}
    for item in cache:
        if len(saved["false_positives"]) >= max_examples and len(saved["false_negatives"]) >= max_examples:
            break
        pred_lanes = postprocess_item(item, cfg, params)
        gt_lanes = item["gt_lanes"]
        tp, fp, fn, matches = match_lanes_by_iou(
            pred_lanes, gt_lanes, item["width"], item["height"],
            iou_threshold=IOU_MATCH_THRESHOLD, line_width=PIXEL_LINE_WIDTH,
        )
        if fp == 0 and fn == 0:
            continue

        image = cv2.imread(item["image_path"])
        if image is None:
            continue
        matched_pred_idx = {m["pred_idx"] for m in matches}
        matched_gt_idx = {m["gt_idx"] for m in matches}

        if fp > 0 and len(saved["false_positives"]) < max_examples:
            panel = image.copy()
            panel = _draw_lanes(panel, gt_lanes, _COLOR_GT, thickness=2)
            fp_lanes = [ln for i, ln in enumerate(pred_lanes) if i not in matched_pred_idx]
            panel = _draw_lanes(panel, fp_lanes, (0, 0, 255), thickness=3)  # highlight in red
            other_pred = [ln for i, ln in enumerate(pred_lanes) if i in matched_pred_idx]
            panel = _draw_lanes(panel, other_pred, (0, 165, 255), thickness=1)  # matched, dim
            label = f"FALSE POSITIVE  {item['scene']} frame={item['frame_id']}  ({fp} unmatched pred lane(s))"
            cv2.putText(panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                       (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                       (0, 0, 200), 1, cv2.LINE_AA)
            out_path = fp_dir / f"{model_label.replace(' ', '_')}_{item['scene']}_{item['frame_id']}.png"
            cv2.imwrite(str(out_path), panel)
            saved["false_positives"].append(out_path)

        if fn > 0 and len(saved["false_negatives"]) < max_examples:
            panel = image.copy()
            panel = _draw_lanes(panel, pred_lanes, _COLOR_PRED, thickness=1)
            fn_lanes = [ln for i, ln in enumerate(gt_lanes) if i not in matched_gt_idx]
            panel = _draw_lanes(panel, fn_lanes, (0, 220, 0), thickness=3)  # highlight in green
            other_gt = [ln for i, ln in enumerate(gt_lanes) if i in matched_gt_idx]
            panel = _draw_lanes(panel, other_gt, (0, 120, 0), thickness=1)  # matched, dim
            label = f"FALSE NEGATIVE  {item['scene']} frame={item['frame_id']}  ({fn} missed GT lane(s))"
            cv2.putText(panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                       (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                       (0, 140, 0), 1, cv2.LINE_AA)
            out_path = fn_dir / f"{model_label.replace(' ', '_')}_{item['scene']}_{item['frame_id']}.png"
            cv2.imwrite(str(out_path), panel)
            saved["false_negatives"].append(out_path)

    return saved


def save_temporal_impact_examples(baseline_cache, temporal_cache, cfg,
                                  baseline_params, temporal_params,
                                  temporal_label, out_root, max_examples=None,
                                  f1_margin=0.10):
    """Task Section 6: "cases where temporal fusion worsens the
    prediction" / "corrects an otherwise incorrect prediction". Compares
    the SAME per-item pixel-F1 (compute_pixel_f1, the function
    evaluate_iou_cache's polyline_pixel_f1 is built from) between the
    single-frame baseline and a temporal variant, matched by dataset
    index `idx` (both caches come from build_prediction_cache() over the
    same dataset, so idx identifies the same frame in both). This does
    NOT invent a new error category: a frame only counts as
    "temporal-corrected" or "temporal-worsened" if the two caches'
    existing pixel-F1 values actually differ by more than `f1_margin`.
    Saves to `out_root/temporal/{helped,hurt}/`."""
    max_examples = max_examples or 12
    out_root = Path(out_root) / "temporal"
    helped_dir = out_root / "helped"
    hurt_dir = out_root / "hurt"
    helped_dir.mkdir(parents=True, exist_ok=True)
    hurt_dir.mkdir(parents=True, exist_ok=True)

    baseline_by_idx = {it["idx"]: it for it in baseline_cache}
    saved = {"helped": [], "hurt": []}

    for item_t in temporal_cache:
        if len(saved["helped"]) >= max_examples and len(saved["hurt"]) >= max_examples:
            break
        item_b = baseline_by_idx.get(item_t["idx"])
        if item_b is None:
            continue

        pred_b = postprocess_item(item_b, cfg, baseline_params)
        pred_t = postprocess_item(item_t, cfg, temporal_params)
        gt_lanes = item_t["gt_lanes"]
        _, _, f1_b, *_ = compute_pixel_f1(pred_b, gt_lanes, item_b["width"], item_b["height"])
        _, _, f1_t, *_ = compute_pixel_f1(pred_t, gt_lanes, item_t["width"], item_t["height"])
        delta = f1_t - f1_b
        if abs(delta) < f1_margin:
            continue

        image = cv2.imread(item_t["image_path"])
        if image is None:
            continue
        left = _draw_lanes(image.copy(), gt_lanes, _COLOR_GT, thickness=2)
        left = _draw_lanes(left, pred_b, (255, 128, 0), thickness=2)     # baseline: orange
        right = _draw_lanes(image.copy(), gt_lanes, _COLOR_GT, thickness=2)
        right = _draw_lanes(right, pred_t, _COLOR_PRED, thickness=2)     # temporal: red
        for panel, cap in ((left, f"Baseline  pixel_f1={f1_b:.3f}"),
                          (right, f"{temporal_label}  pixel_f1={f1_t:.3f}")):
            cv2.putText(panel, cap, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                       (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(panel, cap, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                       (0, 0, 0), 1, cv2.LINE_AA)
        combined = cv2.hconcat([left, right])

        bucket, target_dir = ("helped", helped_dir) if delta > 0 else ("hurt", hurt_dir)
        if len(saved[bucket]) >= max_examples:
            continue
        fname = f"{item_t['scene']}_{item_t['frame_id']}_delta{delta:+.3f}.png"
        out_path = target_dir / fname
        cv2.imwrite(str(out_path), combined)
        saved[bucket].append(out_path)

    return saved


# ============================================================
# 14c. Temporal consistency videos (task Section 7)
# ============================================================

def _find_contiguous_scene_run(cache, min_length=8):
    """Longest run of CONSECUTIVE frame_ids (delta==1) within a single
    scene, using the cache's own (scene, frame_id) fields. Mirrors the
    consecutive-pair logic already used by compute_temporal_jitter(), so
    the video and the jitter metric agree on what counts as "consecutive"
    and neither ever bridges a sequence-boundary gap or splices frames
    from two different scenes together (task Section 7: "do not
    accidentally join unrelated sequences"). Returns a chronologically
    sorted list of cache items, or [] if no run of at least `min_length`
    exists in this cache (train/val/test subsets are leakage-purged
    blocks that are NOT guaranteed to contain long contiguous runs -- see
    the compute_temporal_jitter docstring for the same caveat)."""
    by_scene = {}
    for item in cache:
        by_scene.setdefault(item["scene"], []).append(item)

    best_run = []
    for items in by_scene.values():
        items = sorted(items, key=lambda it: it["frame_id"])
        run = [items[0]] if items else []
        best_for_scene = list(run)
        for prev, cur in zip(items, items[1:]):
            if cur["frame_id"] - prev["frame_id"] == 1:
                run.append(cur)
            else:
                run = [cur]
            if len(run) > len(best_for_scene):
                best_for_scene = list(run)
        if len(best_for_scene) > len(best_run):
            best_run = best_for_scene

    return best_run if len(best_run) >= min_length else []


def save_temporal_consistency_video(baseline_cache, temporal_cache, cfg,
                                    baseline_params, temporal_params,
                                    temporal_label, out_path, scene_name=None,
                                    fps=None, max_frames=None):
    """Task Section 7: consecutive-frame video with GT + baseline +
    temporal overlays, confidence/gate info, frame index, chronological
    order preserved. Both caches must come from build_prediction_cache()
    over the SAME dataset (so `idx` values line up). Uses
    _find_contiguous_scene_run() to pick a real consecutive sequence
    rather than assuming the split's frame ordering is already temporally
    contiguous. Returns the output path, or None if no sufficiently long
    contiguous run was available (this is reported, not silently
    swallowed -- see the caller in run_pipeline)."""
    fps = fps or getattr(cfg, "temporal_video_fps", 8)
    max_frames = max_frames or getattr(cfg, "temporal_video_max_frames", 60)

    run = _find_contiguous_scene_run(
        [it for it in temporal_cache if scene_name is None or it["scene"] == scene_name],
        min_length=8,
    )
    if not run:
        print(
            f"[temporal-video] no contiguous run of >=8 consecutive frames found "
            f"for scene={scene_name!r} in this cache -- skipping "
            f"({out_path}). This is expected for small val/test subsets; "
            f"widen the subset or evaluate on a dedicated temporally-"
            f"contiguous split if a video is required for every scene."
        )
        return None
    run = run[:max_frames]

    baseline_by_idx = {it["idx"]: it for it in baseline_cache}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer_cv = None
    try:
        for i, item_t in enumerate(run):
            item_b = baseline_by_idx.get(item_t["idx"])
            image = cv2.imread(item_t["image_path"])
            if image is None:
                continue
            gt_lanes = item_t["gt_lanes"]
            pred_t = postprocess_item(item_t, cfg, temporal_params)
            pred_b = postprocess_item(item_b, cfg, baseline_params) if item_b is not None else []

            panel = image.copy()
            panel = _draw_lanes(panel, gt_lanes, _COLOR_GT, thickness=3)
            panel = _draw_lanes(panel, pred_b, (255, 128, 0), thickness=2)   # baseline: orange
            panel = _draw_lanes(panel, pred_t, _COLOR_PRED, thickness=2)     # temporal: red

            info = f"{item_t['scene']}  frame={item_t['frame_id']}  ({i+1}/{len(run)})"
            if item_t.get("gate_mean") is not None:
                info += f"  gate={item_t['gate_mean']:.2f}"
            for text, y, scale, color in (
                (info, 24, 0.55, (255, 255, 255)),
                ("green=GT  orange=baseline  red=" + temporal_label, panel.shape[0] - 12, 0.45,
                 (255, 255, 255)),
            ):
                cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                           (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                           color, 1, cv2.LINE_AA)

            if writer_cv is None:
                h, w = panel.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer_cv = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
                if not writer_cv.isOpened():
                    print(f"[temporal-video] failed to open VideoWriter for {out_path} "
                         f"-- mp4v codec may be unavailable in this OpenCV build.")
                    return None
            writer_cv.write(panel)
    finally:
        if writer_cv is not None:
            writer_cv.release()

    return out_path if out_path.exists() else None


# ============================================================
# 15. Comparison helpers
# ============================================================

DELTA_METRICS = [
    "iou_f1", "iou_precision", "iou_recall", "iou_detection_accuracy",
    "iou_f1_rainy", "iou_f1_occlusion", "iou_f1_shaky", "iou_f1_transition",
    "polyline_pixel_f1", "polyline_pixel_f2_safety", "lane_position_jitter_px",
]


def add_baseline_deltas(df, baseline_label=BASELINE_LABEL, metrics=DELTA_METRICS):
    df = df.copy()
    base = df.loc[df["model"] == baseline_label]
    if base.empty:
        return df
    base_row = base.iloc[0]
    for metric in metrics:
        if metric in df.columns:
            df[f"delta_{metric}_vs_baseline"] = df[metric] - base_row[metric]
    return df


def evaluate_all_models(caches, cfg_by_key, params_by_key, protocol_name):
    rows, confusions = [], {}
    for key, cache in caches.items():
        metrics, confusion = evaluate_iou_cache(
            cache, cfg_by_key[key], params_by_key[key], MODEL_LABELS[key])
        metrics["postprocessing_protocol"] = protocol_name
        rows.append(metrics); confusions[key] = confusion
    return rows, confusions


# ============================================================
# 15b. Comparison report
# ============================================================

METRIC_INDEX = [
    ("iou_f1", "higher", "Lane-level F1 (IoU-matched)."),
    ("polyline_pixel_f1", "higher", "Pixel-level F1 between drawn and GT masks."),
    ("polyline_pixel_f2_safety", "higher",
     "Recall-weighted (beta=2) pixel F-score: penalises missed lanes (false "
     "negatives) more than spurious ones. ADAS safety proxy, not a replacement "
     "for polyline_pixel_f1."),
    ("lane_position_jitter_px", "lower",
     "Mean |Δx| of predicted lane position between consecutive frames, in "
     "original-image pixels. The metric temporal fusion (v02/v03/v04) exists "
     "to win on; low n_pairs (see jitter_n_pairs) means low confidence."),
    ("iou_f1_rainy / _occlusion / _shaky / _transition", "higher", "Robustness per condition."),
    ("meets_realtime_budget", "n/a", f"Whether latency fits a {ADAS_TARGET_FPS:.0f} FPS "
     f"({ADAS_LATENCY_BUDGET_MS:.1f} ms) embedded budget."),
]


def print_metric_index(width=104):
    print("\n" + "=" * width)
    print("METRIC INDEX")
    print("=" * width)
    for name, direction, meaning in METRIC_INDEX:
        print(textwrap.fill(meaning, width=width,
                            initial_indent=f"  {name}  [{direction}]: ",
                            subsequent_indent="      "))
    print("=" * width)


COMPARISON_SECTIONS = {
    "Lane-level": [("iou_f1", "higher"), ("iou_precision", "higher"),
                   ("iou_recall", "higher"), ("iou_detection_accuracy", "higher")],
    "Pixel-level": [("polyline_pixel_f1", "higher"),
                    ("pixel_precision", "higher"), ("pixel_recall", "higher")],
    "ADAS safety / stability": [("polyline_pixel_f2_safety", "higher"),
                                ("lane_position_jitter_px", "lower"),
                                ("jitter_n_pairs", None)],
    "Conditions": [("iou_f1_rainy", "higher"), ("iou_f1_occlusion", "higher"),
                   ("iou_f1_shaky", "higher"), ("iou_f1_transition", "higher")],
    "Failure": [("empty_prediction_files", "lower")],
    "Efficiency": [("latency_milliseconds", "lower"), ("frames_per_second", "higher"),
                   ("parameters_total_millions", "lower"),
                   ("giga_macs", "lower"), ("peak_memory_megabytes", "lower"),
                   ("meets_realtime_budget", None)],
}


def _fmt_value(value, decimals=4):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    return f"{value:.{decimals}f}" if np.isfinite(value) else "n/a"


def _best_model(values, direction):
    if direction not in ("higher", "lower"):
        return None
    finite = {m: float(v) for m, v in values.items()
              if isinstance(v, (int, float, np.integer, np.floating))
              and not isinstance(v, (bool, np.bool_)) and np.isfinite(float(v))}
    if len(finite) < 2 or len(set(round(v, 10) for v in finite.values())) < 2:
        return None
    pick = max if direction == "higher" else min
    return pick(finite, key=finite.get)


def _print_side_by_side(title, rows, models):
    if not rows:
        return
    name_w = max(len(r[0]) for r in rows) + 1
    col_w = max(max(len(m) for m in models), 14) + 2
    total_w = name_w + (col_w + 1) * len(models)
    print(f"\n{title}")
    print("-" * total_w)
    print(f"{'metric':<{name_w}}" + "".join(f" {m:>{col_w}}" for m in models))
    print("-" * total_w)
    for name, direction, values, display in rows:
        best = _best_model(values, direction)
        cells = []
        for m in models:
            text = display[m] if display and m in display else _fmt_value(values.get(m))
            cells.append(f" {text + ('*' if m == best else ' '):>{col_w}}")
        print(f"{name:<{name_w}}" + "".join(cells))
    print("-" * total_w)


def print_model_comparison(results_df, title, baseline_label=None):
    baseline_label = baseline_label or BASELINE_LABEL
    df = results_df.set_index("model")
    models = [m for m in MODEL_ORDER if m in df.index]
    print("\n" + "#" * 100)
    print(title)
    print("#" * 100)
    for section, metrics in COMPARISON_SECTIONS.items():
        rows = [(name, direction, {m: df.loc[m, name] for m in models}, None)
                for name, direction in metrics if name in df.columns]
        _print_side_by_side(section, rows, models)
    others = [m for m in models if m != baseline_label]
    if baseline_label in df.index and others:
        delta_metrics = ["iou_f1", "polyline_pixel_f1", "iou_f1_occlusion",
                         "polyline_pixel_f2_safety", "lane_position_jitter_px"]
        rows = []
        for metric in delta_metrics:
            if metric not in df.columns:
                continue
            base = float(df.loc[baseline_label, metric])
            values = {m: float(df.loc[m, metric]) - base for m in others}
            display = {m: (f"{v:+.4f}" if np.isfinite(v) else "n/a") for m, v in values.items()}
            rows.append((metric, "higher", values, display))
        _print_side_by_side("Difference from Baseline", rows, others)


def print_training_comparison(histories):
    summary = summarize_training_histories(histories)
    if summary.empty:
        return
    s = summary.set_index("model")
    models = [m for m in MODEL_ORDER if m in s.index]
    spec = [
        ("epochs_trained", None),
        ("best_epoch", None),
        ("polyline_pixel_f1_at_best_epoch", "higher"),
        ("anchor_f1_at_best_epoch", "higher"),
        ("train_x_loss_at_best_epoch", "lower"),
    ]
    rows = [(c, d, {m: s.loc[m, c] for m in models}, None) for c, d in spec if c in s.columns]
    _print_side_by_side("TRAINING SUMMARY", rows, models)


def print_multi_seed_comparison(agg_df, deltas_df, metrics=None):
    if agg_df.empty:
        return
    metrics = metrics or ["iou_f1", "polyline_pixel_f1",
                          "iou_f1_occlusion"]
    models = [m for m in MODEL_ORDER if m in set(agg_df["model"])]
    n_seeds = int(agg_df["n_seeds"].max())
    rows = []
    for metric in metrics:
        sub = agg_df[agg_df["metric"] == metric].set_index("model")
        if sub.empty:
            continue
        values = {m: float(sub.loc[m, "mean"]) for m in models if m in sub.index}
        display = {}
        for m in values:
            std = sub.loc[m, "std"]
            display[m] = (f"{values[m]:.3f} ± {std:.3f}" if np.isfinite(std)
                          else f"{values[m]:.3f}")
        rows.append((metric, "higher", values, display))
    _print_side_by_side(f"MULTI-SEED RESULTS (n={n_seeds})", rows, models)


# NOTE (this revision): the previous version of this file had a duplicate,
# unused `export_arch_images()` helper here that re-implemented
# export_model_architecture() (section 5d) with three bugs: it shadowed
# the module-level `Path`/`SummaryWriter` imports with local ones, it
# opened a brand-new SummaryWriter/log_dir instead of accepting the run's
# existing writer via get_tb_writer(), and it had no marker-file guard so
# calling it once per seed would regenerate identical diagrams four times.
# It was never called anywhere in run_pipeline (the call site there was
# broken -- see the "TAC-UFLD v1.1" header notes), so it has been removed
# rather than fixed; run_pipeline now calls
# _export_architecture_if_missing() -> export_model_architecture(), the
# single, correct implementation.

# ============================================================
# 16. Pipeline driver
# ============================================================

def run_pipeline(seed, result_root):
    seed_holder["seed"] = seed
    result_root = Path(result_root)
    export_root = result_root / "exports"
    plots_root = result_root / "plots"
    result_root.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    # (this revision) Warn early about a nearly-full drive instead of
    # discovering it hours later at the export step -- see
    # _check_disk_space's and save_lanes_txt's docstrings.
    _check_disk_space(result_root, min_free_mb=2000, label=f"seed {seed} results")

    cfg.seed = seed
    set_seed(seed)

    split_rows, scene_records, missing_scenes = prepare_scene_splits(
        DATASET_ROOT, SCENE_PLAN,
        {"train": N_TRAIN_SUBSET, "val": N_VAL_SUBSET, "test": N_TEST_SUBSET},
        seed=seed
    )
    if missing_scenes:
        print("Scenes missing:", ", ".join(missing_scenes))
    print("Split sizes:", {k: len(v) for k, v in split_rows.items()})

    train_dataset = ELASTemporalDataset(split_rows["train"], scene_records, cfg, augment=True)
    val_dataset = ELASTemporalDataset(split_rows["val"], scene_records, cfg, augment=False)
    test_dataset = ELASTemporalDataset(split_rows["test"], scene_records, cfg, augment=False)

    sanity_check_plot(val_dataset, plots_root / "sample_targets.png")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    baseline_ckpt_path = result_root / "baseline_final.pth"

    # ---------------- TensorBoard writers for this seed (task Sec. 3/4/6) ----
    # One writer per model, unique log_dir under TENSORBOARD_ROOT/seed_<n>/<key>
    # -- see get_tb_writer()'s docstring on why writers are never shared
    # across models/seeds/trials. Optuna trial writers (task Section 5) are
    # created separately inside train_and_eval_variant_trial, namespaced
    # under this same seed's tensorboard root, passed below.
    #
    # BUGFIX (this revision): these four per-seed writers were never
    # explicitly closed (only the Optuna trial writers were, via
    # close_tb_writer). Across SEEDS=[42,1,7,2026] that is 16 SummaryWriter
    # instances -- each holding an open event-file handle and the traced
    # graph from log_model_graph_to_tensorboard -- left alive in
    # _active_tb_writers for the rest of the process. This was a plausible
    # contributor to the slow-growing host-RAM pressure behind the
    # DataLoader worker MemoryError reported earlier (a 2.25 MiB array
    # failing to allocate is not itself a large allocation -- it means the
    # process was already starved by something else). They are now closed
    # at the end of this function (see close_tb_writer calls below).
    tb_seed_root = TENSORBOARD_ROOT / f"seed_{seed}"
    tb_enabled = getattr(cfg, "tensorboard_enabled", True)
    tb_writers = {}
    if tb_enabled:
        for key in ("baseline", "v02", "v03", "v04"):
            tb_writers[key] = get_tb_writer(tb_seed_root / key)
    else:
        tb_writers = {k: None for k in ("baseline", "v02", "v03", "v04")}

    # ---------------- baseline ----------------
    baseline_model = SingleFrameUFLDLikeModel(cfg).to(cfg.device)

    # Torchview architecture diagram (task Section 2): generated once from
    # the freshly constructed model, independent of training/seed weights
    # -- the module structure this diagrams is identical before and after
    # training, and generating it here means it exists even if training
    # later fails partway through. Saved to the experiment-independent
    # ARCHITECTURE_ROOT, not under this seed's result_root, and skipped on
    # subsequent seeds via the on-disk marker file (module structure never
    # depends on the seed).
    _export_architecture_if_missing(
        baseline_model, cfg, temporal=False, label="baseline_ufld",
        out_dir=ARCHITECTURE_ROOT,
    )

    def _train_baseline():
        history = train_model(
            baseline_model, train_loader, val_loader, cfg,
            epochs=cfg.epochs_baseline, mode="baseline", temporal=False,
            lr_backbone=cfg.lr, lr_head=cfg.lr, lr_fusion=cfg.lr,
            lambda_temporal=0.0, save_path=baseline_ckpt_path,
            val_dataset=val_dataset,
            writer=tb_writers["baseline"], tb_prefix="baseline",
        )
        history.to_csv(result_root / "history_baseline.csv", index=False)
        return history

    history_baseline = _train_baseline()

    # ---------------- Optuna ----------------
    best_params = {"v02": {}, "v03": {}, "v04": {}}
    if RUN_OPTUNA_HPO:
        for variant in ("v02", "v03", "v04"):
            study = run_variant_hpo(
                variant, train_loader, val_loader, val_dataset, cfg,
                baseline_ckpt_path, result_root=result_root, plots_root=plots_root,
                tensorboard_root=tb_seed_root,
            )
            best_params[variant] = dict(study.best_trial.params)
            best_params[variant]["best_validation_pixel_f1"] = study.best_trial.value
            study.trials_dataframe().to_csv(
                result_root / f"optuna_trials_{variant}.csv", index=False)
            print(f"\nBest {variant} trial value: {study.best_trial.value:.4f}")
            print(f"  params: {study.best_trial.params}")
        pd.DataFrame([{"variant": v, **p} for v, p in best_params.items()]).to_csv(
            result_root / "optuna_best_params.csv", index=False)

    # ---------------- v0.2 ----------------
    cfg_v02 = copy.deepcopy(cfg)
    p_v02 = best_params["v02"]
    cfg_v02.exist_threshold = p_v02.get("exist_threshold", 0.35)
    cfg_v02.lr = p_v02.get("lr", 5e-5)
    cfg_v02.weight_decay = p_v02.get("weight_decay", 5e-4)
    cfg_v02.lambda_coord = p_v02.get("lambda_coord", 20.0)

    temporal_model_v02 = TACUFLDTemporalModel(cfg_v02).to(cfg_v02.device)
    # BUGFIX (this revision): the baseline's backbone (full-width
    # LightweightBackbone) and the temporal models' current-frame backbone
    # (half-width LightweightBackboneTemporal) are different architectures
    # by design now (see LightweightBackboneTemporal's docstring), so
    # `temporal_model_v02.backbone.load_state_dict(baseline_model.backbone.state_dict())`
    # would crash immediately on the first shape mismatch -- removed.
    # The head is still warm-startable: load_partial_state_dict copies
    # every head.* tensor whose shape matches (everything except
    # head.pre.0.weight, whose input-channel count now differs) instead of
    # the previous unconditional `.load_state_dict(...)`, which required
    # an exact match on every key.
    load_partial_state_dict(temporal_model_v02.head, baseline_model.head.state_dict(),
                            label="v02 head")

    history_v02 = train_model(
        temporal_model_v02, train_loader, val_loader, cfg_v02,
        epochs=cfg.epochs_v02, mode="v02", temporal=True,
        lr_backbone=cfg_v02.lr * 0.1, lr_head=cfg_v02.lr,
        lr_fusion=p_v02.get("lr_fusion", 5e-4),
        lambda_temporal=p_v02.get("lambda_temporal", 0.5),
        val_dataset=val_dataset,
        save_path=result_root / "tac_ufld_v02_final.pth",
        writer=tb_writers["v02"], tb_prefix="v02",
    )
    history_v02.to_csv(result_root / "history_v02.csv", index=False)

    # ---------------- v0.3 (gated alias) ----------------
    cfg_v03 = copy.deepcopy(cfg)
    p_v03 = best_params["v03"]
    cfg_v03.exist_threshold = p_v03.get("exist_threshold", 0.35)
    cfg_v03.lr = p_v03.get("lr", 5e-5)
    cfg_v03.weight_decay = p_v03.get("weight_decay", 5e-4)

    temporal_model_v03 = TACUFLDGatedTemporalModel(cfg_v03).to(cfg_v03.device)
    load_partial_state_dict(temporal_model_v03.head, baseline_model.head.state_dict(),
                            label="v03 head")

    history_v03 = train_model(
        temporal_model_v03, train_loader, val_loader, cfg_v03,
        epochs=cfg.epochs_v03, mode="v03", temporal=True,
        lr_backbone=cfg_v03.lr * 0.1, lr_head=cfg_v03.lr,
        lr_fusion=p_v03.get("lr_fusion", 5e-4),
        lambda_temporal=p_v03.get("lambda_temporal", 0.5),
        val_dataset=val_dataset,
        save_path=result_root / "tac_ufld_v03_final.pth",
        writer=tb_writers["v03"], tb_prefix="v03",
    )
    history_v03.to_csv(result_root / "history_v03.csv", index=False)

    # ---------------- v0.4 (warm-start from v0.2) ----------------
    cfg_v04 = copy.deepcopy(cfg_v02)
    p_v04 = best_params["v04"]
    cfg_v04.exist_threshold = p_v04.get("exist_threshold", 0.35)
    cfg_v04.lr = p_v04.get("lr", 5e-5)
    cfg_v04.weight_decay = p_v04.get("weight_decay", 5e-4)
    cfg_v04.lambda_coord = p_v04.get("lambda_coord", 20.0)

    temporal_model_v04 = TACUFLDTemporalModel(cfg_v04).to(cfg_v04.device)
    # v02 and v04 are the SAME architecture (both TACUFLDTemporalModel), so
    # this full-state-dict warm start remains exact -- unaffected by the
    # backbone/head architecture split above.
    temporal_model_v04.load_state_dict(temporal_model_v02.state_dict())

    history_v04 = train_model(
        temporal_model_v04, train_loader, val_loader, cfg_v04,
        epochs=cfg.epochs_v04, mode="v04", temporal=True,
        lr_backbone=cfg_v04.lr * 0.1, lr_head=cfg_v04.lr,
        lr_fusion=p_v04.get("lr_fusion", 2e-4),
        lambda_temporal=p_v04.get("lambda_temporal", 0.5),
        val_dataset=val_dataset,
        save_path=result_root / "tac_ufld_v04_final.pth",
        writer=tb_writers["v04"], tb_prefix="v04",
    )
    history_v04.to_csv(result_root / "history_v04.csv", index=False)

    histories = {
        "baseline": history_baseline, "v02": history_v02,
        "v03": history_v03, "v04": history_v04,
    }
    for key, history in histories.items():
        plot_model_history(history, MODEL_LABELS[key],
                           plots_root / f"training_curves_{key}.png")
    plot_all_models_convergence(histories, plots_root / "convergence_all_models.png")

    training_summary_df = summarize_training_histories(histories)
    training_summary_df.to_csv(result_root / "training_summary_at_best_epoch.csv", index=False)
    print_training_comparison(histories)

    # ---------------- post-processing sweeps ----------------
    model_specs = {
        "baseline": (baseline_model, cfg, False),
        "v02": (temporal_model_v02, cfg_v02, True),
        "v03": (temporal_model_v03, cfg_v03, True),
        "v04": (temporal_model_v04, cfg_v04, True),
    }
    cfg_by_key = {k: s[1] for k, s in model_specs.items()}

    tuned_params = {}
    for key, (model, cfg_use, temporal) in model_specs.items():
        print(f"Sweeping post-processing for {MODEL_LABELS[key]}...")
        best_pp, sweep_df = sweep_postprocessing(
            model, val_dataset, cfg_use, temporal,
            param_grid=POSTPROCESS_SWEEP_GRID, base_params=POSTPROCESS_BASE_PARAMS,
        )
        sweep_df.to_csv(result_root / f"{key}_postprocess_sweep_val.csv", index=False)
        plot_postprocessing_sweep_heatmap(
            sweep_df, MODEL_LABELS[key], plots_root / f"postprocess_sweep_{key}.png")
        tuned_params[key] = {n: best_pp[n] for n in
                             ("threshold", "min_points", "min_lane_length_fraction",
                              "duplicate_distance", "y_step", "poly_degree")}
        print(f"  best: {tuned_params[key]}")

    common_params = {k: dict(POSTPROCESS_COMMON_FIXED_PARAMS) for k in model_specs}

    # ---------------- test evaluation ----------------
    print("Building test prediction caches...")
    caches = {k: build_prediction_cache(m, test_dataset, cu, temporal=t)
              for k, (m, cu, t) in model_specs.items()}

    tuned_rows, confusion_by_model = evaluate_all_models(
        caches, cfg_by_key, tuned_params, "validation_tuned")
    common_rows, _ = evaluate_all_models(
        caches, cfg_by_key, common_params, "common_fixed")

    plot_confusion_grid(confusion_by_model, plots_root / "confusion_matrices_all_models.png")

    efficiency_df = pd.DataFrame([
        compute_model_efficiency(m, cu, temporal=t, label=MODEL_LABELS[k])
        for k, (m, cu, t) in model_specs.items()
    ])
    efficiency_df.to_csv(result_root / "model_efficiency.csv", index=False)

    # ADAS temporal-stability metric, computed once per model straight from
    # the test caches already built above (same predictions the accuracy
    # metrics use, so no extra inference passes).
    jitter_df = pd.DataFrame([
        {"model": MODEL_LABELS[k], **compute_temporal_jitter(caches[k], cfg_by_key[k])}
        for k in model_specs
    ])
    jitter_df.to_csv(result_root / "temporal_jitter.csv", index=False)

    def _finalize(rows):
        df = pd.DataFrame(rows).merge(efficiency_df, on="model", how="left")
        df = df.merge(jitter_df, on="model", how="left")
        df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
        df = df.sort_values("model").reset_index(drop=True)
        df["model"] = df["model"].astype(str)
        return add_baseline_deltas(df)

    iou_df = _finalize(tuned_rows)
    iou_common_df = _finalize(common_rows)
    iou_df.to_csv(result_root / "final_iou_compatible_results.csv", index=False)
    iou_common_df.to_csv(result_root / "final_iou_results_common_fixed_postprocessing.csv", index=False)

    print_model_comparison(iou_df, f"SEED {seed} | TEST (validation-tuned)")
    print_model_comparison(iou_common_df, f"SEED {seed} | TEST (common-fixed)")

    main_metrics = ["iou_f1", "iou_precision", "iou_recall", "iou_detection_accuracy"]
    plot_results_matrix(iou_df, main_metrics, plots_root / "final_results_matrix.png")
    plot_metric_comparison_bars(iou_df, main_metrics,
                                plots_root / "metric_comparison_validation_tuned.png",
                                "Test metrics (validation-tuned)")
    plot_delta_vs_baseline(iou_df, main_metrics, plots_root / "difference_from_baseline.png")
    plot_condition_heatmap(iou_df, plots_root / "iou_f1_per_condition.png")
    plot_efficiency_tradeoff(iou_df, plots_root / "efficiency_tradeoff.png")
    plot_adas_readiness(iou_df, plots_root / "adas_readiness.png")
    print_adas_readiness_report(iou_df, result_root / "adas_readiness_report.csv")

    # ---------------- qualitative comparison images ----------------
    comparison_root = plots_root / "scenario_comparisons"
    before_after_root = plots_root / "before_after"
    for key, (model, cfg_use, temporal) in model_specs.items():
        save_scenario_comparison_images(
            caches[key], cfg_use, tuned_params[key], MODEL_LABELS[key],
            comparison_root,
        )
        # (this revision) explicit "street image with vs. without lane
        # detection" per model, requested alongside the richer GT-vs-
        # prediction panels above.
        save_before_after_images(
            caches[key], cfg_use, tuned_params[key], MODEL_LABELS[key],
            before_after_root, n_samples=3,
        )
    print(f"Scenario comparison images written to: {comparison_root}")
    print(f"Before/after images written to: {before_after_root}")

    # ---------------- ablation ----------------
    ablation_modes = {
        "baseline": [("full", None)],
        "v02": [("full", None), ("last_frame_only", "last_frame"),
                ("uniform_fusion", "uniform_fusion"), ("no_warp", "no_warp")],
        "v03": [("full", None), ("last_frame_only", "last_frame"),
                ("uniform_fusion", "uniform_fusion"), ("no_warp", "no_warp")],
        "v04": [("full", None), ("last_frame_only", "last_frame"),
                ("uniform_fusion", "uniform_fusion"), ("no_warp", "no_warp")],
    }
    ablation_rows = []
    for key, (model, cfg_use, temporal) in model_specs.items():
        full_metrics = None
        for mode_name, ablation_mode in ablation_modes[key]:
            cache = caches[key] if ablation_mode is None else build_prediction_cache(
                model, test_dataset, cfg_use, temporal=temporal,
                ablation_mode=ablation_mode)
            metrics, _ = evaluate_iou_cache(cache, cfg_use, tuned_params[key],
                                            MODEL_LABELS[key], verbose=False)
            if mode_name == "full":
                full_metrics = metrics
            ablation_rows.append({
                "model": MODEL_LABELS[key], "ablation": mode_name,
                "iou_f1": metrics["iou_f1"],
                "iou_precision": metrics["iou_precision"],
                "iou_recall": metrics["iou_recall"],
                "iou_detection_accuracy": metrics["iou_detection_accuracy"],
                "delta_iou_f1_vs_full": (np.nan if mode_name == "full"
                                         else metrics["iou_f1"] - full_metrics["iou_f1"]),
            })
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(result_root / "ablation_results.csv", index=False)
    print("Ablation:")
    print(ablation_df.to_string(index=False))
    plot_ablation(ablation_df, plots_root / "ablation.png")

    # ---------------- exports ----------------
    export_dirs = {"baseline": "baseline", "v02": "tac_ufld_v02",
                   "v03": "tac_ufld_v03", "v04": "tac_ufld_v04"}
    for key, folder in export_dirs.items():
        manifest = export_cache_predictions(caches[key], cfg_by_key[key],
                                            tuned_params[key], export_root / folder)
        manifest.to_csv(result_root / f"export_manifest_{key}.csv", index=False)

    for split_name, rows in split_rows.items():
        with open(result_root / f"{split_name}_subset.txt", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(f"{r['scene']}\t{r['frame_id']}\t{r['image_path']}\n")

    # (this revision) close this seed's four per-model TensorBoard writers
    # -- see the docstring at their creation above.
    for writer in tb_writers.values():
        if writer is not None:
            close_tb_writer(writer)

    return iou_df


# ============================================================
# 18. Multi-seed statistical analysis
# ============================================================

def aggregate_multi_seed_metrics(combined_df, model_order=None, metrics=None,
                                 baseline_label=None):
    from scipy import stats as _stats
    if model_order is None:
        model_order = MODEL_ORDER
    if baseline_label is None:
        baseline_label = BASELINE_LABEL
    if metrics is None:
        metrics = ["iou_f1", "iou_precision", "iou_recall", "iou_detection_accuracy",
                   "polyline_pixel_f1", "polyline_pixel_f2_safety",
                   "lane_position_jitter_px",
                   "iou_f1_rainy", "iou_f1_occlusion",
                   "iou_f1_shaky", "iou_f1_transition"]
    rows = []
    for model in model_order:
        sub = combined_df[combined_df["model"] == model]
        if sub.empty:
            continue
        n = len(sub)
        for metric in metrics:
            if metric not in sub.columns:
                continue
            vals = sub[metric].dropna().values.astype(float)
            if len(vals) == 0:
                continue
            mean = float(np.mean(vals))
            if len(vals) > 1:
                std = float(np.std(vals, ddof=1))
                se = std / np.sqrt(len(vals))
                tcrit = _stats.t.ppf(0.975, df=len(vals) - 1)
                ci_half = tcrit * se
            else:
                std = float("nan"); ci_half = float("nan")
            rows.append({
                "model": model, "metric": metric, "mean": mean, "std": std,
                "ci95_low": mean - ci_half if np.isfinite(ci_half) else float("nan"),
                "ci95_high": mean + ci_half if np.isfinite(ci_half) else float("nan"),
                "n_seeds": n,
            })
    return pd.DataFrame(rows)


def paired_wilcoxon_vs_baseline(combined_df, metrics=None, baseline_label=None,
                                model_order=None):
    from scipy import stats as _stats
    if baseline_label is None:
        baseline_label = BASELINE_LABEL
    if model_order is None:
        model_order = [m for m in MODEL_ORDER if m != baseline_label]
    if metrics is None:
        metrics = ["iou_f1", "polyline_pixel_f1", "iou_f1_occlusion",
                   "polyline_pixel_f2_safety", "lane_position_jitter_px"]
    base = combined_df[combined_df["model"] == baseline_label]
    base_by_seed = {r["seed"]: r for _, r in base.iterrows()}
    rows = []
    for model in model_order:
        sub = combined_df[combined_df["model"] == model]
        if sub.empty:
            continue
        for metric in metrics:
            if metric not in sub.columns:
                continue
            deltas = []
            for _, row in sub.iterrows():
                if row["seed"] not in base_by_seed:
                    continue
                b_val = base_by_seed[row["seed"]][metric]
                m_val = row[metric]
                if pd.isna(b_val) or pd.isna(m_val):
                    continue
                deltas.append(float(m_val) - float(b_val))
            if len(deltas) < 2:
                continue
            deltas = np.asarray(deltas)
            mean_delta = float(np.mean(deltas))
            std_delta = float(np.std(deltas, ddof=1))
            se = std_delta / np.sqrt(len(deltas))
            tcrit = _stats.t.ppf(0.975, df=len(deltas) - 1)
            ci_half = tcrit * se
            try:
                stat, p = _stats.wilcoxon(deltas, alternative="two-sided")
            except ValueError:
                stat, p = float("nan"), 1.0
            rows.append({
                "model": model, "metric": metric, "mean_delta": mean_delta,
                "std_delta": std_delta,
                "ci95_delta_low": mean_delta - ci_half,
                "ci95_delta_high": mean_delta + ci_half,
                "wilcoxon_stat": float(stat), "p_value": float(p),
                "significant_at_0.05": bool(p < 0.05) if np.isfinite(p) else False,
                "n_seeds": len(deltas),
            })
    return pd.DataFrame(rows)


# ============================================================
# 17. Main
# ============================================================

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true",
                        help="Run coordinate sanity check only, then exit.")
    args = parser.parse_args()

    print("Device:", cfg.device)
    print("Result root:", RESULT_ROOT)

    if SMOKE_EPOCHS < 50 or len(SEEDS) < 6:
        print(
            f"[CONFIG] Running in smoke-test mode: SMOKE_EPOCHS={SMOKE_EPOCHS} "
            f"(final comparison target: >=50), SEEDS={SEEDS} (final comparison "
            f"target: >=6). Per-seed results and single-seed 'best' markers below "
            f"are directional only -- do not treat them as the final ADAS "
            f"go/no-go comparison until both are scaled up and "
            f"paired_wilcoxon_vs_baseline() has more than one seed to test."
        )

    if args.sanity:
        run_sanity_check()
        sys.exit(0)

    all_seed_results = []
    for seed in SEEDS:
        print(f"\n{'=' * 60}\nRunning pipeline for seed={seed}\n{'=' * 60}")
        seed_iou_df = run_pipeline(seed, RESULT_ROOT / f"seed_{seed}")
        seed_iou_df = seed_iou_df.copy()
        seed_iou_df["seed"] = seed
        all_seed_results.append(seed_iou_df)

    combined_df = pd.concat(all_seed_results, ignore_index=True)
    combined_df.to_csv(RESULT_ROOT / "all_seeds_raw_results.csv", index=False)

    agg_df = aggregate_multi_seed_metrics(combined_df)
    agg_df.to_csv(RESULT_ROOT / "multi_seed_aggregate.csv", index=False)

    deltas_df = paired_wilcoxon_vs_baseline(combined_df)
    deltas_df.to_csv(RESULT_ROOT / "multi_seed_paired_tests.csv", index=False)

    print(f"\n{'=' * 60}\nFINAL COMPARISON seeds={SEEDS}\n{'=' * 60}")
    print_multi_seed_comparison(agg_df, deltas_df)
    print_metric_index()

    close_all_tb_writers()