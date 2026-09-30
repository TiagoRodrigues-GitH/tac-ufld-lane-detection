# TAC-UFLD v0.4 — hand-off for the full experiment

**For:** the supervisor running the full experiment. **From:** Tiago Rodrigues. **Date:** 2026-09-30. **Package:** `tac-ufld-handoff-v0.4.0-<date>.zip` (git branch `handoff/v0.4`; v0.3 is on `handoff/v0.3`).

**What changed in v0.4, in one paragraph.** The first pilot showed UFLD overfitting after one epoch (held-out lane F1 0.05) and temporal models that barely used their history. An augmentation ablation, decided on validation, found the cause: without geometric augmentation UFLD memorises lane positions; with it the UFLD baseline reaches 0.86 to 0.88 held-out lane F1. `configs/elas.yaml` now uses that recipe (plus current-frame degradation) for every model. v0.4 also adds two temporal models (aligned fusion, ConvGRU), controls for extra training (`*_baseline_ct`) and for extra layers (`lite_v05_static`), a Kalman-tracker reference, per-condition results and a history-length ablation. Findings so far: `docs/PILOT_V2_FINDINGS.md`; details: `docs/TEMPORAL_IMPROVEMENTS.md`; plan: `docs/ROADMAP.md`.

## 1. What to run

```powershell
# setup (Windows; Linux is the same with python3 / source .venv/bin/activate)
py -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu126     # match your CUDA driver
pip install -e ".[all]"
$env:ELAS_ROOT = "D:\datasets\dataset_elas_v1"                          # or keep the default relative path
python -m tac_ufld doctor                                               # environment, GPU, datasets
python -m pytest -q                                                     # ~10 min; see docs/TESTING.md

# checks before the long run
python -m tac_ufld sanity --config configs/elas.yaml --real             # shapes, losses, gradients, warm starts
python -m tac_ufld check-data --config configs/elas.yaml                # splits, leakage, label overlays

# the full experiment (12 models and controls x 6 seeds, HPO with equal budgets, <= 50 epochs)
python -m tac_ufld run --config configs/elas.yaml --confirm
python -m tac_ufld run --config configs/elas.yaml --confirm --resume    # after any interruption
# or the core subset (9 models; about 25 % shorter)
python -m tac_ufld run --config configs/elas.yaml --confirm --variants ufld_baseline ufld_baseline_ct ufld_v03 ufld_v07 lite_baseline lite_baseline_ct lite_v05 lite_v05_static lite_v06
tensorboard --logdir results/elas_full/tensorboard

# after the run
python -m tac_ufld site --runs results/elas_full --title "TAC-UFLD full results"
python -m tac_ufld ui

# another dataset (CULane shown; tusimple / openlane are the same). Download first: docs/DATASET_DOWNLOAD_GUIDE.md
python scripts/check_dataset.py culane D:\datasets\CULane                # plain Python: READY / NOT READY
$env:CULANE_ROOT = "D:\datasets\CULane"                                 # then enabled: true in configs/datasets.yaml
python -m tac_ufld validate-dataset --dataset culane                    # counts, lane order, history, overlays
python scripts/smoke_datasets.py --datasets culane --device cuda        # a few batches through every stage
python -m tac_ufld run --dataset culane --confirm                       # the full run (--resume after interruptions)
```

Switching datasets is only `--dataset`; the "HOW TO SWITCH DATASETS" block at the top of `src/tac_ufld/cli.py` (also `python -m tac_ufld --help`) explains each step. The CULane and TuSimple configs use the official UFLD training augmentation (rotation ±6°, ±100 px vertical and ±200 px horizontal shifts) plus current-frame degradation; OpenLane uses the ELAS recipe (no official UFLD setting).

`--confirm` is required for the full configs (a run without it stops before creating anything). `--resume` continues an interrupted variant from its last epoch and reuses finished HPO studies and seeds. The run writes `results/elas_full/report/REPORT.md` (with the protocol actually used and every setting that differs from the defaults), `results.xlsx`, figures, per-frame CSVs, overlays and a test video. With 12 models, geometric augmentation (the lite models need most of the 50 epochs) and HPO, expect about 4 to 5 days on an RTX 3050 and roughly a day on a GPU four to five times faster (`docs/EXPERIMENT_MATRIX.md`). The core subset above saves about a quarter.

## 2. Protocol (unchanged from v0.2; recipe changed in v0.4)

The v0.4 training recipe for every model: photometric augmentation as before, plus geometric augmentation (shift, zoom, rotation, perspective, crop; one homography per clip, labels re-encoded exactly) and current-frame degradation (30 % of clips). Delete the `geometric` block and `current_frame_prob` in `configs/elas.yaml` to reproduce the v0.3 recipe. New pairs in the statistics: every temporal model is also compared with `*_baseline_ct` (same extra training), `lite_v05` with `lite_v05_static` (same layers, no time), and every model with its own output filtered by the Kalman tracker.

Held-out test scenes `BR_S02`, `VIX_S05`, `VV_S03` (never seen by training, validation or HPO); train/val/seen-scene test from 60-frame blocks of the other seven scenes with a ≥ 10-frame purge gap; split independent of the training seed and leakage-checked before training; 6 seeds; fresh seeded Optuna studies with the same budget for every model including the baselines; checkpoint selection, early stopping, HPO and post-processing tuning on validation `lane_f1_iou50` only; test evaluated once; paired Wilcoxon + Holm against the same-family baseline (`ufld_v02/v03/v04` vs `ufld_baseline`, `lite_v05` vs `lite_baseline`). Primary metric `lane_f1_iou50`; secondary `lane_f1_iou35`; also precision, recall, F2, TP/FP/FN, pixel F1, anchor F1, jitter, efficiency.

## 3. Status

### Already implemented and tested (v0.2, re-verified)
ELAS geometry (real-data regression test), missing/unknown slots, ROI ignore rows, anchor metric (wrong position = FP + FN), CULane-style lane F1 with Hungarian matching, official UFLD ResNet-18 with strict ImageNet loading, v0.2/v0.3/v0.4, lite baseline/v0.5, warm starts, scene hold-out + purge + leakage checks, seeded HPO, 6-seed statistics with Holm, tuned and common post-processing, YAML validation, end-to-end synthetic run. The 50 original tests still pass unchanged.

### Already implemented but not validated
* `--resume` had never been exercised and was wrong (it treated partially trained variants as finished). **Fixed and now tested** (epoch-level resume, `docs/PREPROCESSING_AUGMENTATION.md`).

### Newly implemented and tested
* Dataset registry (`configs/datasets.yaml`): enable/disable per dataset, `${VAR}` roots, explicit errors, disabled datasets never scanned; `datasets`, `validate-dataset` commands.
* CULane adapter fixes (temporal step 90, official segmentation-label slots, existence-flag cross-check), new TuSimple and OpenLane adapters, validation carving, dataset-native metrics (TuSimple `LaneEval` port) — all on synthetic copies of the real layouts.
* Preprocessing ablations (gray, gray3, edge, Canny, Hough, RGB+edge) with adapted first convolution; RGB default bit-identical.
* Geometric augmentation with exact label re-encoding (alignment test > 95 % of targets on the drawn lane after random transforms); extended photometric augmentation; the original augmentation bit-identical.
* Label smoothing, optional UFLD head dropout, configurable lite dropout, epoch-level resume, run confirmation guard, `sanity` command.
* Streaming inference with per-stream history-feature caching (exact against the clip forward pass for every temporal model, isolated streams, resets, dropped frames), video input read incrementally.
* ONNX export with the streaming split (encoder/head/clip), ONNX Runtime FP32 (matches PyTorch to 1e-5), INT8 with validation-frame calibration, TensorRT FP32/FP16/INT8 engines on the RTX 3050, numerical and task-level checks, `deployment.json` model cards.
* Hardware budget benchmark (measured) + real-time simulation (assumed slowdowns, clearly separated).
* Streamlit UI (never trains; tested to start headless and to import no training code).
* Docker / Compose: CPU image built and the test suite run inside it; GPU image built and run with `--gpus all` on the RTX 3050 (sanity checks on real ELAS data), `docs/DOCKER.md`.
* Static results page for GitHub Pages; hand-off packager; `doctor`.
* **GPU pilot** of the full protocol (2 seeds, ≤ 4 epochs, no HPO): completed in 2 h 22 min. Findings in `docs/PILOT_FINDINGS.md`.

### v0.4: newly implemented and tested (`docs/TEMPORAL_IMPROVEMENTS.md`)
* Overfitting remedies for UFLD: backbone learning-rate multiplier (also an HPO parameter), frozen early stages with frozen BatchNorm statistics, 8 more ELAS training scenes as an ablation arm, cross-dataset initialisation from official UFLD / own checkpoints (`model.init_checkpoint`; tested on a synthetic official-format checkpoint only).
* Current-frame degradation (history kept clean), applied to every model.
* New models: `ufld_v06` (aligned fusion), `ufld_v07` and `lite_v06` (ConvGRU; start exactly as their baseline; carried-state streaming mode), equal-training controls `ufld_baseline_ct` / `lite_baseline_ct`, all paired in the report.
* Evaluation: validation-tuned output Kalman tracker for every model, per-condition F1 table, carried-state results; ablation specs with grids, explicit `allow` lists, shared `common` settings and reuse of identical baseline checkpoints (`configs/ablations/history.yaml`).
* Results page: model explanations and differences, overfitting and history ablations, roadmap (`docs/ROADMAP.md`).
* Robustness evaluation (`python -m tac_ufld robustness --runs <run>`): every checkpoint re-evaluated on the held-out scenes with the current frame occluded / blurred / darkened / noisy and the history clean (fixed per-frame corruptions, identical for every model).
* Multi-dataset preparation: `scripts/check_dataset.py` (download check, plain Python), `scripts/smoke_datasets.py` with `configs/{culane,tusimple,openlane}_smoke.yaml` (every stage, a few batches, synthetic copy when the data is absent), `docs/DATASET_DOWNLOAD_GUIDE.md` for whoever downloads the data, internal and native metrics on separate log lines.

### Newly implemented but not validated
* CULane, TuSimple and OpenLane on **real data** (no copy was available): run `python -m tac_ufld validate-dataset --dataset <name>` first. OpenLane's native metric settings follow our reading of the official 2D evaluation, not cross-checked.
* ONNX Runtime on GPU (the PyPI GPU build needs CUDA 13; ran on CPU here).
* A full training run inside Docker (the GPU container itself was verified).
* CARLA frame source.

### Requires a GPU
The full experiment, the ablations, `export --tensorrt`, latency benchmarks meant for comparison.

### Requires TensorRT or a Jetson
Engine building on the target (`scripts/jetson/build_engines.sh`), Jetson latency/memory (`scripts/jetson/measure_on_jetson.sh`). The Jetson profiles in `configs/deploy/` use **assumed** slowdowns (Orin Nano ×7, AGX Orin ×1.7); only a device measurement can replace them.

## 4. Remaining research decisions

1. **Overfitting of the UFLD family on ELAS.** In the pilot every UFLD run peaks at epoch 1 and then overfits; held-out F1 is 0.04–0.06. Decide whether to run the augmentation/regularisation ablation (`configs/ablations/augmentation.yaml`, ~2 h) before the full run and adopt a winner as the protocol for *all* models, or keep the current recipe and rely on HPO (learning rate, weight decay).
2. **Training-length confound for temporal models.** Temporal models warm-start from their baseline's best checkpoint and then train further; in the pilot most of lite v0.5's gain over the lite baseline came from that extra training (history ablation +0.017 of +0.145). Options: keep (the supervisor's protocol) and report the history ablation next to every comparison; or give the baseline an equal continued-training budget from its own best checkpoint.
3. **v0.4 warm start**: from the baseline (current, equal budget) or from v0.2 (`model.v04_warm_start: v02`, the original protocol, confounded by epochs).
4. **Primary metric strictness.** On ELAS, IoU ≥ 0.5 with 12-px lines allows about 4 px of offset while the annotation itself deviates 1–4 px; P = R = F1 on ELAS (see `docs/DATASETS.md`). Keep IoU 0.5 primary for CULane comparability (current) or promote IoU 0.35 for ELAS-only claims (it must be declared before the run).
5. **Whether to include preprocessing variants at all**; they are ablation-only options.
6. **Deployment precision**: FP16 is lossless on the pilot checkpoints; INT8 costs little for the UFLD models and more for lite v0.5 (`docs/DEPLOYMENT.md`); decide after measuring on the Jetson.
7. **License** of the repository (ported code from the official UFLD repository is MIT).

## 5. Original versus current implementation

| Aspect | Supervisor notebook (S) / ELAS script (U) | v0.2 | v0.3 (this hand-off) |
|---|---|---|---|
| Dataset | S: CULane; U: ELAS with wrong row geometry | ELAS corrected; CULane ported | + registry, TuSimple, OpenLane, CULane fixes, native metrics |
| Baseline | S: official UFLD; U: small CNN | official UFLD (ImageNet) + lite pair | unchanged |
| Temporal models | S: v0.2/v0.3/v0.4; U: v0.3/v0.4 lost | restored + lite v0.5 | unchanged (+ streaming hooks) |
| Split | S: no test; U: per-seed split, test near train | held-out scenes, purge, leakage check | unchanged |
| HPO / stats | S: seeded, variants only; U: unseeded, 4 seeds | equal budgets, 6 seeds, Holm | unchanged |
| Augmentation | U: photometric | photometric | + geometric with label re-encoding, extended photometric (off by default) |
| Input | RGB | RGB | + preprocessing ablations (off by default) |
| Resume | — | skipped partially trained variants | epoch-level, correct |
| Inference | batch evaluation | batch evaluation | + streaming with feature caching, UI |
| Deployment | — | — | ONNX, ONNX Runtime, TensorRT, FP16/INT8, Jetson scripts, budget simulation |
| Reproducibility | notebooks/scripts | package, YAML, tests | + Docker, `doctor`, confirmation guard, results page, hand-off ZIP |

Function-level traceability: `docs/MIGRATION.md`. The supervisor's reference code is not modified (`reference/` is unchanged; the notebook itself is not redistributed).

## 6. Known limitations

* No real-data validation for CULane, TuSimple, OpenLane.
* Pilot numbers are indicative (2 seeds, short training, no HPO).
* The Jetson results are simulations with assumed slowdowns.
* CULane native metric draws polylines, not the official cubic splines.
* Streaming preprocessing (PIL resize on CPU) dominates latency; moving it to the GPU must keep PIL's antialiased bilinear resampling.
* On Windows, DataLoader worker start-up adds ~20–30 s per training/validation loader.
* `lite_v05` INT8 and GridSample precision: see `docs/DEPLOYMENT.md`.

## 7. Package contents

`src/` (package), `configs/` (experiments, datasets, ablations, deployment profiles), `tests/`, `docs/`, `reference/` (original ELAS script, unchanged), `Dockerfile`, `docker-compose.yml`, `scripts/jetson/`, `README.md`, `pyproject.toml`, `requirements.txt`, and `pilot_results/elas_pilot/` (reports, CSVs, plots of the pilot; no checkpoints). Excluded: datasets, checkpoints, exported models, virtual environments, caches (`MANIFEST.txt` lists every file with its SHA-256).
