# TAC-UFLD: UFLD baseline vs. temporal lane detection

A reproducible research pipeline that compares the **Ultra-Fast Lane Detection (UFLD)** baseline (Qin et al., ECCV 2020) with lightweight **temporal** variants that fuse information from previous video frames. The reference dataset is **ELAS**; CULane, TuSimple and OpenLane adapters are included. The package covers the whole path from data to deployment: leakage-safe splits, training with multi-seed statistics, streaming inference with cached history features, ONNX / TensorRT export for embedded targets, a Streamlit interface and a static results page.

The package merges the supervisor's reference notebook (CULane, official UFLD, temporal variants v0.2–v0.4) and the ELAS development script, fixes the problems found in the first audit ([docs/AUDIT_REPORT.md](docs/AUDIT_REPORT.md)), and v0.3 adds the items audited in [docs/AUDIT_V03.md](docs/AUDIT_V03.md). **Supervisor hand-off: [docs/HANDOFF.md](docs/HANDOFF.md).**

> **Status (v0.3.0).** 159 tests pass (1 skipped: ONNX Runtime CUDA provider) on Windows with an RTX 3050, and the suite also passes in the Linux CPU Docker image ([docs/TESTING.md](docs/TESTING.md)). A GPU **pilot** of the full protocol (6 models × 2 seeds, ≤ 4 epochs, no HPO) completed in 2 h 22 min on an RTX 3050: [docs/PILOT_FINDINGS.md](docs/PILOT_FINDINGS.md) and the results page. Pilot numbers are indicative; the full experiment (`configs/elas.yaml`, 6 seeds, HPO) has not been run yet. CULane / TuSimple / OpenLane are tested on synthetic copies of their layouts only; TensorRT ran on the development GPU, not on a Jetson.

---

## Contents

- [Quick start](#quick-start)
- [Commands](#commands)
- [What was fixed](#what-was-fixed)
- [Models](#models)
- [Experimental protocol](#experimental-protocol)
- [Datasets](#datasets)
- [Streaming and deployment](#streaming-and-deployment)
- [Interface and results page](#interface-and-results-page)
- [Docker](#docker)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Documentation](#documentation)
- [References](#references)

---

## Quick start

Requirements: Python ≥ 3.10, Git, an NVIDIA GPU for training (CPU works for tests and inference).

```powershell
git clone https://github.com/TiagoRodrigues-GitH/tac-ufld-lane-detection.git
cd tac-ufld-lane-detection
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu126   # match your CUDA driver
pip install -e ".[all]"                                               # + dev, ui, deploy extras
python -m tac_ufld doctor                                             # environment, GPU, datasets
```

ELAS: download from the [ELAS repository](https://github.com/rodrigoberriel/ego-lane-analysis-system); each scene needs `config.xml`, `groundtruth.xml` and `images/images/lane_<id>.png`. The default location is `../../datasets/dataset_elas_v1` relative to the project; otherwise set `ELAS_ROOT` (or edit `configs/datasets.yaml`). The ImageNet ResNet-18 weights (45 MB) are downloaded once to `~/.cache/torch/hub`. Optional desktop TensorRT: `pip install -e ".[tensorrt]"` (2.3 GB download).

## Commands

```powershell
pytest -q                                                        # tests (~10 min)
python -m tac_ufld check-data --config configs/elas.yaml          # splits, leakage, label overlays
python -m tac_ufld sanity --config configs/elas.yaml --real      # model checks, no training
python -m tac_ufld run --config configs/elas_smoke.yaml          # plumbing smoke run, ~9 min (numbers meaningless)
python -m tac_ufld run --config configs/elas_pilot.yaml          # GPU pilot, ~2.5 h on an RTX 3050
python -m tac_ufld run --config configs/elas.yaml --confirm      # full experiment, ~1.5-2 days on an RTX 3050
python -m tac_ufld run --config configs/elas.yaml --confirm --resume   # continue after an interruption
python -m tac_ufld ablate --spec configs/ablations/augmentation.yaml    # plan; add --confirm to run
python -m tac_ufld datasets | validate-dataset --dataset tusimple
python -m tac_ufld export --checkpoint <ckpt> --int8 --tensorrt fp16 --eval-frames 240
python -m tac_ufld stream --checkpoint <ckpt> --video drive.mp4
python -m tac_ufld benchmark --checkpoints <ckpt> ... --profile configs/deploy/jetson_orin_nano_assumed.yaml
python -m tac_ufld ui                                            # Streamlit interface
python -m tac_ufld site --runs results/elas_pilot                # static results page (GitHub Pages)
python -m tac_ufld package --include-results results/elas_pilot  # hand-off ZIP
```

Configs marked `requires_confirmation` (the full ELAS, CULane, TuSimple and OpenLane runs) refuse to start without `--confirm`, so a multi-day run is never started by accident. `--resume` continues an interrupted variant from its last epoch and reuses finished HPO studies and seeds.

---

## What was fixed

| # | Problem | Now |
|---|---|---|
| 1 | **ELAS labels were geometrically wrong** (p1..p4 at ROI heights 0, ⅓, ⅔, 1; the data shows 0, ¼, ½, 1: straight lanes fit a line within 1–4 px instead of 16–22 px). | Corrected, regression-tested on the real data; unusable points ignored; rows outside the ROI ignored; slots from the annotation. |
| 2 | **Anchor metric inflated** (wrong positions counted as neither FP nor FN). | A wrong position counts as FP + FN. |
| 3 | **Confounded comparison** (custom CNN as "UFLD"; v0.3/v0.4 designs lost; 4× smaller temporal backbone). | Official UFLD re-implemented (ResNet-18, ImageNet); supervisor's v0.2–v0.4 restored; lite pair with the same backbone; warm starts within each family. |
| 4 | **Test not independent** (per-seed split; test frames 9 frames from training frames). | Held-out scenes, seed-independent split, purge gap, leakage checker. |
| 5 | **Statistics could not support claims.** | 6 seeds, exact Wilcoxon + Holm, `underpowered` flag; seeded fresh HPO with equal budgets. |
| 6 | v0.3: `--resume` treated a partially trained variant as finished. | Completion proven by the history file; epoch-level resume. |
| 7 | v0.3: CULane `temporal_step: 30` found no history in the 90-frame drivers; slots from a geometric rule. | Step 90 (supervisor's value); official segmentation-label slots. |
| 8 | v0.3: ONNX export left the model in training mode and changed BatchNorm statistics. | Fixed and guarded; numerical checks against PyTorch. |

---

## Models

| Key | Description | Warm start | Paired with |
|---|---|---|---|
| `ufld_baseline` | UFLD, ResNet-18, single frame, official focal loss | ImageNet | — |
| `ufld_v02` | shared ResNet-18 on 3 frames + learnable per-frame weights; + weighted existence BCE + temporal consistency | `ufld_baseline` | `ufld_baseline` |
| `ufld_v03` | per-pixel gated fusion over frames; + gate prior | `ufld_baseline` | `ufld_baseline` |
| `ufld_v04` | v0.2 architecture + soft-argmax coordinate loss | `ufld_baseline` (`v04_warm_start: v02` for the original protocol) | `ufld_baseline` |
| `ufld_v06` (v0.4) | lite v0.5's learned warp + residual gate on UFLD layer-4 features (aligned fusion) | `ufld_baseline` | `ufld_baseline`, `ufld_baseline_ct` |
| `ufld_v07` (v0.4) | ConvGRU over the frames, zero-initialised residual read-out; can carry one state per stream | `ufld_baseline` | `ufld_baseline`, `ufld_baseline_ct` |
| `ufld_baseline_ct` (v0.4) | control: the baseline trained again from its best checkpoint with the temporal variants' schedule | `ufld_baseline` | `ufld_baseline` |
| `lite_baseline` | 4-block CNN + per-anchor MLP head | scratch | — |
| `lite_v05` | same backbone + learned feature warping + residual gated fusion | `lite_baseline` | `lite_baseline`, `lite_baseline_ct` |
| `lite_v06` (v0.4) | same backbone + ConvGRU fusion | `lite_baseline` | `lite_baseline`, `lite_baseline_ct` |
| `lite_baseline_ct` (v0.4) | control for the lite family | `lite_baseline` | `lite_baseline` |

Plain-language descriptions of every model and how they differ: the results page and
`src/tac_ufld/models/descriptions.py`. The v0.4 tools for making the temporal models earn their cost
(overfitting remedies, history-length ablation, current-frame degradation, aligned and recurrent fusion,
equal-training controls, Kalman reference): [docs/TEMPORAL_IMPROVEMENTS.md](docs/TEMPORAL_IMPROVEMENTS.md).
Plan, including the embedded phase: [docs/ROADMAP.md](docs/ROADMAP.md).

Every model predicts, for each row anchor and lane slot, one of `griding_num + 1` classes (a horizontal cell or "no lane"), decoded with the official soft-argmax; existence is `1 − p(no lane)`.

## Experimental protocol

1. **Data**: the dataset is parsed into records; `check-data` writes overlays in which the encoded targets must lie on the painted markings.
2. **Split** (ELAS): held-out test scenes `BR_S02`, `VIX_S05`, `VV_S03`; train / val / seen-scene test from 60-frame blocks of the other seven scenes with a ≥ 10-frame purge gap; independent of the training seed. Sizes: train 2,890 · val 1,019 · seen-scene test 630 · held-out test 2,610 frames.
3. **HPO** (seeded TPE): the same budget for every variant, baselines included, scored on validation.
4. **Training**: each seed trains every variant with its best hyper-parameters; checkpoints and early stopping on validation `lane_f1_iou50`.
5. **Post-processing**: tuned on validation for every model, plus one common fixed protocol.
6. **Testing**: held-out scenes (primary) and seen scenes (secondary); temporal ablation with history replaced by the current frame.
7. **Report**: mean ± std and 95 % CI, paired tests against the same-family baseline, efficiency, the protocol actually used and every non-default setting.

| Metric | Definition |
|---|---|
| `lane_f1_iou50` (**primary**), precision, recall, F2, TP/FP/FN | lanes drawn 30 px wide at 1640 px (scaled: 12 px at 640), Hungarian matching, TP if IoU ≥ 0.5 |
| `lane_f1_iou35` | same at IoU 0.35 |
| `pixel_f1`, `anchor_f1` | mask overlap; per (anchor, lane) cell with wrong position = FP + FN (tolerance 10 px) |
| `jitter_px` | mean change of predicted x between consecutive frames |
| `native_*` | dataset-native metrics (CULane, TuSimple `LaneEval`, OpenLane), never compared across datasets |
| efficiency | parameters, GMACs, batch-1 latency, FPS, peak GPU memory |

On ELAS both ego lanes are annotated in essentially every frame and every model predicts both, so precision = recall = F1 at the lane level: errors are misplaced lanes. IoU 0.5 with 12-px lines allows ≈ 4 px of offset, close to the annotation's own 1–4 px deviation; IoU 0.35 shows the lenient view.

## Datasets

`configs/datasets.yaml` enables datasets individually (only ELAS by default); disabled datasets are never scanned, and an enabled dataset with a missing root is an explicit error. Details, formats, validation findings and per-dataset protocols: [docs/DATASETS.md](docs/DATASETS.md).

| Dataset | Adapter status |
|---|---|
| ELAS | validated on real data |
| CULane | synthetic layout tests; real-data check: `validate-dataset --dataset culane` |
| TuSimple | new; synthetic layout tests |
| OpenLane (2D) | new; synthetic layout tests |

## Streaming and deployment

`StreamingLaneDetector` (`reset_stream`, `infer_frame`, `infer_video`) keeps per-stream state and caches the per-frame features of the history frames, which is exact because in eval mode each frame's features depend on that frame only (tested against the full forward pass). On the RTX 3050 this halves a temporal model's compute (≈ 16 → 8 ms). `export` writes the streaming split (`encoder.onnx`, `head.onnx`, `clip.onnx`), FP16 and calibrated INT8 graphs, TensorRT engines, a `deployment.json` model card and numerical + task-level checks. Jetson procedure, precision results and the budget simulation: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Interface and results page

`python -m tac_ufld ui` opens a Streamlit interface: select checkpoints, run them on uploaded images or videos (processed frame by frame), compare models side by side with lane confidences, inspect architectures (graph or layer table), training curves, experiment metrics and latency, and download overlays and summaries. It never imports the training code. `python -m tac_ufld site --runs results/<run>` writes a static page (`site/`); `bash scripts/publish_pages.sh results/<run>` builds it and pushes it to the `gh-pages` branch (then, once: *Settings → Pages → Deploy from a branch → gh-pages / (root)*). On a private repository GitHub Pages needs a paid plan (GitHub Pro is free with the Student Developer Pack), and the published page is public.

## Docker

```powershell
docker compose run --rm doctor
docker compose run --rm pilot
docker compose up ui            # http://localhost:8501
```

Datasets are mounted read-only, results and the weight cache as volumes; nothing is baked into the image. GPU, WSL 2 and CPU instructions: [docs/DOCKER.md](docs/DOCKER.md).

## Outputs

```
results/<name>/
├── config_resolved.yaml, environment.json, run.log, all_results.csv
├── data/            split_manifest.csv, split_summary.csv, split_report.json, dataset_stats.csv, label_check/
├── hpo/             <variant>_trials.csv, <variant>_best.json, best_params.json
├── seed_<n>/        checkpoints/<variant>.pt (+ config, model card), history_<variant>.csv, plots/,
│                    postprocess/<variant>_{val_sweep.csv,tuned.json}, per_frame/, predictions/, visuals/, results.csv
├── deploy/          <variant>_seed<n>/ (ONNX, FP16, INT8, engines, deployment.json, numerical_checks.json)
├── tensorboard/
└── report/          REPORT.md (with the protocol used), config_changes.json, results.xlsx, aggregate_*.csv,
                     paired_tests_test_tuned.csv, temporal_ablation.csv, efficiency.csv, figures
```

## Configuration

Every setting lives in YAML (`configs/`); unknown keys are rejected and cross-field rules validated. Main sections: `data` (dataset, root, scenes, image size, anchors, temporal context, `preprocessing`, `augmentation` incl. `geometric`, `split`), `model` (variants, backbone, pretrained, warm starts, dropout), `train` (seeds, optimiser, schedule, clipping, early stopping, loss incl. `label_smoothing`, `save_last`), `hpo`, `evaluation`. Options: [docs/PREPROCESSING_AUGMENTATION.md](docs/PREPROCESSING_AUGMENTATION.md).

## Project structure

```
src/tac_ufld/
├── config.py, cli.py, commands.py      YAML schema; CLI
├── data/        adapters (elas, culane, tusimple, openlane), registry, splits, targets,
│                preprocess, geometric, transforms, dataset, validation
├── models/      resnet, ufld (baseline + v0.2-v0.4), lite (baseline + v0.5), registry
├── losses.py, metrics.py, decoding.py, postprocess.py
├── training/    trainer (resume), hpo
├── evaluation/  predictor, evaluator, native metrics, efficiency, stats, hardware simulation
├── inference/   model card, checkpoint loading, backends, streaming
├── deploy/      ONNX export, ONNX Runtime, TensorRT, FP16/INT8, checks
├── ui/          Streamlit app (core.py is testable without Streamlit)
├── visualization/, reporting.py, experiment.py, ablation.py, sanity.py, site.py, package.py, doctor.py, sim/
configs/         elas{,_smoke,_pilot}.yaml, culane/tusimple/openlane.yaml, datasets.yaml, ablations/, deploy/
scripts/jetson/  engine build and on-device measurement
tests/           unit, integration, fixtures of the CULane/TuSimple/OpenLane layouts
docs/            hand-off, audits, datasets, deployment, Docker, experiment matrix, testing, pilot findings
reference/       the original ELAS script, unchanged
```

## Testing

```powershell
pytest -q                     # everything available locally
pytest -m "not dataset"       # without the local ELAS copy
pytest -m network             # ImageNet download + strict loading
```

Commands, results and skip reasons: [docs/TESTING.md](docs/TESTING.md).

## Documentation

| Document | Content |
|---|---|
| [HANDOFF.md](docs/HANDOFF.md) | supervisor hand-off: what to run, status by category, open decisions |
| [PILOT_FINDINGS.md](docs/PILOT_FINDINGS.md) | what the GPU pilot shows |
| [EXPERIMENT_MATRIX.md](docs/EXPERIMENT_MATRIX.md) | every experiment with command and time budget |
| [DATASETS.md](docs/DATASETS.md) | registry, adapters, protocols, validation |
| [PREPROCESSING_AUGMENTATION.md](docs/PREPROCESSING_AUGMENTATION.md) | input ablations, augmentation, training controls, ablation runner |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | streaming, ONNX, TensorRT, INT8, Jetson, budget simulation, CARLA |
| [DOCKER.md](docs/DOCKER.md) | images, Compose, GPU on Linux / WSL 2 |
| [TESTING.md](docs/TESTING.md) | test commands and results |
| [AUDIT_V03.md](docs/AUDIT_V03.md), [AUDIT_REPORT.md](docs/AUDIT_REPORT.md), [MIGRATION.md](docs/MIGRATION.md) | audits and function-level traceability |

**License.** None chosen yet. The ported model and loss code comes from the supervisor's notebook and the official UFLD repository (MIT); agree on a license with the supervisor before making the repository public.

## References

* Z. Qin, H. Wang, X. Li. *Ultra Fast Structure-aware Deep Lane Detection*, ECCV 2020. [arXiv:2004.11757](https://arxiv.org/abs/2004.11757) · [code](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
* Z. Qin, P. Zhang, X. Li. *Ultra Fast Deep Lane Detection with Hybrid Anchor Driven Ordinal Classification*, TPAMI 2022. [arXiv:2206.07389](https://arxiv.org/abs/2206.07389)
* R. F. Berriel et al. *Ego-Lane Analysis System (ELAS): Dataset and Algorithms*, Image and Vision Computing 2017. [arXiv:1806.05984](https://arxiv.org/abs/1806.05984)
* X. Pan et al. *Spatial As Deep: Spatial CNN for Traffic Scene Understanding* (CULane), AAAI 2018.
* TuSimple lane detection benchmark, 2017. · H. Chen et al. *PersFormer* / OpenLane, ECCV 2022.
