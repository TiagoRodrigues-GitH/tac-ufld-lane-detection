# TAC-UFLD: UFLD baseline vs. temporal lane detection

This is a reproducible research pipeline. It compares the **Ultra-Fast Lane Detection (UFLD)** baseline (Qin et al., ECCV 2020) with lightweight **temporal** variants that fuse information from previous video frames. It starts with the **ELAS** dataset; a CULane adapter is included. The code is modular, configured with YAML files, and tested.

| | |
|---|---|
| **Author** | Tiago Rodrigues · Universidade Tecnológica Federal do Paraná (UTFPR) |
| **Date** | 2026-09-29 |
| **Context** | Research project, supervised (lane detection for ADAS) |
| **Stack** | Python · PyTorch · Optuna |

> **Resumo (PT).** Pipeline reprodutível que compara o detector de faixas UFLD com variantes temporais leves, no dataset brasileiro ELAS e no OpenLane, com protocolo estatístico (várias sementes, testes pareados) e robustez.

The package merges two earlier single-file programs: the supervisor's reference notebook (CULane, official UFLD, temporal variants v0.2–v0.4) and the ELAS development script. It also fixes the problems found in the code audit ([docs/AUDIT_REPORT.md](docs/AUDIT_REPORT.md)). [docs/MIGRATION.md](docs/MIGRATION.md) maps every original function to its new location.

> **Status.** The pipeline has been verified: 50 tests pass, including an end-to-end run on synthetic data, a regression test for the label geometry on the real ELAS data, and strict loading of the ImageNet weights. A smoke run on real ELAS data completes on the GPU. **No research results have been produced yet.** Results produced by the old scripts are invalid because of the label bug (see [Fixes](#what-was-fixed)).
>
> *Preliminary sanity check (not a result):* the UFLD baseline trained for 3 epochs, 1 seed, without HPO learns (lane F1 @ IoU 0.5 = 0.34 on seen-scene test frames), but reaches only 0.06 on the held-out scenes. It also overfits within 2 epochs (train focal 1.13 → 0.08 while validation rises). Generalisation to unseen roads is therefore the central question. The previous protocol could not measure it, because its test frames came from the training scenes.

---

## Contents

- [What was fixed](#what-was-fixed)
- [Models](#models)
- [Experimental protocol](#experimental-protocol)
- [Setup (Windows + VS Code)](#setup-windows--vs-code)
- [Running](#running)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Limitations and roadmap](#limitations-and-roadmap)
- [References](#references)

---

## What was fixed

| # | Problem in the previous code | Now |
|---|---|---|
| 1 | **ELAS labels were geometrically wrong.** Points p1..p4 were placed at ROI heights 0, ⅓, ⅔, 1. Across all 22 scenes the data shows they sit at **0, ¼, ½, 1**: straight lanes fit a line within 1–4 px, versus 16–22 px before. Every training target and every metric was affected. | Corrected and covered by a regression test on the real data. Points that cannot be drawn are *ignored*, not trained as "no lane". Rows outside each scene's ROI are ignored (as with the official UFLD ignore label). Left/right slots come from the annotation, not from sorting by x. |
| 2 | **The anchor-level metric was inflated.** Anchors predicted at the wrong position counted as neither FP nor FN (half the anchors wrong still gave F1 = 1.0). | A wrong position counts as FP + FN, as in the supervisor's code. |
| 3 | **The comparison was confounded.** The "UFLD baseline" was a custom CNN; v0.3/v0.4 had lost their designs; the temporal models used a 4× smaller backbone trained from scratch at 0.1× learning rate. | The official UFLD is re-implemented (ResNet-18, ImageNet weights). The supervisor's v0.2/v0.3/v0.4 are restored. The ELAS script's warped fusion is kept as `lite_v05` and paired with `lite_baseline`, which has the **same** backbone. Every temporal model is warm-started from its own family's baseline. |
| 4 | **Test data was not independent.** The split was redrawn for every seed, and test frames were as close as 9 frames to training frames in the same scenes. | Whole **scenes are held out** for testing. The split depends only on `split_seed`, with a purge gap ≥ the temporal context. A leakage checker stops the run on any overlap. |
| 5 | **Statistics could not support claims.** With 4 seeds a Wilcoxon test cannot reach p < 0.05. Optuna was unseeded and reused old studies. Only the temporal models were tuned. | 6 seeds by default, exact paired Wilcoxon with Holm correction, and an explicit `underpowered` flag. Seeded, fresh HPO studies with the **same budget for the baselines**. |

Further changes:
- Validation-tuned post-processing for *every* model, plus one common fixed protocol.
- A single declared selection metric used everywhere.
- CULane-style lane F1 at IoU 0.5 with Hungarian matching as the primary metric.
- Windows-safe I/O for paths such as `Residência`.

---

## Models

| Key | Description | Origin | Warm start | Paired with |
|---|---|---|---|---|
| `ufld_baseline` | UFLD, ResNet-18, single frame, official focal loss | supervisor (official `parsingNet`) | ImageNet | — |
| `ufld_v02` | Shared ResNet-18 on each of 3 frames + learnable per-frame weights; + weighted existence BCE + temporal consistency | supervisor v0.2 | `ufld_baseline` | `ufld_baseline` |
| `ufld_v03` | Per-pixel gated fusion over frames; + gate prior | supervisor v0.3 | `ufld_baseline` | `ufld_baseline` |
| `ufld_v04` | v0.2 architecture + soft-argmax coordinate loss | supervisor v0.4 | `ufld_baseline` (set `v04_warm_start: v02` for the original protocol) | `ufld_baseline` |
| `lite_baseline` | 4-block CNN + per-anchor MLP head (row classification) | ELAS script | scratch | — |
| `lite_v05` | Same backbone + learned feature warping + residual gated fusion | ELAS script | `lite_baseline` | `lite_baseline` |

Every model predicts, for each row anchor and lane slot, one of `griding_num + 1` classes (a horizontal cell or "no lane"), as in UFLD. Location is decoded with the official soft-argmax. Existence is `1 − p(no lane)`.

---

## Experimental protocol

1. **Data.** ELAS scenes are parsed into records with the corrected geometry. `check-data` writes overlays in which the encoded targets (cyan dots) must lie on the painted markings.
2. **Split** (fixed, independent of the training seed; defaults in `configs/elas.yaml`):
   * `test` — held-out scenes `BR_S02`, `VIX_S05` and `VV_S03` (unseen roads; complete sequences for jitter).
   * `train` / `val` / `seen_test` — 60-frame blocks of the other seven scenes. Frames within 10 frames of another split are purged.

   Current sizes: **train 2,890 · val 1,019 · seen-scene test 630 · held-out test 2,610 frames.**
3. **HPO** (Optuna, seeded TPE): each variant, baselines included, gets `n_trials` trials of `epochs` epochs on a training subset, scored on validation. Test data is never used.
4. **Training**: for each seed, each variant is trained with its best hyper-parameters. Checkpoints are selected and training is stopped early on validation lane F1 @ IoU 0.5.
5. **Post-processing**: the existence threshold, minimum points and polynomial refinement are tuned **on validation** for every model. A common fixed protocol is reported as well.
6. **Testing**: the held-out-scene test (primary) and the seen-scene test (secondary). Temporal models are also evaluated with their history replaced by the current frame (temporal ablation).
7. **Report**: mean ± std and 95 % CI over seeds, paired tests against the same-family baseline, efficiency, figures, Excel.

**Metrics**

| Metric | Definition |
|---|---|
| `lane_f1_iou50` (**primary**), precision, recall, F2, TP/FP/FN | Lanes drawn with CULane's 30 px width scaled to the image (12 px at 640 px), matched one-to-one with the Hungarian algorithm; TP if IoU ≥ 0.5 |
| `lane_f1_iou35` | Same at IoU 0.35 (lenient; the ELAS script's old primary metric) |
| `pixel_f1` | Micro-averaged overlap of drawn lane masks (custom; not comparable with published results) |
| `anchor_f1` | Per (row anchor, lane) cell; a wrong position is FP + FN; tolerance 10 px |
| `jitter_px` | Mean change in predicted x (original pixels) between consecutive frames |
| `condition_f1_iou50_<tag>` | Lane F1 per scene condition (rainy, occlusion, shaky, transition, nominal) |
| efficiency | parameters, GMACs, batch-1 latency, FPS, peak GPU memory |

> **How strict IoU 0.5 is on ELAS.** With a 12 px line width, IoU ≥ 0.5 allows only about 4 px of perpendicular offset. The ELAS annotation itself deviates 1–4 px from a straight line (see `label_geometry_check.csv`), so label noise is close to the tolerance. IoU 0.5 is kept as primary for comparability with CULane; `lane_f1_iou35` shows how results change under a more lenient match. Model comparisons are paired on the same frames, so both metrics are valid for *relative* claims.

---

## Setup (Windows + VS Code)

Requirements: Python ≥ 3.10, an NVIDIA GPU is recommended (CPU works for tests), and Git.

```powershell
# 1. Clone and enter the project
git clone https://github.com/TiagoRodrigues-GitH/tac-ufld-lane-detection.git
cd tac-ufld-lane-detection

# 2. Virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. PyTorch for your CUDA version (see https://pytorch.org/get-started/locally/)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 4. This package + dev tools
pip install -e ".[dev]"
```

In VS Code, choose **Python: Select Interpreter → .venv**. Tests appear in the Testing panel (pytest).

**Dataset.** Download ELAS from the [ELAS repository](https://github.com/rodrigoberriel/ego-lane-analysis-system). Each scene needs `config.xml`, `groundtruth.xml` and `images/images/lane_<id>.png`. By default the data is expected at `../../datasets/dataset_elas_v1` relative to the project folder. You can change this with any of:

```powershell
$env:TAC_UFLD_DATA_ROOT = "D:\datasets\dataset_elas_v1"      # environment variable, or
python -m tac_ufld run --config configs/elas.yaml --data-root "D:\datasets\dataset_elas_v1"
```

The ImageNet ResNet-18 weights (45 MB) are downloaded once to `~/.cache/torch/hub`.

---

## Running

```powershell
# 0. Tests (about 3 min)
pytest -q

# 1. Check data, split, leakage and label geometry -> results/elas_full/data/
python -m tac_ufld check-data --config configs/elas.yaml

# 2. Smoke run: every stage on a small subset, a few minutes. Numbers are not results.
python -m tac_ufld run --config configs/elas_smoke.yaml

# 3. Full experiment (6 variants x 6 seeds, HPO) -- long; see the runtime note below
python -m tac_ufld run --config configs/elas.yaml

# Useful options
python -m tac_ufld run --config configs/elas.yaml --variants ufld_baseline ufld_v02 --seeds 1 2 3 4 5 6
python -m tac_ufld run --config configs/elas.yaml --resume      # continue after an interruption
python -m tac_ufld run --config configs/elas.yaml --no-hpo      # use config defaults
tensorboard --logdir results/elas_full/tensorboard
```

`--resume` reuses finished HPO results, checkpoints and evaluated seeds in the same output folder. Delete the folder (or use `--name`) after changing the config.

**Runtime.** On an RTX 3050 (6 GB), the smoke run takes about 9 minutes. The full configuration trains 6 variants × 6 seeds × up to 50 epochs on about 2,900 frames, plus HPO (20 trials × 10 epochs per variant). Expect **several days** of GPU time. To reduce it: run a subset of variants (`--variants`), fewer HPO trials, or `--no-hpo` with `hpo.load_best_params` from an earlier run. Keep ≥ 6 seeds for the final comparison.

---

## Outputs

```
results/<name>/
├── config_resolved.yaml, environment.json, run.log
├── data/            split_manifest.csv, split_summary.csv, dataset_stats.csv,
│                    label_geometry_check.csv, label_check/*.png
├── hpo/             <variant>_trials.csv, <variant>_best.json, best_params.json
├── seed_<n>/
│   ├── checkpoints/<variant>.pt        best validation checkpoint (+ hyper-parameters, config hash)
│   ├── history_<variant>.csv, plots/   training curves (PNG + PDF)
│   ├── postprocess/<variant>_val_sweep.csv
│   ├── per_frame/<variant>_<split>.csv
│   ├── predictions/<split>/<variant>/  CULane-format .lines.txt (first seed)
│   ├── visuals/<split>/<variant>/{errors,successes}/*.png   original | ground truth | prediction
│   ├── visuals/test_sequence.mp4       consecutive test frames, all models
│   └── results.csv
├── all_results.csv
├── tensorboard/
└── report/          REPORT.md, results.xlsx, aggregate_*.csv, paired_tests_test_tuned.csv,
                     temporal_ablation.csv, efficiency.csv, figures (PNG + PDF)
```

---

## Configuration

Every setting lives in YAML (`configs/`). Unknown keys are rejected, and cross-field rules are validated (e.g. `purge_frames ≥ temporal_step × (num_frames − 1)`). Main sections:

| Section | Key fields |
|---|---|
| `data` | `scenes`, `scene_tags`, `img_h/img_w`, `num_row_anchors`, `row_anchor_range`, `griding_num`, `num_frames`, `temporal_step`, `augmentation`, `split` |
| `data.split` | `split_seed`, `test_scenes`, `block_size`, `purge_frames`, `val_fraction`, `seen_test_fraction`, `max_*_frames` |
| `model` | `variants`, `backbone`, `pretrained`, `v04_warm_start`, `lite_history_encoder` |
| `train` | `seeds`, `epochs`, `optimizer`, `lr`, `lr_fusion`, `scheduler`, `warmup_iters`, `early_stopping_patience`, `lambda_temporal`, `lambda_coord`, `loss` |
| `hpo` | `enabled`, `n_trials`, `epochs`, `sampler_seed`, `max_train_frames`, `search_space` (per parameter: `low`, `high`, `log`, optional `variants`) |
| `evaluation` | `iou_thresholds` (first = primary), `selection_metric`, `common_postprocess`, `postprocess_grid`, `anchor_tolerance_px` |

Outputs are tagged with a hash of the resolved config (`environment.json`, checkpoints).

---

## Project structure

```
src/tac_ufld/
├── config.py            YAML -> typed dataclasses, validation, config hash
├── data/
│   ├── types.py         FrameRecord (common representation for every dataset)
│   ├── base.py          LaneDatasetAdapter interface
│   ├── elas.py          ELAS adapter (corrected geometry)
│   ├── culane.py        CULane adapter (ported, not yet validated on real data)
│   ├── splits.py        scene hold-out + temporal blocks + leakage checks
│   ├── targets.py       row-anchor target encoding (with ignore index)
│   ├── transforms.py    image loading, normalisation, augmentation
│   └── dataset.py       torch Dataset of temporal clips
├── models/
│   ├── resnet.py        torchvision-compatible ResNet-18/34
│   ├── ufld.py          UFLDNet + v0.2/v0.3/v0.4 temporal models
│   ├── lite.py          lightweight baseline + v0.5 warped fusion
│   └── registry.py      variant definitions, warm start, training order
├── decoding.py          logits -> existence + soft-argmax position
├── losses.py            official UFLD losses + variant terms
├── metrics.py           anchor / lane (IoU, Hungarian) / pixel / temporal metrics
├── postprocess.py       predictions -> lane polylines
├── training/            trainer.py (loop), hpo.py (Optuna)
├── evaluation/          predictor, evaluator, efficiency, stats
├── visualization/       overlays (Unicode-safe), plots
├── reporting.py         CSV / Excel / Markdown / .lines.txt export
├── experiment.py        end-to-end orchestration
└── cli.py               command-line interface
configs/                 elas.yaml, elas_smoke.yaml, culane.yaml
tests/                   unit + integration tests (synthetic ELAS generator in conftest.py)
docs/                    AUDIT_REPORT.md, MIGRATION.md
reference/               the original ELAS script, unchanged, for traceability
```

---

## Testing

```powershell
pytest -q                 # everything available offline and locally
pytest -m "not dataset"   # without the local ELAS copy
pytest -m network         # downloads the ImageNet weights and checks strict loading
```

The tests cover:
* ELAS geometry (synthetic and real data);
* missing-point handling and lane slots;
* target encoding and ignore regions;
* the anchor-metric fix, Hungarian matching and CULane line width;
* split disjointness, purge gaps and seed independence;
* forward shapes and warm starts for all models;
* the focal loss against the supervisor's formula;
* Wilcoxon/Holm statistics;
* a full end-to-end run on synthetic ELAS scenes.

---

## Limitations and roadmap

* **No results yet.** The next step is the full ELAS experiment (`configs/elas.yaml`).
* **UFLD fidelity.** The auxiliary segmentation branch (`use_aux`) is not used, because ELAS has no segmentation masks (the supervisor's code did not use it either). There is no geometric augmentation.
* **ELAS specifics.** Frames without any lane (372) are excluded by default. Condition tags exist only for the 10 planned scenes. Unknown scenes are not treated as "nominal".
* **CULane.** The adapter is ported but untested on real data. The step between stored frames (`temporal_step`) must be checked against your copy.
* **Not yet implemented:** TuSimple and OpenLane adapters (a new `LaneDatasetAdapter` subclass each), preprocessing experiments (grayscale / edges / Canny / Hough as configurable ablations, to be kept only if they help), Docker, and export for embedded deployment (ONNX / TensorRT, streaming inference that caches history features).
* **License.** None chosen yet. The ported model and loss code comes from the supervisor's notebook and the official UFLD repository (MIT); agree on a license with the supervisor before making the repository public.

---

## References

* Z. Qin, H. Wang, X. Li. *Ultra Fast Structure-aware Deep Lane Detection*, ECCV 2020. [arXiv:2004.11757](https://arxiv.org/abs/2004.11757) · [code](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
* Z. Qin, P. Zhang, X. Li. *Ultra Fast Deep Lane Detection with Hybrid Anchor Driven Ordinal Classification*, TPAMI 2022. [arXiv:2206.07389](https://arxiv.org/abs/2206.07389)
* R. F. Berriel et al. *Ego-Lane Analysis System (ELAS): Dataset and Algorithms*, Image and Vision Computing 2017. [arXiv:1806.05984](https://arxiv.org/abs/1806.05984) · [code/data](https://github.com/rodrigoberriel/ego-lane-analysis-system)
* X. Pan et al. *Spatial As Deep: Spatial CNN for Traffic Scene Understanding* (CULane), AAAI 2018.
