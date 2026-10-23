# CULane plan (2026-10-23): improved prompt, problems, decisions, how to run

**PT (resumo).** O artigo 1 compara modelos leves e temporais com o UFLD v1 original no CULane, o benchmark do
artigo do UFLD. Plano principal (core) no PC: 6 modelos, 7 sementes, baselines com 50 épocas e ajuste fino de 15
épocas nos modelos que partem do baseline; semente 1 em ~2,5 dias, 7 sementes em ~15-16 dias. Plano completo
(full) no cluster: 10 modelos, 50 épocas para todos. Tudo roda por um arquivo só: `dist/tac_ufld_run.py`.

## 1. Improved prompt (use it to restart the work in any session)

> **Goal.** Article 1 (IEEE ADAS journal): a lightweight, temporal lane detector for embedded ADAS, compared
> fairly with the original UFLD v1 (Qin et al., ECCV 2020) on CULane, the benchmark of that article.
>
> **Baseline fidelity (gate before any claim).** UFLD v1 ResNet-18 trained with the article's recipe (288x800,
> 18 row anchors, 200 cells, 4 lanes, auxiliary segmentation loss, SGD lr 0.1, momentum 0.9, weight decay
> 1e-4, batch 32, warm-up 695 iterations, decay at epochs 25 and 38, 50 epochs, ImageNet weights) must reach the
> published CULane F1 68.4 (+/- 0.5) with the CULane protocol (IoU 0.5, 30 px lines at 1640x590,
> spline-interpolated lanes): `configs/culane_ufld_reproduction.yaml`. Also evaluate the authors' released
> `culane_18.pth` with our pipeline (checks the model port and the evaluator at once). No HPO on UFLD v1.
>
> **Splits.** Official lists: train 88,880 / val 9,675 / test 34,680 (9 categories). Optuna uses its own split
> carved from train by whole clips (hpo_train / hpo_val), never the official val or test. The official val selects
> the checkpoint and the post-processing; the test set is read once, at the end.
>
> **Models.** Core: UFLD v1, lite baseline, lite baseline_ct (equal training), lite TSM, lite TSM static
> (capacity), lite v0.6 occ. Full adds UFLD v0.7 / v0.8, UFLD baseline_ct and lite v0.6. Dropped after pilot 3:
> distillation (teacher weaker than the students), lite v0.7 (attention ignores the history), lite v0.5.
> Warm-started models score their starting weights as epoch 0. Controls inherit the tuned settings of the model
> they control (`hpo.inherit`).
>
> **Budget.** 7 seeds. Baselines: fixed 50 epochs, as UFLD (the article has no early stopping), best epoch on the
> official val. Warm-started models: fixed fine-tuning budget (core 15 epochs, full 50), epoch-0 candidate.
>
> **Metrics (IEEE).** CULane F1 at IoU 0.5 in total and per category, FP on cross, precision, recall, FP/FN per
> frame, frame accuracy, position error near/far, jitter, the Kalman-tracker reference, robustness under 4
> protocols (current, persistent, independent, held-out) plus the real conditions (crowded = real occlusion by
> vehicles, dazzle, night, shadow), cost (parameters, GMACs at 288x800, latency; later Jetson / TensorRT FP16).
> Statistics: 7 paired seeds, exact Wilcoxon, Holm over the 2 declared primary comparisons (lite_tsm vs
> ufld_baseline, lite_tsm vs lite_tsm_static), effect sizes and 95% CIs.
>
> **Scope.** Lane change and LDW are evaluated on ELAS (30 fps, labelled events), not on CULane (no labels, frames
> 1-3 s apart). OpenLane is parked. CARLA closed loop is future work (system validation).

## 2. Problems found and their status

| # | Problem | Status |
|---|---|---|
| 1 | HPO trials scored on the validation set that also selects the checkpoint | **fixed**: `hpo_split` (Optuna's own split from train, whole clips); `hpo/hpo_split.json` proves zero overlap |
| 2 | Warm-started models never scored their starting weights (OpenLane: lite_baseline_ct 0.386 vs 0.465 at start) | **fixed**: epoch-0 candidate |
| 3 | Multi-step decay at absolute epochs 25/38: 10-epoch HPO trials and short fine-tuning never decayed | **fixed**: milestones scale with each run's budget |
| 4 | Controls tuned separately from the model they control | **fixed**: `hpo.inherit` |
| 5 | CULane metric drew straight segments; the official tool interpolates lanes with splines | **fixed**: `evaluation.lane_interpolation: auto` (spline for CULane, internal and native metrics) |
| 6 | Fusion layers at lr 5e-4 under SGD 0.1 (pending item A1) | **fixed** in the CULane configs: `lr_fusion: 0.1`, tuned by HPO |
| 7 | No FP count for the CULane "cross" category | **fixed**: `condition_fp_iou50_<category>` |
| 8 | Robustness on 34,680 test frames x 4 protocols x 7 seeds would take days | **fixed**: `evaluation.robustness_max_frames` (seeded subset; clean F1 recomputed on the same frames) |
| 9 | Comparison pictures overlaid every model in one image | **added**: per-category grids (one panel per model, TP/FP/FN) and `model_comparison.mp4` |
| 10 | CULane keeps every 30th / 90th frame: history is 1 s or 3 s old | **decided**: `temporal_step: 30` (1 s); 90frame drivers fall back to the current frame; the temporal claim rests on ELAS |
| 11 | Auxiliary target drawn from the anchor points, not read from `laneseg_label_w16` | documented approximation: one cell at stride 8, like the official 16 px masks after resizing; same vertical extent as the anchors |
| 12 | Our CULane metric not yet cross-checked with the official tool | **open**: evaluate `culane_18.pth` (download it by browser from the UFLD repository) and/or the official C++ tool |
| 13 | Test images have no masks in the default download list | download `laneseg_label_w16_test.zip` too (ego-lane slots on test) |
| 14 | Nothing committed in git (pending item A11) | **open**: commit before the run (needs your OK) |
| 15 | OpenLane: small, narrow validation and a filtered test set | parked with the run in `results/openlane_full_7models_PARKED_*` |

## 3. Compute (measured 2026-10-23, RTX 3050 6 GB, 288x800, AMP; data loading adds ~15%)

| Model | img/s (train) | min/epoch | 50 epochs | 15 epochs |
|---|---|---|---|---|
| UFLD v1 (batch 32, 3.1 GB) | 69 | ~25 | ~21 h | - |
| UFLD v0.7 / v0.8 (3 frames, batch 16, 3.7 GB) | 31 | ~55 | ~46 h | ~14 h |
| lite baseline (batch 32, 1.9 GB) | 221 | ~9 | ~7.5 h | ~2.2 h |
| lite TSM / v0.6 / v0.6 occ (3 frames, batch 16, 2.2 GB) | 72 | ~25 | ~21 h | ~6 h |

| Plan | Where | Seed 1 | 7 seeds |
|---|---|---|---|
| core (6 models, fine-tune 15 epochs) | this PC | ~2.5 days (Optuna ~10 h + training ~49 h + evaluation ~2 h) | ~15-16 days |
| full (10 models, all 50 epochs) | cluster, >= 16 GB GPU | ~25-35 h on an A100 | 7 seeds in parallel: ~2 days after the search |
| reproduction (UFLD v1, article recipe) | either | ~21 h here, ~3 h on an A100 | 1 seed |

## 4. How to run

Everything goes through one file, `dist/tac_ufld_run.py` (rebuild it after a code change with
`python scripts/supervisor/build_runner.py`). Put it in any folder with `datasets/CULane/` next to it:

```
python tac_ufld_run.py setup
python tac_ufld_run.py smoke --synthetic     # every stage on generated data, minutes
python tac_ufld_run.py check                 # dataset completeness + label overlays
python tac_ufld_run.py smoke                 # every stage on a few real frames
python tac_ufld_run.py run --plan core       # or --plan full / --plan reproduction
python tac_ufld_run.py summary --plan core
python tac_ufld_run.py pack --plan core
```

On a SLURM cluster: `scripts/cluster/culane.slurm` (header), after `scripts/cluster/setup_env.sh`.
