# Experiment matrix

Budgets below are for the development GPU (RTX 3050 6 GB, Windows) and come from the measured pilot (per epoch on the full ELAS training split: UFLD baseline ≈ 1.4 min, UFLD v0.2–v0.4 ≈ 2.5–4 min, lite baseline ≈ 0.6 min, lite v0.5 ≈ 2 min, plus ≈ 10 min of evaluation per seed). A larger GPU divides them roughly by its speed ratio.

| # | Experiment | Command | Varies | Fixed | Seeds | Estimated time (RTX 3050) | Status |
|---|---|---|---|---|---|---|---|
| 0 | Tests | `pytest -q` | — | — | — | ~10 min | done: see `docs/TESTING.md` |
| 1 | Plumbing smoke | `run --config configs/elas_smoke.yaml` | — | tiny images, 160 frames, 2 epochs, no ImageNet | 2 | ~9 min | done (before and after the changes) |
| 2 | **GPU pilot** | `run --config configs/elas_pilot.yaml` | — | full protocol, ≤ 4 epochs, no HPO | 2 | 2 h 22 min (measured) | **done**: `docs/PILOT_FINDINGS.md`, results page |
| 2b | **Second GPU pilot (v0.4 recipe)** | `run --config configs/elas_pilot_v2.yaml` | 11 models and controls | geometric augmentation + current-frame degradation, ≤ 4 epochs, no HPO | 2 | 4 h 47 min (measured) | **done**: `docs/PILOT_V2_FINDINGS.md`, results page |
| 2c | Robustness to a degraded current frame | `robustness --runs results/elas_pilot_v2` (and `results/elas_lite_long`) | occlusion / blur / darkening / noise | trained checkpoints, history clean | 2 | 66 min for 22 checkpoints, 23 min for 10 lite checkpoints (measured) | **done** for both runs |
| 2d | Longer lite run + capacity control | `run --config configs/elas_lite_long.yaml` | 5 lite models | ≤ 16 epochs, patience 4 | 2 | 6 h 12 min (measured) | **done** |
| 3 | **Full ELAS comparison (primary)** | `run --config configs/elas.yaml --confirm` | 12 models and controls (or the 9-model core subset) | protocol of `docs/HANDOFF.md`, v0.4 recipe | 6 | ≈ 4–5 days (core subset ≈ 25 % less) | **to run by the supervisor** |
| 4 | Augmentation / regularisation ablation | `ablate --spec configs/ablations/augmentation.yaml --confirm` | photometric / none / geometric / flip / extended photometric / dropout + label smoothing | pilot budget, UFLD baseline | 3 | 1 h 56 min (measured) | **done**: geometric augmentation chosen on validation |
| 4b | Overfitting remedies (phase 2 of #4) | same spec, `--only backbone_lr_0p1 freeze_stem_layer1 current_frame_degradation small_head more_scenes` | backbone lr ×0.1 / frozen stem + layer1 / current-frame degradation / 256-unit head / 18 scenes | geometric recipe, UFLD baseline | 3 | 2 h 04 min (measured) | **done**: none fixes the overfitting alone |
| 4c | History length / step ablation | `ablate --spec configs/ablations/history.yaml --confirm` | step × frames grid (steps 1–15, 2–5 frames) | UFLD baseline + v0.4 (chosen on validation), 180-frame blocks | 2 | 2 h 56 min for 4 arms (measured) | **done** |
| 5 | Preprocessing ablation | `ablate --spec configs/ablations/preprocessing.yaml --confirm` | rgb / gray3 / gray / edge / canny / hough / rgb_edge / CLAHE | pilot budget, UFLD baseline | 3 | ≈ 8 × 3 × 6 min ≈ 2.5 h | planned, exploratory |
| 6 | Temporal-context ablation | part of #3 (history replaced by the current frame at test time) | history on/off | same checkpoints | 6 | included | runs automatically |
| 7 | v0.4 warm-start protocol | `run --config configs/elas.yaml --confirm --name elas_v04_from_v02` with `model.v04_warm_start: v02` | warm start of v0.4 | — | 6 | ≈ +4 h | optional (supervisor's original protocol) |
| 8 | Lite history encoder | `model.lite_history_encoder: tiny` | shared vs tiny encoder for v0.5 | — | 6 | ≈ +6 h | optional |
| 9 | CULane | `scripts/check_dataset.py`, `validate-dataset --dataset culane`, `scripts/smoke_datasets.py --datasets culane`, then `run --dataset culane --confirm` | 4 UFLD variants (+ v0.4 models if budget allows) | official lists, SGD recipe, official UFLD augmentation | 6 | days (88,880 training frames) | needs the dataset; smoke-tested on a synthetic copy |
| 10 | TuSimple | same steps with `tusimple` | 4 UFLD variants | official split + carved val, official UFLD augmentation | 6 | ≈ 1 day | needs the dataset; smoke-tested on a synthetic copy |
| 11 | OpenLane | same steps with `openlane` | 4 UFLD variants | official split + held-out segments, ELAS recipe | 6 | days | needs the dataset; smoke-tested on a synthetic copy |
| 11b | Every-dataset smoke test | `python scripts/smoke_datasets.py` | dataset | 1 epoch, ≤ 32 training frames, 6 models, CPU | 1 | 6–17 min per dataset on a busy CPU (measured) | **done**: ELAS real + synthetic CULane / TuSimple / OpenLane |
| 12 | Deployment check | `export --checkpoint <best> --int8 --eval-frames 240 --tensorrt fp32 fp16 int8` | backend / precision | trained checkpoint | — | ~10 min per model | done on the pilot checkpoints |
| 13 | Budget simulation | `benchmark --checkpoints ... --profile configs/deploy/<profile>.yaml` | profile, mode | trained checkpoints | — | ~2 min | done (desktop + two assumed Jetson profiles) |
| 14 | Jetson measurement | `scripts/jetson/measure_on_jetson.sh` | device | engines built on the device | — | ~15 min | needs a Jetson |

Reading rules for every row: the test split is evaluated once, after all selection; primary metric `lane_f1_iou50`; temporal models are compared only with their own family's baseline, on the same seeds; fewer than 6 seeds is flagged `underpowered`; dataset-native metrics are never compared across datasets.
