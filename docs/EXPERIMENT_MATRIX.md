# Experiment matrix

Budgets below are for the development GPU (RTX 3050 6 GB, Windows) and come from the measured pilot (per epoch on the full ELAS training split: UFLD baseline ≈ 1.4 min, UFLD v0.2–v0.4 ≈ 2.5–4 min, lite baseline ≈ 0.6 min, lite v0.5 ≈ 2 min, plus ≈ 10 min of evaluation per seed). A larger GPU divides them roughly by its speed ratio.

| # | Experiment | Command | Varies | Fixed | Seeds | Estimated time (RTX 3050) | Status |
|---|---|---|---|---|---|---|---|
| 0 | Tests | `pytest -q` | — | — | — | ~10 min | done: see `docs/TESTING.md` |
| 1 | Plumbing smoke | `run --config configs/elas_smoke.yaml` | — | tiny images, 160 frames, 2 epochs, no ImageNet | 2 | ~9 min | done (before and after the changes) |
| 2 | **GPU pilot** | `run --config configs/elas_pilot.yaml` | — | full protocol, ≤ 4 epochs, no HPO | 2 | 2 h 22 min (measured) | **done**: `docs/PILOT_FINDINGS.md`, results page |
| 3 | **Full ELAS comparison (primary)** | `run --config configs/elas.yaml --confirm` | 6 variants | protocol of `docs/HANDOFF.md` | 6 | ≈ 1.5–2 days (HPO ≈ 15–20 h, training ≈ 20–24 h) | **to run by the supervisor** |
| 4 | Augmentation / regularisation ablation | `ablate --spec configs/ablations/augmentation.yaml --confirm` | photometric / none / geometric / flip / extended photometric / dropout + label smoothing | pilot budget, UFLD baseline | 3 | ≈ 6 arms × 3 seeds × 6 min ≈ 2 h | planned (motivated by the pilot's overfitting) |
| 5 | Preprocessing ablation | `ablate --spec configs/ablations/preprocessing.yaml --confirm` | rgb / gray3 / gray / edge / canny / hough / rgb_edge / CLAHE | pilot budget, UFLD baseline | 3 | ≈ 8 × 3 × 6 min ≈ 2.5 h | planned, exploratory |
| 6 | Temporal-context ablation | part of #3 (history replaced by the current frame at test time) | history on/off | same checkpoints | 6 | included | runs automatically |
| 7 | v0.4 warm-start protocol | `run --config configs/elas.yaml --confirm --name elas_v04_from_v02` with `model.v04_warm_start: v02` | warm start of v0.4 | — | 6 | ≈ +4 h | optional (supervisor's original protocol) |
| 8 | Lite history encoder | `model.lite_history_encoder: tiny` | shared vs tiny encoder for v0.5 | — | 6 | ≈ +6 h | optional |
| 9 | CULane | `validate-dataset --dataset culane`, then `run --dataset culane --confirm` | 4 UFLD variants | official lists, SGD recipe | 6 | days (88k frames) | needs the dataset |
| 10 | TuSimple | `validate-dataset --dataset tusimple`, then `run --dataset tusimple --confirm` | 4 UFLD variants | official split + carved val | 6 | ≈ 1 day | needs the dataset |
| 11 | OpenLane | `validate-dataset --dataset openlane`, then `run --dataset openlane --confirm` | 4 UFLD variants | official split + held-out segments | 6 | days | needs the dataset |
| 12 | Deployment check | `export --checkpoint <best> --int8 --eval-frames 240 --tensorrt fp32 fp16 int8` | backend / precision | trained checkpoint | — | ~10 min per model | done on the pilot checkpoints |
| 13 | Budget simulation | `benchmark --checkpoints ... --profile configs/deploy/<profile>.yaml` | profile, mode | trained checkpoints | — | ~2 min | done (desktop + two assumed Jetson profiles) |
| 14 | Jetson measurement | `scripts/jetson/measure_on_jetson.sh` | device | engines built on the device | — | ~15 min | needs a Jetson |

Reading rules for every row: the test split is evaluated once, after all selection; primary metric `lane_f1_iou50`; temporal models are compared only with their own family's baseline, on the same seeds; fewer than 6 seeds is flagged `underpowered`; dataset-native metrics are never compared across datasets.
