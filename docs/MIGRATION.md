# Migration and traceability

This package merges two single-file programs:

| Tag | Source | SHA-256 at merge time |
|---|---|---|
| **S** | Supervisor's reference notebook `Hyper Mega Fast Lane Detection_fullexp_singleexe.ipynb` (CULane; official UFLD; v0.2 / v0.3 / v0.4) | `ef20b1f6…c4338` |
| **U** | ELAS development script `tac_ufld_elas_pipeline_version_01_corrected.py` (kept unchanged in `reference/`) | `dcf701ab…1a452` |

The audit that motivated the merge is in [AUDIT_REPORT.md](AUDIT_REPORT.md). Its findings are referred to as C1–C8.

## Fixes (audit findings)

| # | Finding | Fix | Where | Test |
|---|---|---|---|---|
| 1 | **C1**: ELAS points p1..p4 were placed at ROI fractions 0, ⅓, ⅔, 1 | Rows are now at **0, ¼, ½, 1** (straight-line residual 1–4 px instead of 16–22 px). Single-point sides are "unknown" (ignored), all-NaN sides are "absent". Left/right slots come from the XML, not from sorting by x. | `data/elas.py` | `tests/test_elas.py` (incl. real-data regression) |
| 1b | Rows outside a scene's ROI were trained as "no lane" | They get `IGNORE_INDEX` (as the official UFLD `ignore_lb`), and evaluation is clipped to the ROI | `data/targets.py`, `postprocess.py` | `tests/test_targets.py` |
| 2 | **C2**: the anchor metric dropped wrong-position anchors | A wrong position counts as FP + FN (S's rule) | `metrics.py::anchor_counts` | `tests/test_metrics.py` |
| 3 | **C3/C4**: the baseline was not UFLD; v03/v04 had lost their designs; temporal models had a 4× smaller backbone trained from scratch at 0.1× LR | The official UFLD is re-implemented (ResNet-18, ImageNet weights, torchvision-compatible). S's v02/v03/v04 are restored with their losses. U's model becomes `lite_v05`, paired with `lite_baseline`, which has the same backbone. Every temporal variant is warm-started from its own family's baseline. | `models/` | `tests/test_models.py`, `tests/test_losses.py` |
| 4 | **C5**: split drawn per seed; test frames 9 frames from training frames; no unseen scenes | Scene-held-out test split, a fixed `split_seed` independent of training seeds, and a purge gap ≥ temporal context span. A leakage checker raises on violations. | `data/splits.py` | `tests/test_splits.py` |
| 5 | **C6/C7**: HPO unseeded and persisted; 4 seeds cannot reach significance; only the variants were tuned | Seeded TPE with fresh in-memory studies; the baselines get the same budget; 6 seeds by default; exact paired Wilcoxon + Holm; the `underpowered` flag makes low seed counts explicit | `training/hpo.py`, `evaluation/stats.py` | `tests/test_config_and_stats.py` |
| 6 | S: no test split; the baseline threshold was fixed while variants were tuned | Held-out test; post-processing tuned on validation for **every** model, plus a common fixed protocol | `experiment.py` | integration test |
| 7 | **C8**: non-standard primary metric (IoU 0.35, pixel F1) | Primary = CULane-style lane F1 at IoU 0.5 (line width scaled 30 px @1640 → 12 px @640), Hungarian matching; IoU 0.35 and pixel F1 remain secondary | `metrics.py`, `evaluation/evaluator.py` | `tests/test_metrics.py` |
| 8 | Several selection criteria (anchor F1 / pixel F1 / IoU F1) | One declared `evaluation.selection_metric` for checkpoints, early stopping, Optuna pruning and the post-processing sweep | `training/trainer.py` | — |
| 9 | `cv2.imread` fails on non-ASCII Windows paths; dataset root was relative to the CWD | PIL / `imdecode` I/O; paths relative to the project root, with an env-var override | `data/transforms.py`, `visualization/overlays.py`, `config.py` | integration test |

## Function map

### From S (supervisor)

| S | New location | Status |
|---|---|---|
| `parsingNet` (official repo, imported) | `models/resnet.py`, `models/ufld.py::UFLDNet` | re-implemented (identical architecture, flatten size derived from the input size) |
| `OfficialUFLDAdapter`, `adapt_official_ufld_output` | `models/ufld.py::UFLDSingleFrame`, `decoding.py` | ported; location = official soft-argmax instead of argmax |
| `TemporalWeightedFusion`, `TACUFLDTemporalModel` | `models/ufld.py` (v02, v04) | ported |
| `GatedTemporalFusion`, `TACUFLDGatedTemporalModel` | `models/ufld.py` (v03) | ported; bias generalised to T ≠ 3 |
| `official_ufld_classification_loss` | `losses.py::softmax_focal_loss` | ported + ignore index; equality test against S's formula |
| `weighted_bce`, `coord_loss`, `temporal_consistency_loss`, `main_loss(mode)` | `losses.py`, `models/registry.py` (per-variant constants) | ported |
| `train_model` | `training/trainer.py` | merged with U's loop |
| `train_and_eval_v0x_trial`, `run_optuna_for_variant`, `VARIANT_REGISTRY` | `training/hpo.py`, `models/registry.py` | generalised, config-driven search space |
| `load_baseline_components` | `models/registry.py::warm_start` | generalised, strict |
| `lane_to_mask`, `compute_lane_iou`, `match_lanes_by_iou` | `metrics.py` | greedy → Hungarian (as in the official CULane evaluator) |
| `refine_lane_polyfit`, `suppress_duplicate_lanes`, `postprocess_item` | `postprocess.py` | ported; `poly_degree: 0` = official raw points |
| `save_lanes_txt`, `export_cache_predictions` | `reporting.py::export_lines_txt` | ported |
| `load_culane_subset`, `read_lines_txt`, `build_culane_file_index` | `data/culane.py` | ported; fixed lane slots; official test list; **not validated on real CULane yet** |
| `save_lane_detection_confusion_matrix` | `visualization/plots.py::plot_error_counts` | replaced (TP/FP/FN bars; TN is undefined for lanes) |

### From U (ELAS script)

| U | New location | Status |
|---|---|---|
| `discover_scene_paths`, `parse_elas_config`, `parse_elas_groundtruth`, `resolve_image_path` | `data/elas.py` | ported, **C1 fixed** |
| `prepare_scene_splits`, `_allocate_quota` | `data/splits.py` | rewritten (scene hold-out, seed-independent, leakage check); weighted quotas dropped |
| `lanes_to_targets` | `data/targets.py` | ported, with the ignore index and fixed slots |
| `ELASTemporalDataset`, `_augment_sequence` | `data/dataset.py`, `data/transforms.py` | ported |
| `LightweightBackbone`, `ConvBlock`, `DepthwiseSeparableConvBlock`, `TinyHistoryEncoder`, `LanePixelHead` | `models/lite.py` | ported; head outputs official layout; `y_head` dropped (weight was 0) |
| `LearnableFlowWarp`, `ResidualTemporalFusion`, `TACUFLDTemporalModel` | `models/lite.py::LiteWarpTemporal` (`lite_v05`) | ported; full-width shared backbone (fair pairing) |
| `LightweightBackboneTemporal` (half width) | — | dropped (confounded the comparison) |
| `main_loss` 6-term mix | — | replaced by the UFLD focal loss + variant terms (no ablation supported the mix) |
| `compute_anchor_metrics` | `metrics.py::anchor_counts` | **C2 fixed** |
| `compute_pixel_f1`, `evaluate_iou_cache`, per-condition F1, `compute_temporal_jitter` | `metrics.py`, `evaluation/evaluator.py` | ported |
| `sweep_postprocessing` | `evaluation/evaluator.py::Evaluator.sweep` | ported; selects on the declared metric |
| cosine LR, gradient clipping, early stopping, `diagnose_epoch` | `training/trainer.py` | ported |
| `compute_model_efficiency` and helpers | `evaluation/efficiency.py` | ported |
| `aggregate_multi_seed_metrics`, `paired_wilcoxon_vs_baseline` | `evaluation/stats.py` | ported + Holm, exact test, power flag |
| scenario / FP-FN / temporal-video images | `visualization/overlays.py` | ported (Unicode-safe) |
| TensorBoard per-epoch scalars | `training/trainer.py` | ported (graph/image logging dropped) |
| torchview diagrams, Graphviz search, pip auto-install on import, disk-space guard | — | dropped (not needed for correctness; install dependencies explicitly) |
| ADAS TRL3/TRL4 thresholds | — | dropped: no cited source (audit C8). Add them back as documented internal targets when a source exists |

## v0.3 changes (2026-09-29)

Audit of the starting point: [AUDIT_V03.md](AUDIT_V03.md). Nothing in the model definitions, losses, metrics, split protocol or selection rules changed; defaults reproduce v0.2 exactly (tested for the input pipeline and the photometric augmentation).

| Area | Change | Where | Test |
|---|---|---|---|
| Resume (bug) | a partially trained variant was treated as finished; now completion = history CSV; epoch-level `<variant>.last.pt` (optimizer, scheduler, scaler, RNG, loader state); finished HPO studies reused | `training/trainer.py`, `experiment.py` | `test_registry_cli.py`, integration |
| Full-run guard | `requires_confirmation` + `--confirm`; nothing created before the check | `cli.py`, `experiment.py`, configs | `test_full_run_requires_confirmation` |
| CULane (bug) | `temporal_step` 30 → 90 (supervisor's value); official segmentation-label slots; existence-flag check; stride report | `data/culane.py`, `configs/culane.yaml` | `test_datasets.py` |
| Datasets | registry `configs/datasets.yaml`; TuSimple, OpenLane adapters; validation carving; native metrics; `validate-dataset` | `data/registry.py`, `data/tusimple.py`, `data/openlane.py`, `data/splits.py`, `evaluation/native.py`, `data/validation.py` | `test_datasets.py`, `test_registry_cli.py` |
| Input | preprocessing ablations; `conv1` adaptation for 1/4 channels | `data/preprocess.py`, `models/resnet.py` | `test_preprocess_augment.py` |
| Augmentation | geometric (homography + label re-encoding), extended photometric | `data/geometric.py`, `data/transforms.py` | `test_preprocess_augment.py` |
| Training options | label smoothing, UFLD head dropout, lite dropout config | `losses.py`, `models/` | `test_tools.py` (sanity), existing loss tests |
| Checkpoints | self-describing: resolved config + model card; tuned post-processing JSON | `experiment.py`, `inference/card.py` | `test_streaming.py` |
| Inference | streaming with history-feature caching; backends | `inference/` | `test_streaming.py` |
| Deployment | ONNX (streaming split), ONNX Runtime, TensorRT, FP16/INT8, numerical + task checks; export bugs fixed (wrapper train mode, BatchNorm update) | `deploy/` | `test_deploy.py` |
| Benchmark | measured latency + budget simulation | `evaluation/hardware.py` | `test_tools.py` |
| UI, site, packaging, doctor, Docker | new | `ui/`, `site.py`, `package.py`, `doctor.py`, `Dockerfile`, `docker-compose.yml` | `test_ui.py`, `test_site.py`, `test_tools.py`, `test_docker.py` |
| Tooling | `sanity`, `ablate`, `export`, `stream`, `benchmark`, `ui`, `site`, `package`, `doctor`, `carla-demo` commands | `commands.py` | `test_tools.py` |

## v0.4 changes (2026-09-30)

Everything is opt-in; v0.3 configurations train and evaluate exactly as before (tested). Details:
`docs/TEMPORAL_IMPROVEMENTS.md`.

| Area | Change | Where | Test |
|---|---|---|---|
| Overfitting | `train.lr_backbone_mult` (separate optimizer group only when not 1; HPO parameter), `train.freeze_backbone_stages` (frozen BatchNorm statistics), `model.init_checkpoint` / `init_scope` (official UFLD or own checkpoints) | `training/trainer.py`, `models/registry.py`, `experiment.py`, `config.py` | `test_temporal_v04.py` |
| Augmentation | current-frame degradation (occlude / blur / darken / noise on the last frame only); draws no random numbers when off | `data/transforms.py`, `config.py` | `test_temporal_v04.py` |
| Models | `ufld_v06` (aligned fusion), `ufld_v07` / `lite_v06` (ConvGRU, `recurrent_step`), controls `ufld_baseline_ct` / `lite_baseline_ct`; `VariantSpec.fusion`, `budget_reference`; `LiteWarpTemporal` renamed `LiteTemporal` (alias kept, state-dict keys unchanged) | `models/fusion.py`, `models/ufld.py`, `models/lite.py`, `models/registry.py` | `test_temporal_v04.py`, `test_streaming.py` (all temporal variants) |
| Evaluation | Kalman reference (`input = kalman`), carried-state evaluation (`input = carry`), per-condition table, equal-training pairs | `evaluation/tracking.py`, `evaluation/recurrent_eval.py`, `experiment.py` | `test_temporal_v04.py`, `test_integration.py` |
| Streaming | `mode="carry"` for recurrent models, optional per-stream Kalman tracker (`stream --mode carry --kalman`) | `inference/streaming.py`, `inference/backends.py`, `commands.py` | `test_temporal_v04.py` |
| Ablations | `common`, `allow`, `grid`, `reuse_single_frame`; reference arm runs first; new arms in `augmentation.yaml`; `history.yaml` | `ablation.py`, `configs/ablations/` | `test_temporal_v04.py`, `test_tools.py` |
| Logging (bug) | a process running several experiments kept writing into every earlier run's `run.log` | `utils.py` | `test_temporal_v04.py` |
| Results page | light blue / white theme, model explanations, ablation sections, roadmap; `site --notes --roadmap` | `site.py`, `models/descriptions.py`, `docs/ROADMAP.md` | `test_site.py` |
