# Code Audit — Supervisor reference vs. current ELAS pipeline

**Date:** 2026-09-29
**Files audited (neither was modified):**

| Role | File | SHA-256 (at audit time) |
|---|---|---|
| Supervisor reference (S) | `Hyper Mega Fast Lane Detection_fullexp_singleexe.ipynb` (1 code cell, ~3.1k lines) | `ef20b1f6…c4338` |
| Current development (U) | `tac_ufld_elas_pipeline_version_01_corrected.py` (4,972 lines) | `dcf701ab…1a452` |

**Method:** I read both files in full. I then ran read-only tests against the local ELAS copy (`../datasets/dataset_elas_v1`, 22 scenes) and against your module (imported, not executed). The test scripts are in the session scratchpad and can be added to `tests/` later.

**Evidence labels:** **[VERIFIED]** = shown by a test or an unambiguous code path. **[STATIC]** = from reading the code only. **[UNCERTAIN]** = needs an experiment to settle.

---

## 1. Executive summary

1. **The two files are not two versions of the same experiment.** S is a **CULane** pipeline built on the **official UFLD `parsingNet` (ResNet‑18)**. U is an **ELAS** pipeline built on a **custom lightweight CNN** with a custom head. They also differ in dataset, resolution, anchors, grid, number of lanes, IoU threshold and model family. Their numbers cannot be compared with each other.
2. **All results currently in `results/v1_elas/` are invalid** because of a ground-truth geometry bug in U (§4, C1) **[VERIFIED]**. The results must be regenerated after the fix.
3. **U has a verified metric bug.** `compute_anchor_metrics` ignores anchors that were predicted at the wrong position, so anchor-level F1 is inflated. In a synthetic test where every anchor was predicted 40 bins off, it counted zero FP and zero FN (§4, C2) **[VERIFIED]**.
4. **U's baseline is not UFLD, and U's v02/v03/v04 no longer implement S's v02/v03/v04** (§3, §4 C3–C4). Because the temporal models also use a 4× smaller backbone trained from scratch at 0.1× learning rate, the current baseline-vs-temporal comparison cannot isolate the effect of temporal information **[VERIFIED]**.
5. **U has real improvements over S that should be kept:**
   * a held-out test set that is used only after all selection;
   * validation-tuned post-processing for *every* model, plus a common fixed protocol;
   * augmentation, cosine LR, early stopping and gradient clipping;
   * efficiency metrics, overlays, FP/FN galleries and jitter.
6. **S has methodological weaknesses of its own:** no test split (validation is used for checkpointing, HPO, threshold selection *and* the final report), an unfair fixed threshold for the baseline, and `pretrained=False`.

**Recommendation: a careful merge.** Keep **U** as the foundation for the ELAS data handling, splits, evaluation and reporting, after fixing C1, C2 and C5. Restore **S**'s model definitions as the reference implementations: the official UFLD baseline and S's v02/v03/v04 with their variant-specific losses and warm-starts. Keep U's warped/residual temporal design as an **additional, separately named variant** instead of letting it overwrite v02–v04. The plan is in §7.

---

## 2. What each file implements

| Item | Supervisor (S) | Current (U) |
|---|---|---|
| Dataset | CULane, official `train_gt.txt` / `val_gt.txt` lists | ELAS, 10 scenes from `SCENE_PLAN`, custom XML parser |
| Splits | train (random subset) + val. **No test** | train/val/test from 60-frame temporal blocks, 4-frame purge, **re-drawn per seed** |
| Input | 800×288, [0,1], no normalization | 512×384 (aspect-preserving), [0,1], no normalization |
| Anchors / grid / lanes | 18 official CULane anchors / 200 / 4 | 48 linspace anchors / 64 / 2 |
| Baseline | `OfficialUFLDAdapter` → official `parsingNet(backbone="18", pretrained=False)`, focal CE over G+1 classes (`official_ufld_classification_loss`) | `SingleFrameUFLDLikeModel` = 4×ConvBlock CNN (0.46 M backbone params) + custom per-(lane, anchor) MLP `LanePixelHead`, 6-term loss |
| v02 | Shared ResNet‑18 per frame + `TemporalWeightedFusion` (3 learnable softmax weights, init 0.1/0.2/0.7), warm-started from baseline, + weighted exist-BCE | `TACUFLDTemporalModel`: half-width current-frame CNN (0.12 M) + `TinyHistoryEncoder` + `LearnableFlowWarp` + `ResidualTemporalFusion`; only the head is partially warm-started |
| v03 | `GatedTemporalFusion` (per-pixel softmax gates over frames, conv gate net) + gate-prior loss | **Same class as v02**; only the gate bias init differs (−0.5 vs −2.0) |
| v04 | v02 architecture + **soft-argmax coordinate loss** + different BCE weights, warm-started from v02 | **Same architecture and same loss as v02**, warm-started from v02 → effectively "v02 trained longer" |
| Temporal context | 3 frames, step 90 (CULane file ids), silent fallback to the current frame | 3 frames, step 2, fallback **with a warning** |
| Optimizer | AdamW, constant LR, no clipping | AdamW, cosine LR, gradient clipping, early stopping (patience 8) |
| HPO | Optuna TPE **seeded**, in memory, 20 trials × 10 epochs, objective = val anchor F1 | Optuna TPE **unseeded**, SQLite with `load_if_exists=True`, 6 × 3, objective = val pixel F1 |
| Metrics | anchor F1 (correct), lane IoU F1 (IoU 0.5, width 30 px @1640) | anchor F1 (**buggy**), lane IoU F1 (IoU **0.35**, width 12 px @640), "polyline pixel F1", F2, per-condition F1, jitter |
| Reporting | CSVs, bar plots, confusion matrix, `.lines.txt` export | Extensive: overlays, FP/FN galleries, videos, TensorBoard, ablations, efficiency, 4 seeds + Wilcoxon |
| Saved results | Notebook output is a **2-image smoke test** (all F1 = 0) — no usable numbers | seeds 1 and 42 complete, seed 7 incomplete; all invalid (C1) |

---

## 3. Feature-by-feature comparison

Classification is always from **U's point of view relative to S**, or relative to the stated goal ("UFLD baseline following the paper").

| # | Feature | Evidence | Class | Note |
|---|---|---|---|---|
| 1 | Dataset adapter | S `load_culane_subset`; U `parse_elas_groundtruth` L509 | uncertain → **needs fix** | U-only code. X values and p1(far)→p4(near) order are correct, but the **row spacing is wrong (C1)** |
| 2 | Lane-slot assignment | S & U `lanes_to_targets` (sort by mean x) | equivalent (both weak) | A frame with only the right lane annotated fills slot 0 ("left"). Affects 28/16,992 ELAS frames (0.16 %) **[VERIFIED]** — low impact |
| 3 | "No lane" class | U L943 `cls_target = grid_size` | **improvement** | Matches UFLD's G+1 formulation |
| 4 | Baseline architecture | S `OfficialUFLDAdapter`; U L1143, L1351 | **regression** vs goal | U's baseline is a small custom CNN, not UFLD. The comment "faithful UFLD" (L1224) is inaccurate |
| 5 | Baseline head | U `LanePixelHead` L1223 | changed | G+1 softmax, existence = 1 − p(bg), soft-argmax location: conceptually UFLD-like. The per-anchor shared MLP differs from UFLD's global FC |
| 6 | Baseline loss | S focal CE only; U `main_loss` L1735 | changed / uncertain | U sums 6 terms (masked CE + focal exist on derived logits + x-regression ×35 + full-grid CE + structural + y). The official CULane config sets the structural/shape weights to 0 [to confirm against the repo]. There is no ablation showing U's mix helps |
| 7 | v02 design | see §2 | **changed** (new model) | The temporal backbone is 4× smaller and trained from scratch at `lr*0.1` (L3045, L4596) → confounded vs baseline |
| 8 | v03 design | U L1567 | **missing** | S's gated per-pixel softmax fusion and its gate-prior loss are gone |
| 9 | v04 design | U L4629–4651 | **missing** | S's coordinate loss is gone; U's v04 is v02 fine-tuned again. The "deep" label (L441) is inaccurate |
| 10 | Per-variant losses | S `main_loss(mode=…)`; U same loss for all modes | **regression** | The defining differences between S's variants were removed |
| 11 | Temporal consistency loss | S L1042; U L1878 | equivalent | Both compare per-frame outputs of t and t−1 |
| 12 | Missing-frame handling | U `get_temporal_paths` L989 | **improvement** | Warns instead of silently duplicating frames |
| 13 | Augmentation | U `_augment_sequence` L1019 | improvement (untested) | Photometric + erasing + frame dropout; no geometric augmentation (official UFLD uses rotation/shift) |
| 14 | Optimizer / schedule | U L2413–2422, L2480–2491 | improvement | Cosine LR, gradient clipping; best-state restore L2640 |
| 15 | Early stopping in HPO | U L2436 (`patience = 1`) | uncertain | With 3 HPO epochs, one flat epoch ends the trial |
| 16 | HPO pruning | U L2608 | **regression** | Prunes if `anchor_f1 < 0.35`, which uses the buggy anchor metric (C2) |
| 17 | HPO reproducibility | S `TPESampler(seed=…)`; U L3087–3093 | **regression** | U's sampler is unseeded; the persisted SQLite study with `load_if_exists=True` would mix trials from older code if rerun with the same `STUDY_VERSION_TAG` |
| 18 | HPO budget fairness | both | equivalent (both weak) | Neither tunes the baseline, while the variants get Optuna budgets |
| 19 | Eval threshold fairness | S `model_registry` baseline 0.50 fixed; U `sweep_postprocessing` for all models + common-fixed protocol | **improvement** | |
| 20 | Anchor metric | S `compute_anchor_metrics` (wrong position → FP+FN); U L1936 | **regression** | **[VERIFIED]** see C2 |
| 21 | Lane IoU F1 | S IoU 0.5, width 30 @1640 px; U L3124–3125 | uncertain | U's width is correctly scaled (30×640/1640 ≈ 11.7 → 12), but the **IoU threshold of 0.35 is non-standard and more lenient**. Report 0.5 as primary |
| 22 | Pixel F1 | U `compute_pixel_f1` L3386 | addition | Arithmetic is correct (micro-averaged union masks of 12-px polylines). It is **custom** and cannot be compared with published numbers |
| 23 | Held-out test | U `run_pipeline` L4693+ | **improvement** | The test set is touched only after HPO, checkpointing and the post-processing sweep **[STATIC, traced]** |
| 24 | Split leakage protection | U `prepare_scene_splits` L599 | uncertain / weak | See C5 |
| 25 | Multi-seed statistics | U L4827–4919 | improvement in intent | With 4 seeds the two-sided Wilcoxon minimum p is 0.125, so it can never reach significance. Seeds also change the split |
| 26 | Selection-metric consistency | U L2428 (checkpoint = pixel F1), L3584 (post-process sweep = IoU F1) | uncertain | Two different selection criteria in one pipeline |
| 27 | Efficiency | U L3668–3761 | addition | Measured on the host device, not an embedded target. History frames are re-encoded every call (a streaming implementation could cache them) |
| 28 | Visualization / reporting | U L3886–4237 | addition | Useful. `cv2.imread` **fails on non-ASCII absolute paths** such as `Residência` **[VERIFIED]**; it works today only because `DATASET_ROOT` is CWD-relative (L363) |
| 29 | Determinism | U L88 `cudnn.benchmark = True` | minor regression | Non-deterministic kernels |
| 30 | Pretraining / normalization | S `pretrained=False`; U from scratch | equivalent (both deviate) | The UFLD paper uses an ImageNet-pretrained ResNet with ImageNet normalization |

---

## 4. Issues that can invalidate results or prevent reproducibility (ranked)

### C1 — ELAS ground-truth rows are placed at the wrong heights (critical, U only) **[VERIFIED]**
`elas_point_rows_y` (L498–506) places p1..p4 at `linspace(roi_y, roi_y+roi_h, 4)`, i.e. at 0, ⅓, ⅔ and 1 of the ROI height.

**Test:** for every annotated lane side in all 22 ELAS scenes (≈31k sides), fit a straight line x = a·y + b under two row hypotheses and measure the maximum deviation. On straight roads a correct hypothesis gives near-zero deviation.

| Hypothesis | Median max-residual per scene | Median gap ratio (p3→p4)/(p2→p3) |
|---|---|---|
| Current code: 0, ⅓, ⅔, 1 | **15.8–22.5 px** | predicts 1.0 |
| 0, ¼, ½, 1 of ROI height | **1.0–4.6 px** | **1.76–2.02 observed** |

A visual overlay on BR_S01 frame 0 shows the (0, ¼, ½, 1) polyline lying exactly on the painted markings, while the current polyline bends inward by up to ~35 px at p3.

**Impact:**
* Every training target (`lanes_to_targets`) and every evaluation GT polyline is distorted, so all models learned a kinked lane shape.
* All pixel and IoU metrics were scored against the wrong geometry.
* The existing `gt_overlay_check.png` looked plausible only because it draws points, not the polyline.

**Fix:** use fractions `[0, 0.25, 0.5, 1.0]`, add a regression test (median collinearity residual < 5 px), and optionally confirm against the ELAS source code.

### C2 — Anchor metric drops wrong-position predictions (high, U) **[VERIFIED]**
In `compute_anchor_metrics` (L1936–1956), anchors where both prediction and GT are positive but the bin error is > 1 are counted as neither FP nor FN.

**Synthetic test:**
* All anchors predicted 40 bins off → tp = fp = fn = 0.
* Half of the anchors at the wrong position → P = R = F1 = **1.00** (correct value: 0.50).

**Impact:** `anchor_f1` in every history CSV and in TensorBoard is inflated, and HPO pruning (L2608) relies on it.

**Fix:** restore S's rule (`wrong_position` → +1 FP and +1 FN).

### C3 — The baseline-vs-temporal comparison is confounded (high, U) **[VERIFIED]**
* Baseline: 1.82 M params, backbone 0.46 M, 2.06 GMACs.
* v02: 1.51 M params, backbone **0.12 M**, 0.80 GMACs.
* The temporal backbones start from scratch with `lr_backbone = lr*0.1` (L3045, L4596), while the baseline trains its backbone at the full LR.

Any accuracy difference therefore mixes capacity, initialization and LR with "temporal information". The existing (already invalid) results are consistent with this: the baseline beats all temporal variants in seed 1.

### C4 — v03 and v04 do not implement their intended designs (high, U) **[STATIC, unambiguous]**
* v03 is v02 with a different gate-bias initialization.
* v04 is v02 re-trained from v02's weights with the same loss, so a v04 "gain" would come only from extra epochs.
* S's gated fusion (v03) and coordinate loss (v04) are missing.

### C5 — Split methodology limits what the test set can show (high for generalization claims, U) **[VERIFIED]**
* **Every test frame comes from a scene that is also in training.** The nearest training frame is only 9 frames away at minimum (median 55).
* This measures within-scene interpolation, not generalization to new roads.
* **The split is re-drawn per seed** (`prepare_scene_splits(..., seed=seed)`, L4467): 195 of seed 1's 500 test frames were seed 42's *training* frames. Across-seed variance therefore mixes split and initialization. `iou_f1_rainy` goes from 0.00 (seed 42) to 0.65 (seed 1).

**Fix:**
* Fix the split independently of the training seed.
* Add a **scene-held-out** test split (at least one scene per condition tag).
* Keep the temporal-block split only as a secondary "seen-scene" test.

### C6 — HPO is not reproducible (medium, U) **[STATIC]**
The Optuna sampler is unseeded and the SQLite storage uses `load_if_exists=True`. Trial counts are currently clean (6 per study, checked), so this is latent, but any rerun without bumping `STUDY_VERSION_TAG` would merge trials from different code versions.

### C7 — Statistics cannot support significance claims yet (medium, U) **[VERIFIED by arithmetic]**
With n = 4 seeds, the smallest possible two-sided Wilcoxon p-value is 2/2⁴ = 0.125. You need at least 6 paired seeds (the minimum p is then 0.031), preferably with a fixed split.

### C8 — Non-standard metrics and invented targets (medium, U) **[STATIC]**
* IoU 0.35, "polyline pixel F1" and F2 are custom metrics.
* The TRL3/TRL4 thresholds (L853–858) have no cited source.
* Report CULane-style IoU 0.5 as the primary lane metric. Label the rest as internal, and do not present the TRL table as a standard.

### S-side issues (relevant if S code is restored)
* **No test split.** Validation drives checkpointing, Optuna, threshold choice and the final report, which is optimistic.
* The baseline is evaluated at a fixed threshold of 0.50 while the variants use Optuna-tuned thresholds (`load_evaluation_threshold`).
* v04 warm-starts from v02 and trains 50 more epochs, so it is confounded by training length (same as U).
* `pretrained=False`, no normalization and no augmentation mean "Official UFLD" refers to the architecture only, not the paper's training recipe.
* Temporal fallback silently duplicates the current frame.
* Its only saved output is a 2-image smoke test.

### Minor (both or U)
* `flow_smoothness_loss` reads the global `cfg` (L1889).
* `DATASET_ROOT` depends on the current working directory.
* The comments are long and sometimes inaccurate (e.g. L1125 "reduced from 10" for a value of 12).
* The 5k-line single file makes unit testing hard.
* The single-side slot bug (C-table #2) is low impact.

---

## 5. Experimental consistency: what differs by setup, not by model

The following differ between S and U independently of any model change: dataset, image size, anchor count and placement, grid size, lane count, GT format, IoU threshold, line width, split protocol, threshold selection, optimizer schedule, epochs/patience, HPO space and budget, and seed handling. **No claim of the form "U's model is better/worse than S's" is possible from the existing outputs.**

Within U, the four models share the data, split, metric and post-processing protocol. However, they differ in backbone capacity, initialization and LR (C3), and the GT is wrong for all of them (C1).

---

## 6. What you can trust today

**Trust (verified or directly traced):**
* ELAS scene discovery, the XML field parsing of x values, the p1 = far / p4 = near ordering, and the per-scene filename caching.
* The target encoding (bins, background class = `grid_size`) — once C1 is fixed.
* Lane IoU matching (greedy one-to-one, identical logic to S) and the pixel-F1 arithmetic.
* Test isolation: the test set is used only after all selection steps.
* Early stopping with best-state restore, cosine LR, gradient clipping.
* Efficiency counting and the visualization/reporting utilities (apart from `cv2.imread` with non-ASCII paths).

**Needs verification before use:**
* Augmentation benefit.
* U's 6-term loss vs plain UFLD focal CE.
* The 0.35 IoU threshold.
* The post-processing sweep's choice of IoU F1 as the selection metric.
* Jitter (few consecutive pairs; it depends on the split).

**Do not trust:**
* Anything in `results/v1_elas/` (C1).
* `anchor_f1` (C2).
* The labels "v03 gated" and "v04 deep" (C4).
* Multi-seed significance claims (C7).

---

## 7. Recommended minimal, traceable merge plan

Each step is small, testable and committed separately. Nothing is rewritten wholesale.

0. **Version control first.** Run `git init` and commit both files unchanged, tagged `audit-2026-09-29`. Move the notebook to `reference/` and never edit it.
1. **Fix C1** (one function) and add `tests/test_elas_geometry.py`, which asserts a median collinearity residual < 5 px on every scene. Regenerate the GT overlay as a polyline. Archive `results/v1_elas` as `results/_invalid_pre_c1_fix`.
2. **Fix C2** (restore S's FP/FN rule) and add a unit test with the synthetic cases above.
3. **Fix the split:** make it independent of the training seed, add a scene-held-out test split, and save manifests once and reuse them.
4. **Restore the reference models from S**, adapted to ELAS (2 lanes, ELAS anchors):
   * `OfficialUFLDAdapter` as `baseline_ufld` (clone the official repo; offer `pretrained=True` + ImageNet normalization as the paper-faithful setting);
   * S's v02/v03/v04 with their variant-specific losses and warm-starts.
5. **Keep U's models as clearly named extra variants:** `lite_baseline` (U's single-frame CNN) and `v05_warp` (U's warped residual fusion). Compare `v05_warp` against `lite_baseline` with the **same current-frame backbone and the same LR**, so the temporal effect can be isolated.
6. **Fix HPO fairness and reproducibility:** use `TPESampler(seed=…)`, put the code hash and seed in the study name, use fresh storage per experiment, give the baseline the same HPO budget, and prune on pixel F1 rather than anchor F1.
7. **Fix the metrics:** make CULane-style IoU 0.5 (scaled width) the primary lane-level metric, and report 0.35 and pixel F1 as secondary. Select both checkpoints and post-processing on one declared metric.
8. Run a short smoke run, then a full run with ≥ 6 seeds. Only after that, split the file into modules (`datasets/`, `models/`, `losses/`, `metrics/`, `train/`, `report/`) with unit tests, as planned for the next stage.

**Which codebase is the foundation?** U, for data, evaluation and reporting infrastructure (after steps 1–3). S, as the source of truth for model and loss definitions (step 4).
