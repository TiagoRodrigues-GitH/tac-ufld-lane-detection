# Datasets

Four datasets share one internal representation (`FrameRecord`: lanes in original pixels, fixed lane slots, a known/unknown flag per slot, an optional valid band, sequence and frame ids, and metadata). Coordinate conventions, lane ordering, splits and official metrics differ per dataset; each adapter documents its own and writes it into every report (`protocol` block).

| Dataset | Slots | Status | Split used | Native metric (reported separately) |
|---|---|---|---|---|
| ELAS | 2 (ego-left, ego-right) | **validated on real data** (regression test, pilot run) | held-out scenes + 60-frame blocks with purge gap | none |
| CULane | 4 | tested on a synthetic copy of the real layout; **not run on real CULane** | official `train_gt` / `val_gt` / `test` lists | lane F1 @ IoU 0.5, 30 px at 1640×590, all lanes |
| TuSimple | 4 | tested on a synthetic copy; **not run on real TuSimple** | official train (validation carved from it) / official `test_label.json` | official `LaneEval` accuracy, FP, FN (+ F1 of 1−FP, 1−FN) |
| OpenLane (2D) | 4 | tested on a synthetic copy; **not run on real OpenLane** | official `training` (validation = held-out segments) / official `validation` as test | CULane-style F1 @ IoU 0.5, 30 px at 1920×1280, all lanes |

No CULane, TuSimple or OpenLane copy was available on the development machine. Their adapters are checked against fixtures that reproduce the published folder structure, list/JSON formats, image sizes and frame-id conventions (`tests/dataset_fixtures.py`). Since v0.4, every stage also runs end to end on those synthetic copies, at each dataset's real geometry and with its native metric (`scripts/smoke_datasets.py`, `tests/test_dataset_tools.py`). **Run `scripts/check_dataset.py` and `validate-dataset` on the real data before training** (below).

## Selecting datasets: `configs/datasets.yaml`

```yaml
datasets:
  elas:     {enabled: true,  root: "${ELAS_ROOT:-../../datasets/dataset_elas_v1}", config: configs/elas.yaml}
  culane:   {enabled: false, root: "${CULANE_ROOT}",   config: configs/culane.yaml}
  tusimple: {enabled: false, root: "${TUSIMPLE_ROOT}", config: configs/tusimple.yaml}
  openlane: {enabled: false, root: "${OPENLANE_ROOT}", config: configs/openlane.yaml}
```

Rules (tested in `tests/test_registry_cli.py`):

* only **enabled** datasets can be used; a disabled dataset's root is never expanded or read;
* `${VAR}` / `${VAR:-default}` are expanded; an unset variable without default, or a missing folder, is an explicit error for an enabled dataset;
* a run whose config names a disabled dataset is rejected, and `--dataset X` with a config of another dataset is rejected;
* root precedence: `--data-root` > `TAC_UFLD_DATA_ROOT` (legacy, ELAS) > registry root > `data.root`;
* ELAS stays the default (`python -m tac_ufld run` without `--config` uses `configs/elas.yaml`).

```powershell
python -m tac_ufld datasets                     # status of every dataset (disabled ones are not scanned)
python -m tac_ufld datasets --check             # parse every enabled dataset and count frames
$env:CULANE_ROOT = "D:\datasets\CULane"         # then set culane.enabled: true
python -m tac_ufld validate-dataset --dataset culane
python -m tac_ufld check-data --dataset culane
python -m tac_ufld run --dataset culane --confirm
python -m tac_ufld run --all-datasets --smoke    # every enabled dataset, each with its smoke config
```

## Real-data validation: `validate-dataset`

`python -m tac_ufld validate-dataset --dataset <name> [--root <path>]` (allowed on a disabled dataset when `--root` is given) writes `results/validate_<name>/validation_report.json` and overlays. It checks, per split: frame and sequence counts; image sizes of a sample against the declared size; lane points inside the image; slot occupancy; **left-to-right order of the slots** at their lowest shared row; availability of history frames *inside* sequences for the configured `temporal_step`; duplicate keys; adapter statistics (missing images, skipped lanes, slot/flag mismatches). CULane additionally reports the stored-frame stride of every clip. The overlays draw each slot in its own colour, every annotated lane in grey and the encoded row-anchor targets as dots: the dots must sit on the markings.

## Download, check, smoke test

* **Download**: `docs/DATASET_DOWNLOAD_GUIDE.md`, written for whoever downloads the data. It gives the links, which archives to take, their sizes and the expected folder trees.
* **Check a download** with plain Python; the check only reads files: `python scripts/check_dataset.py <dataset> <root>`. It reports file counts per split against the published counts, the annotation files, a sample of image/label pairs (plus history frames) and the image size, then gives a READY / NOT READY verdict.
* **Smoke test** every stage on every dataset, a few batches each: `python scripts/smoke_datasets.py`. When a dataset's root is set, it uses the real data with `configs/<dataset>_smoke.yaml`; otherwise it uses a synthetic copy of the dataset's layout. Output goes to `results/smoke_datasets/`.
* **Step by step**: see "HOW TO SWITCH DATASETS" at the top of `src/tac_ufld/cli.py`, also printed by `python -m tac_ufld --help`.

## Temporal sampling per dataset: what is and is not possible

History frame `k` of a sample is `frame_id − k · temporal_step`. When that frame doesn't exist (a sequence start, or a frame the dataset doesn't store), the nearest newer available frame is repeated. This never fails silently: the count is logged, a warning appears above 5 %, and `validate-dataset` reports the availability.

| Dataset | What is stored and annotated | History in the config | Possible | Not possible |
|---|---|---|---|---|
| ELAS | Continuous scene videos; every frame is stored and annotated | `temporal_step: 2`, 3 frames | Any step and window length (the history ablation grid covers steps 1–15 and 2–5 frames); recurrent state carried over a whole scene; jitter between consecutive annotated frames | Nothing specific; the limits are the small size (10 scenes in the pilots) and 2 ego lanes only |
| CULane | Clips store only every 30th (`*_30frame`) or 90th (`*_90frame`) video frame; every stored frame is annotated | `temporal_step: 90` (3 s at 30 fps; present in both folder types) | History 1 s apart in the 30-frame drivers or 3 s apart in all of them; carried state along the stored frames | Short-range history (< 1 s) does not exist. With step 30, the 90-frame drivers have no history. Jitter compares frames 1–3 s apart, so it measures less than on continuous video |
| TuSimple | 1-s clips of 20 consecutive frames (20 fps); only frame 20 is annotated | `temporal_step: 2` (frames 20, 18, 16) | Up to 19 history frames per sample, with any step 1–9 for 3 frames; unannotated history is fine as input | No annotated consecutive frames, so no jitter and no evaluation with state carried across frames; memory spans at most 1 s |
| OpenLane (2D) | Waymo segments at 10 Hz; annotated frames along each segment | `temporal_step: 1` (0.1 s) | Any step and window length; state carried over a segment; jitter | 3D lanes are not used by this 2D pipeline |

## ELAS (reference dataset)

Unchanged from v0.2 (see `docs/AUDIT_REPORT.md`, finding C1): p1..p4 lie at 0, ¼, ½, 1 of the scene's ROI height; single-point sides are *unknown* (ignored), all-NaN sides are *absent*; slots come from the XML (`left`/`right`); rows outside the ROI are ignored. Split: held-out test scenes `BR_S02`, `VIX_S05`, `VV_S03`; the other seven scenes in 60-frame blocks (val 20 %, seen-scene test 15 %) with a purge gap ≥ 10 frames; independent of the training seed; leakage-checked. Horizontal flipping is allowed with slot swap (ego-left ↔ ego-right).

**Metric note.** ELAS annotates both ego lanes in essentially every frame and every trained model predicts both slots, so at the lane level FP = FN and precision = recall = F1 = F2: lane errors are *misplaced* lanes (each one counts once as FP and once as FN), not missed ones. The pilot confirms this (5,212 predicted and 5,212 annotated lanes on 2,610 held-out frames for every model).

## CULane

Layout: `driver_*_30frame` / `driver_*_90frame` folders with `<clip>.MP4/<frame>.jpg` + `.lines.txt`, `laneseg_label_w16/`, `list/{train_gt,val_gt,test}.txt`, `list/test_split/test<k>_<category>.txt`.

Validation findings and changes (v0.3):

1. **Temporal step.** Frame ids are video-frame numbers; clip folders store every 30th frame (`*_30frame`) or every 90th (`*_90frame`). The ported config used `temporal_step: 30`, which finds **no history** in the 90-frame drivers (the temporal models would silently see the current frame three times). The supervisor's notebook used 90 (3 s at 30 fps), which exists in both folder types. `configs/culane.yaml` now uses 90 (with `purge_frames: 180`), the dataset warns when > 5 % of history frames are missing, and `validate-dataset` reports the stride histogram.
2. **Lane slots.** The official UFLD code takes a lane's slot from the segmentation label (`laneseg_label_w16`) at the lane's middle point (UFLD v2 `cache_dataset.py`). The port used a geometric rule (side of the image centre), which disagrees during lane changes. The adapter now uses the label when it exists (train/val) and falls back to the geometric rule only for the test split (no labels there; test evaluation uses every annotated lane anyway). The list's existence flags are cross-checked (`flag_mismatch` statistic).
3. The official test categories (`normal`, `crowd`, `hlight`, `shadow`, `noline`, `arrow`, `curve`, `cross`, `night`) become condition tags; `cross` has no lanes, so only its FP count is meaningful (as in the official evaluation).
4. Native metric: every annotated lane (not only the 4 slots), 30 px lines at 1640×590, IoU ≥ 0.5, Hungarian matching. The official C++ evaluator interpolates lanes with cubic splines; this implementation draws the annotated polylines, so numbers are close but not identical to published ones.

## TuSimple (new)

Layout: `train_set/label_data_{0313,0531,0601}.json`, `train_set/clips/<drive>/<clip>/{1..20}.jpg`, `test_set/clips/...`, `test_label.json` (often downloaded separately; the adapter looks in the root and `test_set/`). One JSON line per annotated frame (frame 20 of each 1-s clip): `lanes` (x per `h_samples` row, −2 = absent), `h_samples`, `raw_file`.

* **Slots**: the official UFLD rule (`scripts/convert_tusimple.py`): lanes shorter than 90 px end to end are dropped; the rest are split by the sign of their slope angle; on each side the two lanes closest to the centre (steepest) fill slots 1, 0 (left) and 2, 3 (right). Lanes not placed (short, or a third lane on one side) are kept for the native metric.
* **Valid band**: rows outside `[min(h_samples), max(h_samples)]` are unannotated and ignored.
* **Temporal context**: sequence = clip folder, frame id = 20; history frames 18 and 16 (`temporal_step: 2`, 20 fps).
* **Split**: TuSimple has no official validation set. Validation is carved from the training clips only: clips of each drive are ordered by clip number, cut into blocks of `block_size` consecutive clips, 10 % of blocks go to validation, and clips within `purge_frames` positions of another split are dropped. The official test set shares drives (e.g. `0531`) with training; this is the benchmark's protocol, reported in `split_report.json` (`shared_groups_train_test`), not a scene-disjoint test.
* **Native metric**: `LaneEval.bench` ported line by line (threshold 20 px / cos(angle) using a least-squares slope of each GT lane; a lane matches if ≥ 85 % of its points are within the threshold; FP/FN rates per image; images with more than 4 GT lanes drop the worst lane). Predictions are sampled at `h_samples` (−2 outside the predicted extent).
* Config: 288×800, 56 row anchors 64..284 (= `h_samples` 160..710), 100 cells, 4 lanes (official UFLD TuSimple setting).

## OpenLane (new, 2D)

Layout: `images/{training,validation}/segment-*/<timestamp>.jpg`, `lane3d_1000/{training,validation}/segment-*/<timestamp>.json` (or `lane3d_300`), optional `lane3d_1000/test/<scenario>_case/...` subsets.

* **Slots** from each lane's `attribute` (1 = left-left, 2 = left, 3 = right, 4 = right-right). Duplicate attributes keep the lane with more visible points (`duplicate_attribute` statistic). Attribute-0 lanes stay in `meta["eval_lanes"]` for the native metric: a 4-slot model can never predict them, which caps its native recall.
* **Points**: `uv` points with `visibility` ≤ 0.5 or outside the image are dropped; lanes need ≥ 2 points.
* **Metadata**: camera `intrinsic` / `extrinsic`, timestamps, categories and track ids are kept in `meta` (3D coordinates are not used by this 2D pipeline).
* **Frame ids** are positions in the segment's time-ordered frame list (10 Hz), so `temporal_step: 1` = 0.1 s.
* **Split**: OpenLane's `validation` set is the test set (as in the literature); validation for selection holds out whole training segments (`val_strategy: sequences`).
* **Native metric**: CULane-style F1 @ IoU 0.5 with 30 px lines at 1920×1280 over every annotated lane. This matches our reading of OpenLane's 2D evaluation settings but was **not cross-checked** against the official evaluator.
* Scenario tags (`curve`, `night`, ...) become condition tags when the scenario folders are present.

## Common behaviour

* Missing lanes: `slot_known=True, lane=None` = no lane (trained as "no lane"); `slot_known=False` = unusable annotation (ignored by losses and metrics).
* Coordinates are converted to model pixels only in target encoding (`x_model = x_orig · W/w_orig`), with the same half-open image bounds everywhere.
* Splits never depend on the training seed; `check_no_leakage` runs before every experiment (frame duplicates, held-out groups, purge gaps along the split order).
* Horizontal flip augmentation uses each adapter's `flip_permutation` (ELAS `(1, 0)`, others `(3, 2, 1, 0)`).
