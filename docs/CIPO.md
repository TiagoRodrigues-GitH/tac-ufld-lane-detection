# CIPO: closest in-path object detection (design note)

Goal: add to the lane detector what VisionPilot's **AutoSpeed** provides: detect the objects ahead, pick the
*closest in-path object* (CIPO), and estimate its distance. Lanes, the ego lane and lane changes already exist
(`tac_ufld/ego.py`, review 2026-10-18, section 5).

Status (2026-10-19, second pass): **implemented and measured.**

* `cipo.zip` was downloaded by the user, verified against the SHA-256 below, and extracted to
  `datasets/OpenLane/cipo`.
* The reader, the head, the train/eval script and the tests exist (section 8).
* The head was trained on a frozen UFLD-R18 lane model and measured on the local test split (section 9).
* Range, closing speed, TTC / THW and vehicles beside were added (section 10).

Everything marked **measured** was run on this machine. Everything else is a design choice, an estimate or a
cited figure.

## 1. Data status

### 1.1 What was on disk at the first pass (measured 2026-10-19, before the download)

`LaneDetection/datasets/OpenLane` holds:

* `images/{training,validation}` (50 + 50 segments, 19,827 frames).
* `images_training_1..4` (55 more training segments).
* `lane3d_1000/{training,validation,test}` (71,974 label JSONs).
* `OpenLane-V/`.

There is **no `cipo/` folder** and no CIPO file anywhere below `datasets/`. A lane label holds only
`extrinsic, intrinsic, lane_lines, file_path, pose` (one validation file checked), so no box is hidden inside the
lane labels either. `_downloads/OpenLane` holds the image parts and the two `lane3d_1000` tars, nothing else.
`docs/DATASET_DOWNLOAD_GUIDE.md` lists `cipo.zip` and `scene.zip` under "Not needed", which is why they are
missing.

> Correction to `docs/REVIEW_2026-10-18.md` section 5: OpenLane *provides* CIPO boxes, but the local copy does
> **not** include them ("already on disk" applies to the images and lanes only).

### 1.2 Which archive provides them

| archive | content | size | SHA-256 (mirror) |
|---|---|---|---|
| **`cipo.zip`** | `cipo/{training,validation}/segment-*/<timestamp>.jpg.json`, one file per image, all 1,000 segments | 467,271,721 B (446 MiB) | `056dfd400af25192dae0043e0facfe7742461ea7404e00f8c2fed24da1190c77` |
| `scene.zip` (optional) | `scene/SCENE/scene.json`: scene / weather / time-of-day per segment | 22,220 B | `7e1fdb27f1b925369b00a80b0ca6d9b0f465451328878185ea56a20cb4c5e14c` |

Sources:

* **Official:** the OpenLane data page ([`data/README.md`](https://github.com/OpenDriveLab/OpenLane/blob/main/data/README.md))
  links a download form (`https://forms.gle/BzxxkUZDuPTqFKgu9`). OpenLane asks you to register with the Waymo
  Open Dataset first (Waymo licence terms apply). The official Google Drive folder
  (`18upnDfB-VVuQf3GPiv_JQn1-BUOcAotk`, id from `scripts/download_openlane.py`) holds `cipo.zip` and `scene.zip`
  next to the image parts (per `docs/DATASET_DOWNLOAD_GUIDE.md`). Drive often refuses with "too many users"
  (`_downloads/download_openlane.log`).
* **Mirror:** the Hugging Face mirror that `scripts/download_openlane.py` already uses for the images
  (`AlayaNeW/OpenDriveLab___OpenLane`, `raw/cipo.zip`). The sizes and SHA-256 above are the mirror's
  (**measured** through its tree API, 2026-10-19).

Done 2026-10-19: the user downloaded the official file to `_downloads/OpenLane_cipo/cipo.zip`, its SHA-256
matches the table, and it is extracted to `OpenLane/cipo`. To repeat it from the mirror:

```powershell
curl.exe -L -C - -o "$env:OPENLANE_ROOT\..\_downloads\OpenLane\cipo.zip" https://huggingface.co/datasets/AlayaNeW/OpenDriveLab___OpenLane/resolve/main/raw/cipo.zip
(Get-FileHash "$env:OPENLANE_ROOT\..\_downloads\OpenLane\cipo.zip" -Algorithm SHA256).Hash   # must equal the table
Expand-Archive "$env:OPENLANE_ROOT\..\_downloads\OpenLane\cipo.zip" -DestinationPath $env:OPENLANE_ROOT  # -> OpenLane\cipo\
```

### 1.3 Annotation format (OpenLane `anno_criterion/CIPO/README.md`)

One JSON per image:

* `raw_file_path`: the image path.
* `results`: a list of boxes. Each box has:
  * `x`, `y`: top-left corner, in pixels of the 1920x1280 image.
  * `width`, `height`.
  * `id`: the **importance level**, a string "1".."4".
  * `trackid`: a tracking id, unique within the segment.
  * `type`: 0 unknown, 1 vehicle, 2 pedestrian, 3 sign, 4 cyclist.

The importance levels (paraphrased from the criterion):

| level | meaning |
|---|---|
| 1 | **the CIPO**: the closest object within the reaction distance with more than 50 % of its body in the ego lane. At most one per frame |
| 2 | objects interacting with the real or virtual ego-lane lines (cut-in, cut-out) |
| 3 | objects mainly within the reaction distance / drivable area, in the left, ego or right lane, occlusion below 50 % (oncoming vehicles included) |
| 4 | objects unlikely to affect the path (distant, off the drivable area, parked) |

### 1.4 What the release really contains (measured 2026-10-19 on the verified `cipo.zip`)

**Archive contents.**

* The archive holds 198,068 JSON files: 158,081 training and 39,987 validation, for 798 + 202 segments. Each file
  is named `cipo/<split>/<segment>/<timestamp>.jpg.json`.
* There are also `__MACOSX/` entries; the reader skips them.

**Differences from the README**, all handled by `parse_cipo`:

* The box list key is **`result`**, not `results`.
* The level `id` is an **int**, not a string.
* About 0.1 % of the boxes carry the level under **`attribute`** instead of `id` (78 of about 93k in a 2 % sample).

**The 95 segments of the lane run's split** (`results/openlane/data/split_manifest.csv`: 41 train, 7 val, 47 test):

| quantity | value |
|---|---|
| CIPO frames with an image on disk | 8,134 train / 1,388 val / 9,317 test (every frame of these segments has a CIPO file) |
| boxes | 409,415, about 22 per frame |
| boxes by level | 1: 10,522 / 2: 1,818 / 3: 34,753 / 4: 362,322 |
| boxes by type | vehicle 324,761 / pedestrian 81,414 / cyclist 3,240; **no** sign or unknown boxes |
| frames with a level-1 box | 56 % (10,504 with one, 9 with two, which breaks "at most one"; 8,326 with none) |
| level-1 box width at 480x320 | median 30 px, p10 7.9 px; 35 % narrower than 16 px, 51 % narrower than 32 px |
| all boxes at 480x320 | median width 9.2 px; 45 % narrower than 8 px (mostly distant level-4 objects) |

**Head stride.** The level-1 sizes settle the stride question of section 3: at stride 16, a third of the CIPOs
would be smaller than one cell, so the head works at **stride 8**.

**The lane labels hold what the range and headway work needs** (section 10):

* the camera intrinsics and extrinsics per frame (camera height 2.116 m, f_y about 2,074 px);
* a 4x4 vehicle-to-world `pose`. Frame timestamps advance about 10^7 units per 0.1 s, so 1 unit = 10 ns; the
  pose gives the ego speed (for example 0.8 m per frame = 8 m/s).

**What is not in OpenLane:** there is **no** 3-D box or range for the CIPO objects. Waymo's laser boxes are not
part of the release; they need the original Waymo TFRecords under the Waymo licence.

OpenLane ships an official evaluation, `eval/CIPO_evaluation`. It is built on the COCO evaluation (`eval.py
--anno_txt ... --res_txt ...`). Result files mirror the annotation layout, with a `score` field added to every
box. Its README gives no baseline table, and I found no published CIPO AP table in the repository (no
literature search was done).

## 2. Scope

| in scope | not in scope |
|---|---|
| 2-D boxes with an importance level, on the shared backbone of the lane models | 3-D detection (the Waymo laser boxes are not part of OpenLane) |
| choosing the CIPO by geometry against the predicted ego lane | multi-object tracking (`trackid` is used only for evaluation and smoothing) |
| a monocular distance proxy, and a time-to-collision cue | longitudinal control (ACC/AEB); the CARLA loop is a separate task |

## 3. Architecture: a light head on the shared features

```
image -> backbone (shared with the lane head) -+-> lane head (unchanged)            -> lanes -> ego lane (ego.py)
                                               +-> neck (tiny FPN) -> CIPO head -> boxes + level --+
                                                                                                   +-> in-path test -> CIPO + distance
```

**Feature taps.** No edits to the model files are needed:

* UFLD (ResNet-18): `UFLDNet.model(x)` already returns `(x2, x3, x4)`. Their sizes are 128 ch at stride 8,
  256 ch at stride 16 and 512 ch at stride 32. The neck reads them **before** the lane head's 8-channel squeeze.
* Lite: `LightweightBackbone.net` is a `Sequential` of four stride-2 blocks, so `net[:3]` gives 96 ch at
  stride 8 and `net[3]` gives 128 ch at stride 16.
* Temporal models: the head reads the **current** frame's features. The fused features are an ablation for
  later.

**Neck.** 1x1 lateral convolutions to 64 channels, nearest-neighbour upsampling to stride 8, summed, then one
depthwise-separable 3x3 block. A YOLOP-style shared encoder, much thinner (Wu et al. 2022).

**Head.** An anchor-free centre head at stride 8 (two depthwise-separable 3x3 blocks, then a 1x1 output):

* a heatmap with 4 channels (importance levels 1-4, sigmoid);
* a box size or (l, t, r, b) distances (4);
* a sub-cell centre offset (2);
* optionally, the object type (5).

Decoding is **NMS-free**: 3x3 max-pool peak picking plus top-k. YOLOv10 removes NMS for the same reason
(Wang et al. 2024): no NMS in the deployed graph, so the ONNX/TensorRT export stays a plain tensor program.
The repository's deploy path relies on that.

**Cost (estimated** with a prototype of this head on random inputs at 480x320, the OpenLane network size):

| | params | MACs |
|---|---|---|
| ResNet-18 taps | 0.073 M | 0.07 G |
| lite taps | 0.030 M | 0.055 G |

For scale, the lite lane model is 2.49 M params and 1.99 GMACs, and UFLD-R18 is 21.8 M and 7.12 GMACs (review
R3, measured at 384x512).

**Why stride 8.** The Waymo front camera has f_y ≈ 2,074 px (measured, below), so a 1.8 m wide car at 50 m is
about 75 px wide in the original image and about 19 px at 480x320. That is 2.3 cells at stride 8, but 1.2 at
stride 16. **Settled by the data** (section 1.4): the median level-1 box is 30 px wide at 480x320, and 35 % are
narrower than 16 px, so stride 8 it is.

**Reference design point: a separate detector.** YOLOv10-N (about 2.3 M params, 6.7 GFLOPs at 640², as reported)
or RT-DETR-R18 (about 20 M params, 60 GFLOPs, as reported; Zhao et al. 2024), fine-tuned on the CIPO boxes and
using the same in-path rule (section 5). Verify these figures against each paper's table before citing them.

The paper's claim is the trade-off:

* the shared head costs 1-3 % of the lane backbone's MACs (0.055-0.07 G);
* YOLOv10-N, scaled to 480x320, is about 1.3 GMACs. This is an estimate: 6.7 GFLOPs at 640², with the usual
  convention FLOPs = 2 x MACs, scaled by the pixel ratio 0.375.

So a separate detector costs about 20x the head, and is 65 % of the lite lane model on its own.

## 4. Training

**F: frozen backbone (recommended first).** Load the trained lane checkpoint of the same seed, freeze the
backbone with BN in eval mode, and train only the neck and head.

* The lane outputs stay **bit-identical**, so every lane result in the article remains valid, and the head is
  a pure add-on with its own cost.
* It is cheap: with augmentation off, the backbone features of the training frames can even be cached once.
* Risk: the lite backbone is trained from scratch on lanes only, so its features may carry little object
  information. Expect frozen-lite to be clearly weaker than frozen-ResNet (which has ImageNet init).

**J: joint training (ablation, NOT implemented).** Loss `L = L_lane + lambda_det * L_det`, with `lambda_det` in
{0.5, 1, 2} chosen on validation. On the same seeds, compare the lane F1 of J with the lane-only model, as an
exploratory Holm family (the primary lane family is untouched). Multi-task training can help or hurt the lanes;
YOLOP reports both effects across its tasks. J needs the lane targets of the main training pipeline, so it
belongs in the experiment code, not in the standalone script.

**Losses (as implemented).**

* Heatmap: CenterNet-style penalty-reduced focal loss (alpha 2, beta 4), with a Gaussian of CenterNet's radius
  for the box size.
* Box: L1 on the sub-cell centre offset and on log(w, h) in cells, both weight 1. There is no GIoU term.
* There is no extra weight on the level-1 channel yet. If level-1 AP lags, add one and choose it on validation.

**Classes and ignore regions.**

* Types 1, 2 and 4 are targets.
* Types 0 (unknown) and 3 (sign) become ignore regions: no loss, because signs are not path obstacles. The
  local data has none of them (section 1.4).

**Augmentation** (`--augment`). The repository's `GeometricAugmenter` (`data/geometric.py`) samples one
homography. It is applied to the image, and `warp_boxes` applies it to the boxes:

* transform the four corners and take their axis-aligned box;
* clip it to the image;
* drop the box if less than 30 % of its area remains.

The settings are: flip p = 0.5, shift 5 % / 3 %, zoom 0.9-1.1, crop 0.85-1. A flip mirrors the boxes; the
importance level does not change under a flip.

**Data and split (as implemented).**

* Frames: every CIPO frame of the chosen segments whose image is on disk. Lane labels are read only for
  validation and test, where they are needed.
* Split: the **segments of the lane run's own `split_manifest.csv`**. A train segment stays train; a val or test
  segment stays val or test. So the frozen backbone never saw a validation or test segment. Inside these
  segments, every CIPO frame is used, not only the frames with attributed lanes that the lane run kept.
* OpenLane-V uses the same Waymo frames. There, the boxes would move by `y - 480` (the band crop), and boxes
  entirely above the band would be dropped (not implemented).

## 5. "In my lane": the geometric decision

On OpenLane the ego boundaries are slots **1 and 2** (attributes 2 = left, 3 = right; `data/openlane.py`).
`ego.ego_lane_state` reads slots 0 and 1 (the ELAS convention), so the CIPO code passes `[lanes[1], lanes[2]]`.

For each detected box `(x, y, w, h)` of type 1, 2 or 4:

1. Take the **contact row** `v_b = y + h`, where the object meets the road.
2. Find the ego-lane span at that row: `xl = ego.x_at(left, v_b)` and `xr = ego.x_at(right, v_b)`. `x_at`
   extrapolates linearly beyond the polyline, so near objects below the last lane point still get a span
   (`cipo.ego_span`).
   * **One boundary missing (as implemented):** add a nominal 3.5 m lane on a flat road from the visible
     boundary; its width in pixels is `3.5 f_x (v_b - c_y) / (f_y h)`. A width remembered from earlier frames
     would be the streaming refinement.
   * **Both missing, or the extrapolated boundaries cross:** use a virtual 3.5 m lane centred on the camera
     column. Level 2 of the criterion also speaks of "virtual" ego-lane lines.
3. **In path** if `|[x, x+w] ∩ [xl, xr]| / w >= 0.5`. This is the 50 % body-in-ego-lane condition of level 1.
4. **CIPO** = the in-path box with score >= tau, nearer than a maximum distance `d_max` (the "reaction
   distance"), with the largest `v_b`, that is, the smallest flat-ground distance (section 6). `tau` and `d_max`
   are chosen on validation (`train_cipo.tune_rule`).

Three variants are reported, so the paper can say what the geometry adds:

* (a) the learned level-1 peak alone;
* (b) geometry over all detected objects, ignoring the learned level;
* (c) geometry as a filter, ranked by the learned level-1 score.

**During a lane change** (`LaneChangeDetector` reports ±1), the target lane should also count as "path", which
is how production ACC selects targets. This is not implemented. OpenLane has no lane-change label; it can be
derived from `pose` (review section 5).

## 6. Distance proxy

OpenLane gives 2-D boxes only, with no range. Two proxies:

* **Flat ground:** `Z = f_y * H / (v_b - v_h)`.
  * `H = 2.116 m`: the camera height from the extrinsic, **measured** on 34 labels (2.115-2.117).
  * `f_y ≈ 2,074 px` (2,044-2,088), from the intrinsic.
  * `v_h` is the horizon row. `train_cipo.py` uses `v_h = c_y` (pitch 0). OpenLane's lane `xyz` projects to
    its `uv` without the extrinsic rotation (median 2.9 px), so the camera axes are already the road-aligned
    axes. The pitch of the road ahead is estimated from the lanes in section 10.
* **Lane-ground reference** (no lidar needed): the ground-truth ego boundaries carry 3-D points in a camera-centred
  frame (x forward, y left, z up). **Measured:** lane points lie at z = -2.41 m (median; p10 -3.40, p90 -1.96),
  which matches a camera 2.12 m above a road that is not flat. Interpolate the boundaries' forward coordinate at
  image row `v_b`, and you get a distance that includes slope (the `up_down_case` test subset).

Metrics:

* the abs-rel error of the flat-ground proxy against the lane-ground reference, on ground-truth level-1 boxes
  (this validates the proxy) and on the predicted CIPO (end to end);
* the share of boxes within 10 %;
* both by range bin (0-20, 20-40, 40-80 m).

**Exclusions:** boxes cut by the bottom image edge (very close vehicles) have no visible contact row. They are
flagged and excluded from the distance metrics.

**Time to collision:** see section 10 (range models with error bars, Kalman filter on range and range rate,
TTC / THW warning rule, vehicles beside).

## 7. Metrics and protocol

| group | metric | definition | why |
|---|---|---|---|
| detection (primary) | **AP@0.5, level 1** | COCO 101-point AP at IoU 0.5, class = level 1, over the test frames | the CIPO itself |
| detection | AP@0.5 per level, AP@0.5 pooled over levels 1-4, COCO AP@[.5:.95] | same, other classes and thresholds | comparability with the official `CIPO_evaluation` |
| CIPO decision | **CIPO frame accuracy** | per frame, the selected CIPO vs the ground-truth level-1 box: correct if both are absent, or if IoU >= 0.5 | the AutoSpeed output |
| | false-CIPO rate / missed-CIPO rate | frames with a wrong or phantom CIPO (phantom-braking risk) / frames with a missed CIPO | the ADAS-facing error types, as in review section 8 |
| in-path (decomposed) | rule validity | share of ground-truth level-1 boxes that the rule calls in-path **with ground-truth lanes** | does the geometry match the annotators' criterion? |
| | selection with oracle boxes | ground-truth boxes (all levels) + **ground-truth** lanes -> is the level-1 box picked? Then the same with **predicted** lanes | separates the cost of lane errors from detection errors |
| | end to end | predicted boxes + predicted lanes = the CIPO frame accuracy above | the system |
| distance | abs-rel, share within 10 %, by range | section 6 | AutoSpeed's distance output |
| stability | CIPO flicker and switches per minute | flicker: the predicted CIPO appears or disappears while the labelled one does not; switch: it changes `trackid` while the labelled level-1 `trackid` does not | the article's temporal theme |
| cost | params, MACs (`evaluation/efficiency.count_macs`), latency | head alone, and the full lane + CIPO model, vs a separate detector | the embedded claim |

**Official numbers (to do).** The paper's AP figures should come from the official `eval/CIPO_evaluation`. Its
result files carry a `type` per box, and the head does not predict a type yet. So the internal AP (COCO
101-point, `cipo.average_precisions`) is the only one measured so far. Before publication, add a type head (or
check how the official tool uses `type`) and cross-check once.

**Statistics (to do).** Seeds as in the lane runs. Confidence intervals by bootstrapping over **segments**,
because frames are correlated (review M4).

**Breakdowns (to do):**

* `scene.zip`: weather, night, scene type.
* the lane3d test cases: curve, intersection, night, merge/split, up/down, extreme weather.

**Bounds of the claim.**

* Local subset: about 100 segments with images, about 1/10 of OpenLane.
* No published CIPO baseline was found, so the result is a capability demonstration and a cost trade-off, not a
  state-of-the-art claim.

## 8. Implementation (as built, 2026-10-19)

All files are new. Nothing edits `models/registry.py`, `models/lite.py`, `models/ufld.py` or `ego.py` (they are
only imported). The scripts are standalone, outside the registry and the main experiment, so the lane study is
unaffected.

| file | content |
|---|---|
| `src/tac_ufld/data/openlane_cipo.py` | `parse_cipo` (every format variant of section 1.4); `CipoSource`: the `cipo/` folder **or `cipo.zip` read directly**; `build_frames`: joins boxes, images and (val/test) lane labels, camera and pose; caps each split BEFORE reading, as contiguous blocks per segment; `read_split_manifest`; CenterNet targets (`gaussian_radius`, `encode_targets`, ignore regions); `warp_boxes`; `CipoDataset` |
| `src/tac_ufld/models/cipo_head.py` | `FeatureTap`: forward hooks on the lane model's backbone, so ONE pass gives the lanes and the features (ResNet layer2-4; lite blocks 3-4; temporal models refused); `CipoHead`: tiny FPN + centre head; `cipo_loss`; NMS-free `decode`; `HeadProbe` for MAC counting |
| `src/tac_ufld/cipo.py` | `CameraModel`, `ego_span`, `in_path_fraction`, `path_geometry`, `select_cipo` (learned / geometric / hybrid), lane-ground reference distance, COCO-style `average_precisions`, `cipo_frame_metrics`, `cipo_stability` |
| `src/tac_ufld/proximity.py` | section 10: range models and error model, horizon from lanes, size prior, fusion, Kalman range tracker, TTC / THW / FCW rule, side vehicles, ego speed |
| `scripts/train_cipo.py` | trains the head on a FROZEN lane checkpoint; tunes the decision rules on val; scores test once; writes `metrics.json`, `history.csv`, `head.pt` |
| `scripts/cipo_range_check.py` | section 10 measurement on labelled boxes (no detector, CPU) |
| `tests/test_cipo.py` (18 tests), `tests/test_proximity.py` (10) | synthetic OpenLane tree from ONE flat-ground pinhole camera (exact answers): reader (folder = zip), splits and caps, targets -> decode round trip, focal loss and ignore regions, the feature tap on UFLD and lite (lane output unchanged), head cost, in-path geometry, rules, AP hand cases, frame metrics, stability, distance; range models vs finite differences, pitch from lanes, Kalman, tracker, FCW, side gaps; both scripts end to end |

Usage (from the repository root, with the venv):

```powershell
python scripts/train_cipo.py --lane-ckpt results/openlane/seed_1/checkpoints/ufld_baseline.pt `
    --openlane-root $env:OPENLANE_ROOT --out results/cipo/ufld_baseline_s1 --epochs 12 --augment
python scripts/cipo_range_check.py --manifest results/openlane/data/split_manifest.csv `
    --openlane-root $env:OPENLANE_ROOT --out results/cipo/range_check
python -m pytest tests/test_cipo.py tests/test_proximity.py -q
```

Not implemented (named, so nobody assumes otherwise):

* joint training;
* a type head and the official-format result writer;
* temporal lane models as the backbone;
* the lane-change target lane;
* bootstrap confidence intervals and per-scene breakdowns.

## 9. Measured (2026-10-19)

### 9.1 Tests and smoke run

* **Tests:** `tests/test_cipo.py` (18) and `tests/test_proximity.py` (10) pass. The existing suite without the
  ELAS dataset tests still passes (348 passed, 1 skipped, CPU).
* **Smoke run on real frames** (CPU, because the GPU was busy): `results/cipo/smoke_ufld_cpu`. It used 287 train,
  98 val and 188 test frames, 2 epochs, read straight from `cipo.zip`. It ran end to end: frames, training, rule
  tuning, test metrics, `metrics.json`. Its detection numbers mean nothing: the head was still predicting
  nothing.

### 9.2 First real run: frozen UFLD-R18, one seed

**Setup.**

* **Run:** `results/cipo/ufld_baseline_s1`; log in `results/cipo_ufld_baseline_s1.log`.
* **Backbone:** `results/openlane/seed_1/checkpoints/ufld_baseline.pt`, frozen.
* **Data:** all frames of the lane run's segments: 8,134 train, 1,388 val, 9,317 test.
* **Training:** 12 epochs, batch 16, AdamW 2e-3, flip / zoom augmentation, AMP, on the RTX 3050. An epoch took
  80-120 s, about 30 min in all, plus 6 min of frame loading and 11 min of test evaluation.
* **Selection:** best validation level-1 AP@0.5 = 0.376 at epoch 9.

**Detection (test, 9,317 frames):**

| AP@0.5 level 1 | AP@0.75 level 1 | AP@[.5:.95] level 1 | AP@0.5 level 2 / 3 / 4 | AP@0.5 any level |
|---|---|---|---|---|
| **0.148** | 0.028 | 0.055 | 0.004 / 0.150 / 0.022 | 0.043 |

* **Validation overstates test (0.376 vs 0.148).** Validation is 7 segments (about 780 level-1 boxes) and also
  chose the epoch; the test set is 47 segments.
* **"Level 1" is a relation, not an appearance.** The same lead car is level 1 when it is the nearest in-path
  object within the reaction distance, and level 3 otherwise (test AP on level 3 is 0.150). A small head on
  frozen lane features cannot learn that relation well.
* **Small objects dominate the class-agnostic AP:** 45 % of the boxes are under 8 px at 480x320.

**CIPO decision (test).** The rules were tuned on validation:

* learned: level-1 threshold 0.4;
* geometric: score 0.3, d_max 80 m;
* hybrid: score 0.3, level-1 threshold 0.1, d_max 80 m.

56.6 % of the test frames have a level-1 box, so "never a CIPO" scores 0.434 accuracy.

| rule | accuracy | precision | recall | false CIPO | missed CIPO | flicker / switches per min |
|---|---|---|---|---|---|---|
| learned (level-1 peak) | 0.512 | 0.445 | 0.156 | 0.110 | 0.477 | 41 / 7.9 |
| geometric (nearest in-path) | 0.492 | 0.394 | 0.245 | 0.213 | 0.427 | 63 / 27 |
| **hybrid** (in-path filter + level-1 score) | **0.539** | **0.504** | 0.219 | 0.122 | 0.442 | 43 / 13 |
| *oracle boxes + ground-truth lanes* | 0.626 | 0.656 | 0.407 | 0.120 | 0.336 | - |
| *oracle boxes + predicted lanes* | 0.657 | 0.748 | 0.459 | 0.087 | 0.306 | - |

For the oracle rows, the labelled boxes of all levels go through the geometric rule, with d_max = 45 m tuned on
validation.

**In-path decomposition (test, 2,874 level-1 boxes in frames with both ground-truth ego boundaries):**

* **Rule validity:** the 50 % rule calls 82.6 % of the annotated level-1 objects in-path with the ground-truth
  lanes, and 84.1 % with the predicted lanes.
* **Ego span at the box row,** with ground-truth lanes:
  * from both lanes: 2,346 boxes;
  * virtual lane: 500 boxes, because the two extrapolated boundaries crossed. Most of these are far boxes above
    the end of the labelled lanes; extrapolating along the lane's direction would fix them.
  * at or above the horizon: 28 boxes.
* **Even with perfect boxes, "nearest in-path within d_max" agrees with the annotators on only about 63-66 % of
  frames** (recall 0.41-0.46). The level-1 criterion also involves the reaction distance and occlusion, which a
  fixed d_max only approximates. This is the ceiling of the geometric decision as implemented. Better boxes alone
  cannot pass it.

**Distance (test).**

* **Labelled level-1 boxes,** flat road with pitch 0, against the lane-ground reference (2,198 boxes):
  * median error 7.8 %, 58 % within 10 %;
  * by range: 6.6 % (0-20 m), 17.1 % (20-40 m), 27.3 % (40-80 m), 33.7 % (> 80 m).
  * The pitch from the lanes (section 10) lowers these: level-1 median from 7.8 % to 5.3 %, and all objects at
    40-80 m from 36.6 % to 7.9 %.
* **Predicted CIPOs that match a level-1 box** (hybrid, 824 boxes): median error 14.7 %.

**Stability.** The CIPO flickers 40-60 times per minute with no temporal smoothing. Temporal fusion (the
article's theme) or the range tracker of section 10 is the obvious next step.

**Cost (480x320, RTX 3050, batch 1, FP32, PyTorch with Python overhead):**

| | params | GMACs | latency |
|---|---|---|---|
| lane model (UFLD-R18 + row-anchor head, full input) | 43.4 M | 5.58 | 8.4 ms |
| **CIPO head** | **0.073 M** | **0.069** (1.2 % of the lane model) | +2.8 ms (11.2 ms lane + head) |

**Reading the result.** As a cheap add-on, the frozen-backbone head works end to end at 1.2 % extra compute, but
its level-1 detection is weak (AP@0.5 = 0.15). It beats the trivial decision by about 10 points of accuracy,
with a false-CIPO rate of 12 %. It is not yet good enough to act on.

**Next, in order of expected value:**

1. **Joint training** (option J, section 4): unfreeze the backbone, and keep the lane loss as a regulariser.
2. **More context in the level decision:** a wider head, a stride-16 context branch, or the level-1 decision
   made by the geometric rule on class-agnostic boxes (class-agnostic AP would then be the detector's metric).
3. **Temporal smoothing** of the CIPO (track-level decision).
4. **A bigger and better validation set:** the lane run's 7 validation segments are too few. Bootstrap over test
   segments for the confidence intervals.
5. **The lite backbone, frozen** (`lite_baseline.pt`; its lane head predates the 2026-10-18 fix, but the
   backbone features are what count). Not run: the GPU went back to the CARLA session.

All of this is one seed, one backbone and a frozen setting: an indicative first measurement, not a result for
the paper.

## 10. Range, closing speed, TTC / headway and vehicles beside (the "predictor")

Goal: the basis of a future forward-collision / proximity warning. The code is `src/tac_ufld/proximity.py`; the
measurement on real data is `scripts/cipo_range_check.py`; the tests are `tests/test_proximity.py`.

### 10.1 Monocular range from the contact row, with pitch

A road point at forward range d, seen by a camera h metres above the road and pitched down by theta, appears at
the angle `phi = theta + alpha` below the horizontal, with `alpha = atan((v - c_y) / f_y)`. Hence:

    d = h / tan(theta + atan((v - c_y) / f_y))         (theta = 0:  d = f_y h / (v - c_y))

* **Inputs** come from the frame's own OpenLane intrinsics and extrinsics: h = 2.116 m, f_y about 2,074 px.
* **Pitch** comes from the vanishing point of the ego-lane boundaries (`horizon_from_lanes`). Fit a line to the
  nearest half of each boundary and intersect the two lines; that row is the horizon of the road plane just
  ahead, and `theta = atan((c_y - v_h) / f_y)`. This absorbs the car's pitch (braking, load) and gentle slopes,
  which a fixed extrinsic cannot. The tests recover a 1-2 degree pitch exactly from synthetic lanes.

**Error model** (first order, `ground_range_sigma`). With `d = h cot(phi)`, `|dd/dphi| = h + d^2/h`. So:

    sigma_d = (h + d^2/h) * sqrt((f_y / (f_y^2 + (v - c_y)^2) * sigma_v)^2 + sigma_theta^2)
           ~ d^2 / (f_y h) * sigma_v   +   d^2 / h * sigma_theta

**The error grows with d^2.** One pixel of row error costs d^2 / (f_y h): 0.02 m at 10 m, 0.37 m at 40 m and
1.5 m at 80 m. A pitch error of 0.2 degrees costs 0.17 m, 2.6 m and 10.6 m at the same ranges. The pitch, not
the pixel, dominates beyond about 20 m, which is why the lane vanishing point matters. The tests check the model
against finite differences, and the d^2 growth (sigma at 40 m = 4 x sigma at 20 m).

### 10.2 Size prior

    d = f W / w_px     sigma_d^2 = (d sigma_W / W)^2 + (d^2 sigma_px / (f W))^2

Here W is the vehicle **width** or, for pedestrians and cyclists, the **height** (`SIZE_PRIORS`). The priors
are design assumptions, not published values: car 1.8 +- 0.15 m, pedestrian 1.7 +- 0.10 m, cyclist
1.75 +- 0.15 m.

* **Error:** the prior's spread makes the error grow with d, only linearly, so beyond some range it beats the
  contact model.
* **Bias:** a vehicle seen at an angle, or a truck, has a wider box than 1.8 m and looks nearer. Section 10.6
  measures this bias.

`fuse_ranges` combines the two estimates with inverse-variance weights.

### 10.3 Closing speed: a constant-velocity Kalman filter per CIPO

The state is `(d, d_dot)`, with `F = [[1, dt], [0, 1]]` and white-noise acceleration discretised as
`Q = q [[dt^3/3, dt^2/2], [dt^2/2, dt]]`. This is the discretised Wiener velocity model (Saerkkae, *Bayesian
Filtering and Smoothing*, 2013, ch. 4). The measurement is the fused range, with the range-dependent variance of
10.1-10.2.

`CipoRangeTracker` handles the stream:

* the box continues the track if IoU >= 0.3 with the previous CIPO box **and** the range passes a 3-sigma
  innovation gate;
* otherwise a new object became the CIPO and the filter restarts;
* without a box, it predicts for up to 5 frames.

The closing speed is `-d_dot` from the filter, never a frame difference: at 10 Hz and sigma_d = 1 m, a frame
difference has a standard deviation of about 14 m/s. `q = 2 m^2/s^3` is a tuning assumption, to be set against
CARLA ground truth.

### 10.4 TTC, headway and the warning rule

* `TTC = d / (-d_dot)` while closing (closing speed above 0.5 m/s), otherwise inf. This is a constant-velocity
  TTC: it ignores the lead vehicle's deceleration, so it is late for a braking lead. The fix would be a
  constant-acceleration filter.
* `THW = d / v_ego`. The ego speed comes from the vehicle speed signal, or on OpenLane from consecutive `pose`s.
* `FcwMonitor` warns when TTC < `ttc_s`, or, if enabled, THW < `thw_s`. The condition must hold for 2
  consecutive frames, and TTC is used only once the closing speed's sigma is below 3 m/s.

Thresholds and their sources:

* **TTC: verified.** NHTSA's NCAP FCW confirmation test requires the alert at **TTC >= 2.1 s** (stopped lead,
  ego at 72.4 km/h), **>= 2.4 s** (decelerating lead) and **>= 2.0 s** (lead at 32.2 km/h). Source: the NHTSA
  confirmation test reports, e.g. Dynamic Research for NHTSA, 9 Nov 2012, 2013 Ford Taurus, read 2026-10-19.
  The default `ttc_s = 2.4 s` is the least that passes all three. The pipeline latency (detection, filter
  convergence, the 2-frame debounce of 0.2 s) and the constant-velocity optimism come on top, so the margin has
  to be calibrated in CARLA.
* **ISO 15623:2013** (forward vehicle collision warning systems) defines the performance requirements and test
  procedures. Its text is paywalled and its numbers are not reproduced here.
* **THW: no verified published threshold was found**, so `thw_s` is off by default. Set it explicitly, as a
  design choice.

### 10.5 Vehicles beside (forward-adjacent zone)

`lateral_position` takes the contact row's range d. It converts image columns to metres at that range,
`X = (u - c_x) d / f_x`, and returns:

* the lane: ego / left / right (the in-path rule of section 5 decides "ego");
* the lateral position of the box centre;
* the gap from the box's nearest edge to the ego-lane boundary on its side (negative = over the line, e.g. a
  cut-in);
* the gap to the ego car's side (half-width 0.95 m, an assumption).

`adjacent_close` flags a left or right object within 15 m and with less than 1.0 m to the ego side. Both
numbers are assumptions, to calibrate in CARLA.

**Limit:** a front camera only sees the zone ahead of the ego car's front. The real blind spot beside and
behind the driver, the zone of ISO 17387 lane-change decision aids, needs side or rear sensors.

### 10.6 Measured on OpenLane (labelled boxes, no detector)

**Setup.** `scripts/cipo_range_check.py` on the 47 test segments, 9,317 frames, run 2026-10-19 (CPU, frame
loading 925 s; `results/cipo/range_check/`).

* **Objects:** every labelled vehicle, pedestrian and cyclist whose contact row lies within the rows covered by
  the ground-truth ego lanes, and whose box is not cut by the image bottom. That gives 27,186 objects.
* **Reference range:** the lanes' own 3-D points at that row, which are not an independent sensor.
* **Assumed errors:** 2 px row error and 0.2 degrees pitch error.

**Range, on the 17,414 objects whose frame gives a lane vanishing point** (64 %; the others lack one of the two
ego boundaries or have non-converging lines). Each cell is the median |error| / reference, then the share within
10 %:

| reference range | n | flat road, pitch 0 | **pitch from the lanes** | size prior | fused |
|---|---|---|---|---|---|
| all | 17,414 | 19.2 % / 30 % | **5.7 % / 69 %** | 22.5 % / 21 % | 9.2 % / 53 % |
| 0-20 m | 4,128 | 6.3 % / 71 % | **3.2 % / 86 %** | 21.2 % / 23 % | 4.2 % / 84 % |
| 20-40 m | 5,412 | 18.4 % / 27 % | **4.5 % / 79 %** | 18.5 % / 26 % | 7.3 % / 63 % |
| 40-80 m | 5,805 | 36.6 % / 12 % | **7.9 % / 60 %** | 28.8 % / 17 % | 16.6 % / 32 % |
| > 80 m | 2,069 | 43.8 % / 9 % | **18.2 % / 30 %** | 25.2 % / 18 % | 23.0 % / 23 % |

**Findings:**

* **The pitch is the dominant error, and the lanes fix most of it.** The pitch read off the lanes has a median of
  +0.48 degrees, with p05 -1.47 and p95 +1.60 degrees. Ignoring it (`flat`) is fine within 20 m and useless
  beyond 40 m, exactly as the d^2 error model predicts. With the lane vanishing point, the median error is
  5.7 % overall and 7.9 % at 40-80 m.
* **The width prior is biased:** vehicles come out 15 % too near (median signed error -0.15). Boxes of vehicles
  seen at an angle include their side, and trucks are wider than 1.8 m. For ego-lane vehicles, seen mostly from
  behind, the bias is -12 %. Fused with the contact range, it makes things worse (9.2 % vs 5.7 %). Use it only
  as a fallback when no lane horizon exists, or correct its bias first. Pedestrians' height prior is unbiased
  (+1 %).
* **On the level-1 objects alone** (2,201; 1,484 with a horizon), the pitch-corrected range has a median error
  of 5.3 % and 71 % within 10 %.
* **The error model is too optimistic.** Only 76 % of the objects fall within 2 sigma, where a calibrated model
  gives 95 %. The median |error| / sigma is 0.97, where a Gaussian gives 0.67, so the assumed sigmas are about
  1.5x too small. That is about 0.3 degrees of pitch error and/or 3 px of row error. Part of this is the
  reference's own error.

**Closing speed (level-1 tracks).** These are 1,933 frames in runs of at least 10 consecutive frames. The
filter's measurement is the fused range with its sigma. The reference is the slope of the reference range over
+-0.5 s; 1,734 frames have both.

| closing speed from | MAE | RMSE |
|---|---|---|
| frame differences of the range | 9.15 m/s | 28.5 m/s |
| **constant-velocity Kalman filter** (q = 2 m^2/s^3) | **1.92 m/s** | **4.52 m/s** |

The filter is what makes a TTC usable at all: frame differences are about 5x worse.

**Warnings and headway on the same tracks** (ego speed from the poses):

* TTC < 2.4 s on 13 frames; the debounced FCW rule fired on 9 frames. Waymo's logs contain no collisions, so
  these are close approaches or false alarms. Labelling them needs the video, which was not done.
* THW over 1,663 frames with the ego moving: median 4.7 s; no frame under 1 s; 7.7 % under 2 s.

**Not validated here:**

* true range: OpenLane has no 3-D object boxes;
* detector-in-the-loop range: the CIPO head's boxes instead of the labels;
* the side-vehicle flag.

CARLA (10.7) is the place for all three.

### 10.7 What CARLA validation needs (next step)

* **Ground truth:** per frame, the true range and range rate of the lead and adjacent vehicles from the
  simulator's actor transforms. Also the ego speed, and the camera intrinsics, height and pitch as rendered.
* **Scenarios:** the three NCAP FCW scenarios (stopped, decelerating and slower lead vehicle at the speeds
  above), cut-ins from both sides, and slopes, through ScenarioRunner / OpenSCENARIO.
* **Model inputs:** the detector's boxes (the head of section 3 fine-tuned on CARLA frames, or ground-truth
  boxes first, to isolate the range and filter errors).
* **Metrics:**
  * range abs-rel by range bin, and the coverage of the error model;
  * closing-speed RMSE;
  * TTC at the first warning vs the NCAP criteria;
  * false warnings per hour in free driving;
  * the adjacent-close flag's precision and recall against the true lateral gap.
* **Tuning on that ground truth:** `q`, `sigma_v`, `sigma_theta`, the size priors and the warning margins.

## References (keys in the article's `references.bib`)

* `wu2022yolop`: YOLOP, a shared encoder with task heads.
* `wang2024yolov10`: YOLOv10, NMS-free real-time detection.
* `zhao2024rtdetr`: RT-DETR.
* `chen2022openlane`: PersFormer and the OpenLane benchmark, including CIPO.
* Optional: `yu2020bdd100k`, `han2022yolopv2`.
