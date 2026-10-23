# Lane data from the CARLA simulator

CARLA renders driving scenes in Unreal Engine and knows the exact road geometry, so every frame
comes with lane boundaries for free, in any weather and with as much traffic (occlusion) as needed.
`scripts/carla_collect.py` turns that into clips in the TuSimple layout, which the existing
TuSimple adapter reads: no new adapter, and the temporal models get their 19 history frames.

## Install (Windows, done on this PC on 2026-10-01)

| What | Where |
|---|---|
| CARLA 0.9.16 packaged release (Unreal Engine 4.26; no Unreal Editor needed) | `C:\CARLA\CARLA_0.9.16` |
| Python 3.12 client (`carla==0.9.16`, numpy, opencv, pillow) | `C:\CARLA\venv312` |

The package is the Windows zip from <https://github.com/carla-simulator/carla/releases> (7.8 GB,
about 20 GB unpacked). The base package has Town01–05 and Town10HD; `AdditionalMaps_0.9.16.zip`
(7.3 GB, unpacked into the same folder) adds Town06/07 and others. CARLA 0.10 (Unreal Engine 5)
asks for 16 GB of video memory; on a 6 GB RTX 3050 use 0.9.16 with `-quality-level=Low`.

The CARLA client supports Python 3.10–3.12 only, hence the separate venv; the TAC-UFLD venv
(Python 3.14) is not used for collection.

## Collect

```powershell
# 1. server (no window; Low quality fits 6 GB of video memory)
C:\CARLA\CARLA_0.9.16\CarlaUE4.exe -RenderOffScreen -quality-level=Low

# 2. clips: training towns and a different test town (scene-disjoint test)
C:\CARLA\venv312\Scripts\python.exe scripts\carla_collect.py --out D:\carla_lanes --town Town04 --clips 200
C:\CARLA\venv312\Scripts\python.exe scripts\carla_collect.py --out D:\carla_lanes --town Town10HD --clips 200 --seed 2
C:\CARLA\venv312\Scripts\python.exe scripts\carla_collect.py --out D:\carla_lanes --town Town05 --clips 100 --split test
```

Each clip is 20 frames at 20 Hz (1 s, as in TuSimple), 1280×720; only frame 20 is labelled.
Weather cycles through 12 presets (noon, sunset, night; dry, wet, rain). `--traffic` sets the
number of other vehicles (default 40).

## Use with TAC-UFLD

```powershell
$env:TUSIMPLE_ROOT = "D:\carla_lanes"
python -m tac_ufld validate-dataset --dataset tusimple   # overlays: check the labels by eye first
```

then enable `tusimple` in `configs/datasets.yaml` (or copy `configs/tusimple.yaml` to a
`carla.yaml` with its own output folder) and run as for the other datasets.

## Design

| Part | Responsibility | Tested |
|---|---|---|
| `CameraModel`, `sample_rows`, `occluded_fraction`, `usable_lanes` | pinhole projection, TuSimple row sampling, occlusion share: pure NumPy, no CARLA import | `tests/test_carla_collect.py` |
| `lane_boundaries` | lane edges from the CARLA road map (waypoints) | on the simulator |
| `synchronous_world` | synchronous mode at 20 Hz; restores the previous settings on exit, even after an error | on the simulator |
| `Scene` | ego car, NPC traffic, RGB and semantic cameras; destroys every actor it spawned, also when a spawn fails | on the simulator |
| `ClipWriter` | TuSimple file layout, label and meta lines, clip numbering across runs | `tests/test_carla_collect.py` |
| `collect_clips`, `main` | compose the parts | on the simulator |

The script was smoke-tested on 2026-10-02 (3 clips, labels checked on overlays) and refactored on 2026-10-04
into the parts above (same output format). **Re-run the 3-clip smoke test before the first real collection**:
the refactored simulator code has not run against CARLA yet (the GPU was busy with OpenLane).

## What the labels are (and are not)

- Geometric ground truth from the CARLA map: the boundaries of the ego lane and the outer
  boundaries of the neighbouring lanes in the same direction (up to 4), projected with the exact
  camera pose and sampled at the TuSimple rows (`h_samples` 160…710). They are not painted-marking
  detections: where a road has no painted line, the boundary is still labelled.
- Boundaries hidden behind vehicles are labelled, like TuSimple. `<split>_meta.jsonl` stores, per
  clip, `occluded_fraction` (share of labelled points covered by a vehicle or person in the
  semantic camera), weather, town and speed, to build an occlusion subset.
- Synthetic images: results on CARLA alone are not evidence for real roads. Use it for
  controlled occlusion/weather experiments and pre-training; report real-data results (ELAS,
  OpenLane) separately.

## Closed-loop lane keeping (LDW / LKA), added 2026-10-19

The lane model now drives the car. Each simulation frame works like this:

1. The front camera image goes to the detector.
2. `tac_ufld.ego` (`EgoLaneTracker` + `lane_keeping`) turns the lanes into an LDW decision and an LKA steering
   request. The turn signal is read back from the car's light state, as a car reads it from CAN.
3. A scripted "driver" decides who steers. The LKA steers, or the driver does during a scripted drift or a
   signalled lane change.
4. A PI controller holds 60 km/h.
5. CARLA's road map gives ground truth: lateral offset from the lane centre, lane width, heading error and
   curvature. It is logged next to every output of the system.

Three detector modes use the same loop:

| mode | lanes that steer | what it measures |
|---|---|---|
| `oracle` | CARLA map ego-lane edges projected into the image (perfect perception) | the controller and LDW alone; used to tune the gains |
| `model` (closed loop) | the trained model | the whole system |
| `model --shadow` | ground truth; the model rides along and its lanes, offset and LDW are logged | perception and warnings, without a failing loop ending the run |

### Run

The CARLA client needs Python 3.10–3.12 and the model needs the project venv (Python 3.14 + torch). The script runs
in `C:\CARLA\venv312` and starts the detector as a socket server in `..\.venv` by itself
(`src/tac_ufld/sim/detector_bridge.py`: JSON plus raw RGB bytes, no pickle). With `--detector-addr HOST:PORT`
the server can also run on another machine, such as a Jetson.

```powershell
C:\CARLA\CARLA_0.9.16\CarlaUE4.exe -RenderOffScreen -quality-level=Low      # restart it before each town, see below
$py = "C:\CARLA\venv312\Scripts\python.exe"
& $py scripts\carla_lka.py run --scenario straight lead drift lane_change --detector oracle --label oracle --out results\carla_lka\final
& $py scripts\carla_lka.py run --scenario straight lead drift lane_change --detector model --device cuda `
    --checkpoint results\review_lite_tsm_pilot\seed_1\checkpoints\lite_tsm.pt --label lite_tsm --out results\carla_lka\final
& $py scripts\carla_lka.py run ... --shadow --label lite_tsm_shadow                     # same, the model only observes
& $py scripts\carla_lka.py sweep --scenario straight curves --k-offset 0.3 0.6 1.0 --k-heading 0 0.5 1.0 --out results\carla_lka\sweep_oracle
& $py scripts\carla_lka.py summarize results\carla_lka\final                             # metrics.csv again from the frame logs
```

Each run writes `<tag>.csv` (one row per frame), `<tag>.json` (configuration, start point, detector, metrics) and,
with `--snapshot-every N`, overlay frames in `frames/`. `metrics.csv` has one row per run. Useful options:
`--k-offset`, `--k-heading`, `--max-steer`, `--warn-tlc`, `--speed-kmh`, `--fps`, `--weather`, `--seed`, `--town`,
camera `--fov --cam-x --cam-z --horizon`, `--video`.

**This PC (16 GB RAM, RTX 3050 6 GB):** full Town06 crashes the server when the map is switched. Town06_Opt
without optional layers (`MapLayer.NONE`; roads and markings always load) loads in 15 s, so it is the `curves`
default (`--map-layers`). It works only as the first map after a server start: switching to it from full Town04
also crashed. Restart CARLA between towns. The script orders scenarios by town.

### Scenarios

| name | town | what happens | feeds |
|---|---|---|---|
| `straight` | Town04 | straightest 0.7 km found (max curvature 1/1540 m); LKA only | lateral RMS, crossings, false LDW |
| `curves` | Town06_Opt | route turning about 40°, radius >= 100 m, straight warm-up | the same on curves |
| `lead` | Town04 | `straight` with a truck 15 m ahead in the same lane (gap held at 9.5-16.8 m) | occlusion of the far markings |
| `drift` | Town04 | 4 scripted inattentive drifts at 0.4 m/s. Two with the LKA off (LDW test). Two released 0.5 m before the line with the LKA on (Euro NCAP LKA-style test). Each starts from the lane centre | LDW lead time and misses, LKA excursion |
| `lane_change` | Town04 | left then right lane change, turn signal on 1 s before | suppression, false LDW, lane-change detection |

Common settings:

- **Car and camera.** Lincoln MKZ 2020 at 60 km/h, 30 Hz (the ELAS rate, so the temporal models see the history
  spacing they were trained with). Camera 640x480, FOV 70°, 2.0 m ahead of the car's centre, 1.3 m high, pitched
  so the horizon sits at 0.6 H. Together this reproduces the ELAS view: lane about 520 px wide at the bottom row,
  no hood in view.
- **Tracker and gains.** Tracker rows at 0.95 H and 0.72 H (3.7 m and 10.6 m ahead of the camera). Gains
  `k_offset 0.6`, `k_heading 0`, `max_steer 0.3`, LDW `warn_tlc_s 1.0` and `warn_margin 0.05`.
  `half_width = 0.92 m / 3.5 m`, from the car's bounding box.
- **Disturbance.** CARLA is deterministic: a car spawned centred on a straight road never leaves the centre. Every
  run therefore adds a seeded random steering disturbance (Ornstein-Uhlenbeck, sigma 0.003 normalised steer,
  tau 2 s). It is the same for every detector.

### Metrics (`src/tac_ufld/sim/lka_metrics.py`)

- **Line crossing:** the body overlaps a boundary, `|offset| + half_width > lane_width / 2`. One event per
  overlap, so a full lane change counts once. A crossing is LKA-controlled when the LKA was steering and the turn
  signal was not on toward that side.
- **Rates:** crossings per km and LDW warnings per km (events, gaps under 0.3 s merged).
- **LDW lead time:** crossing onset minus warning onset, for the drifts with the LKA off. Missed means no warning
  on that side from 4 s before to 1 s after the crossing.
- **Time to first departure:** time after the 3 s warm-up of the first LKA-controlled crossing.
- **Lateral deviation RMS:** over `keep` phases only.
- **LKA excursion:** the maximum of `|offset| + half_width - lane_width / 2` after a release; < 0 means the body
  stayed inside.
- **Perception:** `offset_mae_lw`, the model's offset at the reference row against the projected ground truth.

### Results (2026-10-19, seed 1, ClearNoon)

Results CSV: `results/carla_lka/final/metrics.csv` (25 runs). Gain sweeps: `results/carla_lka/sweep_oracle/metrics.csv`
(27 runs) and `results/carla_lka/sweep_lite_tsm/metrics.csv` (6). Checkpoints: `lite_tsm` from
`review_lite_tsm_pilot` and `lite_v05` from `review_lite_aligned_pilot` (both trained on ELAS).

Closed loop:

| detector | scenario | km | lateral RMS (m) | LKA crossings / km | first departure (s) | LDW / km | LDW lead (s) | LKA excursion (m) | end |
|---|---|---|---|---|---|---|---|---|---|
| oracle | straight | 0.67 | 0.021 | 0 | - | 0 | - | - | completed |
| oracle | curves | 0.67 | 0.030 | 0 | - | 1.5 | - | - | completed |
| oracle | lead | 0.67 | 0.021 | 0 | - | 0 | - | - | completed |
| oracle | drift | 1.07 | 0.033 | 0 | - | 12.2 | 1.15 (2/2 warned) | -0.44 | completed |
| oracle | lane_change | 0.60 | 0.047 | 0 | - | 5.0 | - | - | completed (2/2 changes) |
| lite_v05 | straight | 0.67 | 0.615 | 1.5 | 0.9 | 4.5 | - | - | completed |
| lite_v05 | lead | 0.67 | 0.596 | 3.0 | 0.9 | 9.0 | - | - | completed |
| lite_v05 | curves | 0.08 | 1.07 | 38.5 | 1.3 | 38.5 | - | - | hit the sidewalk at 4.7 s |
| lite_v05 | drift | 0.18 | 1.10 | 16.5 | 1.4 | 16.5 | - | - | hit a wall at 10.9 s |
| lite_v05 | lane_change | 0.09 | 1.06 | 35.1 | 1.5 | 23.4 | - | - | hit the guardrail at 5.2 s |
| lite_tsm | straight | 0.54 | 0.98 | 39.1 | 0.6 | 74.4 | - | - | hit the guardrail at 32.2 s |
| lite_tsm | lead | 0.38 | 1.50 | 15.8 | 0.6 | 50.1 | - | - | hit the guardrail at 22.8 s |
| lite_tsm | curves | 0.05 | 1.06 | 43.8 | 0.8 | 87.5 | - | - | hit the sidewalk at 2.8 s |
| lite_tsm | drift | 0.03 | 1.24 | 59.7 | 0.8 | 59.7 | - | - | hit the guardrail at 2.0 s |
| lite_tsm | lane_change | 0.04 | 1.10 | 48.7 | 0.8 | 97.4 | - | - | hit the guardrail at 2.5 s |

Shadow mode (ground truth steers, so the car's path equals the oracle row; the model only observes):

| detector | scenario | offset MAE (lane widths) | offset correlation r with ground truth | LDW / km (car centred: false warnings) | drift crossings warned | LDW during lane change |
|---|---|---|---|---|---|---|
| lite_v05 | straight / lead / curves | 0.081 / 0.076 / 0.019 | 0.04 / 0.14 / -0.22 | 34.5 / 37.5 / 0 | - | - |
| lite_v05 | drift / lane_change | 0.040 / 0.073 | -0.06 / -0.03 | 0 / 0 | 0 of 2 | 0 |
| lite_tsm | straight / lead / curves | 0.156 / 0.152 / 0.050 | 0.10 / 0.13 / 0.00 | 34.5 / 45.0 / 123 | - | - |
| lite_tsm | drift / lane_change | 0.055 / 0.093 | 0.01 / 0.13 | 4.7 / 26.7 | 1 of 2 (0.53 s ahead) | 0 |

Oracle gain sweep: lateral RMS in cm, straight (Town04) / curves (Town06). With the LKA released toward the line,
excursion was -0.34 to -0.47 m for every pair (the body stayed inside), and LDW lead time was 1.15 s for every
pair.

| k_offset | k_heading 0 | 0.5 | 1.0 |
|---|---|---|---|
| 0.3 | 4.4 / 8.2 | 4.0 / 8.4 | 3.9 / 20.4 |
| 0.6 | 2.1 / 3.0 | 2.1 / 5.6 | 2.2 / 12.1 |
| 1.0 | 1.3 / 1.4 | 1.3 / 4.4 | 1.4 / 8.5 |

`lite_tsm` closed-loop sweep on `straight` (`k_offset` 0.3/0.6/1.0, `k_heading` 0/0.5): every pair left the lane
within 0.5-2.1 s and hit the barrier or terrain within 19-30 s. Gains are not the problem.

### Findings

1. **The loop, the controller and the LDW work with perfect lanes.**
   - Lateral RMS is 2-5 cm, with no LKA crossing in any scenario.
   - LDW warns 1.13-1.15 s before both drift crossings and misses none (wheel-based TLC, ego.py 2026-10-19).
   - Released drifts stop 0.44 m inside the line.
   - During lane changes the turn signal suppresses LKA and LDW on 99.4 % of frames (the remaining frame is the
     one-frame light-state delay), and `LaneChangeDetector` finds both changes.
2. **Both ELAS-trained light models fail in CARLA, from perception rather than control.**
   - In shadow mode their predicted offset does not follow the car (|r| <= 0.22, slope about 0 in the drift and
     lane-change runs).
   - They draw a lane 134-300 px wide where the real one is about 450 px at the reference row: a near-constant
     prior, not detected markings.
   - In closed loop they leave the lane within 0.6-1.5 s. `lite_v05` survives the straight and lead runs only
     because the road is straight (it holds a constant 0.6 m bias). Shadow-mode LDW gives up to 123 false
     warnings per km with the car centred (35-45 on the straight and lead runs) and catches 0-1 of 2 real
     departures.
   - The camera geometry matches ELAS and the bridge gives correct lanes on an ELAS frame. The cause is the
     appearance gap: CARLA's light low-contrast asphalt and markings at Low quality, against ELAS's dark roads.
   - This says nothing about the models on real roads (see above). For a closed-loop test, fine-tune on CARLA clips
     or use another simulator with realistic road textures. Shadow mode is the quick check: offset correlation and
     MAE against ground truth. **Done on 2026-10-19:** see "Closing the ELAS -> CARLA perception gap" below; both
     models then complete every scenario.
3. **`k_heading > 0` makes curves worse.** The vanishing-point heading cue (ego.py 2026-10-19) has no bias on
   straights, where it makes no difference. On curves it adds a curvature term: measured +7 to +10 lane widths
   per 1/m, against about 7.6 from the geometry, kappa·(d_ref+d_far)/2·d_ref/W. With `k_heading = 1` the car settles
   toward the inside, about 16 m per 1/m (0.13 m at R 125 m). Default `k_heading = 0`: the look-ahead offset at
   the reference row already damps the loop. A curvature-free heading needs a third row (quadratic fit).
4. **LDW chatter.**
   - After each lane change, once the signal goes off, 3 warnings alternate sides while the LKA settles in the new
     lane. Production LDW holds off for 1-2 s after the signal; `lane_keeping` does not.
   - The same happens while the LKA brings the car back after a drift: at `k_offset 0.6` it returns fast enough
     for the time to the opposite line to fall under 1 s. The drift run has 8 / 13 / 31 LDW events at
     `k_offset` 0.3 / 0.6 / 1.0 (oracle, `k_heading` 0), all after a crossing or a release. A steering-rate limit
     or a lower gain far from the line would help.

Limits:

- One seed, one weather and one speed.
- Scripted scenarios, not ScenarioRunner/OpenSCENARIO (the next step for Euro NCAP-style repeatability).
- Synchronous mode, so detector latency (18-30 ms on the RTX 3050) is not added to the loop.
- The car body's half-width stands in for the wheel track in the crossing test.

### Design

| Part | Responsibility | Tested |
|---|---|---|
| `lka_loop.CameraRig`, `image_polyline`, `signed_offset`, `pure_pursuit_steer`, `heading_hold_steer`, `curvature_steer`, `SpeedController`, `SteeringDisturbance`, `route_*` | camera geometry, driver laws, route shape: no CARLA | `tests/test_carla_lka.py` |
| `lka_loop.ScenarioDirector` | phases: warm-up, keep, align, drifts, release, lane change | `tests/test_carla_lka.py` |
| kinematic closed loop (test only) | bicycle model, `CameraRig`, `EgoLaneTracker`, `lane_keeping`: steering signs, curve holding, LDW before the wheel crosses | `tests/test_carla_lka.py` |
| `lka_metrics` | frame-log schema, CSV, all metrics | `tests/test_carla_lka.py` |
| `detector_bridge` | socket protocol, model server (project venv), client | `tests/test_carla_lka.py` (fake model); real model checked by hand |
| `lka_loop.synchronous_world`, `pick_start`, `Rig`, `run_scenario` | town, start point, actors (always destroyed), the loop | on the simulator (runs above) |
| `scripts/carla_lka.py` | `run` / `sweep` / `serve` / `summarize` | on the simulator |

## Closing the ELAS -> CARLA perception gap (fine-tuning), added 2026-10-19

The closed loop above failed because the ELAS-trained models do not see CARLA's roads. Fine-tuning them on CARLA
lane data, kept away from the closed-loop test towns, closes most of that gap.

### Data (`scripts/carla_collect_elas.py`)

The data is in the **ELAS layout**, not the TuSimple one of `carla_collect.py`:

- **Why not TuSimple.** The light models have the ELAS head: 2 slots (ego-left, ego-right), 640x480 frames, row
  anchors over 59-99.5 % of the height. The TuSimple adapter has 4 slots, requires 1280x720 and labels only frame
  20 of each 20-frame clip.
- **Scenes.** Each scene is a continuous 30 Hz drive. Every frame is saved (the history frames, as in ELAS), and
  every 3rd frame gets a label.
- **Labels.** p1..p4 at rows 320/360/400/480 (about 3-19 m ahead), from the CARLA map projected with the exact
  camera pose. Only painted boundaries are labelled: a side whose `LaneMarkingType` is NONE, Grass, Curb or Other
  is written as absent. Frames on or just before a junction are not labelled, and neither are frames below 3 m/s.
- **Camera.** The closed-loop camera, with a random change per scene: FOV +-4°, height +-0.15 m, horizon
  +-0.03 H, yaw +-1°.
- **Driving.** The Traffic Manager's autopilot drives, with a random lateral offset in the lane (+-0.7 m, redrawn
  every 4 s) and occasional lane changes. 25 other cars, and 12 weathers cycled per scene.
- **Checks.** Overlays in `<root>/_overlays/`; the project's own label-check images came from
  `scripts/carla_finetune.py --dry-run`.

| town | role | scenes | labelled frames |
|---|---|---|---|
| Town01 | train | 4 | 937 |
| Town02 | train | 2 | 234 |
| Town03_Opt (no buildings) | train | 3 | 242 |
| Town10HD_Opt | train | 11 | 1115 |
| Town05 | **validation** (held-out town: early stopping, post-processing tuning) | 2 | 386 |
| Town07 | **test** (collected 2026-10-20, never trained or selected on) | 8 | 587 |
| Town04, Town06 | **closed-loop test only** (refused by both scripts) | - | - |

Saved under `C:\CARLA\lane_data\carla_elas` (1.9 GB, outside the repo; `CARLA_LANES_ROOT` overrides the path).

Collection notes for this PC:

- The server crashed when loading full Town03 (as with Town06), and twice during a scene while another GPU job was
  running.
- Scenes ending in traffic queues are cut after 5 s at rest (`--stuck-s`).
- Urban towns lose 50-80 % of the candidate labels to junctions and unpainted edges; that is by design.

### Fine-tuning (`scripts/carla_finetune.py`, `configs/carla_finetune.yaml`)

```powershell
python scripts\carla_finetune.py --config configs\carla_finetune.yaml --variants lite_tsm lite_v05   # project venv
```

- **Start point.** Each model starts from its own ELAS checkpoint. `ExperimentRunner.train_variant` is called
  with `warm_from` = that checkpoint: identical state-dict keys, a strict full load. `lite_v05` therefore keeps its
  trained warp fusion. A plain `tac_ufld run` would first train a fresh `lite_baseline` and warm-start from it
  instead.
- **Configuration.** The checkpoint's own: architecture, preprocessing, anchors and augmentation. Only epochs
  (10, early stopping patience 3), learning rate (2e-4, half the ELAS rate) and the data change.
- **Splits.**
  - train: 3,498 frames = the CARLA training towns + every 3rd frame of 4 ELAS training scenes (970 frames, to
    keep the real-road skills);
  - validation: Town05, 386 frames;
  - ELAS test: the checkpoint's held-out scenes BR_S02, VIX_S05 and VV_S03 (2,610 frames), to check for
    forgetting.
- **Scoring.** The original ("before") and the fine-tuned ("after") checkpoint are scored the same way, with
  post-processing tuned on Town05 for both. `<variant>_tuned.json` sits next to the new checkpoint, so the closed
  loop uses it.
- **Training runs.** `lite_tsm`: best epoch 5 of 8, about 3 min per epoch. `lite_v05`: best epoch 5 of 8, about
  11 min per epoch (it encodes every history frame). A first try with 4 epochs at 1e-4 stopped while validation F1
  was still rising (0.03 -> 0.22).

Offline results (`results/carla_finetune_<variant>/carla_finetune.csv`).

**Caveat: the Town05 number is a validation F1, not a test F1.**

- In this run the script's `seen_test` split *is* the validation set: the same 386 frames of
  `CARLA_Town05_s5_00` (278) and `CARLA_Town05_s5_01` (108). Those frames chose the best epoch (5) and the
  post-processing.
- The "after" value (0.282 / 0.296) is therefore the selected maximum (`lite_tsm`: val F1 0.2817 at epoch 5) and
  is optimistic. The "before" value used no selection beyond post-processing tuning, so the gap is overstated by
  an unknown amount.
- The held-out evidence is the closed loop in Town04/Town06 below.
- The CSV labels the split `carla_val`. The proper test number is the held-out Town07 test below: lane F1 0.12
  -> **0.50** (`lite_tsm`) and 0.16 -> **0.46** (`lite_v05`).

**Protocol of the lane F1** (identical to the ELAS protocol, `tac_ufld.evaluation.evaluator`):

- CULane-style lane F1: every lane drawn as a 30 px line scaled to the image width (30·640/1640 = 12 px at 640),
  Hungarian matching, a match when IoU >= 0.5 (`lane_f1_iou50`).
- Scored inside the scene's labelled band (`valid_y_range` = ROI) on every labelled frame: 386 Town05 frames,
  both ego boundaries where painted.
- Same post-processing sweep as the ELAS runs, tuned on the validation frames.

It is the same metric as the ELAS F1, but **not comparable in difficulty**:

- the CARLA band starts at row 320 (about 19 m ahead) against 340 in ELAS (about 12 m), and far points are where
  12 px lines miss;
- half of Town05 is heavy rain with a black-sky rendering artefact at Low quality, and the other half a sharp
  curve;
- 386 frames from 2 scenes.

At IoU 0.5 with 12 px lines, a lane off by about 6 px over its length is a miss, so the anchor error (17-24 px
after fine-tuning) explains the low F1. For lane keeping, the closed-loop and shadow metrics below matter more
than this F1.

| model | split | lane F1 before -> after | anchor error (px) | lane-centre error (px) |
|---|---|---|---|---|
| lite_tsm | CARLA Town05, **validation** (selected) | 0.026 -> 0.282 | 96.6 -> 24.2 | 52.5 -> 6.9 |
| lite_tsm | ELAS test scenes (held out) | 0.911 -> **0.939** | 2.51 -> 2.41 | 3.74 -> 3.15 |
| lite_v05 | CARLA Town05, **validation** (selected) | 0.034 -> 0.296 | 81.2 -> 17.2 | 46.9 -> 14.0 |
| lite_v05 | ELAS test scenes (held out) | 0.925 -> **0.930** | 2.33 -> 2.65 | 3.32 -> 4.33 |

Mixing in ELAS frames prevented forgetting: ELAS test F1 (held out, the checkpoints' own test scenes) went up
slightly for both models, probably from more varied training data.

### Closed loop before -> after fine-tuning (Town04 / Town06: never seen in fine-tuning)

Same scenarios, seed, gains (`k_offset 0.6`, `k_heading 0`), disturbance and weather as the earlier results. Runs
are tagged `*_carla` in `results/carla_lka/final/metrics.csv` (55 runs with the UFLD rows below; `model_run`
names the checkpoint's run).

| model | scenario | lateral RMS (m) | LKA crossings / km | first departure (s) | end |
|---|---|---|---|---|---|
| lite_tsm | straight | 0.98 -> **0.12** | 39.1 -> 1.5 | 0.6 -> 38.7 | hit the guardrail at 32 s -> completed |
| lite_tsm | curves | 1.06 -> **0.37** | 43.8 -> 3.0 | 0.8 -> 17.6 | hit the sidewalk at 2.8 s -> completed |
| lite_tsm | lead | 1.50 -> **0.14** | 15.8 -> 1.5 | 0.6 -> 38.5 | hit the guardrail at 23 s -> completed |
| lite_tsm | drift | 1.24 -> **0.04** | 59.7 -> 0.9 | 0.8 -> 11.2 | hit the guardrail at 2.0 s -> completed |
| lite_tsm | lane_change | 1.10 -> **0.06** | 48.7 -> 0 | 0.8 -> none | hit the guardrail at 2.5 s -> completed |
| lite_v05 | straight | 0.62 -> **0.16** | 1.5 -> 1.5 | 0.9 -> 38.4 | completed -> completed |
| lite_v05 | curves | 1.07 -> **0.43** | 38.5 -> 1.5 | 1.3 -> 17.9 | hit the sidewalk at 4.7 s -> completed |
| lite_v05 | lead | 0.60 -> **0.21** | 3.0 -> 1.5 | 0.9 -> 38.2 | completed -> completed |
| lite_v05 | drift | 1.10 -> **0.09** | 16.5 -> 0.9 | 1.4 -> 23.7 | hit a wall at 10.9 s -> completed |
| lite_v05 | lane_change | 1.06 -> **0.07** | 35.1 -> 0 | 1.5 -> none | hit the guardrail at 5.2 s -> completed |
| oracle (reference) | the 5 above | 0.021 / 0.030 / 0.021 / 0.033 / 0.047 | 0 | none | completed |

In the drift test (closed loop) the fine-tuned models warn 0.82 s (`lite_tsm`) and 0.65 s (`lite_v05`) before the
LDW drift crossings (oracle 1.15 s). After a release toward the line they stop the car 0.45 m inside it, as the
oracle does.

Shadow mode (ground truth steers; perception and warnings of the model):

| model | offset MAE straight / lead (lane widths) | false LDW / km, straight / lead / curves | drift crossings warned (lead time) | offset correlation r in the drift run | lane changes detected |
|---|---|---|---|---|---|
| lite_tsm | 0.156 / 0.152 -> **0.021 / 0.019** | 34.5 / 45.0 / 123 -> **4.5 / 4.5 / 12.0** | 1 of 2 (0.53 s) -> **2 of 2 (0.55 s)** | 0.01 -> 0.16 | 0 -> **2 of 2** |
| lite_v05 | 0.081 / 0.076 -> **0.012 / 0.013** | 34.5 / 37.5 / 0 -> **1.5 / 1.5 / 13.5** | 0 of 2 -> **2 of 2 (0.85 s)** | -0.06 -> **0.70** | 0 -> **2 of 2** |

(`lite_v05` had no curve warnings before because its offset did not move at all.)

**UFLD v1 baseline for reference** (ResNet-18, ImageNet-pretrained, trained on ELAS only: no CARLA data;
`results/elas_pilot_v2/seed_1/checkpoints/ufld_baseline.pt`, tags `*__ufld_baseline*`):

| scenario | closed loop: lateral RMS (m) / LKA crossings per km / end | shadow: offset MAE (lane widths) / false LDW per km |
|---|---|---|
| straight | **0.065** / 0 / completed | 0.009 / 3.0 |
| lead | 0.149 / 1.5 / completed | 0.014 / 28.5 |
| drift | 0.037 / 0 / completed; LDW 0.97 s ahead, 2 of 2 warned; release stops 0.44 m inside the line | 0.013 (offset r 0.97) |
| lane_change | 0.053 / 0 / completed; both lane changes detected | 0.045 / 5.0 |
| curves (Town06_Opt) | 1.86 / 7.5 / **left the road at 23.9 s** | 0.012 / 10.5 |

The full-size baseline sees CARLA lanes well without any CARLA data: best offset error of all models, better
straight-road RMS than the fine-tuned light models (0.065 vs 0.12-0.16 m). It still loses the road on the
Town06_Opt curve, where both fine-tuned light models stay on (0.37 / 0.43 m). So the light models' domain gap
comes from the light models themselves, not from the CARLA images. Possible causes, not separated here: the small
backbone's capacity, and pretraining. The light backbone is trained from scratch on ELAS (`models/lite.py`); the
ResNet-18 starts from ImageNet. This is relevant for the article's
light-vs-baseline comparison (robustness to a domain shift), and the demo video shows it frame by frame.

Reading:

- Fine-tuning on about 2,500 CARLA frames from other towns turns two models that could not stay on the road into
  usable ones in an unseen town.
- `lite_tsm` steers better: lower RMS in 5 of 5 scenarios.
- `lite_v05` perceives the lateral position better: offset MAE, drift correlation 0.70 vs 0.16, LDW lead time.
  Its offset is smoother but lags, which costs RMS in closed loop.
- **Remaining gap to the oracle:**
  - lateral RMS is 5-10x the oracle's on straights and 12-15x on curves, with 1-3 crossings per km where the
    oracle has none;
  - LDW lead time is 0.55-0.85 s against 1.15 s;
  - the curves (Town06_Opt, empty scenery, R about 125 m) are the weakest case.
- **Next steps:**
  - more highway-like training data: Town05's highway is the validation town, and full Town03/Town06 cannot be
    loaded on this PC;
  - multiple seeds;
  - a lower LKA gain for detectors with this much offset noise.

Run-to-run noise: a repeated shadow run of the original `lite_tsm` (`results/carla_lka/recheck`) gives the same
lateral RMS (0.0214 m) and offset MAE (0.156 vs 0.161), but 23 vs 31 LDW events. CARLA physics and GPU inference
are not bit-exact, and a jittery detector flickers around the LDW threshold, so treat LDW counts as about +-30 %.

### Held-out CARLA test: Town07 (2026-10-20)

**The test set.** Town07 is used nowhere else: not trained on, not selected on, not a closed-loop town.

- Rural two-lane roads with curves, double yellow centre lines and white edge lines.
- Collected with `scripts/carla_collect_elas.py --town Town07 --seed 9 --scenes 8 --target-labelled 600`: 587
  labelled frames in 8 scenes. Weathers run from noon to night (ClearNight, WetNight); 25 other cars. The full
  map loaded fine on this PC.
- Scored with `scripts/carla_finetune.py --evaluate-only` (`carla.test_towns: [Town07]`), with no retraining.
  Post-processing was tuned on the Town05 validation frames only, and the checkpoints are the ones selected on
  Town05.
- Same lane-F1 protocol as above. Files: `results/carla_finetune_<variant>/carla_finetune_heldout.csv` and
  `heldout/`.

| model | lane F1 (IoU 0.5) | precision / recall | lane F1 (IoU 0.35) | anchor error (px) | lane-centre error (px) |
|---|---|---|---|---|---|
| lite_tsm, ELAS checkpoint | 0.118 | 0.112 / 0.126 | 0.172 | 52.7 | 37.8 |
| **lite_tsm, fine-tuned on CARLA** | **0.500** | 0.489 / 0.510 | **0.694** | 8.8 | **10.8** |
| lite_v05, ELAS checkpoint | 0.159 | 0.150 / 0.169 | 0.207 | 44.7 | 34.8 |
| **lite_v05, fine-tuned on CARLA** | 0.458 | 0.456 / 0.460 | 0.663 | **6.9** | 14.2 |
| UFLD v1 baseline, ELAS only (reference, not fine-tuned) | 0.410 | 0.387 / 0.437 | 0.560 | 20.7 | 22.9 |

Reading:

- This is the **test** number the Town05 validation F1 could not give. Fine-tuning raises held-out lane F1 from
  0.12-0.16 to 0.46-0.50, and the anchor error falls 5-6x (to 7-9 px).
- The held-out F1 is higher than the selected Town05 validation F1 (0.28-0.30): Town05 was the harder set (heavy
  rain, a sharp curve). Lane F1 values are set-dependent, so compare models on the same set only.
- On this unseen town both fine-tuned light models beat the ELAS-only UFLD v1 baseline offline: F1 0.50 / 0.46 vs
  0.41, anchor error 9 / 7 px vs 21 px. In the closed loop the baseline centred better on the Town04 straight;
  together, fine-tuning lets the light models match or pass the full-size baseline in a new domain.
- The lane-F1 caveat above still holds: 12 px lines at IoU 0.5 are strict, so the IoU 0.35 column and the pixel
  errors are useful companions.
- One seed and one test town; night scenes are included.
- The ELAS pilot 3 checkpoints (`results/elas_pilot_v3`, `lite_tsm` 0.943 on the ELAS test) were not fine-tuned
  or scored here. All "before" rows are the checkpoints the fine-tuning started from.

### Design (fine-tuning and viewer)

| Part | Responsibility | Checked |
|---|---|---|
| `scripts/carla_collect_elas.py` | ELAS-layout scenes, painted-boundary labels, camera jitter, stuck/junction rules, test towns refused | overlays and `--dry-run` label check (above) |
| `scripts/carla_finetune.py` | splits from the config (train / validation / optional test towns, disjointness enforced), fine-tune from the checkpoint itself, before/after scoring, `--dry-run`, `--evaluate-only` (with `reference_checkpoints` scored as they are) | the runs above; the Town07 test |
| `configs/carla_finetune.yaml` | checkpoints, towns, ELAS mix, schedule overrides | - |
| `scripts/carla_drive.py` | live window / H.264 recording, model panels and HUD, manual / autopilot / LKA driving, watchdog for a stuck client | offline (CPU models, synthetic frames), window path with SDL's dummy driver, the demo recording |

## How to open CARLA and watch the models / Como abrir o CARLA e ver os modelos

### Português

`scripts/carla_drive.py` abre uma janela com um painel por fonte, lado a lado:

- a câmera;
- o ground truth do CARLA (faixas do mapa projetadas na imagem);
- um painel por modelo (`--checkpoint`, repetível), com as faixas do modelo, o GT em verde fino e o HUD (offset,
  troca de faixa, LDW, comando de direção do LKA, latência).

Ele inicia o servidor CARLA se nenhum estiver aberto (fora da tela, qualidade baixa).

1. **Antes:** pause a fila da GPU (atalho *Fila GPU - Pausar*). A placa tem 6 GB: o CARLA usa 3-4 GB e cada
   modelo 0,3-1 GB. Feche o Chrome com muitas abas, o LM Studio e o Docker, se não forem usados (o CARLA cai sem
   memória). Um job de GPU por vez.
2. **Rodar** (PowerShell, na pasta do projeto):

   ```powershell
   $py = "C:\CARLA\venv312\Scripts\python.exe"
   # dirigir você mesmo, com 2 modelos (UFLD v1 e o lite v0.5 ajustado no CARLA)
   & $py scripts\carla_drive.py --town Town04 --drive manual `
       --checkpoint results\elas_pilot_v2\seed_1\checkpoints\ufld_baseline.pt `
       --checkpoint results\carla_finetune_lite_v05\seed_1\checkpoints\lite_v05.pt
   # o piloto automático dirige, com trânsito; L passa a direção para o modelo selecionado
   & $py scripts\carla_drive.py --town Town04 --drive autopilot --traffic 30 --checkpoint ... --checkpoint ...
   # o modelo 1 dirige desde o início (controle de velocidade a 60 km/h)
   & $py scripts\carla_drive.py --town Town04 --drive lka --checkpoint ...
   # gravar um vídeo H.264 sem janela (60 s simulados)
   & $py scripts\carla_drive.py --town Town04 --drive autopilot --traffic 30 --duration-s 60 --no-window `
       --record results\carla_lka\demo.mp4 --checkpoint ... --checkpoint ...
   ```

3. **Teclas:** `W`/`S` acelera/freia, `A`/`D` vira, `Q`/`E` seta esquerda/direita (liga/desliga), `L` liga/desliga
   o LKA (o modelo escolhido assume a direção), `1`..`9` escolhe o modelo que dirige, `G` mostra/esconde o GT nos
   painéis dos modelos, `ESC` sai.
4. **Mapas:** Town04 (rodovia) abre normalmente. Town06 e Town03 completos derrubam o servidor neste PC: use
   `--map-layers minimal` (mapa `_Opt` sem prédios). Se trocar de mapa, prefira reabrir o servidor (feche o
   `CarlaUE4`).
5. **Notas:** a simulação é síncrona (cada quadro espera todos os modelos), por isso com vários modelos ela roda
   mais devagar que o tempo real: 3 modelos rodam a cerca de 4 quadros/s, e 60 s simulados levam cerca de
   7 min. O vídeo gravado toca em tempo simulado. Town04 e Town06 nunca foram usados no ajuste fino: são
   cidades inéditas para todos os modelos.
6. **Vídeo de demonstração:** `results/carla_lka/demo_town04_autopilot.mp4`. São 60 s em Town04, com piloto
   automático e 30 carros. Painéis: câmera | ground truth | UFLD v1 baseline (ELAS) | lite_tsm (ELAS) |
   lite_tsm ajustado no CARLA | legenda.

### English

`scripts/carla_drive.py` opens one window with a panel per source, side by side:

- the camera;
- the CARLA ground truth (map lanes projected into the image);
- one panel per model (`--checkpoint`, repeatable), with the model's lanes, thin green ground truth and a HUD
  (offset, lane change, LDW, LKA steering request, latency).

It starts the CARLA server if none is running (off-screen, Low quality).

1. **Before:** pause the GPU queue (*Fila GPU - Pausar* shortcut). The card has 6 GB: CARLA takes 3-4 GB and each
   model 0.3-1 GB. Close Chrome with many tabs, LM Studio and Docker if unused (CARLA exits when memory runs
   out). One GPU job at a time.
2. **Run:** the commands above (PowerShell, in the project folder). Modes:
   - `--drive manual`: you drive.
   - `--drive autopilot`: the Traffic Manager drives; `L` hands the steering to the selected model with cruise
     control.
   - `--drive lka`: model 1 steers from the start.
   - `--record x.mp4`: also writes the side-by-side panels as H.264 (ffmpeg); `--no-window` with `--duration-s`
     records without a window.
3. **Keys:** `W`/`S` throttle/brake, `A`/`D` steer, `Q`/`E` left/right turn signal (toggle), `L` LKA on/off,
   `1`..`9` which model steers, `G` ground truth on the model panels, `ESC` quit.
4. **Maps:** Town04 (highway) loads fine. Full Town06 and Town03 crash the server on this PC: use
   `--map-layers minimal` (the `_Opt` map without buildings). To change towns, restart the server.
5. **Notes:** the simulation is synchronous (every frame waits for every model), so several models run slower
   than real time: 3 models run at about 4 frames/s, and 60 simulated seconds take about 7 min. The recording
   plays in simulated time. Town04 and Town06 were never used for fine-tuning: they are unseen towns for every
   model.
6. **Demo video:** `results/carla_lka/demo_town04_autopilot.mp4`. It is 60 s in Town04, on autopilot with 30
   cars. Panels: camera | ground truth | UFLD v1 baseline (ELAS) | lite_tsm (ELAS) | lite_tsm fine-tuned on
   CARLA | legend.
