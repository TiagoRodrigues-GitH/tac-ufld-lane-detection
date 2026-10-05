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
