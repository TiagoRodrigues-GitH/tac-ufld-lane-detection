# Testing

## Commands

```powershell
pytest -q                                   # everything available locally (~10 min on the dev machine)
pytest -q -m "not dataset"                  # without the local ELAS copy
pytest -q -m network                        # ImageNet download + strict weight loading
pytest -q tests/test_streaming.py           # one area
python -m tac_ufld run --config configs/elas_smoke.yaml     # GPU plumbing smoke, ~9 min
python -m tac_ufld run --config configs/elas_pilot.yaml     # GPU pilot, ~2.5 h
```

In Docker: `docker compose run --rm tests` (GPU image) or `docker compose --profile cpu run --rm tests-cpu`.

## Coverage by requirement

| Requirement | Tests |
|---|---|
| Adapters parse fixtures (CULane, TuSimple, OpenLane; ELAS synthetic + real) | `test_datasets.py`, `test_elas.py` |
| Dataset activation / disabling, missing roots, invalid configs | `test_registry_cli.py` |
| Leakage-safe splits (incl. carved validation) | `test_splits.py`, `test_datasets.py` |
| Geometric augmentation keeps image-label alignment; clipping, missing points, lane order, determinism | `test_preprocess_augment.py` |
| Preprocessing shapes and values; non-RGB models | `test_preprocess_augment.py` |
| Models instantiate and run; warm starts; gradients | `test_models.py`, `test_tools.py::test_sanity...` |
| Losses and metrics on known examples (incl. TuSimple `LaneEval`) | `test_losses.py`, `test_metrics.py`, `test_datasets.py` |
| Streaming state reset and isolation; cache exactness | `test_streaming.py` |
| ONNX export and numerical checks; INT8; FP16 (CUDA provider) | `test_deploy.py` |
| Budget simulation, benchmark, stream and export commands | `test_tools.py` |
| UI starts without training and imports no training code | `test_ui.py` |
| Docker configuration | `test_docker.py` |
| Full training never starts by accident | `test_registry_cli.py::test_full_run_requires_confirmation` |
| End-to-end pipeline (ELAS synthetic, TuSimple fixture) | `test_integration.py`, `test_registry_cli.py` |
| Results page | `test_site.py` |
| v0.4: current-frame degradation (history untouched, no random draws when off), backbone lr groups, frozen stages and BatchNorm, official-checkpoint initialisation, new variants registered and paired, recurrent models start as their baseline, carry = window until the chain exceeds the clip, Kalman tracker (noise reduction, causality, gating, gaps), ablation grid / allow / reuse, capacity control built after augmentation, one run log per experiment | `test_temporal_v04.py` |
| Multi-dataset preparation: download checker (flags missing parts with the fix, compares published counts, finds data one level down, reads only), smoke configs keep each full config's geometry and augmentation, every stage end to end on synthetic CULane / TuSimple / OpenLane copies at the real geometry with the native metric logged separately | `test_dataset_tools.py` |
| v0.4 end to end: new fusions, +CT and static controls, Kalman and carried-state rows, report sections | `test_integration.py::test_v04_pipeline_with_new_models_controls_and_temporal_references` |

## Results

Recorded on 2026-09-29 (v0.3) and 2026-09-30 (v0.4).

| Run | Environment | Result | Skipped (reason) |
|---|---|---|---|
| Baseline before any change (`fff07b6`) | Windows 11, Python 3.14.3, PyTorch 2.14.0+cu126, RTX 3050 | **50 passed** in 2 min 09 s | none |
| Final suite, host | same + onnx 1.23, onnxruntime-gpu 1.30 (CPU provider), TensorRT 11.3, Streamlit 1.64 | **159 passed, 1 skipped, 0 failed** in 11 min 16 s | `test_fp16_on_cuda_execution_provider`: ONNX Runtime's CUDA provider needs CUDA 13, PyTorch here is CUDA 12.6 |
| Final suite, Docker CPU image (`tac-ufld:cpu`, current source mounted, ELAS mounted read-only) | Linux (WSL 2), Python 3.12.14, PyTorch 2.8.0+cpu | **152 passed, 2 skipped, 2 failed** in 7 min 32 s; the 2 failures were environment assumptions of the tests (no `.git` / `.empty` inside the image), fixed and re-run: pass/skip | ONNX Runtime CUDA provider (none in the image); `docker` CLI (not inside the container); packaging (not a git checkout) |
| v0.4 suite, host (while a GPU training run was using the machine) | as the v0.3 host run | **220 passed, 1 skipped, 0 failed** in 21 min 27 s | same as v0.3 (ONNX Runtime CUDA provider) |
| v0.4 + multi-dataset preparation, host (GPU training running in parallel) | as above | **229 passed, 1 skipped, 0 failed** in 41 min 16 s; the three new smoke tests take 3–9 min each on the busy CPU | same (ONNX Runtime CUDA provider) |

Real-data checks included in both suites: ELAS geometry regression on all 22 scenes (`test_real_elas_points_are_collinear`). ELAS regression of the multi-dataset changes: three pilot-2 checkpoints (UFLD baseline, v0.3, lite v0.5, seed 1) re-evaluated on the 2,610 held-out frames with the new code give the stored lane F1 exactly (0.881236, 0.872794, 0.708749); pixel F1, anchor F1 and jitter differ by at most 5e-5 (mixed-precision GPU inference). The network test (`pytest -m network`, ImageNet download + strict load) passed in both host runs; the Docker run deselected it.

GPU runs:

| Run | Result |
|---|---|
| ELAS smoke, before the changes (`configs/elas_smoke.yaml`) | exit 0, 8 min 36 s; lane F1 = 0 for every model (plumbing config) |
| **ELAS GPU pilot** after the changes (`configs/elas_pilot.yaml`, commit `96b4505`) | exit 0, 2 h 22 min; results in `docs/PILOT_FINDINGS.md` |
| ELAS smoke on the final code (commit `2b60ae7`) | exit 0, 21 min 39 s (the machine was also building the GPU Docker image; 8 min 36 s alone before the changes); every stage and output present, incl. `report/config_changes.json` and the protocol section |
| Model sanity, real ELAS batch | all six variants pass, on the host (CPU) and inside the GPU container (`--gpus all`) |
| Export + TensorRT FP32/FP16/INT8 on three pilot checkpoints | all engines built and ran; validation F1 table in `docs/DEPLOYMENT.md` |
| `validate-dataset --dataset elas` (real data) | OK: 7,561 frames, 10 scenes, 0 lane-order violations, 0 % missing history |

Not run anywhere (no data / hardware): CULane, TuSimple, OpenLane real data; Jetson; CARLA; ONNX Runtime CUDA provider.
