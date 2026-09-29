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

## Results

Recorded on 2026-09-29.

| Run | Environment | Result | Skipped (reason) |
|---|---|---|---|
| Baseline before any change (`fff07b6`) | Windows 11, Python 3.14.3, PyTorch 2.14.0+cu126, RTX 3050 | **50 passed** in 2 min 09 s | none |
| Final suite, host | same + onnx 1.23, onnxruntime-gpu 1.30 (CPU provider), TensorRT 11.3, Streamlit 1.64 | **159 passed, 1 skipped, 0 failed** in 11 min 16 s | `test_fp16_on_cuda_execution_provider`: ONNX Runtime's CUDA provider needs CUDA 13, PyTorch here is CUDA 12.6 |
| Final suite, Docker CPU image (`tac-ufld:cpu`, current source mounted, ELAS mounted read-only) | Linux (WSL 2), Python 3.12.14, PyTorch 2.8.0+cpu | **152 passed, 2 skipped, 2 failed** in 7 min 32 s; the 2 failures were environment assumptions of the tests (no `.git` / `.empty` inside the image), fixed and re-run: pass/skip | ONNX Runtime CUDA provider (none in the image); `docker` CLI (not inside the container); packaging (not a git checkout) |

Real-data checks included in both suites: ELAS geometry regression on all 22 scenes (`test_real_elas_points_are_collinear`). The network test (`pytest -m network`, ImageNet download + strict load) passed in both host runs; the Docker run deselected it.

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
