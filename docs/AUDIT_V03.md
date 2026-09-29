# Audit before the v0.3 work (2026-09-29)

State of the repository at commit `fff07b6` (v0.2.0), before any change in this round. The older audit that motivated v0.2 is `docs/AUDIT_REPORT.md`.

## How it was checked

* Read every module under `src/tac_ufld/`, the configs, the tests, `docs/` and the supervisor's notebook (outside the repository, for the CULane port).
* Offline test suite: **50 passed, 0 skipped** in 2 min 09 s (`pytest -q`, Python 3.14.3, PyTorch 2.14.0+cu126, RTX 3050 6 GB; the real-ELAS regression and the ImageNet strict-load tests ran).
* ELAS GPU smoke run, from a frozen copy of the commit: `python -m tac_ufld run --config configs/elas_smoke.yaml` → **exit 0 in 8 min 36 s**; every stage ran (HPO 2×1, 6 variants × 2 seeds × 2 epochs, evaluation, report). All lane F1 values were 0.000: the smoke config (no ImageNet weights, 192×256, 160 frames, 2 epochs) checks plumbing, not learning.

## Implemented and tested

ELAS adapter with the corrected geometry (real-data regression test); missing/unknown lane slots and ROI ignore rows; row-anchor targets; anchor metric with wrong positions as FP + FN; CULane-style lane F1 (Hungarian); official UFLD (ResNet-18, strict ImageNet load); v0.2/v0.3/v0.4 and lite baseline/v0.5 with warm starts; scene hold-out split with purge gaps, seed independence and leakage checks; seeded fresh Optuna studies with equal budgets; 6 seeds, exact Wilcoxon + Holm; validation-tuned and common post-processing; YAML validation; end-to-end synthetic run.

## Implemented but not validated

* CULane adapter (ported, synthetic unit tests only, no CULane copy).
* `--resume` (never exercised; see risk 1).
* Efficiency numbers (measured with history re-encoded every call: an upper bound for streaming).

## Missing (requested in this round)

Dataset registry and selection; TuSimple and OpenLane adapters; preprocessing ablations; geometric and extended photometric augmentation; label smoothing, optional dropout, epoch-level resume; streaming inference with feature caching; ONNX / TensorRT export, FP16/INT8, numerical checks; hardware budget simulation; UI; Docker; static results page; hand-off package.

## Risks found

1. **Resume could skip unfinished training.** A variant was considered trained as soon as its best checkpoint existed, but that file is written at the first improvement, so an interrupted variant was reported as finished. *Fixed:* completion requires the history CSV; interrupted variants continue from `<variant>.last.pt`.
2. **Full training could start by accident.** `python -m tac_ufld run` with no arguments started `configs/elas.yaml` (days of GPU). *Fixed:* configs marked `requires_confirmation` need `--confirm`; nothing is created before the check.
3. **CULane temporal step.** `temporal_step: 30` finds no history in the `*_90frame` drivers. *Fixed:* 90, the supervisor's value (docs/DATASETS.md).
4. **CULane lane slots** came from a geometric rule instead of the official segmentation-label ids. *Fixed* when labels exist.
5. **Check overlays** would have written one image per clip on CULane (~9k). *Fixed:* capped at 40 sequences.
6. **Smoke config learns nothing**, so it cannot show whether models train. *Added* `configs/elas_pilot.yaml` (full protocol, short budget, ~2.5 h).
7. Label-geometry check warns for `VIX_S05` (median straight-line residual 11.7 px on the split's frames): that scene has curves, so the straight-line check is expected to fail there; its real-data regression test (all frames) passes.

## Exact commands used for the baseline check

```powershell
cd tac-ufld-lane-detection
..\.venv\Scripts\python.exe -m pytest -q
..\.venv\Scripts\python.exe -m tac_ufld run --config configs/elas_smoke.yaml --name elas_smoke_pre_changes
```
