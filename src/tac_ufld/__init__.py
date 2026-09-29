"""TAC-UFLD: a reproducible research pipeline comparing the UFLD baseline
(Ultra-Fast Lane Detection, Qin et al., ECCV 2020) against lightweight
temporal lane-detection variants, starting with the ELAS dataset.

Module map
----------
config          YAML -> typed dataclasses, validation, config hashing
data            dataset adapters (ELAS, CULane), splits, targets, torch Dataset
models          UFLD re-implementation, temporal variants, lightweight models
losses          official UFLD losses + variant-specific auxiliary terms
metrics         anchor, lane (IoU-matched), pixel and temporal metrics
postprocess     row-anchor predictions -> lane polylines
training        Trainer (optimisation loop) and Optuna HPO
evaluation      prediction, evaluation, efficiency and multi-seed statistics
visualization   overlays and figures
experiment      end-to-end orchestration (the only module with side effects)
"""

__version__ = "0.2.0"
