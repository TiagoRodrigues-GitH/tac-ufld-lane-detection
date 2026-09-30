"""TAC-UFLD: a reproducible research pipeline comparing the UFLD baseline
(Ultra-Fast Lane Detection, Qin et al., ECCV 2020) against lightweight
temporal lane-detection variants, with streaming and embedded deployment.

Module map
----------
config          YAML -> typed dataclasses, validation, ${VAR} expansion, config hashing
data            dataset adapters (ELAS, CULane, TuSimple, OpenLane), registry,
                splits, targets, preprocessing, augmentation, torch Dataset
models          UFLD re-implementation, temporal variants, lightweight models
losses          official UFLD losses + variant-specific auxiliary terms
metrics         anchor, lane (IoU-matched), pixel and temporal metrics
postprocess     row-anchor predictions -> lane polylines
training        Trainer (optimisation loop, resume) and Optuna HPO
evaluation      prediction, evaluation, dataset-native metrics, efficiency,
                multi-seed statistics, hardware budget simulation
inference       checkpoint loading, model card, streaming with feature caching
deploy          ONNX export, ONNX Runtime / TensorRT backends, FP16 / INT8, checks
ui              Streamlit interface (inference and results only)
visualization   overlays, videos and figures
experiment      end-to-end orchestration (the only module that trains)
ablation        fair ablation runner; sanity: pre-training model checks
site / package / doctor   results page, hand-off ZIP, environment checks
"""

__version__ = "0.4.0"
