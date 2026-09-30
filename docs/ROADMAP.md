# Roadmap

Status markers (read by `python -m tac_ufld site`): `[done]`, `[progress]`, `[next]`, `[later]`.
Goal: show that temporal information lets a **lightweight** lane detector match or beat the single-frame UFLD
baseline, and that it can run on an embedded automotive system. Embedded optimisation comes last, after the
accuracy question is answered under the full protocol.

## Phase 1: protocol and first pilot (v0.3)
- [done] Leakage-safe ELAS protocol: three held-out road scenes, 60-frame blocks with a purge gap, six seeds, HPO with the same budget for every model, validation-only selection.
- [done] GPU pilot (2 seeds, up to 4 epochs): lite v0.5 is best on unseen roads; every UFLD run peaks at epoch 1 and then overfits; history adds little to any model.
- [done] Deployment tooling: ONNX, TensorRT FP32 / FP16 / INT8 on the desktop GPU, streaming with cached history features.

## Phase 2: make a lightweight temporal model beat the UFLD baseline (v0.4)
- [done] Fix the UFLD overfitting: geometric augmentation lifts the UFLD baseline from 0.07 to 0.86 held-out lane F1 (`configs/ablations/augmentation.yaml`, chosen on validation); extended photometric augmentation, flips and dropout + label smoothing do not help.
- [progress] The other overfitting levers as single changes to the original recipe: a 10x lower backbone learning rate, frozen early layers, a smaller UFLD head, current-frame degradation and 8 more ELAS training scenes (remaining arms of the same ablation). Cross-dataset initialisation (`model.init_checkpoint`) is implemented for when CULane / TuSimple weights are available.
- [progress] Longer history: grid of frame steps 1 to 15 and 2 to 5 frames with one shared leakage-safe split (`configs/ablations/history.yaml`).
- [done] Current-frame degradation: only the current frame is occluded, blurred, darkened or made noisy while the history stays clean, so a temporal model must use its history (`data.augmentation.current_frame_prob`).
- [done] Align before fusing: UFLD v0.6 (lite v0.5's learned warp and gate on UFLD features).
- [done] Recurrent state: UFLD v0.7 and lite v0.6 (ConvGRU); streaming can carry one state per camera (`--mode carry`).
- [done] Equal-training controls: the +CT baselines get the same extra training as the temporal models.
- [done] Evaluate where time should help: lane F1 per scene condition, jitter, and a validation-tuned Kalman tracker as the cheap temporal reference.
- [done] Second pilot with the v0.4 models and recipe (`configs/elas_pilot_v2.yaml`): UFLD baseline 0.881, best temporal UFLD (v0.7) 0.895; the lite models need more epochs.
- [done] Capacity control `lite_v05_static`: lite v0.5 fed the current frame in every position, to separate a temporal gain from the extra fusion layers.
- [progress] Lite family with a longer budget (up to 16 epochs) and the capacity control (`configs/elas_lite_long.yaml`).
- [next] Train the recurrent models on long sequences so that one carried state per camera works (carry mode drifts after 3-frame training).

## Phase 3: full protocol (supervisor)
- [next] Full run: 6 seeds, HPO, up to 50 epochs, with the recipe and models chosen from the pilots (`configs/elas.yaml`).
- [later] Pretraining on CULane / TuSimple, then fine-tuning on ELAS (needs the datasets or the official UFLD weights).

## Phase 4: embedded automotive deployment (last step)
- [later] Measure on the real device: build the TensorRT engines on a Jetson and run the measurement script; today's Jetson figures are simulations with assumed slowdowns.
- [later] Take preprocessing off the CPU: the CPU resize is about 9 ms of a 14 to 19 ms frame. Resize on the camera's image processor or the GPU with zero-copy buffers, and retrain with that same resize.
- [later] Crop to the road: the row anchors cover only the lower 41 % of the image, so cropping cuts compute by about 2.4 times; a lower input resolution is the second option. Both need retraining.
- [later] TensorRT FP16 by default (lossless in the pilot); INT8 only if FP16 is too slow, calibrated on the car's camera or with quantization-aware training for the lite models.
- [later] Prefer the lite family: about 8 times fewer parameters and 3.6 times less compute than UFLD.
- [later] Robustness to dropped frames: train with frame drops and space the history by timestamps, since the car's camera may run at a different frame rate than ELAS.
- [later] Production runtime: C++ TensorRT or DeepStream instead of Python, capture / inference / post-processing in parallel, always process the latest frame, locked clocks, thermal tests inside the enclosure.
