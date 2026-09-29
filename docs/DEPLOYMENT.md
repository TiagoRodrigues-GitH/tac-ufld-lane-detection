# Streaming inference and embedded deployment

Status labels used here: **validated (desktop)** = ran and checked on the development machine (RTX 3050, Windows 11); **not validated** = written and, where possible, unit-tested, but never run on the target hardware.

| Component | Status |
|---|---|
| Streaming inference, per-stream state, feature caching, resets, dropped frames | validated (desktop), exactness tests vs the clip forward pass |
| ONNX export (FP32, streaming split) + ONNX Runtime CPU | validated (desktop), all six variants, real checkpoints |
| ONNX INT8 (static Q/DQ, calibrated on validation frames) on ONNX Runtime CPU | validated (desktop), accuracy loss measured |
| ONNX Runtime CUDA / FP16 | **not validated**: the PyPI `onnxruntime-gpu` 1.30 needs CUDA 13, the venv's PyTorch is CUDA 12.6 |
| TensorRT FP32 / FP16 / INT8 engines + runtime | validated (desktop, TensorRT 11.3 on the RTX 3050) |
| TensorRT on Jetson (JetPack) | **not validated**: no Jetson available; procedure and scripts below |
| Budget simulation for Jetson | simulation only; the slowdown factors are assumptions |
| CARLA frame source | **not validated**: no CARLA server available |

## Streaming inference

```python
from tac_ufld.inference.loading import load_model
from tac_ufld.inference.streaming import StreamingLaneDetector

loaded = load_model("results/elas_pilot/seed_1/checkpoints/lite_v05.pt", device="cuda")
det = StreamingLaneDetector(loaded, mode="cached")          # or "recompute"
res = det.infer_frame(rgb_uint8, stream_id="cam0", timestamp=t)   # one frame of one stream
det.reset_stream("cam0")                                    # e.g. at a sequence boundary
for rgb, res in det.infer_video("drive.mp4"):               # incremental, one frame in memory
    ...
```

`FrameResult` holds the lanes in original pixels per slot, each lane's mean existence probability, the raw existence/position arrays, the history frame indices actually used, the number of fallbacks, the reset reason, frames dropped since the previous call, and the latency split (`preprocess_ms`, `model_ms`, `postprocess_ms`).

**What is cached, and why it is exact.** In eval mode each model's per-frame backbone output depends on that frame only (shared weights, BatchNorm running statistics, dropout off). Fusion depends on the current frame: v0.2/v0.4 weights, v0.3 gates, and v0.5's warp and gate. So the cache holds one feature map per past frame, and each step runs one backbone pass plus fusion and head. With the `tiny` history encoder of v0.5 a frame's history features differ from its current-frame features, so both are cached. The detector refuses to cache in training mode. `tests/test_streaming.py` checks the cached result against the full clip forward pass for every temporal variant, including dropped frames.

**Temporal rule.** History of frame `t` = frames `t − k·temporal_step`, `k = T−1..1`, as in training. A missing frame (stream start, drop, `history_length` limit) is replaced by the nearest newer frame of the context, which is the dataset's rule. Frame indices come from `frame_index`, from `timestamp × nominal_fps`, or from a per-stream counter; drops can only be detected in the first two cases. State resets on `reset_stream`, a new `sequence_id`, a new source frame size, or indices going backwards. Streams never share state (`test_streams_are_isolated`).

**Measured (RTX 3050, batch 1, 640×480 frames of the held-out clip, `results/benchmarks/desktop.md`):**

| Model | Mode | Preprocess ms | Model ms | Postprocess ms | Total ms |
|---|---|---|---|---|---|
| UFLD baseline | single | 10.1 | 7.9 | 1.1 | 19.1 |
| UFLD v0.2 | cached | 8.7 | 7.9 | 1.4 | 18.0 |
| UFLD v0.2 | recompute | 9.3 | 16.1 | 1.4 | 26.8 |
| Lite baseline | single | 8.4 | 4.2 | 1.4 | 14.0 |
| Lite v0.5 | cached | 8.5 | 8.5 | 1.3 | 18.3 |
| Lite v0.5 | recompute | 9.8 | 14.4 | 1.4 | 25.7 |

Caching brings the temporal models' compute down to the single-frame baseline's. Preprocessing (a PIL bilinear resize on the CPU, the same operation as in training) is now the largest cost; moving it to the GPU is an optimisation, but it must reproduce PIL's resampling exactly (antialiased bilinear) or it changes the model's input.

## Export

```powershell
python -m tac_ufld export --checkpoint results/elas_pilot/seed_1/checkpoints/ufld_v02.pt `
    --int8 --calib-frames 64 --check-frames 24 --eval-frames 240 --tensorrt fp32 fp16 int8
```

Output (`<run>/deploy/<variant>_seed<n>/`):

| File | Content |
|---|---|
| `model.onnx` | single-frame models: `frame (1,C,H,W)` → `logits (1,G+1,A,L)` |
| `encoder.onnx` | temporal models: `frame` → `current_features` [+ `history_features` for the tiny encoder] |
| `head.onnx` | `history_features (1,T−1,C',h,w)`, `current_features (1,C',h,w)` → `logits` |
| `clip.onnx` | `clip (1,T,C,H,W)` → `logits` (non-streaming reference) |
| `fp16/`, `int8/` | FP16 and Q/DQ INT8 graphs with their own `deployment.json` |
| `engines/*.engine` | TensorRT engines + `build_info.<precision>.json` (TensorRT version, GPU, compute capability) |
| `deployment.json` | model card (input size, channel order, resize, mean/std, preprocessing), output layout and decoding, row anchors, post-processing parameters and their source, every graph's inputs/outputs/shapes/ONNX operators, the temporal-state specification, and the numerical checks |

**Temporal state in deployment.** It is not inside the graphs. The runtime keeps a ring buffer of per-frame `history_features` (bytes per frame in `deployment.json`), selects the `T−1` entries by the temporal rule and feeds them to `head.onnx` with the current frame's features. `tac_ufld.inference.streaming` does this for every backend (PyTorch, ONNX Runtime, TensorRT). A single-frame export of a temporal model would not be a streaming deployment.

**Export pitfalls fixed during this work.** (1) `torch.onnx.export` restores the *wrapper's* original mode after exporting, and a fresh wrapper starts in train mode, so the whole model was left in training mode (BatchNorm on batch statistics). (2) The following shape probe then updated BatchNorm running statistics. Both were caught by the numerical check (single-frame UFLD differed by 0.05 in logits). The exporter now puts wrappers in eval mode and refuses to finish if the model's state changed. `LanePixelHead`'s `AdaptiveAvgPool2d` with non-divisible sizes (24×32 → 18×100) is replaced during export by an exactly equivalent pair of matrix products (tested).

## Precision and numerical checks

Every backend runs the same frames through the same streaming code as PyTorch FP32 (`deploy/check.py`). Engineering tolerances: FP32 logits ≤ 1e-3 / existence ≤ 1e-4 / position ≤ 0.05 px; FP16 5e-2 / 1e-2 / 0.5 px; INT8 approximate (existence ≤ 0.15, position ≤ 3 px). Task accuracy decides: `--eval-frames` scores each backend with the experiment's evaluator on validation frames, using the exact training-time temporal context.

Pilot checkpoints (seed 1), 240 validation frames, lane F1 @ IoU 0.5:

| Model | PyTorch FP32 | ONNX Runtime FP32 | ONNX Runtime INT8 | TensorRT FP32 | TensorRT FP16 | TensorRT INT8 |
|---|---|---|---|---|---|---|
| UFLD baseline | 0.1732 | 0.1732 | 0.1776 (+0.004) | 0.1732 | 0.1732 | 0.1732 |
| UFLD v0.2 | 0.0789 | 0.0789 | 0.0636 (−0.015) | 0.0789 | 0.0789 | 0.0768 (−0.002) |
| Lite v0.5 | 0.5636 | 0.5636 | 0.5570 (−0.007) | 0.5636 | 0.5636 | 0.5658 (+0.002) |

Numerics on 16 validation frames (streaming): ONNX Runtime FP32 max |Δ logits| ≤ 8e-6 for all three. TensorRT FP32 (TF32 off) ≤ 5.5e-4 and FP16 ≤ 9e-3 for the UFLD models (within tolerance). For lite v0.5 TensorRT FP32 reaches 1.9e-3 and FP16 5.9e-2, slightly outside the strict tolerances because TensorRT's `GridSample` differs from PyTorch's; the lanes and the validation F1 are unchanged. INT8 moves lanes by a few pixels on average (max |Δx| 3–21 px in the worst anchor cell) and changes validation F1 by −0.015 to +0.004 on 240 frames, which is within the noise of that sample for most rows. An earlier INT8 configuration (asymmetric activations, every operator quantized) lost 0.042 F1 on lite v0.5; the TensorRT-compatible configuration below is also the more accurate one.

* **FP16** is converted with `onnxconverter-common` (FP32 inputs/outputs). `GridSample` (v0.5's warp) stays in FP32, because TensorRT requires the sampled tensor and the grid to have the same type.
* **INT8** uses ONNX Runtime static quantization calibrated on validation frames (never test): Q/DQ format, symmetric per-channel INT8 (zero point 0, as TensorRT requires). Only Conv/Gemm/MatMul are quantized; biases stay FP32 (TensorRT dequantizes only 8-bit tensors); the first convolution stays FP32 (TensorRT has no INT8 kernel for it, and it is the most sensitive layer). The temporal head is calibrated on feature pairs produced by the FP32 encoder in stream order. Limitations: accuracy depends on the calibration set; ONNX Runtime and TensorRT INT8 kernels differ; the clip graph is not quantized.
* **TensorRT 11** builds only strongly typed networks: precision comes from the ONNX types (FP32, FP16 or Q/DQ graph), not from builder flags; the FP32 engine disables TF32. With TensorRT 10 the code sets the FP16/INT8 builder flags on the FP32 graph instead.

## Jetson deployment (not validated)

1. **Match versions.** JetPack fixes CUDA, cuDNN and TensorRT (e.g. JetPack 6.x: CUDA 12.x, TensorRT 10.x). Install PyTorch from NVIDIA's JetPack wheels (not PyPI) and check `python3 -c "import torch, tensorrt; print(torch.__version__, torch.cuda.is_available(), tensorrt.__version__)"`.
2. **Export on the desktop** (ONNX is portable): `python -m tac_ufld export --checkpoint <ckpt> --int8 --fp16`. Copy the deployment folder **without** `engines/`, the checkpoint (the streaming runtime reads the model card and the config from it) and the repository.
3. **Build engines on the Jetson:** `./scripts/jetson/build_engines.sh <deployment_dir> fp16` (uses `trtexec`; engines from another GPU or TensorRT version cannot be used). On TensorRT 8.x (JetPack 5), build from the FP32 ONNX with `--fp16` / `--int8` instead of the typed graphs.
4. **Power and clocks.** Choose the power mode (`sudo nvpmodel -m <mode>`) and pin clocks (`sudo jetson_clocks`) before measuring; record both.
5. **Measure:** `./scripts/jetson/measure_on_jetson.sh <ckpt> <deployment_dir> <video> fp16` runs the streaming benchmark with the TensorRT backend and logs memory and power with `tegrastats`. Replace the assumed slowdown in `configs/deploy/jetson_*_assumed.yaml` with the measured latency.
6. **Check accuracy on the device:** `python3 -m tac_ufld export ... --eval-frames 240 --tensorrt fp16` (needs the validation frames on the device).
7. Consider the preprocessing cost (CPU resize) on the Jetson's CPU; it was ~9 ms per frame on a desktop CPU.

## Hardware budget simulation

```powershell
python -m tac_ufld benchmark --checkpoints <ckpt> ... --profile configs/deploy/jetson_orin_nano_assumed.yaml --video <clip>
```

The benchmark first measures batch-1 streaming latency (split by stage) and peak GPU memory on the local machine, for each temporal model in `cached` and `recompute` mode. It then simulates a camera at `target_fps` with random sensor drops (`drop_rate`) feeding one worker. Service times are drawn from the measured samples multiplied by `slowdown`, and the policy is `latest` (skip stale frames) or `fifo`. Reported: processed FPS, frames skipped while busy, p50/p95/p99 end-to-end latency, deadline misses, **history fallbacks** (frames whose history frames were never processed), and whether the FPS, latency and memory budgets are met. The report keeps "measured on this machine" and "simulated" in separate sections. Profiles: `desktop` (no slowdown) and two Jetson profiles with **assumed** slowdowns (Orin Nano ×7, AGX Orin ×1.7, from peak-throughput ratios). A simulation can say whether a configuration *would* meet the budgets on hardware of that speed; it cannot establish Jetson performance.

Pilot result: every model meets 30 FPS on the desktop. Under the assumed Orin Nano slowdown none does, and 79–92 % of the temporal models' history frames fall back (the model effectively loses its temporal context when the worker is overloaded). Under the assumed AGX Orin slowdown only the lite baseline meets all budgets.

## CARLA (optional, not validated)

`python -m tac_ufld carla-demo --checkpoint <ckpt> --town Town04 --frames 300` connects to a running CARLA 0.9.x server (install the matching `carla` Python package), drives an ego vehicle on autopilot with a front RGB camera, and runs the streaming detector. Output goes to `results/simulator/` (overlay video + summary). CARLA frames have no lane ground truth in this pipeline and come from a different domain, so they are for qualitative inspection only and never enter dataset evaluation.
