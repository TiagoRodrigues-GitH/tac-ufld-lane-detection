"""FP16 conversion and INT8 post-training quantization of exported graphs.

FP16 (ONNX Runtime): weights and activations converted with
``onnxconverter-common`` keeping FP32 inputs/outputs; intended for the CUDA
execution provider. For TensorRT, FP16 is a builder flag on the FP32 ONNX
(``deploy/tensorrt.py``) - no separate FP16 file is needed.

INT8 (ONNX Runtime static quantization, QDQ format, per-channel weights,
SYMMETRIC activations and weights - zero point 0, as TensorRT requires):
activation ranges are calibrated on real frames that the caller takes from
the VALIDATION split (never test). For temporal models the encoder is
calibrated on the frames and the head on the feature tensors the FP32
encoder produces for those frames in stream order, so the head sees
realistic (history, current) pairs. The QDQ graphs can also be built by
TensorRT with INT8 enabled (explicit quantization).

Limitations: INT8 accuracy depends on the calibration set and must be checked
with ``deploy_eval`` on the validation split; ONNX Runtime's CPU INT8 kernels
and TensorRT's INT8 kernels are not bit-identical; the clip graph is not
quantized (it is only a non-streaming reference).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from tac_ufld.inference.loading import FramePreprocessor, LoadedModel


def convert_fp16(deployment_dir: str | Path, out_dir: str | Path) -> dict:
    import onnx
    from onnxconverter_common import float16

    # GridSample (lite v0.5 warp) stays FP32: its sampling grid is computed in
    # FP32 and TensorRT requires input and grid of the same type.
    block = list(getattr(float16, "DEFAULT_OP_BLOCK_LIST", [])) + ["GridSample"]
    src, out = Path(deployment_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "deployment.json").read_text(encoding="utf-8"))
    for g in manifest["graphs"].values():
        model = float16.convert_float_to_float16(onnx.load(str(src / g["file"])), keep_io_types=True,
                                                 op_block_list=block)
        onnx.save(model, str(out / g["file"]))
    manifest["precision"] = "fp16"
    manifest["precision_note"] = ("onnxconverter-common float16, FP32 inputs/outputs, GridSample kept FP32; for the "
                                  "ONNX Runtime CUDA provider and TensorRT (strongly typed)")
    (out / "deployment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _first_convs(path: Path) -> list[str]:
    """Names of the Conv nodes that read a graph input directly (the image)."""
    import onnx

    graph = onnx.load(str(path)).graph
    inputs = {i.name for i in graph.input}
    return [n.name for n in graph.node if n.op_type == "Conv" and n.input and n.input[0] in inputs]


class _Reader:
    """ONNX Runtime CalibrationDataReader over a list of feed dicts."""

    def __init__(self, feeds: list[dict[str, np.ndarray]]) -> None:
        self._it = iter(feeds)

    def get_next(self):
        return next(self._it, None)


def quantize_int8(loaded: LoadedModel, deployment_dir: str | Path, out_dir: str | Path,
                  calibration_frames: list[np.ndarray], method: str = "percentile") -> dict:
    """``calibration_frames``: consecutive RGB frames from the validation split."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    from tac_ufld.deploy.ort_backend import OnnxRuntimeBackend

    src, out = Path(deployment_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "deployment.json").read_text(encoding="utf-8"))
    pre = FramePreprocessor(loaded.cfg)
    frames = [pre(f).unsqueeze(0).numpy().astype(np.float32) for f in calibration_frames]
    calib = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
             "percentile": CalibrationMethod.Percentile}[method]
    feeds: dict[str, list[dict]] = {}
    if manifest["temporal_state"] is None:
        feeds["model"] = [{"frame": f} for f in frames]
    else:
        fp32 = OnnxRuntimeBackend(src, providers=["CPUExecutionProvider"])
        encoded = [fp32.encode(f) for f in frames]
        t = manifest["temporal_state"]["num_frames"]
        step = manifest["temporal_state"]["temporal_step"]
        feeds["encoder"] = [{"frame": f} for f in frames]
        head = []
        for i, enc in enumerate(encoded):
            hist = [encoded[max(0, i - k * step)]["history"] for k in range(t - 1, 0, -1)]
            head.append({"history_features": np.stack(hist, axis=1), "current_features": enc["current"]})
        feeds["head"] = head
    # Biases stay FP32: TensorRT only dequantizes 8-bit (not int32) tensors.
    extra = {"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False,
             "CalibMovingAverage": method == "minmax"}
    for name, graph in manifest["graphs"].items():
        target = out / graph["file"]
        if name not in feeds:  # clip graph: kept in FP32 (non-streaming reference only)
            shutil.copyfile(src / graph["file"], target)
            continue
        # Only Conv / Gemm / MatMul are quantized (TensorRT-friendly Q/DQ placement),
        # and the first convolution on the image stays FP32: with 3-4 input channels
        # it has no INT8 kernel in TensorRT, and it is the most sensitive layer.
        first_convs = _first_convs(src / graph["file"])
        quantize_static(str(src / graph["file"]), str(target), _Reader(feeds[name]), quant_format=QuantFormat.QDQ,
                        per_channel=True, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        calibrate_method=calib, op_types_to_quantize=["Conv", "Gemm", "MatMul"],
                        nodes_to_exclude=first_convs, extra_options=extra)
    manifest["precision"] = "int8"
    manifest["precision_note"] = (f"ONNX Runtime static quantization, QDQ, symmetric per-channel INT8, {method} "
                                  f"calibration on {len(frames)} validation frames; clip graph kept FP32")
    manifest["int8_calibration"] = {"frames": len(frames), "method": method, "source": "validation split"}
    (out / "deployment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
