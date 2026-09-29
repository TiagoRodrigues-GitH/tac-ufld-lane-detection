"""ONNX export (streaming split), ONNX Runtime numerical checks, FP16 / INT8."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from tac_ufld.models.registry import VARIANTS
from tests.test_streaming import _cfg, _checkpoint

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from tac_ufld.deploy.check import numerical_check, synthetic_frames  # noqa: E402
from tac_ufld.deploy.onnx_export import MatmulAdaptiveAvgPool2d, export_model  # noqa: E402
from tac_ufld.deploy.ort_backend import OnnxRuntimeBackend  # noqa: E402
from tac_ufld.inference.loading import load_model  # noqa: E402


@pytest.mark.parametrize("in_hw,out_hw", [((24, 32), (18, 100)), ((12, 16), (18, 50)), ((8, 8), (4, 4)), ((7, 9), (3, 5))])
def test_matmul_pool_equals_adaptive_pool(in_hw, out_hw):
    x = torch.randn(2, 5, *in_hw)
    assert torch.allclose(MatmulAdaptiveAvgPool2d(in_hw, out_hw)(x), nn.AdaptiveAvgPool2d(out_hw)(x), atol=1e-6)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_export_and_onnxruntime_match_pytorch(tmp_path, variant):
    cfg = _cfg()
    loaded = load_model(_checkpoint(tmp_path, variant, cfg))
    manifest = export_model(loaded, tmp_path / "deploy")
    spec = VARIANTS[variant]
    expected = {"encoder", "head", "clip"} if spec.temporal else {"model"}
    assert set(manifest["graphs"]) == expected
    assert (manifest["temporal_state"] is not None) == spec.temporal
    saved = json.loads((tmp_path / "deploy" / "deployment.json").read_text(encoding="utf-8"))
    assert saved["card"]["input"]["channels"] == 3 and saved["card"]["output"]["logits_shape"][1] == 25
    backend = OnnxRuntimeBackend(tmp_path / "deploy", providers=["CPUExecutionProvider"])
    result = numerical_check(loaded, backend, synthetic_frames(6, size=(120, 160)), "fp32")
    assert result["pass"], result
    assert result["lane_presence_disagreements"] == 0


def test_tiny_history_encoder_exports_two_feature_outputs(tmp_path):
    cfg = _cfg(**{"model.lite_history_encoder": "tiny"})
    loaded = load_model(_checkpoint(tmp_path, "lite_v05", cfg))
    manifest = export_model(loaded, tmp_path / "d")
    assert set(manifest["graphs"]["encoder"]["outputs"]) == {"current_features", "history_features"}
    backend = OnnxRuntimeBackend(tmp_path / "d", providers=["CPUExecutionProvider"])
    assert numerical_check(loaded, backend, synthetic_frames(6, size=(120, 160)), "fp32")["pass"]


@pytest.mark.parametrize("variant", ["ufld_baseline", "ufld_v02"])
def test_int8_quantization_with_calibration(tmp_path, variant):
    from tac_ufld.deploy.precision import quantize_int8

    cfg = _cfg()
    loaded = load_model(_checkpoint(tmp_path, variant, cfg))
    export_model(loaded, tmp_path / "fp32")
    frames = synthetic_frames(12, size=(120, 160), seed=1)
    manifest = quantize_int8(loaded, tmp_path / "fp32", tmp_path / "int8", frames[:8], method="minmax")
    assert manifest["precision"] == "int8" and manifest["int8_calibration"]["frames"] == 8
    ops = {n.op_type for g in manifest["graphs"].values() if g["file"] != "clip.onnx"
           for n in onnx.load(str(tmp_path / "int8" / g["file"])).graph.node}
    assert "QuantizeLinear" in ops and "DequantizeLinear" in ops
    backend = OnnxRuntimeBackend(tmp_path / "int8", providers=["CPUExecutionProvider"])
    result = numerical_check(loaded, backend, frames[8:], "int8")
    assert np.isfinite(result["max_abs_exist"]) and result["max_abs_exist"] < 0.5


def test_fp16_on_cuda_execution_provider(tmp_path):
    from tac_ufld.deploy.ort_backend import ort_providers
    from tac_ufld.deploy.precision import convert_fp16

    if "CUDAExecutionProvider" not in ort_providers():
        pytest.skip("ONNX Runtime CUDA execution provider not available")
    cfg = _cfg()
    loaded = load_model(_checkpoint(tmp_path, "ufld_v03", cfg))
    export_model(loaded, tmp_path / "fp32")
    convert_fp16(tmp_path / "fp32", tmp_path / "fp16")
    backend = OnnxRuntimeBackend(tmp_path / "fp16", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    if "CUDAExecutionProvider" not in backend.active_providers.values():
        pytest.skip("CUDA execution provider failed to initialise")
    result = numerical_check(loaded, backend, synthetic_frames(6, size=(120, 160)), "fp16")
    assert result["pass"], result


@pytest.mark.parametrize("variant", ["ufld_v02", "lite_v05"])
def test_tensorrt_engines_match_pytorch(tmp_path, variant):
    from tac_ufld.deploy.tensorrt import TensorRTBackend, available, build_deployment
    from tac_ufld.deploy.precision import quantize_int8

    if not available():
        pytest.skip("TensorRT not installed or no CUDA device")
    cfg = _cfg()
    loaded = load_model(_checkpoint(tmp_path, variant, cfg))
    export_model(loaded, tmp_path / "d")
    frames = synthetic_frames(10, size=(120, 160), seed=2)
    quantize_int8(loaded, tmp_path / "d", tmp_path / "d" / "int8", frames[:6], method="minmax")
    loaded.model.to("cuda")
    loaded.device = "cuda"
    for precision in ("fp32", "fp16", "int8"):
        info = build_deployment(tmp_path / "d", precision)
        assert set(info["graphs"]) == {"encoder", "head", "clip"}
        result = numerical_check(loaded, TensorRTBackend(tmp_path / "d", precision), frames[6:], precision)
        if precision == "fp32":
            assert result["max_abs_exist"] < 1e-3, result
        elif precision == "fp16":
            assert result["max_abs_exist"] < 2e-2, result
        else:
            assert np.isfinite(result["max_abs_exist"])
