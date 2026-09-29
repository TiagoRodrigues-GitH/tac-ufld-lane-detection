"""TensorRT engine building and inference.

Engines are built from the exported ONNX graphs (``encoder`` / ``head`` /
``clip`` or ``model``) and saved under ``<deployment>/engines/``. An engine
is specific to the GPU architecture, the TensorRT version and (on Jetson) the
JetPack release: build it ON the target device (``scripts/jetson/
build_engines.sh`` uses ``trtexec``); ``build_info.json`` records what an
engine was built with and loading warns on a mismatch.

Precision (TensorRT 11 only builds strongly typed networks: the precision
comes from the ONNX tensor types; with TensorRT 10 the builder flags are
used on the FP32 graph instead):
* ``fp32`` - the FP32 ONNX;
* ``fp16`` - the FP16 ONNX from ``precision.convert_fp16`` (TensorRT 11), or
  the FP32 ONNX with the FP16 builder flag (TensorRT 10);
* ``int8`` - explicit quantization: the symmetric Q/DQ ONNX from
  ``precision.quantize_int8`` (calibrated on validation frames). Implicit,
  calibrator-based INT8 is deprecated/removed and not used.

Inference uses PyTorch CUDA tensors as device buffers (no pycuda).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch

LOGGER = logging.getLogger(__name__)


def available() -> bool:
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def _device_info() -> dict:
    import tensorrt as trt

    props = torch.cuda.get_device_properties(0)
    return {"tensorrt": trt.__version__, "gpu": props.name, "compute_capability": f"{props.major}.{props.minor}",
            "cuda_runtime": torch.version.cuda}


def build_engine(onnx_path: str | Path, engine_path: str | Path, precision: str = "fp16",
                 workspace_mb: int = 1024, strongly_typed: bool = False) -> dict:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    weak_typing = hasattr(trt.BuilderFlag, "FP16")  # removed in TensorRT 11
    strongly_typed = strongly_typed or not weak_typing
    if strongly_typed and hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT could not parse {onnx_path}: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if precision == "fp32" and hasattr(trt.BuilderFlag, "TF32"):
        # TF32 tensor-core math is on by default on Ampere+; the FP32 engine is
        # the numerical reference, so use IEEE FP32.
        config.clear_flag(trt.BuilderFlag.TF32)
    if not strongly_typed:
        if precision in ("fp16", "int8"):
            config.set_flag(trt.BuilderFlag.FP16)
        if precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    return {"onnx": str(onnx_path), "engine": engine_path.name, "precision": precision,
            "strongly_typed": strongly_typed, "build_seconds": round(time.time() - t0, 1),
            "size_mb": round(engine_path.stat().st_size / 2**20, 2), **_device_info()}


def build_deployment(deployment_dir: str | Path, precision: str = "fp16", workspace_mb: int = 1024,
                     onnx_dir: str | Path | None = None) -> dict:
    """Build one engine per graph of ``deployment_dir`` (FP32 ONNX), or of
    ``onnx_dir`` (e.g. the INT8 QDQ or FP16 export) when given."""
    import tensorrt as trt

    root = Path(deployment_dir)
    strongly = not hasattr(trt.BuilderFlag, "FP16")
    if onnx_dir is None and precision == "fp16" and strongly:
        onnx_dir = root / "fp16"
        if not (onnx_dir / "deployment.json").exists():
            from tac_ufld.deploy.precision import convert_fp16

            convert_fp16(root, onnx_dir)
    if onnx_dir is None and precision == "int8":
        onnx_dir = root / "int8"
    src = Path(onnx_dir) if onnx_dir else root
    if not (src / "deployment.json").exists():
        raise FileNotFoundError(f"{src} has no deployment.json (for int8 run export with --int8 first)")
    manifest = json.loads((src / "deployment.json").read_text(encoding="utf-8"))
    out = root / "engines"
    info = {"precision": precision, "onnx_source": str(src), "graphs": {}}
    for name, graph in manifest["graphs"].items():
        info["graphs"][name] = build_engine(src / graph["file"], out / f"{name}.{precision}.engine", precision,
                                            workspace_mb, strongly_typed=strongly)
    (out / f"build_info.{precision}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


_DTYPES = {}


def _torch_dtype(trt_dtype):
    import tensorrt as trt

    if not _DTYPES:
        _DTYPES.update({trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
                        trt.int8: torch.int8, trt.bool: torch.bool})
    return _DTYPES[trt_dtype]


class _Engine:
    def __init__(self, path: Path) -> None:
        import tensorrt as trt

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"cannot deserialize {path} (built for another GPU / TensorRT version?)")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    def __call__(self, feeds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        stream = torch.cuda.current_stream()
        keep = []
        for name in self.inputs:
            t = feeds[name].to("cuda", dtype=_torch_dtype(self.engine.get_tensor_dtype(name))).contiguous()
            keep.append(t)
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())
        outs = {}
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            t = torch.empty(shape, dtype=_torch_dtype(self.engine.get_tensor_dtype(name)), device="cuda")
            self.context.set_tensor_address(name, t.data_ptr())
            outs[name] = t
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return outs


class TensorRTBackend:
    """Same interface as ``TorchBackend`` / ``OnnxRuntimeBackend``."""

    def __init__(self, deployment_dir: str | Path, precision: str = "fp16") -> None:
        root = Path(deployment_dir)
        manifest = json.loads((root / "deployment.json").read_text(encoding="utf-8"))
        info_path = root / "engines" / f"build_info.{precision}.json"
        if info_path.exists():
            built = json.loads(info_path.read_text(encoding="utf-8"))
            now = _device_info()
            first = next(iter(built["graphs"].values()))
            if (first["tensorrt"], first["gpu"]) != (now["tensorrt"], now["gpu"]):
                LOGGER.warning("engines were built with TensorRT %s on %s; running TensorRT %s on %s",
                               first["tensorrt"], first["gpu"], now["tensorrt"], now["gpu"])
        self.temporal = manifest["temporal_state"] is not None
        self.engines = {name: _Engine(root / "engines" / f"{name}.{precision}.engine") for name in manifest["graphs"]}
        self.name = f"tensorrt-{precision}"
        self.device = "cuda"

    @staticmethod
    def _t(x) -> torch.Tensor:
        return x if isinstance(x, torch.Tensor) else torch.from_numpy(x)

    def encode(self, frame) -> dict[str, torch.Tensor]:
        frame = self._t(frame)
        if not self.temporal:
            return {"current": frame}
        outs = self.engines["encoder"]({"frame": frame})
        return {"current": outs["current_features"],
                "history": outs.get("history_features", outs["current_features"])}

    def head(self, history: list, current) -> torch.Tensor:
        if not self.temporal:
            return self.engines["model"]({"frame": self._t(current)})["logits"].float()
        hist = torch.stack([self._t(h).to("cuda") for h in history], dim=1)
        return self.engines["head"]({"history_features": hist, "current_features": self._t(current)})["logits"].float()

    def full(self, clip) -> torch.Tensor:
        clip = self._t(clip)
        if not self.temporal:
            return self.engines["model"]({"frame": clip[:, -1]})["logits"].float()
        return self.engines["clip"]({"clip": clip})["logits"].float()

    def synchronize(self) -> None:
        torch.cuda.synchronize()
