"""ONNX Runtime backend for exported deployments (``deployment.json``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tac_ufld.inference.backends import to_numpy


_CUDA_WORKS: bool | None = None


def _cuda_provider_works() -> bool:
    """A provider can be listed but fail to load (e.g. onnxruntime-gpu built
    for CUDA 13 next to a CUDA 12 PyTorch). Probe once with a tiny graph."""
    global _CUDA_WORKS
    if _CUDA_WORKS is None:
        import onnxruntime as ort
        from onnx import TensorProto, helper

        graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "probe",
                                  [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        ort.set_default_logger_severity(4)
        try:
            sess = ort.InferenceSession(model.SerializeToString(),
                                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            _CUDA_WORKS = sess.get_providers()[0] == "CUDAExecutionProvider"
        except Exception:
            _CUDA_WORKS = False
        finally:
            ort.set_default_logger_severity(2)
    return _CUDA_WORKS


def ort_providers(prefer_gpu: bool = True) -> list[str]:
    """CUDA first when the installed ONNX Runtime can actually load it. Import
    torch before ONNX Runtime so its CUDA/cuDNN DLLs are already loaded (Windows)."""
    import torch  # noqa: F401  (loads the CUDA runtime DLLs shared with ONNX Runtime)
    import onnxruntime as ort

    providers = []
    if (prefer_gpu and "CUDAExecutionProvider" in ort.get_available_providers() and torch.cuda.is_available()
            and _cuda_provider_works()):
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


class OnnxRuntimeBackend:
    def __init__(self, deployment_dir: str | Path, providers: list[str] | None = None,
                 graph_files: dict[str, str] | None = None) -> None:
        import onnxruntime as ort

        self.dir = Path(deployment_dir)
        self.manifest = json.loads((self.dir / "deployment.json").read_text(encoding="utf-8"))
        self.temporal = self.manifest["temporal_state"] is not None
        self.providers = providers or ort_providers()
        files = {name: g["file"] for name, g in self.manifest["graphs"].items()}
        files.update(graph_files or {})
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.sessions = {name: ort.InferenceSession(str(self.dir / f), sess_options=opts, providers=self.providers)
                         for name, f in files.items()}
        self.active_providers = {name: s.get_providers()[0] for name, s in self.sessions.items()}
        self.name = f"onnxruntime-{'cuda' if 'CUDAExecutionProvider' in self.active_providers.values() else 'cpu'}"

    @staticmethod
    def _f32(x) -> np.ndarray:
        return np.ascontiguousarray(to_numpy(x), dtype=np.float32)

    def encode(self, frame) -> dict[str, np.ndarray]:
        frame = self._f32(frame)
        if not self.temporal:
            return {"current": frame}  # model.onnx runs in head()
        outs = self.sessions["encoder"].run(None, {"frame": frame})
        return {"current": outs[0], "history": outs[1] if len(outs) > 1 else outs[0]}

    def head(self, history: list, current) -> np.ndarray:
        if not self.temporal:
            return self.sessions["model"].run(None, {"frame": self._f32(current)})[0]
        hist = np.stack([self._f32(h) for h in history], axis=1)
        return self.sessions["head"].run(None, {"history_features": hist,
                                                "current_features": self._f32(current)})[0]

    def full(self, clip) -> np.ndarray:
        clip = self._f32(clip)
        if not self.temporal:
            return self.sessions["model"].run(None, {"frame": clip[:, -1]})[0]
        return self.sessions["clip"].run(None, {"clip": clip})[0]

    def synchronize(self) -> None:  # ONNX Runtime calls are synchronous
        return None
