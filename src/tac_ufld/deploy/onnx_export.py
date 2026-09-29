"""ONNX export of trained models, including the streaming split of temporal models.

Exported graphs (batch 1, static shapes; ``deployment.json`` records every
name, shape and convention):

* single-frame models: ``model.onnx``   frame (1, C, H, W) -> logits
* temporal models:
    - ``encoder.onnx``  frame (1, C, H, W) -> current_features [, history_features]
      (the per-frame features that the runtime caches between calls);
    - ``head.onnx``     history_features (1, T-1, C', h, w), current_features
      (1, C', h, w) -> logits (fusion + classifier, run every frame);
    - ``clip.onnx``     clip (1, T, C, H, W) -> logits (non-streaming reference).

The temporal state is NOT inside the graphs: the caller keeps a ring buffer of
per-frame ``history_features`` and passes the T-1 entries selected by the
temporal rule (``tac_ufld.inference.streaming``). Exporting only a single-frame
graph of a temporal model would not be a streaming deployment.

``LanePixelHead`` uses ``AdaptiveAvgPool2d`` with output sizes that do not
divide the input (e.g. 24x32 -> 18x100), which ONNX exporters do not support;
during export it is replaced by an exactly equivalent separable averaging
(two matrix products), verified by the numerical check.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tac_ufld.inference.loading import LoadedModel

LOGGER = logging.getLogger(__name__)
OPSET = 17


# ------------------------------------------------------------------ helpers


def adaptive_pool_matrix(in_size: int, out_size: int) -> torch.Tensor:
    """(out, in) averaging matrix with PyTorch's adaptive pooling bins."""
    m = torch.zeros(out_size, in_size)
    for i in range(out_size):
        start = math.floor(i * in_size / out_size)
        end = math.ceil((i + 1) * in_size / out_size)
        m[i, start:end] = 1.0 / (end - start)
    return m


class MatmulAdaptiveAvgPool2d(nn.Module):
    """Exact replacement of ``AdaptiveAvgPool2d`` for a fixed input size."""

    def __init__(self, in_hw: tuple[int, int], out_hw: tuple[int, int]) -> None:
        super().__init__()
        self.register_buffer("ph", adaptive_pool_matrix(in_hw[0], out_hw[0]))
        self.register_buffer("pw", adaptive_pool_matrix(in_hw[1], out_hw[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.matmul(self.ph.to(x.dtype), x), self.pw.to(x.dtype).t())


@contextlib.contextmanager
def exportable(model: nn.Module, example_frame: torch.Tensor, num_history: int):
    """Temporarily swap non-exportable modules for exact equivalents."""
    swapped = []
    pools = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.AdaptiveAvgPool2d)]
    if pools:
        shapes = {}
        hooks = [m.register_forward_hook(lambda mod, inp, out, n=name: shapes.__setitem__(n, inp[0].shape[-2:]))
                 for name, m in pools]
        with torch.no_grad():  # one pass through every path records each pool's input size
            enc = model.encode_frames(example_frame)
            model.forward_features([enc.get("history", enc["current"])] * num_history, enc["current"])
        for h in hooks:
            h.remove()
        for name, m in pools:
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            attr = name.rsplit(".", 1)[-1]
            out = m.output_size if isinstance(m.output_size, tuple) else (m.output_size, m.output_size)
            setattr(parent, attr, MatmulAdaptiveAvgPool2d(tuple(shapes[name]), tuple(out)).to(example_frame.device))
            swapped.append((parent, attr, m))
    try:
        yield model
    finally:
        for parent, attr, original in swapped:
            setattr(parent, attr, original)


class EncoderGraph(nn.Module):
    def __init__(self, model: nn.Module, with_history: bool) -> None:
        super().__init__()
        self.model, self.with_history = model, with_history

    def forward(self, frame: torch.Tensor):
        enc = self.model.encode_frames(frame)
        return (enc["current"], enc["history"]) if self.with_history else enc["current"]


class HeadGraph(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, history_features: torch.Tensor, current_features: torch.Tensor) -> torch.Tensor:
        history = [history_features[:, i] for i in range(history_features.shape[1])]
        return self.model.forward_features(history, current_features)["logits"]


class LogitsGraph(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["logits"]


def _export(module: nn.Module, args: tuple, path: Path, input_names: list[str], output_names: list[str],
            opset: int) -> str:
    """Try the TorchScript exporter first (mature operator coverage, stable
    graphs for TensorRT), then the dynamo exporter.

    The wrapper is put in eval mode first: ``torch.onnx.export`` restores the
    wrapper's original mode afterwards, recursively, and a fresh wrapper
    starts in train mode - which would silently switch the wrapped model
    (BatchNorm, dropout) to training."""
    module.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for dynamo in (False, True):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(module, args, str(path), input_names=input_names, output_names=output_names,
                                  opset_version=opset, do_constant_folding=True, dynamo=dynamo,
                                  **({} if dynamo else {"training": torch.onnx.TrainingMode.EVAL}))
            import onnx

            onnx.checker.check_model(str(path))
            return "dynamo" if dynamo else "torchscript"
        except Exception as exc:  # try the other exporter, report both if neither works
            errors.append(f"{'dynamo' if dynamo else 'torchscript'}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"ONNX export of {path.name} failed:\n" + "\n".join(errors))


def onnx_ops(path: Path) -> list[str]:
    import onnx

    return sorted({node.op_type for node in onnx.load(str(path)).graph.node})


# --------------------------------------------------------------------- export


def export_model(loaded: LoadedModel, out_dir: str | Path, opset: int = OPSET) -> dict:
    """Export ``loaded`` to ``out_dir`` and write ``deployment.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = loaded.model.eval().float().cpu()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    d = loaded.cfg.data
    c = loaded.card["input"]["channels"]
    frame = torch.randn(1, c, d.img_h, d.img_w)
    graphs: dict[str, dict] = {}
    with exportable(model, frame, loaded.num_frames - 1), torch.no_grad():
        if not loaded.temporal:
            how = _export(LogitsGraph(model), (frame,), out / "model.onnx", ["frame"], ["logits"], opset)
            graphs["model"] = {"file": "model.onnx", "inputs": {"frame": list(frame.shape)},
                               "outputs": {"logits": list(model(frame)["logits"].shape)}, "exporter": how}
        else:
            enc = model.encode_frames(frame)
            separate = enc["history"] is not enc["current"]
            t = loaded.num_frames
            hist = torch.stack([enc["history"]] * (t - 1), dim=1)
            enc_outputs = ["current_features"] + (["history_features"] if separate else [])
            how = _export(EncoderGraph(model, separate), (frame,), out / "encoder.onnx", ["frame"], enc_outputs, opset)
            graphs["encoder"] = {"file": "encoder.onnx", "inputs": {"frame": list(frame.shape)},
                                 "outputs": {"current_features": list(enc["current"].shape),
                                             **({"history_features": list(enc["history"].shape)} if separate else {})},
                                 "exporter": how}
            how = _export(HeadGraph(model), (hist, enc["current"]), out / "head.onnx",
                          ["history_features", "current_features"], ["logits"], opset)
            logits_shape = list(model.forward_features(list(hist.unbind(1)), enc["current"])["logits"].shape)
            graphs["head"] = {"file": "head.onnx", "inputs": {"history_features": list(hist.shape),
                                                              "current_features": list(enc["current"].shape)},
                              "outputs": {"logits": logits_shape}, "exporter": how}
            clip = torch.randn(1, t, c, d.img_h, d.img_w)
            how = _export(LogitsGraph(model), (clip,), out / "clip.onnx", ["clip"], ["logits"], opset)
            graphs["clip"] = {"file": "clip.onnx", "inputs": {"clip": list(clip.shape)},
                              "outputs": {"logits": logits_shape}, "exporter": how}
    changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
    if model.training or changed:
        raise RuntimeError(f"export modified the model (training={model.training}, changed tensors: {changed[:5]})")
    loaded.model.to(loaded.device)
    for g in graphs.values():
        g["onnx_ops"] = onnx_ops(out / g["file"])
    manifest = {
        "format": "tac_ufld.deployment/1",
        "variant": loaded.variant,
        "source_checkpoint": str(loaded.checkpoint),
        "opset": opset,
        "precision": "fp32",
        "card": loaded.card,
        "postprocess": {"params": loaded.postprocess.as_dict(), "source": loaded.postprocess_source},
        "graphs": graphs,
        "temporal_state": None if not loaded.temporal else {
            "description": "ring buffer of per-frame history_features (or current_features when there is no "
                           "separate history output), keyed by frame index; history of frame t = frames "
                           "t - k*temporal_step, k = T-1..1; a missing frame is replaced by the nearest newer "
                           "frame of the context (training rule); reset on sequence change / size change",
            "num_frames": loaded.num_frames,
            "temporal_step": loaded.card["temporal_step"],
            "feature_shape": graphs["encoder"]["outputs"].get("history_features",
                                                              graphs["encoder"]["outputs"]["current_features"]),
            "bytes_per_cached_frame": int(np.prod(graphs["encoder"]["outputs"].get(
                "history_features", graphs["encoder"]["outputs"]["current_features"])) * 4),
        },
        "config": loaded.cfg.to_dict(),
    }
    (out / "deployment.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    LOGGER.info("exported %s to %s (%s)", loaded.variant, out, ", ".join(graphs))
    return manifest
