"""UI logic without Streamlit: discovery of runs and checkpoints, inference on
images and videos, architecture summaries, convention checks. Imports only
inference / visualisation code (tested: no training module is loaded)."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from tac_ufld.inference.loading import LoadedModel, discover_checkpoints, load_model
from tac_ufld.inference.streaming import FrameResult, StreamingLaneDetector
from tac_ufld.visualization.overlays import VideoSink, draw_prediction

CONVENTION_KEYS = ("input.height", "input.width", "input.channels", "input.preprocessing.mode", "num_lanes",
                   "num_row_anchors", "griding_num", "dataset")


# ------------------------------------------------------------------ discovery


def list_runs(results_root: str | Path) -> list[Path]:
    root = Path(results_root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and ((p / "all_results.csv").exists()
                                                               or any(p.glob("seed_*/checkpoints/*.pt"))))


def list_checkpoints(results_root: str | Path) -> pd.DataFrame:
    rows = discover_checkpoints(results_root)
    return pd.DataFrame(rows, columns=["path", "run", "seed", "variant", "size_mb"])


def checkpoint_label(row) -> str:
    return f"{row['run']} / seed {row['seed']} / {row['variant']}"


# ---------------------------------------------------------------- conventions


def _get(card: dict, dotted: str):
    node = card
    for key in dotted.split("."):
        node = node.get(key) if isinstance(node, dict) else None
    return node


def convention_warnings(models: list[LoadedModel]) -> list[str]:
    """Differences in input size, preprocessing or output layout between the
    selected models (their outputs are then not directly comparable)."""
    warnings = []
    if len(models) < 2:
        return warnings
    for key in CONVENTION_KEYS:
        values = {m.variant + f" ({m.checkpoint.parents[2].name})": _get(m.card, key) for m in models}
        if len({json.dumps(v, sort_keys=True, default=str) for v in values.values()}) > 1:
            warnings.append(f"`{key}` differs: " + ", ".join(f"{k}={v}" for k, v in values.items()))
    return warnings


# ------------------------------------------------------------------ inference


@dataclass
class ImagePrediction:
    variant: str
    overlay_bgr: np.ndarray
    result: FrameResult
    note: str


def predict_image(loaded: LoadedModel, rgb: np.ndarray) -> ImagePrediction:
    """Single image. Temporal models see the image as their whole history
    (static history), which is stated in the note."""
    det = StreamingLaneDetector(loaded)
    res = det.infer_frame(rgb, "ui-image", frame_index=0)
    res = det.infer_frame(rgb, "ui-image", frame_index=loaded.card["temporal_step"])  # warm, cached path timing
    note = ("temporal model on a single image: history frames = the same image (static history)"
            if loaded.temporal else "single-frame model")
    return ImagePrediction(loaded.variant, draw_prediction(rgb, res, loaded.card["display_name"]), res, note)


def lanes_table(pred: ImagePrediction) -> pd.DataFrame:
    rows = []
    for slot, (lane, conf) in enumerate(zip(pred.result.lanes, pred.result.lane_confidence)):
        rows.append({"model": pred.variant, "slot": slot, "detected": lane is not None,
                     "confidence (mean existence)": None if conf is None else round(conf, 3),
                     "points": 0 if lane is None else len(lane),
                     "x at bottom (px)": None if lane is None else round(float(lane[np.argmax(lane[:, 1]), 0]), 1)})
    return pd.DataFrame(rows)


def process_video(loaded: LoadedModel, video_path: Path, out_path: Path, max_frames: int | None = None,
                  every: int = 1, progress=None) -> dict:
    """Stream a video file frame by frame (never loaded whole into memory)
    and write an overlay MP4."""
    det = StreamingLaneDetector(loaded)
    sink = VideoSink(out_path, fps=max(1.0, 10.0 / every))
    latencies, fallbacks, n = [], 0, 0
    for rgb, res in det.infer_video(video_path, every=every, max_frames=max_frames):
        sink.write(draw_prediction(rgb, res, loaded.card["display_name"]))
        latencies.append(res.latency_ms)
        fallbacks += res.fallbacks
        n += 1
        if progress is not None and max_frames:
            progress(min(1.0, n / max_frames))
    sink.close()
    df = pd.DataFrame(latencies[3:] or latencies)
    return {"frames": n, "history_fallbacks": fallbacks, "video": str(out_path),
            **{f"{k}_mean": float(df[k].mean()) for k in df.columns}}


def save_upload(data: bytes, suffix: str) -> Path:
    """Write an uploaded file to a temporary file in chunks."""
    tmp = Path(tempfile.mkdtemp(prefix="tac_ufld_ui_")) / f"upload{suffix}"
    with open(tmp, "wb") as fh:
        for start in range(0, len(data), 1 << 20):
            fh.write(data[start:start + (1 << 20)])
    return tmp


# ---------------------------------------------------------------- architecture


def parameter_table(model: nn.Module) -> pd.DataFrame:
    rows = []
    for name, module in model.named_children():
        for sub_name, sub in [(name, module), *[(f"{name}.{n}", m) for n, m in module.named_children()]]:
            rows.append({"module": sub_name, "type": type(sub).__name__,
                         "parameters": sum(p.numel() for p in sub.parameters()),
                         "trainable": sum(p.numel() for p in sub.parameters() if p.requires_grad)})
    return pd.DataFrame(rows)


@torch.no_grad()
def layer_shapes(loaded: LoadedModel, max_rows: int = 400) -> pd.DataFrame:
    """Output shape of every leaf module for one forward pass (fallback when
    graph rendering is unavailable)."""
    model = loaded.model
    d = loaded.cfg.data
    x = torch.zeros(1, loaded.num_frames, loaded.card["input"]["channels"], d.img_h, d.img_w, device=loaded.device)
    rows, hooks = [], []

    def hook(name):
        def fn(mod, inp, out):
            if len(rows) < max_rows:
                shape = tuple(out.shape) if isinstance(out, torch.Tensor) else type(out).__name__
                rows.append({"module": name, "type": type(mod).__name__, "output shape": str(shape),
                             "parameters": sum(p.numel() for p in mod.parameters(recurse=False))})
        return fn

    for name, mod in model.named_modules():
        if name and not list(mod.children()):
            hooks.append(mod.register_forward_hook(hook(name)))
    try:
        model(x)
    finally:
        for h in hooks:
            h.remove()
    return pd.DataFrame(rows)


def graph_svg(loaded: LoadedModel) -> str | None:
    """torchview + Graphviz rendering when both are installed, else None."""
    try:
        import shutil

        from torchview import draw_graph

        if shutil.which("dot") is None:
            return None
        d = loaded.cfg.data
        x = torch.zeros(1, loaded.num_frames, loaded.card["input"]["channels"], d.img_h, d.img_w,
                        device=loaded.device)
        graph = draw_graph(loaded.model, input_data=x, depth=2, device=loaded.device, expand_nested=False)
        return graph.visual_graph.pipe(format="svg").decode("utf-8")
    except Exception:
        return None


# ------------------------------------------------------------------- results


def run_tables(run: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted((run / "report").glob("*.csv")):
        out[path.stem] = pd.read_csv(path)
    if (run / "all_results.csv").exists():
        out["all_results"] = pd.read_csv(run / "all_results.csv")
    return out


def histories(run: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run.glob("seed_*/history_*.csv")):
        df = pd.read_csv(path)
        df["seed"] = path.parent.name.removeprefix("seed_")
        df["variant"] = path.stem.removeprefix("history_")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


__all__ = ["checkpoint_label", "convention_warnings", "graph_svg", "histories", "layer_shapes", "lanes_table",
           "list_checkpoints", "list_runs", "load_model", "parameter_table", "predict_image", "process_video",
           "run_tables", "save_upload"]
