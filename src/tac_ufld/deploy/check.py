"""PyTorch-versus-exported-model numerical checks.

Both backends run the same frames through the same streaming detector (same
preprocessing, cache and fallback rules), so the comparison covers exactly
what a deployment computes. Reported per precision:

* ``max_abs_logits``   clip graph (or single-frame graph) vs PyTorch logits;
* ``max_abs_exist``    existence probabilities, all cells;
* ``max_abs_x_px``     soft-argmax position (model pixels) where the PyTorch
                       existence is >= 0.5 (position is irrelevant elsewhere);
* ``lane_presence_disagreements`` slots drawn by one backend but not the other;
* ``mean_lane_dx_px``  mean horizontal distance of lanes drawn by both.

Tolerances (``TOLERANCES``) are engineering limits, not accuracy claims:
FP32 must match to rounding; FP16 within half a model pixel; INT8 is
approximate and must also be judged by task accuracy (``deploy_eval``).
"""

from __future__ import annotations

import numpy as np
import torch

from tac_ufld.inference.backends import TorchBackend
from tac_ufld.inference.loading import FramePreprocessor, LoadedModel
from tac_ufld.inference.streaming import StreamingLaneDetector

TOLERANCES = {
    "fp32": {"logits": 1e-3, "exist": 1e-4, "x_px": 0.05},
    "fp16": {"logits": 5e-2, "exist": 1e-2, "x_px": 0.5},
    "int8": {"logits": None, "exist": 0.15, "x_px": 3.0},
}


def synthetic_frames(n: int, size=(480, 640), seed: int = 0) -> list[np.ndarray]:
    """Road-like frames (lane markings that drift) when no real frames are given."""
    import cv2

    rng = np.random.default_rng(seed)
    h, w = size
    frames = []
    for i in range(n):
        img = (rng.normal(90, 12, (h, w, 3))).clip(0, 255).astype(np.uint8)
        for side in (-1, 1):
            x_bottom = w / 2 + side * 0.38 * w + 6 * np.sin(i / 3)
            cv2.line(img, (int(x_bottom), h - 1), (int(w / 2 + side * 0.04 * w), int(0.55 * h)), (240, 240, 240), 6)
        frames.append(img)
    return frames


def numerical_check(loaded: LoadedModel, backend, frames: list[np.ndarray], precision: str = "fp32",
                    reference: TorchBackend | None = None) -> dict:
    tol = TOLERANCES[precision]
    ref_backend = reference or TorchBackend(loaded.model, loaded.device)
    ref = StreamingLaneDetector(loaded, backend=ref_backend)
    cand = StreamingLaneDetector(loaded, backend=backend)
    max_exist = max_x = 0.0
    disagreements, dxs = 0, []
    for i, frame in enumerate(frames):
        r, c = ref.infer_frame(frame, frame_index=i), cand.infer_frame(frame, frame_index=i)
        max_exist = max(max_exist, float(np.abs(r.exist - c.exist).max()))
        on = r.exist >= 0.5
        if on.any():
            max_x = max(max_x, float(np.abs(r.x_model - c.x_model)[on].max()))
        for lr, lc in zip(r.lanes, c.lanes):
            if (lr is None) != (lc is None):
                disagreements += 1
            elif lr is not None:
                ys = np.intersect1d(np.round(lr[:, 1], 3), np.round(lc[:, 1], 3))
                if len(ys):
                    dxs.append(float(np.abs(np.interp(ys, lr[:, 1], lr[:, 0]) - np.interp(ys, lc[:, 1], lc[:, 0])).mean()))
    pre = FramePreprocessor(loaded.cfg)
    t = loaded.num_frames
    clip = torch.stack([pre(frames[min(k, len(frames) - 1)]) for k in range(t)]).unsqueeze(0)
    ref_logits, cand_logits = ref_backend.full(clip), backend.full(clip)  # single-frame models use clip[:, -1]
    max_logits = float(np.abs(np.asarray(_np(ref_logits)) - np.asarray(_np(cand_logits))).max())
    result = {
        "precision": precision, "frames": len(frames), "backend": getattr(backend, "name", type(backend).__name__),
        "reference": ref_backend.name, "max_abs_logits": max_logits, "max_abs_exist": max_exist,
        "max_abs_x_px": max_x, "lane_presence_disagreements": disagreements,
        "mean_lane_dx_px": float(np.mean(dxs)) if dxs else 0.0, "tolerances": tol,
    }
    checks = [max_exist <= tol["exist"], max_x <= tol["x_px"]]
    if tol["logits"] is not None:
        checks.append(max_logits <= tol["logits"])
    result["pass"] = bool(all(checks))
    return result


def deploy_eval(loaded: LoadedModel, backend, records: list, adapter) -> dict:
    """Task-level accuracy of a backend on dataset records (use VALIDATION
    records): each frame's temporal context is exactly the dataset's (same
    history paths and fallbacks), run through the backend's streaming split
    (encoder + head), scored with the experiment's evaluator and the model's
    post-processing. Comparing backends on the same records isolates the
    effect of export / precision."""
    from collections import OrderedDict

    from PIL import Image

    from tac_ufld.data.dataset import TemporalLaneDataset
    from tac_ufld.decoding import decode_logits
    from tac_ufld.evaluation.evaluator import Evaluator
    from tac_ufld.evaluation.predictor import Predictions

    cfg = loaded.cfg
    ds = TemporalLaneDataset(records, adapter, cfg.data, loaded.row_anchors, loaded.num_frames, augment=False)
    pre = FramePreprocessor(cfg)
    cache: OrderedDict = OrderedDict()
    exist_all, x_all = [], []
    for paths in ds.context_paths:
        encs = []
        for p in paths:
            if p not in cache:
                with Image.open(p) as img:
                    cache[p] = backend.encode(pre(np.asarray(img.convert("RGB"))).unsqueeze(0))
                if len(cache) > 32:
                    cache.popitem(last=False)
            encs.append(cache[p])
        history = [e.get("history", e["current"]) for e in encs[:-1]] if loaded.temporal else []
        logits = torch.as_tensor(_np(backend.head(history, encs[-1]["current"])))
        exist, bins = decode_logits(logits)
        exist_all.append(exist[0].numpy())
        x_all.append((bins[0] * (cfg.data.img_w - 1) / (logits.shape[1] - 2)).numpy())
    preds = Predictions(np.asarray(exist_all, np.float32), np.asarray(x_all, np.float32),
                        np.full(len(records), np.nan, np.float32))
    metrics = Evaluator(cfg, loaded.row_anchors).evaluate(preds, records, loaded.postprocess).metrics
    keys = [k for k in metrics if k.startswith(("lane_f1", "lane_precision", "lane_recall", "pixel_f1", "anchor_f1"))]
    return {"backend": getattr(backend, "name", type(backend).__name__), "frames": len(records),
            **{k: metrics[k] for k in keys}}


def _np(x) -> np.ndarray:
    return x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float32)
