"""Model card: every convention needed to run a trained model outside the
training pipeline (UI, streaming, ONNX / TensorRT). Stored in each checkpoint
and in ``deployment.json`` next to exported models."""

from __future__ import annotations

import dataclasses

from tac_ufld.config import ExperimentConfig
from tac_ufld.data.preprocess import channel_stats, channels_for_mode
from tac_ufld.data.targets import make_row_anchors
from tac_ufld.models.registry import resolve_spec


def model_card(cfg: ExperimentConfig, variant: str) -> dict:
    d = cfg.data
    spec = resolve_spec(variant, cfg)
    mean, std = channel_stats(d.preprocessing, d.normalize)
    anchors = make_row_anchors(d.img_h, d.num_row_anchors, d.row_anchor_range)
    return {
        "variant": variant,
        "display_name": spec.display_name,
        "family": spec.family,
        "temporal": spec.temporal,
        "num_frames": d.num_frames if spec.temporal else 1,
        "temporal_step": d.temporal_step,
        "dataset": d.dataset,
        "input": {
            "height": d.img_h, "width": d.img_w,
            "channels": channels_for_mode(d.preprocessing.mode),
            "channel_order": "RGB" if d.preprocessing.mode in ("rgb", "rgb_edge") else d.preprocessing.mode,
            "resize": "PIL bilinear from the original frame to (width, height)",
            "value_range_before_norm": "[0, 1]",
            "mean": mean, "std": std,
            "preprocessing": dataclasses.asdict(d.preprocessing),
        },
        "output": {
            "logits_shape": ["B", d.griding_num + 1, d.num_row_anchors, d.num_lanes],
            "layout": "(batch, griding_num + 1 classes, row anchors, lane slots); last class = no lane",
            "existence": "1 - softmax(logits)[:, -1]",
            "location": "soft-argmax over the first griding_num classes; "
                        "x_model = bin * (width - 1) / (griding_num - 1)",
            "row_anchors_model_px": [float(a) for a in anchors],
        },
        "num_lanes": d.num_lanes,
        "griding_num": d.griding_num,
        "num_row_anchors": d.num_row_anchors,
        "row_anchor_range": list(d.row_anchor_range),
    }
