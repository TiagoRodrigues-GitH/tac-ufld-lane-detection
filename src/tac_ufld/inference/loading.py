"""Load trained checkpoints for inference and discover them on disk."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tac_ufld.config import ExperimentConfig, _build, load_config
from tac_ufld.data.preprocess import Preprocessor, channel_stats, normalize_channels
from tac_ufld.data.targets import make_row_anchors
from tac_ufld.data.transforms import rgb_array_to_tensor
from tac_ufld.inference.card import model_card
from tac_ufld.models.registry import build_model
from tac_ufld.postprocess import PostprocessParams


@dataclass
class LoadedModel:
    model: nn.Module
    cfg: ExperimentConfig
    card: dict
    variant: str
    checkpoint: Path
    device: str
    postprocess: PostprocessParams
    postprocess_source: str
    meta: dict = field(default_factory=dict)
    kalman: dict | None = None      # validation-tuned output tracker parameters, if the run tuned them

    @property
    def temporal(self) -> bool:
        return bool(self.card["temporal"])

    @property
    def num_frames(self) -> int:
        return int(self.card["num_frames"])

    @property
    def model_size(self) -> tuple[int, int]:
        return self.cfg.data.img_w, self.cfg.data.img_h

    @property
    def row_anchors(self) -> np.ndarray:
        d = self.cfg.data
        return make_row_anchors(d.img_h, d.num_row_anchors, d.row_anchor_range)


def config_from_checkpoint(payload: dict, config_path: str | Path | None = None) -> ExperimentConfig:
    if "config" in payload:
        cfg = _build(ExperimentConfig, payload["config"])
        cfg.validate()
        return cfg
    if config_path is None:
        raise ValueError("this checkpoint predates self-describing checkpoints (no 'config' entry); "
                         "pass the YAML config it was trained with")
    return load_config(config_path)


def _tuned_postprocess(checkpoint: Path, variant: str) -> tuple[PostprocessParams, str] | None:
    """Validation-tuned post-processing written next to the checkpoint by the
    experiment (``seed_<n>/postprocess/<variant>_tuned.json``)."""
    path = checkpoint.parent.parent / "postprocess" / f"{variant}_tuned.json"
    if path.exists():
        params = json.loads(path.read_text(encoding="utf-8"))["params"]
        return PostprocessParams(**params), f"validation-tuned ({path.name})"
    return None


def _tuned_kalman(checkpoint: Path, variant: str) -> dict | None:
    path = checkpoint.parent.parent / "postprocess" / f"{variant}_tuned.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("kalman")
    return None


def load_model(checkpoint: str | Path, device: str = "cpu", config_path: str | Path | None = None) -> LoadedModel:
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = config_from_checkpoint(payload, config_path)
    variant = payload.get("variant") or checkpoint.stem
    model = build_model(variant, cfg, pretrained=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    card = payload.get("card") or model_card(cfg, variant)
    tuned = _tuned_postprocess(checkpoint, variant)
    if tuned is None:
        pp, source = PostprocessParams.from_config(cfg.evaluation.common_postprocess), "common fixed protocol"
    else:
        pp, source = tuned
    meta = {k: payload.get(k) for k in ("epoch", "score", "selection_metric", "seed", "version", "config_hash",
                                         "hyperparams")}
    return LoadedModel(model, cfg, card, variant, checkpoint, device, pp, source, meta,
                       kalman=_tuned_kalman(checkpoint, variant))


class FramePreprocessor:
    """Original RGB frame (H, W, 3, uint8) -> normalised model input (C, h, w),
    exactly the dataset pipeline without augmentation."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        d = cfg.data
        self.width, self.height = d.img_w, d.img_h
        self.preprocessor = Preprocessor(d.preprocessing)
        self.mean, self.std = channel_stats(d.preprocessing, d.normalize)

    def __call__(self, rgb: np.ndarray) -> torch.Tensor:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"expected an (H, W, 3) uint8 RGB frame, got {rgb.shape} {rgb.dtype}")
        frame = rgb_array_to_tensor(rgb, self.width, self.height)
        frame = self.preprocessor(frame.unsqueeze(0))[0]
        return normalize_channels(frame, self.mean, self.std)


def discover_checkpoints(results_root: str | Path) -> list[dict]:
    """Every ``<run>/seed_<n>/checkpoints/<variant>.pt`` below ``results_root``."""
    out = []
    root = Path(results_root)
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*/seed_*/checkpoints/*.pt")):
        if path.name.endswith(".last.pt"):
            continue
        out.append({"path": path, "run": path.parents[2].name, "seed": path.parents[1].name.removeprefix("seed_"),
                    "variant": path.stem, "size_mb": path.stat().st_size / 2**20})
    return out
