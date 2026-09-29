"""Typed experiment configuration loaded from YAML.

Every tunable number lives in a YAML file under ``configs/``. This module
turns it into nested dataclasses, rejects unknown keys (so a typo can never
silently fall back to a default), validates cross-field constraints, and
produces a stable hash used to tag outputs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_ENV = "TAC_UFLD_DATA_ROOT"


class ConfigError(ValueError):
    """Raised for malformed or inconsistent configuration."""


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------


@dataclass
class AugmentationConfig:
    """Photometric augmentation applied identically to every frame of a clip."""

    enabled: bool = True
    prob: float = 0.8
    brightness: float = 0.18
    contrast: float = 0.18
    saturation: float = 0.15
    noise_std: float = 0.015
    erasing_prob: float = 0.15
    erasing_scale: list[float] = field(default_factory=lambda: [0.02, 0.08])
    frame_dropout_prob: float = 0.10


@dataclass
class SplitConfig:
    """Leakage-safe split protocol (ELAS). Independent of the training seed."""

    split_seed: int = 2026
    test_scenes: list[str] = field(default_factory=list)
    block_size: int = 60
    purge_frames: int = 10
    val_fraction: float = 0.2
    seen_test_fraction: float = 0.15
    max_train_frames: int | None = None
    max_val_frames: int | None = None
    max_test_frames: int | None = None
    max_seen_test_frames: int | None = None


@dataclass
class DataConfig:
    dataset: str = "elas"
    root: str = "../../datasets/dataset_elas_v1"
    scenes: list[str] = field(default_factory=list)
    scene_tags: dict[str, dict[str, bool]] = field(default_factory=dict)
    img_h: int = 384
    img_w: int = 512
    num_lanes: int = 2
    num_row_anchors: int = 18
    row_anchor_range: list[float] = field(default_factory=lambda: [0.59, 0.995])
    griding_num: int = 100
    num_frames: int = 3
    temporal_step: int = 2
    normalize: str = "imagenet"
    include_frames_without_lanes: bool = False
    num_workers: int = 2
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)


@dataclass
class ModelConfig:
    variants: list[str] = field(
        default_factory=lambda: [
            "ufld_baseline", "ufld_v02", "ufld_v03", "ufld_v04",
            "lite_baseline", "lite_v05",
        ]
    )
    backbone: str = "18"
    pretrained: bool = True
    # "baseline": v04 warm-starts from the UFLD baseline like v02/v03 (fair:
    # equal training budget). "v02": the supervisor's original protocol.
    v04_warm_start: str = "baseline"
    # "shared": v05 encodes history frames with the same backbone as the
    # current frame (isolates the fusion effect). "tiny": the cheaper
    # depthwise-separable history encoder from the original ELAS script.
    lite_history_encoder: str = "shared"


@dataclass
class LossConfig:
    focal_gamma: float = 2.0
    sim_loss_w: float = 0.0   # official UFLD CULane config value
    shp_loss_w: float = 0.0   # official UFLD CULane config value


@dataclass
class TrainConfig:
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    epochs: int = 50
    batch_size: int = 8
    optimizer: str = "adam"
    lr: float = 4.0e-4
    lr_fusion: float = 5.0e-4
    weight_decay: float = 1.0e-4
    momentum: float = 0.9
    scheduler: str = "cos"
    warmup_iters: int = 100
    multi_steps: list[int] = field(default_factory=lambda: [25, 38])
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 10
    amp: bool = True
    deterministic: bool = False
    lambda_temporal: float = 0.01
    lambda_coord: float = 0.35
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class SearchParam:
    low: float
    high: float
    log: bool = False
    variants: list[str] | None = None  # None -> applies to every variant

    def applies_to(self, variant: str) -> bool:
        return self.variants is None or variant in self.variants


@dataclass
class HPOConfig:
    enabled: bool = True
    n_trials: int = 20
    epochs: int = 10
    sampler_seed: int = 42
    seed: int = 42
    max_train_frames: int | None = 2000
    variants: list[str] | None = None  # None -> every configured variant
    load_best_params: str | None = None  # path to a previous best_params.json
    search_space: dict[str, SearchParam] = field(default_factory=dict)


@dataclass
class PostprocessConfig:
    threshold: float = 0.5
    min_points: int = 3
    poly_degree: int = 0
    duplicate_distance: float = 10.0
    y_step: float = 2.0


@dataclass
class EvalConfig:
    iou_thresholds: list[float] = field(default_factory=lambda: [0.5, 0.35])
    culane_line_width: float = 30.0
    culane_image_width: float = 1640.0
    anchor_tolerance_px: float = 10.0
    selection_metric: str = "lane_f1_iou50"
    common_postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    postprocess_grid: dict[str, list[float]] = field(
        default_factory=lambda: {
            "threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
            "min_points": [2, 3, 4],
            "poly_degree": [0, 1, 2],
        }
    )
    latency_runs: int = 50
    n_visual_examples: int = 12
    visualize_seeds: int = 1
    tensorboard: bool = True


@dataclass
class ExperimentConfig:
    name: str = "elas_experiment"
    output_dir: str = "results"
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # -- derived helpers -----------------------------------------------------

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def data_root(self) -> Path:
        """Dataset root: env override, absolute path, or relative to the project root."""
        override = os.environ.get(DATA_ROOT_ENV)
        root = Path(override) if override else Path(self.data.root)
        return root if root.is_absolute() else (PROJECT_ROOT / root).resolve()

    def output_root(self) -> Path:
        out = Path(self.output_dir)
        out = out if out.is_absolute() else PROJECT_ROOT / out
        return out / self.name

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

    def min_split_gap(self) -> int:
        """Minimum frame gap between splits: at least the temporal context span."""
        span = self.data.temporal_step * (self.data.num_frames - 1)
        return max(self.data.split.purge_frames, span)

    def validate(self) -> None:
        from tac_ufld.models.registry import VARIANTS  # local import avoids a cycle

        d, s, t, e = self.data, self.data.split, self.train, self.evaluation
        span = d.temporal_step * (d.num_frames - 1)
        if s.purge_frames < span:
            raise ConfigError(
                f"split.purge_frames={s.purge_frames} must be >= temporal context span "
                f"temporal_step*(num_frames-1)={span}, otherwise a validation/test "
                f"sample's history frames can be training images."
            )
        if not 0 <= s.val_fraction < 1 or not 0 <= s.seen_test_fraction < 1:
            raise ConfigError("split fractions must be in [0, 1)")
        if s.val_fraction + s.seen_test_fraction >= 1:
            raise ConfigError("val_fraction + seen_test_fraction must be < 1")
        unknown = [v for v in self.model.variants if v not in VARIANTS]
        if unknown:
            raise ConfigError(f"unknown model variants {unknown}; known: {sorted(VARIANTS)}")
        if self.model.v04_warm_start not in ("baseline", "v02"):
            raise ConfigError("model.v04_warm_start must be 'baseline' or 'v02'")
        if self.model.lite_history_encoder not in ("shared", "tiny"):
            raise ConfigError("model.lite_history_encoder must be 'shared' or 'tiny'")
        if d.normalize not in ("imagenet", "none"):
            raise ConfigError("data.normalize must be 'imagenet' or 'none'")
        if t.optimizer not in ("adam", "adamw", "sgd"):
            raise ConfigError("train.optimizer must be adam|adamw|sgd")
        if t.scheduler not in ("cos", "multi", "none"):
            raise ConfigError("train.scheduler must be cos|multi|none")
        if not e.iou_thresholds:
            raise ConfigError("evaluation.iou_thresholds must not be empty")
        lo, hi = d.row_anchor_range
        if not 0 <= lo < hi <= 1:
            raise ConfigError("data.row_anchor_range must satisfy 0 <= low < high <= 1")
        if d.img_h % 32 or d.img_w % 32:
            raise ConfigError("img_h and img_w must be multiples of 32 (ResNet stride)")
        if d.num_frames < 1:
            raise ConfigError("data.num_frames must be >= 1")
        for name in self.hpo.search_space:
            if name not in ("lr", "lr_fusion", "weight_decay", "lambda_temporal", "lambda_coord"):
                raise ConfigError(f"unsupported HPO parameter '{name}'")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


def _build(cls: type, data: Any) -> Any:
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"unknown key(s) {sorted(unknown)} in section {cls.__name__}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce(hints[f.name], data[f.name])
    return cls(**kwargs)


def _coerce(tp: Any, value: Any) -> Any:
    if dataclasses.is_dataclass(tp):
        return _build(tp, value)
    origin = typing.get_origin(tp)
    if origin is dict:
        _, val_t = typing.get_args(tp)
        if dataclasses.is_dataclass(val_t):
            return {k: _build(val_t, v) for k, v in (value or {}).items()}
    return value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load and validate a YAML config. ``overrides`` uses dotted keys, e.g.
    ``{"train.seeds": [1, 2]}``."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    cfg = _build(ExperimentConfig, raw)
    cfg.validate()
    return cfg


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
