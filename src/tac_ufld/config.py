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
import re
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_ENV = "TAC_UFLD_DATA_ROOT"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    """Raised for malformed or inconsistent configuration."""


def expand_env(value: str, context: str = "") -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}``. An unset variable without a
    default is an error (never a silent empty path)."""

    def repl(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ and os.environ[name] != "":
            return os.environ[name]
        if default is not None:
            return default
        where = f" (in {context})" if context else ""
        raise ConfigError(f"environment variable {name} is not set{where}")

    return _ENV_PATTERN.sub(repl, value)


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------


@dataclass
class GeometricAugConfig:
    """Geometric augmentation. Every transform is one homography applied to
    all frames of a clip AND to the lane polylines, so labels stay aligned.
    Disabled by default (baseline reproducibility)."""

    enabled: bool = False
    prob: float = 0.5
    translate_x: float = 0.0      # max |shift| as a fraction of the width
    translate_y: float = 0.0      # max |shift| as a fraction of the height
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0])   # isotropic zoom range
    rotate_deg: float = 0.0       # max |rotation| about the image centre
    perspective: float = 0.0      # max corner displacement as a fraction of the size
    crop_scale: list[float] = field(default_factory=lambda: [1.0, 1.0])  # crop side fraction, resized back
    hflip_prob: float = 0.0       # needs a dataset slot permutation (left <-> right)
    # A row may be labelled "no lane" only if at least this fraction of its
    # visible content comes from the annotated band (ELAS ROI); otherwise the
    # row is ignored. 1.0 = strict.
    min_row_valid: float = 1.0
    border: str = "constant"      # constant (black) | replicate | reflect


@dataclass
class AugmentationConfig:
    """Photometric augmentation applied identically to every frame of a clip.
    Fields up to ``frame_dropout_prob`` are the original (ELAS script)
    augmentation; the later photometric options are off by default so the
    random stream, and therefore every existing run, is unchanged."""

    enabled: bool = True
    prob: float = 0.8
    brightness: float = 0.18
    contrast: float = 0.18
    saturation: float = 0.15
    noise_std: float = 0.015
    erasing_prob: float = 0.15
    erasing_scale: list[float] = field(default_factory=lambda: [0.02, 0.08])
    frame_dropout_prob: float = 0.10
    gamma: float = 0.0            # gamma sampled log-uniformly in [1/(1+g), 1+g]
    hue: float = 0.0              # max hue rotation as a fraction of 360 degrees
    blur_prob: float = 0.0
    blur_sigma: list[float] = field(default_factory=lambda: [0.3, 1.5])
    motion_blur_prob: float = 0.0
    motion_blur_kernel: list[int] = field(default_factory=lambda: [3, 9])
    shadow_prob: float = 0.0
    shadow_strength: list[float] = field(default_factory=lambda: [0.3, 0.7])
    geometric: GeometricAugConfig = field(default_factory=GeometricAugConfig)
    # Current-frame degradation: with this probability ONLY the current frame
    # (the one whose lanes are the target) is occluded, blurred, darkened or
    # made noisy, while the history frames stay clean. A temporal model can
    # then only recover the lanes by using its history, so it learns to use
    # it. Applied to every model (single-frame baselines see the same
    # corrupted current frame), so the training recipe stays identical across
    # the pair. Sampled independently of ``prob``; 0 = off (no random draws).
    current_frame_prob: float = 0.0
    current_frame_ops: list[str] = field(default_factory=lambda: ["occlude", "blur", "darken", "noise"])
    current_occlusion_boxes: list[int] = field(default_factory=lambda: [1, 3])
    current_occlusion_scale: list[float] = field(default_factory=lambda: [0.03, 0.12])  # area fraction per box
    current_occlusion_band: list[float] = field(default_factory=lambda: [0.55, 1.0])    # box centres, fraction of height
    current_blur_sigma: list[float] = field(default_factory=lambda: [2.0, 5.0])
    current_darken: list[float] = field(default_factory=lambda: [0.15, 0.5])             # brightness factor
    current_noise_std: list[float] = field(default_factory=lambda: [0.08, 0.2])


CURRENT_FRAME_OPS = ("occlude", "blur", "darken", "noise")


HPO_PARAMS = ("lr", "lr_fusion", "weight_decay", "lambda_temporal", "lambda_coord", "lr_backbone_mult")
PREPROCESS_MODES = ("rgb", "gray", "gray3", "edge", "canny", "hough", "rgb_edge")
PRE_OPS = ("blur", "clahe", "equalize")


@dataclass
class PreprocessConfig:
    """Input representation (preprocessing ablations). ``rgb`` is the
    original behaviour. The same object is used for training, evaluation,
    streaming inference and deployment (it is stored in every checkpoint)."""

    mode: str = "rgb"             # see PREPROCESS_MODES
    pre_ops: list[str] = field(default_factory=list)  # applied in this order before the mode
    blur_ksize: int = 5
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    sobel_ksize: int = 3
    edge_normalization: str = "max"   # max (per image) | fixed
    edge_clip: float = 1.0            # 'fixed': magnitude / edge_clip, clipped to [0, 1]
    canny_low: int = 50
    canny_high: int = 150
    canny_l2: bool = True
    hough_threshold: int = 30
    hough_min_line_length: int = 20
    hough_max_line_gap: int = 10
    hough_thickness: int = 2
    hough_min_angle_deg: float = 15.0  # drop near-horizontal segments (not lane-like)
    edge_source: str = "sobel"        # rgb_edge: sobel | canny
    feature_mean: float = 0.5         # normalisation of edge/canny/hough channels
    feature_std: float = 0.5


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
    # Only for datasets whose official split has no validation set (TuSimple,
    # OpenLane): how validation is carved from the official training split.
    # "blocks" = temporal blocks within each split group + purge gap;
    # "sequences" = whole sequences held out.
    val_strategy: str = "blocks"


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
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
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
    # Optional dropout before the last Linear of the UFLD head (0 = official
    # UFLD, no dropout). Parameter-free, so state dicts are unchanged.
    head_dropout: float = 0.0
    # Hidden width of the UFLD fully connected head (official: 2048, about
    # 10 M of the 21.8 M parameters at 384x512). Smaller = fewer parameters to
    # memorise a small training set (an overfitting remedy, not official UFLD).
    ufld_head_hidden: int = 2048
    # Dropout of the lite models (values of the original ELAS script).
    lite_dropout: float = 0.05
    lite_head_dropout: float = 0.15
    # Initialise the models that are not warm-started (the baselines) from
    # another checkpoint instead of ImageNet only: an official UFLD
    # checkpoint (e.g. culane_18.pth / tusimple_18.pth) or one of our own
    # checkpoints trained on CULane / TuSimple / OpenLane. Every tensor whose
    # name and shape match is copied (the head usually differs: other lane
    # count, anchors and grid); ``init_scope: backbone`` copies the backbone
    # only. Temporal variants inherit it through their warm start.
    init_checkpoint: str | None = None
    init_scope: str = "backbone"       # backbone | all_matching
    # Aligned fusion (ufld_v06): maximum flow displacement in feature cells
    # (UFLD features have stride 32, so 4 cells = 128 px at 512 px width).
    aligned_max_disp: float = 4.0
    # Recurrent fusion (ufld_v07, lite_v06): ConvGRU hidden channels.
    recurrent_hidden: int = 64


@dataclass
class LossConfig:
    focal_gamma: float = 2.0
    sim_loss_w: float = 0.0   # official UFLD CULane config value
    shp_loss_w: float = 0.0   # official UFLD CULane config value
    # Uniform label smoothing of the focal classification term (0 = official).
    label_smoothing: float = 0.0


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
    # Write <variant>.last.pt every epoch so an interrupted run continues
    # mid-training with --resume (optimizer, scheduler, RNG and history).
    save_last: bool = True
    # Backbone learning rate = lr * lr_backbone_mult (head and fusion keep
    # theirs). < 1 keeps the pretrained features closer to ImageNet, a
    # standard remedy for overfitting on small datasets. 1 = original.
    lr_backbone_mult: float = 1.0
    # Freeze the first N backbone stages (weights fixed, BatchNorm statistics
    # frozen). ResNet: 1 = stem (conv1/bn1), 2 = + layer1, 3 = + layer2,
    # 4 = + layer3, 5 = whole backbone. Lite: N of the 4 conv blocks.
    freeze_backbone_stages: int = 0


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
    # Output-level Kalman tracker applied to every model's predictions (a
    # causal filter over the frames of each sequence), tuned on validation.
    # It is the cheap temporal reference a learned temporal model must beat.
    kalman: bool = True
    kalman_grid: dict[str, list[float]] = field(
        default_factory=lambda: {"q": [0.1, 1.0, 10.0], "alpha": [0.0, 0.5, 0.8]}
    )
    # Recurrent models are also evaluated frame by frame with the hidden
    # state carried across the whole sequence (longer memory than the
    # training window); "window" mode (exactly the training computation) is
    # the main result.
    carry_state_eval: bool = True


@dataclass
class ExperimentConfig:
    name: str = "elas_experiment"
    output_dir: str = "results"
    device: str = "auto"
    description: str = ""
    # Long (multi-day) configurations set this; `run` then refuses to start
    # without --confirm, so a full experiment is never launched by accident.
    requires_confirmation: bool = False
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
        """Dataset root: env override, absolute path, or relative to the project
        root. ``${VAR}`` / ``${VAR:-default}`` in ``data.root`` are expanded."""
        override = os.environ.get(DATA_ROOT_ENV)
        root = Path(override) if override else Path(expand_env(self.data.root, "data.root"))
        return root if root.is_absolute() else (PROJECT_ROOT / root).resolve()

    @property
    def in_channels(self) -> int:
        """Model input channels implied by the preprocessing mode."""
        from tac_ufld.data.preprocess import channels_for_mode  # local import avoids a cycle

        return channels_for_mode(self.data.preprocessing.mode)

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
            if name not in HPO_PARAMS:
                raise ConfigError(f"unsupported HPO parameter '{name}' (supported: {HPO_PARAMS})")
        m = self.model
        if m.init_scope not in ("backbone", "all_matching"):
            raise ConfigError("model.init_scope must be 'backbone' or 'all_matching'")
        if m.ufld_head_hidden < 8:
            raise ConfigError("model.ufld_head_hidden must be >= 8")
        if m.aligned_max_disp <= 0 or m.recurrent_hidden < 4:
            raise ConfigError("model.aligned_max_disp must be > 0 and model.recurrent_hidden >= 4")
        if t.lr_backbone_mult <= 0:
            raise ConfigError("train.lr_backbone_mult must be > 0")
        if not 0 <= t.freeze_backbone_stages <= 5:
            raise ConfigError("train.freeze_backbone_stages must be in [0, 5]")
        unknown_kf = set(e.kalman_grid) - {"q", "r", "alpha", "gate_px"}
        if unknown_kf or any(not v for v in e.kalman_grid.values()):
            raise ConfigError(f"evaluation.kalman_grid: unknown key(s) {sorted(unknown_kf)} or empty value list "
                              "(supported: q, r, alpha, gate_px)")
        from tac_ufld.data import KNOWN_DATASETS  # local import avoids a cycle

        if d.dataset.lower() not in KNOWN_DATASETS:
            raise ConfigError(f"data.dataset '{d.dataset}' unknown; supported: {list(KNOWN_DATASETS)}")
        if s.val_strategy not in ("blocks", "sequences"):
            raise ConfigError("data.split.val_strategy must be 'blocks' or 'sequences'")
        self._validate_preprocessing()
        self._validate_augmentation()
        for name in ("head_dropout", "lite_dropout", "lite_head_dropout"):
            if not 0 <= getattr(m, name) < 1:
                raise ConfigError(f"model.{name} must be in [0, 1)")
        if not 0 <= t.loss.label_smoothing < 1:
            raise ConfigError("train.loss.label_smoothing must be in [0, 1)")
        lane_metrics = ("lane_f1_", "lane_f2_", "lane_precision_", "lane_recall_")
        if not e.selection_metric.startswith(lane_metrics):
            raise ConfigError(
                f"evaluation.selection_metric '{e.selection_metric}' must be a lane-level metric "
                f"(lane_f1_iouXX, lane_f2_..., ...): the post-processing sweep only computes those"
            )
        tags = {f"iou{int(round(th * 100)):02d}" for th in e.iou_thresholds}
        if e.selection_metric.rsplit("_", 1)[-1] not in tags:
            raise ConfigError(
                f"evaluation.selection_metric '{e.selection_metric}' uses an IoU threshold that is "
                f"not in evaluation.iou_thresholds {e.iou_thresholds}"
            )

    def _validate_preprocessing(self) -> None:
        p = self.data.preprocessing
        if p.mode not in PREPROCESS_MODES:
            raise ConfigError(f"data.preprocessing.mode must be one of {PREPROCESS_MODES}")
        bad = [op for op in p.pre_ops if op not in PRE_OPS]
        if bad:
            raise ConfigError(f"unknown data.preprocessing.pre_ops {bad}; supported: {PRE_OPS}")
        if p.blur_ksize % 2 == 0 or p.blur_ksize < 1:
            raise ConfigError("data.preprocessing.blur_ksize must be an odd positive integer")
        if p.sobel_ksize not in (1, 3, 5, 7):
            raise ConfigError("data.preprocessing.sobel_ksize must be 1, 3, 5 or 7")
        if p.edge_normalization not in ("max", "fixed") or p.edge_clip <= 0:
            raise ConfigError("data.preprocessing.edge_normalization must be max|fixed, edge_clip > 0")
        if not 0 <= p.canny_low <= p.canny_high:
            raise ConfigError("data.preprocessing requires 0 <= canny_low <= canny_high")
        if p.edge_source not in ("sobel", "canny"):
            raise ConfigError("data.preprocessing.edge_source must be sobel|canny")
        if p.feature_std <= 0:
            raise ConfigError("data.preprocessing.feature_std must be > 0")

    def _validate_augmentation(self) -> None:
        a = self.data.augmentation
        g = a.geometric
        for name in ("prob", "erasing_prob", "frame_dropout_prob", "blur_prob", "motion_blur_prob", "shadow_prob"):
            if not 0 <= getattr(a, name) <= 1:
                raise ConfigError(f"data.augmentation.{name} must be in [0, 1]")
        if a.gamma < 0 or not 0 <= a.hue <= 0.5:
            raise ConfigError("data.augmentation.gamma must be >= 0 and hue in [0, 0.5]")
        for name in ("blur_sigma", "motion_blur_kernel", "shadow_strength", "erasing_scale"):
            lo, hi = getattr(a, name)
            if lo > hi or lo < 0:
                raise ConfigError(f"data.augmentation.{name} must be [low, high] with 0 <= low <= high")
        if not 0 <= g.prob <= 1 or not 0 <= g.hflip_prob <= 1:
            raise ConfigError("data.augmentation.geometric probabilities must be in [0, 1]")
        if not 0 <= g.translate_x < 0.5 or not 0 <= g.translate_y < 0.5:
            raise ConfigError("data.augmentation.geometric.translate_* must be in [0, 0.5)")
        if not 0 <= g.rotate_deg <= 30:
            raise ConfigError("data.augmentation.geometric.rotate_deg must be in [0, 30] (small rotations)")
        if not 0 <= g.perspective < 0.25:
            raise ConfigError("data.augmentation.geometric.perspective must be in [0, 0.25)")
        for name in ("scale", "crop_scale"):
            lo, hi = getattr(g, name)
            if not 0 < lo <= hi or (name == "crop_scale" and hi > 1):
                raise ConfigError(f"data.augmentation.geometric.{name} must be [low, high], 0 < low <= high"
                                  + (" <= 1" if name == "crop_scale" else ""))
        if not 0 <= a.current_frame_prob <= 1:
            raise ConfigError("data.augmentation.current_frame_prob must be in [0, 1]")
        bad_ops = [op for op in a.current_frame_ops if op not in CURRENT_FRAME_OPS]
        if bad_ops or (a.current_frame_prob > 0 and not a.current_frame_ops):
            raise ConfigError(f"data.augmentation.current_frame_ops must be a non-empty subset of {CURRENT_FRAME_OPS}")
        for name in ("current_occlusion_boxes", "current_occlusion_scale", "current_occlusion_band",
                     "current_blur_sigma", "current_darken", "current_noise_std"):
            lo, hi = getattr(a, name)
            if lo > hi or lo < 0:
                raise ConfigError(f"data.augmentation.{name} must be [low, high] with 0 <= low <= high")
        if a.current_occlusion_band[1] > 1 or a.current_occlusion_scale[1] >= 1 or a.current_darken[1] > 1:
            raise ConfigError("data.augmentation.current_occlusion_band/current_darken must be <= 1 and "
                              "current_occlusion_scale < 1")
        if not 0 < g.min_row_valid <= 1:
            raise ConfigError("data.augmentation.geometric.min_row_valid must be in (0, 1]")
        if g.border not in ("constant", "replicate", "reflect"):
            raise ConfigError("data.augmentation.geometric.border must be constant|replicate|reflect")


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
