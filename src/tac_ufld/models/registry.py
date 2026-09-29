"""Variant registry: the single place that defines what each model is.

Loss constants for v02/v03/v04 are the supervisor's values from
``main_loss`` in the reference notebook. ``reference`` names the
single-frame model a temporal variant is statistically paired with: each
temporal model is compared against the baseline that shares its backbone,
so the comparison isolates the temporal pathway (audit finding C3).
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from tac_ufld.config import ExperimentConfig


@dataclass(frozen=True)
class ExistBCE:
    weight: float
    neg_weight: float
    pos_weight: float


@dataclass(frozen=True)
class VariantSpec:
    key: str
    display_name: str
    family: str                       # "ufld" | "lite"
    temporal: bool
    reference: str | None             # paired single-frame model for statistics
    warm_start: str | None            # variant whose trained weights initialise this one
    exist_bce: ExistBCE | None = None
    gate_prior: tuple[float, ...] | None = None
    gate_prior_weight: float = 0.0
    coord_loss: bool = False
    temporal_consistency: bool = False
    flow_smoothness_weight: float = 0.0
    origin: str = ""


VARIANTS: dict[str, VariantSpec] = {
    "ufld_baseline": VariantSpec(
        "ufld_baseline", "UFLD baseline (ResNet-18)", "ufld", False, None, None,
        origin="supervisor notebook: OfficialUFLDAdapter / official parsingNet",
    ),
    "ufld_v02": VariantSpec(
        "ufld_v02", "UFLD v0.2 weighted fusion", "ufld", True, "ufld_baseline", "ufld_baseline",
        exist_bce=ExistBCE(1.2, neg_weight=2.5, pos_weight=1.0), temporal_consistency=True,
        origin="supervisor notebook: TACUFLDTemporalModel + main_loss(mode='v02')",
    ),
    "ufld_v03": VariantSpec(
        "ufld_v03", "UFLD v0.3 gated fusion", "ufld", True, "ufld_baseline", "ufld_baseline",
        exist_bce=ExistBCE(1.0, neg_weight=1.7, pos_weight=1.15),
        gate_prior=(0.15, 0.25, 0.60), gate_prior_weight=0.01, temporal_consistency=True,
        origin="supervisor notebook: TACUFLDGatedTemporalModel + main_loss(mode='v03')",
    ),
    "ufld_v04": VariantSpec(
        "ufld_v04", "UFLD v0.4 coord loss", "ufld", True, "ufld_baseline", "ufld_baseline",
        exist_bce=ExistBCE(1.0, neg_weight=1.25, pos_weight=1.85), coord_loss=True,
        temporal_consistency=True,
        origin="supervisor notebook: TACUFLDTemporalModel + main_loss(mode='v04')",
    ),
    "lite_baseline": VariantSpec(
        "lite_baseline", "Lite single-frame", "lite", False, None, None,
        origin="ELAS script: SingleFrameUFLDLikeModel (LightweightBackbone + LanePixelHead)",
    ),
    "lite_v05": VariantSpec(
        "lite_v05", "Lite v0.5 warped fusion", "lite", True, "lite_baseline", "lite_baseline",
        temporal_consistency=True, flow_smoothness_weight=0.01,
        origin="ELAS script: TACUFLDTemporalModel (LearnableFlowWarp + ResidualTemporalFusion)",
    ),
}


def resolve_spec(key: str, cfg: ExperimentConfig) -> VariantSpec:
    """Apply config-dependent choices (v04 warm-start source)."""
    spec = VARIANTS[key]
    if key == "ufld_v04" and cfg.model.v04_warm_start == "v02":
        from dataclasses import replace

        spec = replace(spec, warm_start="ufld_v02")
    return spec


def training_order(keys: list[str], cfg: ExperimentConfig) -> list[str]:
    """Variants sorted so every warm-start source is trained first. Missing
    dependencies are added automatically."""
    ordered: list[str] = []

    def visit(key: str) -> None:
        if key in ordered:
            return
        dep = resolve_spec(key, cfg).warm_start
        if dep:
            visit(dep)
        ordered.append(key)

    for key in keys:
        visit(key)
    return ordered


def build_model(key: str, cfg: ExperimentConfig, pretrained: bool | None = None) -> nn.Module:
    """Instantiate a variant. ``pretrained`` defaults to the config value;
    warm-started variants skip the ImageNet download (weights are overwritten)."""
    from tac_ufld.models.lite import LanePixelHead, LightweightBackbone, LiteSingleFrame, LiteWarpTemporal
    from tac_ufld.models.ufld import (
        GatedTemporalFusion, TemporalWeightedFusion, UFLDNet, UFLDSingleFrame, UFLDTemporal,
    )

    spec = resolve_spec(key, cfg)
    d, m = cfg.data, cfg.model
    in_channels = cfg.in_channels
    if pretrained is None:
        pretrained = m.pretrained and spec.warm_start is None
    if spec.family == "ufld":
        ufld = UFLDNet(d.num_lanes, d.num_row_anchors, d.griding_num, d.img_h, d.img_w,
                       backbone=m.backbone, pretrained=pretrained, in_channels=in_channels,
                       head_dropout=m.head_dropout)
        if not spec.temporal:
            return UFLDSingleFrame(ufld)
        fusion = (GatedTemporalFusion(512, d.num_frames) if key == "ufld_v03"
                  else TemporalWeightedFusion(d.num_frames))
        return UFLDTemporal(ufld, fusion)
    backbone = LightweightBackbone(in_channels, 128, dropout=m.lite_dropout)
    head = LanePixelHead(128, d.num_lanes, d.num_row_anchors, d.griding_num, dropout=m.lite_head_dropout)
    if not spec.temporal:
        return LiteSingleFrame(backbone, head)
    return LiteWarpTemporal(backbone, head, d.num_frames, history_encoder=m.lite_history_encoder,
                            in_channels=in_channels, dropout=m.lite_dropout)


def warm_start(target: nn.Module, source_state: dict) -> list[str]:
    """Copy trained weights into ``target``. Identical architecture (same
    state-dict keys, e.g. v02 -> v04) -> full strict load; otherwise every
    shared top-level sub-module (``ufld``, ``backbone``, ``head``) is loaded
    strictly. Returns the list of loaded sub-modules."""
    if set(source_state) == set(target.state_dict()):
        target.load_state_dict(source_state, strict=True)
        return ["<all>"]
    loaded = []
    for name in ("ufld", "backbone", "head"):
        sub = getattr(target, name, None)
        prefix = f"{name}."
        part = {k[len(prefix):]: v for k, v in source_state.items() if k.startswith(prefix)}
        if sub is not None and part:
            sub.load_state_dict(part, strict=True)
            loaded.append(name)
    if not loaded:
        raise RuntimeError("warm start found no shared sub-module between source and target")
    return loaded
