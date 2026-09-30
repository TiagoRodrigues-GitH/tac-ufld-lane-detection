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
    fusion: str = ""                  # weighted | gated | warp | aligned | recurrent ("" = single frame)
    # Equal-training control: the single-frame model that received the SAME
    # extra training from the same starting checkpoint (``*_baseline_ct``).
    # A temporal gain over it cannot come from more epochs.
    budget_reference: str | None = None
    # "real" history, or "current": the clip is the current frame repeated
    # (capacity control: same layers and training, no temporal information).
    history: str = "real"
    # The capacity-control twin of a temporal model (``*_static``): a gain
    # over it cannot come from the extra fusion layers.
    static_reference: str | None = None


VARIANTS: dict[str, VariantSpec] = {
    "ufld_baseline": VariantSpec(
        "ufld_baseline", "UFLD baseline (ResNet-18)", "ufld", False, None, None,
        origin="supervisor notebook: OfficialUFLDAdapter / official parsingNet",
    ),
    "ufld_baseline_ct": VariantSpec(
        "ufld_baseline_ct", "UFLD baseline + continued training", "ufld", False, None, "ufld_baseline",
        origin="v0.4 control: the baseline trained again from its own best checkpoint with the temporal "
               "variants' budget",
    ),
    "ufld_v02": VariantSpec(
        "ufld_v02", "UFLD v0.2 weighted fusion", "ufld", True, "ufld_baseline", "ufld_baseline",
        exist_bce=ExistBCE(1.2, neg_weight=2.5, pos_weight=1.0), temporal_consistency=True,
        origin="supervisor notebook: TACUFLDTemporalModel + main_loss(mode='v02')",
        fusion="weighted", budget_reference="ufld_baseline_ct",
    ),
    "ufld_v03": VariantSpec(
        "ufld_v03", "UFLD v0.3 gated fusion", "ufld", True, "ufld_baseline", "ufld_baseline",
        exist_bce=ExistBCE(1.0, neg_weight=1.7, pos_weight=1.15),
        gate_prior=(0.15, 0.25, 0.60), gate_prior_weight=0.01, temporal_consistency=True,
        origin="supervisor notebook: TACUFLDGatedTemporalModel + main_loss(mode='v03')",
        fusion="gated", budget_reference="ufld_baseline_ct",
    ),
    "ufld_v04": VariantSpec(
        "ufld_v04", "UFLD v0.4 coord loss", "ufld", True, "ufld_baseline", "ufld_baseline",
        exist_bce=ExistBCE(1.0, neg_weight=1.25, pos_weight=1.85), coord_loss=True,
        temporal_consistency=True,
        origin="supervisor notebook: TACUFLDTemporalModel + main_loss(mode='v04')",
        fusion="weighted", budget_reference="ufld_baseline_ct",
    ),
    "ufld_v06": VariantSpec(
        "ufld_v06", "UFLD v0.6 aligned fusion", "ufld", True, "ufld_baseline", "ufld_baseline",
        temporal_consistency=True, flow_smoothness_weight=0.01,
        origin="v0.4: lite v0.5's LearnableFlowWarp + ResidualTemporalFusion on UFLD layer-4 features",
        fusion="aligned", budget_reference="ufld_baseline_ct",
    ),
    "ufld_v07": VariantSpec(
        "ufld_v07", "UFLD v0.7 recurrent (ConvGRU)", "ufld", True, "ufld_baseline", "ufld_baseline",
        temporal_consistency=True,
        origin="v0.4: ConvGRU over UFLD layer-4 features, zero-initialised residual read-out",
        fusion="recurrent", budget_reference="ufld_baseline_ct",
    ),
    "lite_baseline": VariantSpec(
        "lite_baseline", "Lite single-frame", "lite", False, None, None,
        origin="ELAS script: SingleFrameUFLDLikeModel (LightweightBackbone + LanePixelHead)",
    ),
    "lite_baseline_ct": VariantSpec(
        "lite_baseline_ct", "Lite baseline + continued training", "lite", False, None, "lite_baseline",
        origin="v0.4 control: the lite baseline trained again from its own best checkpoint with the temporal "
               "variants' budget",
    ),
    "lite_v05": VariantSpec(
        "lite_v05", "Lite v0.5 warped fusion", "lite", True, "lite_baseline", "lite_baseline",
        temporal_consistency=True, flow_smoothness_weight=0.01,
        origin="ELAS script: TACUFLDTemporalModel (LearnableFlowWarp + ResidualTemporalFusion)",
        fusion="warp", budget_reference="lite_baseline_ct", static_reference="lite_v05_static",
    ),
    "lite_v05_static": VariantSpec(
        "lite_v05_static", "Lite v0.5 static history (capacity control)", "lite", True, "lite_baseline",
        "lite_baseline", temporal_consistency=True, flow_smoothness_weight=0.01,
        origin="v0.4 control: lite v0.5 with every history frame replaced by the current frame",
        fusion="warp", budget_reference="lite_baseline_ct", history="current",
    ),
    "lite_v06": VariantSpec(
        "lite_v06", "Lite v0.6 recurrent (ConvGRU)", "lite", True, "lite_baseline", "lite_baseline",
        temporal_consistency=True,
        origin="v0.4: ConvGRU over lite backbone features, zero-initialised residual read-out",
        fusion="recurrent", budget_reference="lite_baseline_ct",
    ),
}

# Short labels for tables, plots and the results page (display_name is longer).
SHORT_LABELS = {
    "ufld_baseline": "UFLD baseline", "ufld_baseline_ct": "UFLD baseline +CT", "ufld_v02": "UFLD v0.2 weighted",
    "ufld_v03": "UFLD v0.3 gated", "ufld_v04": "UFLD v0.4 coord", "ufld_v06": "UFLD v0.6 aligned",
    "ufld_v07": "UFLD v0.7 ConvGRU", "lite_baseline": "Lite baseline", "lite_baseline_ct": "Lite baseline +CT",
    "lite_v05": "Lite v0.5 warped", "lite_v05_static": "Lite v0.5 static", "lite_v06": "Lite v0.6 ConvGRU",
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
    from tac_ufld.models.fusion import AlignedResidualFusion, ConvGRUFusion
    from tac_ufld.models.lite import LanePixelHead, LightweightBackbone, LiteSingleFrame, LiteTemporal
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
                       head_dropout=m.head_dropout, head_hidden=m.ufld_head_hidden)
        if not spec.temporal:
            return UFLDSingleFrame(ufld)
        channels = 512
        if spec.fusion == "gated":
            fusion = GatedTemporalFusion(channels, d.num_frames)
        elif spec.fusion == "aligned":
            fusion = AlignedResidualFusion(channels, d.num_frames, hidden=64, max_disp=m.aligned_max_disp)
        elif spec.fusion == "recurrent":
            fusion = ConvGRUFusion(channels, m.recurrent_hidden)
        else:
            fusion = TemporalWeightedFusion(d.num_frames)
        return UFLDTemporal(ufld, fusion)
    backbone = LightweightBackbone(in_channels, 128, dropout=m.lite_dropout)
    head = LanePixelHead(128, d.num_lanes, d.num_row_anchors, d.griding_num, dropout=m.lite_head_dropout)
    if not spec.temporal:
        return LiteSingleFrame(backbone, head)
    fusion = ConvGRUFusion(backbone.out_channels, m.recurrent_hidden) if spec.fusion == "recurrent" else None
    return LiteTemporal(backbone, head, d.num_frames, history_encoder=m.lite_history_encoder,
                        in_channels=in_channels, dropout=m.lite_dropout, fusion=fusion)


# ------------------------------------------------------------------ training helpers


def backbone_module(model: nn.Module) -> nn.Module:
    """The (pretrained) feature extractor of any variant: the ResNet of the
    UFLD family, the 4-block CNN of the lite family."""
    if hasattr(model, "ufld"):
        return model.ufld.model
    if hasattr(model, "backbone"):
        return model.backbone
    raise TypeError(f"{type(model).__name__} has no known backbone")


def backbone_stages(model: nn.Module) -> list[nn.Module]:
    """Backbone stages from the input side: ResNet [stem, layer1..4]; lite the 4 conv blocks."""
    bb = backbone_module(model)
    if hasattr(bb, "layer1"):
        return [nn.ModuleList([bb.conv1, bb.bn1]), bb.layer1, bb.layer2, bb.layer3, bb.layer4]
    return list(bb.net)


def freeze_stages(model: nn.Module, n: int) -> list[nn.Module]:
    """Freeze the first ``n`` backbone stages (no gradients). Returns them so
    the trainer can keep their BatchNorm layers in eval mode."""
    stages = backbone_stages(model)[:n]
    for stage in stages:
        for p in stage.parameters():
            p.requires_grad_(False)
    return stages


def init_from_checkpoint(model: nn.Module, path: str, scope: str = "backbone") -> dict[str, list[str]]:
    """Copy every tensor of ``path`` whose name and shape match ``model``.

    Accepts our checkpoints (``{"state_dict": ...}``), official UFLD
    checkpoints (``{"model": ...}``, keys ``model.*`` / ``pool.*`` / ``cls.*``,
    optionally prefixed ``module.``; they map to ``ufld.*`` here) and plain
    state dicts. ``scope="backbone"`` restricts the copy to the backbone.
    Returns the loaded and the skipped (shape mismatch) target keys."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload
    if isinstance(payload, dict):
        state = payload.get("state_dict") or payload.get("model") or payload
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items() if hasattr(v, "shape")}
    target = model.state_dict()
    bb_prefix = "ufld.model." if hasattr(model, "ufld") else "backbone."
    loaded, skipped, new_state = [], [], {}
    for key, value in state.items():
        name = key if key in target else (f"ufld.{key}" if f"ufld.{key}" in target else None)
        if name is None or (scope == "backbone" and not name.startswith(bb_prefix)):
            continue
        if tuple(target[name].shape) != tuple(value.shape):
            skipped.append(name)
            continue
        new_state[name] = value.to(target[name].dtype)
        loaded.append(name)
    if not loaded:
        raise RuntimeError(f"{path}: no tensor matches this model (scope '{scope}')")
    model.load_state_dict(new_state, strict=False)
    return {"loaded": loaded, "skipped": skipped}


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
