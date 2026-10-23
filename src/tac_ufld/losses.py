"""Losses.

Official UFLD terms (cfzd/Ultra-Fast-Lane-Detection, ``utils/loss.py``):
``softmax_focal_loss`` (SoftmaxFocalLoss, gamma=2), ``parsing_relation_loss``
(similarity loss) and ``parsing_relation_dis`` (shape loss). The official
CULane config sets both relation weights to 0; they are configurable here.

Variant-specific auxiliary terms are ported from the supervisor's
``main_loss``: weighted existence BCE (v02/v03/v04), gate prior (v03),
soft-argmax coordinate loss (v04), temporal consistency (all temporal
variants). ``flow_smoothness_loss`` comes from the ELAS script (v05).
Every term respects ``IGNORE_INDEX``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from tac_ufld.config import LossConfig
from tac_ufld.data.targets import IGNORE_INDEX
from tac_ufld.decoding import exist_logits, expected_x
from tac_ufld.models.registry import VariantSpec


def softmax_focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0,
                       label_smoothing: float = 0.0) -> torch.Tensor:
    """Official SoftmaxFocalLoss: NLL of (1 - p)^gamma * log p over G+1 classes.

    ``label_smoothing`` = eps mixes in a uniform target over the G+1 classes:
    (1 - eps) * focal NLL + eps * mean_c[-(1 - p_c)^gamma log p_c]. With
    eps = 0 the result is exactly the official loss."""
    valid = target != IGNORE_INDEX
    if not valid.any():
        return logits.sum() * 0.0
    logits = logits.float()
    log_p = F.log_softmax(logits, dim=1)
    weighted = torch.pow(1.0 - log_p.exp(), gamma) * log_p
    nll = F.nll_loss(weighted, target, ignore_index=IGNORE_INDEX)
    if label_smoothing <= 0:
        return nll
    uniform = -weighted.mean(dim=1)[valid].mean()
    return (1.0 - label_smoothing) * nll + label_smoothing * uniform


def parsing_relation_loss(logits: torch.Tensor) -> torch.Tensor:
    """Official similarity loss: smooth-L1 between logits of adjacent row anchors."""
    diff = logits[:, :, :-1, :] - logits[:, :, 1:, :]
    return F.smooth_l1_loss(diff, torch.zeros_like(diff))


def parsing_relation_dis(logits: torch.Tensor) -> torch.Tensor:
    """Official shape loss: L1 between consecutive first-order differences of
    the expected location over the first half of the row anchors."""
    n, dim, rows, cols = logits.shape
    probs = F.softmax(logits[:, : dim - 1], dim=1)
    pos = (probs * torch.arange(dim - 1, device=logits.device, dtype=probs.dtype).view(1, -1, 1, 1)).sum(1)
    diffs = [pos[:, i] - pos[:, i + 1] for i in range(rows // 2)]
    if len(diffs) < 2:
        return logits.sum() * 0.0
    return sum(F.l1_loss(diffs[i], diffs[i + 1]) for i in range(len(diffs) - 1)) / (len(diffs) - 1)


def distillation_loss(student: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor,
                      temperature: float) -> torch.Tensor:
    """Hinton et al. (2015) knowledge distillation on the row-anchor classification: per (anchor, lane) cell,
    T^2 * KL(softmax(z_t / T) || softmax(z_s / T)) over the G+1 classes, averaged over the annotated cells
    (``valid``; outside them the teacher was never trained). T^2 keeps the gradient scale independent of T."""
    if not valid.any():
        return student.sum() * 0.0
    t = float(temperature)
    log_ps = F.log_softmax(student.float() / t, dim=1)
    log_pt = F.log_softmax(teacher.float() / t, dim=1)
    kl = (log_pt.exp() * (log_pt - log_ps)).sum(dim=1)          # (B, A, L)
    return kl[valid].mean() * t * t


def weighted_exist_bce(logits: torch.Tensor, exist: torch.Tensor, valid: torch.Tensor,
                       neg_weight: float, pos_weight: float) -> torch.Tensor:
    if not valid.any():
        return logits.sum() * 0.0
    ex_logits = exist_logits(logits)
    bce = F.binary_cross_entropy_with_logits(ex_logits, exist, reduction="none")
    weights = torch.where(exist > 0.5, torch.full_like(bce, pos_weight), torch.full_like(bce, neg_weight))
    return (bce * weights)[valid].mean()


def coord_loss(logits: torch.Tensor, x_target: torch.Tensor, exist: torch.Tensor,
               valid: torch.Tensor, img_w: int) -> torch.Tensor:
    """Smooth-L1 between the soft-argmax x and the target x (normalised), positives only."""
    mask = valid & (exist > 0.5)
    if not mask.any():
        return logits.sum() * 0.0
    pred = expected_x(logits, img_w) / (img_w - 1)
    return F.smooth_l1_loss(pred[mask], (x_target / (img_w - 1))[mask])


def temporal_consistency_loss(logits_t: torch.Tensor, logits_prev: torch.Tensor,
                              exist_weight: float = 0.25) -> torch.Tensor:
    """Supervisor's temporal_consistency_loss on per-frame predictions."""
    cls_term = F.smooth_l1_loss(logits_t[:, :-1].float(), logits_prev[:, :-1].float())
    ex_term = F.smooth_l1_loss(exist_logits(logits_t), exist_logits(logits_prev))
    return cls_term + exist_weight * ex_term


def gate_prior_loss(gates: torch.Tensor, prior: tuple[float, ...]) -> torch.Tensor:
    mean_gate = gates.float().mean(dim=(0, 2, 3, 4))
    target = torch.tensor(prior, device=gates.device, dtype=mean_gate.dtype)
    if target.numel() != mean_gate.numel():
        target = torch.linspace(0.5, 1.5, mean_gate.numel(), device=gates.device)
        target = target / target.sum()
    return F.mse_loss(mean_gate, target)


def flow_smoothness_loss(flows: list[torch.Tensor]) -> torch.Tensor:
    total = 0.0
    for flow in flows:
        total = total + (flow[..., :, 1:] - flow[..., :, :-1]).abs().mean()
        total = total + (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()
    return total / max(len(flows), 1)


@dataclass
class Hyperparams:
    """Values that Optuna may tune; defaults come from the train config."""

    lr: float
    lr_fusion: float
    weight_decay: float
    lambda_temporal: float
    lambda_coord: float
    lr_backbone_mult: float = 1.0   # backbone lr = lr * lr_backbone_mult

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


class VariantLoss:
    """Total training loss for one variant. ``focal`` is the component shared
    by every model and is the one logged as the comparable loss.

    What it does: always the official UFLD focal classification loss over the G+1 classes of every (anchor,
    lane) cell; then, depending on the ``VariantSpec``, the optional terms: UFLD similarity/shape losses,
    auxiliary segmentation, weighted existence BCE, gate prior, soft-argmax coordinate loss, flow smoothness, and
    a temporal-consistency term between the single-frame predictions of the last two frames. Cells marked
    ``IGNORE_INDEX`` (outside the annotated band, unknown slots) never contribute."""

    def __init__(self, spec: VariantSpec, loss_cfg: LossConfig, hp: Hyperparams, img_w: int,
                 distill_weight: float = 0.0, distill_temperature: float = 2.0) -> None:
        self.spec, self.cfg, self.hp, self.img_w = spec, loss_cfg, hp, img_w
        self.distill_weight, self.distill_temperature = distill_weight, distill_temperature

    def __call__(self, model: nn.Module, outputs: dict, batch: dict,
                 include_temporal: bool = True) -> tuple[torch.Tensor, dict[str, float]]:
        logits, target, valid = outputs["logits"], batch["cls"], batch["valid"]
        focal = softmax_focal_loss(logits, target, self.cfg.focal_gamma, self.cfg.label_smoothing)
        total = focal
        logs = {"focal": focal.detach()}
        if self.cfg.sim_loss_w:
            term = parsing_relation_loss(logits.float())
            total = total + self.cfg.sim_loss_w * term
            logs["sim"] = term.detach()
        if self.cfg.shp_loss_w:
            term = parsing_relation_dis(logits.float())
            total = total + self.cfg.shp_loss_w * term
            logs["shape"] = term.detach()
        if "aux_seg" in outputs and "seg" in batch:  # official UFLD auxiliary segmentation (training only)
            term = F.cross_entropy(outputs["aux_seg"].float(), batch["seg"])
            total = total + self.cfg.aux_loss_w * term
            logs["aux_seg"] = term.detach()
        spec = self.spec
        if spec.teacher is not None and "teacher_logits" in outputs and self.distill_weight > 0:
            term = distillation_loss(logits, outputs["teacher_logits"], valid, self.distill_temperature)
            total = total + self.distill_weight * term
            logs["kd"] = term.detach()
        if spec.exist_bce is not None:
            term = weighted_exist_bce(logits, batch["exist"], valid, spec.exist_bce.neg_weight,
                                      spec.exist_bce.pos_weight)
            total = total + spec.exist_bce.weight * term
            logs["exist_bce"] = term.detach()
        if spec.gate_prior is not None and "gates" in outputs:
            term = gate_prior_loss(outputs["gates"], spec.gate_prior)
            total = total + spec.gate_prior_weight * term
            logs["gate_prior"] = term.detach()
        if spec.coord_loss:
            term = coord_loss(logits, batch["x"], batch["exist"], valid, self.img_w)
            total = total + self.hp.lambda_coord * term
            logs["coord"] = term.detach()
        if spec.flow_smoothness_weight and outputs.get("flows"):
            term = flow_smoothness_loss(outputs["flows"])
            total = total + spec.flow_smoothness_weight * term
            logs["flow_smooth"] = term.detach()
        images = batch["images"]
        if (include_temporal and spec.temporal_consistency and self.hp.lambda_temporal > 0
                and images.shape[1] > 1 and hasattr(model, "per_frame_logits")):
            t = images.shape[1]
            prev_logits, cur_logits = model.per_frame_logits(images, (t - 2, t - 1))
            term = temporal_consistency_loss(cur_logits, prev_logits)
            total = total + self.hp.lambda_temporal * term
            logs["temporal"] = term.detach()
        logs["total"] = total.detach()
        return total, {k: float(v) for k, v in logs.items()}
