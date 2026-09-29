from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tac_ufld.config import LossConfig
from tac_ufld.data.targets import IGNORE_INDEX
from tac_ufld.losses import (
    Hyperparams, VariantLoss, parsing_relation_dis, parsing_relation_loss, softmax_focal_loss,
)
from tac_ufld.models.registry import VARIANTS


def _supervisor_focal(logits, target, gamma=2.0):
    """Reference: official_ufld_classification_loss from the supervisor notebook."""
    scores = F.softmax(logits, dim=1)
    return F.nll_loss(torch.pow(1.0 - scores, gamma) * F.log_softmax(logits, dim=1), target)


def test_focal_matches_supervisor_reference():
    torch.manual_seed(0)
    logits = torch.randn(3, 11, 6, 2)
    target = torch.randint(0, 11, (3, 6, 2))
    assert softmax_focal_loss(logits, target) == pytest.approx(float(_supervisor_focal(logits, target)), rel=1e-5)


def test_ignored_cells_do_not_contribute():
    torch.manual_seed(0)
    logits = torch.randn(2, 11, 6, 2)
    target = torch.randint(0, 11, (2, 6, 2))
    masked = target.clone()
    masked[:, :3] = IGNORE_INDEX
    expected = _supervisor_focal(logits[:, :, 3:], target[:, 3:])
    assert softmax_focal_loss(logits, masked) == pytest.approx(float(expected), rel=1e-5)
    all_ignored = torch.full_like(target, IGNORE_INDEX)
    assert float(softmax_focal_loss(logits, all_ignored)) == 0.0


def test_relation_losses_are_finite():
    logits = torch.randn(2, 11, 8, 2)
    assert torch.isfinite(parsing_relation_loss(logits))
    assert torch.isfinite(parsing_relation_dis(logits))


@pytest.mark.parametrize("key", list(VARIANTS))
def test_variant_losses_backpropagate(key):
    from tac_ufld.config import PROJECT_ROOT, load_config
    from tac_ufld.models.registry import build_model

    cfg = load_config(PROJECT_ROOT / "configs" / "elas_smoke.yaml", {
        "data.img_h": 64, "data.img_w": 96, "data.griding_num": 24, "data.num_row_anchors": 8})
    model = build_model(key, cfg, pretrained=False)
    images = torch.randn(2, 3, 3, 64, 96)
    cls = torch.randint(0, 25, (2, 8, 2))
    cls[:, :2] = IGNORE_INDEX
    batch = {"images": images, "cls": cls, "valid": cls != IGNORE_INDEX,
             "exist": ((cls >= 0) & (cls < 24)).float(), "x": torch.rand(2, 8, 2) * 95}
    hp = Hyperparams(lr=1e-3, lr_fusion=1e-3, weight_decay=0.0, lambda_temporal=0.05, lambda_coord=0.5)
    loss, logs = VariantLoss(VARIANTS[key], LossConfig(), hp, 96)(model, model(images), batch)
    loss.backward()
    assert torch.isfinite(loss) and "focal" in logs
    if VARIANTS[key].temporal:
        assert "temporal" in logs
