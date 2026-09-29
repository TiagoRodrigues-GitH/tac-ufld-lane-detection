"""Decode (B, G+1, A, L) logits into existence probabilities and x positions.

Location follows the official UFLD inference: the expectation of the cell
index under the softmax restricted to the G location cells. Existence is
``1 - p(no lane)``, thresholded later (the threshold is tuned on validation
for every model, including the baseline).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def decode_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(exist_prob, expected_bin)``, both (B, A, L) float32."""
    logits = logits.float()
    probs = F.softmax(logits, dim=1)
    exist = 1.0 - probs[:, -1]
    grid = F.softmax(logits[:, :-1], dim=1)
    bins = torch.arange(grid.shape[1], device=logits.device, dtype=grid.dtype).view(1, -1, 1, 1)
    return exist, (grid * bins).sum(dim=1)


def exist_logits(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """logit(1 - p(no lane)): the existence logit used by the variant BCE terms."""
    p_no = F.softmax(logits.float(), dim=1)[:, -1]
    return torch.logit((1.0 - p_no).clamp(eps, 1.0 - eps))


def expected_x(logits: torch.Tensor, img_w: int) -> torch.Tensor:
    """Soft-argmax x position in model pixels (differentiable)."""
    _, expected_bin = decode_logits(logits)
    griding_num = logits.shape[1] - 1
    return expected_bin * (img_w - 1) / (griding_num - 1)
