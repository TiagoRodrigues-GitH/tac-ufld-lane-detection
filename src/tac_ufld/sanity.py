"""Model sanity checks before any long run (``python -m tac_ufld sanity``).

For every variant, in warm-start order:

* output shape (B, G+1, A, L) and finite outputs;
* finite total loss and finite, non-zero gradients - reported per parameter
  group (backbone / head / fusion), so a detached fusion module is caught;
* one optimiser step actually changes the weights;
* warm start: every shared tensor equals the source model's tensor after
  loading (strict), and the loaded sub-modules are the expected ones.

Uses one real training batch when the dataset is available (``--real``),
otherwise synthetic inputs of the configured size. Never trains a model.
"""

from __future__ import annotations

import torch

from tac_ufld.config import ExperimentConfig
from tac_ufld.data.targets import IGNORE_INDEX
from tac_ufld.losses import Hyperparams, VariantLoss
from tac_ufld.models.registry import build_model, resolve_spec, training_order, warm_start

EXPECTED_WARM_START = {"ufld": ["ufld"], "lite": ["backbone", "head"]}


def _synthetic_batch(cfg: ExperimentConfig, frames: int, batch: int = 2) -> dict[str, torch.Tensor]:
    d = cfg.data
    g = torch.Generator().manual_seed(0)
    cls = torch.randint(0, d.griding_num + 1, (batch, d.num_row_anchors, d.num_lanes), generator=g)
    cls[:, :2] = IGNORE_INDEX
    exist = ((cls >= 0) & (cls < d.griding_num)).float()
    return {"images": torch.randn(batch, frames, cfg.in_channels, d.img_h, d.img_w, generator=g),
            "cls": cls, "valid": cls != IGNORE_INDEX, "exist": exist,
            "x": torch.rand(batch, d.num_row_anchors, d.num_lanes, generator=g) * (d.img_w - 1)}


def _group(name: str) -> str:
    if name.startswith("fusion") or name.startswith("history_encoder"):
        return "fusion"
    if name.startswith(("ufld.cls", "ufld.pool", "head.")):
        return "head"
    return "backbone"


def run_sanity(cfg: ExperimentConfig, variants: list[str], device: str = "cpu",
               real_batches: dict[int, dict] | None = None) -> list[dict]:
    rows, states = [], {}
    for key in training_order(variants, cfg):
        spec = resolve_spec(key, cfg)
        frames = cfg.data.num_frames if spec.temporal else 1
        torch.manual_seed(0)
        model = build_model(key, cfg, pretrained=False).to(device)
        row = {"variant": key, "warm_start_from": spec.warm_start or "-", "warm_start_ok": True, "loaded": "-"}
        if spec.warm_start:
            source = states[spec.warm_start]
            loaded = warm_start(model, source)
            target = model.state_dict()
            shared = [k for k in source if k in target]
            row["loaded"] = ",".join(loaded)
            row["warm_start_ok"] = bool(shared) and all(torch.equal(target[k].cpu(), source[k].cpu()) for k in shared)
            if loaded != ["<all>"]:
                row["warm_start_ok"] &= loaded == EXPECTED_WARM_START[spec.family]
        batch = (real_batches or {}).get(frames) or _synthetic_batch(cfg, frames)
        batch = {k: v.to(device) for k, v in batch.items()}
        model.train()
        out = model(batch["images"])
        logits = out["logits"]
        d = cfg.data
        expected = (batch["images"].shape[0], d.griding_num + 1, d.num_row_anchors, d.num_lanes)
        row["output_shape_ok"] = tuple(logits.shape) == expected
        row["outputs_finite"] = bool(torch.isfinite(logits).all())
        hp = Hyperparams(cfg.train.lr, cfg.train.lr_fusion, cfg.train.weight_decay, cfg.train.lambda_temporal,
                         cfg.train.lambda_coord)
        loss_fn = VariantLoss(spec, cfg.train.loss, hp, d.img_w)
        loss, logs = loss_fn(model, out, batch)
        row["loss"] = loss.item()
        row["loss_terms"] = ",".join(sorted(logs))
        row["loss_finite"] = bool(torch.isfinite(loss))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss.backward()
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        optimizer.step()
        row["step_changes_weights"] = any(not torch.equal(before[k], v) for k, v in model.named_parameters())
        # Gradient coverage on the SECOND step: zero-initialised output layers
        # (the lite head) legitimately block gradients at step 0 only.
        optimizer.zero_grad(set_to_none=True)
        loss_fn(model, model(batch["images"]), batch)[0].backward()
        groups: dict[str, list[float]] = {}
        bad = 0
        for name, p in model.named_parameters():
            if p.grad is None:
                groups.setdefault(_group(name), []).append(0.0)
                continue
            if not torch.isfinite(p.grad).all():
                bad += 1
            groups.setdefault(_group(name), []).append(float(p.grad.abs().sum()))
        row["non_finite_grads"] = bad
        for g, values in groups.items():
            row[f"grad_{g}_nonzero_frac"] = round(sum(v > 0 for v in values) / len(values), 3)
        grads_ok = bad == 0 and all(row.get(f"grad_{g}_nonzero_frac", 1.0) > 0 for g in groups)
        row["ok"] = bool(row["output_shape_ok"] and row["outputs_finite"] and row["loss_finite"] and grads_ok
                         and row["step_changes_weights"] and row["warm_start_ok"])
        rows.append(row)
        states[key] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return rows
