"""Training loop shared by every variant.

Merged from both original scripts:
* supervisor: per-group learning rates (base network vs fusion module),
  best-checkpoint selection on validation, Optuna reporting;
* ELAS script: warm-up + cosine schedule, gradient clipping, early stopping
  with best-state restore, overfitting diagnostics, TensorBoard scalars.

Checkpoint selection, early stopping, Optuna pruning and the post-processing
sweep all use ONE declared metric (``evaluation.selection_metric``,
default lane F1 at IoU 0.5 on validation), removing the old mix of anchor
F1 / pixel F1 / IoU F1 criteria.
"""

from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from tac_ufld.config import ExperimentConfig
from tac_ufld.data.types import FrameRecord
from tac_ufld.evaluation.evaluator import Evaluator
from tac_ufld.evaluation.predictor import predict
from tac_ufld.losses import Hyperparams, VariantLoss
from tac_ufld.models.registry import VariantSpec
from tac_ufld.postprocess import PostprocessParams

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainResult:
    history: pd.DataFrame
    best_epoch: int
    best_score: float
    checkpoint_path: Path | None


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        spec: VariantSpec,
        cfg: ExperimentConfig,
        hp: Hyperparams,
        device: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        val_records: list[FrameRecord],
        evaluator: Evaluator,
        epochs: int,
        patience: int,
        writer=None,
        trial=None,
        tag: str = "",
    ) -> None:
        self.model, self.spec, self.cfg, self.hp, self.device = model, spec, cfg, hp, device
        self.train_loader, self.val_loader, self.val_records = train_loader, val_loader, val_records
        self.evaluator, self.epochs, self.patience = evaluator, epochs, patience
        self.writer, self.trial, self.tag = writer, trial, tag or spec.key
        self.loss_fn = VariantLoss(spec, cfg.train.loss, hp, cfg.data.img_w)
        self.amp = cfg.train.amp and device.startswith("cuda")
        self.selection = cfg.evaluation.selection_metric
        self.val_params = PostprocessParams.from_config(cfg.evaluation.common_postprocess)

    # ------------------------------------------------------------------ setup

    def _optimizer(self) -> torch.optim.Optimizer:
        fusion = list(self.model.fusion_parameters()) if hasattr(self.model, "fusion_parameters") else []
        fusion_ids = {id(p) for p in fusion}
        base = [p for p in self.model.parameters() if id(p) not in fusion_ids and p.requires_grad]
        groups = [{"params": base, "lr": self.hp.lr}]
        if fusion:
            groups.append({"params": fusion, "lr": self.hp.lr_fusion})
        t = self.cfg.train
        if t.optimizer == "sgd":
            return torch.optim.SGD(groups, momentum=t.momentum, weight_decay=self.hp.weight_decay)
        cls = torch.optim.AdamW if t.optimizer == "adamw" else torch.optim.Adam
        return cls(groups, weight_decay=self.hp.weight_decay)

    def _scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
        t = self.cfg.train
        per_epoch = max(len(self.train_loader), 1)
        total = max(self.epochs * per_epoch, 1)

        def factor(it: int) -> float:
            warm = 1.0 if t.warmup_iters <= 0 else min(1.0, 0.1 + 0.9 * it / t.warmup_iters)
            if t.scheduler == "cos":
                return warm * 0.5 * (1.0 + math.cos(math.pi * min(it, total) / total))
            if t.scheduler == "multi":
                return warm * 0.1 ** sum(it >= s * per_epoch for s in t.multi_steps)
            return warm

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    # ------------------------------------------------------------------- loop

    def _train_epoch(self, optimizer, scheduler, scaler) -> dict[str, float]:
        self.model.train()
        sums: dict[str, float] = {}
        n_seen = 0
        autocast = torch.autocast("cuda") if self.amp else nullcontext()
        for batch in self.train_loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                outputs = self.model(batch["images"])
                loss, logs = self.loss_fn(self.model, outputs, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{self.tag}: non-finite loss {logs}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if self.cfg.train.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            bs = batch["images"].shape[0]
            n_seen += bs
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v * bs
        return {f"train_{k}": v / max(n_seen, 1) for k, v in sums.items()}

    def fit(self, checkpoint_path: Path | None = None, extra_meta: dict | None = None) -> TrainResult:
        optimizer = self._optimizer()
        scheduler = self._scheduler(optimizer)
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        history, best_state = [], None
        best_score, best_loss, best_epoch, stale = -np.inf, np.inf, 0, 0
        for epoch in range(1, self.epochs + 1):
            start = time.time()
            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            row.update(self._train_epoch(optimizer, scheduler, scaler))
            preds = predict(self.model, self.val_loader, self.device, self.cfg.data.img_w,
                            amp=self.amp, loss_fn=self.loss_fn)
            val = self.evaluator.evaluate(preds, self.val_records, self.val_params)
            row.update({f"val_{k}": v for k, v in val.metrics.items()})
            score = val.metrics[self.selection]
            row["seconds"] = time.time() - start
            row["flags"] = self._diagnose(history[-1] if history else None, row)
            improved = score > best_score + 1e-4 or (abs(score - best_score) <= 1e-4 and preds.loss < best_loss)
            if improved:
                best_score, best_loss, best_epoch, stale = score, preds.loss, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if checkpoint_path is not None:
                    self._save(checkpoint_path, best_state, epoch, score, extra_meta)
            else:
                stale += 1
            history.append(row)
            LOGGER.info(
                "[%s] epoch %02d/%d  train_loss=%.4f  val_focal=%.4f  val_%s=%.4f  anchor_f1=%.4f  %s%s",
                self.tag, epoch, self.epochs, row.get("train_total", float("nan")), preds.loss,
                self.selection, score, val.metrics.get("anchor_f1", float("nan")),
                "BEST" if improved else f"no improvement {stale}/{self.patience}",
                f"  [{row['flags']}]" if row["flags"] else "",
            )
            self._log_tensorboard(row, epoch)
            if self.trial is not None:
                import optuna

                self.trial.report(float(score), epoch)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
            if self.patience > 0 and stale >= self.patience:
                LOGGER.info("[%s] early stopping at epoch %d (best epoch %d)", self.tag, epoch, best_epoch)
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return TrainResult(pd.DataFrame(history), best_epoch, float(best_score), checkpoint_path)

    # --------------------------------------------------------------- helpers

    def _save(self, path: Path, state: dict, epoch: int, score: float, extra: dict | None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": state, "variant": self.spec.key, "epoch": epoch,
                    "selection_metric": self.selection, "score": score,
                    "hyperparams": self.hp.as_dict(), **(extra or {})}, path)

    @staticmethod
    def _diagnose(prev: dict | None, row: dict) -> str:
        flags = []
        train, val = row.get("train_focal"), row.get("val_focal_loss")
        if train and val and val > 1.5 * train:
            flags.append(f"val/train focal ratio {val / train:.2f}")
        if prev is not None and train is not None and val is not None:
            if train < prev.get("train_focal", train) - 1e-3 and val > prev.get("val_focal_loss", val) + 1e-3:
                flags.append("overfitting: train loss down, val loss up")
        return "; ".join(flags)

    def _log_tensorboard(self, row: dict, epoch: int) -> None:
        if self.writer is None:
            return
        for key, value in row.items():
            if isinstance(value, (int, float)) and np.isfinite(value) and key != "epoch":
                group = "train" if key.startswith("train_") else ("val" if key.startswith("val_") else "misc")
                self.writer.add_scalar(f"{self.tag}/{group}/{key}", value, epoch)


def load_checkpoint(path: Path, model: nn.Module, device: str = "cpu") -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    return payload
