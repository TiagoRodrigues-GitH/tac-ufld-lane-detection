"""CSV / Excel / Markdown export of experiment results."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Minimal Markdown table (no external dependency)."""
    if df.empty:
        return "_(no rows)_\n"

    def fmt(v) -> str:
        if isinstance(v, (float, np.floating)):
            return "n/a" if not np.isfinite(v) else format(float(v), floatfmt)
        return str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body]) + "\n"


def write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> Path | None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
        return path
    except ImportError:
        LOGGER.warning("openpyxl not installed; skipping %s (CSV files are still written)", path.name)
        return None


def export_lines_txt(records, lanes_per_record, out_dir: Path) -> int:
    """Write predictions in CULane ``.lines.txt`` format (one lane per line,
    ``x y x y ...`` in original pixels), ported from the supervisor's
    ``save_lanes_txt`` / ``export_cache_predictions``. Returns files written."""
    count = 0
    for rec, lanes in zip(records, lanes_per_record):
        path = out_dir / rec.sequence / f"{rec.frame_id:05d}.lines.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for lane in lanes:
            if lane is not None:
                pts = lane[np.argsort(lane[:, 1])[::-1]]  # bottom -> top, as in CULane files
                lines.append(" ".join(f"{x:.2f} {y:.2f}" for x, y in pts))
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        count += 1
    return count


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value and all(isinstance(k, str) for k in value) and key != "scene_tags":
            out.update(_flatten(value, name + "."))
        else:
            out[name] = value
    return out


def config_diff(cfg) -> dict[str, dict]:
    """Every resolved setting that differs from the package defaults, so the
    report states exactly which protocol was used."""
    from tac_ufld.config import ExperimentConfig

    now, ref = _flatten(cfg.to_dict()), _flatten(ExperimentConfig().to_dict())
    return {k: {"default": ref.get(k), "value": v} for k, v in sorted(now.items()) if ref.get(k) != v}


def protocol_section(cfg, split_info: dict, best_params: dict, variants: list[str], changes: dict) -> list[str]:
    """Markdown describing the protocol actually used (split, selection,
    tuning budget, training controls, augmentation, preprocessing)."""
    t, a, g, p, e = cfg.train, cfg.data.augmentation, cfg.data.augmentation.geometric, \
        cfg.data.preprocessing, cfg.evaluation
    geo = ("off" if not (a.enabled and g.enabled) else
           f"p={g.prob}, translate=({g.translate_x}, {g.translate_y}), scale={g.scale}, rotate=±{g.rotate_deg}°, "
           f"perspective={g.perspective}, crop={g.crop_scale}, hflip={g.hflip_prob}")
    extra_photo = {k: getattr(a, k) for k in ("gamma", "hue", "blur_prob", "motion_blur_prob", "shadow_prob")
                   if getattr(a, k)}
    protocol = split_info.get("protocol", {})
    lines = ["\n## Protocol\n",
             f"* Dataset: `{cfg.data.dataset}`; split source: {split_info.get('split_source', 'n/a')}; "
             f"validation: {split_info.get('validation', 'blocks within the non-test scenes')}.",
             f"* Dataset protocol: {protocol.get('split', 'n/a')}. Native metric: {protocol.get('native_metric', 'n/a')}.",
             f"* Selection metric (checkpoints, early stopping, HPO, post-processing sweep): "
             f"`{e.selection_metric}` on validation only; test is evaluated once at the end.",
             f"* Training: {t.optimizer} lr={t.lr} (fusion {t.lr_fusion}), weight decay={t.weight_decay}, "
             f"scheduler={t.scheduler} (warm-up {t.warmup_iters} it), grad clip={t.grad_clip_norm}, "
             f"early stopping patience={t.early_stopping_patience}, epochs<={t.epochs}, AMP={t.amp}, "
             f"label smoothing={t.loss.label_smoothing}, head dropout={cfg.model.head_dropout}.",
             f"* HPO: {'on' if cfg.hpo.enabled else 'off'} ({cfg.hpo.n_trials} trials x {cfg.hpo.epochs} epochs "
             f"per variant, same budget for baselines); tuned: "
             f"{ {v: best_params.get(v) for v in variants if best_params.get(v)} or 'none'}.",
             f"* Photometric augmentation: {'on' if a.enabled else 'off'} (p={a.prob}; extra: {extra_photo or 'none'}); "
             f"geometric: {geo}.",
             f"* Input representation: `{p.mode}` (pre-ops {p.pre_ops or 'none'}), {cfg.in_channels} channel(s).",
             f"* Settings that differ from the package defaults: {len(changes)} (see `config_changes.json`)."]
    return lines


def summary_table(agg: pd.DataFrame, metrics: list[str], labels: dict[str, str]) -> pd.DataFrame:
    """Wide table: one row per variant, 'mean ± std' per metric."""
    rows = []
    for variant, sub in agg.groupby("variant", sort=False):
        row = {"model": labels.get(variant, variant)}
        sub = sub.set_index("metric")
        for m in metrics:
            if m in sub.index:
                mean, std, n = sub.loc[m, "mean"], sub.loc[m, "std"], int(sub.loc[m, "n"])
                row[m] = f"{mean:.4f}" + (f" ± {std:.4f}" if np.isfinite(std) else "") + ("" if n > 1 else " (n=1)")
        rows.append(row)
    return pd.DataFrame(rows)
