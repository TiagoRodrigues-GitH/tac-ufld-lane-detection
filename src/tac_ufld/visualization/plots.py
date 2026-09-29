"""Publication-style figures (matplotlib, Agg backend, PNG + PDF)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#F0E442", "#000000"]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_history(history: pd.DataFrame, title: str, selection: str, path: Path) -> None:
    if history.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for col, label in (("train_focal", "train focal"), ("val_focal_loss", "val focal"), ("train_total", "train total")):
        if col in history:
            axes[0].plot(history["epoch"], history[col], marker="o", ms=3, label=label)
    axes[0].set(title=f"{title}: loss", xlabel="epoch", ylabel="loss")
    axes[0].legend()
    for col, label in ((f"val_{selection}", selection), ("val_anchor_f1", "anchor F1"), ("val_pixel_f1", "pixel F1")):
        if col in history:
            axes[1].plot(history["epoch"], history[col], marker="o", ms=3, label=label)
    axes[1].set(title=f"{title}: validation", xlabel="epoch", ylabel="score", ylim=(0, 1.02))
    axes[1].legend()
    _save(fig, path)


def plot_metric_bars(agg: pd.DataFrame, metrics: list[str], labels: dict[str, str], path: Path, title: str) -> None:
    agg = agg[agg["metric"].isin(metrics)]
    if agg.empty:
        return
    variants = list(dict.fromkeys(agg["variant"]))
    x = np.arange(len(metrics))
    width = 0.8 / max(len(variants), 1)
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(metrics) + 2), 4.5))
    for i, variant in enumerate(variants):
        sub = agg[agg["variant"] == variant].set_index("metric").reindex(metrics)
        err = np.vstack([sub["mean"] - sub["ci95_low"], sub["ci95_high"] - sub["mean"]])
        err = np.nan_to_num(err, nan=0.0)
        ax.bar(x + (i - (len(variants) - 1) / 2) * width, sub["mean"], width, yerr=err, capsize=3,
               label=labels.get(variant, variant), color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x, metrics, rotation=15, ha="right")
    ax.set(ylim=(0, 1.05), title=title, ylabel="mean over seeds (95 % CI)")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, path)


def plot_condition_heatmap(results: pd.DataFrame, labels: dict[str, str], path: Path) -> None:
    cols = [c for c in results.columns if c.startswith("condition_f1_")]
    if not cols:
        return
    data = results.groupby("variant", sort=False)[cols].mean()
    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 3, 0.6 * len(data) + 1.8))
    im = ax.imshow(data.to_numpy(dtype=float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), [c.split("_")[-1] for c in cols])
    ax.set_yticks(range(len(data)), [labels.get(v, v) for v in data.index])
    for (r, c), v in np.ndenumerate(data.to_numpy(dtype=float)):
        ax.text(c, r, "n/a" if np.isnan(v) else f"{v:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("Lane F1 per scene condition (mean over seeds)")
    _save(fig, path)


def plot_efficiency(eff: pd.DataFrame, results: pd.DataFrame, metric: str, labels: dict[str, str], path: Path) -> None:
    if eff.empty or metric not in results:
        return
    acc = results.groupby("variant", sort=False)[metric].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, row in enumerate(eff.itertuples()):
        if row.variant in acc:
            ax.scatter(row.latency_ms, acc[row.variant], s=80 + 40 * row.params_millions,
                       color=PALETTE[i % len(PALETTE)], edgecolor="black", label=labels.get(row.variant, row.variant))
    ax.set(xlabel="latency per frame, batch 1 (ms)", ylabel=metric, title="Accuracy vs latency (size = parameters)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_error_counts(results: pd.DataFrame, tag: str, labels: dict[str, str], path: Path) -> None:
    cols = [f"lane_tp_{tag}", f"lane_fp_{tag}", f"lane_fn_{tag}"]
    if not set(cols) <= set(results.columns):
        return
    data = results.groupby("variant", sort=False)[cols].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(data))
    for i, (col, name) in enumerate(zip(cols, ("TP", "FP", "FN"))):
        ax.bar(x + (i - 1) * 0.27, data[col], 0.27, label=name, color=PALETTE[i])
    ax.set_xticks(x, [labels.get(v, v) for v in data.index], rotation=15, ha="right")
    ax.set(title=f"Lane-level TP / FP / FN ({tag}, mean over seeds)", ylabel="lanes")
    ax.legend()
    _save(fig, path)
