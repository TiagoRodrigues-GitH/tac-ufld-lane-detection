"""Multi-seed statistics (audit finding C7).

* Means, standard deviations and t-based 95 % confidence intervals across
  seeds (the split is fixed, so seed variance = initialisation + data order).
* Paired comparisons (same seed) of each temporal variant against its
  same-family single-frame reference: mean difference, 95 % CI, exact
  two-sided Wilcoxon signed-rank p-value, and Holm-Bonferroni adjusted
  p-values across all comparisons.
* ``min_attainable_p`` makes the power limit explicit: with n paired seeds
  the smallest two-sided exact Wilcoxon p is 2 / 2**n (0.125 for n = 4,
  0.031 for n = 6), so fewer than 6 seeds can never reach p < 0.05.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def min_attainable_p(n: int) -> float:
    return 2.0 / 2 ** n if n > 0 else 1.0


def _ci(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean() - half), float(values.mean() + half)


def aggregate(results: pd.DataFrame, metrics: list[str], group: str = "variant") -> pd.DataFrame:
    rows = []
    for name, sub in results.groupby(group, sort=False):
        for metric in metrics:
            if metric not in sub:
                continue
            vals = sub[metric].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            lo, hi = _ci(vals)
            rows.append({group: name, "metric": metric, "n": vals.size, "mean": vals.mean(),
                         "std": vals.std(ddof=1) if vals.size > 1 else float("nan"),
                         "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (NaN entries are kept)."""
    p = np.asarray(pvalues, dtype=float)
    ok = np.flatnonzero(np.isfinite(p))
    adjusted = np.full_like(p, np.nan)
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def paired_comparisons(results: pd.DataFrame, pairs: list[tuple[str, str]], metrics: dict[str, bool],
                       alpha: float = 0.05) -> pd.DataFrame:
    """``pairs`` = (variant, reference); ``metrics`` maps name -> higher_is_better.
    ``results`` needs columns variant, seed and the metrics."""
    rows = []
    for variant, reference in pairs:
        a = results[results["variant"] == variant].set_index("seed")
        b = results[results["variant"] == reference].set_index("seed")
        seeds = sorted(set(a.index) & set(b.index))
        for metric, higher in metrics.items():
            if metric not in results or not seeds:
                continue
            delta = (a.loc[seeds, metric] - b.loc[seeds, metric]).dropna().to_numpy(dtype=float)
            n = delta.size
            p = float("nan")
            if n >= 2 and np.any(delta != 0):
                p = float(stats.wilcoxon(delta, alternative="two-sided", method="exact").pvalue) \
                    if not np.any(delta == 0) else float(stats.wilcoxon(delta, alternative="two-sided").pvalue)
            lo, hi = _ci(delta)
            rows.append({"variant": variant, "reference": reference, "metric": metric, "n_seeds": n,
                         "mean_delta": delta.mean() if n else float("nan"), "ci95_low": lo, "ci95_high": hi,
                         "better_direction": "higher" if higher else "lower",
                         "wilcoxon_p": p, "min_attainable_p": min_attainable_p(n)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_holm"] = holm(df["wilcoxon_p"].tolist())
        df["significant"] = df["p_holm"] < alpha
        df["underpowered"] = df["min_attainable_p"] >= alpha
    return df
