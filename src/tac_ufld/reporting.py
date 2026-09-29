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
