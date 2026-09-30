"""Static results page (``python -m tac_ufld site``) for GitHub Pages.

Reads finished runs (reports, histories, visuals), ablation summaries,
benchmark reports, deployment checks and an optional demo video, and writes
a self-contained site: ``index.html`` (full document, for GitHub Pages),
``page.html`` (the same content without the document shell, for embedding)
and ``assets/``. Charts are inline SVG computed here, to scale, and themed
through CSS variables; the page needs no JavaScript and no external service
except Google Fonts (with fallbacks). Every number comes from the run
folders; the only hand-written text is the findings list, the roadmap and
the model descriptions (``tac_ufld.models.descriptions``).
"""

from __future__ import annotations

import datetime as dt
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from tac_ufld import __version__
from tac_ufld.config import PROJECT_ROOT
from tac_ufld.models.descriptions import DIFFERENCES, KALMAN_NOTE, MODEL_NOTES
from tac_ufld.models.registry import SHORT_LABELS, VARIANTS

FAMILY = {k: s.family for k, s in VARIANTS.items()}
LABELS = dict(SHORT_LABELS)
ORDER = list(VARIANTS)

CSS = """
/* One light look on purpose (white page, blue only where it carries meaning:
   links, the header rule, highlighted notes and the UFLD family; the lite
   family is teal). Every colour is set explicitly, so the page looks the
   same whatever the viewer's theme. */
:root {
  color-scheme: light;
  --bg: #ffffff; --surface: #f6f9fd; --ink: #15202b; --muted: #586779; --rule: #dde6ef;
  --accent: #2f7ec9; --accent-ink: #1f5f9e; --accent-soft: #eef5fc;
  --ufld: #2f7ec9; --lite: #1f9e9a; --good: #1f8a5b; --bad: #c0392b; --warn: #a86a00;
  --display: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", "Cascadia Mono", Consolas, monospace;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 15px/1.6 var(--body); margin: 0; }
main.wrap { max-width: 1080px; margin: 0 auto; padding-inline: 20px; padding-block: 0 64px; display: grid; gap: 48px; }
header.top { display: grid; gap: 12px; padding-block: 32px 24px; border-bottom: 3px solid var(--accent); }
.eyebrow { font: 600 12px/1 var(--mono); letter-spacing: .12em; text-transform: uppercase; color: var(--accent-ink); }
h1, h2, h3 { font-family: var(--display); font-weight: 600; text-wrap: balance; margin: 0; letter-spacing: -.01em; }
h1 { font-size: clamp(32px, 5.4vw, 48px); line-height: 1.05; }
h2 { font-size: 27px; line-height: 1.15; }
h3 { font-size: 18px; line-height: 1.3; }
p { margin: 0; max-width: 70ch; }
.meta { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { font: 500 12.5px/1 var(--mono); background: var(--surface); border: 1px solid var(--rule); border-radius: 4px;
  padding: 6px 8px; font-variant-numeric: tabular-nums; }
.chip.warn { border-color: var(--warn); color: var(--warn); }
section { display: grid; gap: 16px; min-width: 0; }
section > h2 { padding-top: 4px; }
.lede { color: var(--muted); max-width: 74ch; }
.findings { display: grid; gap: 10px; padding: 18px 22px; background: var(--accent-soft); border-left: 4px solid var(--accent);
  border-radius: 0 6px 6px 0; }
.findings li { margin: 0 0 6px; max-width: 78ch; }
.findings ul { margin: 0; padding-left: 20px; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr)); gap: 16px; }
.panel { background: var(--bg); border: 1px solid var(--rule); border-radius: 6px; padding: 16px; display: grid; gap: 10px; min-width: 0; }
.cards { display: grid; gap: 12px; }
.card { background: var(--bg); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 18px; display: grid;
  gap: 8px; min-width: 0; }
.card.temporal { border-left: 4px solid var(--accent); }
.card.control { border-left: 4px solid var(--warn); }
.card .head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 12px; }
.card .origin { font-size: 12.5px; color: var(--muted); }
.card .cost { font: 500 12.5px/1.4 var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; margin-left: auto; }
.card .body { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.card p { font-size: 14.5px; max-width: none; }
.card .diff { color: var(--muted); }
.card .k { font: 600 11px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted);
  display: block; margin-bottom: 4px; }
.family-title { font: 600 13px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted);
  display: flex; align-items: center; gap: 8px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
td.wrap { white-space: normal; min-width: 180px; }
th { font: 600 11.5px/1.3 var(--mono); text-transform: uppercase; letter-spacing: .06em; color: var(--muted);
  background: var(--surface); }
td.num, th.num { text-align: right; }
tr.group td { font: 600 12px/1.2 var(--mono); color: var(--muted); background: var(--surface); text-transform: uppercase;
  letter-spacing: .06em; }
.fam { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 8px; vertical-align: baseline; }
.fam.ufld { background: var(--ufld); } .fam.lite { background: var(--lite); }
.pill { font: 600 11px/1 var(--mono); padding: 4px 7px; border-radius: 99px; border: 1px solid currentColor; white-space: nowrap; }
.pill.ok { color: var(--good); } .pill.no { color: var(--bad); } .pill.na { color: var(--muted); }
.pill.role-baseline { color: var(--muted); } .pill.role-control { color: var(--warn); } .pill.role-temporal { color: var(--accent-ink); }
.pill.done { color: var(--good); } .pill.progress { color: var(--accent-ink); } .pill.next { color: var(--warn); }
.pill.later { color: var(--muted); }
svg { display: block; max-width: 100%; height: auto; font-family: var(--mono); }
svg text { fill: var(--muted); font-size: 11px; }
svg .label { fill: var(--ink); font-size: 12px; font-family: var(--body); }
svg .grid { stroke: var(--rule); stroke-width: 1; }
svg .ufld { fill: var(--ufld); } svg .lite { fill: var(--lite); }
svg .ufld-s { stroke: var(--ufld); } svg .lite-s { stroke: var(--lite); }
svg .dot { fill: var(--bg); stroke: var(--ink); stroke-width: 1.4; }
svg .box { fill: var(--surface); stroke: var(--rule); stroke-width: 1.2; }
svg .box-hi { fill: var(--accent-soft); stroke: var(--accent); stroke-width: 1.6; }
svg .arrow { stroke: var(--muted); stroke-width: 1.4; fill: none; }
svg .arrowhead { fill: var(--muted); }
svg .title { fill: var(--ink); font: 600 13px var(--body); }
svg .small { fill: var(--muted); font: 11.5px var(--body); }
svg .zero { stroke: var(--ink); stroke-width: 1; opacity: .5; }
svg .pos { fill: var(--accent); } svg .neg { fill: var(--bad); opacity: .8; }
svg .cell-text { fill: var(--ink); font-size: 12px; }
svg .seg-pre { fill: var(--muted); opacity: .5; } svg .seg-model { fill: var(--accent); } svg .seg-post { fill: var(--ink); opacity: .7; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12.5px; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
video, .shots img { width: 100%; max-width: 100%; border: 1px solid var(--rule); border-radius: 4px; background: #000; }
.shots { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 14px; }
figure { margin: 0; display: grid; gap: 6px; min-width: 0; }
figcaption { font-size: 12.5px; color: var(--muted); }
.phase { display: grid; gap: 8px; }
.phase ul { margin: 0; padding-left: 0; list-style: none; display: grid; gap: 8px; }
.phase li { display: grid; grid-template-columns: 96px 1fr; gap: 12px; align-items: baseline; max-width: 90ch; }
.phase li > span:first-child { justify-self: start; }
pre { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 14px; overflow-x: auto;
  font: 13px/1.55 var(--mono); margin: 0; }
code { font-family: var(--mono); font-size: .92em; }
a { color: var(--accent-ink); text-underline-offset: 3px; }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
footer { color: var(--muted); font-size: 12.5px; border-top: 1px solid var(--rule); padding-top: 16px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 185px), 1fr)); gap: 12px; }
.kpi { border: 1px solid var(--rule); border-top: 3px solid var(--accent); border-radius: 6px; padding: 12px 14px;
  display: grid; gap: 4px; align-content: start; min-width: 0; }
.kpi .v { font: 600 25px/1.15 var(--display); font-variant-numeric: tabular-nums; color: var(--ink); }
.kpi .k { font: 600 11px/1.2 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent-ink); }
.kpi .l { font-size: 13px; line-height: 1.45; color: var(--muted); }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 460px), 1fr)); gap: 16px; }
.charts > .panel, .plan > .panel { align-content: start; }
.en { display: block; color: var(--muted); font-size: 12.5px; font-weight: 400; }
th .en { text-transform: none; letter-spacing: 0; font-family: var(--body); }
.h2en { font-size: 16px; margin-top: 4px; font-family: var(--body); }
table.choice td.wrap { min-width: 190px; } table.choice td { vertical-align: top; }
.plan { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr)); gap: 16px; }
.plan ul, .plan ol { margin: 0; padding-left: 20px; display: grid; gap: 6px; }
.plan li { font-size: 14.5px; }
svg .clean { fill: var(--bg); stroke: var(--ink); stroke-width: 1.6; } svg .hit { stroke: var(--bad); stroke-width: 2; opacity: .55; }
@media (max-width: 720px) { .card .body { grid-template-columns: 1fr; gap: 8px; } .card .cost { margin-left: 0; } }
@media (max-width: 560px) { main.wrap { padding-inline: 16px; } h2 { font-size: 23px; }
  .phase li { grid-template-columns: 1fr; gap: 4px; } }
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" '
         'href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="https://fonts.googleapis.com/'
         'css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">')


def esc(v) -> str:
    return html.escape(str(v))


def fmt(v, digits: int = 3) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return esc(v)
    return "n/a" if not np.isfinite(f) else f"{f:.{digits}f}"


def _md(text: str) -> str:
    """Tiny Markdown subset: **bold** and `code`."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", re.sub(r"`(.+?)`", r"<code>\1</code>", esc(text)))


def _label(v: str) -> str:
    if v.endswith("+kf"):
        return f"{LABELS.get(v[:-3], v[:-3])} + Kalman"
    return LABELS.get(v, v)


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


# ------------------------------------------------------------------ SVG charts


def bar_chart(rows: list[dict], metric_label: str, width: int = 760, left: int = 150) -> str:
    """Horizontal bars (mean) with per-seed dots, one row per model, 0..1 scale.
    Rows keep their order; a thin gap separates the model families. A row may
    carry its own ``label``."""
    right, row_h, top = 60, 30, 26
    gaps = sum(1 for a, b in zip(rows, rows[1:]) if FAMILY.get(a["variant"]) != FAMILY.get(b["variant"]))
    height = top + row_h * len(rows) + 12 * gaps + 30
    plot_w = width - left - right
    x = lambda v: left + plot_w * max(0.0, min(1.0, v))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(metric_label)} per model">']
    for t in np.linspace(0, 1, 6):
        parts.append(f'<line class="grid" x1="{x(t):.1f}" y1="{top - 8}" x2="{x(t):.1f}" y2="{height - 26}"/>'
                     f'<text x="{x(t):.1f}" y="{height - 10}" text-anchor="middle">{t:.1f}</text>')
    y = top
    for i, r in enumerate(rows):
        if i and FAMILY.get(rows[i - 1]["variant"]) != FAMILY.get(r["variant"]):
            y += 12
        fam = FAMILY.get(r["variant"].removesuffix("+kf"), "ufld")
        label = r.get("label") or _label(r["variant"])
        parts.append(f'<text class="label" x="{left - 10}" y="{y + 16}" text-anchor="end">{esc(label)}</text>')
        parts.append(f'<rect class="{fam}" x="{left}" y="{y + 5}" width="{x(r["mean"]) - left:.1f}" height="16" rx="2"/>')
        for s in r["seeds"]:
            parts.append(f'<circle class="dot" cx="{x(s):.1f}" cy="{y + 13}" r="4"/>')
        end = max([r["mean"], *r["seeds"]])  # label after the bar and every seed dot
        parts.append(f'<text x="{x(end) + 10:.1f}" y="{y + 17}">{r["mean"]:.3f}</text>')
        y += row_h
    parts.append("</svg>")
    return "".join(parts)


def delta_chart(rows: list[dict], width: int = 760, label_w: int = 190) -> str:
    """Signed horizontal bars around zero (mean difference) with per-seed dots."""
    right, row_h, top = 60, 28, 22
    vals = [v for r in rows for v in [r["mean"], *r["seeds"]] if np.isfinite(v)] or [0.0]
    lim = max(0.02, max(abs(v) for v in vals)) * 1.15
    height = top + row_h * len(rows) + 30
    plot_w = width - label_w - right
    x = lambda v: label_w + plot_w * (v + lim) / (2 * lim)
    ticks = np.linspace(-lim, lim, 5)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="difference in lane F1">']
    for t in ticks:
        parts.append(f'<line class="grid" x1="{x(t):.1f}" y1="{top - 6}" x2="{x(t):.1f}" y2="{height - 26}"/>'
                     f'<text x="{x(t):.1f}" y="{height - 10}" text-anchor="middle">{t:+.2f}</text>')
    parts.append(f'<line class="zero" x1="{x(0):.1f}" y1="{top - 6}" x2="{x(0):.1f}" y2="{height - 26}"/>')
    for i, r in enumerate(rows):
        y = top + i * row_h
        parts.append(f'<text class="label" x="{label_w - 10}" y="{y + 15}" text-anchor="end">{esc(r["label"])}</text>')
        m = r["mean"]
        x0, x1 = sorted((x(0), x(m)))
        parts.append(f'<rect class="{"pos" if m >= 0 else "neg"}" x="{x0:.1f}" y="{y + 4}" width="{x1 - x0:.1f}" '
                     f'height="15" rx="2"/>')
        for s in r["seeds"]:
            parts.append(f'<circle class="dot" cx="{x(s):.1f}" cy="{y + 11.5}" r="3.5"/>')
        tx = max([x(m), *[x(s) for s in r["seeds"]]]) + 8
        parts.append(f'<text x="{tx:.1f}" y="{y + 16}">{m:+.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def dumbbell_chart(rows: list[dict], a_label: str, b_label: str, width: int = 760, left: int = 190) -> str:
    """Two values per row on one scale: ``a`` (hollow dot) and ``b`` (filled,
    family colour), joined by a line. Used for clean vs degraded lane F1."""
    right, row_h, top = 70, 30, 34
    vals = [v for r in rows for v in (r["a"], r["b"]) if np.isfinite(v)] or [0.0, 1.0]
    lo = max(0.0, np.floor(min(vals) * 10) / 10 - 0.1)
    height = top + row_h * len(rows) + 30
    plot_w = width - left - right
    x = lambda v: left + plot_w * (v - lo) / (1.0 - lo)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(a_label)} and {esc(b_label)} per model">',
             f'<circle class="clean" cx="{left + 6}" cy="12" r="5"/><text x="{left + 16}" y="16">{esc(a_label)}</text>',
             f'<circle class="ufld" cx="{left + 200}" cy="12" r="5"/><text x="{left + 210}" y="16">{esc(b_label)}</text>']
    for t in np.arange(np.ceil(lo * 10) / 10, 1.0001, 0.1):
        parts.append(f'<line class="grid" x1="{x(t):.1f}" y1="{top - 8}" x2="{x(t):.1f}" y2="{height - 26}"/>'
                     f'<text x="{x(t):.1f}" y="{height - 10}" text-anchor="middle">{t:.1f}</text>')
    for i, r in enumerate(rows):
        y = top + i * row_h + 12
        fam = FAMILY.get(r["variant"], "ufld")
        parts.append(f'<text class="label" x="{left - 10}" y="{y + 4}" text-anchor="end">{esc(r["label"])}</text>')
        parts.append(f'<line class="hit" x1="{x(r["a"]):.1f}" y1="{y}" x2="{x(r["b"]):.1f}" y2="{y}"/>')
        parts.append(f'<circle class="clean" cx="{x(r["a"]):.1f}" cy="{y}" r="5"/>'
                     f'<circle class="{fam}" cx="{x(r["b"]):.1f}" cy="{y}" r="5.5"/>')
        parts.append(f'<text x="{max(x(r["a"]), x(r["b"])) + 10:.1f}" y="{y + 4}">{r["b"]:.3f} '
                     f'({r["b"] - r["a"]:+.3f})</text>')
    parts.append("</svg>")
    return "".join(parts)


def heatmap(values: dict[tuple[int, int], float], rows: list[int], cols: list[int], row_label: str,
            col_label: str, width: int = 520, signed: bool = False) -> str:
    """Grid of cells (rows x cols) coloured by value; the value is printed in every cell."""
    left, top, cell_h = 90, 40, 38
    cell_w = (width - left - 10) / max(len(cols), 1)
    height = top + cell_h * len(rows) + 16
    finite = [v for v in values.values() if np.isfinite(v)] or [0.0]
    lo, hi = (min(finite), max(finite)) if not signed else (-max(abs(v) for v in finite), max(abs(v) for v in finite))
    span = (hi - lo) or 1.0
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(row_label)} by {esc(col_label)}">',
             f'<text class="small" x="{left + (width - left) / 2:.1f}" y="14" text-anchor="middle">{esc(col_label)}</text>']
    for j, c in enumerate(cols):
        parts.append(f'<text x="{left + (j + .5) * cell_w:.1f}" y="32" text-anchor="middle">{c}</text>')
    parts.append(f'<text class="small" x="12" y="{top + cell_h * len(rows) / 2:.1f}" transform="rotate(-90 12 '
                 f'{top + cell_h * len(rows) / 2:.1f})" text-anchor="middle">{esc(row_label)}</text>')
    for i, rv in enumerate(rows):
        y = top + i * cell_h
        parts.append(f'<text x="{left - 10}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="end">{rv}</text>')
        for j, c in enumerate(cols):
            v = values.get((rv, c), float("nan"))
            xx = left + j * cell_w
            if np.isfinite(v):
                a = (v - lo) / span if not signed else abs(v) / (span / 2)
                cls = "neg" if signed and v < 0 else "pos"
                text = f"{v:+.3f}" if signed else f"{v:.3f}"
                parts.append(f'<rect class="box" x="{xx + 1:.1f}" y="{y + 1}" width="{cell_w - 2:.1f}" height="{cell_h - 2}"/>'
                             f'<rect class="{cls}" x="{xx + 1:.1f}" y="{y + 1}" width="{cell_w - 2:.1f}" height="{cell_h - 2}" '
                             f'opacity="{0.12 + 0.6 * max(0.0, min(1.0, a)):.2f}"/>'
                             f'<text class="cell-text" x="{xx + cell_w / 2:.1f}" y="{y + cell_h / 2 + 4:.1f}" '
                             f'text-anchor="middle">{text}</text>')
            else:
                parts.append(f'<rect class="box" x="{xx + 1:.1f}" y="{y + 1}" width="{cell_w - 2:.1f}" height="{cell_h - 2}"/>'
                             f'<text x="{xx + cell_w / 2:.1f}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="middle">not run</text>')
    parts.append("</svg>")
    return "".join(parts)


def curves_chart(hist: pd.DataFrame, metric: str, width: int = 330, height: int = 190) -> str:
    """Small multiple: metric vs epoch for one model, one line per seed."""
    left, right, top, bottom = 38, 12, 14, 26
    epochs = hist["epoch"].to_numpy()
    xmax = max(2, int(epochs.max()))
    x = lambda e: left + (width - left - right) * (e - 1) / (xmax - 1)
    y = lambda v: top + (height - top - bottom) * (1 - max(0.0, min(1.0, v)))
    fam = FAMILY.get(hist["variant"].iloc[0], "ufld")
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(metric)} per epoch">']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line class="grid" x1="{left}" y1="{y(t):.1f}" x2="{width - right}" y2="{y(t):.1f}"/>'
                     f'<text x="{left - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{t:.2f}</text>')
    for e in range(1, xmax + 1):
        parts.append(f'<text x="{x(e):.1f}" y="{height - 8}" text-anchor="middle">{e}</text>')
    dashes = ["", "5 4", "2 3", "8 3"]
    for k, (seed, sub) in enumerate(hist.groupby("seed")):
        sub = sub.sort_values("epoch")
        pts = " ".join(f"{x(e):.1f},{y(v):.1f}" for e, v in zip(sub["epoch"], sub[metric]))
        parts.append(f'<polyline class="{fam}-s" points="{pts}" fill="none" stroke-width="2.2" '
                     f'stroke-dasharray="{dashes[k % 4]}"/>')
        best = sub.loc[sub[metric].idxmax()]
        parts.append(f'<circle class="dot" cx="{x(best["epoch"]):.1f}" cy="{y(best[metric]):.1f}" r="3.5"/>')
    parts.append("</svg>")
    return "".join(parts)


def latency_chart(rows: list[dict], width: int = 760) -> str:
    """Stacked horizontal bars: preprocess / model / postprocess (ms)."""
    left, right, row_h, top = 210, 70, 28, 22
    height = top + row_h * len(rows) + 30
    vmax = max(r["total"] for r in rows) * 1.08
    step = 5 if vmax <= 40 else 10
    x = lambda v: left + (width - left - right) * v / vmax
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="latency per frame">']
    for t in range(0, int(vmax) + 1, step):
        parts.append(f'<line class="grid" x1="{x(t):.1f}" y1="{top - 6}" x2="{x(t):.1f}" y2="{height - 26}"/>'
                     f'<text x="{x(t):.1f}" y="{height - 10}" text-anchor="middle">{t}</text>')
    for i, r in enumerate(rows):
        y = top + i * row_h
        parts.append(f'<text class="label" x="{left - 10}" y="{y + 14}" text-anchor="end">{esc(r["label"])}</text>')
        x0 = left
        for key, cls in (("pre", "seg-pre"), ("model", "seg-model"), ("post", "seg-post")):
            w = x(r[key]) - left
            parts.append(f'<rect class="{cls}" x="{x0:.1f}" y="{y + 3}" width="{w:.1f}" height="16"/>')
            x0 += w
        parts.append(f'<text x="{x0 + 6:.1f}" y="{y + 15}">{r["total"]:.1f} ms</text>')
    parts.append("</svg>")
    return "".join(parts)


def pipeline_svg(width: int = 900) -> str:
    """Shared skeleton of every model: frames -> backbone -> fusion -> head -> lanes."""
    h = 250
    parts = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="model pipeline: frames, shared backbone, '
             'temporal fusion, row-anchor head, lanes">',
             '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             'orient="auto-start-reverse"><path class="arrowhead" d="M0,0 L10,5 L0,10 z"/></marker></defs>']
    # frames
    for i, lab in enumerate(("t - 4", "t - 2", "t (current)")):
        y = 36 + i * 62
        parts.append(f'<rect class="box" x="10" y="{y}" width="120" height="46" rx="4"/>'
                     f'<text class="title" x="70" y="{y + 28}" text-anchor="middle">{lab}</text>')
        parts.append(f'<path class="arrow" d="M130,{y + 23} L176,{y + 23}" marker-end="url(#ah)"/>')
    parts.append('<text class="small" x="70" y="22" text-anchor="middle">camera frames (step 2)</text>')
    # backbone
    parts.append('<rect class="box" x="180" y="36" width="170" height="170" rx="6"/>'
                 '<text class="title" x="265" y="92" text-anchor="middle">Backbone</text>'
                 '<text class="small" x="265" y="114" text-anchor="middle">same weights</text>'
                 '<text class="small" x="265" y="131" text-anchor="middle">for every frame</text>'
                 '<text class="small" x="265" y="158" text-anchor="middle">ResNet-18 (UFLD)</text>'
                 '<text class="small" x="265" y="175" text-anchor="middle">4-block CNN (lite)</text>'
                 '<text class="small" x="265" y="228" text-anchor="middle">past frames: cached features</text>')
    parts.append('<path class="arrow" d="M350,121 L396,121" marker-end="url(#ah)"/>')
    # fusion
    parts.append('<rect class="box-hi" x="400" y="20" width="220" height="202" rx="6"/>'
                 '<text class="title" x="510" y="46" text-anchor="middle">Temporal fusion</text>'
                 '<text class="small" x="510" y="64" text-anchor="middle">where the models differ</text>')
    for i, (name, who) in enumerate((("none", "baselines, +CT"), ("weighted sum", "v0.2, v0.4"),
                                     ("per-cell gates", "v0.3"), ("warp + gate", "v0.5, v0.6"),
                                     ("ConvGRU", "v0.7, lite v0.6"))):
        y = 90 + i * 25
        parts.append(f'<text class="label" x="416" y="{y}">{name}</text>'
                     f'<text class="small" x="606" y="{y}" text-anchor="end">{who}</text>')
    parts.append('<path class="arrow" d="M620,121 L666,121" marker-end="url(#ah)"/>')
    # head
    parts.append('<rect class="box" x="670" y="66" width="110" height="110" rx="6"/>'
                 '<text class="title" x="725" y="104" text-anchor="middle">Row-anchor</text>'
                 '<text class="title" x="725" y="121" text-anchor="middle">head</text>'
                 '<text class="small" x="725" y="142" text-anchor="middle">18 rows x 100</text>'
                 '<text class="small" x="725" y="157" text-anchor="middle">cells per lane</text>')
    parts.append('<path class="arrow" d="M780,121 L816,121" marker-end="url(#ah)"/>')
    parts.append('<rect class="box" x="820" y="86" width="72" height="70" rx="6"/>'
                 '<text class="title" x="856" y="118" text-anchor="middle">lanes</text>'
                 '<text class="small" x="856" y="136" text-anchor="middle">(+ Kalman)</text>')
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------------ sections


def _agg(df: pd.DataFrame, split: str, protocol: str, metric: str, inp: str = "full") -> list[dict]:
    sub = df[(df["split"] == split) & (df["protocol"] == protocol) & (df["input"] == inp)]
    rows = []
    for v in [v for v in ORDER if v in set(sub["variant"])]:
        vals = sub[sub["variant"] == v][metric].to_numpy(dtype=float)
        rows.append({"variant": v, "mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
                     "seeds": vals.tolist()})
    return rows


def _mean_std(vals: np.ndarray, digits: int) -> str:
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return "n/a"
    return f"{vals.mean():.{digits}f}" + (f" ± {vals.std(ddof=1):.{digits}f}" if len(vals) > 1 else "")


def results_table(df: pd.DataFrame, split: str, protocol: str) -> str:
    sub = df[(df["split"] == split) & (df["protocol"] == protocol) & (df["input"] == "full")]
    cols = [("lane_f1_iou50", "Lane F1 @0.5"), ("lane_f1_iou35", "Lane F1 @0.35"), ("pixel_f1", "Pixel F1"),
            ("anchor_f1", "Anchor F1"), ("jitter_px", "Jitter px"), ("lane_fp_iou50", "FP"), ("lane_fn_iou50", "FN")]
    cols = [(c, n) for c, n in cols if c in sub]
    head = "".join(f'<th class="num">{esc(n)}</th>' for _, n in cols)
    body = []
    for v in [v for v in ORDER if v in set(sub["variant"])]:
        s = sub[sub["variant"] == v]
        digits = {"lane_fp_iou50": 0, "lane_fn_iou50": 0, "jitter_px": 2}
        cells = "".join(f'<td class="num">{_mean_std(s[c].to_numpy(dtype=float), digits.get(c, 3))}</td>'
                        for c, _ in cols)
        body.append(f'<tr><td><span class="fam {FAMILY[v]}"></span>{esc(LABELS[v])}</td>{cells}</tr>')
    return (f'<div class="scroll"><table><thead><tr><th>Model</th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _verdict(r) -> str:
    if r.significant:
        return '<span class="pill ok">significant</span>'
    if r.underpowered:
        return '<span class="pill na">underpowered</span>'
    return '<span class="pill no">not significant</span>'


def paired_table(df: pd.DataFrame | None, first: str = "Model", metric: str = "lane_f1_iou50") -> str:
    if df is None or df.empty:
        return ""
    df = df[df["metric"] == metric]
    rows = []
    for r in df.itertuples():
        rows.append(f'<tr><td>{esc(_label(r.variant))}</td><td>{esc(_label(r.reference))}</td>'
                    f'<td class="num">{r.n_seeds}</td><td class="num">{r.mean_delta:+.3f}</td>'
                    f'<td class="num">[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]</td><td class="num">{fmt(r.wilcoxon_p)}</td>'
                    f'<td class="num">{fmt(r.p_holm)}</td><td>{_verdict(r)}</td></tr>')
    return (f'<div class="scroll"><table><thead><tr><th>{esc(first)}</th><th>Compared with</th><th class="num">Seeds</th>'
            '<th class="num">Δ lane F1</th><th class="num">95 % CI</th><th class="num">Wilcoxon p</th>'
            '<th class="num">Holm p</th><th>Verdict</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def _history_gain_table(path: Path) -> str:
    df = _read(path)
    if df is None or df.empty:
        return ""
    col = [c for c in df.columns if c.startswith("mean_gain")][0]
    rows = "".join(f'<tr><td>{esc(_label(r.variant))}</td><td class="num">{getattr(r, col):+.4f}</td>'
                   f'<td class="num">{r.n_seeds}</td></tr>' for r in df.itertuples())
    return ('<div class="scroll"><table><thead><tr><th>Temporal model</th><th class="num">F1 gain from real history</th>'
            f'<th class="num">Seeds</th></tr></thead><tbody>{rows}</tbody></table></div>')


def _copy_image(src: Path, dst: Path, width: int = 960) -> None:
    import cv2

    from tac_ufld.visualization.overlays import imread_bgr

    img = imread_bgr(src)
    if img.shape[1] > width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    buf.tofile(str(dst))


def run_duration(run: Path) -> str | None:
    log = run / "run.log"
    if not log.exists():
        return None
    times = re.findall(r"^(\d\d:\d\d:\d\d) ", log.read_text(encoding="utf-8", errors="replace"), flags=re.M)
    if len(times) < 2:
        return None
    t0, t1 = (dt.datetime.strptime(t, "%H:%M:%S") for t in (times[0], times[-1]))
    minutes = int(((t1 - t0).total_seconds() % 86400) // 60)
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"


def models_section(efficiency: pd.DataFrame | None, present: set[str]) -> str:
    """Architecture diagram, comparison table and one card per model."""
    eff = {} if efficiency is None else {r.variant: r for r in efficiency.itertuples()}
    keys = [k for k in ORDER if k in present] or ORDER

    def cost(k: str) -> str:
        r = eff.get(k)
        if r is None:
            return ""
        frames = f", {int(r.input_frames)} frames per clip" if hasattr(r, "input_frames") and r.input_frames > 1 else ""
        return f"{r.params_millions:.2f} M params · {r.gmacs:.1f} GMACs per clip{frames}"

    rows = []
    for fam, title in (("ufld", "UFLD family (ResNet-18, ImageNet)"), ("lite", "Lite family (4-block CNN, from scratch)")):
        rows.append(f'<tr class="group"><td colspan="7"><span class="fam {fam}"></span>{esc(title)}</td></tr>')
        for k in [k for k in keys if FAMILY[k] == fam]:
            n = MODEL_NOTES[k]
            r = eff.get(k)
            params = "" if r is None else f"{r.params_millions:.2f}"
            rows.append(f'<tr><td>{esc(LABELS[k])}</td><td><span class="pill role-{n["role"]}">{n["role"]}</span></td>'
                        f'<td>{esc(n["frames"])}</td><td>{esc(n["history"])}</td><td>{esc(n["aligns"])}</td>'
                        f'<td>{esc(n["memory"])}</td><td class="num">{params}</td></tr>')
    table = ('<div class="scroll"><table><thead><tr><th>Model</th><th>Role</th><th>Frames used</th>'
             '<th>How history enters</th><th>Aligns motion</th><th>Memory</th><th class="num">Params (M)</th></tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table></div>')
    cards = []
    for fam, title in (("ufld", "UFLD family"), ("lite", "Lite family")):
        items = []
        for k in [k for k in keys if FAMILY[k] == fam]:
            n = MODEL_NOTES[k]
            items.append(f'<article class="card {n["role"]}"><div class="head"><h3><span class="fam {fam}"></span>'
                         f'{esc(LABELS[k])}</h3><span class="pill role-{n["role"]}">{n["role"]}</span>'
                         f'<span class="origin">{esc(n["origin"])}</span><span class="cost">{esc(cost(k))}</span></div>'
                         f'<div class="body"><p><span class="k">What it does</span>{esc(n["what"])}</p>'
                         f'<p class="diff"><span class="k">Difference</span>{esc(n["differs"])}</p></div></article>')
        cards.append(f'<div class="family-title"><span class="fam {fam}"></span>{title}</div>'
                     f'<div class="cards">{"".join(items)}</div>')
    diffs = "".join(f"<li>{esc(d)}</li>" for d in DIFFERENCES)
    return ('<section id="models"><h2>The models</h2><p class="lede">Two families, each a single-frame baseline and '
            'temporal variants built on it. All of them read the current frame and predict the two ego lanes of '
            'that frame; the temporal ones also read earlier frames.</p>'
            f'<div class="panel scroll">{pipeline_svg()}</div>'
            f'<div class="findings"><ul>{diffs}</ul></div>{table}{"".join(cards)}'
            f'<p class="lede"><strong>Reference, not a model:</strong> {esc(KALMAN_NOTE)}</p></section>')


def _augmentation_section(root: Path) -> str:
    agg = _read(root / "ablation_augmentation" / "ablation_aggregate.csv")
    paired = _read(root / "ablation_augmentation" / "ablation_paired_tests.csv")
    res = _read(root / "ablation_augmentation" / "ablation_results.csv")
    if agg is None or res is None:
        return ""
    metric = "lane_f1_iou50"
    sub = res[res["split"] == "test"] if "split" in res else res
    arms = list(dict.fromkeys(sub["arm"]))
    rows = []
    for arm in arms:
        vals = sub[sub["arm"] == arm][metric].to_numpy(dtype=float)
        val_scores, best_epochs = [], []
        for h in sorted((root / f"ablation_augmentation__{arm}").glob("seed_*/history_ufld_baseline.csv")):
            hd = pd.read_csv(h)
            best = hd.loc[hd[f"val_{metric}"].idxmax()]
            val_scores.append(float(best[f"val_{metric}"]))
            best_epochs.append(int(best["epoch"]))
        verdict = ""
        if paired is not None and not paired.empty:
            p = paired[(paired["variant"] == f"{arm}/ufld_baseline") & (paired["metric"] == metric)]
            if len(p):
                r = next(p.itertuples())
                verdict = f'{r.mean_delta:+.3f} {_verdict(r)}'
        rows.append((arm, vals, np.asarray(val_scores), best_epochs, verdict))
    rows.sort(key=lambda r: -np.nanmean(r[2]) if len(r[2]) else 0.0)
    body = "".join(f'<tr><td>{esc(a.replace("_", " "))}</td><td class="num">{_mean_std(val, 3)}</td>'
                   f'<td class="num">{_mean_std(v, 3)}</td><td class="num">{", ".join(map(str, e)) or "n/a"}</td>'
                   f'<td>{verdict or "reference"}</td></tr>' for a, v, val, e, verdict in rows)
    chart = bar_chart([{"variant": "ufld_baseline", "label": a.replace("_", " "), "mean": float(np.nanmean(v)),
                        "seeds": v.tolist()} for a, v, _, _, _ in rows], "lane F1 per arm", width=820, left=210)
    return ('<section id="overfitting"><h2>Fixing the UFLD overfitting</h2><p class="lede">The first pilot showed '
            'every UFLD run peaking at epoch 1. Each arm below changes one thing in the training of the UFLD baseline '
            '(same split, seeds, epochs and evaluation) and is compared with the original recipe (photometric '
            'augmentation). The recipe for later runs is chosen on the validation column; the held-out column is '
            'reported, never used for choosing. "Best epochs" are the validation-selected epochs per seed; the '
            '"more scenes" arm has its own, larger validation set. Three seeds cannot reach significance, so the '
            f'verdicts are indicative.</p><div class="panel"><h3>Held-out lane F1 per arm</h3>{chart}</div>'
            '<div class="scroll"><table><thead><tr><th>Arm</th><th class="num">Validation lane F1</th>'
            '<th class="num">Held-out lane F1</th><th class="num">Best epochs</th><th>Held-out Δ vs reference</th>'
            f'</tr></thead><tbody>{body}</tbody></table></div></section>')


def _history_section(root: Path) -> str:
    res = _read(root / "ablation_history" / "ablation_results.csv")
    if res is None or res.empty:
        return ""
    metric = "lane_f1_iou50"
    res = res[res["split"] == "test"] if "split" in res else res
    temporal = [v for v in res["variant"].unique() if VARIANTS.get(v) and VARIANTS[v].temporal]
    parts = []
    for v in temporal:
        base = VARIANTS[v].reference
        gains, f1s = {}, {}
        steps, frames = set(), set()
        for arm, sub in res.groupby("arm"):
            m = re.match(r"s(\d+)_t(\d+)", arm)
            if not m:
                continue
            s, t = int(m.group(1)), int(m.group(2))
            a = sub[sub["variant"] == v].set_index("seed")[metric]
            b = sub[sub["variant"] == base].set_index("seed")[metric]
            seeds = sorted(set(a.index) & set(b.index))
            if not seeds:
                continue
            steps.add(s)
            frames.add(t)
            f1s[(t, s)] = float(a.loc[seeds].mean())
            gains[(t, s)] = float((a.loc[seeds] - b.loc[seeds]).mean())
        if not gains:
            continue
        rows, cols = sorted(frames), sorted(steps)
        parts.append(f'<div class="panel"><h3>{esc(LABELS[v])}: lane F1 gain over {esc(LABELS[base])}</h3>'
                     f'{heatmap(gains, rows, cols, "frames in the clip", "frame step (history frame k is t - k x step)", signed=True)}'
                     f'</div><div class="panel"><h3>{esc(LABELS[v])}: held-out lane F1</h3>'
                     f'{heatmap(f1s, rows, cols, "frames in the clip", "frame step")}</div>')
    if not parts:
        return ""
    return ('<section id="history"><h2>How far back to look</h2><p class="lede">History-length ablation: every '
            'arm uses the same split (purge gap large enough for the longest history) and the same trained baseline, '
            'so only the temporal context changes. Cells are means over seeds on the held-out scenes; "not run" '
            'arms can be added with <code>ablate --only</code>.</p><div class="grid2">' + "".join(parts) + "</div></section>")


def _roadmap_section(path: Path) -> str:
    if not path.exists():
        return ""
    phases, current = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = {"title": line[3:].strip(), "items": []}
            phases.append(current)
        elif line.startswith("- ") and current is not None:
            m = re.match(r"- \[(done|progress|next|later)\]\s*(.*)", line)
            status, text = (m.group(1), m.group(2)) if m else ("later", line[2:])
            current["items"].append((status, text))
    names = {"done": "done", "progress": "in progress", "next": "next", "later": "later"}
    html_phases = "".join(
        f'<div class="phase panel"><h3>{_md(p["title"])}</h3><ul>'
        + "".join(f'<li><span class="pill {s}">{names[s]}</span><span>{_md(t)}</span></li>' for s, t in p["items"])
        + "</ul></div>" for p in phases)
    return ('<section id="roadmap"><h2>Roadmap</h2><p class="lede">First make a lightweight temporal model beat the '
            'UFLD baseline under the full protocol; only then optimise for an in-car embedded system.</p>'
            f'{html_phases}</section>')


def _temporal_block(run: Path, df: pd.DataFrame) -> str:
    """Paired comparisons (plain and equal-training), Kalman reference, carried
    state and the test-time history ablation of one run, as HTML."""
    report = run / "report"
    paired = _read(report / "paired_tests_test_tuned.csv")
    temporal_part = ""
    if paired is not None and not paired.empty:
        pm = paired[paired["metric"] == "lane_f1_iou50"]
        ref, var = pm["reference"].astype(str), pm["variant"].astype(str)
        is_ct = ref.str.endswith("_ct") | var.str.endswith("_ct")
        is_static = ref.str.endswith("_static") | var.str.endswith("_static")
        plain = pm[~is_ct & ~is_static]
        budget = pm[is_ct & ~is_static]
        capacity = pm[is_static & ~is_ct]
        sub = df[(df["split"] == "test") & (df["protocol"] == "tuned") & (df["input"] == "full")]

        def deltas(frame: pd.DataFrame) -> list[dict]:
            rows = []
            for r in frame.itertuples():
                a = sub[sub["variant"] == r.variant].set_index("seed")["lane_f1_iou50"]
                b = sub[sub["variant"] == r.reference].set_index("seed")["lane_f1_iou50"]
                common = sorted(set(a.index) & set(b.index))
                rows.append({"label": f"{_label(r.variant)} − {_label(r.reference)}", "mean": float(r.mean_delta),
                             "seeds": [float(a[c] - b[c]) for c in common]})
            return rows

        temporal_part += (f'<div class="panel"><h3>Temporal model minus its single-frame baseline (held-out lane F1)</h3>'
                          f'{delta_chart(deltas(plain), label_w=280)}</div>{paired_table(plain, "Temporal model")}')
        if not budget.empty:
            temporal_part += ('<h3>Equal-training control</h3><p class="lede">Temporal models start from their '
                              'baseline\'s best checkpoint and train further. The +CT baseline gets the same extra '
                              'training; a temporal gain that survives this comparison is not an effect of more epochs.</p>'
                              f'<div class="panel">{delta_chart(deltas(budget), label_w=280)}</div>{paired_table(budget)}')
        if not capacity.empty:
            temporal_part += ('<h3>Capacity control: extra layers or earlier frames?</h3><p class="lede">Lite v0.5 '
                              'static is lite v0.5 with every history frame replaced by the current frame, in training '
                              'and at test time: the same layers, warm start and budget, but no temporal information. '
                              '"Lite v0.5 − static" is what the earlier frames add; "static − baseline" is what the '
                              'extra fusion layers add on their own.</p>'
                              f'<div class="panel">{delta_chart(deltas(capacity), label_w=280)}</div>'
                              f'{paired_table(capacity)}')
    kalman = _read(report / "kalman_reference.csv")
    kpaired = _read(report / "paired_tests_kalman.csv")
    if kalman is not None and not kalman.empty:
        piv = kalman.set_index(["variant", "metric"])["mean"]

        def val(v: str, m: str, digits: int) -> str:
            return fmt(piv[(v, m)], digits) if (v, m) in piv.index else "n/a"

        models = [v for v in ORDER if v in set(kalman["variant"])]
        body = "".join(f'<tr><td><span class="fam {FAMILY[v]}"></span>{esc(LABELS[v])}</td>'
                       f'<td class="num">{val(v, "lane_f1_iou50", 3)}</td><td class="num">{val(v + "+kf", "lane_f1_iou50", 3)}</td>'
                       f'<td class="num">{val(v, "jitter_px", 2)}</td><td class="num">{val(v + "+kf", "jitter_px", 2)}</td></tr>'
                       for v in models)
        kp = None if kpaired is None else kpaired[~kpaired["variant"].astype(str).str.endswith("+kf")]
        temporal_part += ('<h3>Against the Kalman tracker</h3><p class="lede">The same predictions filtered by the '
                          'output Kalman tracker (tuned on validation). This is what time gives almost for free; a '
                          'learned temporal model should beat its baseline + Kalman (paired table below).</p>'
                          '<div class="scroll"><table><thead><tr><th>Model</th><th class="num">Lane F1</th>'
                          '<th class="num">Lane F1 + Kalman</th><th class="num">Jitter px</th>'
                          f'<th class="num">Jitter px + Kalman</th></tr></thead><tbody>{body}</tbody></table></div>'
                          + paired_table(kp, "Temporal model"))
    robust = _read(report / "robustness_summary.csv")
    if robust is not None and not robust.empty:
        piv = robust.set_index(["variant", "input", "op"])["lane_f1"]
        clean_f1 = robust.groupby("variant")["clean_lane_f1"].first()
        ops = [o for o in ("occlude", "blur", "darken", "noise") if o in set(robust["op"])]
        models = [v for v in ORDER if v in set(robust["variant"])]

        def cell(v: str, inp: str, op: str) -> str:
            return fmt(piv[(v, inp, op)], 3) if (v, inp, op) in piv.index else "n/a"

        body = "".join(
            f'<tr><td><span class="fam {FAMILY[v]}"></span>{esc(LABELS[v])}</td><td class="num">{fmt(clean_f1.get(v), 3)}</td>'
            f'<td class="num"><strong>{cell(v, "degraded", "all")}</strong></td>'
            + "".join(f'<td class="num">{cell(v, "degraded", o)}</td>' for o in ops)
            + f'<td class="num">{cell(v, "degraded+kalman", "all")}</td></tr>' for v in models)
        chart_rows = [{"variant": v, "mean": float(piv[(v, "degraded", "all")]),
                       "seeds": []} for v in models if (v, "degraded", "all") in piv.index]
        seeds_file = _read(report / "robustness.csv")
        if seeds_file is not None:
            for r in chart_rows:
                s_ = seeds_file[(seeds_file["variant"] == r["variant"]) & (seeds_file["input"] == "degraded")
                                & (seeds_file["op"] == "all")]
                r["seeds"] = s_["lane_f1"].astype(float).tolist()
        temporal_part += ('<h3>When the current frame is degraded</h3><p class="lede">The held-out test scenes again, '
                          'but every current frame is corrupted (an occluding box, strong blur, darkening or noise; the '
                          'same corruption for every model) while the earlier frames stay clean. This is where earlier '
                          'frames should pay off: a single-frame model only sees the damaged image. Validation-tuned '
                          'post-processing, nothing re-tuned.</p>'
                          f'<div class="panel"><h3>Held-out lane F1 with a degraded current frame</h3>'
                          f'{bar_chart(chart_rows, "lane F1 degraded")}</div>'
                          '<div class="scroll"><table><thead><tr><th>Model</th><th class="num">Clean</th>'
                          '<th class="num">Degraded</th>' + "".join(f'<th class="num">{o}</th>' for o in ops)
                          + '<th class="num">Degraded + Kalman</th></tr></thead>'
                          f'<tbody>{body}</tbody></table></div>')
    carry = _read(report / "carry_state.csv")
    if carry is not None and not carry.empty:
        col = [c for c in carry.columns if c.startswith("mean_gain")][0]
        body = "".join(f'<tr><td>{esc(_label(r.variant))}</td><td class="num">{getattr(r, col):+.4f}</td>'
                       f'<td class="num">{int(r.n_seeds)}</td></tr>' for r in carry.itertuples())
        temporal_part += ('<h3>Recurrent models with a carried state</h3><p class="lede">Held-out lane F1 when the '
                          'ConvGRU keeps one state per stream across the whole scene, minus the window mode used in '
                          'training (a zero state at every clip).</p><div class="scroll"><table><thead><tr>'
                          '<th>Model</th><th class="num">Carry − window</th><th class="num">Seeds</th></tr></thead>'
                          f'<tbody>{body}</tbody></table></div>')
    hist_tbl = _history_gain_table(report / "temporal_ablation.csv")
    if hist_tbl:
        temporal_part += ('<h3>Is it the history?</h3><p class="lede">Test-time ablation: history frames replaced by '
                          'the current frame. The gain is what the real history contributes to each trained model.</p>'
                          + hist_tbl)
    return temporal_part


def _extra_run_section(run: Path, main_run: Path, main_df: pd.DataFrame) -> str:
    """A further run (e.g. the lite family with a longer budget): its results,
    its temporal comparisons, and a head-to-head chart that puts its models
    next to the other family's models from the main run."""
    import yaml

    df = pd.read_csv(run / "all_results.csv")
    resolved = run / "config_resolved.yaml"
    cfg = yaml.safe_load(resolved.read_text(encoding="utf-8")) if resolved.exists() else {}
    main_cfg_path = main_run / "config_resolved.yaml"
    main_cfg = yaml.safe_load(main_cfg_path.read_text(encoding="utf-8")) if main_cfg_path.exists() else {}
    epochs = cfg.get("train", {}).get("epochs", "?")
    main_epochs = main_cfg.get("train", {}).get("epochs", "?")
    here = set(df["variant"])
    families = {FAMILY[v] for v in here}
    rows = _agg(df, "test", "tuned", "lane_f1_iou50")
    others = [dict(r, label=f"{_label(r['variant'])} ({main_epochs} ep)") for r in _agg(main_df, "test", "tuned", "lane_f1_iou50")
              if FAMILY[r["variant"]] not in families]
    mine = [dict(r, label=f"{_label(r['variant'])} ({epochs} ep)") for r in rows]
    head = (f'<div class="panel"><h3>Head to head on the held-out roads</h3>'
            f'{bar_chart(others + mine, "lane F1 head to head", width=820, left=210)}'
            f'<p class="lede">Same split, recipe and evaluation; the numbers in brackets are the maximum epochs of '
            f'each run (early stopping may end sooner). The {", ".join(sorted(families))} models come from '
            f'<code>{esc(run.name)}</code>, the others from <code>{esc(main_run.name)}</code>.</p></div>') if others else ""
    title = {"elas_lite_long": "The lite family with a longer budget"}.get(run.name, f"Run {run.name}")
    return (f'<section id="{esc(run.name)}"><h2>{esc(title)}</h2><p class="lede">{esc(cfg.get("description", ""))}</p>'
            f'{head}<h3>Held-out test scenes (mean ± std over seeds)</h3>{results_table(df, "test", "tuned")}'
            f'{_temporal_block(run, df)}</section>')


def _md_lists(path: Path) -> dict[str, list[str]]:
    """``## Heading`` -> its ``- item`` / ``1. item`` lines, in order."""
    out, current = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = out.setdefault(line[3:].strip(), [])
        elif current is not None and re.match(r"(- |\d+\. )", line):
            current.append(re.sub(r"^(- |\d+\. )", "", line).strip())
    return out


def _family_views(run: Path, df: pd.DataFrame, extra_runs: list[Path]):
    """Held-out test rows of the main run, of a further lite-only run (e.g. a
    longer lite training) when given, both combined (the lite family taken
    from the further run), and the efficiency table of both runs."""
    def test_rows(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[(frame["split"] == "test") & (frame["protocol"] == "tuned") & (frame["input"] == "full")]

    main = test_rows(df)
    lite_run, lite = None, None
    for r in extra_runs:
        r = r if r.is_absolute() else PROJECT_ROOT / r
        if (r / "all_results.csv").exists():
            d = test_rows(pd.read_csv(r / "all_results.csv"))
            if not d.empty and all(FAMILY.get(v) == "lite" for v in d["variant"].unique()):
                lite_run, lite = r, d
    both = pd.concat([main[main["variant"].map(FAMILY) != ("lite" if lite is not None else "none")],
                      lite if lite is not None else main.iloc[0:0]], ignore_index=True)
    effs = [e for e in (_read(run / "report" / "efficiency.csv"),
                        _read(lite_run / "report" / "efficiency.csv") if lite_run else None) if e is not None]
    eff = pd.concat(effs).drop_duplicates("variant", keep="last").set_index("variant") if effs else None
    return main, lite_run, lite, both, eff


def _summary_section(run: Path, df: pd.DataFrame, extra_runs: list[Path], summary: Path | None) -> str:
    """One-screen overview for a short presentation: key figures and the four
    most telling charts, all computed from the run folders, plus the
    hand-written challenges and next steps (``docs/RESEARCH_SUMMARY.md``)."""
    metric = "lane_f1_iou50"
    root = run.parent
    report = run / "report"
    main, lite_run, lite, both, eff = _family_views(run, df, extra_runs)
    f1 = both.groupby("variant")[metric].mean()
    seeds = lambda frame, v: frame[frame["variant"] == v].sort_values("seed")[metric].astype(float).tolist()
    temporal_ufld = [v for v in f1.index if FAMILY.get(v) == "ufld" and VARIANTS[v].temporal]
    kpis, panels = [], []

    aug = _read(root / "ablation_augmentation" / "ablation_results.csv")
    if aug is not None and {"photometric", "geometric"} <= set(aug["arm"]):
        a = aug[aug["split"] == "test"] if "split" in aug else aug
        before, after = a[a["arm"] == "photometric"][metric].mean(), a[a["arm"] == "geometric"][metric].mean()
        kpis.append(("Overfitting fixed", f"{before:.2f} → {after:.2f}",
                     "UFLD baseline, held-out lane F1: original augmentation vs + geometric augmentation (3 seeds)"))
        arm_rows = []
        for arm in dict.fromkeys(a["arm"]):
            v = a[a["arm"] == arm][metric].astype(float)
            arm_rows.append({"variant": "ufld_baseline", "label": arm.replace("_", " "), "mean": float(v.mean()),
                             "seeds": v.tolist()})
        arm_rows.sort(key=lambda r: -r["mean"])
        panels.append(("1 · Overfitting: what fixed it", bar_chart(arm_rows, "held-out lane F1 per training recipe",
                                                                    width=540, left=150),
                       "UFLD baseline trained with each recipe (same split, seeds and epochs); held-out lane F1. "
                       "Geometric augmentation (shift, zoom, rotation, perspective) is the fix; flips and heavy "
                       "regularisation are not."))

    if "ufld_baseline" in f1 and temporal_ufld:
        best = max(temporal_ufld, key=lambda v: f1[v])
        kpis.append(("Clean frames", f"{f1['ufld_baseline']:.3f} vs {f1[best]:.3f}",
                     f"UFLD baseline vs the best temporal model ({LABELS[best]}): within seed noise "
                     f"({len(seeds(both, best))} seeds)"))

    rob_rows, occ_kpi = [], None
    for source, frame in ((run, main), (lite_run, lite)):
        rob = _read(source / "report" / "robustness_summary.csv") if source is not None else None
        per_seed = _read(source / "report" / "robustness.csv") if source is not None else None
        if rob is None:
            continue
        occ = rob[(rob["input"] == "degraded") & (rob["op"] == "occlude")].set_index("variant")
        for v in [v for v in ORDER if v in occ.index]:
            if source == run and lite is not None and FAMILY[v] == "lite":
                continue
            if v.endswith("_ct") and FAMILY[v] == "lite":
                continue
            rob_rows.append({"variant": v, "label": LABELS[v], "a": float(occ.loc[v, "clean_lane_f1"]),
                             "b": float(occ.loc[v, "lane_f1"])})
        cands = [v for v in occ.index if FAMILY.get(v) == "ufld" and VARIANTS[v].temporal]
        if source == run and cands and "ufld_baseline" in occ.index and per_seed is not None:
            best = max(cands, key=lambda v: occ.loc[v, "lane_f1"])
            ps = per_seed[(per_seed["input"] == "degraded") & (per_seed["op"] == "occlude")]
            gain = (ps[ps["variant"] == best].set_index("seed")["lane_f1"]
                    - ps[ps["variant"] == "ufld_baseline"].set_index("seed")["lane_f1"]).dropna()
            every = "in every seed" if len(gain) and (gain > 0).all() else "not in every seed"
            occ_kpi = ("Occluded current frame", f"{occ.loc['ufld_baseline', 'lane_f1']:.3f} → "
                       f"{occ.loc[best, 'lane_f1']:.3f}",
                       f"UFLD baseline vs {LABELS[best]} when the current frame is occluded and the earlier frames "
                       f"are clean: {gain.mean():+.3f}, {every}")
    if occ_kpi:
        kpis.append(occ_kpi)
    if rob_rows:
        panels.append(("2 · Where time helps: an occluded current frame",
                       dumbbell_chart(rob_rows, "clean current frame", "occluded current frame", width=540, left=150),
                       "Held-out lane F1 with clean frames (hollow) and with the current frame occluded by boxes "
                       "while the earlier frames stay clean (filled); the number is the occluded F1 and its change. "
                       "Single-frame models only see the damaged image."))

    kal = _read(report / "kalman_reference.csv")
    if kal is not None and not kal.empty:
        k = kal[kal["metric"] == "jitter_px"].set_index("variant")["mean"]
        red = [1 - k[v + "+kf"] / k[v] for v in k.index if not v.endswith("+kf") and v + "+kf" in k.index]
        if red:
            kpis.append(("Kalman tracker on the output", f"−{100 * float(np.mean(red)):.0f} % jitter",
                         "for every model, with unchanged lane F1: the cheap temporal reference a learned model "
                         "has to beat"))

    if eff is not None and "ufld_baseline" in f1 and "lite_v05" in f1 and "lite_v05" in eff.index:
        pb, pl = eff.loc["ufld_baseline", "params_millions"], eff.loc["lite_v05", "params_millions"]
        kpis.append(("Lightweight model", f"{f1['lite_v05']:.3f} · {pl:.1f} M",
                     f"Lite v0.5 lane F1 and parameters, against {f1['ufld_baseline']:.3f} with {pb:.1f} M for the "
                     f"UFLD baseline ({pb / pl:.0f}× fewer parameters)"))
    if eff is not None:
        size_rows = []
        for v in [v for v in ORDER if v in f1.index and v in eff.index and not v.endswith("_ct")]:
            e = eff.loc[v]
            size_rows.append({"variant": v, "mean": float(f1[v]), "seeds": seeds(both, v),
                              "label": f"{LABELS[v]} · {e['params_millions']:.1f} M"})
        src = f" Lite models from <code>{esc(lite_run.name)}</code> (longer training)." if lite_run else ""
        panels.append(("3 · Accuracy against size", bar_chart(size_rows, "held-out lane F1 with model size",
                                                              width=540, left=190),
                       f"Held-out lane F1 (clean frames) and parameters. Compute per frame: UFLD {eff.loc['ufld_baseline', 'gmacs']:.1f} GMAC, "
                       f"lite {eff.loc['lite_baseline', 'gmacs']:.1f} GMAC; a temporal model streaming with cached features "
                       "adds only its fusion (it recomputes 3 frames otherwise)."
                       + src))

    hist = _read(root / "ablation_history" / "ablation_results.csv")
    if hist is not None and not hist.empty:
        h = hist[hist["split"] == "test"] if "split" in hist else hist
        tv = [v for v in h["variant"].unique() if VARIANTS.get(v) and VARIANTS[v].temporal]
        rows = []
        for arm in sorted(h["arm"].unique(), key=lambda a: tuple(int(n) for n in re.findall(r"\d+", a))):
            m = re.match(r"s(\d+)_t(\d+)", arm)
            if not m or not tv:
                continue
            sub = h[h["arm"] == arm]
            a_ = sub[sub["variant"] == tv[0]].set_index("seed")[metric]
            b_ = sub[sub["variant"] == VARIANTS[tv[0]].reference].set_index("seed")[metric]
            d = (a_ - b_).dropna()
            if len(d):
                rows.append({"label": f"{m.group(2)} frames, step {m.group(1)}", "mean": float(d.mean()),
                             "seeds": d.tolist()})
        if rows:
            panels.append((f"4 · How far back to look ({LABELS[tv[0]]})", delta_chart(rows, width=540, label_w=150),
                           f"Held-out lane F1 of {LABELS[tv[0]]} minus the UFLD baseline for each history length and "
                           "spacing (history frame k = t − k·step; same split and baseline for every arm). "
                           "Spacing changes little; five frames are worse than three."))

    plan = _md_lists(summary) if summary is not None and summary.exists() else {}
    tiles = "".join(f'<div class="kpi"><span class="k">{esc(k)}</span><span class="v">{esc(v)}</span>'
                    f'<span class="l">{esc(l)}</span></div>' for k, v, l in kpis)
    charts = "".join(f'<figure class="panel"><h3>{esc(t)}</h3>{svg}<figcaption>{cap}</figcaption></figure>'
                     for t, svg, cap in panels)
    ordered = lambda title: "ol" if title.lower().startswith(("next", "próximos")) else "ul"
    lists = "".join(f'<div class="panel"><h3>{esc(title)}</h3><{ordered(title)}>'
                    + "".join(f"<li>{_md(i)}</li>" for i in items)
                    + f'</{ordered(title)}></div>' for title, items in plan.items())
    return ('<section id="summary"><h2>Where the research stands</h2><p class="lede">Question: can a lightweight '
            'lane detector that also looks at the previous frames beat the single-frame UFLD baseline, and run on '
            'an in-car embedded system? Everything below comes from pilot runs on ELAS (2–3 seeds, short training, '
            'no hyper-parameter search), so it shows directions, not final claims.</p>'
            f'<div class="kpis">{tiles}</div><div class="charts">{charts}</div>'
            + (f'<div class="plan">{lists}</div>' if lists else "") + "</section>")


def _bi(pt: str, en: str) -> str:
    """Portuguese first, English below it in muted text (both already HTML)."""
    return f'<span lang="pt">{pt}</span><span class="en" lang="en">{en}</span>'


def _pt(text: str) -> str:
    """Decimal comma for numbers inside Portuguese text."""
    return re.sub(r"(?<![\w.])(\d+)\.(\d+)", r"\1,\2", text)  # not in names like v0.3


def _choice_section(run: Path, df: pd.DataFrame, extra_runs: list[Path], notes: Path | None) -> str:
    """Which model to compare with the baseline: per situation, which model
    beats the UFLD baseline (numbers computed from the runs), a side-by-side
    table of the candidates, and the hand-written recommendation and model
    notes (``docs/MODEL_CHOICE.md``). Portuguese first, English below."""
    import yaml

    metric, base, occ = "lane_f1_iou50", "ufld_baseline", "occlude"
    main, lite_run, lite, both, eff = _family_views(run, df, extra_runs)
    f1 = both.groupby("variant")[metric].mean()
    if base not in f1:
        return ""
    sources = [run] + ([lite_run] if lite_run else [])
    own = lambda source, v: not (source == run and lite_run is not None and FAMILY.get(v) == "lite")
    frame_of = lambda v: lite if lite is not None and FAMILY[v] == "lite" else main
    seed_f1 = lambda v: frame_of(v)[frame_of(v)["variant"] == v].set_index("seed")[metric].astype(float)
    rob, rob_seed, hist_gain = {}, {}, {}
    for source in sources:  # every number comes from the run that trained the model
        s = _read(source / "report" / "robustness_summary.csv")
        ps = _read(source / "report" / "robustness.csv")
        ta = _read(source / "report" / "temporal_ablation.csv")
        for r in ([] if s is None else s[s["input"] == "degraded"].itertuples()):
            if own(source, r.variant):
                rob[(r.variant, r.op)] = float(r.lane_f1)
        for r in ([] if ps is None else ps[ps["input"] == "degraded"].itertuples()):
            if own(source, r.variant):
                rob_seed[(r.variant, r.op, int(r.seed))] = float(r.lane_f1)
        if ta is not None:
            col = next(c for c in ta.columns if c.startswith("mean_gain"))
            hist_gain.update({r.variant: float(getattr(r, col)) for r in ta.itertuples() if own(source, r.variant)})

    def rob_delta_seeds(a: str, b: str, op: str) -> pd.Series:
        sa = pd.Series({s: v for (m, o, s), v in rob_seed.items() if m == a and o == op}, dtype=float)
        sb = pd.Series({s: v for (m, o, s), v in rob_seed.items() if m == b and o == op}, dtype=float)
        return (sa - sb).dropna()

    L = lambda v: esc(LABELS[v])
    temporal = [v for v in f1.index if FAMILY[v] == "ufld" and VARIANTS[v].temporal]
    yes = lambda every: ("Sim, em todas as sementes", "Yes, in every seed") if every else ("Sim, na média", "Yes, on average")
    rows = []  # (situation, verdict, best model, numbers), each a (pt, en) pair

    if temporal:  # 1. clean frames
        best = max(temporal, key=lambda v: f1[v])
        d = f1[best] - f1[base]
        hg = hist_gain.get(best, float("nan"))
        every = bool(((seed_f1(best) - seed_f1(base)).dropna() > 0).all())
        temporal_gain = d > 0 and every and np.isfinite(hg) and hg >= 0.01
        rows.append((("Quadros limpos", "Clean frames"),
                     yes(True) if temporal_gain else ("Não claramente", "Not clearly"),
                     (L(best), L(best)) if temporal_gain else ("nenhum ainda", "none yet"),
                     (f"{L(best)}: {d:+.3f} ({f1[best]:.3f} × {f1[base]:.3f})"
                      + (f"; sem o histórico perde só {hg:.3f}" if np.isfinite(hg) else "")
                      + ("" if temporal_gain or not np.isfinite(hg) else ", então o ganho não é temporal"),
                      f"{L(best)}: {d:+.3f} ({f1[best]:.3f} vs {f1[base]:.3f})"
                      + (f"; without its history it loses only {hg:.3f}" if np.isfinite(hg) else "")
                      + ("" if temporal_gain or not np.isfinite(hg) else ", so the gain is not temporal"))))
    occ_d = {v: rob[(v, occ)] - rob[(base, occ)] for v in temporal if (v, occ) in rob and (base, occ) in rob}
    if occ_d:  # 2. occluded current frame
        ranked = sorted(occ_d, key=lambda v: -occ_d[v])
        best = ranked[0]
        every = bool(len(rob_delta_seeds(best, base, occ)) and (rob_delta_seeds(best, base, occ) > 0).all())
        top = ", ".join(f"{L(v)} {occ_d[v]:+.3f}" for v in ranked[:3])
        ct = (f"; controle com o mesmo treino {rob[('ufld_baseline_ct', occ)]:.3f}",
              f"; equal-training control {rob[('ufld_baseline_ct', occ)]:.3f}") if ("ufld_baseline_ct", occ) in rob else ("", "")
        rows.append((("Quadro atual ocluído (histórico limpo)", "Occluded current frame (clean history)"),
                     yes(every) if occ_d[best] > 0 else ("Não", "No"), (L(best), L(best)),
                     (f"{top} (baseline {rob[(base, occ)]:.3f}{ct[0]})", f"{top} (baseline {rob[(base, occ)]:.3f}{ct[1]})")))
    other = [(v, op, rob[(v, op)] - rob[(base, op)]) for v in temporal for op in ("blur", "darken", "noise")
             if (v, op) in rob and (base, op) in rob]
    if other:  # 3. other corruptions
        lo, hi = min(d for *_, d in other), max(d for *_, d in other)
        clear = hi >= 0.03
        best_v, best_op, _ = max(other, key=lambda x: x[2])
        rows.append((("Quadro atual borrado, escuro ou com ruído", "Blurred, dark or noisy current frame"),
                     ("Sim", "Yes") if clear else ("Não", "No"),
                     (L(best_v), L(best_v)) if clear else ("nenhum", "none"),
                     (f"todas as diferenças entre {lo:+.3f} e {hi:+.3f}", f"every difference between {lo:+.3f} and {hi:+.3f}")))
    cond = _read(run / "report" / "test_conditions.csv")
    if cond is not None:  # 4. the scene tagged as occluded (natural occlusions)
        c = cond[cond["metric"] == "condition_f1_iou50_occlusion"].set_index("variant")["mean"]
        cands = [v for v in temporal if v in c.index]
        if base in c.index and cands:
            best = max(cands, key=lambda v: c[v])
            ctx = (f"; +CT {c['ufld_baseline_ct']:.3f}" if "ufld_baseline_ct" in c.index else "")
            rows.append((("Cena de teste com oclusões reais", "Held-out scene with real occlusions"),
                         ("Sim, mas é uma única cena", "Yes, but it is a single scene") if c[best] > c[base] else ("Não", "No"),
                         (L(best), L(best)),
                         (f"{L(best)} {c[best]:.3f} × baseline {c[base]:.3f}{ctx}",
                          f"{L(best)} {c[best]:.3f} vs baseline {c[base]:.3f}{ctx}")))
    kal = _read(run / "report" / "kalman_reference.csv")
    if kal is not None and not kal.empty:  # 5. smoothness
        k = kal[kal["metric"] == "jitter_px"].set_index("variant")["mean"]
        if base in k.index and base + "+kf" in k.index:
            learned = [k[v] for v in temporal if v in k.index]
            red = 100 * (1 - k[base + "+kf"] / k[base])
            learned_wins = bool(learned) and min(learned) < 0.9 * k[base]
            rows.append((("Estabilidade (jitter)", "Smoothness (jitter)"),
                         ("Sim", "Yes") if learned_wins else ("Não por um modelo aprendido", "Not by a learned model"),
                         ("filtro de Kalman na saída", "Kalman filter on the output"),
                         (f"baseline {k[base]:.2f} px → {k[base + '+kf']:.2f} px com Kalman (−{red:.0f} %); "
                          f"modelos temporais {min(learned):.2f}–{max(learned):.2f} px",
                          f"baseline {k[base]:.2f} px → {k[base + '+kf']:.2f} px with Kalman (−{red:.0f} %); "
                          f"temporal models {min(learned):.2f}–{max(learned):.2f} px")))
    lv = "lite_v05"
    if lv in f1 and eff is not None and lv in eff.index and (lv, occ) in rob and (base, occ) in rob:  # 6. lightweight
        dc, do = f1[lv] - f1[base], rob[(lv, occ)] - rob[(base, occ)]
        pb, pl = eff.loc[base, "params_millions"], eff.loc[lv, "params_millions"]

        def epochs(r: Path | None) -> str:
            p = (r or run) / "config_resolved.yaml"
            return str(yaml.safe_load(p.read_text(encoding="utf-8")).get("train", {}).get("epochs", "?")) if p.exists() else "?"

        verdict = (("Sob oclusão sim; em quadros limpos não", "Under occlusion yes; on clean frames no") if do > 0 >= dc
                   else yes(True) if do > 0 and dc > 0 else ("Não", "No"))
        rows.append((("Modelo leve para embarcado (× UFLD baseline)", "Lightweight model for embedded use (vs UFLD baseline)"),
                     verdict, (L(lv), L(lv)),
                     (f"limpos {dc:+.3f}, ocluídos {do:+.3f}, {pb / pl:.0f}× menos parâmetros ({pl:.1f} M × {pb:.1f} M); "
                      f"treinos diferentes: até {epochs(lite_run)} × {epochs(run)} épocas",
                      f"clean {dc:+.3f}, occluded {do:+.3f}, {pb / pl:.0f}× fewer parameters ({pl:.1f} M vs {pb:.1f} M); "
                      f"different training: up to {epochs(lite_run)} vs {epochs(run)} epochs")))
    ls = "lite_v05_static"
    if (lv, occ) in rob and (ls, occ) in rob:  # 7. does the lite model use its history?
        ds = rob_delta_seeds(lv, ls, occ)
        per = " / ".join(f"{d:+.3f}" for d in ds)
        rows.append((("O modelo leve usa o histórico? (× mesmas camadas sem histórico, com oclusão)",
                      "Does the lite model use its history? (vs same layers without history, occluded)"),
                     yes(bool(len(ds) and (ds > 0).all())) if rob[(lv, occ)] > rob[(ls, occ)] else ("Não", "No"),
                     (L(lv), L(lv)),
                     (f"{rob[(lv, occ)]:.3f} × {rob[(ls, occ)]:.3f} (por semente {per})",
                      f"{rob[(lv, occ)]:.3f} vs {rob[(ls, occ)]:.3f} (per seed {per})")))
    if not rows:
        return ""
    head = ("<tr>" + "".join(f"<th>{_bi(p, e)}</th>" for p, e in (("Situação", "Situation"),
                                                                 ("Supera a baseline?", "Better than the baseline?"),
                                                                 ("Melhor modelo", "Best model"), ("Números", "Numbers")))
            + "</tr>")
    body = "".join("<tr>" + "".join(f'<td class="wrap">{_bi(_pt(p), e)}</td>' for p, e in row) + "</tr>" for row in rows)

    plan = _md_lists(notes) if notes is not None and notes.exists() else {}
    model_notes = {}
    for item in plan.get("Notas por modelo", []):
        m = re.match(r"`(\w+)`:\s*(.*?)\s*\|\|\s*(.*)", item)
        if m:
            model_notes[m.group(1)] = (m.group(2), m.group(3))
    cand = []
    for v in [v for v in ORDER if v in f1.index]:
        o = rob.get((v, occ))
        do = o - rob[(base, occ)] if o is not None and (base, occ) in rob else None
        params = f"{eff.loc[v, 'params_millions']:.1f}" if eff is not None and v in eff.index else "n/a"
        hg = hist_gain.get(v)
        note = model_notes.get(v, ("", ""))
        cand.append(f'<tr><td><span class="fam {FAMILY[v]}"></span>{L(v)}</td><td class="num">{fmt(f1[v])}</td>'
                    f'<td class="num">{fmt(o) if o is not None else "n/a"}</td>'
                    f'<td class="num">{"—" if v == base else (f"{do:+.3f}" if do is not None else "n/a")}</td>'
                    f'<td class="num">{params}</td><td class="num">{f"{hg:.3f}" if hg is not None else "—"}</td>'
                    f'<td class="wrap">{_bi(_md(note[0]), _md(note[1])) if note[0] else ""}</td></tr>')
    cand_head = ("<tr>" + "".join(f'<th{c}>{_bi(p, e)}</th>' for p, e, c in (
        ("Modelo", "Model", ""), ("F1 limpos", "Clean F1", ' class="num"'), ("F1 ocluídos", "Occluded F1", ' class="num"'),
        ("Δ oclusão × baseline", "Δ occluded vs baseline", ' class="num"'), ("Parâmetros (M)", "Parameters (M)", ' class="num"'),
        ("Perda sem histórico", "Cost without history", ' class="num"'), ("Nota", "Note", ""))) + "</tr>")
    rec = "".join(f'<div class="panel" lang="{"pt" if lang == "Recomendação" else "en"}"><h3>{esc(lang)}</h3><ul>'
                  + "".join(f"<li>{_md(i)}</li>" for i in plan[lang]) + "</ul></div>"
                  for lang in ("Recomendação", "Recommendation") if plan.get(lang))
    src = _bi(f"Modelos lite: execução <code>{esc(lite_run.name)}</code> (treino mais longo).",
              f"Lite models: run <code>{esc(lite_run.name)}</code> (longer training).") if lite_run else ""
    return ('<section id="choice"><h2>Qual modelo comparar com a baseline<span class="en h2en">Which model to compare '
            'with the baseline</span></h2>'
            '<p class="lede">' + _bi("Lane F1 nas cenas de teste nunca vistas, média de 2 sementes; o quadro atual "
                                   "ocluído vem da avaliação de robustez (o histórico continua limpo). Resultados de "
                                   "piloto: indicam a direção, não provam.",
                                   "Lane F1 on the held-out scenes, mean of 2 seeds; the occluded current frame comes "
                                   "from the robustness evaluation (the history stays clean). Pilot results: they "
                                   "show the direction, they do not prove it.") + src + '</p>'
            f'<div class="scroll"><table class="choice"><thead>{head}</thead><tbody>{body}</tbody></table></div>'
            f'<div class="plan">{rec}</div>'
            f'<h3>{_bi("Os candidatos lado a lado", "The candidates side by side")}</h3>'
            f'<div class="scroll"><table><thead>{cand_head}</thead><tbody>{"".join(cand)}</tbody></table></div>'
            '<p class="lede">' + _bi("“Perda sem histórico”: quanto o lane F1 cai quando o histórico é trocado pelo "
                                   "quadro atual no teste (quadros limpos).",
                                   "“Cost without history”: how much lane F1 drops when the history is replaced by "
                                   "the current frame at test time (clean frames).") + "</p></section>")


def build_site(runs: list[Path], out: Path, title: str, extra: list[Path] | None = None,
               notes: Path | None = None, roadmap: Path | None = None, summary: Path | None = None,
               choice: Path | None = None) -> Path:
    """``runs[0]`` is the run shown; ablation summaries, benchmarks and the demo
    video are read from ``<results>/ablation_*``, ``<results>/benchmarks`` and
    ``<results>/demo`` next to it (``extra`` is reserved for further report files)."""
    out.mkdir(parents=True, exist_ok=True)
    assets = out / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir()
    run = runs[0] if runs[0].is_absolute() else PROJECT_ROOT / runs[0]
    results_root = run.parent
    df = pd.read_csv(run / "all_results.csv")
    env = json.loads((run / "environment.json").read_text(encoding="utf-8")) if (run / "environment.json").exists() else {}
    seeds = sorted(df["seed"].unique().tolist())
    duration = run_duration(run)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    notes_path = notes or PROJECT_ROOT / "docs" / "PILOT_FINDINGS.md"
    roadmap_path = roadmap or PROJECT_ROOT / "docs" / "ROADMAP.md"
    efficiency = _read(run / "report" / "efficiency.csv")
    present = set(df["variant"])
    s = []

    # header
    chips = ["dataset ELAS", f"{len(present)} models", f"{len(seeds)} seeds",
             "held-out scenes BR_S02 · VIX_S05 · VV_S03", env.get("gpu", "GPU n/a")]
    if duration:
        chips.append(f"GPU time {duration}")
    s.append(f'<header class="top"><div class="eyebrow">TAC-UFLD v{__version__} · {esc(run.name)}</div>'
             f'<h1>{esc(title)}</h1><p class="lede">Can a lightweight lane detector that looks at the previous '
             'frames beat the single-frame UFLD baseline? UFLD (ResNet-18) and a small CNN, each with temporal '
             'variants, trained on the ELAS ego-lane dataset and tested on three road scenes never used for training '
             'or model selection.</p><div class="meta">'
             + "".join(f'<span class="chip">{esc(c)}</span>' for c in chips)
             + '<span class="chip warn">indicative pilot, not the full protocol</span></div></header>')

    # one-screen overview (key figures, four charts, challenges, next steps)
    s.append(_summary_section(run, df, runs[1:], summary or PROJECT_ROOT / "docs" / "RESEARCH_SUMMARY.md"))
    # which model to compare with the baseline (Portuguese first)
    s.append(_choice_section(run, df, runs[1:], choice or PROJECT_ROOT / "docs" / "MODEL_CHOICE.md"))

    # findings (hand-written analysis)
    # (a Portuguese version next to the notes, ``<notes>.pt.md``, is shown first)
    if notes_path.exists():
        versions = [("pt", "O que os resultados mostram", notes_path.with_name(notes_path.stem + ".pt.md")),
                    ("en", "What the results show", notes_path)]
        blocks = []
        for lang, heading, path in versions:
            if path.exists():
                items = [l[2:].strip() for l in path.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
                blocks.append(f'<h2 lang="{lang}">{heading}</h2><div class="findings" lang="{lang}"><ul>'
                              + "".join(f"<li>{_md(i)}</li>" for i in items) + "</ul></div>")
        s.append('<section id="findings">' + "".join(blocks) + "</section>")

    s.append(models_section(efficiency, present))

    # results
    test_rows = _agg(df, "test", "tuned", "lane_f1_iou50")
    seen_rows = _agg(df, "seen_test", "tuned", "lane_f1_iou50")
    legend = ('<div class="legend"><span><i class="sw" style="background:var(--ufld)"></i>UFLD family</span>'
              '<span><i class="sw" style="background:var(--lite)"></i>Lite family</span>'
              '<span>bars = mean over seeds, dots = individual seeds</span></div>')
    s.append('<section id="results"><h2>Lane F1 on unseen roads</h2><p class="lede">CULane-style lane F1: each lane '
             'drawn 12 px wide (30 px at 1640 px, scaled to 640 px), matched one-to-one, correct when IoU ≥ 0.5. '
             'Post-processing tuned on validation only. On ELAS every model predicts both ego lanes in every frame, '
             'so precision = recall = F1: the errors are misplaced lanes, not missed ones.</p>'
             f'<div class="panel"><h3>Held-out test scenes (unseen roads)</h3>{bar_chart(test_rows, "lane F1 held-out")}</div>'
             + (f'<div class="panel"><h3>Seen-scene test blocks</h3>{bar_chart(seen_rows, "lane F1 seen")}</div>' if seen_rows else "")
             + f'{legend}<h3>Held-out test scenes, all metrics (mean ± std over seeds)</h3>{results_table(df, "test", "tuned")}'
             + (f'<h3>Seen-scene test</h3>{results_table(df, "seen_test", "tuned")}' if seen_rows else "") + '</section>')

    # does temporal context help?
    report = run / "report"
    temporal_part = _temporal_block(run, df)
    if temporal_part:
        n = len(seeds)
        s.append('<section id="temporal"><h2>Does temporal context help?</h2><p class="lede">Each temporal model is '
                 'compared with the single-frame model of its own family on the same seeds (exact Wilcoxon, Holm '
                 f'correction). With {n} seed{"s" if n != 1 else ""} no difference can reach p &lt; 0.05 unless there '
                 'are at least 6 seeds, so the verdicts here are indicative.</p>' + temporal_part + "</section>")

    # where temporal should help
    cond = _read(report / "test_conditions.csv")
    if cond is not None and not cond.empty:
        metrics = [m for m in dict.fromkeys(cond["metric"]) if m.startswith("condition_f1")] + ["jitter_px"]
        heads = "".join(f'<th class="num">{esc(m.replace("condition_f1_iou50_", "").replace("_px", " px"))}</th>'
                        for m in metrics)
        piv = cond.set_index(["variant", "metric"])
        body = []
        for v in [v for v in ORDER if v in set(cond["variant"])]:
            cells = "".join(f'<td class="num">{fmt(piv.loc[(v, m), "mean"], 2 if m == "jitter_px" else 3) if (v, m) in piv.index else "n/a"}</td>'
                            for m in metrics)
            body.append(f'<tr><td><span class="fam {FAMILY[v]}"></span>{esc(LABELS[v])}</td>{cells}</tr>')
        s.append('<section id="conditions"><h2>Where temporal information should help</h2><p class="lede">Held-out '
                 'lane F1 per scene condition from the ELAS scene tags (whole scenes, so these are three scenes, not '
                 'frame-level subsets) and jitter, the mean frame-to-frame change of the predicted lane position '
                 '(lower is steadier; it also contains real lane motion).</p><div class="scroll"><table><thead><tr>'
                 f'<th>Model</th>{heads}</tr></thead><tbody>{"".join(body)}</tbody></table></div></section>')

    # further runs (e.g. the lite family with a longer budget)
    for extra_run in runs[1:]:
        extra_run = extra_run if extra_run.is_absolute() else PROJECT_ROOT / extra_run
        if (extra_run / "all_results.csv").exists():
            s.append(_extra_run_section(extra_run, run, df))

    # ablations
    s.append(_augmentation_section(results_root))
    s.append(_history_section(results_root))

    # training curves
    hists = []
    for p in sorted(run.glob("seed_*/history_*.csv")):
        h = pd.read_csv(p)
        h["seed"] = p.parent.name.removeprefix("seed_")
        h["variant"] = p.stem.removeprefix("history_")
        hists.append(h)
    if hists:
        hist = pd.concat(hists, ignore_index=True)
        cells = []
        for v in [v for v in ORDER if v in set(hist["variant"])]:
            cells.append(f'<figure class="panel"><h3><span class="fam {FAMILY[v]}"></span>{esc(LABELS[v])}</h3>'
                         f'{curves_chart(hist[hist["variant"] == v], "val_lane_f1_iou50")}'
                         '<figcaption>validation lane F1 per epoch; solid = first seed, dashed = second, dot = best '
                         'epoch (the checkpoint kept)</figcaption></figure>')
        s.append('<section id="curves"><h2>Training curves</h2><p class="lede">Temporal models and +CT controls start '
                 'from their family baseline\'s best checkpoint.</p><div class="grid2">' + "".join(cells) + "</div></section>")

    # demo video
    demo = next(iter(sorted(results_root.glob("demo/*_h264.mp4"))), None)
    if demo:
        shutil.copyfile(demo, assets / "demo.mp4")
        poster = demo.with_name(demo.name.replace("_h264.mp4", "_poster.jpg"))
        poster_attr = ""
        if poster.exists():
            shutil.copyfile(poster, assets / "demo_poster.jpg")
            poster_attr = ' poster="assets/demo_poster.jpg"'
        s.append('<section id="demo"><h2>Streaming on a held-out road</h2><p class="lede">240 consecutive frames of '
                 'scene BR_S02 (never seen in training), run frame by frame through the streaming detector: left UFLD '
                 'baseline, right Lite v0.5 with cached history features (first pilot checkpoints). Red = ego-left '
                 'slot, yellow = ego-right slot, numbers = mean existence probability of each lane.</p>'
                 f'<video controls muted playsinline preload="metadata"{poster_attr} src="assets/demo.mp4"></video></section>')

    # latency + budgets
    bench_files = sorted(results_root.glob("benchmarks/*.json"))
    if bench_files:
        by_name = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in bench_files}
        desk = by_name.get("desktop") or next(iter(by_name.values()))
        rows = [{"label": f"{LABELS.get(r['measured']['variant'], r['measured']['variant'])} · {r['measured']['mode']}",
                 "pre": r["measured"]["preprocess_ms_mean"], "model": r["measured"]["model_ms_mean"],
                 "post": r["measured"]["postprocess_ms_mean"], "total": r["measured"]["total_ms_mean"]}
                for r in desk["results"]]
        budget_rows = []
        for name, rep in by_name.items():
            prof = rep["profile"]
            for r in rep["results"]:
                m, sim = r["measured"], r["simulated"]
                pill = '<span class="pill ok">met</span>' if sim["all_budgets_met"] else '<span class="pill no">missed</span>'
                budget_rows.append(f'<tr><td>{esc(name)}</td><td class="num">×{prof["slowdown"]}</td>'
                                   f'<td>{esc(LABELS.get(m["variant"], m["variant"]))}</td><td>{esc(m["mode"])}</td>'
                                   f'<td class="num">{sim["processed_fps"]:.1f}</td><td class="num">{sim["latency_p95_ms"]:.1f}</td>'
                                   f'<td class="num">{100 * sim["history_fallback_rate"]:.0f} %</td><td>{pill}</td></tr>')
        gpu = desk["results"][0]["measured"]["device"]
        s.append('<section id="latency"><h2>Latency and real-time budgets</h2><p class="lede">Measured at batch 1 on '
                 f'{esc(gpu)} with 640×480 frames of the held-out clip (first pilot checkpoints). "cached" reuses the '
                 'stored features of previous frames; "recompute" re-encodes all three frames every time. '
                 'Preprocessing (a PIL bilinear resize on the CPU, identical to training) is the largest cost.</p>'
                 f'<div class="panel">{latency_chart(rows)}<div class="legend"><span><i class="sw" style="background:var(--muted);opacity:.5"></i>preprocess</span>'
                 '<span><i class="sw" style="background:var(--accent)"></i>model</span><span><i class="sw" style="background:var(--ink);opacity:.7"></i>postprocess</span></div></div>'
                 '<p class="lede">Simulated 30 FPS camera: service times = measured × slowdown. The Jetson slowdowns are '
                 'assumptions from peak-throughput ratios, not measurements on a Jetson. "History fallbacks" are frames '
                 'whose history frames were skipped because the worker was busy.</p>'
                 '<div class="scroll"><table><thead><tr><th>Profile</th><th class="num">Slowdown</th><th>Model</th><th>Mode</th>'
                 '<th class="num">Processed FPS</th><th class="num">p95 latency ms</th><th class="num">History fallbacks</th>'
                 f'<th>Budgets</th></tr></thead><tbody>{"".join(budget_rows)}</tbody></table></div></section>')

    # deployment checks (any run that has them)
    dep_rows = []
    for p in sorted({*run.glob("deploy/*/numerical_checks.json"), *results_root.glob("elas_pilot/deploy/*/numerical_checks.json")}):
        c = json.loads(p.read_text(encoding="utf-8"))
        variant = p.parent.name.rsplit("_seed", 1)[0]
        acc = {e["backend"]: e["lane_f1_iou50"] for e in c.get("task_accuracy_val", [])}
        ref = acc.get("torch-fp32")
        for key, r in c.items():
            if not isinstance(r, dict) or key in ("frames",):
                continue
            backend = key.replace("_", " ")
            name = {"onnxruntime_fp32": "onnxruntime-cpu", "onnxruntime_int8": "onnxruntime-int8",
                    "tensorrt_fp32": "tensorrt-fp32", "tensorrt_fp16": "tensorrt-fp16", "tensorrt_int8": "tensorrt-int8"}.get(key)
            f1 = acc.get(name)
            f1_txt = "n/a" if f1 is None else f"{f1:.4f}" + ("" if ref is None else f" ({f1 - ref:+.4f})")
            if "pass" in r:
                pill = '<span class="pill ok">within tolerance</span>' if r["pass"] else '<span class="pill no">outside tolerance</span>'
                dep_rows.append(f'<tr><td>{esc(LABELS.get(variant, variant))}</td><td>{esc(backend)}</td>'
                                f'<td class="num">{r["max_abs_logits"]:.1e}</td><td class="num">{r["max_abs_exist"]:.1e}</td>'
                                f'<td class="num">{r["mean_lane_dx_px"]:.2f}</td><td class="num">{f1_txt}</td><td>{pill}</td></tr>')
            else:
                reason = r.get("failed") or r.get("skipped") or ""
                dep_rows.append(f'<tr><td>{esc(LABELS.get(variant, variant))}</td><td>{esc(backend)}</td>'
                                f'<td colspan="4">{esc(reason[:120])}</td><td><span class="pill na">not built</span></td></tr>')
    if dep_rows:
        s.append('<section id="export"><h2>Export and precision checks</h2><p class="lede">Each exported model runs '
                 'the same frames as PyTorch through the same streaming code (first pilot checkpoints). Val lane F1 '
                 'is measured on 240 validation frames with the exact training-time temporal context; the difference '
                 'to PyTorch FP32 is in brackets. TensorRT ran on the development GPU, not on a Jetson.</p>'
                 '<div class="scroll"><table><thead><tr><th>Model</th><th>Backend</th>'
                 '<th class="num">max |Δ logits|</th><th class="num">max |Δ exist|</th><th class="num">mean lane Δx px</th>'
                 '<th class="num">Val lane F1</th><th>Numerics</th></tr></thead><tbody>' + "".join(dep_rows)
                 + "</tbody></table></div></section>")

    # qualitative examples
    shots = []
    best_temporal = max((r for r in test_rows if VARIANTS[r["variant"]].temporal), key=lambda r: r["mean"], default=None)
    for v in [v for v in ("ufld_baseline", best_temporal["variant"] if best_temporal else None) if v]:
        for bucket in ("successes", "errors"):
            files = sorted((run / f"seed_{seeds[0]}" / "visuals" / "test" / v / bucket).glob("*.png"))
            if files:
                dst = assets / f"{v}_{bucket}.jpg"
                _copy_image(files[len(files) // 2], dst, 1200)
                shots.append(f'<figure><img src="assets/{dst.name}" alt="{esc(LABELS[v])} {bucket[:-1]} example" '
                             f'loading="lazy"><figcaption>{esc(LABELS[v])}, held-out test, a typical '
                             f'{"success" if bucket == "successes" else "error"}: original | ground truth (green) | '
                             f'prediction (red) - {esc(files[len(files) // 2].stem)}</figcaption></figure>')
    if shots:
        s.append('<section id="examples"><h2>Examples from the held-out scenes</h2><div class="shots">' + "".join(shots)
                 + "</div></section>")

    s.append(_roadmap_section(roadmap_path))

    # reproduce
    commands = [("python -m tac_ufld doctor", ""), ("python -m pytest -q", ""),
                ("python -m tac_ufld ablate --spec configs/ablations/augmentation.yaml --confirm", "overfitting remedies"),
                ("python -m tac_ufld run --config configs/elas_pilot_v2.yaml", "pilot with the v0.4 models"),
                ("python -m tac_ufld ablate --spec configs/ablations/history.yaml --confirm", "history length"),
                ("python -m tac_ufld run --config configs/elas.yaml --confirm", "full protocol"),
                ("python -m tac_ufld ui", "inspect checkpoints"),
                (f"python -m tac_ufld site --runs results/{run.name}", "rebuild this page")]
    width = max(len(c) for c, _ in commands) + 3
    block = "\n".join(esc(c.ljust(width) + (f"# {n}" if n else "")).rstrip() for c, n in commands)
    s.append(f'<section id="reproduce"><h2>Reproduce and continue</h2><pre>{block}</pre>'
             '<p class="lede">The full experiment differs from the pilots in seeds (6 instead of 2), epochs (up to 50 '
             'with patience 10) and hyper-parameter search (the same budget for every model). Only the full run can '
             'support claims.</p></section>')
    s.append(f'<footer>Generated {dt.date.today().isoformat()} from git {esc(commit)} by <code>python -m tac_ufld site</code>. '
             'Numbers come from the run folders; the findings, the roadmap and the model descriptions are written by '
             'hand.</footer>')

    body = f'<main class="wrap">{"".join(x for x in s if x)}</main>'
    page_title = "TAC-UFLD Pilot Results"
    head = f"<title>{page_title}</title>{FONTS}<style>{CSS}</style>"
    (out / "page.html").write_text(head + body, encoding="utf-8")
    (out / "index.html").write_text(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width, initial-scale=1, viewport-fit=cover">{head}</head><body>{body}</body></html>',
        encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return out / "index.html"
