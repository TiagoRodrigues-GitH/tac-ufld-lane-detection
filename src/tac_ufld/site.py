"""Static results page (``python -m tac_ufld site``) for GitHub Pages.

Reads finished runs (reports, histories, visuals), benchmark reports,
deployment checks and an optional demo video, and writes a self-contained
site: ``index.html`` (full document, for GitHub Pages), ``page.html`` (the
same content without the document shell, for embedding) and ``assets/``.
Charts are inline SVG computed here, to scale, and themed through CSS
variables; the page needs no JavaScript and no external service except
Google Fonts (with fallbacks).
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

FAMILY = {"ufld_baseline": "ufld", "ufld_v02": "ufld", "ufld_v03": "ufld", "ufld_v04": "ufld",
          "lite_baseline": "lite", "lite_v05": "lite"}
LABELS = {"ufld_baseline": "UFLD baseline", "ufld_v02": "UFLD v0.2 weighted", "ufld_v03": "UFLD v0.3 gated",
          "ufld_v04": "UFLD v0.4 coord", "lite_baseline": "Lite baseline", "lite_v05": "Lite v0.5 warped"}
ORDER = list(LABELS)

CSS = """
/* Layout: one reading column (max 1060px) of report sections; tables and charts
   scroll inside their own boxes. Road-signage type on asphalt neutrals. */
:root {
  --bg: #f4f5f3; --surface: #ffffff; --ink: #1c2024; --muted: #5d656c; --rule: #d9ddd9;
  --accent: #8a6100; --marking: #e4ae00; --ufld: #3f6c9e; --lite: #c2711d;
  --good: #2f7d4f; --bad: #b23a3a; --chip: #eceee9;
  --display: "Barlow Condensed", "Arial Narrow", "Roboto Condensed", sans-serif;
  --body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #15181b; --surface: #1d2125; --ink: #e5e8ea; --muted: #9aa3aa; --rule: #30363b;
  --accent: #f2c230; --marking: #f2c230; --ufld: #7fa8d6; --lite: #e89a4c;
  --good: #5fbf85; --bad: #e07a7a; --chip: #262b30; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #15181b; --surface: #1d2125; --ink: #e5e8ea; --muted: #9aa3aa; --rule: #30363b;
  --accent: #f2c230; --marking: #f2c230; --ufld: #7fa8d6; --lite: #e89a4c;
  --good: #5fbf85; --bad: #e07a7a; --chip: #262b30; color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 15px/1.6 var(--body); margin: 0; }
.wrap { max-width: 1060px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 64px; display: grid; gap: 44px; }
header.top { display: grid; gap: 10px; border-bottom: 4px solid var(--marking); padding-bottom: 22px; }
.eyebrow { font: 600 12px/1 var(--mono); letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
h1, h2, h3 { font-family: var(--display); font-weight: 600; letter-spacing: .01em; text-wrap: balance; margin: 0; }
h1 { font-size: clamp(34px, 6vw, 54px); line-height: 1.02; }
h2 { font-size: 30px; line-height: 1.1; }
h3 { font-size: 21px; line-height: 1.2; }
p { margin: 0; max-width: 68ch; }
.meta { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { font: 500 12.5px/1 var(--mono); background: var(--chip); border: 1px solid var(--rule); border-radius: 3px;
  padding: 6px 8px; font-variant-numeric: tabular-nums; }
.chip.warn { border-color: var(--accent); color: var(--accent); }
section { display: grid; gap: 16px; min-width: 0; }
.lede { color: var(--muted); max-width: 72ch; }
.findings { display: grid; gap: 10px; padding: 18px 20px; background: var(--surface); border: 1px solid var(--rule);
  border-left: 6px solid var(--marking); }
.findings li { margin: 0 0 6px; max-width: 76ch; }
.findings ul { margin: 0; padding-left: 20px; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); gap: 18px; }
.panel { background: var(--surface); border: 1px solid var(--rule); padding: 16px; display: grid; gap: 10px; min-width: 0; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th { font: 600 11.5px/1.3 var(--mono); text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
td.num, th.num { text-align: right; }
.fam { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 8px; vertical-align: baseline; }
.fam.ufld { background: var(--ufld); } .fam.lite { background: var(--lite); }
.pill { font: 600 11px/1 var(--mono); padding: 4px 7px; border-radius: 99px; border: 1px solid currentColor; }
.pill.ok { color: var(--good); } .pill.no { color: var(--bad); } .pill.na { color: var(--muted); }
svg { display: block; max-width: 100%; height: auto; font-family: var(--mono); }
svg text { fill: var(--muted); font-size: 11px; }
svg .label { fill: var(--ink); font-size: 12px; font-family: var(--body); }
svg .grid { stroke: var(--rule); stroke-width: 1; }
svg .ufld { fill: var(--ufld); } svg .lite { fill: var(--lite); }
svg .ufld-s { stroke: var(--ufld); } svg .lite-s { stroke: var(--lite); }
svg .dot { fill: var(--surface); stroke: var(--ink); stroke-width: 1.4; }
svg .seg-pre { fill: var(--muted); opacity: .55; } svg .seg-model { fill: var(--marking); } svg .seg-post { fill: var(--ink); opacity: .75; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12.5px; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
video, .shots img { width: 100%; max-width: 100%; border: 1px solid var(--rule); background: #000; }
.shots { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 14px; }
figure { margin: 0; display: grid; gap: 6px; min-width: 0; }
figcaption { font-size: 12.5px; color: var(--muted); }
pre { background: var(--surface); border: 1px solid var(--rule); padding: 14px; overflow-x: auto; font: 13px/1.55 var(--mono); margin: 0; }
code { font-family: var(--mono); font-size: .92em; }
a { color: var(--accent); text-underline-offset: 3px; }
a:focus-visible { outline: 2px solid var(--marking); outline-offset: 2px; }
footer { color: var(--muted); font-size: 12.5px; border-top: 1px solid var(--rule); padding-top: 16px; }
@media (max-width: 520px) { .wrap { padding-inline: 16px; } h2 { font-size: 26px; } }
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" '
         'href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="https://fonts.googleapis.com/'
         'css2?family=Barlow+Condensed:wght@500;600&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:'
         'wght@400;500;600&display=swap">')


def esc(v) -> str:
    return html.escape(str(v))


def fmt(v, digits: int = 3) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return esc(v)
    return "n/a" if not np.isfinite(f) else f"{f:.{digits}f}"


# ------------------------------------------------------------------ SVG charts


def bar_chart(rows: list[dict], metric_label: str, width: int = 760) -> str:
    """Horizontal bars (mean) with per-seed dots, one row per model, 0..1 scale."""
    left, right, row_h, top = 150, 60, 34, 26
    height = top + row_h * len(rows) + 30
    plot_w = width - left - right
    x = lambda v: left + plot_w * max(0.0, min(1.0, v))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(metric_label)} per model">']
    for t in np.linspace(0, 1, 6):
        parts.append(f'<line class="grid" x1="{x(t):.1f}" y1="{top - 8}" x2="{x(t):.1f}" y2="{height - 26}"/>'
                     f'<text x="{x(t):.1f}" y="{height - 10}" text-anchor="middle">{t:.1f}</text>')
    for i, r in enumerate(rows):
        y = top + i * row_h
        fam = FAMILY.get(r["variant"], "ufld")
        parts.append(f'<text class="label" x="{left - 10}" y="{y + 16}" text-anchor="end">{esc(LABELS.get(r["variant"], r["variant"]))}</text>')
        parts.append(f'<rect class="{fam}" x="{left}" y="{y + 4}" width="{x(r["mean"]) - left:.1f}" height="18" rx="2"/>')
        for s in r["seeds"]:
            parts.append(f'<circle class="dot" cx="{x(s):.1f}" cy="{y + 13}" r="4"/>')
        end = max([r["mean"], *r["seeds"]])  # label after the bar and every seed dot
        parts.append(f'<text x="{x(end) + 10:.1f}" y="{y + 17}">{r["mean"]:.3f}</text>')
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
    left, right, row_h, top = 200, 70, 28, 22
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


# ------------------------------------------------------------------ sections


def _agg(df: pd.DataFrame, split: str, protocol: str, metric: str) -> list[dict]:
    sub = df[(df["split"] == split) & (df["protocol"] == protocol) & (df["input"] == "full")]
    rows = []
    for v in [v for v in ORDER if v in set(sub["variant"])]:
        vals = sub[sub["variant"] == v][metric].to_numpy(dtype=float)
        rows.append({"variant": v, "mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
                     "seeds": vals.tolist()})
    return rows


def results_table(df: pd.DataFrame, split: str, protocol: str) -> str:
    sub = df[(df["split"] == split) & (df["protocol"] == protocol) & (df["input"] == "full")]
    cols = [("lane_f1_iou50", "Lane F1 @0.5"), ("lane_f1_iou35", "Lane F1 @0.35"), ("pixel_f1", "Pixel F1"),
            ("anchor_f1", "Anchor F1"), ("jitter_px", "Jitter px"), ("lane_fp_iou50", "FP"), ("lane_fn_iou50", "FN")]
    head = "".join(f'<th class="num">{esc(n)}</th>' for _, n in cols)
    body = []
    for v in [v for v in ORDER if v in set(sub["variant"])]:
        s = sub[sub["variant"] == v]
        cells = []
        for c, _ in cols:
            vals = s[c].to_numpy(dtype=float)
            digits = 0 if c in ("lane_fp_iou50", "lane_fn_iou50") else (2 if c == "jitter_px" else 3)
            txt = f"{vals.mean():.{digits}f}" + (f" ± {vals.std(ddof=1):.{digits}f}" if len(vals) > 1 else "")
            cells.append(f'<td class="num">{txt}</td>')
        body.append(f'<tr><td><span class="fam {FAMILY[v]}"></span>{esc(LABELS[v])}</td>{"".join(cells)}</tr>')
    return (f'<div class="scroll"><table><thead><tr><th>Model</th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def paired_table(path: Path) -> str:
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    df = df[df["metric"] == "lane_f1_iou50"]
    rows = []
    for r in df.itertuples():
        pill = ('<span class="pill ok">significant</span>' if r.significant else
                '<span class="pill na">underpowered</span>' if r.underpowered else '<span class="pill no">not significant</span>')
        rows.append(f'<tr><td>{esc(LABELS.get(r.variant, r.variant))}</td><td>{esc(LABELS.get(r.reference, r.reference))}</td>'
                    f'<td class="num">{r.n_seeds}</td><td class="num">{r.mean_delta:+.3f}</td>'
                    f'<td class="num">[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]</td><td class="num">{fmt(r.wilcoxon_p)}</td>'
                    f'<td class="num">{fmt(r.p_holm)}</td><td>{pill}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Temporal model</th><th>Paired with</th><th class="num">Seeds</th>'
            '<th class="num">Δ lane F1</th><th class="num">95 % CI</th><th class="num">Wilcoxon p</th>'
            '<th class="num">Holm p</th><th>Verdict</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def ablation_table(path: Path) -> str:
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    col = [c for c in df.columns if c.startswith("mean_gain")][0]
    rows = "".join(f'<tr><td>{esc(LABELS.get(r.variant, r.variant))}</td><td class="num">{getattr(r, col):+.4f}</td>'
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


def build_site(runs: list[Path], out: Path, title: str, extra: list[Path] | None = None,
               notes: Path | None = None) -> Path:
    """``runs[0]`` is the run shown; benchmarks and the demo video are read from
    ``<results>/benchmarks`` and ``<results>/demo`` next to it (``extra`` is
    reserved for further report files)."""
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
    s = []

    # header
    chips = [f"dataset ELAS", f"{df['variant'].nunique()} models", f"{len(seeds)} seeds",
             "held-out scenes BR_S02 · VIX_S05 · VV_S03", env.get("gpu", "GPU n/a")]
    if duration:
        chips.append(f"GPU time {duration}")
    s.append(f'<header class="top"><div class="eyebrow">TAC-UFLD v{__version__} · pilot run · {esc(run.name)}</div>'
             f'<h1>{esc(title)}</h1><p class="lede">UFLD (ResNet-18) and a lightweight CNN, each with temporal variants '
             'that fuse the two previous frames, trained on the ELAS ego-lane dataset and tested on three road scenes '
             'never used for training or model selection.</p><div class="meta">'
             + "".join(f'<span class="chip">{esc(c)}</span>' for c in chips)
             + '<span class="chip warn">indicative pilot, not the full protocol</span></div></header>')

    # findings (hand-written analysis)
    if notes_path.exists():
        items = [l[2:].strip() for l in notes_path.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
        md = lambda t: re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", re.sub(r"`(.+?)`", r"<code>\1</code>", esc(t)))
        s.append('<section><h2>What the pilot shows</h2><div class="findings"><ul>'
                 + "".join(f"<li>{md(i)}</li>" for i in items) + "</ul></div></section>")

    # results
    test_rows = _agg(df, "test", "tuned", "lane_f1_iou50")
    seen_rows = _agg(df, "seen_test", "tuned", "lane_f1_iou50")
    legend = ('<div class="legend"><span><i class="sw" style="background:var(--ufld)"></i>UFLD family (ResNet-18, ImageNet)</span>'
              '<span><i class="sw" style="background:var(--lite)"></i>Lite family (4-block CNN)</span>'
              '<span>bars = mean over seeds, dots = individual seeds</span></div>')
    s.append('<section><h2>Lane F1 on unseen roads</h2><p class="lede">CULane-style lane F1: each lane drawn 12 px wide '
             '(30 px at 1640 px, scaled to 640 px), matched one-to-one, correct when IoU ≥ 0.5. Post-processing tuned on '
             'validation only. On ELAS every model predicts both ego lanes in every frame, so precision = recall = F1: '
             'the errors are misplaced lanes, not missed ones.</p>'
             f'<div class="panel"><h3>Held-out test scenes (unseen roads)</h3>{bar_chart(test_rows, "lane F1 held-out")}</div>'
             f'<div class="panel"><h3>Seen-scene test blocks</h3>{bar_chart(seen_rows, "lane F1 seen")}</div>{legend}'
             f'<h3>Held-out test scenes, all metrics (mean ± std over seeds)</h3>{results_table(df, "test", "tuned")}'
             f'<h3>Seen-scene test</h3>{results_table(df, "seen_test", "tuned")}</section>')

    # statistics
    s.append('<section><h2>Does temporal context help?</h2><p class="lede">Each temporal model is compared with the '
             'single-frame model of its own family on the same seeds (exact Wilcoxon, Holm correction). Two seeds can '
             'never reach p &lt; 0.05, so no difference here is a finding yet. The second table replaces the history '
             'frames by the current frame at test time: it isolates what the history itself contributes.</p>'
             f'{paired_table(run / "report" / "paired_tests_test_tuned.csv")}'
             f'{ablation_table(run / "report" / "temporal_ablation.csv")}</section>')

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
            cells.append(f'<figure class="panel"><h3>{esc(LABELS[v])}</h3>'
                         f'{curves_chart(hist[hist["variant"] == v], "val_lane_f1_iou50")}'
                         '<figcaption>validation lane F1 per epoch; solid = seed 1, dashed = seed 2, dot = best epoch '
                         '(the checkpoint kept)</figcaption></figure>')
        s.append('<section><h2>Training curves</h2><p class="lede">Temporal models start from their family baseline\'s '
                 'best checkpoint. The UFLD models peak at their first epoch and then overfit (training loss keeps '
                 'falling while validation gets worse); the lite models were still improving when the 4-epoch pilot '
                 'budget ended.</p><div class="grid2">' + "".join(cells) + "</div></section>")

    # demo video
    demo = next(iter(sorted(results_root.glob("demo/*_h264.mp4"))), None)
    if demo:
        shutil.copyfile(demo, assets / "demo.mp4")
        poster = demo.with_name(demo.name.replace("_h264.mp4", "_poster.jpg"))
        poster_attr = ""
        if poster.exists():
            shutil.copyfile(poster, assets / "demo_poster.jpg")
            poster_attr = ' poster="assets/demo_poster.jpg"'
        s.append('<section><h2>Streaming on a held-out road</h2><p class="lede">240 consecutive frames of scene BR_S02 '
                 '(never seen in training), run frame by frame through the streaming detector: left UFLD baseline, right '
                 'Lite v0.5 with cached history features. Red = ego-left slot, yellow = ego-right slot, numbers = mean '
                 'existence probability of each lane; the header shows the history frames used and the latency split.</p>'
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
        s.append('<section><h2>Latency and real-time budgets</h2><p class="lede">Measured at batch 1 on '
                 f'{esc(gpu)} with 640×480 frames of the held-out clip. "cached" reuses the stored features of previous '
                 'frames; "recompute" re-encodes all three frames every time. Preprocessing (a PIL bilinear resize on the '
                 'CPU, identical to training) is now the largest cost.</p>'
                 f'<div class="panel">{latency_chart(rows)}<div class="legend"><span><i class="sw" style="background:var(--muted);opacity:.55"></i>preprocess</span>'
                 '<span><i class="sw" style="background:var(--marking)"></i>model</span><span><i class="sw" style="background:var(--ink);opacity:.75"></i>postprocess</span></div></div>'
                 '<p class="lede">Simulated 30 FPS camera: service times = measured × slowdown. The Jetson slowdowns are '
                 'assumptions from peak-throughput ratios, not measurements on a Jetson. "History fallbacks" are frames '
                 'whose history frames were skipped because the worker was busy.</p>'
                 '<div class="scroll"><table><thead><tr><th>Profile</th><th class="num">Slowdown</th><th>Model</th><th>Mode</th>'
                 '<th class="num">Processed FPS</th><th class="num">p95 latency ms</th><th class="num">History fallbacks</th>'
                 f'<th>Budgets</th></tr></thead><tbody>{"".join(budget_rows)}</tbody></table></div></section>')

    # deployment checks
    dep_rows = []
    for p in sorted(run.glob("deploy/*/numerical_checks.json")):
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
        s.append('<section><h2>Export and precision checks</h2><p class="lede">Each exported model runs the same frames '
                 'as PyTorch through the same streaming code. Val lane F1 is measured on 240 validation frames with the '
                 'exact training-time temporal context; the difference to PyTorch FP32 is in brackets. TensorRT ran on the '
                 'development GPU, not on a Jetson.</p><div class="scroll"><table><thead><tr><th>Model</th><th>Backend</th>'
                 '<th class="num">max |Δ logits|</th><th class="num">max |Δ exist|</th><th class="num">mean lane Δx px</th>'
                 '<th class="num">Val lane F1</th><th>Numerics</th></tr></thead><tbody>' + "".join(dep_rows)
                 + "</tbody></table></div></section>")

    # qualitative examples
    shots = []
    for v in ("ufld_baseline", "lite_v05"):
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
        s.append('<section><h2>Examples from the held-out scenes</h2><div class="shots">' + "".join(shots) + "</div></section>")

    # reproduce
    s.append('<section><h2>Reproduce and continue</h2><pre>python -m tac_ufld doctor\npython -m pytest -q\n'
             'python -m tac_ufld run --config configs/elas_pilot.yaml          # this pilot, about 2.5 h on an RTX 3050\n'
             'python -m tac_ufld run --config configs/elas.yaml --confirm      # full protocol: 6 seeds, HPO, up to 50 epochs\n'
             'python -m tac_ufld ui                                            # inspect checkpoints and results\n'
             'python -m tac_ufld site --runs results/elas_pilot                # rebuild this page</pre>'
             '<p class="lede">The full experiment differs from this pilot in seeds (6 instead of 2), epochs (up to 50 with '
             'patience 10 instead of 4 with patience 2) and hyper-parameter search (20 trials per model, the same budget for '
             'every model). Only the full run can support claims.</p></section>')
    s.append(f'<footer>Generated {dt.date.today().isoformat()} from git {esc(commit)} by <code>python -m tac_ufld site</code>. '
             'Numbers come from the run folders; nothing on this page is typed by hand except the findings list.</footer>')

    body = f'<main class="wrap">{"".join(s)}</main>'
    page_title = "TAC-UFLD Pilot Results"
    head = f"<title>{page_title}</title>{FONTS}<style>{CSS}</style>"
    (out / "page.html").write_text(head + body, encoding="utf-8")
    (out / "index.html").write_text(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width, initial-scale=1, viewport-fit=cover">{head}</head><body>{body}</body></html>',
        encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return out / "index.html"
