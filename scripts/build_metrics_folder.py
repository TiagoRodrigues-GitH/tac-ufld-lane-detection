"""Collect the pilot metrics and draw the main charts into metrics_and_graphics/.

    python scripts/build_metrics_folder.py            # after the runs listed below exist

Everything is read from results/ (nothing is typed by hand), so the folder can
be rebuilt after the full run. Charts are drawn in Portuguese (graphics/pt)
and English (graphics/en), as PNG and SVG. The results-page PDFs in the same
folder are printed separately (see metrics_and_graphics/README.md).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "metrics_and_graphics"
MAIN, LITE = RES / "elas_pilot_v2", RES / "elas_lite_long"
BLUE, TEAL, RED, GREY = "#2f7ec9", "#1f9e9a", "#c0392b", "#586779"
LABEL = {"ufld_baseline": "UFLD baseline", "ufld_baseline_ct": "UFLD baseline +CT", "ufld_v02": "UFLD v0.2 weighted",
         "ufld_v03": "UFLD v0.3 gated", "ufld_v04": "UFLD v0.4 coord", "ufld_v06": "UFLD v0.6 aligned",
         "ufld_v07": "UFLD v0.7 ConvGRU", "lite_baseline": "Lite baseline", "lite_baseline_ct": "Lite baseline +CT",
         "lite_v05": "Lite v0.5 warped", "lite_v05_static": "Lite v0.5 static", "lite_v06": "Lite v0.6 ConvGRU"}
ORDER = list(LABEL)
ARMS = {"photometric": ("original (fotométrico)", "original (photometric)"), "none": ("nenhum", "none"),
        "geometric": ("geométrico", "geometric"), "geometric_flip": ("geométrico + espelhamento", "geometric + flip"),
        "photometric_extended": ("fotométrico estendido", "extended photometric"),
        "regularised": ("dropout + label smoothing + WD", "dropout + label smoothing + WD"),
        "backbone_lr_0p1": ("lr do backbone ×0,1", "backbone lr ×0.1"),
        "freeze_stem_layer1": ("stem + layer1 congelados", "frozen stem + layer1"),
        "current_frame_degradation": ("degradação do quadro atual", "current-frame degradation"),
        "small_head": ("cabeça pequena (256)", "small head (256)"), "more_scenes": ("mais cenas (18)", "more scenes (18)")}
TEXT = {
    "pt": {"f1": "Lane F1 no teste (cenas nunca vistas)", "aug": "1. Overfitting da UFLD: o que resolveu",
           "aug_note": "UFLD baseline, 3 sementes, 4 épocas; barras = média, pontos = sementes",
           "clean": "2. Lane F1 no teste com quadros limpos",
           "clean_note": "Piloto 2 (UFLD, 4 épocas) e execução lite longa (16 épocas); 2 sementes",
           "occ": "3. Quadro atual ocluído, histórico limpo", "clean_lbl": "quadro atual limpo",
           "occ_lbl": "quadro atual ocluído", "ops": "4. Lane F1 por tipo de degradação do quadro atual",
           "op_names": {"occlude": "oclusão", "blur": "desfoque", "darken": "escurecimento", "noise": "ruído"},
           "size": "5. Acurácia versus número de parâmetros", "params": "Parâmetros (milhões, escala log)",
           "hist": "6. Até onde olhar para trás (UFLD v0.4 − baseline)", "hist_x": "Ganho de lane F1 sobre a baseline",
           "hist_lbl": "{t} quadros, passo {s}", "kal": "7. Jitter com e sem o filtro de Kalman na saída",
           "jit": "Jitter médio (px, menor = mais estável)", "without": "sem Kalman", "with": "com Kalman",
           "lat": "8. Latência por quadro na RTX 3050 (lote 1)", "ms": "Tempo por quadro (ms)",
           "pre": "pré-processamento", "model": "modelo", "post": "pós-processamento",
           "modes": {"single": "quadro único", "cached": "em cache", "recompute": "recalculado"}},
    "en": {"f1": "Held-out lane F1 (unseen scenes)", "aug": "1. UFLD overfitting: what fixed it",
           "aug_note": "UFLD baseline, 3 seeds, 4 epochs; bars = mean, dots = seeds",
           "clean": "2. Held-out lane F1 on clean frames",
           "clean_note": "Pilot 2 (UFLD, 4 epochs) and the longer lite run (16 epochs); 2 seeds",
           "occ": "3. Occluded current frame, clean history", "clean_lbl": "clean current frame",
           "occ_lbl": "occluded current frame", "ops": "4. Lane F1 per current-frame degradation",
           "op_names": {"occlude": "occlusion", "blur": "blur", "darken": "darkening", "noise": "noise"},
           "size": "5. Accuracy against parameter count", "params": "Parameters (millions, log scale)",
           "hist": "6. How far back to look (UFLD v0.4 − baseline)", "hist_x": "Lane F1 gain over the baseline",
           "hist_lbl": "{t} frames, step {s}", "kal": "7. Jitter with and without the output Kalman filter",
           "jit": "Mean jitter (px, lower = steadier)", "without": "without Kalman", "with": "with Kalman",
           "lat": "8. Per-frame latency on the RTX 3050 (batch 1)", "ms": "Time per frame (ms)",
           "pre": "preprocessing", "model": "model", "post": "postprocessing",
           "modes": {"single": "single frame", "cached": "cached", "recompute": "recompute"}},
}


def test_rows(run: Path) -> pd.DataFrame:
    df = pd.read_csv(run / "all_results.csv")
    return df[(df.split == "test") & (df.protocol == "tuned") & (df.input == "full")]


def family_color(v: str) -> str:
    return TEAL if v.startswith("lite") else BLUE


def save(fig, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=180)
    fig.savefig(folder / f"{name}.svg")
    plt.close(fig)


def dec(lang: str, x: float, digits: int = 3) -> str:
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if lang == "pt" else s


def copy_metrics() -> None:
    m = OUT / "metrics"
    m.mkdir(parents=True, exist_ok=True)
    files = {
        "pilot2_heldout_mean_std.csv": MAIN / "report/aggregate_test_tuned.csv",
        "pilot2_seen_scenes_mean_std.csv": MAIN / "report/aggregate_seen_test_tuned.csv",
        "pilot2_all_results_per_seed.csv": MAIN / "all_results.csv",
        "pilot2_paired_tests.csv": MAIN / "report/paired_tests_test_tuned.csv",
        "pilot2_robustness_summary.csv": MAIN / "report/robustness_summary.csv",
        "pilot2_robustness_per_seed.csv": MAIN / "report/robustness.csv",
        "pilot2_kalman.csv": MAIN / "report/kalman_reference.csv",
        "pilot2_conditions.csv": MAIN / "report/test_conditions.csv",
        "pilot2_history_removal_at_test.csv": MAIN / "report/temporal_ablation.csv",
        "pilot2_carried_state.csv": MAIN / "report/carry_state.csv",
        "pilot2_efficiency.csv": MAIN / "report/efficiency.csv",
        "lite_long_heldout_mean_std.csv": LITE / "report/aggregate_test_tuned.csv",
        "lite_long_all_results_per_seed.csv": LITE / "all_results.csv",
        "lite_long_paired_tests.csv": LITE / "report/paired_tests_test_tuned.csv",
        "lite_long_robustness_summary.csv": LITE / "report/robustness_summary.csv",
        "lite_long_robustness_per_seed.csv": LITE / "report/robustness.csv",
        "lite_long_kalman.csv": LITE / "report/kalman_reference.csv",
        "lite_long_efficiency.csv": LITE / "report/efficiency.csv",
        "ablation_augmentation_results.csv": RES / "ablation_augmentation/ablation_results.csv",
        "ablation_augmentation_validation.csv": RES / "ablation_augmentation/ablation_validation.csv",
        "ablation_augmentation_paired_tests.csv": RES / "ablation_augmentation/ablation_paired_tests.csv",
        "ablation_history_results.csv": RES / "ablation_history/ablation_results.csv",
        "elas_split_frames_per_scene.csv": MAIN / "data/split_summary.csv",
    }
    sheets = {}
    for name, src in files.items():
        if src.exists():
            shutil.copy2(src, m / name)
            sheets[name[:-4][:31]] = pd.read_csv(src)
    bench = RES / "benchmarks/desktop.json"
    if bench.exists():
        rows = [r["measured"] for r in json.loads(bench.read_text(encoding="utf-8"))["results"]]
        lat = pd.DataFrame(rows)
        lat.to_csv(m / "latency_desktop_rtx3050.csv", index=False)
        sheets["latency_desktop_rtx3050"] = lat
    with pd.ExcelWriter(m / "all_metrics.xlsx") as xw:
        for name, frame in sheets.items():
            frame.to_excel(xw, sheet_name=name, index=False)
    print(f"{len(sheets)} tables -> {m}")


def charts(lang: str) -> None:
    T = TEXT[lang]
    g = OUT / "graphics" / lang
    main, lite = test_rows(MAIN), test_rows(LITE)
    both = pd.concat([main[~main.variant.str.startswith("lite")], lite])
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # 1 augmentation ablation
    aug = pd.read_csv(RES / "ablation_augmentation/ablation_results.csv")
    aug = aug[aug.split == "test"]
    arms = aug.groupby("arm").lane_f1_iou50.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, arm in enumerate(arms):
        v = aug[aug.arm == arm].lane_f1_iou50.to_numpy()
        ax.barh(i, v.mean(), color=BLUE if arm == "geometric" else "#9dbfe3")
        ax.scatter(v, [i] * len(v), s=14, color="white", edgecolor="#15202b", zorder=3)
        ax.text(max(v.max(), v.mean()) + 0.01, i, dec(lang, v.mean()), va="center", fontsize=8)
    ax.set_yticks(range(len(arms)), [ARMS.get(a, (a, a))[0 if lang == "pt" else 1] for a in arms])
    ax.set_xlim(0, 1)
    ax.set_xlabel(T["f1"])
    ax.set_title(T["aug"], loc="left", fontweight="bold")
    fig.text(0.01, 0.01, T["aug_note"], fontsize=7, color=GREY)
    save(fig, g, "01_augmentation_ablation")

    # 2 clean held-out F1
    models = [v for v in ORDER if v in set(both.variant)]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for i, v in enumerate(models[::-1]):
        s = both[both.variant == v].lane_f1_iou50.to_numpy()
        ax.barh(i, s.mean(), color=family_color(v))
        ax.scatter(s, [i] * len(s), s=14, color="white", edgecolor="#15202b", zorder=3)
        ax.text(max(s.max(), s.mean()) + 0.01, i, dec(lang, s.mean()), va="center", fontsize=8)
    ax.set_yticks(range(len(models)), [LABEL[v] for v in models[::-1]])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(T["f1"])
    ax.set_title(T["clean"], loc="left", fontweight="bold")
    fig.text(0.01, 0.01, T["clean_note"], fontsize=7, color=GREY)
    save(fig, g, "02_clean_heldout_f1")

    # 3 occlusion dumbbell and 4 per-op
    rob = pd.concat([
        pd.read_csv(MAIN / "report/robustness_summary.csv").query("~variant.str.startswith('lite')", engine="python"),
        pd.read_csv(LITE / "report/robustness_summary.csv")])
    occ = rob[(rob.input == "degraded") & (rob.op == "occlude")].set_index("variant")
    models = [v for v in ORDER if v in occ.index and v != "lite_baseline_ct"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for i, v in enumerate(models[::-1]):
        a, b = occ.loc[v, "clean_lane_f1"], occ.loc[v, "lane_f1"]
        ax.plot([a, b], [i, i], color=RED, alpha=0.5, lw=2)
        ax.scatter([a], [i], s=36, facecolor="white", edgecolor="#15202b", zorder=3,
                   label=T["clean_lbl"] if i == 0 else None)
        ax.scatter([b], [i], s=36, color=family_color(v), zorder=3, label=T["occ_lbl"] if i == 0 else None)
        ax.text(max(a, b) + 0.015, i, f"{dec(lang, b)} ({'+' if b - a >= 0 else ''}{dec(lang, b - a)})",
                va="center", fontsize=8)
    ax.set_yticks(range(len(models)), [LABEL[v] for v in models[::-1]])
    ax.set_xlim(0.1, 1.08)
    ax.set_xlabel(T["f1"])
    ax.set_title(T["occ"], loc="left", fontweight="bold")
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    save(fig, g, "03_occluded_current_frame")

    deg = rob[(rob.input == "degraded") & (rob.op.isin(["occlude", "blur", "darken", "noise"]))]
    piv = deg.pivot_table(index="variant", columns="op", values="lane_f1")
    ufld = [v for v in ORDER if v in piv.index and v.startswith("ufld")]
    ops = ["occlude", "blur", "darken", "noise"]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    width = 0.8 / len(ops)
    colors = [RED, "#8e9bab", "#5b6b7c", "#b7c2ce"]
    for j, op in enumerate(ops):
        ax.bar(np.arange(len(ufld)) + j * width, piv.loc[ufld, op], width, color=colors[j], label=T["op_names"][op])
    ax.set_xticks(np.arange(len(ufld)) + 0.3, [LABEL[v].replace("UFLD ", "") for v in ufld], rotation=20, ha="right")
    ax.set_ylim(0.6, 0.95)
    ax.set_ylabel(T["f1"])
    ax.set_title(T["ops"] + " (UFLD)", loc="left", fontweight="bold")
    ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper left")
    save(fig, g, "04_degradation_per_type")

    # 5 accuracy vs parameters
    eff = pd.concat([pd.read_csv(MAIN / "report/efficiency.csv"), pd.read_csv(LITE / "report/efficiency.csv")])
    eff = eff.drop_duplicates("variant", keep="last").set_index("variant")
    f1 = both.groupby("variant").lane_f1_iou50.mean()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    shown = [v for v in ORDER if v in f1.index and v in eff.index and not v.endswith("_ct")]
    for v in shown:
        ax.scatter(eff.loc[v, "params_millions"], f1[v], s=40, color=family_color(v), zorder=3)
    for v in [v for v in shown if v.startswith("lite")]:
        ax.annotate(LABEL[v], (eff.loc[v, "params_millions"], f1[v]), textcoords="offset points", xytext=(6, -3),
                    fontsize=7)
    # the UFLD models sit in one cluster: labels stacked to the right with leader lines
    ufld = sorted([v for v in shown if v.startswith("ufld")], key=lambda v: -f1[v])
    for i, v in enumerate(ufld):
        ax.annotate(f"{LABEL[v]}  {dec(lang, f1[v])}", (eff.loc[v, "params_millions"], f1[v]),
                    xytext=(34, 0.78 - i * 0.055), fontsize=7, va="center", ha="left",
                    arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.6, "relpos": (0, 0.5)})
    ax.set_xscale("log")
    ax.set_xlim(1.8, 120)
    ax.set_xticks([2, 3, 5, 10, 20, 50, 100], ["2", "3", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel(T["params"])
    ax.set_ylabel(T["f1"])
    ax.set_title(T["size"], loc="left", fontweight="bold")
    save(fig, g, "05_accuracy_vs_parameters")

    # 6 history length
    hist = pd.read_csv(RES / "ablation_history/ablation_results.csv")
    hist = hist[hist.split == "test"]
    rows = []
    for arm, sub in hist.groupby("arm"):
        s, t = (int(x) for x in arm[1:].split("_t"))
        a = sub[sub.variant == "ufld_v04"].set_index("seed").lane_f1_iou50
        b = sub[sub.variant == "ufld_baseline"].set_index("seed").lane_f1_iou50
        d = (a - b).dropna()
        rows.append((t, s, d.mean(), d.to_numpy()))
    rows.sort()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, (t, s, m, d) in enumerate(rows[::-1]):
        ax.barh(i, m, color=BLUE if m > 0 else RED)
        ax.scatter(d, [i] * len(d), s=14, color="white", edgecolor="#15202b", zorder=3)
        ax.text(max(m, d.max()) + 0.002, i, ("+" if m >= 0 else "") + dec(lang, m), va="center", fontsize=8)
    ax.axvline(0, color="#15202b", lw=0.8)
    ax.set_yticks(range(len(rows)), [T["hist_lbl"].format(t=t, s=s) for t, s, _, _ in rows[::-1]])
    ax.set_xlabel(T["hist_x"])
    ax.set_title(T["hist"], loc="left", fontweight="bold")
    save(fig, g, "06_history_length")

    # 7 Kalman jitter
    kal = pd.concat([pd.read_csv(MAIN / "report/kalman_reference.csv"), pd.read_csv(LITE / "report/kalman_reference.csv")])
    k = kal[kal.metric == "jitter_px"].drop_duplicates("variant", keep="last").set_index("variant")["mean"]
    models = [v for v in ORDER if v in k.index and v + "+kf" in k.index and not v.endswith("_ct")]
    fig, ax = plt.subplots(figsize=(7, 4.0))
    y = np.arange(len(models))
    ax.barh(y + 0.2, [k[v] for v in models], 0.4, color="#9dbfe3", label=T["without"])
    ax.barh(y - 0.2, [k[v + "+kf"] for v in models], 0.4, color=BLUE, label=T["with"])
    ax.set_yticks(y, [LABEL[v] for v in models])
    ax.invert_yaxis()
    ax.set_xlabel(T["jit"])
    ax.set_title(T["kal"], loc="left", fontweight="bold")
    ax.legend(fontsize=7, frameon=False)
    save(fig, g, "07_kalman_jitter")

    # 8 latency
    bench = RES / "benchmarks/desktop.json"
    if bench.exists():
        rows = [r["measured"] for r in json.loads(bench.read_text(encoding="utf-8"))["results"]]
        fig, ax = plt.subplots(figsize=(7, 4.0))
        labels = [f"{LABEL.get(r['variant'], r['variant'])} · {T['modes'].get(r['mode'], r['mode'])}" for r in rows]
        pre = [r["preprocess_ms_mean"] for r in rows]
        mod = [r["model_ms_mean"] for r in rows]
        post = [r["postprocess_ms_mean"] for r in rows]
        yy = np.arange(len(rows))
        ax.barh(yy, pre, color="#b7c2ce", label=T["pre"])
        ax.barh(yy, mod, left=pre, color=BLUE, label=T["model"])
        ax.barh(yy, post, left=np.add(pre, mod), color="#15202b", label=T["post"])
        ax.set_yticks(yy, labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(T["ms"])
        ax.set_title(T["lat"], loc="left", fontweight="bold")
        ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower right")
        save(fig, g, "08_latency_rtx3050")
    print(f"charts -> {g}")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    copy_metrics()
    for lang in ("pt", "en"):
        charts(lang)
    figs = OUT / "graphics" / "report_figures"
    figs.mkdir(parents=True, exist_ok=True)
    for run in (MAIN, LITE):
        for png in (run / "report").glob("*.png"):
            shutil.copy2(png, figs / f"{run.name}_{png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
