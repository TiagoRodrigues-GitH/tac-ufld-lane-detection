"""TAC-UFLD Streamlit interface.

    python -m tac_ufld ui                      # or: streamlit run src/tac_ufld/ui/app.py -- --results results

Inference and inspection of finished runs only: the interface loads
checkpoints and result folders, it never imports the training code and
never starts training.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from tac_ufld.ui import core


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.environ.get("TAC_UFLD_RESULTS", "results"))
    known, _ = p.parse_known_args(sys.argv[1:])
    return known


st.set_page_config(page_title="TAC-UFLD lane detection", layout="wide")
ARGS = _args()


@st.cache_resource(show_spinner="Loading model...")
def _load(path: str, device: str):
    return core.load_model(path, device)


def _png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


# ------------------------------------------------------------------ sidebar

st.sidebar.title("TAC-UFLD")
st.sidebar.caption("Lane detection: UFLD baseline vs temporal variants. Inference and results only - "
                   "this interface never trains.")
results_root = st.sidebar.text_input("Results folder", ARGS.results)
devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
device = st.sidebar.selectbox("Device", devices)
ckpts = core.list_checkpoints(results_root)
selected: list = []
if ckpts.empty:
    st.sidebar.warning(f"No checkpoints under `{results_root}` (expected <run>/seed_<n>/checkpoints/*.pt).")
else:
    labels = {core.checkpoint_label(r): str(r["path"]) for _, r in ckpts.iterrows()}
    default = list(labels)[:2]
    picked = st.sidebar.multiselect("Models to compare", list(labels), default=default)
    selected = [labels[p] for p in picked]
extra = st.sidebar.text_input("Or a checkpoint path")
if extra.strip():
    selected.append(extra.strip())

models = []
for path in selected:
    try:
        models.append(_load(path, device))
    except Exception as exc:  # show, do not crash the whole page
        st.sidebar.error(f"{Path(path).name}: {exc}")

for warning in core.convention_warnings(models):
    st.warning("Models use different conventions - outputs are not directly comparable. " + warning)

tabs = st.tabs(["Image", "Video", "Models", "Training curves", "Experiment metrics", "Latency"])

# -------------------------------------------------------------------- image

with tabs[0]:
    st.subheader("Lane detection on one image")
    upload = st.file_uploader("Image", type=["png", "jpg", "jpeg", "bmp"], key="img")
    if upload is not None and models:
        rgb = np.asarray(Image.open(io.BytesIO(upload.getvalue())).convert("RGB"))
        preds = [core.predict_image(m, rgb) for m in models]
        cols = st.columns(len(preds))
        for col, pred, m in zip(cols, preds, models):
            with col:
                st.image(cv2.cvtColor(pred.overlay_bgr, cv2.COLOR_BGR2RGB), caption=f"{m.card['display_name']}",
                         use_container_width=True)
                st.caption(pred.note + f"; post-processing: {m.postprocess_source}")
                st.download_button("Download overlay (PNG)", _png_bytes(pred.overlay_bgr),
                                   file_name=f"{Path(upload.name).stem}_{m.variant}.png", key=f"dl_{m.variant}_{id(m)}")
        table = pd.concat([core.lanes_table(p) for p in preds], ignore_index=True)
        st.dataframe(table, use_container_width=True)
        latency = pd.DataFrame([{"model": p.variant, **p.result.latency_ms} for p in preds])
        st.dataframe(latency.round(2), use_container_width=True)
        summary = {"image": upload.name, "models": [
            {"variant": p.variant, "checkpoint": str(m.checkpoint), "latency_ms": p.result.latency_ms,
             "lanes": [None if l is None else np.round(l, 1).tolist() for l in p.result.lanes],
             "confidence": p.result.lane_confidence} for p, m in zip(preds, models)]}
        st.download_button("Download summary (JSON)", json.dumps(summary, indent=2), file_name="summary.json")
    elif not models:
        st.info("Select at least one model in the sidebar.")

# -------------------------------------------------------------------- video

with tabs[1]:
    st.subheader("Streaming inference on a video")
    st.caption("Frames are decoded and processed one at a time (the video is never loaded into memory). "
               "Temporal models reuse cached features of previous frames.")
    vid = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    c1, c2 = st.columns(2)
    max_frames = c1.slider("Max frames", 10, 600, 120, step=10)
    every = c2.slider("Process every n-th frame", 1, 5, 1)
    if vid is not None and models and st.button("Run on video"):
        path = core.save_upload(vid.getvalue(), Path(vid.name).suffix)
        for m in models:
            bar = st.progress(0.0, text=f"{m.variant}")
            out = path.with_name(f"overlay_{m.variant}.mp4")
            info = core.process_video(m, path, out, max_frames=max_frames, every=every,
                                      progress=lambda f, b=bar, v=m.variant: b.progress(f, text=v))
            st.markdown(f"**{m.card['display_name']}** - {info['frames']} frames, "
                        f"mean total {info.get('total_ms_mean', float('nan')):.1f} ms/frame, "
                        f"history fallbacks {info['history_fallbacks']}")
            data = out.read_bytes()
            st.video(data)
            st.download_button(f"Download {m.variant} overlay video", data, file_name=out.name, key=f"v_{m.variant}")

# ------------------------------------------------------------------- models

with tabs[2]:
    st.subheader("Architecture")
    if models:
        m = models[st.selectbox("Model", range(len(models)), format_func=lambda i: models[i].variant)]
        st.json(m.card, expanded=False)
        params = core.parameter_table(m.model)
        st.metric("Parameters", f"{sum(p.numel() for p in m.model.parameters()) / 1e6:.2f} M")
        st.dataframe(params, use_container_width=True)
        svg = core.graph_svg(m)
        if svg:
            st.image(svg, caption="computational graph (torchview, depth 2)")
        else:
            st.info("Graph rendering unavailable (needs torchview + Graphviz `dot`); showing every layer's output "
                    "shape instead.")
            st.dataframe(core.layer_shapes(m), use_container_width=True, height=400)
        st.caption(f"Checkpoint {m.checkpoint} - epoch {m.meta.get('epoch')}, "
                   f"{m.meta.get('selection_metric')} = {m.meta.get('score')}")

# --------------------------------------------------------- training curves

runs = core.list_runs(results_root)
with tabs[3]:
    st.subheader("Training and validation curves")
    if runs:
        run = st.selectbox("Run", runs, format_func=lambda p: p.name, key="run_curves")
        hist = core.histories(run)
        if hist.empty:
            st.info("No history files in this run.")
        else:
            metric = st.selectbox("Metric", [c for c in hist.columns if c.startswith(("val_", "train_"))],
                                  index=0)
            wide = hist.pivot_table(index="epoch", columns=["variant", "seed"], values=metric)
            wide.columns = [f"{v} s{s}" for v, s in wide.columns]
            st.line_chart(wide)
            st.dataframe(hist, use_container_width=True, height=300)
    else:
        st.info("No finished runs found.")

# ------------------------------------------------------- experiment metrics

with tabs[4]:
    st.subheader("Experiment metrics")
    if runs:
        run = st.selectbox("Run", runs, format_func=lambda p: p.name, key="run_metrics")
        report = run / "report" / "REPORT.md"
        tables = core.run_tables(run)
        if report.exists():
            with st.expander("REPORT.md", expanded=True):
                st.markdown(report.read_text(encoding="utf-8"))
        for name, df in tables.items():
            if name != "all_results":
                with st.expander(name):
                    st.dataframe(df, use_container_width=True)
        for png in sorted((run / "report").glob("*.png")):
            st.image(str(png), caption=png.stem)
        if "all_results" in tables:
            st.download_button("Download all_results.csv", tables["all_results"].to_csv(index=False),
                               file_name=f"{run.name}_all_results.csv")

# ------------------------------------------------------------------ latency

with tabs[5]:
    st.subheader("Inference latency (batch 1, this machine)")
    st.caption("Streaming latency split into preprocessing, model and post-processing. For temporal models "
               "'cached' reuses previous frames' features; 'recompute' re-encodes every history frame.")
    if models and st.button("Measure"):
        from tac_ufld.deploy.check import synthetic_frames
        from tac_ufld.evaluation.hardware import BudgetProfile, measure
        from tac_ufld.inference.streaming import StreamingLaneDetector

        frames = synthetic_frames(20)
        profile = BudgetProfile(warmup=5, measure_frames=30)
        rows = []
        for m in models:
            for mode in (["cached", "recompute"] if m.temporal else ["cached"]):
                meas = measure(StreamingLaneDetector(m, mode=mode), frames, profile, mode if m.temporal else "single")
                rows.append(meas.summary())
        df = pd.DataFrame(rows)
        cols = ["variant", "mode", "backend", "device", "preprocess_ms_mean", "model_ms_mean", "postprocess_ms_mean",
                "total_ms_mean", "total_ms_p95", "fps_total_mean", "peak_gpu_memory_mb"]
        st.dataframe(df[cols].round(2), use_container_width=True)
        st.bar_chart(df.set_index(df["variant"] + " " + df["mode"])[["preprocess_ms_mean", "model_ms_mean",
                                                                     "postprocess_ms_mean"]])
        st.download_button("Download latency (CSV)", df.to_csv(index=False), file_name="latency.csv")
    bench_dir = Path(results_root) / "benchmarks"
    for md in sorted(bench_dir.glob("*.md")) if bench_dir.exists() else []:
        with st.expander(f"Benchmark report: {md.stem}"):
            st.markdown(md.read_text(encoding="utf-8"))
