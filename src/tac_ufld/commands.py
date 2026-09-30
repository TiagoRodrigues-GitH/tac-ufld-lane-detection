"""Sub-commands beyond run / check-data / datasets / validate-dataset.
Each is implemented in its own module; heavy imports happen inside the
handlers so ``--help`` stays fast.

    sanity      model shapes, finite losses, gradients, warm starts (no training)
    ablate      preprocessing / augmentation ablations (plan by default, --confirm to run)
    export      ONNX (+ FP16 / INT8 / TensorRT) with numerical checks -> deployment folder
    stream      streaming inference on a video or image folder -> overlay video + JSON
    benchmark   latency / memory on this machine + real-time budget simulation
    ui          Streamlit interface (inference and results only; never trains)
    robustness  held-out lane F1 with a degraded current frame (history clean)
    site        static HTML results page (GitHub Pages) from finished runs
    package     supervisor hand-off ZIP (code, configs, docs; no data/checkpoints)
    doctor      environment, GPU, optional dependency and dataset checks
    carla-demo  optional qualitative demo on CARLA camera frames
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac_ufld.config import PROJECT_ROOT, ConfigError

HANDLERS: dict = {}


def _handler(name):
    def deco(fn):
        HANDLERS[name] = fn
        return fn
    return deco


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("sanity", help="model checks before training: shapes, losses, gradients, warm starts")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "elas_pilot.yaml"))
    p.add_argument("--variants", nargs="+")
    p.add_argument("--device", default="cpu")
    p.add_argument("--real", action="store_true", help="use one real training batch (needs the dataset)")

    p = sub.add_parser("ablate", help="run an ablation spec (prints the plan unless --confirm)")
    p.add_argument("--spec", required=True)
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--only", nargs="+", help="run only these arms")
    p.add_argument("--summarize-only", action="store_true")

    p = sub.add_parser("export", help="ONNX export + numerical checks (+ FP16, INT8, TensorRT)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", help="deployment folder (default: <run>/deploy/<variant>_seed<n>)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--fp16", action="store_true", help="ONNX FP16 copy for the ONNX Runtime CUDA provider")
    p.add_argument("--int8", action="store_true", help="INT8 QDQ, calibrated on validation frames")
    p.add_argument("--calib-frames", type=int, default=64)
    p.add_argument("--tensorrt", nargs="*", choices=["fp32", "fp16", "int8"], help="build TensorRT engines")
    p.add_argument("--check-frames", type=int, default=24)
    p.add_argument("--synthetic", action="store_true", help="synthetic frames for checks/calibration (no dataset)")
    p.add_argument("--eval-frames", type=int, default=0,
                   help="task-level accuracy of every exported precision on N validation frames")
    p.add_argument("--data-root")

    p = sub.add_parser("stream", help="streaming inference on a video or an image folder")
    p.add_argument("--checkpoint", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--images", help="folder of frames, processed in name order")
    p.add_argument("--backend", choices=["torch", "onnx", "tensorrt"], default="torch")
    p.add_argument("--deployment", help="deployment folder for onnx/tensorrt")
    p.add_argument("--precision", default="fp32")
    p.add_argument("--mode", choices=["cached", "recompute", "carry"], default="cached",
                   help="carry = one hidden state per stream (recurrent models ufld_v07 / lite_v06)")
    p.add_argument("--kalman", action="store_true",
                   help="filter the lanes with the output Kalman tracker (validation-tuned values when the run "
                        "tuned them, defaults otherwise)")
    p.add_argument("--every", type=int, default=1, help="process every n-th frame (simulates drops)")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--out", help="overlay video path (default: results/streams/<name>.mp4)")
    p.add_argument("--device", default="auto")

    p = sub.add_parser("benchmark", help="latency/memory on this machine + budget simulation")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--profile", default=str(PROJECT_ROOT / "configs" / "deploy" / "desktop.yaml"))
    p.add_argument("--modes", nargs="+", default=["cached", "recompute"], choices=["cached", "recompute"],
                   help="temporal inference modes to compare (single-frame models always run once)")
    p.add_argument("--backend", choices=["torch", "onnx", "tensorrt"], default="torch")
    p.add_argument("--deployments", nargs="+", help="deployment folders aligned with --checkpoints")
    p.add_argument("--video", help="real frames instead of synthetic ones")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", help="report path without extension (default: results/benchmarks/<profile>)")
    for flag, typ in (("--target-fps", float), ("--max-latency-ms", float), ("--memory-budget-mb", float),
                      ("--batch-size", int), ("--precision", str), ("--drop-rate", float), ("--slowdown", float),
                      ("--frames", int), ("--measure-frames", int)):
        p.add_argument(flag, type=typ)
    p.add_argument("--input-size", help="camera frame size HxW, e.g. 720x1280")

    p = sub.add_parser("ui", help="Streamlit interface (never trains)")
    p.add_argument("--results", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--headless", action="store_true")

    p = sub.add_parser("robustness", help="held-out lane F1 with a degraded current frame (history clean)")
    p.add_argument("--runs", nargs="+", required=True, help="finished run folders (e.g. results/elas_pilot_v2)")
    p.add_argument("--seed", type=int, default=2026, help="seed of the per-frame degradations")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=2)

    p = sub.add_parser("site", help="static results page for GitHub Pages")
    p.add_argument("--runs", nargs="+", required=True, help="results folders (e.g. results/elas_pilot)")
    p.add_argument("--out", default=str(PROJECT_ROOT / "site"))
    p.add_argument("--title", default="TAC-UFLD pilot results")
    p.add_argument("--extra", nargs="*", default=[], help="extra JSON/Markdown reports to embed (benchmarks, checks)")
    p.add_argument("--notes", help="Markdown file whose '- ' bullets are the findings (default docs/PILOT_FINDINGS.md)")
    p.add_argument("--roadmap", help="roadmap Markdown (default docs/ROADMAP.md)")

    p = sub.add_parser("package", help="supervisor hand-off ZIP")
    p.add_argument("--out", default=str(PROJECT_ROOT / "dist"))
    p.add_argument("--include-results", nargs="*", default=[], help="result folders whose reports (not checkpoints) to add")

    p = sub.add_parser("doctor", help="environment, GPU and dependency checks")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("carla-demo", help="optional: lanes on CARLA camera frames (qualitative only)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--town", default="Town04")
    p.add_argument("--out", default=str(PROJECT_ROOT / "results" / "simulator"))


# -------------------------------------------------------------------- helpers


def _device(name: str) -> str:
    import torch

    return ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name


def _read_rgb(path) -> "np.ndarray":
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def split_records(cfg, split: str, n: int, data_root: str | None = None) -> tuple[list, object]:
    """``n`` records of ``split`` (runs of consecutive frames) and the adapter."""
    import copy

    from tac_ufld.data import build_adapter
    from tac_ufld.data.registry import DEFAULT_REGISTRY, load_registry
    from tac_ufld.data.splits import build_splits, consecutive_frames

    cfg = copy.deepcopy(cfg)
    if data_root:
        cfg.data.root = data_root
    elif not cfg.data_root().is_dir() and DEFAULT_REGISTRY.exists():
        reg = load_registry()
        if cfg.data.dataset in reg and reg[cfg.data.dataset].enabled:
            cfg.data.root = str(reg[cfg.data.dataset].resolve_root())
    adapter = build_adapter(cfg)
    splits, _ = build_splits(cfg, adapter)
    return consecutive_frames(getattr(splits, split), n), adapter


def split_frames(cfg, split: str, n: int, data_root: str | None = None) -> list:
    """Consecutive RGB frames of ``split`` for calibration / checks."""
    records, _ = split_records(cfg, split, n, data_root)
    return [_read_rgb(r.image_path) for r in records]


def make_backend(kind: str, loaded, deployment: str | None, precision: str, device: str):
    from tac_ufld.inference.backends import TorchBackend

    if kind == "torch":
        return TorchBackend(loaded.model, device, precision if precision in ("fp32", "fp16") else "fp32")
    if deployment is None:
        raise ConfigError(f"--backend {kind} needs a deployment folder (python -m tac_ufld export)")
    if kind == "onnx":
        from tac_ufld.deploy.ort_backend import OnnxRuntimeBackend

        sub = Path(deployment) / precision if precision in ("fp16", "int8") else Path(deployment)
        return OnnxRuntimeBackend(sub)
    from tac_ufld.deploy.tensorrt import TensorRTBackend, available

    if not available():
        raise ConfigError("TensorRT is not installed / no CUDA device")
    return TensorRTBackend(deployment, precision)


# -------------------------------------------------------------------- handlers


@_handler("sanity")
def cmd_sanity(args) -> int:
    import pandas as pd

    from tac_ufld.config import load_config
    from tac_ufld.sanity import run_sanity

    cfg = load_config(args.config)
    variants = args.variants or cfg.model.variants
    batches = None
    if args.real:
        from tac_ufld.data import build_adapter
        from tac_ufld.data.dataset import TemporalLaneDataset
        from tac_ufld.data.splits import build_splits, consecutive_frames
        from tac_ufld.data.targets import make_row_anchors

        d = cfg.data
        adapter = build_adapter(cfg)
        splits, _ = build_splits(cfg, adapter)
        anchors = make_row_anchors(d.img_h, d.num_row_anchors, d.row_anchor_range)
        batches = {}
        sample = consecutive_frames(splits.train, 2, runs=2)  # mid-sequence: real history frames exist
        for frames in {1, d.num_frames}:
            ds = TemporalLaneDataset(sample, adapter, d, anchors, frames, augment=True)
            items = [ds[i] for i in range(2)]
            batches[frames] = {k: __import__("torch").stack([it[k] for it in items]) for k in items[0] if k != "index"}
    rows = run_sanity(cfg, variants, _device(args.device), batches)
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", 30, "display.width", 200):
        print(df.to_string(index=False))
    return 0 if df["ok"].all() else 1


@_handler("ablate")
def cmd_ablate(args) -> int:
    from tac_ufld.ablation import arm_configs, load_spec, run_ablation, summarize

    spec = load_spec(args.spec)
    configs = arm_configs(spec)
    ref = configs[spec.reference_arm]
    if args.summarize_only:
        print(summarize(spec, configs, ref.output_root().parent / spec.name))
        return 0
    print(f"Ablation '{spec.name}': {len(configs)} arms x {len(ref.train.seeds)} seeds x "
          f"{len(ref.model.variants)} variant(s), <= {ref.train.epochs} epochs each (base {spec.base_config})")
    for arm, cfg in configs.items():
        overrides = spec.arms[arm] or {}
        print(f"  {arm:<22} in_channels={cfg.in_channels}  {overrides or '(reference)'}")
    if not args.confirm:
        print("\nPlan only. Re-run with --confirm to train every arm.")
        return 0
    print(run_ablation(args.spec, only=args.only))
    return 0


@_handler("export")
def cmd_export(args) -> int:
    import torch

    from tac_ufld.deploy.check import numerical_check, synthetic_frames
    from tac_ufld.deploy.onnx_export import export_model
    from tac_ufld.deploy.ort_backend import OnnxRuntimeBackend, ort_providers
    from tac_ufld.inference.loading import load_model

    ckpt = Path(args.checkpoint)
    loaded = load_model(ckpt, "cpu")
    seed = loaded.meta.get("seed")
    out = Path(args.out) if args.out else ckpt.parents[2] / "deploy" / f"{loaded.variant}_seed{seed}"
    manifest = export_model(loaded, out, args.opset)
    n_check, n_calib = args.check_frames, args.calib_frames if args.int8 else 0
    if args.synthetic:
        frames = synthetic_frames(n_check + n_calib)
        source = "synthetic"
    else:
        frames = split_frames(loaded.cfg, "val", n_check + n_calib, args.data_root)
        source = "validation split"
    calib, check = frames[:n_calib], frames[n_calib:]
    checks = {"frames": len(check), "frame_source": source}
    providers = ort_providers()
    checks["onnxruntime_fp32"] = numerical_check(loaded, OnnxRuntimeBackend(out, providers), check, "fp32")
    if args.fp16:
        from tac_ufld.deploy.precision import convert_fp16

        convert_fp16(out, out / "fp16")
        backend = OnnxRuntimeBackend(out / "fp16", providers)
        if "CUDAExecutionProvider" in backend.active_providers.values():
            checks["onnxruntime_fp16"] = numerical_check(loaded, backend, check, "fp16")
        else:
            checks["onnxruntime_fp16"] = {"skipped": "ONNX Runtime CUDA provider unavailable (FP16 targets GPU)"}
    if args.int8:
        from tac_ufld.deploy.precision import quantize_int8

        quantize_int8(loaded, out, out / "int8", calib)
        checks["onnxruntime_int8"] = numerical_check(loaded, OnnxRuntimeBackend(out / "int8", ["CPUExecutionProvider"]),
                                                     check, "int8")
    if args.tensorrt is not None:
        from tac_ufld.deploy.tensorrt import TensorRTBackend, available, build_deployment

        if not available():
            checks["tensorrt"] = {"skipped": "TensorRT not installed or no CUDA device"}
        else:
            loaded.model.to("cuda")
            loaded.device = "cuda"
            for precision in args.tensorrt or ["fp16"]:
                if precision == "int8" and not (out / "int8").exists():
                    checks["tensorrt_int8"] = {"skipped": "run with --int8 first (QDQ calibration)"}
                    continue
                try:
                    build_deployment(out, precision, onnx_dir=out / "int8" if precision == "int8" else None)
                    checks[f"tensorrt_{precision}"] = numerical_check(loaded, TensorRTBackend(out, precision),
                                                                      check, precision)
                except Exception as exc:  # record, keep the other precisions
                    checks[f"tensorrt_{precision}"] = {"failed": f"{type(exc).__name__}: {str(exc)[:400]}"}
    if args.eval_frames:
        from tac_ufld.deploy.check import deploy_eval
        from tac_ufld.inference.backends import TorchBackend

        records, adapter = split_records(loaded.cfg, "val", args.eval_frames, args.data_root)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded.model.to(device)
        loaded.device = device
        evals = [deploy_eval(loaded, TorchBackend(loaded.model, device), records, adapter),
                 deploy_eval(loaded, OnnxRuntimeBackend(out, providers), records, adapter)]
        if (out / "int8").exists():
            evals.append(deploy_eval(loaded, OnnxRuntimeBackend(out / "int8", ["CPUExecutionProvider"]),
                                     records, adapter) | {"backend": "onnxruntime-int8"})
        for precision in args.tensorrt or []:
            if (out / "engines" / f"build_info.{precision}.json").exists():
                from tac_ufld.deploy.tensorrt import TensorRTBackend

                evals.append(deploy_eval(loaded, TensorRTBackend(out, precision), records, adapter))
        checks["task_accuracy_val"] = evals
        for e in evals:
            print(f"val lane_f1_iou50 {e['backend']:<20} {e['lane_f1_iou50']:.4f}  ({e['frames']} frames)")
    manifest["numerical_checks"] = checks
    (out / "deployment.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (out / "numerical_checks.json").write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")
    for name, r in checks.items():
        if isinstance(r, dict) and "pass" in r:
            print(f"{name:<18} pass={r['pass']}  max|logits|={r['max_abs_logits']:.2e}  "
                  f"max|exist|={r['max_abs_exist']:.2e}  max|x|={r['max_abs_x_px']:.3f}px")
        elif isinstance(r, dict):
            print(f"{name:<18} {r}")
    print(f"deployment folder: {out}")
    return 0 if all(r.get("pass", True) for r in checks.values() if isinstance(r, dict)
                    and not str(r.get("precision", "")).startswith("int8")) else 1


def _frame_source(args):
    if args.video:
        return None
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    if not files:
        raise ConfigError(f"no images in {args.images}")
    return files


@_handler("stream")
def cmd_stream(args) -> int:
    import numpy as np

    from tac_ufld.inference.loading import load_model
    from tac_ufld.inference.streaming import StreamingLaneDetector
    from tac_ufld.visualization.overlays import VideoSink, draw_prediction

    device = _device(args.device)
    loaded = load_model(args.checkpoint, device)
    backend = make_backend(args.backend, loaded, args.deployment, args.precision, device)
    tracker = None
    if args.kalman:
        from tac_ufld.evaluation.tracking import KalmanParams

        tracker = KalmanParams(**loaded.kalman) if loaded.kalman else KalmanParams(threshold=loaded.postprocess.threshold)
    det = StreamingLaneDetector(loaded, backend=backend, mode=args.mode, tracker=tracker)
    name = Path(args.video or args.images).stem
    out = Path(args.out) if args.out else PROJECT_ROOT / "results" / "streams" / f"{name}_{loaded.variant}.mp4"
    records, sink = [], VideoSink(out, fps=10)
    if args.video:
        iterator = det.infer_video(args.video, every=args.every, max_frames=args.max_frames)
    else:
        def iterator_images():
            files = _frame_source(args)[:: args.every][: args.max_frames]
            for i, f in enumerate(files):
                rgb = _read_rgb(f)
                yield rgb, det.infer_frame(rgb, "images", frame_index=i * args.every)

        iterator = iterator_images()
    for rgb, res in iterator:
        sink.write(draw_prediction(rgb, res, loaded.card["display_name"]))
        records.append({"frame_index": res.frame_index, "fallbacks": res.fallbacks, "reset": res.reset_reason,
                        "lane_confidence": res.lane_confidence, **res.latency_ms,
                        "lanes": [None if l is None else np.round(l, 1).tolist() for l in res.lanes]})
    sink.close()
    summary = {"frames": len(records), "backend": backend.name, "mode": args.mode, "video": str(out),
               "kalman": None if tracker is None else tracker.as_dict()}
    for key in ("preprocess_ms", "model_ms", "postprocess_ms", "total_ms"):
        vals = [r[key] for r in records[3:]] or [r[key] for r in records]
        summary[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


@_handler("benchmark")
def cmd_benchmark(args) -> int:
    from tac_ufld.deploy.check import synthetic_frames
    from tac_ufld.evaluation.hardware import evaluate_budget, load_profile, measure, write_report
    from tac_ufld.inference.loading import load_model
    from tac_ufld.inference.streaming import StreamingLaneDetector

    size = [int(v) for v in args.input_size.lower().split("x")] if args.input_size else [None, None]
    profile = load_profile(args.profile, target_fps=args.target_fps, max_latency_ms=args.max_latency_ms,
                           memory_budget_mb=args.memory_budget_mb, batch_size=args.batch_size,
                           precision=args.precision, drop_rate=args.drop_rate, slowdown=args.slowdown,
                           frames=args.frames, measure_frames=args.measure_frames,
                           input_height=size[0], input_width=size[1])
    device = _device(args.device)
    if args.video:
        from tac_ufld.utils import open_video
        import cv2

        cap, frames = open_video(args.video), []
        while len(frames) < 60:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(cv2.resize(bgr, (profile.input_width, profile.input_height)), cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        frames = synthetic_frames(60, size=(profile.input_height, profile.input_width))
    rows = []
    deployments = args.deployments or [None] * len(args.checkpoints)
    for ckpt, dep in zip(args.checkpoints, deployments):
        loaded = load_model(ckpt, device)
        backend = make_backend(args.backend, loaded, dep, profile.precision, device)
        modes = args.modes if loaded.temporal else ["single"]
        for mode in modes:
            det = StreamingLaneDetector(loaded, backend=backend, mode="cached" if mode == "single" else mode)
            m = measure(det, frames, profile, mode)
            rows.append({"measured": m.summary(),
                         "simulated": evaluate_budget(m, profile, loaded.num_frames, loaded.card["temporal_step"])})
    out = Path(args.out) if args.out else PROJECT_ROOT / "results" / "benchmarks" / profile.name
    write_report(out, profile, rows)
    print(out.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


@_handler("ui")
def cmd_ui(args) -> int:
    import subprocess

    app = Path(__file__).parent / "ui" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port)]
    if args.headless:
        cmd += ["--server.headless", "true"]
    cmd += ["--", "--results", args.results]
    return subprocess.call(cmd)


@_handler("robustness")
def cmd_robustness(args) -> int:
    from tac_ufld.evaluation.robustness import evaluate_run
    from tac_ufld.utils import setup_logging

    setup_logging()
    device = _device(args.device)
    for run in args.runs:
        summary = evaluate_run(Path(run), device, seed=args.seed, workers=args.workers)
        overall = summary[summary["op"] == "all"][["variant", "input", "clean_lane_f1", "lane_f1", "drop_vs_clean"]]
        print()
        print(run)
        print(overall.round(3).to_string(index=False))
    return 0


@_handler("site")
def cmd_site(args) -> int:
    from tac_ufld.site import build_site

    print(build_site([Path(r) for r in args.runs], Path(args.out), args.title, [Path(e) for e in args.extra],
                     notes=Path(args.notes) if args.notes else None,
                     roadmap=Path(args.roadmap) if args.roadmap else None))
    return 0


@_handler("package")
def cmd_package(args) -> int:
    from tac_ufld.package import build_package

    path, report = build_package(Path(args.out), [Path(r) for r in args.include_results])
    print(report)
    print(f"\nZIP: {path}")
    return 0


@_handler("doctor")
def cmd_doctor(args) -> int:
    from tac_ufld.doctor import format_report, run_checks

    report = run_checks()
    print(json.dumps(report, indent=2, default=str) if args.json else format_report(report))
    return 0 if report["ok"] else 1


@_handler("carla-demo")
def cmd_carla(args) -> int:
    from tac_ufld.sim.carla_source import run_carla_demo

    return run_carla_demo(args)
