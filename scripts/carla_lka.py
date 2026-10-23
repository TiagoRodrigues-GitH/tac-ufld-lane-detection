"""Closed-loop lane keeping in CARLA with the TAC-UFLD lane detector (docs/CARLA.md, "Closed-loop lane keeping").

Runs in the CARLA client venv (Python 3.12). With ``--detector model`` the trained model runs in the project venv
(Python 3.14 + torch), started automatically as a socket server (``tac_ufld.sim.detector_bridge``)::

    $py = "C:\\CARLA\\venv312\\Scripts\\python.exe"
    & $py scripts\\carla_lka.py run --scenario all --detector oracle --out results\\carla_lka\\oracle
    & $py scripts\\carla_lka.py run --scenario all --detector model \\
        --checkpoint results\\review_lite_tsm_pilot\\seed_1\\checkpoints\\lite_tsm.pt --out results\\carla_lka\\lite_tsm
    & $py scripts\\carla_lka.py sweep --scenario straight curves --k-offset 0.3 0.6 1.0 --k-heading 0 0.5 1.0 \\
        --out results\\carla_lka\\sweep
    & $py scripts\\carla_lka.py summarize results\\carla_lka\\oracle        # metrics.csv again from the logs

    # detector server by hand (project venv), e.g. on another machine; then run with --detector-addr HOST:PORT
    python scripts\\carla_lka.py serve --checkpoint <ckpt> --host 0.0.0.0 --port 5599

Each run writes ``<out>/<tag>.csv`` (one row per frame, ``lka_metrics.COLUMNS``) and ``<out>/<tag>.json``
(configuration, start point, detector, metrics); ``<out>/metrics.csv`` has one row per run.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac_ufld.ego import LaneKeepingParams  # noqa: E402
from tac_ufld.sim import lka_metrics  # noqa: E402
from tac_ufld.sim.lka_loop import SCENARIOS, CameraRig, RunConfig  # noqa: E402

META = ("tag", "scenario", "detector", "model_run", "loop", "town", "terminated", "speed_kmh", "fps", "k_offset", "k_heading",
        "max_steer", "warn_tlc_s", "half_width", "disturbance", "y_ref_frac", "y_far_frac", "seed")


def default_project_python() -> Path:
    env = os.environ.get("TAC_UFLD_PYTHON")
    if env:
        return Path(env)
    venv = REPO.parent / ".venv"
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def open_detector(args: argparse.Namespace, out: Path):
    """None for the oracle; else an object with info / reset / infer / close (``detector_bridge``)."""
    if args.detector == "oracle":
        return None
    from tac_ufld.sim import detector_bridge as bridge

    if args.detector_addr:
        host, port = args.detector_addr.rsplit(":", 1)
        return bridge.RemoteLaneDetector(host, int(port))
    if not args.checkpoint:
        raise SystemExit("--detector model needs --checkpoint (or --detector-addr)")
    try:
        import torch  # noqa: F401  # the model can run in this interpreter
    except ImportError:
        python = Path(args.detector_python) if args.detector_python else default_project_python()
        if not python.exists():
            raise SystemExit(f"project Python not found at {python}: pass --detector-python or set TAC_UFLD_PYTHON")
        print(f"starting the detector server with {python} (log: {out / 'detector.log'})", flush=True)
        return bridge.spawn_server(python, args.checkpoint, out / "detector.log", args.device, args.mode, args.kalman)
    return bridge.StreamingHandler(args.checkpoint, args.device, args.mode, args.kalman)


def run_config(args: argparse.Namespace, k_offset: float, k_heading: float) -> RunConfig:
    keeping = LaneKeepingParams(k_offset=k_offset, k_heading=k_heading, max_steer=args.max_steer,
                                warn_tlc_s=args.warn_tlc, warn_margin=args.warn_margin)
    rig = CameraRig(fov_deg=args.fov, x_m=args.cam_x, z_m=args.cam_z, horizon_frac=args.horizon)
    return RunConfig(fps=args.fps, speed_kmh=args.speed_kmh, warmup_s=args.warmup_s, rig=rig, y_ref_frac=args.y_ref,
                     y_far_frac=args.y_far, keeping=keeping, weather=args.weather, seed=args.seed, town=args.town,
                     duration_s=args.duration_s, map_layers=args.map_layers, disturbance=args.disturbance,
                     snapshot_every=args.snapshot_every,
                     video=args.video, shadow=args.shadow)


def scenario_names(requested: list[str]) -> list[str]:
    names = list(SCENARIOS) if "all" in requested else requested
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenario(s) {unknown}; choose from {list(SCENARIOS)} or all")
    return sorted(names, key=lambda n: (SCENARIOS[n].town, list(SCENARIOS).index(n)))  # fewer town loads


def drive(args: argparse.Namespace, grid: list[tuple[float, float]], tag_gains: bool) -> None:
    from tac_ufld.sim.lka_loop import load_carla, run_scenario

    carla = load_carla()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    detector = open_detector(args, out)
    detector_info = detector.info() if detector is not None else {"variant": "oracle"}
    try:
        for name in scenario_names(args.scenario):
            for k_offset, k_heading in grid:
                cfg = run_config(args, k_offset, k_heading)
                tag = f"{name}__ko{k_offset:g}_kh{k_heading:g}" if tag_gains else name
                tag += f"__{args.label}" if args.label else ""
                t0 = time.time()
                rows, info = run_scenario(client, SCENARIOS[name], cfg, detector, out, tag)
                lka_metrics.write_log(out / f"{tag}.csv", rows)
                metrics = lka_metrics.summarize(lka_metrics.columns_from_rows(rows), cfg.fps)
                record = {"tag": tag, "scenario": name, "detector": detector_info, "config": _jsonable(asdict(cfg)),
                          "info": info, "metrics": metrics, "wall_s": round(time.time() - t0, 1)}
                (out / f"{tag}.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
                print(f"[{tag}] {len(rows)} frames in {record['wall_s']} s: " + ", ".join(
                    f"{k} {metrics[k]:.3g}" for k in ("lat_rms_m", "crossings_lka", "ldw_events", "ldw_lead_median_s",
                                                      "lka_max_excursion_m", "both_lanes_frac") if k in metrics),
                      flush=True)
    finally:  # a crashed simulator still leaves the finished runs summarised
        if detector is not None:
            detector.close()
        summarize(out)


def _jsonable(value):
    return json.loads(json.dumps(value, default=str))


def summarize(out: Path, fps_default: float = 30.0) -> Path:
    """metrics.csv from every ``<tag>.json`` + ``<tag>.csv`` in ``out`` (metrics recomputed from the frame logs)."""
    rows = []
    for meta_path in sorted(out.glob("*.json")):
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        log_path = out / f"{record['tag']}.csv"
        if "metrics" not in record or not log_path.exists():
            continue
        cfg, keeping = record["config"], record["config"]["keeping"]
        fps = float(cfg.get("fps", fps_default))
        metrics = lka_metrics.summarize(lka_metrics.read_log(log_path), fps)
        variant = record["detector"].get("variant")
        loop = "oracle" if variant == "oracle" else "shadow" if record["info"].get("shadow") else "closed"
        ckpt = record["detector"].get("checkpoint")
        model_run = Path(ckpt).parents[2].name if ckpt and len(Path(ckpt).parents) > 2 else ""  # results/<run>/...
        rows.append({"tag": record["tag"], "scenario": record["scenario"], "loop": loop,
                     "detector": variant, "model_run": model_run, "town": record["info"].get("town"),
                     "terminated": record["info"].get("terminated", ""), "speed_kmh": cfg["speed_kmh"], "fps": fps,
                     "k_offset": keeping["k_offset"], "k_heading": keeping["k_heading"],
                     "max_steer": keeping["max_steer"], "warn_tlc_s": keeping["warn_tlc_s"],
                     "half_width": round(keeping["half_width"], 4), "disturbance": cfg.get("disturbance"),
                     "y_ref_frac": cfg["y_ref_frac"],
                     "y_far_frac": cfg["y_far_frac"], "seed": cfg["seed"], **metrics})
    path = out / "metrics.csv"
    if rows:
        fields = list(META) + [k for k in rows[0] if k not in META]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
        print(f"{len(rows)} runs -> {path}")
    return path


def add_run_args(ap: argparse.ArgumentParser, gains_as_lists: bool) -> None:
    ap.add_argument("--scenario", nargs="+", default=["all"], help=f"{', '.join(SCENARIOS)} or all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="", help="suffix of the run tags, so several detectors can share --out")
    ap.add_argument("--detector", choices=["oracle", "model"], default="oracle")
    ap.add_argument("--checkpoint", help="trained checkpoint (results/<run>/seed_<n>/checkpoints/<variant>.pt)")
    ap.add_argument("--detector-python", help="project venv Python for the detector server (default ../.venv)")
    ap.add_argument("--detector-addr", help="HOST:PORT of a running detector server (scripts/carla_lka.py serve)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--mode", default="cached", help="streaming mode: cached | recompute | carry")
    ap.add_argument("--kalman", action="store_true", help="validation-tuned Kalman output tracker of the run")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--timeout", type=float, default=300.0, help="CARLA client timeout, s (loading a town is slow)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--speed-kmh", type=float, default=60.0)
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--duration-s", type=float, default=None, help="override the scenario duration")
    ap.add_argument("--town", default=None, help="override the scenario town")
    ap.add_argument("--map-layers", choices=["all", "minimal"], default=None,
                    help="override the scenario's map layers (curves: minimal, i.e. Town06_Opt without buildings)")
    ap.add_argument("--weather", default="ClearNoon")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--disturbance", type=float, default=0.003,
                    help="random steering disturbance, std of the normalised steer (0 = none)")
    nargs = "+" if gains_as_lists else None
    # defaults from the oracle sweep (docs/CARLA.md): k_heading > 0 biases the car toward the inside of curves
    ap.add_argument("--k-offset", type=float, nargs=nargs, default=[0.6] if gains_as_lists else 0.6)
    ap.add_argument("--k-heading", type=float, nargs=nargs, default=[0.0] if gains_as_lists else 0.0)
    ap.add_argument("--max-steer", type=float, default=0.3)
    ap.add_argument("--warn-tlc", type=float, default=1.0)
    ap.add_argument("--warn-margin", type=float, default=0.05)
    ap.add_argument("--y-ref", type=float, default=0.95, help="tracker reference row, fraction of the height")
    ap.add_argument("--y-far", type=float, default=0.72, help="tracker far row, fraction of the height")
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--cam-x", type=float, default=2.0)
    ap.add_argument("--cam-z", type=float, default=1.3)
    ap.add_argument("--horizon", type=float, default=0.6, help="horizon row, fraction of the height")
    ap.add_argument("--snapshot-every", type=int, default=0, help="overlay JPEG every N frames (0 = none)")
    ap.add_argument("--video", action="store_true", help="overlay MP4 per run")
    ap.add_argument("--shadow", action="store_true",
                    help="the detector rides along (lanes, offset, LDW logged) while ground-truth lanes steer")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_run_args(sub.add_parser("run", help="drive scenarios once"), gains_as_lists=False)
    add_run_args(sub.add_parser("sweep", help="drive scenarios for every LKA gain pair"), gains_as_lists=True)
    serve = sub.add_parser("serve", help="detector server (project venv)")
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default="5599")
    serve.add_argument("--device", default="auto")
    serve.add_argument("--mode", default="cached")
    serve.add_argument("--kalman", action="store_true")
    summ = sub.add_parser("summarize", help="metrics.csv from the per-run logs of a folder")
    summ.add_argument("out")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        drive(args, [(args.k_offset, args.k_heading)], tag_gains=False)
    elif args.cmd == "sweep":
        drive(args, list(itertools.product(args.k_offset, args.k_heading)), tag_gains=True)
    elif args.cmd == "serve":
        from tac_ufld.sim.detector_bridge import main as serve_main

        serve_main(["--checkpoint", args.checkpoint, "--host", args.host, "--port", args.port, "--device", args.device,
                    "--mode", args.mode] + (["--kalman"] if args.kalman else []))
    else:
        summarize(Path(args.out))


if __name__ == "__main__":
    main()
