"""Hardware-oriented benchmark and real-time budget simulation.

Two clearly separated parts:

1. **Measured** (``measure``): batch-1 streaming latency on THIS machine,
   split into preprocessing / model / post-processing, for a given backend
   (PyTorch FP32/FP16, ONNX Runtime, TensorRT) and inference mode
   (single-frame, temporal recompute, temporal cached), plus peak GPU memory.
   Optional batch > 1 throughput (several streams batched, non-streaming).

2. **Simulated** (``simulate``): a discrete-event simulation of a camera at
   ``target_fps`` with random sensor drops, feeding one inference worker whose
   per-frame service times are resampled from the measured latencies, scaled
   by an explicit ``slowdown`` factor. It answers "would this configuration
   meet the declared budgets IF the device were ``slowdown`` x slower than the
   measuring machine?". It does NOT establish Jetson performance: only a
   measurement on the device does (docs/DEPLOYMENT.md).

The simulation also reports what skipped frames do to temporal models: a
frame whose history frames were never processed falls back to newer frames
(the training rule), so an overloaded worker silently degrades the temporal
context. ``history_fallback_rate`` makes that visible.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from tac_ufld.config import ConfigError


@dataclass
class BudgetProfile:
    name: str = "custom"
    description: str = ""
    target_fps: float = 30.0
    max_latency_ms: float = 50.0          # camera capture -> lanes available
    memory_budget_mb: float = 1024.0      # GPU memory available to the model
    input_height: int = 480               # CAMERA frame size (the model input size is fixed by the checkpoint)
    input_width: int = 640
    batch_size: int = 1
    precision: str = "fp32"
    drop_rate: float = 0.0                # simulated sensor/transport frame drops
    slowdown: float = 1.0                 # ASSUMED device latency / measured latency
    slowdown_source: str = "none (desktop)"
    frames: int = 600                     # simulated frames
    policy: str = "latest"                # latest: skip stale frames | fifo: process every frame in order
    seed: int = 0
    warmup: int = 10
    measure_frames: int = 100

    def validate(self) -> None:
        if self.target_fps <= 0 or self.max_latency_ms <= 0 or self.memory_budget_mb <= 0:
            raise ConfigError("target_fps, max_latency_ms and memory_budget_mb must be > 0")
        if not 0 <= self.drop_rate < 1:
            raise ConfigError("drop_rate must be in [0, 1)")
        if self.slowdown <= 0 or self.batch_size < 1:
            raise ConfigError("slowdown must be > 0 and batch_size >= 1")
        if self.policy not in ("latest", "fifo"):
            raise ConfigError("policy must be latest|fifo")
        if self.precision not in ("fp32", "fp16", "int8"):
            raise ConfigError("precision must be fp32|fp16|int8")


def load_profile(path: str | Path | None, **overrides) -> BudgetProfile:
    data = {}
    if path:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(data) - set(BudgetProfile.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"unknown profile key(s) {sorted(unknown)} in {path}")
    data.update({k: v for k, v in overrides.items() if v is not None})
    profile = BudgetProfile(**data)
    profile.validate()
    return profile


# ------------------------------------------------------------------ measure


@dataclass
class Measurement:
    backend: str
    mode: str
    variant: str
    device: str
    samples_ms: dict[str, list[float]] = field(default_factory=dict)
    peak_gpu_memory_mb: float = float("nan")
    batch_throughput_fps: float | None = None

    def summary(self) -> dict:
        out = {"backend": self.backend, "mode": self.mode, "variant": self.variant, "device": self.device,
               "peak_gpu_memory_mb": self.peak_gpu_memory_mb, "batch_throughput_fps": self.batch_throughput_fps}
        for stage, values in self.samples_ms.items():
            arr = np.asarray(values)
            out[f"{stage}_mean"] = float(arr.mean())
            out[f"{stage}_p50"] = float(np.percentile(arr, 50))
            out[f"{stage}_p95"] = float(np.percentile(arr, 95))
        out["fps_total_mean"] = 1000.0 / out["total_ms_mean"] if out.get("total_ms_mean") else float("nan")
        return out


def device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    import platform

    return f"CPU ({platform.processor() or platform.machine()})"


def measure(detector, frames: list[np.ndarray], profile: BudgetProfile, label_mode: str) -> Measurement:
    """Stream ``frames`` (cycled) through ``detector`` and record latencies."""
    device = getattr(detector.backend, "device", "cuda" if torch.cuda.is_available() else "cpu")
    cuda = str(device).startswith("cuda") or "tensorrt" in detector.backend.name or "cuda" in detector.backend.name
    stream = "bench"
    detector.reset_stream(stream)
    n = profile.warmup + profile.measure_frames
    if cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    samples: dict[str, list[float]] = {}
    for i in range(n):
        res = detector.infer_frame(frames[i % len(frames)], stream, frame_index=i)
        if i >= profile.warmup:
            for k, v in res.latency_ms.items():
                samples.setdefault(k, []).append(v)
    peak = float("nan")
    if cuda and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 2**20
    m = Measurement(detector.backend.name, label_mode, detector.loaded.variant, device_name(str(device)), samples, peak)
    if profile.batch_size > 1 and hasattr(detector.backend, "model"):
        m.batch_throughput_fps = _batch_throughput(detector, profile)
    return m


@torch.no_grad()
def _batch_throughput(detector, profile: BudgetProfile, runs: int = 20) -> float:
    """Frames/s when ``batch_size`` streams are processed together (full clip
    forward, no caching): an upper bound for multi-camera batching."""
    loaded = detector.loaded
    c = loaded.card["input"]["channels"]
    d = loaded.cfg.data
    x = torch.randn(profile.batch_size, loaded.num_frames, c, d.img_h, d.img_w,
                    device=detector.backend.device, dtype=detector.backend.dtype)
    model = detector.backend.model
    import time

    for _ in range(3):
        model(x)
    detector.backend.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    detector.backend.synchronize()
    return profile.batch_size * runs / (time.perf_counter() - t0)


# ------------------------------------------------------------------ simulate


def simulate(service_ms: list[float], profile: BudgetProfile, num_frames_model: int = 1,
             temporal_step: int = 1) -> dict:
    """Camera -> single worker discrete-event simulation (seeded)."""
    rng = random.Random(profile.seed)
    period = 1000.0 / profile.target_fps
    service = [s * profile.slowdown for s in service_ms]
    arrivals = [(i, i * period) for i in range(profile.frames) if rng.random() >= profile.drop_rate]
    free_at, processed, skipped = 0.0, [], 0
    latencies = []
    queue: list[tuple[int, float]] = []
    pending = list(arrivals)
    while pending or queue:
        # frames that have arrived by the time the worker is free
        while pending and pending[0][1] <= free_at:
            queue.append(pending.pop(0))
        if not queue:
            queue.append(pending.pop(0))
        if profile.policy == "latest" and len(queue) > 1:
            skipped += len(queue) - 1
            queue = queue[-1:]
        idx, t_arr = queue.pop(0)
        start = max(free_at, t_arr)
        free_at = start + rng.choice(service)
        processed.append(idx)
        latencies.append(free_at - t_arr)
    lat = np.asarray(latencies) if latencies else np.asarray([np.nan])
    done = set(processed)
    fallbacks = total_hist = 0
    if num_frames_model > 1:
        for idx in processed:
            for k in range(1, num_frames_model):
                h = idx - k * temporal_step
                if h >= 0:
                    total_hist += 1
                    fallbacks += int(h not in done)
    duration_s = profile.frames / profile.target_fps
    result = {
        "simulated_frames": profile.frames, "sensor_dropped": profile.frames - len(arrivals),
        "skipped_busy": skipped, "processed": len(processed),
        "processed_fps": len(processed) / duration_s,
        "latency_p50_ms": float(np.nanpercentile(lat, 50)), "latency_p95_ms": float(np.nanpercentile(lat, 95)),
        "latency_p99_ms": float(np.nanpercentile(lat, 99)),
        "deadline_miss_rate": float(np.mean(lat > profile.max_latency_ms)),
        "history_fallback_rate": fallbacks / total_hist if total_hist else 0.0,
        "service_ms_mean_scaled": float(np.mean(service)),
    }
    expected = profile.target_fps * (1 - profile.drop_rate)
    result["meets_fps"] = result["processed_fps"] >= 0.98 * expected
    result["meets_latency_p95"] = result["latency_p95_ms"] <= profile.max_latency_ms
    return result


def evaluate_budget(measurement: Measurement, profile: BudgetProfile, num_frames_model: int,
                    temporal_step: int) -> dict:
    sim = simulate(measurement.samples_ms["total_ms"], profile, num_frames_model, temporal_step)
    mem = measurement.peak_gpu_memory_mb
    sim["meets_memory"] = bool(np.isnan(mem) or mem <= profile.memory_budget_mb)
    sim["memory_note"] = "not measured (CPU backend)" if np.isnan(mem) else f"{mem:.0f} MB peak allocated by PyTorch"
    sim["all_budgets_met"] = bool(sim["meets_fps"] and sim["meets_latency_p95"] and sim["meets_memory"])
    return sim


def write_report(path: Path, profile: BudgetProfile, rows: list[dict]) -> None:
    """``rows``: {"measured": Measurement.summary(), "simulated": evaluate_budget(...)}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.with_suffix(".json")).write_text(json.dumps({"profile": asdict(profile), "results": rows}, indent=2,
                                                      default=str), encoding="utf-8")
    md = [f"# Benchmark: profile `{profile.name}`\n", profile.description + "\n" if profile.description else "",
          "## Measured on this machine (batch 1, streaming)\n",
          "| variant | backend | mode | device | preprocess ms | model ms | postprocess ms | total ms (p95) | FPS | peak GPU MB |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        m = r["measured"]
        md.append(f"| {m['variant']} | {m['backend']} | {m['mode']} | {m['device']} | {m['preprocess_ms_mean']:.2f} | "
                  f"{m['model_ms_mean']:.2f} | {m['postprocess_ms_mean']:.2f} | {m['total_ms_mean']:.2f} "
                  f"({m['total_ms_p95']:.2f}) | {m['fps_total_mean']:.1f} | {m['peak_gpu_memory_mb']:.0f} |")
    md += ["\n## Simulated budget check (NOT a device measurement)\n",
           f"Camera {profile.input_width}x{profile.input_height} at {profile.target_fps} FPS, sensor drop rate "
           f"{profile.drop_rate}, policy `{profile.policy}`, latency budget {profile.max_latency_ms} ms, memory budget "
           f"{profile.memory_budget_mb} MB. Service times = measured x **{profile.slowdown}** "
           f"(assumption: {profile.slowdown_source}).\n",
           "| variant | mode | processed FPS | skipped (busy) | p95 latency ms | deadline misses | history fallbacks | budgets met |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        m, s = r["measured"], r["simulated"]
        md.append(f"| {m['variant']} | {m['mode']} | {s['processed_fps']:.1f} | {s['skipped_busy']} | "
                  f"{s['latency_p95_ms']:.1f} | {100 * s['deadline_miss_rate']:.1f}% | "
                  f"{100 * s['history_fallback_rate']:.1f}% | {'yes' if s['all_budgets_met'] else 'NO'} |")
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
