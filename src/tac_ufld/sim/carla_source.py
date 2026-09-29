"""Optional CARLA integration: camera frames from a running CARLA server fed
to the streaming detector, for QUALITATIVE checks only.

Status: NOT validated (no CARLA server was available during development).
CARLA frames have no lane ground truth here and a different domain from
ELAS/CULane, so they are written to ``results/simulator/`` and never enter
dataset evaluation or reports.

Setup (see docs/DEPLOYMENT.md): start CARLA 0.9.x (``CarlaUE4.exe`` /
``CarlaUE4.sh``), ``pip install carla==<server version>`` in this
environment, then ``python -m tac_ufld carla-demo --checkpoint <ckpt>``.
"""

from __future__ import annotations

import json
import queue
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def carla_frames(host: str, port: int, town: str, n_frames: int, width: int = 640, height: int = 480,
                 fps: float = 20.0, fov: float = 90.0) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame index, timestamp s, RGB array) from an ego vehicle on autopilot."""
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError("the 'carla' Python package is not installed; install the version matching the "
                           "CARLA server (pip install carla==0.9.x)") from exc
    client = carla.Client(host, port)
    client.set_timeout(20.0)
    world = client.get_world()
    if town and not world.get_map().name.endswith(town):
        world = client.load_world(town)
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 1.0 / fps
    world.apply_settings(settings)
    actors = []
    try:
        bp = world.get_blueprint_library()
        vehicle = world.spawn_actor(bp.find("vehicle.tesla.model3"), world.get_map().get_spawn_points()[0])
        actors.append(vehicle)
        vehicle.set_autopilot(True)
        cam_bp = bp.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(width))
        cam_bp.set_attribute("image_size_y", str(height))
        cam_bp.set_attribute("fov", str(fov))
        camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=1.4)), attach_to=vehicle)
        actors.append(camera)
        frames: queue.Queue = queue.Queue()
        camera.listen(frames.put)
        for i in range(n_frames):
            world.tick()
            image = frames.get(timeout=10.0)
            bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
            yield i, float(image.timestamp), np.ascontiguousarray(bgra[:, :, 2::-1])
    finally:
        for actor in reversed(actors):
            actor.destroy()
        world.apply_settings(original)


def run_carla_demo(args) -> int:
    from tac_ufld.inference.loading import load_model
    from tac_ufld.inference.streaming import StreamingLaneDetector
    from tac_ufld.visualization.overlays import VideoSink, draw_prediction

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = load_model(args.checkpoint, device)
    det = StreamingLaneDetector(loaded, nominal_fps=20.0)
    out = Path(args.out) / f"carla_{time.strftime('%Y%m%d_%H%M%S')}"
    sink = VideoSink(out / "overlay.mp4", fps=20)
    latencies = []
    for i, ts, rgb in carla_frames(args.host, args.port, args.town, args.frames):
        res = det.infer_frame(rgb, "carla", timestamp=ts)
        sink.write(draw_prediction(rgb, res, f"CARLA {args.town} - {loaded.card['display_name']}"))
        latencies.append(res.latency_ms["total_ms"])
    sink.close()
    summary = {"frames": len(latencies), "mean_total_ms": float(np.mean(latencies)) if latencies else None,
               "note": "qualitative only: CARLA frames have no lane ground truth in this pipeline"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0
