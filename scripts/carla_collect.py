"""Collect lane-detection clips from the CARLA simulator in TuSimple layout.

Writes what ``tac_ufld.data.tusimple.TuSimpleAdapter`` reads, so CARLA data can be trained and
evaluated with ``--dataset tusimple`` (point TUSIMPLE_ROOT at the output folder)::

    <out>/train_set/clips/<drive>/<clip>/1.jpg .. 20.jpg      (1280x720, 20 frames at 20 Hz)
    <out>/train_set/label_data_carla.json                      (one JSON line per clip, frame 20)
    <out>/test_set/clips/...  and  <out>/test_label.json        (with --split test)
    <out>/<split>_meta.jsonl                                   (town, weather, traffic, occlusion)

Labels are geometric: lane boundaries come from the CARLA road map (waypoints), are projected
into the camera and sampled at the TuSimple rows (h_samples 160..710, step 10; x = -2 where the
boundary is absent or outside the image). Boundaries hidden behind vehicles are still labelled,
as TuSimple annotators did; ``occluded_fraction`` in the meta file gives, per clip, the share of
labelled points covered by a vehicle or person in the semantic camera, so occluded clips can be
selected for the occlusion study.

Needs the CARLA server running (packaged 0.9.16), and the CARLA Python client (Python 3.10-3.12)::

    C:\\CARLA\\CARLA_0.9.16\\CarlaUE4.exe -RenderOffScreen -quality-level=Low
    C:\\CARLA\\venv312\\Scripts\\python.exe scripts\\carla_collect.py --town Town04 --clips 50 --out D:\\carla_lanes

Use different towns for train and test (``--split test``) to keep the test scene-disjoint.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import random
import time
from pathlib import Path

import numpy as np

try:
    import carla
except ImportError as exc:  # pragma: no cover - only on machines with the CARLA client
    raise SystemExit("CARLA client not installed: use the Python 3.12 venv with `pip install carla==0.9.16`") from exc

W, H, FOV = 1280, 720, 90.0
H_SAMPLES = list(range(160, 720, 10))
FRAMES_PER_CLIP = 20
FPS = 20
# Semantic tags (CARLA >= 0.9.14) that hide the road: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle.
OCCLUDERS = np.array([12, 13, 14, 15, 16, 17, 18, 19], dtype=np.uint8)
WEATHERS = ["ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon", "MidRainyNoon", "HardRainNoon",
            "ClearSunset", "CloudySunset", "WetSunset", "SoftRainSunset", "ClearNight", "WetNight"]


def intrinsics() -> np.ndarray:
    f = W / (2.0 * math.tan(math.radians(FOV) / 2.0))
    return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])


def project(points: np.ndarray, camera: carla.Actor, k: np.ndarray) -> np.ndarray:
    """World points (N, 3) -> image points (N, 2); NaN for points behind the camera."""
    world_to_cam = np.array(camera.get_transform().get_inverse_matrix())
    homo = np.c_[points, np.ones(len(points))].T
    cam = world_to_cam @ homo  # UE axes: x forward, y right, z up
    xyz = np.stack([cam[1], -cam[2], cam[0]])  # -> x right, y down, z forward
    uv = (k @ xyz)[:2] / np.where(xyz[2] > 0.5, xyz[2], np.nan)
    return uv.T


def boundary_polylines(world_map: carla.Map, location: carla.Location, length_m: float = 80.0,
                       step_m: float = 1.0) -> list[np.ndarray]:
    """3D polylines of the ego lane boundaries and of the outer boundaries of the adjacent lanes
    (same direction, driving lanes), up to 4 lines, ordered left to right."""
    ego = world_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if ego is None:
        return []
    lanes = [ego]
    left, right = ego.get_left_lane(), ego.get_right_lane()
    def same_dir(wp) -> bool:
        return wp is not None and wp.lane_type == carla.LaneType.Driving and wp.lane_id * ego.lane_id > 0

    if same_dir(left):
        lanes.insert(0, left)
    if same_dir(right):
        lanes.append(right)

    def walk(start: carla.Waypoint, side: float) -> np.ndarray:
        pts, wp, travelled = [], start, 0.0
        while wp is not None and travelled <= length_m:
            t = wp.transform
            r = t.get_right_vector()
            off = side * wp.lane_width / 2.0
            pts.append([t.location.x + r.x * off, t.location.y + r.y * off, t.location.z + r.z * off])
            nxt = wp.next(step_m)
            wp = nxt[0] if nxt else None
            travelled += step_m
        return np.asarray(pts, dtype=np.float64)

    lines = []
    if lanes[0] is not ego:  # outer boundary of the left neighbour
        lines.append(walk(lanes[0], -1.0))
    lines += [walk(ego, -1.0), walk(ego, +1.0)]
    if lanes[-1] is not ego:  # outer boundary of the right neighbour
        lines.append(walk(lanes[-1], +1.0))
    return lines


def sample_rows(uv: np.ndarray) -> list[int]:
    """x at each h_sample row along an image polyline (near -> far); -2 if absent or off-image."""
    xs = []
    for y in H_SAMPLES:
        x_at = -2
        for (x0, y0), (x1, y1) in zip(uv[:-1], uv[1:], strict=True):
            if np.isnan([x0, y0, x1, y1]).any() or y0 == y1:
                continue
            if min(y0, y1) <= y <= max(y0, y1):
                x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                if 0 <= x < W:
                    x_at = int(round(x))
                break  # first crossing from the near end: the visible part of the boundary
        xs.append(x_at)
    return xs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--town", default="Town04")
    ap.add_argument("--clips", type=int, default=20)
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--traffic", type=int, default=40, help="NPC vehicles (occluders)")
    ap.add_argument("--gap-s", type=float, default=3.0, help="driving time between clips")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if not world.get_map().name.endswith(args.town):
        world = client.load_world(args.town)
    world_map = world.get_map()
    settings = world.get_settings()
    # fixed_delta_seconds is None in asynchronous mode, which carla.WorldSettings(...) rejects: keep the values
    original = (settings.synchronous_mode, settings.fixed_delta_seconds, settings.no_rendering_mode)
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 1.0 / FPS
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)

    split_dir = args.out / ("train_set" if args.split == "train" else "test_set")
    label_path = split_dir / "label_data_carla.json" if args.split == "train" else args.out / "test_label.json"
    meta_path = args.out / f"{args.split}_meta.jsonl"
    split_dir.mkdir(parents=True, exist_ok=True)
    k = intrinsics()
    actors = []
    try:
        library = world.get_blueprint_library()
        spawns = world_map.get_spawn_points()
        random.shuffle(spawns)
        ego = world.spawn_actor(random.choice(library.filter("vehicle.lincoln.mkz_2020")), spawns[0])
        actors.append(ego)
        for sp in spawns[1:args.traffic + 1]:
            bp = random.choice([b for b in library.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4])
            npc = world.try_spawn_actor(bp, sp)
            if npc:
                npc.set_autopilot(True, tm.get_port())
                actors.append(npc)
        ego.set_autopilot(True, tm.get_port())
        tm.ignore_lights_percentage(ego, 100.0)

        mount = carla.Transform(carla.Location(x=1.0, z=1.5), carla.Rotation(pitch=-3.0))
        cams = {}
        for name, kind in (("rgb", "sensor.camera.rgb"), ("sem", "sensor.camera.semantic_segmentation")):
            bp = library.find(kind)
            bp.set_attribute("image_size_x", str(W))
            bp.set_attribute("image_size_y", str(H))
            bp.set_attribute("fov", str(FOV))
            cam = world.spawn_actor(bp, mount, attach_to=ego)
            q: queue.Queue = queue.Queue()
            cam.listen(q.put)
            cams[name] = (cam, q)
            actors.append(cam)

        def tick() -> tuple[carla.Image, carla.Image]:
            frame = world.tick()
            out = []
            for _cam, q in cams.values():
                while True:
                    img = q.get(timeout=30.0)
                    if img.frame == frame:
                        out.append(img)
                        break
            return out[0], out[1]

        drive = f"{args.town}-s{args.seed}"
        existing = len(list((split_dir / "clips" / drive).glob("*"))) if (split_dir / "clips" / drive).exists() else 0
        saved, attempts = 0, 0
        for _ in range(40):  # let the cars start moving
            tick()
        while saved < args.clips and attempts < args.clips * 4:
            attempts += 1
            weather = WEATHERS[(saved + args.seed) % len(WEATHERS)]
            world.set_weather(getattr(carla.WeatherParameters, weather))
            for _ in range(int(args.gap_s * FPS)):
                tick()
            clip = f"{existing + saved:05d}"
            clip_dir = split_dir / "clips" / drive / clip
            clip_dir.mkdir(parents=True, exist_ok=True)
            for i in range(1, FRAMES_PER_CLIP + 1):
                rgb, sem = tick()
                rgb.save_to_disk(str(clip_dir / f"{i}.jpg"))
            # label the last frame (20) with the camera pose of that frame
            lines = boundary_polylines(world_map, ego.get_location())
            lanes = [sample_rows(project(pl, cams["rgb"][0], k)) for pl in lines if len(pl) > 1]
            lanes = [xs for xs in lanes if sum(x >= 0 for x in xs) >= 5]
            if len(lanes) < 2:
                for f in clip_dir.iterdir():
                    f.unlink()
                clip_dir.rmdir()
                continue
            tags = np.frombuffer(sem.raw_data, dtype=np.uint8).reshape(H, W, 4)[:, :, 2]  # BGRA: tag in R
            pts = [(x, y) for xs in lanes for x, y in zip(xs, H_SAMPLES, strict=True) if x >= 0]
            occluded = float(np.mean([np.isin(tags[y, x], OCCLUDERS) for x, y in pts]))
            raw = f"clips/{drive}/{clip}/20.jpg"
            with open(label_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"lanes": lanes, "h_samples": H_SAMPLES, "raw_file": raw}) + "\n")
            with open(meta_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"raw_file": raw, "town": args.town, "weather": weather,
                                     "traffic": len(actors) - 3, "occluded_fraction": round(occluded, 4),
                                     "speed_kmh": round(3.6 * ego.get_velocity().length(), 1)}) + "\n")
            saved += 1
            print(f"clip {clip}: {len(lanes)} lanes, weather {weather}, occluded {occluded:.0%}", flush=True)
        print(f"saved {saved} clips in {split_dir} (attempts {attempts})")
    finally:
        for cam, _ in cams.values() if "cams" in locals() else []:
            cam.stop()
        client.apply_batch([carla.command.DestroyActor(a) for a in actors])
        tm.set_synchronous_mode(False)
        restore = world.get_settings()
        restore.synchronous_mode, restore.fixed_delta_seconds, restore.no_rendering_mode = original
        world.apply_settings(restore)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
