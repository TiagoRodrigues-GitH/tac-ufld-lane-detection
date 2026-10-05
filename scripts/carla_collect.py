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

Design (docs/CARLA.md, "Design"): pure geometry (``CameraModel``, ``sample_rows``, ``occluded_fraction``) has
no CARLA dependency and is unit-tested; ``lane_boundaries`` reads the road map; ``synchronous_world`` and
``Scene`` own the simulator state and always restore or destroy it; ``ClipWriter`` owns the file layout;
``collect_clips`` composes them.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import queue
import random
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

if TYPE_CHECKING:  # the CARLA client is needed to collect, not to import this module or test its geometry
    import carla

H_SAMPLES = tuple(range(160, 720, 10))
FRAMES_PER_CLIP = 20
FPS = 20
WARMUP_TICKS = 40                 # let the cars start moving before the first clip
MAX_ATTEMPTS_PER_CLIP = 4         # clips with fewer than MIN_LANES usable lanes are retried
MIN_LANES, MIN_POINTS_PER_LANE = 2, 5
SENSOR_TIMEOUT_S = 30.0
EGO_BLUEPRINT = "vehicle.lincoln.mkz_2020"
CAMERA_X_M, CAMERA_Z_M, CAMERA_PITCH_DEG = 1.0, 1.5, -3.0
# Semantic tags (CARLA >= 0.9.14) that hide the road: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle.
OCCLUDER_TAGS = np.array([12, 13, 14, 15, 16, 17, 18, 19], dtype=np.uint8)
WEATHERS = ("ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon", "MidRainyNoon", "HardRainNoon",
            "ClearSunset", "CloudySunset", "WetSunset", "SoftRainSunset", "ClearNight", "WetNight")


def load_carla() -> Any:
    """The CARLA client module, or a clear exit on machines without it."""
    try:
        import carla
    except ImportError as exc:
        raise SystemExit("CARLA client not installed: use the Python 3.12 venv with `pip install carla==0.9.16`") from exc
    return carla


# --------------------------------------------------------------------------- pure geometry (no CARLA)

@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera matching CARLA's RGB sensor attributes."""
    width: int = 1280
    height: int = 720
    fov_deg: float = 90.0

    def intrinsics(self) -> np.ndarray:
        f = self.width / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))
        return np.array([[f, 0.0, self.width / 2.0], [0.0, f, self.height / 2.0], [0.0, 0.0, 1.0]])

    def project(self, points: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
        """World points (N, 3) -> pixels (N, 2); NaN for points less than 0.5 m in front of the camera.

        ``world_to_camera`` is CARLA's 4x4 inverse transform (Unreal axes: x forward, y right, z up).
        """
        cam = world_to_camera @ np.c_[points, np.ones(len(points))].T
        xyz = np.stack([cam[1], -cam[2], cam[0]])  # -> x right, y down, z forward
        uv = (self.intrinsics() @ xyz)[:2] / np.where(xyz[2] > 0.5, xyz[2], np.nan)
        return uv.T


def sample_rows(uv: np.ndarray, width: int, h_samples: tuple[int, ...] = H_SAMPLES) -> list[int]:
    """x at each row along an image polyline ordered near -> far; -2 where absent or off-image.

    Only the first crossing from the near end counts: it is the visible part of the boundary.
    """
    xs = []
    for y in h_samples:
        x_at = -2
        for (x0, y0), (x1, y1) in itertools.pairwise(uv):
            if np.isnan([x0, y0, x1, y1]).any() or y0 == y1:
                continue
            if min(y0, y1) <= y <= max(y0, y1):
                x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                if 0 <= x < width:
                    x_at = round(x)
                break
        xs.append(x_at)
    return xs


def occluded_fraction(tags: np.ndarray, lanes: list[list[int]], h_samples: tuple[int, ...] = H_SAMPLES,
                      occluders: np.ndarray = OCCLUDER_TAGS) -> float:
    """Share of labelled lane points covered by an occluder in the semantic tag image (H, W)."""
    points = [(y, x) for xs in lanes for x, y in zip(xs, h_samples, strict=True) if x >= 0]
    if not points:
        return 0.0
    rows, cols = np.array(points).T
    return float(np.isin(tags[rows, cols], occluders).mean())


def usable_lanes(lanes: list[list[int]]) -> list[list[int]]:
    return [xs for xs in lanes if sum(x >= 0 for x in xs) >= MIN_POINTS_PER_LANE]


# --------------------------------------------------------------------------- CARLA road map

def _same_direction_driving(lane: carla.Waypoint | None, ego: carla.Waypoint, carla_mod: Any) -> bool:
    return lane is not None and lane.lane_type == carla_mod.LaneType.Driving and lane.lane_id * ego.lane_id > 0


def _walk_boundary(start: carla.Waypoint, side: float, length_m: float, step_m: float) -> np.ndarray:
    """3D points of one lane edge (side -1 left, +1 right) from ``start`` forward."""
    points, waypoint, travelled = [], start, 0.0
    while waypoint is not None and travelled <= length_m:
        t = waypoint.transform
        r = t.get_right_vector()
        offset = side * waypoint.lane_width / 2.0
        points.append([t.location.x + r.x * offset, t.location.y + r.y * offset, t.location.z + r.z * offset])
        ahead = waypoint.next(step_m)
        waypoint = ahead[0] if ahead else None
        travelled += step_m
    return np.asarray(points, dtype=np.float64)


def lane_boundaries(world_map: carla.Map, location: carla.Location, length_m: float = 80.0,
                    step_m: float = 1.0) -> list[np.ndarray]:
    """Ego-lane boundaries plus the outer boundaries of same-direction neighbours: up to 4, left to right."""
    carla_mod = load_carla()
    ego = world_map.get_waypoint(location, project_to_road=True, lane_type=carla_mod.LaneType.Driving)
    if ego is None:
        return []
    left, right = ego.get_left_lane(), ego.get_right_lane()
    lines = []
    if _same_direction_driving(left, ego, carla_mod):
        lines.append(_walk_boundary(left, -1.0, length_m, step_m))
    lines += [_walk_boundary(ego, -1.0, length_m, step_m), _walk_boundary(ego, +1.0, length_m, step_m)]
    if _same_direction_driving(right, ego, carla_mod):
        lines.append(_walk_boundary(right, +1.0, length_m, step_m))
    return lines


# --------------------------------------------------------------------------- simulator state

@contextmanager
def synchronous_world(client: carla.Client, town: str, seed: int) -> Iterator[tuple[carla.World, Any]]:
    """Load the town in synchronous mode at FPS; restore the previous settings on exit, even after errors."""
    world = client.get_world()
    if not world.get_map().name.endswith(town):
        world = client.load_world(town)
    settings = world.get_settings()
    # fixed_delta_seconds is None in asynchronous mode, which carla.WorldSettings(...) rejects: keep the values
    original = (settings.synchronous_mode, settings.fixed_delta_seconds, settings.no_rendering_mode)
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 1.0 / FPS
    world.apply_settings(settings)
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(seed)
    try:
        yield world, traffic_manager
    finally:
        traffic_manager.set_synchronous_mode(False)
        restore = world.get_settings()
        restore.synchronous_mode, restore.fixed_delta_seconds, restore.no_rendering_mode = original
        world.apply_settings(restore)


class Scene:
    """Ego car with an RGB and a semantic camera, plus NPC traffic; destroys everything it spawned on exit."""

    def __init__(self, client: carla.Client, world: carla.World, traffic_manager: Any, camera: CameraModel,
                 traffic: int, rng: random.Random) -> None:
        self._client, self._world, self._camera = client, world, camera
        self._actors: list[carla.Actor] = []
        self._sensors: dict[str, tuple[carla.Sensor, queue.Queue]] = {}
        try:
            self._populate(traffic_manager, traffic, rng)
        except BaseException:  # a failed spawn must not leave cars or cameras in the simulator
            self.close()
            raise

    def _populate(self, traffic_manager: Any, traffic: int, rng: random.Random) -> None:
        world = self._world
        library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        rng.shuffle(spawn_points)
        self.ego = self._spawn(library.find(EGO_BLUEPRINT), spawn_points[0])
        four_wheeled = [b for b in library.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
        self.npc_count = 0
        for point in spawn_points[1:traffic + 1]:
            npc = world.try_spawn_actor(rng.choice(four_wheeled), point)
            if npc:
                npc.set_autopilot(True, traffic_manager.get_port())
                self._actors.append(npc)
                self.npc_count += 1
        self.ego.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.ignore_lights_percentage(self.ego, 100.0)
        for name, kind in (("rgb", "sensor.camera.rgb"), ("sem", "sensor.camera.semantic_segmentation")):
            self._attach_camera(library, name, kind)

    def _spawn(self, blueprint: carla.ActorBlueprint, transform: carla.Transform) -> carla.Actor:
        actor = self._world.spawn_actor(blueprint, transform)
        self._actors.append(actor)
        return actor

    def _attach_camera(self, library: carla.BlueprintLibrary, name: str, kind: str) -> None:
        carla_mod = load_carla()
        blueprint = library.find(kind)
        blueprint.set_attribute("image_size_x", str(self._camera.width))
        blueprint.set_attribute("image_size_y", str(self._camera.height))
        blueprint.set_attribute("fov", str(self._camera.fov_deg))
        mount = carla_mod.Transform(carla_mod.Location(x=CAMERA_X_M, z=CAMERA_Z_M),
                                    carla_mod.Rotation(pitch=CAMERA_PITCH_DEG))
        sensor = self._world.spawn_actor(blueprint, mount, attach_to=self.ego)
        frames: queue.Queue = queue.Queue()
        sensor.listen(frames.put)
        self._sensors[name] = (sensor, frames)
        self._actors.append(sensor)

    @property
    def world_to_rgb_camera(self) -> np.ndarray:
        return np.array(self._sensors["rgb"][0].get_transform().get_inverse_matrix())

    def tick(self) -> dict[str, carla.Image]:
        """Advance one step and return each camera's image of that exact frame, by camera name."""
        frame = self._world.tick()
        images = {}
        for name, (_sensor, frames) in self._sensors.items():
            image = frames.get(timeout=SENSOR_TIMEOUT_S)
            while image.frame != frame:  # drop images of earlier frames
                image = frames.get(timeout=SENSOR_TIMEOUT_S)
            images[name] = image
        return images

    def close(self) -> None:
        carla_mod = load_carla()
        for sensor, _frames in self._sensors.values():
            sensor.stop()
        self._client.apply_batch([carla_mod.command.DestroyActor(a) for a in self._actors])
        self._sensors, self._actors = {}, []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------- output files

class ClipWriter:
    """One split in the TuSimple layout: clip folders, the label line of frame 20, and a meta line per clip."""

    def __init__(self, out: Path, split: str, drive: str) -> None:
        self.split_dir = out / ("train_set" if split == "train" else "test_set")
        self.label_path = self.split_dir / "label_data_carla.json" if split == "train" else out / "test_label.json"
        self.meta_path = out / f"{split}_meta.jsonl"
        self._drive = drive
        self._drive_dir = self.split_dir / "clips" / drive
        self._drive_dir.mkdir(parents=True, exist_ok=True)
        self._next = sum(1 for p in self._drive_dir.iterdir() if p.is_dir())  # continue numbering across runs

    def new_clip(self) -> Path:
        clip_dir = self._drive_dir / f"{self._next:05d}"
        clip_dir.mkdir(exist_ok=True)
        return clip_dir

    def keep(self, clip_dir: Path, lanes: list[list[int]], meta: dict[str, Any]) -> str:
        raw_file = f"clips/{self._drive}/{clip_dir.name}/{FRAMES_PER_CLIP}.jpg"
        self._append(self.label_path, {"lanes": lanes, "h_samples": list(H_SAMPLES), "raw_file": raw_file})
        self._append(self.meta_path, {"raw_file": raw_file, **meta})
        self._next += 1
        return raw_file

    @staticmethod
    def discard(clip_dir: Path) -> None:
        shutil.rmtree(clip_dir)

    @staticmethod
    def _append(path: Path, record: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- collection

def semantic_tags(image: carla.Image, camera: CameraModel) -> np.ndarray:
    """Tag image (H, W) from CARLA's BGRA semantic image (the tag is in the R channel)."""
    return np.frombuffer(image.raw_data, dtype=np.uint8).reshape(camera.height, camera.width, 4)[:, :, 2]


def label_lanes(world_map: carla.Map, scene: Scene, camera: CameraModel) -> list[list[int]]:
    world_to_camera = scene.world_to_rgb_camera
    lines = lane_boundaries(world_map, scene.ego.get_location())
    return usable_lanes([sample_rows(camera.project(line, world_to_camera), camera.width)
                         for line in lines if len(line) > 1])


def collect_clips(world: carla.World, scene: Scene, writer: ClipWriter, camera: CameraModel, clips: int,
                  gap_s: float, town: str, seed: int) -> tuple[int, int]:
    """Drive, record 20-frame clips and label their last frame; returns (saved, attempts)."""
    carla_mod = load_carla()
    world_map = world.get_map()
    for _ in range(WARMUP_TICKS):
        scene.tick()
    saved = attempts = 0
    while saved < clips and attempts < clips * MAX_ATTEMPTS_PER_CLIP:
        attempts += 1
        weather = WEATHERS[(saved + seed) % len(WEATHERS)]
        world.set_weather(getattr(carla_mod.WeatherParameters, weather))
        for _ in range(round(gap_s * FPS)):
            scene.tick()
        clip_dir = writer.new_clip()
        for index in range(1, FRAMES_PER_CLIP + 1):
            images = scene.tick()
            images["rgb"].save_to_disk(str(clip_dir / f"{index}.jpg"))
        lanes = label_lanes(world_map, scene, camera)  # camera pose of frame 20, the labelled one
        if len(lanes) < MIN_LANES:
            writer.discard(clip_dir)
            continue
        occluded = occluded_fraction(semantic_tags(images["sem"], camera), lanes)
        writer.keep(clip_dir, lanes, {"town": town, "weather": weather, "traffic": scene.npc_count,
                                      "occluded_fraction": round(occluded, 4),
                                      "speed_kmh": round(3.6 * scene.ego.get_velocity().length(), 1)})
        saved += 1
        print(f"clip {clip_dir.name}: {len(lanes)} lanes, weather {weather}, occluded {occluded:.0%}", flush=True)
    return saved, attempts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    carla_mod = load_carla()
    client = carla_mod.Client(args.host, args.port)
    client.set_timeout(60.0)
    camera = CameraModel()
    writer = ClipWriter(args.out, args.split, drive=f"{args.town}-s{args.seed}")
    with synchronous_world(client, args.town, args.seed) as (world, traffic_manager), \
            Scene(client, world, traffic_manager, camera, args.traffic, random.Random(args.seed)) as scene:
        saved, attempts = collect_clips(world, scene, writer, camera, args.clips, args.gap_s, args.town, args.seed)
    print(f"saved {saved} clips in {writer.split_dir} (attempts {attempts})")


if __name__ == "__main__":
    main()
