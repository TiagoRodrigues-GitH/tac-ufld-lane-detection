"""Collect CARLA driving scenes in the ELAS layout, seen through the closed-loop camera, for fine-tuning.

Why ELAS and not the TuSimple layout of ``scripts/carla_collect.py``: the light models (``lite_tsm``, ``lite_v05``)
have the ELAS head (2 slots: ego-left, ego-right; 640x480 frames; row anchors over 59-99.5 % of the height),
which the ELAS adapter reads unchanged. The TuSimple adapter has 4 slots, requires 1280x720 and labels one frame per
20-frame clip. Here every frame of a continuous 30 Hz drive is saved (temporal history, as in ELAS) and every
``--label-every``-th frame gets ground truth::

    <out>/<scene>/config.xml                  frame range, 640x480, region of interest (ELAS schema)
    <out>/<scene>/groundtruth.xml             per labelled frame: ego-left / ego-right p1..p4
    <out>/<scene>/images/images/lane_<i>.jpg  every frame
    <out>/<scene>/meta.json                   town, weather, camera, traffic, labelled-frame counts
    <out>/_overlays/<scene>_<i>.jpg           label check (with --overlay-every)

Labels follow the ELAS conventions (``tac_ufld.data.elas``): p1..p4 at 0, 1/4, 1/2 and 1 of the ROI height
(rows 320, 360, 400, 480 here), nan where the boundary is outside the image. They come from the CARLA road map
projected with the exact camera pose, but only where paint exists: a side whose lane marking is not painted
(``LaneMarkingType`` NONE / Grass / Curb / Other) over the labelled range is written as absent, and frames on or
just before a junction get no label at all (the map draws boundaries through junctions where nothing is painted).
Boundaries hidden by vehicles stay labelled.

The camera is the closed-loop rig (``tac_ufld.sim.lka_loop.CameraRig``: 640x480, FOV 70, 2.0 m ahead, 1.3 m high,
horizon at 60 % of the height) with a small random change per scene (FOV, height, horizon, yaw), so the model does
not learn one exact mount. The ego car drives on the Traffic Manager autopilot with a random lateral offset in its
lane (re-drawn every few seconds) and occasional lane changes, so off-centre views are in the data.

Keep the closed-loop test towns (Town04, Town06) out of the training data::

    $py = "C:\\CARLA\\venv312\\Scripts\\python.exe"
    & $py scripts\\carla_collect_elas.py --out C:\\CARLA\\lane_data\\carla_elas --town Town03 --scenes 6 --seed 1
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import random
import sys
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from carla_collect import WEATHERS  # noqa: E402
from tac_ufld.data.elas import ELAS_ROW_FRACTIONS  # noqa: E402
from tac_ufld.sim.lka_loop import CameraRig, load_carla  # noqa: E402

ROI_Y, ROI_H = 320, 160                   # labelled band: rows 320..480 (about 3-19 m ahead)
PAINTED = ("Broken", "Solid", "SolidSolid", "SolidBroken", "BrokenSolid", "BrokenBroken", "BottsDots")
EGO_BLUEPRINT = "vehicle.lincoln.mkz_2020"
SENSOR_TIMEOUT_S = 60.0
TEST_TOWNS = ("Town04", "Town06")


# --------------------------------------------------------------------------- labels (pure)

def row_crossings(uv: np.ndarray, rows: np.ndarray, width: int) -> np.ndarray:
    """x where an image polyline ordered near -> far first crosses each row; nan if never or off-image."""
    out = np.full(len(rows), np.nan)
    for k, y in enumerate(rows):
        for (x0, y0), (x1, y1) in zip(uv[:-1], uv[1:]):
            if not np.isfinite([x0, y0, x1, y1]).all() or y0 == y1:
                continue
            if min(y0, y1) <= y <= max(y0, y1):
                x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                out[k] = x if 0 <= x < width else np.nan
                break
    return out


def side_xml(name: str, xs: np.ndarray | None) -> str:
    """``<left><p1>..</p1>..<p4>..</p4></left>``; an absent side is written with nan points."""
    values = xs if xs is not None else np.full(len(ELAS_ROW_FRACTIONS), np.nan)
    points = "".join(f"<p{k + 1}>{'nan' if not np.isfinite(x) else f'{x:.1f}'}</p{k + 1}>"
                     for k, x in enumerate(values))
    return f"<{name}>{points}</{name}>"


def config_xml(scene: str, n_frames: int, width: int, height: int) -> str:
    return (f'<config><dataset id="{scene}" name="{scene}" path="{scene}/images/">'
            f'<frame_sequence start="0" end="{n_frames - 1}"></frame_sequence>'
            f'<frame_size width="{width}" height="{height}"></frame_size>'
            f'<region_of_interest x="0" y="{ROI_Y}" width="{width}" height="{ROI_H}"></region_of_interest>'
            f'<ipm_points top_left="0" top_right="0" bottom_right="0" bottom_left="0"></ipm_points>'
            f"</dataset></config>")


def jittered_rig(base: CameraRig, rng: random.Random) -> tuple[CameraRig, float]:
    """Per-scene camera: FOV +-4 deg, height +-0.15 m, horizon +-0.03 H; plus a yaw misalignment (deg)."""
    rig = replace(base, fov_deg=base.fov_deg + rng.uniform(-4, 4), z_m=base.z_m + rng.uniform(-0.15, 0.15),
                  horizon_frac=base.horizon_frac + rng.uniform(-0.03, 0.03))
    return rig, rng.uniform(-1.0, 1.0)


# --------------------------------------------------------------------------- CARLA

@contextmanager
def synchronous_world(client: Any, town: str, fps: float, seed: int, map_layers: str = "all"):
    """Town in synchronous mode with the Traffic Manager in lock-step; settings restored on exit. ``minimal``
    loads ``<town>_Opt`` without buildings, foliage or props (roads and markings always load): on this PC the
    full Town03 and Town06 crash the server while loading."""
    carla_mod = load_carla()
    name = f"{town}_Opt" if map_layers == "minimal" and not town.endswith("_Opt") else town
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != name:
        world = (client.load_world(name, map_layers=carla_mod.MapLayer.NONE) if map_layers == "minimal"
                 else client.load_world(name))
    settings = world.get_settings()
    original = (settings.synchronous_mode, settings.fixed_delta_seconds, settings.no_rendering_mode)
    settings.synchronous_mode, settings.fixed_delta_seconds, settings.no_rendering_mode = True, 1.0 / fps, False
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)
    try:
        yield world, tm
    finally:
        tm.set_synchronous_mode(False)
        restore = world.get_settings()
        restore.synchronous_mode, restore.fixed_delta_seconds, restore.no_rendering_mode = original
        world.apply_settings(restore)


def boundary(wp: Any, side: int, length_m: float, step_m: float = 1.0) -> tuple[np.ndarray, list[str], bool]:
    """3D points of one edge (-1 left, +1 right) of ``wp``'s lane, its marking types, and whether a junction is
    on the way."""
    points, marks, junction, cur = [], [], False, wp
    for _ in range(int(length_m / step_m)):
        t, half = cur.transform, cur.lane_width / 2.0
        c, r = t.location, t.get_right_vector()
        points.append((c.x + side * r.x * half, c.y + side * r.y * half, c.z + side * r.z * half))
        marking = cur.left_lane_marking if side < 0 else cur.right_lane_marking
        marks.append(str(marking.type))
        junction |= cur.is_junction
        ahead = cur.next(step_m)
        if not ahead:
            break
        cur = ahead[0]
    return np.asarray(points), marks, junction


def frame_label(world_map: Any, ego: Any, camera: Any, rig: CameraRig, rows: np.ndarray, carla_mod: Any):
    """(left xs, right xs) at ``rows`` or None for a frame without a usable label (junction ahead)."""
    wp = world_map.get_waypoint(ego.get_location(), project_to_road=True, lane_type=carla_mod.LaneType.Driving)
    if wp is None or wp.is_junction:
        return None
    w2c = np.array(camera.get_transform().get_inverse_matrix())
    near, far = rig.ground_distance(rows[-1] - 1) + rig.x_m, rig.ground_distance(rows[0]) + rig.x_m
    sides = []
    for side in (-1, 1):
        points, marks, junction = boundary(wp, side, far + 5.0)
        if junction:
            return None
        labelled = marks[int(near):int(far) + 1] or marks
        painted = np.mean([m.split(".")[-1] in PAINTED for m in labelled])
        xs = row_crossings(rig.project(points, w2c), rows, rig.width) if painted >= 0.5 else None
        sides.append(xs if xs is not None and np.isfinite(xs).sum() >= 2 else None)
    return tuple(sides)


def collect_town(client: Any, args: argparse.Namespace) -> list[dict]:
    import cv2

    carla_mod = load_carla()
    rng = random.Random(f"{args.town}:{args.seed}")
    rows = ROI_Y + np.asarray(ELAS_ROW_FRACTIONS) * ROI_H
    base = CameraRig()
    summaries = []
    with synchronous_world(client, args.town, args.fps, args.seed, args.map_layers) as (world, tm):
        world_map, library = world.get_map(), world.get_blueprint_library()
        spawn_points = world_map.get_spawn_points()
        npcs = []
        four_wheeled = [b for b in library.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
        try:
            for point in rng.sample(spawn_points, min(args.traffic, len(spawn_points) - args.scenes)):
                npc = world.try_spawn_actor(rng.choice(four_wheeled), point)
                if npc:
                    npc.set_autopilot(True, tm.get_port())
                    npcs.append(npc)
            for k in range(args.scenes):
                if args.target_labelled and sum(s["labelled"] for s in summaries) >= args.target_labelled:
                    break
                scene = f"CARLA_{args.town}_s{args.seed}_{k:02d}"
                weather = WEATHERS[(k + args.seed) % len(WEATHERS)]
                world.set_weather(getattr(carla_mod.WeatherParameters, weather))
                rig, yaw = jittered_rig(base, rng)
                summaries.append(_record_scene(world, world_map, tm, library, spawn_points, rig, yaw, rows, scene,
                                               weather, len(npcs), rng, args, carla_mod, cv2))
                print(f"{scene}: {summaries[-1]['labelled']} labelled of {summaries[-1]['frames']} frames, "
                      f"weather {weather}{' (stopped: stuck)' if summaries[-1]['stuck'] else ''}", flush=True)
        finally:
            client.apply_batch([carla_mod.command.DestroyActor(a) for a in npcs])
    return summaries


def _record_scene(world, world_map, tm, library, spawn_points, rig, yaw, rows, scene, weather, n_npcs, rng, args,
                  carla_mod, cv2) -> dict:
    out = Path(args.out) / scene
    images = out / "images" / "images"
    images.mkdir(parents=True, exist_ok=True)
    actors = []
    try:
        ego = None
        for point in rng.sample(spawn_points, len(spawn_points)):
            ego = world.try_spawn_actor(library.find(EGO_BLUEPRINT), point)
            if ego:
                break
        actors.append(ego)
        bp = library.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(rig.width))
        bp.set_attribute("image_size_y", str(rig.height))
        bp.set_attribute("fov", str(rig.fov_deg))
        mount = carla_mod.Transform(carla_mod.Location(x=rig.x_m, z=rig.z_m),
                                    carla_mod.Rotation(pitch=rig.pitch_deg, yaw=yaw))
        camera = world.spawn_actor(bp, mount, attach_to=ego)
        actors.append(camera)
        frames: queue.Queue = queue.Queue()
        camera.listen(frames.put)
        ego.set_autopilot(True, tm.get_port())
        tm.ignore_lights_percentage(ego, 100.0)
        tm.vehicle_percentage_speed_difference(ego, rng.uniform(-30.0, 10.0))
        tm.random_left_lanechange_percentage(ego, 3.0)
        tm.random_right_lanechange_percentage(ego, 3.0)
        n_frames = int(round(args.scene_s * args.fps))
        warmup = int(round(args.warmup_s * args.fps))
        gt, overlays = [], Path(args.out) / "_overlays"
        lane_offset = getattr(tm, "vehicle_lane_offset", None)   # Traffic Manager >= 0.9.14
        written, still = 0, 0
        for i in range(-warmup, n_frames):
            if lane_offset is not None and i % int(args.offset_every_s * args.fps) == 0:
                lane_offset(ego, rng.uniform(-args.max_offset_m, args.max_offset_m))
            frame = world.tick()
            image = frames.get(timeout=SENSOR_TIMEOUT_S)
            while image.frame < frame:
                image = frames.get(timeout=SENSOR_TIMEOUT_S)
            if i < 0:
                continue
            vel = ego.get_velocity()
            speed = math.hypot(vel.x, vel.y)
            still = still + 1 if speed < 1.0 else 0
            if still > args.stuck_s * args.fps:   # queued behind traffic: the rest would be one repeated view
                break
            written = i + 1
            bgr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(rig.height, rig.width, 4)[:, :, :3]
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            buf.tofile(str(images / f"lane_{i}.jpg"))
            if i % args.label_every or speed < args.min_label_speed:
                continue
            label = frame_label(world_map, ego, camera, rig, rows, carla_mod)
            if label is None or (label[0] is None and label[1] is None):
                continue
            gt.append(f'<frame id="{i}"><position>{side_xml("left", label[0])}{side_xml("right", label[1])}'
                      f"</position></frame>")
            if args.overlay_every and len(gt) % args.overlay_every == 1:
                overlays.mkdir(exist_ok=True)
                check = bgr.copy()
                for xs, colour in zip(label, ((0, 0, 255), (255, 80, 0))):
                    for x, y in zip(xs if xs is not None else [], rows):
                        if np.isfinite(x):
                            cv2.circle(check, (int(x), int(min(y, rig.height - 1))), 4, colour, -1)
                cv2.imencode(".jpg", check)[1].tofile(str(overlays / f"{scene}_{i:04d}.jpg"))
    finally:
        for actor in reversed(actors):
            if actor is not None:
                if actor.type_id.startswith("sensor."):
                    actor.stop()
                actor.destroy()
    n_frames, stuck = max(written, 1), written < n_frames
    (out / "config.xml").write_text(config_xml(scene, n_frames, rig.width, rig.height), encoding="utf-8")
    (out / "groundtruth.xml").write_text(
        f'<groundtruth><frames n="{n_frames}">{"".join(gt)}</frames></groundtruth>', encoding="utf-8")
    meta = {"scene": scene, "town": args.town, "seed": args.seed, "weather": weather, "fps": args.fps,
            "frames": n_frames, "stuck": stuck, "labelled": len(gt), "label_every": args.label_every,
            "npc_vehicles": n_npcs,
            "camera": asdict(rig), "camera_pitch_deg": rig.pitch_deg, "camera_yaw_deg": yaw,
            "roi": [ROI_Y, ROI_H]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="dataset root (ASCII path: CARLA and OpenCV write there)")
    ap.add_argument("--town", required=True)
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--scene-s", type=float, default=30.0)
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--fps", type=float, default=30.0, help="ELAS frame rate: temporal history as in training")
    ap.add_argument("--label-every", type=int, default=3)
    ap.add_argument("--target-labelled", type=int, default=0, help="stop the town once this many frames are labelled")
    ap.add_argument("--min-label-speed", type=float, default=3.0, help="m/s: no label for a car (nearly) at rest")
    ap.add_argument("--stuck-s", type=float, default=5.0, help="end a scene after this long below 1 m/s")
    ap.add_argument("--traffic", type=int, default=25, help="NPC vehicles on autopilot (occluders)")
    ap.add_argument("--max-offset-m", type=float, default=0.7, help="random lateral offset of the ego car")
    ap.add_argument("--offset-every-s", type=float, default=4.0)
    ap.add_argument("--overlay-every", type=int, default=0, help="label-check overlay every N labelled frames")
    ap.add_argument("--map-layers", choices=["all", "minimal"], default="all",
                    help="minimal: <town>_Opt without buildings/props (towns that crash the server when loaded)")
    ap.add_argument("--allow-test-town", action="store_true", help="collect in Town04/Town06 anyway")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=1)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.town in TEST_TOWNS and not args.allow_test_town:
        raise SystemExit(f"{args.town} is a closed-loop test town (docs/CARLA.md); pass --allow-test-town to "
                         "collect there anyway")
    carla_mod = load_carla()
    client = carla_mod.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    summaries = collect_town(client, args)
    print(f"{args.town}: {sum(s['labelled'] for s in summaries)} labelled frames in {len(summaries)} scenes "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
