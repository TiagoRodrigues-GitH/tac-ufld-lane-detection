"""Watch the lane models in CARLA, side by side, and let one of them steer (docs/CARLA.md, "How to open CARLA and
watch the models").

One window (pygame) and/or an H.264 video with one panel each for: the camera, the CARLA ground truth (the map's
ego-lane edges projected into the image), and every model given with ``--checkpoint`` (its lanes, the green ground
truth for reference, and a HUD: lateral offset, lane change, LDW, LKA steering request, latency).

Drive modes (``--drive``):

* ``manual``    you drive with the keyboard; ``L`` lets the selected model steer (LKA on top of your steering)
* ``autopilot`` CARLA's Traffic Manager drives (blinkers included); ``L`` hands the steering to the selected model
                with a constant-speed cruise control, ``L`` again gives it back to the autopilot
* ``lka``       the selected model steers from the start (cruise control at ``--speed-kmh``)

Keys: ``W``/``S`` throttle/brake, ``A``/``D`` steer, ``Q``/``E`` left/right turn signal (toggle), ``L`` LKA on/off,
``1``..``9`` which model steers, ``G`` ground truth on the model panels on/off, ``ESC`` quit.

The models run in the project venv (torch), one detector server each (``tac_ufld.sim.detector_bridge``), started by
this script; the script itself runs in the CARLA venv. If no CARLA server answers, it starts one (off-screen,
Low quality)::

    $py = "C:\\CARLA\\venv312\\Scripts\\python.exe"
    & $py scripts\\carla_drive.py --town Town04 --drive manual `
        --checkpoint results\\elas_pilot_v2\\seed_1\\checkpoints\\ufld_baseline.pt `
        --checkpoint results\\carla_finetune_lite_v05\\seed_1\\checkpoints\\lite_v05.pt
    & $py scripts\\carla_drive.py --town Town04 --drive autopilot --traffic 30 --duration-s 60 --no-window `
        --record results\\carla_lka\\demo.mp4 --checkpoint ... --checkpoint ...

The simulation is synchronous (every frame waits for every model), so with several models it runs slower than real
time; the recorded video plays at the simulation rate (``--fps``), i.e. in simulated real time.
"""
from __future__ import annotations

import argparse
import math
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac_ufld.ego import EgoLaneTracker, LaneChangeParams, LaneKeepingParams, ego_lane_state, lane_keeping  # noqa: E402
from tac_ufld.sim import detector_bridge as bridge  # noqa: E402
from tac_ufld.sim.lka_loop import (  # noqa: E402
    SCENARIOS, CameraRig, Rig, RunConfig, SpeedController, image_polyline, lane_edges, load_carla, pick_start,
    signed_offset, synchronous_world, turn_signal_from_lights,
)

CARLA_EXE = Path(r"C:\CARLA\CARLA_0.9.16\CarlaUE4.exe")
GT_COLOUR, LEFT_COLOUR, RIGHT_COLOUR = (0, 200, 0), (0, 0, 255), (255, 80, 0)
WARN_COLOUR, TEXT_COLOUR, ACTIVE_COLOUR = (0, 0, 255), (255, 255, 255), (0, 220, 255)


# --------------------------------------------------------------------------- drawing (no CARLA)

def tile(panels: list[np.ndarray], columns: int, size: tuple[int, int]) -> np.ndarray:
    """Panels (BGR, any size) resized to ``size`` (w, h) and tiled left to right, top to bottom; black fill."""
    import cv2

    w, h = size
    rows = math.ceil(len(panels) / columns)
    canvas = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for k, panel in enumerate(panels):
        r, c = divmod(k, columns)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = cv2.resize(panel, (w, h), interpolation=cv2.INTER_AREA)
    return canvas


def draw_lanes(img: np.ndarray, lanes: list, colours: tuple, thickness: int) -> None:
    import cv2

    for lane, colour in zip(lanes, colours):
        if lane is not None and len(lane) >= 2:
            cv2.polylines(img, [np.round(lane).astype(np.int32).reshape(-1, 1, 2)], False, colour, thickness,
                          cv2.LINE_AA)


def draw_text(img: np.ndarray, lines: list[tuple[str, tuple]], scale: float = 0.55) -> None:
    """Text lines on a dark band at the top: [(text, BGR colour), ...]."""
    import cv2

    band = int(14 + 22 * len(lines) * scale / 0.55)
    img[:band] = (img[:band] * 0.35).astype(np.uint8)
    for k, (text, colour) in enumerate(lines):
        cv2.putText(img, text, (8, int(20 + 22 * k * scale / 0.55)), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1,
                    cv2.LINE_AA)


def legend_panel(size: tuple[int, int], models: list[str]) -> np.ndarray:
    """Colour key and keyboard help, for an empty slot of the grid."""
    import cv2

    w, h = size
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    rows = [("green: ground truth (CARLA map)", GT_COLOUR), ("red / blue: model ego-left / ego-right", TEXT_COLOUR),
            ("LDW in red when it warns", WARN_COLOUR), ("", TEXT_COLOUR)]
    rows += [(f"[{k + 1}] {name}", TEXT_COLOUR) for k, name in enumerate(models)]
    rows += [("", TEXT_COLOUR), ("W/S throttle/brake  A/D steer  Q/E signal", TEXT_COLOUR),
             ("L LKA on/off  1-9 model  G ground truth  ESC quit", TEXT_COLOUR)]
    scale = 0.45 * w / 480
    for k, (text, colour) in enumerate(rows):
        cv2.putText(img, text, (12, int((28 + 24 * k) * w / 480)), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1,
                    cv2.LINE_AA)
    return img


def side_name(value: int) -> str:
    return {-1: "LEFT", 1: "RIGHT"}.get(int(value), "-")


class Watchdog:
    """Ends the process if no frame completes for ``stall_s`` seconds. With the Traffic Manager in synchronous mode
    ``world.tick()`` can block forever inside the CARLA client (seen once on 2026-10-19, after 4 min of a
    recording); the ``cleanup`` callbacks stop ffmpeg and the detector servers first."""

    def __init__(self, stall_s: float, cleanup: list) -> None:
        import threading

        self.stall_s, self.cleanup, self.last = stall_s, cleanup, time.monotonic()
        threading.Thread(target=self._watch, daemon=True).start()

    def beat(self) -> None:
        self.last = time.monotonic()

    def _watch(self) -> None:
        import os

        while True:
            time.sleep(5.0)
            if time.monotonic() - self.last > self.stall_s:
                print(f"no frame for {self.stall_s:.0f} s: the CARLA client is stuck (restart the server); exiting",
                      flush=True)
                for fn in self.cleanup:
                    try:
                        fn()
                    except Exception:  # noqa: BLE001 - best effort on the way out
                        pass
                os._exit(3)


class VideoWriter:
    """H.264 MP4 through ffmpeg (stdin pipe, BGR frames)."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        w, h = size
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=120)

    def kill(self) -> None:
        self.proc.kill()


# --------------------------------------------------------------------------- CARLA server

def ensure_server(host: str, port: int, exe: Path, start: bool, wait_s: float = 240.0):
    """A connected ``carla.Client``; starts ``CarlaUE4.exe`` (off-screen, Low quality) if nothing answers."""
    carla_mod = load_carla()

    def answers() -> bool:
        with socket.socket() as s:
            s.settimeout(1.0)
            return s.connect_ex((host, port)) == 0

    if not answers():
        if not start:
            raise SystemExit(f"no CARLA server on {host}:{port} (start it, or drop --no-start)")
        if not exe.exists():
            raise SystemExit(f"CARLA not found at {exe} (--carla-exe)")
        print(f"starting {exe.name} (off-screen, Low quality)...", flush=True)
        subprocess.Popen([str(exe), "-RenderOffScreen", "-quality-level=Low", f"-carla-rpc-port={port}"])
    deadline = time.monotonic() + wait_s
    while True:
        try:
            client = carla_mod.Client(host, port)
            client.set_timeout(10.0)
            client.get_server_version()
            client.set_timeout(300.0)
            return client
        except RuntimeError:
            if time.monotonic() > deadline:
                raise
            time.sleep(3.0)


# --------------------------------------------------------------------------- models

class ModelPanel:
    """One checkpoint: detector server, ego-lane tracker, and its last decisions."""

    def __init__(self, checkpoint: Path, args: argparse.Namespace, rig: CameraRig, log_dir: Path, k: int) -> None:
        python = Path(args.detector_python) if args.detector_python else REPO.parent / ".venv" / "Scripts" / "python.exe"
        self.detector = bridge.spawn_server(python, checkpoint, log_dir / f"detector_{k}.log", args.device)
        info = self.detector.info()
        run = checkpoint.resolve().parents[2].name if len(checkpoint.resolve().parents) > 2 else ""
        self.name = f"{info['variant']} [{run}]"
        self.tracker = EgoLaneTracker(rig.width / 2, args.y_ref * rig.height, args.y_far * rig.height,
                                      LaneChangeParams(fps=args.fps))
        self.lanes: list = [None, None]
        self.state = self.lk = None
        self.lc, self.latency = 0, float("nan")

    def update(self, rgb: np.ndarray, index: int, signal: int, keeping: LaneKeepingParams) -> None:
        out = self.detector.infer(rgb, frame_index=index)
        self.lanes = (list(out.lanes) + [None, None])[:2]
        self.latency = out.latency_ms
        self.state, self.lc = self.tracker.update(self.lanes)
        self.lk = lane_keeping(self.state, self.tracker.detector, signal, keeping)

    def close(self) -> None:
        self.detector.close()


# --------------------------------------------------------------------------- main loop

class Keyboard:
    """pygame keyboard state -> driver commands (manual control in the style of CARLA's manual_control.py)."""

    def __init__(self) -> None:
        import pygame

        self.pg = pygame
        self.steer = 0.0
        self.quit = False
        self.events: list[str] = []

    def poll(self, dt: float) -> tuple[float, float, float]:
        pg = self.pg
        self.events = []
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit = True
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.quit = True
                elif event.unicode:
                    self.events.append(event.unicode.lower())
        keys = pg.key.get_pressed()
        target = (-0.35 if keys[pg.K_a] else 0.0) + (0.35 if keys[pg.K_d] else 0.0)
        self.steer += float(np.clip(target - self.steer, -1.2 * dt, 1.2 * dt))
        throttle = 0.6 if keys[pg.K_w] else 0.0
        brake = 0.8 if keys[pg.K_s] else 0.0
        return throttle, brake, self.steer


def run(args: argparse.Namespace) -> None:
    import cv2

    carla_mod = load_carla()
    out_dir = Path(args.record).parent if args.record else REPO / "results" / "carla_drive"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = ensure_server(args.host, args.port, Path(args.carla_exe), not args.no_start)
    rig_cam = CameraRig()
    start_speed = 0.0 if args.drive == "manual" else args.speed_kmh   # manual driving starts at rest
    cfg = RunConfig(fps=args.fps, speed_kmh=start_speed, rig=rig_cam, seed=args.seed, weather=args.weather)
    models: list[ModelPanel] = []
    window = None
    writer = None
    npcs: list = []
    tm = None
    try:
        for k, ckpt in enumerate(args.checkpoint):
            print(f"loading model {k + 1}: {ckpt}", flush=True)
            models.append(ModelPanel(Path(ckpt), args, rig_cam, out_dir, k))
        with synchronous_world(client, args.town, args.fps, no_rendering=False, map_layers=args.map_layers) as world:
            world.set_weather(getattr(carla_mod.WeatherParameters, args.weather))
            world_map = world.get_map()
            rng = random.Random(args.seed)
            if args.drive == "lka" or args.start == "straight":
                start, _ = pick_start(world_map, SCENARIOS["straight"], args.speed_kmh / 3.6 * 30, 50.0, rng)
            else:
                start = world_map.get_waypoint(rng.choice(world_map.get_spawn_points()).location)
            needs_tm = args.drive == "autopilot" or args.traffic > 0
            if needs_tm:
                tm = client.get_trafficmanager()
                tm.set_synchronous_mode(True)
                tm.set_random_device_seed(args.seed)
            with Rig(client, world, cfg, start, camera=True, lead=False) as rig:
                library = world.get_blueprint_library()
                cars = [b for b in library.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
                for point in rng.sample(world_map.get_spawn_points(), min(args.traffic, 80)):
                    if point.location.distance(start.transform.location) < 15:
                        continue
                    npc = world.try_spawn_actor(rng.choice(cars), point)
                    if npc:
                        npc.set_autopilot(True, tm.get_port())
                        npcs.append(npc)
                autopilot = args.drive == "autopilot"
                if autopilot:
                    rig.ego.set_autopilot(True, tm.get_port())
                    tm.update_vehicle_lights(rig.ego, True)
                    tm.auto_lane_change(rig.ego, True)
                keeping = LaneKeepingParams(k_offset=args.k_offset, k_heading=args.k_heading, max_steer=0.3,
                                            half_width=rig.half_width / start.lane_width)
                panels_n = 2 + len(models)
                columns = min(args.columns, panels_n)
                with_legend = panels_n % columns != 0   # an empty slot in the grid: put the legend there
                panels_n += with_legend
                pw = args.panel_width - args.panel_width % 2
                size = (pw, (pw * 3 // 4) - (pw * 3 // 4) % 2)
                canvas_size = (columns * size[0], math.ceil(panels_n / columns) * size[1])
                if not args.no_window:
                    import pygame

                    pygame.init()
                    window = pygame.display.set_mode(canvas_size)
                    pygame.display.set_caption("TAC-UFLD in CARLA  |  W/S A/D drive, Q/E signal, L LKA, 1-9 model, "
                                               "G ground truth, ESC quit")
                    keyboard = Keyboard()
                if args.record:
                    writer = VideoWriter(Path(args.record), canvas_size, args.fps)
                legend = legend_panel(size, [m.name for m in models]) if with_legend else None
                lka_on, active, show_gt, signal_cmd = args.drive == "lka", 0, True, 0
                cruise = SpeedController()
                dt = 1.0 / args.fps
                n_frames = int(args.duration_s * args.fps) if args.duration_s else None
                cleanup = [m.detector.process.kill for m in models if m.detector.process is not None]
                watchdog = Watchdog(args.stall_s, cleanup + ([writer.kill] if writer is not None else []))
                i, t0 = 0, time.time()
                print("driving: " + ("ESC to quit" if window else f"{args.duration_s} s"), flush=True)
                while n_frames is None or i < n_frames:
                    rgb = rig.tick()
                    tf = rig.ego.get_transform()
                    vel = rig.ego.get_velocity()
                    speed = math.hypot(vel.x, vel.y)
                    wp = world_map.get_waypoint(tf.location, project_to_road=True,
                                                lane_type=carla_mod.LaneType.Driving)
                    right = wp.transform.get_right_vector()
                    gt_offset = signed_offset((tf.location.x, tf.location.y),
                                              (wp.transform.location.x, wp.transform.location.y), (right.x, right.y))
                    w2c = rig.world_to_camera(tf)
                    gt_lanes = [image_polyline(rig_cam.project(e, w2c), rig_cam.width, rig_cam.height)
                                if len(e) > 1 else None for e in lane_edges(wp)]
                    gt_state = ego_lane_state(gt_lanes, args.y_ref * rig_cam.height, args.y_far * rig_cam.height,
                                              rig_cam.width / 2)
                    signal = turn_signal_from_lights(int(rig.ego.get_light_state()))
                    for m in models:
                        m.update(rgb, i, signal, keeping)

                    throttle = brake = driver_steer = 0.0
                    if window is not None:
                        throttle, brake, driver_steer = keyboard.poll(dt)
                        if keyboard.quit:
                            break
                        for key in keyboard.events:
                            if key == "l":
                                lka_on = not lka_on
                                if autopilot:
                                    rig.ego.set_autopilot(not lka_on, tm.get_port())
                            elif key == "q":
                                signal_cmd = 0 if signal_cmd == -1 else -1
                                rig.set_turn_signal(signal_cmd)
                            elif key == "e":
                                signal_cmd = 0 if signal_cmd == 1 else 1
                                rig.set_turn_signal(signal_cmd)
                            elif key == "g":
                                show_gt = not show_gt
                            elif key.isdigit() and 1 <= int(key) <= len(models):
                                active = int(key) - 1
                    lka_steer = models[active].lk.steer if (models and lka_on and models[active].lk) else 0.0
                    if not (autopilot and not lka_on):
                        if args.drive != "manual":   # cruise control (lka mode, or autopilot handed to the LKA)
                            throttle, brake = cruise.update(args.speed_kmh / 3.6, speed, dt)
                        steer = float(np.clip(driver_steer + lka_steer, -1.0, 1.0))
                        rig.ego.apply_control(carla_mod.VehicleControl(throttle=throttle, steer=steer, brake=brake))

                    # ---- panels
                    camera = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    gt_panel = camera.copy()
                    draw_lanes(gt_panel, gt_lanes, (GT_COLOUR, GT_COLOUR), 3)
                    driver = ("autopilot" if autopilot and not lka_on else
                              f"LKA: model {active + 1}" if lka_on and models else "manual")
                    draw_text(camera, [(f"camera  t {i * dt:5.1f} s  {speed * 3.6:4.0f} km/h", TEXT_COLOUR),
                                       (f"{args.town}  steering: {driver}  signal {side_name(signal)}", TEXT_COLOUR)])
                    draw_text(gt_panel, [("ground truth (CARLA map, ego lane)", GT_COLOUR),
                                         (f"offset {gt_offset:+.2f} m  ({gt_state.offset:+.2f} lane widths)",
                                          TEXT_COLOUR),
                                         (f"lane width {wp.lane_width:.2f} m  lane id {wp.lane_id}", TEXT_COLOUR)])
                    panels = [camera, gt_panel]
                    for k, m in enumerate(models):
                        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        if show_gt:
                            draw_lanes(img, gt_lanes, (GT_COLOUR, GT_COLOUR), 1)
                        draw_lanes(img, m.lanes, (LEFT_COLOUR, RIGHT_COLOUR), 3)
                        steering = lka_on and k == active
                        lk = m.lk
                        lines = [(f"[{k + 1}] {m.name}{'  STEERING' if steering else ''}",
                                  ACTIVE_COLOUR if steering else TEXT_COLOUR),
                                 (f"offset {m.state.offset:+.2f} lw  (GT {gt_state.offset:+.2f})", TEXT_COLOUR),
                                 (f"lane change {side_name(m.lc)}   LDW {side_name(lk.ldw)}",
                                  WARN_COLOUR if lk.ldw else TEXT_COLOUR),
                                 (f"LKA steer {lk.steer:+.3f}{'' if lk.active else ' (off: signal / no lane)'}",
                                  TEXT_COLOUR),
                                 (f"latency {m.latency:5.1f} ms", TEXT_COLOUR)]
                        draw_text(img, lines)
                        panels.append(img)
                    frame = tile(panels + ([legend] if legend is not None else []), columns, size)
                    if writer is not None:
                        writer.write(frame)
                    if window is not None:
                        import pygame

                        surface = pygame.image.frombuffer(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes(),
                                                          canvas_size, "RGB")
                        window.blit(surface, (0, 0))
                        pygame.display.flip()
                    if rig.collisions and args.stop_on_collision:
                        print(f"collision with {rig.collisions[0]}", flush=True)
                        break
                    watchdog.beat()
                    i += 1
                print(f"{i} frames ({i * dt:.1f} s simulated) in {time.time() - t0:.0f} s", flush=True)
    finally:
        if writer is not None:
            writer.close()
            print(f"video: {args.record}", flush=True)
        if npcs:
            client.apply_batch([load_carla().command.DestroyActor(a) for a in npcs])
        if tm is not None:
            tm.set_synchronous_mode(False)
        for m in models:
            m.close()
        if window is not None:
            import pygame

            pygame.quit()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", action="append", default=[], help="a model to show (repeat for several)")
    ap.add_argument("--drive", choices=["manual", "autopilot", "lka"], default="manual")
    ap.add_argument("--town", default="Town04")
    ap.add_argument("--map-layers", choices=["all", "minimal"], default="all",
                    help="minimal: <town>_Opt without buildings/props (Town03/Town06 crash this PC when fully loaded)")
    ap.add_argument("--start", choices=["spawn", "straight"], default="spawn",
                    help="random spawn point, or the straightest route found (lka always uses straight)")
    ap.add_argument("--traffic", type=int, default=0, help="other cars on autopilot")
    ap.add_argument("--weather", default="ClearNoon")
    ap.add_argument("--speed-kmh", type=float, default=60.0, help="cruise speed (lka, or autopilot handed to LKA)")
    ap.add_argument("--k-offset", type=float, default=0.6)
    ap.add_argument("--k-heading", type=float, default=0.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--y-ref", type=float, default=0.95)
    ap.add_argument("--y-far", type=float, default=0.72)
    ap.add_argument("--duration-s", type=float, default=None, help="stop after this much simulated time")
    ap.add_argument("--record", default=None, help="H.264 MP4 of the panels (ffmpeg)")
    ap.add_argument("--no-window", action="store_true", help="no live window (recording only; needs --duration-s)")
    ap.add_argument("--panel-width", type=int, default=480)
    ap.add_argument("--columns", type=int, default=3)
    ap.add_argument("--stop-on-collision", action="store_true")
    ap.add_argument("--stall-s", type=float, default=90.0, help="exit if no frame completes for this long")
    ap.add_argument("--device", default="auto", help="detector device: auto | cuda | cpu")
    ap.add_argument("--detector-python", default=None, help="project venv Python (default ../.venv)")
    ap.add_argument("--carla-exe", default=str(CARLA_EXE))
    ap.add_argument("--no-start", action="store_true", help="never start a CARLA server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    if args.no_window and not args.duration_s:
        ap.error("--no-window needs --duration-s")
    if args.no_window and not args.record:
        ap.error("--no-window without --record shows nothing")
    if args.drive == "lka" and not args.checkpoint:
        ap.error("--drive lka needs at least one --checkpoint")
    return args


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
