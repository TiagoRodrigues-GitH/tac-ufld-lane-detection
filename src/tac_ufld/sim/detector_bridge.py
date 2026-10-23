"""The lane detector behind a local socket, so the CARLA client and the trained model can run in different Pythons.

The CARLA 0.9.16 client exists for Python 3.10-3.12 only (``C:\\CARLA\\venv312``); the project venv (torch) is Python
3.14. The closed loop (``tac_ufld.sim.lka_loop``) runs in the CARLA venv and talks to a detector server started
in the project venv. Nothing is pickled: the two interpreters only exchange JSON and raw RGB bytes.

Protocol (one TCP connection, request/response): every message is a 4-byte big-endian length, a UTF-8 JSON header
of that length, then ``header["nbytes"]`` raw bytes (0 for control messages)::

    {"cmd": "info"}                                     -> {"ok": true, "variant": ..., "num_lanes": ..., ...}
    {"cmd": "infer", "shape": [h, w, 3], "frame_index": i} + RGB uint8 bytes
                                                        -> {"ok": true, "lanes": [[[x, y], ...] | null, ...],
                                                            "confidence": [...], "latency_ms": ..., "fallbacks": ...}
    {"cmd": "reset"}                                    -> {"ok": true}          (new stream: temporal state cleared)
    {"cmd": "bye"}                                      -> closes this connection, the server waits for the next
    {"cmd": "shutdown"}                                 -> {"ok": true}, then the server exits

Server (project venv)::

    python -m tac_ufld.sim.detector_bridge --checkpoint results/<run>/seed_1/checkpoints/lite_v05.pt --port 5599

``scripts/carla_lka.py`` starts it by itself (``spawn_server``) unless ``--detector-addr`` points at a running one,
which can also be another machine (e.g. a Jetson running the deployed model).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LENGTH = struct.Struct(">I")
MAX_HEADER_BYTES = 1 << 20
MAX_PAYLOAD_BYTES = 64 << 20
STREAM_ID = "carla"


# --------------------------------------------------------------------------- framing

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("the peer closed the connection")
        got += k
    return bytes(buf)


def send_message(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    raw = json.dumps({**header, "nbytes": len(payload)}).encode("utf-8")
    sock.sendall(LENGTH.pack(len(raw)) + raw)
    if payload:
        sock.sendall(payload)


def recv_message(sock: socket.socket) -> tuple[dict, bytes]:
    (size,) = LENGTH.unpack(_recv_exact(sock, LENGTH.size))
    if size > MAX_HEADER_BYTES:
        raise ValueError(f"header of {size} bytes refused")
    header = json.loads(_recv_exact(sock, size).decode("utf-8"))
    nbytes = int(header.get("nbytes", 0))
    if not 0 <= nbytes <= MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload of {nbytes} bytes refused")
    return header, _recv_exact(sock, nbytes) if nbytes else b""


def decode_frame(header: dict, payload: bytes) -> np.ndarray:
    shape = tuple(int(v) for v in header["shape"])
    if len(shape) != 3 or shape[2] != 3 or int(np.prod(shape)) != len(payload):
        raise ValueError(f"frame shape {shape} does not match {len(payload)} bytes of RGB uint8")
    return np.frombuffer(payload, dtype=np.uint8).reshape(shape)


# --------------------------------------------------------------------------- results

@dataclass
class LaneOutput:
    lanes: list[np.ndarray | None]      # per slot (0 = ego-left, 1 = ego-right), (N, 2) x, y in frame pixels
    confidence: list[float | None]
    latency_ms: float                   # detector time (preprocess + model + postprocess)
    fallbacks: int                      # history frames replaced by a newer one (stream start, drops)


def lanes_to_json(lanes: list[np.ndarray | None]) -> list:
    return [None if lane is None else np.round(np.asarray(lane, dtype=np.float64), 2).tolist() for lane in lanes]


def lanes_from_json(raw: list) -> list[np.ndarray | None]:
    return [None if lane is None else np.asarray(lane, dtype=np.float32).reshape(-1, 2) for lane in raw]


# --------------------------------------------------------------------------- the model (project venv only)

class StreamingHandler:
    """A trained checkpoint through ``StreamingLaneDetector`` (one stream). Imports torch: project venv only."""

    def __init__(self, checkpoint: str | Path, device: str = "auto", mode: str = "cached",
                 kalman: bool = False) -> None:
        import torch

        from tac_ufld.inference.loading import load_model
        from tac_ufld.inference.streaming import StreamingLaneDetector

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.loaded = load_model(checkpoint, device)
        tracker = None
        if kalman:
            if not self.loaded.kalman:
                raise ValueError("--kalman: this checkpoint has no validation-tuned Kalman parameters")
            from tac_ufld.evaluation.tracking import KalmanParams

            tracker = KalmanParams(**self.loaded.kalman)
        self.detector = StreamingLaneDetector(self.loaded, mode=mode, tracker=tracker)
        self.kalman = tracker is not None

    def info(self) -> dict:
        loaded, d = self.loaded, self.loaded.cfg.data
        return {"variant": loaded.variant, "display_name": loaded.card.get("display_name", loaded.variant),
                "checkpoint": str(loaded.checkpoint), "dataset": d.dataset, "num_lanes": int(d.num_lanes),
                "model_size": list(loaded.model_size), "row_anchor_range": list(d.row_anchor_range),
                "temporal": loaded.temporal, "num_frames": loaded.num_frames,
                "temporal_step": int(loaded.card["temporal_step"]), "mode": self.detector.mode,
                "kalman": self.kalman, "postprocess": loaded.postprocess_source, "device": self.device}

    def reset(self) -> None:
        self.detector.reset_stream(STREAM_ID)

    def infer(self, rgb: np.ndarray, frame_index: int, timestamp: float | None = None) -> LaneOutput:
        res = self.detector.infer_frame(rgb, STREAM_ID, timestamp=timestamp, frame_index=frame_index)
        return LaneOutput(res.lanes, res.lane_confidence, float(res.latency_ms["total_ms"]), int(res.fallbacks))

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- server

def serve(handler, host: str = "127.0.0.1", port: int = 5599, ready=None) -> None:
    """Answer requests until a client sends ``shutdown``; one client at a time. ``ready(port)`` is called once
    the socket listens (port 0 = any free port)."""
    with socket.create_server((host, port)) as server:
        if ready is not None:
            ready(server.getsockname()[1])
        while True:
            conn, _addr = server.accept()
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if not _session(conn, handler):
                    return


def _session(conn: socket.socket, handler) -> bool:
    """Serve one connection; False when the client asked the server to stop."""
    while True:
        try:
            header, payload = recv_message(conn)
        except (ConnectionError, OSError):
            return True
        cmd = header.get("cmd")
        try:
            if cmd == "infer":
                out = handler.infer(decode_frame(header, payload), int(header["frame_index"]), header.get("timestamp"))
                reply = {"lanes": lanes_to_json(out.lanes), "confidence": out.confidence,
                         "latency_ms": out.latency_ms, "fallbacks": out.fallbacks}
            elif cmd == "info":
                reply = handler.info()
            elif cmd == "reset":
                handler.reset()
                reply = {}
            elif cmd == "bye":
                return True
            elif cmd == "shutdown":
                send_message(conn, {"ok": True})
                return False
            else:
                raise ValueError(f"unknown command {cmd!r}")
        except (ValueError, KeyError, TypeError) as exc:
            send_message(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            continue
        send_message(conn, {"ok": True, **reply})


# --------------------------------------------------------------------------- client

class RemoteLaneDetector:
    """Client side (any Python with NumPy): the same ``info / reset / infer / close`` as ``StreamingHandler``."""

    def __init__(self, host: str, port: int, connect_timeout_s: float = 180.0,
                 process: subprocess.Popen | None = None, log_path: Path | None = None) -> None:
        self.process, self.log_path = process, log_path
        deadline = time.monotonic() + connect_timeout_s
        while True:
            try:
                self.sock = socket.create_connection((host, port), timeout=5.0)
                break
            except OSError:
                if process is not None and process.poll() is not None:
                    raise RuntimeError(f"the detector server exited with code {process.returncode}; "
                                       f"see {log_path}") from None
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)
        self.sock.settimeout(300.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _call(self, header: dict, payload: bytes = b"") -> dict:
        send_message(self.sock, header, payload)
        reply, _ = recv_message(self.sock)
        if not reply.get("ok"):
            raise RuntimeError(f"detector server: {reply.get('error')}")
        return reply

    def info(self) -> dict:
        reply = self._call({"cmd": "info"})
        reply.pop("ok", None)
        reply.pop("nbytes", None)
        return reply

    def reset(self) -> None:
        self._call({"cmd": "reset"})

    def infer(self, rgb: np.ndarray, frame_index: int, timestamp: float | None = None) -> LaneOutput:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        r = self._call({"cmd": "infer", "shape": list(rgb.shape), "frame_index": int(frame_index),
                        "timestamp": timestamp}, rgb.tobytes())
        return LaneOutput(lanes_from_json(r["lanes"]), r["confidence"], float(r["latency_ms"]), int(r["fallbacks"]))

    def close(self) -> None:
        """Disconnect; a server this client spawned is shut down too."""
        try:
            send_message(self.sock, {"cmd": "shutdown" if self.process is not None else "bye"})
            if self.process is not None:
                recv_message(self.sock)
        except OSError:
            pass
        finally:
            self.sock.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def spawn_server(python: str | Path, checkpoint: str | Path, log_path: Path, device: str = "auto",
                 mode: str = "cached", kalman: bool = False, host: str = "127.0.0.1") -> RemoteLaneDetector:
    """Start ``python -m tac_ufld.sim.detector_bridge`` (the project venv) and connect to it."""
    src = Path(__file__).resolve().parents[2]
    port = free_port(host)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (str(src), os.environ.get("PYTHONPATH")) if p)}
    cmd = [str(python), "-m", "tac_ufld.sim.detector_bridge", "--checkpoint", str(Path(checkpoint).resolve()),
           "--host", host, "--port", str(port), "--device", device, "--mode", mode] + (["--kalman"] if kalman else [])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=src.parent)
    return RemoteLaneDetector(host, port, process=process, log_path=log_path)


def main(argv: list[str] | None = None) -> None:
    from tac_ufld.inference.streaming import MODES

    ap = argparse.ArgumentParser(description="Serve a trained lane detector to the CARLA closed loop.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--mode", default="cached", choices=MODES)
    ap.add_argument("--kalman", action="store_true", help="validation-tuned Kalman output tracker of the run")
    args = ap.parse_args(argv)
    handler = StreamingHandler(args.checkpoint, args.device, args.mode, args.kalman)
    print(json.dumps(handler.info()), flush=True)
    serve(handler, args.host, args.port, ready=lambda port: print(f"listening on {args.host}:{port}", flush=True))


if __name__ == "__main__":
    main()
