"""Streaming lane detection with per-stream temporal state and feature caching.

What is cached, and why it is safe
----------------------------------
Every temporal model encodes each frame with a backbone whose output, in
eval mode, depends on that frame only (shared weights, BatchNorm running
statistics, dropout off). Fusion (v0.2/v0.4 weights, v0.3 gates, v0.5 warp +
gate) depends on the *current* frame and is recomputed every step. So the
cache stores one feature map per past frame (``encode(frame)["history"]``)
and each step costs one backbone pass plus fusion and head, instead of
``num_frames`` backbone passes. ``tests/test_streaming.py`` checks that the
cached result equals the full clip forward pass.

Temporal context and missing frames
-----------------------------------
History frame ``k`` of frame index ``t`` is ``t - k * temporal_step`` (same
rule as training). A missing history frame (stream start, dropped frame,
``history_length`` limit) is replaced by the nearest newer frame of the
context - exactly what ``TemporalLaneDataset`` does - and reported in
``FrameResult.fallbacks``.

Stream state
------------
Streams are independent (``stream_id``); each keeps its own cache. State is
reset when ``reset_stream`` is called, when ``sequence_id`` changes, when the
source frame size changes, or when frame indices go backwards. Frame indices
come from ``frame_index``, from ``timestamp`` x ``nominal_fps``, or from a
per-stream counter (every call = next frame; drops cannot be detected then).

Recurrent models (ConvGRU fusion)
---------------------------------
``mode="cached"`` runs them exactly as trained (the GRU restarts from zero
over the window of cached features). ``mode="carry"`` keeps ONE hidden
state per chain instead: the state of frame ``t`` is updated from the state
of frame ``t - temporal_step`` (``temporal_step`` interleaved chains, so
consecutive updates are as far apart as in training). Memory then reaches
back to the start of the stream; a missing predecessor restarts the chain.

Output tracker
--------------
``tracker=KalmanParams(...)`` filters the decoded lane points of each
stream with the causal Kalman tracker of ``tac_ufld.evaluation.tracking``.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from tac_ufld.decoding import decode_logits
from tac_ufld.inference.backends import Backend, TorchBackend
from tac_ufld.inference.loading import FramePreprocessor, LoadedModel
from tac_ufld.postprocess import PostprocessParams, lanes_from_prediction

MODES = ("cached", "recompute", "carry")


@dataclass
class FrameResult:
    stream_id: str
    frame_index: int
    timestamp: float | None
    lanes: list[np.ndarray | None]            # per slot, original pixels
    lane_confidence: list[float | None]       # mean existence probability of each drawn lane
    exist: np.ndarray                         # (A, L)
    x_model: np.ndarray                       # (A, L) model pixels
    history_indices: list[int]                # frame index used for each history position (oldest first)
    fallbacks: int                            # history positions filled with a newer frame
    reset_reason: str | None
    dropped_before: int                       # frames missing since the previous call of this stream
    latency_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class _StreamState:
    cache: "OrderedDict[int, object]" = field(default_factory=OrderedDict)
    last_index: int | None = None
    source_size: tuple[int, int] | None = None
    sequence_id: str | None = None
    t0: float | None = None
    frames_seen: int = 0
    tracker: object | None = None


class StreamingLaneDetector:
    def __init__(self, loaded: LoadedModel, backend: Backend | None = None, mode: str = "cached",
                 history_length: int | None = None, nominal_fps: float | None = None,
                 postprocess: PostprocessParams | None = None,
                 valid_y_range: tuple[float, float] | None = None, tracker=None) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        model = loaded.model
        if model.training:
            raise RuntimeError("feature caching needs eval mode (BatchNorm/dropout would make features stale)")
        self.loaded = loaded
        self.backend = backend or TorchBackend(model, loaded.device)
        if mode == "carry" and not (getattr(model, "recurrent", False) and hasattr(self.backend, "step")):
            raise ValueError("mode 'carry' needs a recurrent model (ufld_v07, lite_v06) on the PyTorch backend")
        self.mode = mode
        self.tracker_params = tracker
        self.temporal = loaded.temporal
        self.num_frames = loaded.num_frames
        self.step = int(loaded.card["temporal_step"])
        max_history = self.num_frames - 1
        self.history_length = max_history if history_length is None else int(history_length)
        if not 0 <= self.history_length <= max_history:
            raise ValueError(f"history_length must be in [0, {max_history}] for this model")
        self.nominal_fps = nominal_fps
        self.postprocess = postprocess or loaded.postprocess
        self.valid_y_range = valid_y_range
        self.preprocess = FramePreprocessor(loaded.cfg)
        self.anchors = loaded.row_anchors
        self._streams: dict[str, _StreamState] = {}

    # ------------------------------------------------------------ state

    def reset_stream(self, stream_id: str = "default") -> None:
        self._streams.pop(stream_id, None)

    def streams(self) -> list[str]:
        return list(self._streams)

    def cache_size(self, stream_id: str = "default") -> int:
        state = self._streams.get(stream_id)
        return len(state.cache) if state else 0

    def _frame_index(self, state: _StreamState, timestamp: float | None, frame_index: int | None) -> int:
        if frame_index is not None:
            return int(frame_index)
        if timestamp is not None and self.nominal_fps:
            if state.t0 is None:
                state.t0 = float(timestamp)
            return int(round((float(timestamp) - state.t0) * self.nominal_fps))
        return 0 if state.last_index is None else state.last_index + 1

    def _context(self, index: int, available: set[int]) -> tuple[list[int], int]:
        """History indices (oldest first) after the dataset's fallback rule."""
        wanted = [index - k * self.step for k in range(self.num_frames - 1, 0, -1)]
        usable = [w if (w in available and (index - w) <= self.history_length * self.step) else None
                  for w in wanted]
        chain = usable + [index]
        fallbacks = 0
        for i in range(len(chain) - 2, -1, -1):
            if chain[i] is None:
                chain[i] = chain[i + 1]
                fallbacks += 1
        return chain[:-1], fallbacks

    # ------------------------------------------------------------ inference

    def infer_frame(self, frame, stream_id: str = "default", timestamp: float | None = None,
                    frame_index: int | None = None, sequence_id: str | None = None) -> FrameResult:
        """``frame``: (H, W, 3) uint8 RGB array or an image path."""
        if isinstance(frame, (str, Path)):
            from PIL import Image

            with Image.open(frame) as img:
                frame = np.asarray(img.convert("RGB"))
        state = self._streams.setdefault(stream_id, _StreamState())
        size = (int(frame.shape[1]), int(frame.shape[0]))
        reason = None
        if state.frames_seen and sequence_id is not None and sequence_id != state.sequence_id:
            reason = f"sequence changed ({state.sequence_id} -> {sequence_id})"
        elif state.source_size is not None and size != state.source_size:
            reason = f"frame size changed {state.source_size} -> {size}"
        index_hint = self._frame_index(state, timestamp, frame_index)
        if reason is None and state.last_index is not None and index_hint <= state.last_index:
            reason = f"frame index went backwards ({state.last_index} -> {index_hint})"
        if reason is not None:
            state = self._streams[stream_id] = _StreamState()
            index = self._frame_index(state, timestamp, frame_index)
        else:
            index = index_hint
        dropped = 0 if state.last_index is None else max(0, index - state.last_index - 1)
        state.source_size, state.sequence_id = size, sequence_id
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        x = self.preprocess(frame).unsqueeze(0)
        timings["preprocess_ms"] = (time.perf_counter() - t0) * 1e3

        self.backend.synchronize()
        t1 = time.perf_counter()
        history_idx, fallbacks = [], 0
        if not self.temporal:
            enc = self.backend.encode(x)
            logits = self.backend.head([], enc["current"])
        elif self.mode == "carry":
            enc = self.backend.encode(x)
            previous = index - self.step
            prev_state = state.cache.get(previous)
            history_idx, fallbacks = ([previous], 0) if prev_state is not None else ([], 1)
            logits, state.cache[index] = self.backend.step(enc["current"], prev_state)
            for old in [i for i in state.cache if i <= previous]:
                del state.cache[old]
        else:
            history_idx, fallbacks = self._context(index, set(state.cache))
            if self.mode == "cached":
                enc = self.backend.encode(x)
                # a fallback to the current frame uses its history features, as the
                # dataset feeds the current image into the history encoder
                state.cache[index] = enc.get("history", enc["current"])
                logits = self.backend.head([state.cache[i] for i in history_idx], enc["current"])
            else:  # recompute: keep preprocessed frames, run the whole clip
                state.cache[index] = x
                clip = torch.cat([state.cache[i] for i in history_idx] + [x], dim=0).unsqueeze(0)
                logits = self.backend.full(clip)
            horizon = index - (self.num_frames - 1) * self.step
            for old in [i for i in state.cache if i < horizon]:
                del state.cache[old]
        self.backend.synchronize()
        timings["model_ms"] = (time.perf_counter() - t1) * 1e3

        t2 = time.perf_counter()
        logits_t = logits if isinstance(logits, torch.Tensor) else torch.from_numpy(np.asarray(logits))
        exist_t, bins_t = decode_logits(logits_t.float())
        exist = exist_t[0].cpu().numpy()
        grid = logits_t.shape[1] - 1
        x_model = (bins_t[0] * (self.loaded.cfg.data.img_w - 1) / (grid - 1)).cpu().numpy()
        if self.tracker_params is not None:
            from tac_ufld.evaluation.tracking import LaneKalmanTracker

            if state.tracker is None:
                state.tracker = LaneKalmanTracker(self.tracker_params)
            exist, x_model = state.tracker.update(exist, x_model, index)
        lanes = lanes_from_prediction(exist, x_model, self.anchors, self.postprocess, self.loaded.model_size,
                                      size, self.valid_y_range)
        confidence = []
        for slot, lane in enumerate(lanes):
            on = exist[:, slot] >= self.postprocess.threshold
            confidence.append(float(exist[on, slot].mean()) if lane is not None and on.any() else None)
        timings["postprocess_ms"] = (time.perf_counter() - t2) * 1e3
        timings["total_ms"] = sum(timings.values())

        state.last_index = index
        state.frames_seen += 1
        return FrameResult(stream_id, index, timestamp, lanes, confidence, exist, x_model,
                           history_idx, fallbacks, reason, dropped, timings)

    def infer_video(self, video_path: str | Path, stream_id: str | None = None, max_frames: int | None = None,
                    every: int = 1) -> Iterator[tuple[np.ndarray, FrameResult]]:
        """Read a video incrementally (one frame in memory at a time) and yield
        (RGB frame, result). Frame indices are the video positions, so
        ``every > 1`` behaves like dropped frames for the temporal context."""
        import cv2

        from tac_ufld.utils import open_video

        stream_id = stream_id or f"video:{Path(video_path).name}"
        self.reset_stream(stream_id)
        cap = open_video(video_path)
        try:
            position, produced = -1, 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                position += 1
                if position % every:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                yield rgb, self.infer_frame(rgb, stream_id, timestamp=ts, frame_index=position)
                produced += 1
                if max_frames is not None and produced >= max_frames:
                    break
        finally:
            cap.release()
