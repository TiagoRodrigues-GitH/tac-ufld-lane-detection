"""Streaming inference: exactness of feature caching, stream isolation,
resets, dropped frames and video input."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tac_ufld.config import PROJECT_ROOT, load_config
from tac_ufld.decoding import decode_logits
from tac_ufld.inference.card import model_card
from tac_ufld.inference.loading import FramePreprocessor, load_model
from tac_ufld.inference.streaming import StreamingLaneDetector
from tac_ufld.models.registry import VARIANTS, build_model

TEMPORAL = [k for k, s in VARIANTS.items() if s.temporal]


def _cfg(**extra):
    return load_config(PROJECT_ROOT / "configs" / "elas_smoke.yaml", {
        "data.img_h": 64, "data.img_w": 96, "data.griding_num": 24, "data.num_row_anchors": 8,
        "model.pretrained": False, "device": "cpu", **extra})


def _checkpoint(tmp_path: Path, variant: str, cfg) -> Path:
    torch.manual_seed(0)
    model = build_model(variant, cfg, pretrained=False)
    for m in model.modules():  # non-trivial BatchNorm statistics
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.2, 0.2)
            m.running_var.uniform_(0.5, 1.5)
    with torch.no_grad():  # zero-initialised layers (lite head, ConvGRU read-out) would hide the fusion
        for p in model.parameters():
            if p.dim() > 1 and not p.any():
                p.normal_(0.0, 0.05)
    path = tmp_path / f"{variant}.pt"
    torch.save({"state_dict": model.state_dict(), "variant": variant, "config": cfg.to_dict(),
                "card": model_card(cfg, variant)}, path)
    return path


def _frames(n: int = 12, seed: int = 0, size=(120, 160)) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (*size, 3), dtype=np.uint8) for _ in range(n)]


def _reference(loaded, frames, context: list[int]):
    pre = FramePreprocessor(loaded.cfg)
    clip = torch.stack([pre(frames[i]) for i in context]).unsqueeze(0)
    with torch.no_grad():
        exist, bins = decode_logits(loaded.model(clip)["logits"])
    return exist[0].numpy()


def _dataset_context(t: int, available: set[int], step: int, num_frames: int) -> list[int]:
    """TemporalLaneDataset._context rule, on frame indices."""
    chain = [t - k * step if (t - k * step) in available else None for k in range(num_frames - 1, 0, -1)] + [t]
    for i in range(len(chain) - 2, -1, -1):
        if chain[i] is None:
            chain[i] = chain[i + 1]
    return chain


@pytest.mark.parametrize("variant", TEMPORAL)
@pytest.mark.parametrize("mode", ["cached", "recompute"])
def test_streaming_equals_clip_forward(tmp_path, variant, mode):
    cfg = _cfg()
    loaded = load_model(_checkpoint(tmp_path, variant, cfg))
    det = StreamingLaneDetector(loaded, mode=mode)
    frames = _frames()
    for t in range(len(frames)):
        res = det.infer_frame(frames[t], frame_index=t)
        ctx = ([t] * cfg.data.num_frames if loaded.card.get("static_history")   # capacity control
               else _dataset_context(t, set(range(t)), cfg.data.temporal_step, cfg.data.num_frames))
        np.testing.assert_allclose(res.exist, _reference(loaded, frames, ctx), atol=1e-5)
        assert res.history_indices == ctx[:-1]
        assert det.cache_size() <= (cfg.data.num_frames - 1) * cfg.data.temporal_step + 1


def test_tiny_history_encoder_is_cached_exactly(tmp_path):
    cfg = _cfg(**{"model.lite_history_encoder": "tiny"})
    loaded = load_model(_checkpoint(tmp_path, "lite_v05", cfg))
    det = StreamingLaneDetector(loaded)
    frames = _frames(8)
    for t in range(8):
        res = det.infer_frame(frames[t], frame_index=t)
        ctx = _dataset_context(t, set(range(t)), 2, 3)
        np.testing.assert_allclose(res.exist, _reference(loaded, frames, ctx), atol=1e-5)


def test_dropped_frames_use_the_dataset_fallback(tmp_path):
    cfg = _cfg()
    loaded = load_model(_checkpoint(tmp_path, "ufld_v03", cfg))
    det = StreamingLaneDetector(loaded)
    frames = _frames(8)
    seen = []
    for t in (0, 1, 2, 5, 6):  # frames 3 and 4 dropped
        res = det.infer_frame(frames[t], frame_index=t)
        ctx = _dataset_context(t, set(seen), 2, 3)
        np.testing.assert_allclose(res.exist, _reference(loaded, frames, ctx), atol=1e-5)
        seen.append(t)
    assert res.dropped_before == 0 and ctx == [2, 6, 6] and res.fallbacks == 1
    assert det.infer_frame(frames[7], frame_index=7).history_indices == [5, 5]


def test_streams_are_isolated(tmp_path):
    cfg = _cfg()
    loaded = load_model(_checkpoint(tmp_path, "ufld_v02", cfg))
    a, b = _frames(6, seed=1), _frames(6, seed=2)
    mixed = StreamingLaneDetector(loaded)
    alone_a, alone_b = StreamingLaneDetector(loaded), StreamingLaneDetector(loaded)
    for t in range(6):
        ra, rb = mixed.infer_frame(a[t], "A"), mixed.infer_frame(b[t], "B")
        np.testing.assert_array_equal(ra.exist, alone_a.infer_frame(a[t], "A").exist)
        np.testing.assert_array_equal(rb.exist, alone_b.infer_frame(b[t], "B").exist)
    assert sorted(mixed.streams()) == ["A", "B"]
    mixed.reset_stream("A")
    assert mixed.streams() == ["B"] and mixed.cache_size("B") > 0


def test_resets_on_sequence_size_and_backwards_index(tmp_path):
    cfg = _cfg()
    loaded = load_model(_checkpoint(tmp_path, "lite_v05", cfg))
    det = StreamingLaneDetector(loaded)
    f = _frames(4)
    det.infer_frame(f[0], sequence_id="S1", frame_index=10)
    assert det.infer_frame(f[1], sequence_id="S1", frame_index=12).reset_reason is None
    assert "sequence changed" in det.infer_frame(f[2], sequence_id="S2", frame_index=13).reset_reason
    assert "backwards" in det.infer_frame(f[3], sequence_id="S2", frame_index=5).reset_reason
    other_size = np.zeros((90, 160, 3), dtype=np.uint8)
    res = det.infer_frame(other_size, sequence_id="S2", frame_index=6)
    assert "size changed" in res.reset_reason and res.history_indices == [6, 6]


def test_timestamps_detect_drops_and_history_length_zero_is_static(tmp_path):
    cfg = _cfg()
    loaded = load_model(_checkpoint(tmp_path, "ufld_v04", cfg))
    det = StreamingLaneDetector(loaded, nominal_fps=10.0)
    f = _frames(4)
    det.infer_frame(f[0], timestamp=5.0)
    res = det.infer_frame(f[1], timestamp=5.3)
    assert res.frame_index == 3 and res.dropped_before == 2
    static = StreamingLaneDetector(loaded, history_length=0)
    for t in range(3):
        r = static.infer_frame(f[t], frame_index=t)
    np.testing.assert_allclose(r.exist, _reference(loaded, f, [2, 2, 2]), atol=1e-5)
    with pytest.raises(ValueError):
        StreamingLaneDetector(loaded, history_length=5)


def test_single_frame_model_and_training_mode_guard(tmp_path):
    cfg = _cfg()
    loaded = load_model(_checkpoint(tmp_path, "ufld_baseline", cfg))
    res = StreamingLaneDetector(loaded).infer_frame(_frames(1)[0])
    assert res.history_indices == [] and set(res.latency_ms) >= {"preprocess_ms", "model_ms", "postprocess_ms"}
    loaded.model.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        StreamingLaneDetector(loaded)


def test_infer_video_reads_incrementally_with_unicode_path(tmp_path):
    cfg = _cfg()
    loaded = load_model(_checkpoint(tmp_path, "ufld_v02", cfg))
    ascii_video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(ascii_video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for frame in _frames(9):
        writer.write(frame)
    writer.release()
    video = tmp_path / "vídeo_ç.mp4"
    video.write_bytes(ascii_video.read_bytes())
    det = StreamingLaneDetector(loaded)
    out = list(det.infer_video(video, every=2))
    assert [r.frame_index for _, r in out] == [0, 2, 4, 6, 8]
    assert out[2][1].history_indices == [0, 2]
    assert len(list(det.infer_video(video, max_frames=3))) == 3
