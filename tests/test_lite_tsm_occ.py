"""Light temporal variants of 2026-10-19: the online temporal shift (lite_tsm, its static control) and the
occlusion-gated ConvGRU write (lite_v06_occ). The generic checks (shapes, losses, window streaming, ONNX
encoder/head) run for them through the VARIANTS parametrisations of the other test files."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from tac_ufld.evaluation.efficiency import count_macs, count_parameters
from tac_ufld.inference.loading import FramePreprocessor, load_model
from tac_ufld.inference.streaming import StreamingLaneDetector
from tac_ufld.models.fusion import ConvGRUFusion, OcclusionGatedConvGRUFusion
from tac_ufld.models.lite import LiteTSM, temporal_shift
from tac_ufld.models.registry import VARIANTS, build_model, warm_start
from tests.test_streaming import _cfg, _checkpoint, _frames


def _nontrivial(model: nn.Module) -> nn.Module:
    """Zero-initialised layers (lite cls head, GRU read-out) would hide differences."""
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1 and not p.any():
                p.normal_(0.0, 0.05)
    return model.eval()


class _StreamingStep(nn.Module):
    """What one carry-mode frame computes: encoder + one state update (state = zeros)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, frame: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.model.recurrent_step(self.model.encode_frames(frame)["current"], None)["logits"]}


# ------------------------------------------------------------------ temporal shift


def test_temporal_shift_moves_one_fold_from_the_previous_frame():
    x = torch.randn(2, 3, 16, 4, 5)
    y = temporal_shift(x, 2)
    assert torch.equal(y[:, 0, :2], torch.zeros_like(y[:, 0, :2]))      # first frame: zero padding
    assert torch.equal(y[:, 1:, :2], x[:, :-1, :2])                       # shifted from t-1
    assert torch.equal(y[:, :, 2:], x[:, :, 2:])                          # the rest untouched


def test_tsm_is_the_baseline_network_with_zero_extra_parameters_and_streaming_macs():
    cfg = _cfg()
    base = build_model("lite_baseline", cfg).eval()
    tsm = build_model("lite_tsm", cfg).eval()
    assert isinstance(tsm, LiteTSM) and tsm.folds == (4, 8, 12)           # 1/8 of 32, 64, 96 channels
    assert set(tsm.state_dict()) == set(base.state_dict())
    assert warm_start(tsm, base.state_dict()) == ["<all>"]
    assert count_parameters(tsm) == count_parameters(base)
    frame = torch.randn(1, 3, cfg.data.img_h, cfg.data.img_w)
    base_macs = count_macs(base, frame)
    assert count_macs(_StreamingStep(tsm), frame) == base_macs           # carry: one baseline pass per frame
    head_macs = count_macs(tsm.head, base.backbone(frame))
    clip = frame.unsqueeze(1).expand(1, cfg.data.num_frames, *frame.shape[1:])
    assert count_macs(tsm, clip) == cfg.data.num_frames * (base_macs - head_macs) + head_macs


def test_tsm_uses_history_causally_and_does_not_start_as_its_baseline():
    cfg = _cfg()
    base = _nontrivial(build_model("lite_baseline", cfg))
    tsm = build_model("lite_tsm", cfg).eval()
    warm_start(tsm, base.state_dict())
    x = torch.randn(2, cfg.data.num_frames, 3, cfg.data.img_h, cfg.data.img_w)
    other = x.clone()
    other[:, 0] = torch.randn_like(other[:, 0])
    with torch.no_grad():
        out = tsm(x)["logits"]
        assert not torch.allclose(out, tsm(other)["logits"])              # the oldest frame reaches the output
        assert not torch.allclose(out, base(x[:, -1])["logits"])          # the shift acts from the warm start on
        future = x.clone()
        future[:, -1] = torch.randn_like(future[:, -1])
        p1 = tsm.per_frame_logits(x, (1,))[0]
        assert torch.allclose(p1, tsm.per_frame_logits(future, (1,))[0])  # causal: frame 1 ignores frame 2
        assert torch.allclose(tsm.per_frame_logits(x, (2,))[0], out)


def test_tsm_window_split_and_carried_slices_equal_the_clip_forward():
    cfg = _cfg()
    tsm = _nontrivial(build_model("lite_tsm", cfg))
    x = torch.randn(2, cfg.data.num_frames, 3, cfg.data.img_h, cfg.data.img_w)
    with torch.no_grad():
        ref = tsm(x)["logits"]
        enc = [tsm.encode_frames(x[:, i])["current"] for i in range(x.shape[1])]
        assert torch.allclose(tsm.forward_features(enc[:-1], enc[-1])["logits"], ref, atol=1e-5)
        state = None
        for e in enc:
            out = tsm.recurrent_step(e, state)
            state = out["state"]
        assert torch.allclose(out["logits"], ref, atol=1e-5)
    assert [tuple(s.shape[1:2]) for s in state] == [(4,), (8,), (12,)]   # one slice per shifted block


def test_tsm_carry_mode_is_a_window_of_shift_depth_plus_one_frames(tmp_path):
    """The carried slices give frame t exactly the clip forward over t - k*step, k = depth..0 (depth = number of
    shifted blocks): finite memory, unlike the ConvGRU. With 3 shifted blocks that is 4 frames, one more than the
    3-frame training clip, whose deepest dependency was zero padding."""
    from tests.test_streaming import _reference

    cfg = _cfg()
    loaded = load_model(_checkpoint(tmp_path, "lite_tsm", cfg))
    depth, step = len(loaded.model.shift_blocks), cfg.data.temporal_step
    det = StreamingLaneDetector(loaded, mode="carry")
    frames = _frames(14)
    for t, frame in enumerate(frames):
        res = det.infer_frame(frame, frame_index=t)
        ctx = [t - k * step for k in range(depth, -1, -1) if t - k * step >= 0]
        np.testing.assert_allclose(res.exist, _reference(loaded, frames, ctx), atol=1e-5)
        assert det.cache_size() <= step
    pre = FramePreprocessor(loaded.cfg)
    clip = torch.stack([pre(f) for f in frames[:depth + 2]]).unsqueeze(0).requires_grad_(True)
    loaded.model(clip)["logits"].sum().backward()
    reach = clip.grad.abs().flatten(2).amax(-1)[0]           # per clip position: does the output depend on it?
    # the deepest shift reaches t - depth (which the 3-frame training clip cannot supply), nothing older does
    assert reach[0] == 0 and bool((reach[1:] > 0).all())


def test_static_tsm_control_is_registered_and_cannot_carry(tmp_path):
    from tac_ufld.inference.card import model_card

    cfg = _cfg()
    assert VARIANTS["lite_tsm"].static_reference == "lite_tsm_static"
    assert VARIANTS["lite_tsm_static"].history == "current"
    assert model_card(cfg, "lite_tsm_static")["static_history"] and not model_card(cfg, "lite_tsm")["static_history"]
    for key in ("lite_tsm", "lite_tsm_static", "lite_v06_occ"):
        spec = VARIANTS[key]
        assert spec.family == "lite" and spec.reference == "lite_baseline" and spec.warm_start == "lite_baseline"
        assert spec.budget_reference == "lite_baseline_ct"
    loaded = load_model(_checkpoint(tmp_path, "lite_tsm_static", cfg))
    with pytest.raises(ValueError, match="static-history"):
        StreamingLaneDetector(loaded, mode="carry")
    StreamingLaneDetector(loaded, mode="cached")


# ------------------------------------------------------------- occlusion-gated GRU


class _FixedVisibility(nn.Module):
    """Visibility logits +30 (visible) on the left half of the map, -30 (occluded) on the right half."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.full_like(x[:, :1], 30.0)
        logits[..., x.shape[-1] // 2:] = -30.0
        return logits


def test_occlusion_gate_keeps_the_memory_of_invisible_cells():
    torch.manual_seed(0)
    gated, plain = OcclusionGatedConvGRUFusion(32, hidden=8), ConvGRUFusion(32, hidden=8)
    plain.load_state_dict({k: v for k, v in gated.state_dict().items() if not k.startswith("visibility.")})
    gated.eval()
    plain.eval()
    feats, state = torch.randn(2, 32, 6, 8), torch.randn(2, 8, 6, 8)
    with torch.no_grad():
        start = gated(torch.stack([feats, feats], dim=1))[1]["visibility"]
        assert torch.allclose(start, torch.full_like(start, torch.sigmoid(torch.tensor(3.0)).item()))  # starts open
        gated.visibility = _FixedVisibility()
        new, vis = gated.gated_step(feats, state)
        expected = plain.step(feats, state)
        assert torch.allclose(new[..., :4], expected[..., :4], atol=1e-6)   # visible: the plain ConvGRU write
        assert torch.allclose(new[..., 4:], state[..., 4:], atol=1e-6)      # occluded: the memory is kept
        assert not torch.allclose(expected[..., 4:], state[..., 4:])        # (the plain GRU would overwrite it)
        assert vis.shape == (2, 1, 6, 8)


def test_occlusion_gated_variant_learns_the_visibility():
    cfg = _cfg()
    model = build_model("lite_v06_occ", cfg)
    assert isinstance(model.fusion, OcclusionGatedConvGRUFusion)
    extra = count_parameters(model) - count_parameters(build_model("lite_v06", cfg))
    assert extra == cfg.model.recurrent_hidden * 9 + 1                    # one 3x3 conv to one channel
    _nontrivial(model)                                                     # past the zero-initialised read-out
    x = torch.randn(2, cfg.data.num_frames, 3, cfg.data.img_h, cfg.data.img_w)
    out = model(x)
    assert out["visibility"].shape == (2,) and bool(((out["visibility"] > 0) & (out["visibility"] < 1)).all())
    out["logits"].sum().backward()
    assert model.fusion.visibility.weight.grad is not None and model.fusion.visibility.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------- ONNX carry graph


@pytest.mark.parametrize("variant", ["lite_tsm", "lite_v06_occ"])
def test_onnx_step_graph_carries_the_state_like_pytorch(tmp_path, variant):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from tac_ufld.deploy.onnx_export import export_model
    from tac_ufld.deploy.ort_backend import OnnxRuntimeBackend

    cfg = _cfg()
    loaded = load_model(_checkpoint(tmp_path, variant, cfg))
    manifest = export_model(loaded, tmp_path / "deploy")
    carry = manifest["temporal_state"]["carry"]
    assert manifest["graphs"]["step"]["tuple_state"] == (variant == "lite_tsm")
    assert carry["bytes_per_chain"] > 0 and len(carry["state_shapes"]) == (3 if variant == "lite_tsm" else 1)
    backend = OnnxRuntimeBackend(tmp_path / "deploy", providers=["CPUExecutionProvider"])
    ref, cand = StreamingLaneDetector(loaded, mode="carry"), StreamingLaneDetector(loaded, backend=backend, mode="carry")
    for t, frame in enumerate(_frames(10)):
        r, c = ref.infer_frame(frame, frame_index=t), cand.infer_frame(frame, frame_index=t)
        np.testing.assert_allclose(r.exist, c.exist, atol=1e-4)


# -------------------------------------------------------------------- end to end


def test_new_lite_variants_run_end_to_end(tiny_config):
    """Training (warm start), evaluation with the static-history ablation and carried state, robustness."""
    import pandas as pd

    from tac_ufld.evaluation.robustness import evaluate_run
    from tac_ufld.experiment import ExperimentRunner

    cfg = tiny_config
    cfg.name = "tiny_tsm_occ"
    cfg.train.seeds = [1]
    cfg.train.epochs = 1
    cfg.hpo.enabled = False
    cfg.data.augmentation.current_frame_prob = 0.5
    cfg.evaluation.postprocess_grid = {"threshold": [0.5], "min_points": [2], "poly_degree": [0]}
    variants = ["lite_baseline", "lite_tsm", "lite_tsm_static", "lite_v06_occ"]
    runner = ExperimentRunner(cfg, variants=variants)
    runner.run()
    results = pd.read_csv(runner.out / "all_results.csv")
    assert set(results["variant"]) == set(variants)
    carry = results[results["input"] == "carry"]
    assert set(carry["variant"]) == {"lite_tsm", "lite_v06_occ"}             # not the static control
    static = results[results["input"] == "static_history"]
    assert set(static["variant"]) == {"lite_tsm", "lite_v06_occ"}
    paired = pd.read_csv(runner.out / "report" / "paired_tests_test_tuned.csv")
    assert {("lite_tsm", "lite_baseline"), ("lite_tsm", "lite_tsm_static")} <= set(zip(paired["variant"],
                                                                                   paired["reference"]))
    summary = evaluate_run(runner.out, "cpu", workers=0)
    assert {"current", "persistent", "held_out"} <= set(summary["protocol"])
    assert set(summary["variant"]) == set(variants)


# ------------------------------------------------- robustness protocol "independent"


def test_independent_degradation_draws_every_frame_separately():
    """Same op on every frame, but its own draws per frame (fresh noise, other boxes); reproducible per seed."""
    import random

    from tac_ufld.config import AugmentationConfig
    from tac_ufld.data.transforms import degrade_current_frame

    same = torch.rand(1, 3, 48, 64).expand(3, -1, -1, -1).contiguous()   # identical frames: any difference = draws
    cfg = AugmentationConfig()
    for op in ("occlude", "noise", "blur", "darken"):
        run = lambda: degrade_current_frame(same, cfg, op, random.Random(5), torch.Generator().manual_seed(5),  # noqa: E731
                                            scope="independent")
        out = run()
        assert all(not torch.allclose(out[i], same[i]) for i in range(3)), op       # every frame degraded
        assert not torch.allclose(out[0], out[2]), op                                # ... differently
        assert torch.equal(out, run()), op                                           # reproducible
        clip = degrade_current_frame(same, cfg, op, random.Random(5), torch.Generator().manual_seed(5), scope="clip")
        assert torch.allclose(clip[0], clip[2]), op                                  # unlike "clip"
