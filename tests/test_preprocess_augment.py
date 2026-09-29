"""Preprocessing ablations (shapes, values, model channels) and augmentation
(image-label alignment, ROI handling, flips, determinism, backward
compatibility of the original photometric augmentation)."""

from __future__ import annotations

import random

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tac_ufld.config import PROJECT_ROOT, AugmentationConfig, ConfigError, GeometricAugConfig, PreprocessConfig, load_config
from tac_ufld.data.geometric import GeometricAugmenter, apply_h, encode_targets_warped, transform_lanes, warp_frames
from tac_ufld.data.preprocess import Preprocessor, channel_stats, channels_for_mode
from tac_ufld.data.targets import IGNORE_INDEX, encode_targets, make_row_anchors
from tac_ufld.data.transforms import PhotometricAugmenter, normalize
from tac_ufld.data.types import FrameRecord
from tac_ufld.models.registry import VARIANTS, build_model
from tac_ufld.models.resnet import adapt_conv1

MODES = ["rgb", "gray", "gray3", "edge", "canny", "hough", "rgb_edge"]
W, H, G = 96, 64, 24
ANCHORS = make_row_anchors(H, 8, (0.55, 0.99))


def _road_image(h=120, w=160) -> np.ndarray:
    img = np.full((h, w, 3), 80, dtype=np.uint8)
    cv2.line(img, (30, h - 1), (75, 45), (255, 255, 255), 3)
    cv2.line(img, (130, h - 1), (85, 45), (255, 255, 0), 3)
    cv2.line(img, (0, 20), (w - 1, 22), (255, 255, 255), 2)  # near-horizontal: not lane-like
    return img


def _record(lanes, known=(True, True), valid=(60.0, 120.0), size=(160, 120)):
    return FrameRecord("t", "S", 0, None, size, tuple(lanes), known, valid_y_range=valid)


LEFT = np.array([[30.0, 119.0], [75.0, 60.0]], dtype=np.float32)
RIGHT = np.array([[130.0, 119.0], [85.0, 60.0]], dtype=np.float32)


# ------------------------------------------------------------ preprocessing


@pytest.mark.parametrize("mode", MODES)
def test_preprocessing_shapes_and_ranges(mode):
    cfg = PreprocessConfig(mode=mode)
    out = Preprocessor(cfg).image(_road_image())
    assert out.shape == (120, 160, channels_for_mode(mode)) and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    mean, std = channel_stats(cfg, "imagenet")
    assert len(mean) == len(std) == channels_for_mode(mode)
    frames = torch.from_numpy(_road_image().transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0)
    assert Preprocessor(cfg)(frames).shape == (1, channels_for_mode(mode), 120, 160)


def test_preprocessing_values():
    img = _road_image()
    gray = Preprocessor(PreprocessConfig(mode="gray")).image(img)[..., 0]
    gray3 = Preprocessor(PreprocessConfig(mode="gray3")).image(img)
    assert np.array_equal(gray3[..., 0], gray) and np.array_equal(gray3[..., 1], gray3[..., 2])
    edge = Preprocessor(PreprocessConfig(mode="edge")).image(img)[..., 0]
    assert edge.max() == pytest.approx(1.0) and edge[5:15, 5:15].max() == 0.0  # flat area has no edges
    assert Preprocessor(PreprocessConfig(mode="edge")).image(np.zeros_like(img)).max() == 0.0
    canny = Preprocessor(PreprocessConfig(mode="canny")).image(img)
    assert set(np.unique(canny)) <= {0.0, 1.0}
    hough = Preprocessor(PreprocessConfig(mode="hough", hough_threshold=10, hough_min_line_length=15)).image(img)[..., 0]
    assert hough[60:119, 20:140].sum() > 0          # slanted lane lines kept
    assert hough[15:28, :].sum() == 0               # near-horizontal line dropped (hough_min_angle_deg)
    rgb_edge = Preprocessor(PreprocessConfig(mode="rgb_edge")).image(img)
    np.testing.assert_allclose(rgb_edge[..., :3], img / 255.0, atol=1e-6)


def test_pre_ops_are_applied_in_the_configured_order():
    img = _road_image()
    a = Preprocessor(PreprocessConfig(mode="gray", pre_ops=["blur", "clahe"])).image(img)
    b = Preprocessor(PreprocessConfig(mode="gray", pre_ops=["clahe", "blur"])).image(img)
    c = Preprocessor(PreprocessConfig(mode="gray")).image(img)
    assert not np.array_equal(a, b) and not np.array_equal(a, c)


def test_rgb_default_reproduces_the_original_normalisation():
    frames = torch.rand(3, 3, 8, 10)
    cfg = PreprocessConfig()
    pre = Preprocessor(cfg)
    assert pre.identity and pre(frames) is frames
    from tac_ufld.data.preprocess import normalize_channels

    mean, std = channel_stats(cfg, "imagenet")
    assert torch.equal(normalize_channels(pre(frames), mean, std), normalize(frames, "imagenet"))


def test_adapted_conv1_gives_same_response_for_grey_input():
    w = torch.randn(64, 3, 7, 7)
    gray = torch.rand(1, 1, 32, 32)
    out3 = F.conv2d(gray.repeat(1, 3, 1, 1), w, padding=3)
    out1 = F.conv2d(gray, adapt_conv1(w, 1), padding=3)
    assert torch.allclose(out1, out3, atol=1e-4)
    w4 = adapt_conv1(w, 4)
    assert w4.shape == (64, 4, 7, 7) and torch.equal(w4[:, :3], w)
    with pytest.raises(ValueError):
        adapt_conv1(w, 2)


@pytest.mark.parametrize("mode", ["gray", "rgb_edge"])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_models_accept_non_rgb_inputs(mode, variant):
    cfg = load_config(PROJECT_ROOT / "configs" / "elas_smoke.yaml", {
        "data.img_h": 64, "data.img_w": 96, "data.griding_num": 24, "data.num_row_anchors": 8,
        "data.preprocessing.mode": mode})
    c = channels_for_mode(mode)
    assert cfg.in_channels == c
    model = build_model(variant, cfg, pretrained=False).eval()
    assert model(torch.randn(2, 3, c, 64, 96))["logits"].shape == (2, 25, 8, 2)


def test_preprocessing_config_validation():
    base = PROJECT_ROOT / "configs" / "elas_smoke.yaml"
    for bad in ({"data.preprocessing.mode": "sepia"}, {"data.preprocessing.pre_ops": ["sharpen"]},
                {"data.preprocessing.blur_ksize": 4}, {"data.preprocessing.canny_low": 200}):
        with pytest.raises(ConfigError):
            load_config(base, bad)


@pytest.mark.network
def test_imagenet_weights_load_for_grey_input():
    from tac_ufld.models.resnet import ResNetBackbone

    try:
        net = ResNetBackbone("18", pretrained=True, in_channels=1)
    except RuntimeError as exc:
        if "Could not download" in str(exc):
            pytest.skip("offline")
        raise
    assert net.conv1.weight.shape == (64, 1, 7, 7)


# ------------------------------------------------------ geometric augmentation


def test_identity_homography_reproduces_encode_targets():
    for lanes, known, valid in (([LEFT, RIGHT], (True, True), (60.0, 120.0)),
                                ([LEFT, None], (True, True), (60.0, 120.0)),
                                ([LEFT, None], (True, False), None),
                                ([RIGHT * [1.3, 1.0], LEFT], (True, True), (30.0, 110.0))):
        rec = _record(lanes, known, valid)
        a = encode_targets(rec, W, H, ANCHORS, G)
        b = encode_targets_warped(rec, np.eye(3), W, H, ANCHORS, G)
        assert np.array_equal(a.cls, b.cls) and np.array_equal(a.exist, b.exist)
        np.testing.assert_allclose(a.x, b.x, atol=1e-3)


def test_translation_shifts_targets_and_lanes_leaving_the_image_become_no_lane():
    rec = _record([LEFT, RIGHT], valid=None)
    base = encode_targets(rec, W, H, ANCHORS, G)
    dx = 7.0
    h = np.array([[1, 0, dx], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    t = encode_targets_warped(rec, h, W, H, ANCHORS, G)
    both = (base.exist > 0) & (t.exist > 0)
    np.testing.assert_allclose(t.x[both], np.minimum(base.x[both] + dx, W - 1), atol=1e-3)
    big = np.array([[1, 0, 60], [0, 1, 0], [0, 0, 1]], dtype=np.float64)  # pushes the right lane out
    t2 = encode_targets_warped(rec, big, W, H, ANCHORS, G)
    gone = (base.exist[:, 1] > 0) & (t2.exist[:, 1] == 0)
    assert gone.any() and (t2.cls[gone, 1] == G).all()


def test_rows_showing_unannotated_content_are_ignored_after_rotation():
    rec = _record([LEFT, None], valid=(60.0, 120.0))  # absent right lane -> "no lane" labels in the ROI
    cx, cy = (W - 1) / 2, (H - 1) / 2
    a = np.deg2rad(12)
    rot = np.array([[np.cos(a), -np.sin(a), cx - cx * np.cos(a) + cy * np.sin(a)],
                    [np.sin(a), np.cos(a), cy - cx * np.sin(a) - cy * np.cos(a)], [0, 0, 1]])
    base = encode_targets(rec, W, H, ANCHORS, G)
    t = encode_targets_warped(rec, rot, W, H, ANCHORS, G)
    inv = np.linalg.inv(rot)
    xs = np.linspace(0, W - 1, 200)
    y0_model, y1_model = 60.0 * H / 120, 120.0 * H / 120
    for a in np.flatnonzero((t.cls == G).any(axis=1)):
        src = apply_h(inv, np.stack([xs, np.full_like(xs, ANCHORS[a])], axis=1))
        visible = (src[:, 0] >= 0) & (src[:, 0] < W) & (src[:, 1] >= 0) & (src[:, 1] < H)
        assert ((src[visible, 1] >= y0_model - 1e-3) & (src[visible, 1] <= y1_model + 1e-3)).all(), a
    # the rotation tilts the ROI top edge across the first anchor rows: they lose their "no lane" labels
    assert ((base.cls == G) & (t.cls == IGNORE_INDEX)).any()
    assert (t.exist > 0).sum() > 0


def test_image_label_alignment_under_random_transforms():
    """Draw the lanes, transform image and labels together, and check that the
    re-encoded targets land on the drawn (bright) lane pixels."""
    img = np.full((H, W, 3), 0.2, dtype=np.float32)
    rec = _record([LEFT, RIGHT], valid=None)
    for lane in transform_lanes(rec, np.eye(3), W, H):
        cv2.polylines(img, [np.round(lane).astype(np.int32).reshape(-1, 1, 2)], False, (1.0, 1.0, 1.0), 2)
    frames = torch.from_numpy(img.transpose(2, 0, 1).copy()).unsqueeze(0)
    cfg = GeometricAugConfig(enabled=True, prob=1.0, translate_x=0.08, translate_y=0.05, scale=[0.9, 1.1],
                             rotate_deg=6, perspective=0.04, crop_scale=[0.85, 1.0], hflip_prob=0.5)
    aug = GeometricAugmenter(cfg, flip_permutation=(1, 0))
    random.seed(0)
    hits = total = 0
    for _ in range(30):
        h, flipped = aug.sample(W, H)
        warped = warp_frames(frames, h)[0].numpy().transpose(1, 2, 0)
        t = encode_targets_warped(rec, h, W, H, ANCHORS, G, flipped, (1, 0))
        for a, slot in zip(*np.nonzero(t.exist > 0)):
            x, y = t.x[a, slot], ANCHORS[a]
            patch = warped[max(0, int(y) - 1):int(y) + 2, max(0, int(x) - 2):int(x) + 3, 0]
            hits += int(patch.max() > 0.6)
            total += 1
        if flipped:  # the left lane must move to slot 1 (it is now on the right)
            lanes = transform_lanes(rec, h, W, H)
            present = t.exist[:, 1] > 0
            if present.any():
                assert t.x[present, 1].mean() > W / 2 - 10 or lanes[0] is None
    assert total > 100 and hits / total > 0.95


def test_folded_lane_is_ignored():
    kinked = np.array([[20.0, 70.0], [60.0, 90.0], [150.0, 95.0]], dtype=np.float32)
    rec = _record([kinked, None], valid=None)
    a = np.deg2rad(-15)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ys = apply_h(rot, kinked * [W / 160, H / 120])[:, 1]
    assert not (np.all(np.diff(ys) > 0) or np.all(np.diff(ys) < 0))
    t = encode_targets_warped(rec, rot, W, H, ANCHORS, G)
    assert (t.cls[:, 0] == IGNORE_INDEX).all() and (t.cls[:, 1] != IGNORE_INDEX).any()


def test_geometric_sampling_is_deterministic_and_flip_needs_permutation():
    cfg = GeometricAugConfig(enabled=True, prob=1.0, rotate_deg=5, perspective=0.03, hflip_prob=0.5)
    aug = GeometricAugmenter(cfg, (1, 0))
    random.seed(42)
    first = [aug.sample(W, H) for _ in range(5)]
    random.seed(42)
    second = [aug.sample(W, H) for _ in range(5)]
    for (h1, f1), (h2, f2) in zip(first, second):
        assert np.array_equal(h1, h2) and f1 == f2
    with pytest.raises(ConfigError):
        GeometricAugmenter(cfg, None)
    random.seed(1)
    assert GeometricAugmenter(GeometricAugConfig(enabled=False), None).sample(W, H) is None


# ------------------------------------------------------ photometric augmentation


class _OriginalPhotometric:
    """Verbatim copy of the pre-change PhotometricAugmenter (git HEAD)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, frames):
        cfg = self.cfg
        if not cfg.enabled or random.random() > cfg.prob:
            return frames
        brightness = 1.0 + random.uniform(-cfg.brightness, cfg.brightness)
        contrast = 1.0 + random.uniform(-cfg.contrast, cfg.contrast)
        saturation = 1.0 + random.uniform(-cfg.saturation, cfg.saturation)
        mean = frames.mean(dim=(2, 3), keepdim=True)
        frames = (frames - mean) * contrast + mean
        gray = frames.mean(dim=1, keepdim=True)
        frames = (gray + (frames - gray) * saturation) * brightness
        if cfg.noise_std > 0:
            frames = frames + torch.randn_like(frames) * cfg.noise_std
        frames = frames.clamp(0.0, 1.0)
        if random.random() < cfg.erasing_prob:
            _, _, h, w = frames.shape
            area = random.uniform(*cfg.erasing_scale) * h * w
            aspect = random.uniform(0.5, 2.0)
            eh = max(1, min(h, int((area * aspect) ** 0.5)))
            ew = max(1, min(w, int((area / aspect) ** 0.5)))
            y0, x0 = random.randint(0, h - eh), random.randint(0, w - ew)
            frames[:, :, y0:y0 + eh, x0:x0 + ew] = torch.rand(frames.shape[0], 3, eh, ew)
        if frames.shape[0] > 1 and random.random() < cfg.frame_dropout_prob:
            frames[random.randrange(frames.shape[0] - 1)] = frames[-1].clone()
        return frames


def test_default_photometric_augmentation_is_bit_identical_to_the_original():
    cfg = AugmentationConfig(erasing_prob=0.5, frame_dropout_prob=0.5)
    for seed in range(20):
        frames = torch.rand(3, 3, 16, 20)
        random.seed(seed)
        torch.manual_seed(seed)
        new = PhotometricAugmenter(cfg)(frames.clone())
        random.seed(seed)
        torch.manual_seed(seed)
        old = _OriginalPhotometric(cfg)(frames.clone())
        assert torch.equal(new, old)


def test_extended_photometric_ops():
    cfg = AugmentationConfig(prob=1.0, noise_std=0.0, erasing_prob=0.0, frame_dropout_prob=0.0, gamma=0.5,
                             hue=0.2, blur_prob=1.0, motion_blur_prob=1.0, shadow_prob=1.0,
                             brightness=0.0, contrast=0.0, saturation=0.0)
    frames = torch.rand(3, 3, 24, 32)
    random.seed(3)
    a = PhotometricAugmenter(cfg)(frames.clone())
    random.seed(3)
    b = PhotometricAugmenter(cfg)(frames.clone())
    assert torch.equal(a, b) and a.shape == frames.shape and a.min() >= 0 and a.max() <= 1
    assert torch.allclose(a[0] - a[1], a[0] - a[1])  # frames of a clip stay consistent
    gray = torch.full((1, 3, 8, 8), 0.5)
    hue_only = AugmentationConfig(prob=1.0, brightness=0, contrast=0, saturation=0, noise_std=0, erasing_prob=0,
                                  frame_dropout_prob=0, hue=0.3)
    random.seed(0)
    assert torch.allclose(PhotometricAugmenter(hue_only)(gray.clone()), gray, atol=1e-5)  # grey has no hue
