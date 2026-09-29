"""Metric correctness, including the anchor-metric fix (audit C2)."""

from __future__ import annotations

import numpy as np
import pytest

from tac_ufld.data.targets import IGNORE_INDEX, bin_to_x
from tac_ufld.metrics import (
    anchor_counts, culane_line_width, iou_matrix, lane_mask, match_lanes, pixel_counts, prf, temporal_jitter,
)

G, W = 64, 512


def test_wrong_position_counts_as_fp_and_fn():
    target = np.full((8, 2), 10)
    exist = np.ones((8, 2))
    x = bin_to_x(np.full((8, 2), 50), W, G)                 # confident, 40 cells away
    c = anchor_counts(exist, x, target, 0.5, 1, W, G)
    assert (c.tp, c.fp, c.fn) == (0, 16, 16)


def test_half_wrong_gives_f1_half():
    target = np.full((8, 2), 10)
    target[:, 0] = 50
    x = bin_to_x(np.full((8, 2), 10), W, G)
    c = anchor_counts(np.ones((8, 2)), x, target, 0.5, 1, W, G)
    p, r, f1 = prf(c.tp, c.fp, c.fn)
    assert p == pytest.approx(0.5) and r == pytest.approx(0.5) and f1 == pytest.approx(0.5)


def test_ignored_cells_do_not_count():
    target = np.full((4, 1), IGNORE_INDEX)
    c = anchor_counts(np.ones((4, 1)), np.zeros((4, 1)), target, 0.5, 1, W, G)
    assert (c.tp, c.fp, c.fn, c.tn) == (0, 0, 0, 0)


def test_true_negative_and_tolerance():
    target = np.array([[G], [12]])
    exist = np.array([[0.1], [0.9]])
    x = bin_to_x(np.array([[0], [13]]), W, G)
    c = anchor_counts(exist, x, target, 0.5, 1, W, G)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 0, 0, 1)


def test_culane_width_scaling():
    assert culane_line_width(1640) == 30
    assert culane_line_width(640) == 12


def test_identical_lanes_match_and_far_lanes_do_not():
    lane = np.array([[100, 300], [200, 470]], dtype=np.float32)
    far = lane + [150, 0]
    m_same = lane_mask(lane, 640, 480, 12)
    m_far = lane_mask(far, 640, 480, 12)
    assert match_lanes(iou_matrix([m_same], [m_same]), 0.5).tp == 1
    res = match_lanes(iou_matrix([m_far], [m_same]), 0.5)
    assert (res.tp, res.fp, res.fn) == (0, 1, 1)


def test_hungarian_beats_greedy():
    # greedy would take (0,0)=0.9 then nothing >= 0.5; the optimal assignment keeps two matches
    ious = np.array([[0.9, 0.8], [0.85, 0.0]])
    assert match_lanes(ious, 0.5).tp == 2


def test_pixel_counts():
    a = np.zeros((4, 4), bool)
    a[0] = True
    b = np.zeros((4, 4), bool)
    b[0, :2] = True
    b[1, 0] = True
    assert pixel_counts([a], [b], (4, 4)) == (2, 2, 1)


def test_jitter_uses_only_consecutive_frames():
    ex = np.ones((2, 1))
    frames = [("S", 1, ex, np.array([[10.0], [10.0]])), ("S", 2, ex, np.array([[13.0], [10.0]])),
              ("S", 5, ex, np.array([[99.0], [99.0]]))]
    out = temporal_jitter(frames, 0.5)
    assert out["jitter_pairs"] == 2 and out["jitter_px"] == pytest.approx(1.5)
