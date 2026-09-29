from __future__ import annotations

import numpy as np

from tac_ufld.data.targets import IGNORE_INDEX, bin_to_x, encode_targets, make_row_anchors, x_to_bin
from tac_ufld.data.types import FrameRecord


def _record(lanes, known=(True, True), valid=(60.0, 120.0)):
    return FrameRecord("t", "S", 0, None, (160, 120), tuple(lanes), known, valid_y_range=valid)


LANE = np.array([[40.0, 60.0], [20.0, 120.0]], dtype=np.float32)
ANCHORS = make_row_anchors(64, 16, (0.0, 1.0))  # spans the whole image, some rows outside ROI


def test_bin_roundtrip_within_half_cell():
    x = np.linspace(0, 95, 50)
    err = np.abs(bin_to_x(x_to_bin(x, 96, 24), 96, 24) - x)
    assert err.max() <= (95 / 23) / 2 + 1e-6


def test_rows_outside_valid_range_are_ignored():
    t = encode_targets(_record([LANE, None]), 96, 64, ANCHORS, 24)
    y_orig = ANCHORS * 120 / 64
    outside = (y_orig < 60 - 1e-3) | (y_orig > 120 + 1e-3)
    assert (t.cls[outside] == IGNORE_INDEX).all()
    assert (t.cls[~outside, 1] == 24).all()           # absent lane inside ROI -> "no lane" class
    assert t.exist[~outside, 0].sum() > 0


def test_unknown_slot_is_ignored_everywhere():
    t = encode_targets(_record([LANE, None], known=(True, False)), 96, 64, ANCHORS, 24)
    assert (t.cls[:, 1] == IGNORE_INDEX).all() and not t.valid[:, 1].any()


def test_encoded_x_matches_interpolated_lane():
    t = encode_targets(_record([LANE, None], valid=None), 96, 64, ANCHORS, 24)
    rows = np.flatnonzero(t.exist[:, 0] > 0)
    y = ANCHORS[rows] * 120 / 64
    expected = np.interp(y, [60, 120], [40, 20]) * 96 / 160
    np.testing.assert_allclose(t.x[rows, 0], expected, atol=1e-4)
