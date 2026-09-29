"""ELAS parsing: row geometry (audit C1), missing points, lane slots."""

from __future__ import annotations

import numpy as np
import pytest

from tac_ufld.data.elas import ELAS_ROW_FRACTIONS, ElasAdapter
from tac_ufld.data.targets import encode_targets, make_row_anchors
from tests.conftest import REAL_ELAS


def test_points_sit_at_quarter_fractions_of_roi(synthetic_elas):
    adapter = ElasAdapter(synthetic_elas, scenes=["SYN_A"])
    rec = adapter.load_sequence("SYN_A")[0]
    cfg = adapter.scene_config("SYN_A")
    expected = cfg.roi_y + np.asarray(ELAS_ROW_FRACTIONS) * cfg.roi_h
    np.testing.assert_allclose(rec.lanes[0][:, 1], expected)
    assert ELAS_ROW_FRACTIONS == (0.0, 0.25, 0.5, 1.0)


def test_synthetic_straight_lanes_are_collinear(synthetic_elas):
    rec = ElasAdapter(synthetic_elas, scenes=["SYN_A"]).load_sequence("SYN_A")[3]
    for lane in rec.present_lanes():
        coef = np.polyfit(lane[:, 1], lane[:, 0], 1)
        assert np.abs(np.polyval(coef, lane[:, 1]) - lane[:, 0]).max() < 0.2


def test_missing_side_is_absent_and_single_point_is_unknown(synthetic_elas):
    recs = {r.frame_id: r for r in ElasAdapter(synthetic_elas, scenes=["SYN_A"]).load_sequence("SYN_A")}
    assert recs[5].lanes[1] is None and recs[5].slot_known[1] is True       # all-NaN side -> absent
    assert recs[9].lanes[0] is None and recs[9].slot_known[0] is False      # one point -> unknown


def test_right_only_frame_keeps_right_slot(synthetic_elas, tmp_path):
    from tests.conftest import make_elas_scene

    make_elas_scene(tmp_path / "e", "RIGHT_ONLY", n_frames=3, single_point_left={0, 1, 2})
    rec = ElasAdapter(tmp_path / "e").load_sequence("RIGHT_ONLY")[0]
    # left side unknown (single point), right side present -> must stay in slot 1
    assert rec.lanes[0] is None and rec.lanes[1] is not None
    anchors = make_row_anchors(64, 8, (0.55, 0.99))
    t = encode_targets(rec, 96, 64, anchors, 24)
    assert t.exist[:, 0].sum() == 0 and t.exist[:, 1].sum() > 0


@pytest.mark.dataset
@pytest.mark.skipif(not REAL_ELAS.exists(), reason="local ELAS copy not found")
def test_real_elas_points_are_collinear():
    """Regression test for C1 on the real data: with the correct row geometry
    the four points of (mostly straight) ego lanes lie on a line."""
    adapter = ElasAdapter(REAL_ELAS)
    for scene in adapter.sequences():
        residuals = []
        for rec in adapter.load_sequence(scene):
            for lane in rec.present_lanes():
                if len(lane) == 4:
                    coef = np.polyfit(lane[:, 1], lane[:, 0], 1)
                    residuals.append(np.abs(np.polyval(coef, lane[:, 1]) - lane[:, 0]).max())
        assert np.median(residuals) < 5.0, f"{scene}: median residual {np.median(residuals):.2f}px"
