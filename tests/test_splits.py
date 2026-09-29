"""Split protocol (audit C5): held-out scenes, purge gaps, seed independence."""

from __future__ import annotations

import random

import numpy as np
import pytest

from tac_ufld.config import ConfigError, SplitConfig, load_config, PROJECT_ROOT
from tac_ufld.data.splits import DataSplits, LeakageError, check_no_leakage, split_scenes_and_blocks
from tac_ufld.data.types import FrameRecord


def _records(scenes=("A", "B", "C", "D"), n=300):
    lane = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    return {s: [FrameRecord("t", s, i, None, (10, 10), (lane, None), (True, True)) for i in range(n)]
            for s in scenes}


CFG = SplitConfig(split_seed=7, test_scenes=["D"], block_size=30, purge_frames=10,
                  val_fraction=0.2, seen_test_fraction=0.15)


def test_held_out_scene_only_in_test_and_no_leakage():
    s = split_scenes_and_blocks(_records(), CFG, min_gap=10)
    assert {r.sequence for r in s.test} == {"D"}
    assert "D" not in {r.sequence for r in s.train + s.val + s.seen_test}
    check_no_leakage(s, 10, ["D"])  # does not raise
    assert s.train and s.val and s.seen_test


def test_gap_between_splits_exceeds_min_gap():
    s = split_scenes_and_blocks(_records(), CFG, min_gap=10)
    for seq in "ABC":
        train = np.array([r.frame_id for r in s.train if r.sequence == seq])
        other = np.array([r.frame_id for r in s.val + s.seen_test if r.sequence == seq])
        assert np.abs(train[:, None] - other[None, :]).min() > 10


def test_split_is_deterministic_and_ignores_global_rng():
    random.seed(1)
    np.random.seed(1)
    a = split_scenes_and_blocks(_records(), CFG, 10).manifest()
    random.seed(999)
    np.random.seed(999)
    b = split_scenes_and_blocks(_records(), CFG, 10).manifest()
    assert a.equals(b)


def test_leakage_is_detected():
    recs = _records(("A",), 50)["A"]
    with pytest.raises(LeakageError):
        check_no_leakage(DataSplits(train=recs[:30], val=recs[29:], test=[]))
    with pytest.raises(LeakageError):
        check_no_leakage(DataSplits(train=recs[:20], val=recs[25:], test=[]), min_gap=10)


def test_purge_shorter_than_temporal_span_is_rejected():
    with pytest.raises(ConfigError):
        load_config(PROJECT_ROOT / "configs" / "elas_smoke.yaml",
                    {"data.split.purge_frames": 2, "data.temporal_step": 2, "data.num_frames": 3})
