"""Holm families for the paired tests and sequence-preserving caps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tac_ufld.config import PROJECT_ROOT, load_config
from tac_ufld.data.splits import cap_records
from tac_ufld.data.types import FrameRecord
from tac_ufld.evaluation.stats import min_attainable_p, paired_comparisons, seeds_needed


def _results(n_seeds: int, deltas: dict[str, float]) -> pd.DataFrame:
    rows = []
    for s in range(1, n_seeds + 1):
        rows.append({"variant": "base", "seed": s, "f1": 0.80 + 0.001 * s, "jit": 1.5})
        for v, d in deltas.items():
            rows.append({"variant": v, "seed": s, "f1": 0.80 + 0.001 * s + d + 0.0001 * s, "jit": 1.4})
    return pd.DataFrame(rows)


def test_primary_family_is_corrected_on_its_own():
    """Many exploratory rows must not wipe out a declared primary comparison."""
    res = _results(8, {f"t{i}": 0.02 for i in range(8)})
    pairs = [(f"t{i}", "base") for i in range(8)]
    df = paired_comparisons(res, pairs, {"f1": True, "jit": False}, primary_pairs=[("t0", "base")],
                            primary_metric="f1")
    prim = df[df["family"] == "primary"]
    assert len(prim) == 1 and prim["significant"].item()            # 8 unanimous seeds, family of 1
    assert set(df["family"]) == {"primary", "exploratory:f1", "exploratory:jit"}
    assert (df[df["family"] == "exploratory:f1"]["family_size"] == 7).all()


def test_one_family_over_everything_was_hopeless_with_six_seeds():
    res = _results(6, {f"t{i}": 0.02 for i in range(5)})
    pairs = [(f"t{i}", "base") for i in range(5)]
    df = paired_comparisons(res, pairs, {"f1": True})
    assert not df["significant"].any() and df["underpowered"].all()   # 5 x 0.031 > 0.05
    assert min_attainable_p(6) * 5 > 0.05


def test_seeds_needed():
    assert seeds_needed(1) == 6 and seeds_needed(2) == 7 and seeds_needed(4) == 8


def test_cap_by_sequence_keeps_consecutive_frames():
    recs = [FrameRecord(dataset="x", sequence=f"s{s}", frame_id=f, image_path=f"{s}_{f}", image_size=(10, 10),
                        lanes=(None,), slot_known=(True,)) for s in range(10) for f in range(50)]
    capped = cap_records(recs, 120, "k", by_sequence=True)
    assert len(capped) == 120
    by_seq: dict[str, list[int]] = {}
    for r in capped:
        by_seq.setdefault(r.sequence, []).append(r.frame_id)
    for ids in by_seq.values():
        assert np.all(np.diff(sorted(ids)) == 1)                      # contiguous runs only
    assert len(by_seq) == 3
    assert cap_records(recs, 120, "k", by_sequence=True) == capped    # deterministic
    random_cap = cap_records(recs, 120, "k")
    assert len({r.sequence for r in random_cap}) > 3                  # the old behaviour is unchanged


def test_openlane_pilot_config_is_valid():
    cfg = load_config(PROJECT_ROOT / "configs" / "openlane_pilot.yaml", {"data.root": "."})
    assert cfg.data.split.cap_by_sequence and cfg.evaluation.primary_pairs
    assert all(v in cfg.model.variants for p in cfg.evaluation.primary_pairs for v in p)
