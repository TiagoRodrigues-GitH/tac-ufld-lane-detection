"""Leakage-safe, seed-independent data splits (the audit's finding C5).

Protocol for datasets without an official split (ELAS):

* ``test``      - every frame of the scenes listed in ``split.test_scenes``.
                  These scenes never appear in training, validation or HPO, so
                  this split measures generalisation to unseen roads. Whole
                  scenes are kept, so consecutive frames exist for temporal
                  metrics (jitter).
* ``train`` / ``val`` / ``seen_test`` - the remaining scenes are cut into
                  fixed temporal blocks of ``block_size`` frames; blocks are
                  assigned with a RNG seeded only by ``split_seed`` and the
                  scene name. Frames closer than ``min_gap`` frames to a frame
                  of another split are purged, where ``min_gap`` is at least
                  the temporal context span, so no sample's history frames can
                  belong to another split. ``seen_test`` is a secondary test
                  on seen scenes (comparable to the previous protocol).

The split never depends on the training seed: every seed and every model
sees exactly the same frames. ``check_no_leakage`` enforces all of this.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tac_ufld.config import SplitConfig
from tac_ufld.data.types import FrameRecord

SPLIT_NAMES = ("train", "val", "test", "seen_test")


class LeakageError(RuntimeError):
    """Raised when two splits share frames, scenes or temporal neighbourhoods."""


@dataclass
class DataSplits:
    train: list[FrameRecord]
    val: list[FrameRecord]
    test: list[FrameRecord]
    seen_test: list[FrameRecord] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[FrameRecord]]:
        return {name: getattr(self, name) for name in SPLIT_NAMES}

    def manifest(self) -> pd.DataFrame:
        rows = [
            {"split": name, "dataset": r.dataset, "sequence": r.sequence,
             "frame_id": r.frame_id, "image_path": str(r.image_path)}
            for name, recs in self.as_dict().items() for r in recs
        ]
        return pd.DataFrame(rows, columns=["split", "dataset", "sequence", "frame_id", "image_path"])

    def summary(self) -> pd.DataFrame:
        m = self.manifest()
        if m.empty:
            return m
        return m.groupby(["split", "sequence"]).size().rename("frames").reset_index()


def _distance_to_nearest(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Distance from each query id to the nearest id in sorted ``reference``."""
    if len(reference) == 0:
        return np.full(len(query), np.inf)
    idx = np.searchsorted(reference, query)
    left = reference[np.clip(idx - 1, 0, len(reference) - 1)]
    right = reference[np.clip(idx, 0, len(reference) - 1)]
    return np.minimum(np.abs(query - left), np.abs(query - right)).astype(float)


def _assign_blocks(frame_ids: list[int], cfg: SplitConfig, scene: str) -> dict[int, str]:
    blocks = sorted({fid // cfg.block_size for fid in frame_ids})
    order = blocks[:]
    random.Random(f"{cfg.split_seed}:{scene}").shuffle(order)
    n = len(order)
    n_val = int(round(n * cfg.val_fraction))
    n_seen = int(round(n * cfg.seen_test_fraction))
    if n >= 3:
        n_val = max(n_val, 1 if cfg.val_fraction > 0 else 0)
        n_seen = max(n_seen, 1 if cfg.seen_test_fraction > 0 else 0)
    while n_val + n_seen >= n and (n_val + n_seen) > 0:  # keep >= 1 training block
        if n_seen > 0:
            n_seen -= 1
        else:
            n_val -= 1
    labels = {b: "train" for b in order}
    labels.update({b: "val" for b in order[:n_val]})
    labels.update({b: "seen_test" for b in order[n_val:n_val + n_seen]})
    return labels


def _purge(frame_ids: np.ndarray, labels: np.ndarray, gap: int) -> np.ndarray:
    keep = np.ones(len(frame_ids), dtype=bool)
    rules = (("train", ("val", "seen_test")), ("seen_test", ("val",)))
    for target, others in rules:
        mask = labels == target
        if not mask.any():
            continue
        reference = np.sort(frame_ids[np.isin(labels, others)])
        close = _distance_to_nearest(frame_ids[mask], reference) <= gap
        keep[np.flatnonzero(mask)[close]] = False
    return keep


def cap_records(records: list[FrameRecord], max_n: int | None, key: str) -> list[FrameRecord]:
    """Deterministic random subsample (seeded by ``key``), returned in order."""
    if max_n is None or len(records) <= max_n:
        return records
    chosen = sorted(random.Random(key).sample(range(len(records)), max_n))
    return [records[i] for i in chosen]


def split_scenes_and_blocks(
    records_by_seq: dict[str, list[FrameRecord]], cfg: SplitConfig, min_gap: int
) -> DataSplits:
    unknown = [s for s in cfg.test_scenes if s not in records_by_seq]
    if unknown:
        raise ValueError(f"split.test_scenes not found in loaded scenes: {unknown}")
    out: dict[str, list[FrameRecord]] = {name: [] for name in SPLIT_NAMES}
    for scene in sorted(records_by_seq):
        recs = sorted(records_by_seq[scene], key=lambda r: r.frame_id)
        if scene in cfg.test_scenes:
            out["test"].extend(recs)
            continue
        block_label = _assign_blocks([r.frame_id for r in recs], cfg, scene)
        fids = np.asarray([r.frame_id for r in recs])
        labels = np.asarray([block_label[r.frame_id // cfg.block_size] for r in recs])
        keep = _purge(fids, labels, min_gap)
        for rec, label, k in zip(recs, labels, keep):
            if k:
                out[label].append(rec)
    caps = {"train": cfg.max_train_frames, "val": cfg.max_val_frames,
            "test": cfg.max_test_frames, "seen_test": cfg.max_seen_test_frames}
    for name in SPLIT_NAMES:
        out[name] = cap_records(out[name], caps[name], f"{cfg.split_seed}:cap:{name}")
    splits = DataSplits(**out)
    check_no_leakage(splits, min_gap=min_gap, held_out_sequences=cfg.test_scenes)
    return splits


def check_no_leakage(
    splits: DataSplits, min_gap: int | None = None, held_out_sequences: list[str] | None = None
) -> None:
    """Raise ``LeakageError`` if splits overlap in frames, held-out scenes or
    temporal neighbourhoods (``|frame_id difference| <= min_gap`` within a scene)."""
    groups = splits.as_dict()
    seen: dict[str, str] = {}
    for name, recs in groups.items():
        for r in recs:
            if r.key in seen:
                raise LeakageError(f"frame {r.key} is in both '{seen[r.key]}' and '{name}'")
            seen[r.key] = name
    held_out = set(held_out_sequences or [])
    for name in ("train", "val", "seen_test"):
        bad = held_out & {r.sequence for r in groups[name]}
        if bad:
            raise LeakageError(f"held-out test scene(s) {sorted(bad)} appear in '{name}'")
    if min_gap is None:
        return
    for a, b in (("train", "val"), ("train", "seen_test"), ("val", "seen_test")):
        by_seq: dict[str, list[int]] = {}
        for r in groups[b]:
            by_seq.setdefault(r.sequence, []).append(r.frame_id)
        ref = {seq: np.sort(np.asarray(ids)) for seq, ids in by_seq.items()}
        for r in groups[a]:
            if r.sequence in ref:
                d = _distance_to_nearest(np.asarray([r.frame_id]), ref[r.sequence])[0]
                if d <= min_gap:
                    raise LeakageError(
                        f"{r.key} ('{a}') is only {int(d)} frames from a '{b}' frame (min gap {min_gap})"
                    )
