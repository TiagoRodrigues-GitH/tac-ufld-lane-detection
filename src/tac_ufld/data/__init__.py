"""Dataset adapters, splits, targets and the torch Dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tac_ufld.data.base import LaneDatasetAdapter
from tac_ufld.data.types import FrameRecord

if TYPE_CHECKING:
    from tac_ufld.config import ExperimentConfig

KNOWN_DATASETS = ("elas", "culane", "tusimple", "openlane")


def build_adapter(cfg: "ExperimentConfig") -> LaneDatasetAdapter:
    """Factory: one adapter per supported dataset. Only the selected dataset's
    root is touched (nothing else is scanned)."""
    name = cfg.data.dataset.lower()
    root = cfg.data_root()
    if name == "elas":
        from tac_ufld.data.elas import ElasAdapter

        return ElasAdapter(
            root, scenes=cfg.data.scenes or None, scene_tags=cfg.data.scene_tags,
            include_frames_without_lanes=cfg.data.include_frames_without_lanes,
        )
    if name == "culane":
        from tac_ufld.data.culane import CULaneAdapter

        return CULaneAdapter(root)
    if name == "tusimple":
        from tac_ufld.data.tusimple import TuSimpleAdapter

        return TuSimpleAdapter(root, drives=cfg.data.scenes or None)
    if name == "openlane":
        from tac_ufld.data.openlane import OpenLaneAdapter

        return OpenLaneAdapter(root, segments=cfg.data.scenes or None,
                               skip_unattributed_frames=cfg.data.skip_unattributed_frames)
    raise ValueError(f"dataset '{cfg.data.dataset}' has no adapter (supported: {list(KNOWN_DATASETS)})")


__all__ = ["FrameRecord", "KNOWN_DATASETS", "LaneDatasetAdapter", "build_adapter"]
