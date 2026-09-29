"""Dataset adapters, splits, targets and the torch Dataset."""

from __future__ import annotations

from tac_ufld.config import ExperimentConfig
from tac_ufld.data.base import LaneDatasetAdapter
from tac_ufld.data.types import FrameRecord


def build_adapter(cfg: ExperimentConfig) -> LaneDatasetAdapter:
    """Factory: one adapter per supported dataset."""
    name = cfg.data.dataset.lower()
    if name == "elas":
        from tac_ufld.data.elas import ElasAdapter

        return ElasAdapter(
            cfg.data_root(), scenes=cfg.data.scenes or None, scene_tags=cfg.data.scene_tags,
            include_frames_without_lanes=cfg.data.include_frames_without_lanes,
        )
    if name == "culane":
        from tac_ufld.data.culane import CULaneAdapter

        return CULaneAdapter(cfg.data_root())
    raise ValueError(
        f"dataset '{cfg.data.dataset}' has no adapter yet (implemented: elas, culane; "
        f"planned: tusimple, openlane)"
    )


__all__ = ["FrameRecord", "LaneDatasetAdapter", "build_adapter"]
