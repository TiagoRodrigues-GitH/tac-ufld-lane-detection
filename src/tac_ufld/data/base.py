"""Common interface every dataset adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from tac_ufld.data.types import FrameRecord


class LaneDatasetAdapter(ABC):
    """Translate one dataset's on-disk format into ``FrameRecord`` objects.

    Adapters only parse and index; they never resize images, build targets or
    split data, so every dataset shares the same downstream code.
    """

    name: str = "base"
    num_lanes: int = 0

    @abstractmethod
    def sequences(self) -> list[str]:
        """Names of the video sequences (ELAS scenes, CULane clips) available."""

    @abstractmethod
    def load_sequence(self, sequence: str) -> list[FrameRecord]:
        """All usable annotated frames of ``sequence``, sorted by frame id."""

    @abstractmethod
    def frame_path(self, sequence: str, frame_id: int) -> Path | None:
        """Image path of any frame (annotated or not) used as temporal context."""

    def official_splits(self) -> dict[str, list[FrameRecord]] | None:
        """Datasets with an official split (CULane) override this. ``None``
        means the scene/temporal-block split protocol is used instead."""
        return None

    def load_all(self) -> dict[str, list[FrameRecord]]:
        return {seq: self.load_sequence(seq) for seq in self.sequences()}
