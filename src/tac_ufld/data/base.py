"""Common interface every dataset adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from tac_ufld.data.types import FrameRecord


class LaneDatasetAdapter(ABC):
    """Translate one dataset's on-disk format into ``FrameRecord`` objects.

    Adapters only parse and index; they never resize images, build targets or
    split data, so every dataset shares the same downstream code.

    Class attributes describing the dataset's conventions:

    * ``num_lanes`` - fixed lane slots of the internal representation;
    * ``flip_permutation`` - slot order after a horizontal flip (left <-> right),
      or ``None`` when flipping would break the label semantics;
    * ``protocol`` - the dataset-specific split / evaluation protocol, written
      into every report so results from different datasets are never mixed.
    """

    name: str = "base"
    num_lanes: int = 0
    flip_permutation: tuple[int, ...] | None = None
    protocol: dict[str, str] = {}

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
        """Datasets with an official split override this. ``None`` means the
        scene/temporal-block split protocol is used instead. A missing or
        empty ``val`` entry means validation is carved from ``train``
        (``data.split.val_strategy``) without touching ``test``."""
        return None

    def load_all(self) -> dict[str, list[FrameRecord]]:
        return {seq: self.load_sequence(seq) for seq in self.sequences()}
