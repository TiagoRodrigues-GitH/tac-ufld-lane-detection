"""Dataset-agnostic record types shared by every adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FrameRecord:
    """One annotated frame, in ORIGINAL image pixel coordinates.

    Lanes are stored in fixed slots (ELAS: 0 = ego-left, 1 = ego-right;
    CULane: 0..3 left to right). A slot is in exactly one state:

    * present -> ``lanes[i]`` is an (N >= 2, 2) array of (x, y) points,
      ``slot_known[i]`` is True;
    * absent  -> ``lanes[i]`` is None and ``slot_known[i]`` is True
      (the annotation says there is no lane there);
    * unknown -> ``lanes[i]`` is None and ``slot_known[i]`` is False
      (the annotation exists but cannot be used, e.g. a single point).
      Unknown slots are ignored by the loss and by the metrics.

    ``valid_y_range`` is the vertical band in which the annotation is defined
    (ELAS region of interest). Rows outside it are ignored in training and
    predictions outside it are clipped before evaluation.
    """

    dataset: str
    sequence: str
    frame_id: int
    image_path: Path
    image_size: tuple[int, int]  # (width, height)
    lanes: tuple[np.ndarray | None, ...]
    slot_known: tuple[bool, ...]
    valid_y_range: tuple[float, float] | None = None
    tags: dict[str, bool] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.sequence}/{self.frame_id}"

    @property
    def num_slots(self) -> int:
        return len(self.lanes)

    def present_lanes(self) -> list[np.ndarray]:
        return [lane for lane in self.lanes if lane is not None]

    def has_any_lane(self) -> bool:
        return any(lane is not None for lane in self.lanes)
