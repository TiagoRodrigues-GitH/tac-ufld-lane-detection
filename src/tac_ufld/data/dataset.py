"""PyTorch dataset producing clips of ``num_frames`` frames plus row-anchor targets."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from tac_ufld.config import DataConfig
from tac_ufld.data.base import LaneDatasetAdapter
from tac_ufld.data.targets import encode_targets
from tac_ufld.data.transforms import PhotometricAugmenter, load_frame, normalize
from tac_ufld.data.types import FrameRecord

LOGGER = logging.getLogger(__name__)


class TemporalLaneDataset(Dataset):
    """Items: ``images`` (T, 3, H, W) ordered oldest -> current, and targets for
    the current frame. History frame ``k`` is ``frame_id - k * temporal_step``.
    Missing history frames (sequence start, gaps) are replaced by the nearest
    newer available frame; the count is logged, never silent."""

    def __init__(
        self,
        records: list[FrameRecord],
        adapter: LaneDatasetAdapter,
        cfg: DataConfig,
        row_anchors: np.ndarray,
        num_frames: int,
        augment: bool = False,
    ) -> None:
        self.records = records
        self.cfg = cfg
        self.num_frames = num_frames
        self.augmenter = PhotometricAugmenter(cfg.augmentation) if augment else None
        self.fallback_frames = 0
        self.context_paths = [self._context(r, adapter) for r in records]
        self.targets = [
            encode_targets(r, cfg.img_w, cfg.img_h, row_anchors, cfg.griding_num) for r in records
        ]
        if self.fallback_frames:
            LOGGER.info(
                "%d of %d history frames unavailable (sequence starts/gaps); replaced by newer frames",
                self.fallback_frames, len(records) * max(num_frames - 1, 0),
            )

    def _context(self, record: FrameRecord, adapter: LaneDatasetAdapter) -> list[str]:
        paths = []
        for k in range(self.num_frames - 1, -1, -1):
            if k == 0:
                paths.append(record.image_path)
            else:
                paths.append(adapter.frame_path(record.sequence, record.frame_id - k * self.cfg.temporal_step))
        for i in range(len(paths) - 2, -1, -1):
            if paths[i] is None:
                paths[i] = paths[i + 1]
                self.fallback_frames += 1
        return [str(p) for p in paths]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        frames = torch.stack([load_frame(p, self.cfg.img_w, self.cfg.img_h) for p in self.context_paths[idx]])
        if self.augmenter is not None:
            frames = self.augmenter(frames)
        t = self.targets[idx]
        return {
            "images": normalize(frames, self.cfg.normalize),
            "cls": torch.from_numpy(t.cls),
            "exist": torch.from_numpy(t.exist),
            "x": torch.from_numpy(t.x),
            "valid": torch.from_numpy(t.valid),
            "index": torch.tensor(idx),
        }
