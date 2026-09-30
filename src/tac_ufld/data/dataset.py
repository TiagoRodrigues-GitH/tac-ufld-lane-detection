"""PyTorch dataset producing clips of ``num_frames`` frames plus row-anchor targets.

Per item, in this order (identical for every split; augmentation only when
``augment=True``, i.e. training):

1. load and resize the RGB frames (PIL, bilinear);
2. geometric augmentation: one homography for the whole clip, targets
   re-encoded from the transformed lanes (``data.augmentation.geometric``);
3. photometric augmentation, sampled once per clip;
4. preprocessing to the model's input representation (``data.preprocessing``);
5. per-channel normalisation.

Steps 1, 4 and 5 are exactly what streaming inference and the ONNX/TensorRT
runners do at deployment time (``tac_ufld.inference``).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from tac_ufld.config import DataConfig
from tac_ufld.data.base import LaneDatasetAdapter
from tac_ufld.data.geometric import GeometricAugmenter, encode_targets_warped, warp_frames
from tac_ufld.data.preprocess import Preprocessor, channel_stats, normalize_channels
from tac_ufld.data.targets import encode_targets
from tac_ufld.data.transforms import PhotometricAugmenter, load_frame
from tac_ufld.data.types import FrameRecord

LOGGER = logging.getLogger(__name__)


class TemporalLaneDataset(Dataset):
    """Items: ``images`` (T, C, H, W) ordered oldest -> current, and targets for
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
        static_history: bool = False,
    ) -> None:
        self.records = records
        self.cfg = cfg
        self.num_frames = num_frames
        # Capacity control (``*_static`` variants): the clip is the CURRENT frame
        # repeated num_frames times, built after augmentation so every position
        # carries the same (possibly degraded) image and no temporal information.
        self.static_history = static_history
        self.row_anchors = row_anchors
        self.augmenter = PhotometricAugmenter(cfg.augmentation) if augment else None
        geo = cfg.augmentation.geometric
        self.flip_permutation = getattr(adapter, "flip_permutation", None)
        self.geometric = (GeometricAugmenter(geo, self.flip_permutation)
                          if augment and cfg.augmentation.enabled and geo.enabled else None)
        self.preprocessor = Preprocessor(cfg.preprocessing)
        self.mean, self.std = channel_stats(cfg.preprocessing, cfg.normalize)
        self.fallback_frames = 0
        self.context_paths = [self._context(r, adapter) for r in records]
        self.targets = [
            encode_targets(r, cfg.img_w, cfg.img_h, row_anchors, cfg.griding_num) for r in records
        ]
        if self.fallback_frames:
            total = len(records) * max(num_frames - 1, 0)
            LOGGER.info("%d of %d history frames unavailable (sequence starts/gaps); replaced by newer frames",
                        self.fallback_frames, total)
            if total and self.fallback_frames / total > 0.05:
                LOGGER.warning("%.0f%% of history frames are missing: check data.temporal_step against "
                               "the dataset's stored frame ids", 100.0 * self.fallback_frames / total)

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
        cfg = self.cfg
        paths = self.context_paths[idx][-1:] if self.static_history else self.context_paths[idx]
        frames = torch.stack([load_frame(p, cfg.img_w, cfg.img_h) for p in paths])
        t = self.targets[idx]
        if self.geometric is not None:
            sampled = self.geometric.sample(cfg.img_w, cfg.img_h)
            if sampled is not None:
                h, flipped = sampled
                frames = warp_frames(frames, h, cfg.augmentation.geometric.border)
                t = encode_targets_warped(self.records[idx], h, cfg.img_w, cfg.img_h, self.row_anchors,
                                          cfg.griding_num, flipped, self.flip_permutation,
                                          cfg.augmentation.geometric.min_row_valid)
        if self.augmenter is not None:
            frames = self.augmenter(frames)
        frames = self.preprocessor(frames)
        images = normalize_channels(frames, self.mean, self.std)
        if self.static_history and self.num_frames > 1:
            images = images.expand(self.num_frames, *images.shape[1:]).contiguous()
        return {
            "images": images,
            "cls": torch.from_numpy(t.cls),
            "exist": torch.from_numpy(t.exist),
            "x": torch.from_numpy(t.x),
            "valid": torch.from_numpy(t.valid),
            "index": torch.tensor(idx),
        }
