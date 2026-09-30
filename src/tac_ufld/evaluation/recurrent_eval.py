"""Frame-by-frame evaluation of recurrent models with a carried hidden state.

In "window" mode (training and the main results) a recurrent model runs its
ConvGRU from a zero state over the ``num_frames`` clip, exactly like every
other temporal model. In "carry" mode - what a deployment with ONE state
tensor per stream does - the state is never reset inside a sequence, so the
memory reaches back to the start of the sequence. Consecutive updates of a
state are ``temporal_step`` frames apart, as in training, so there are
``temporal_step`` interleaved state chains.

Only meaningful on whole sequences: the experiment uses it for the held-out
test scenes, where every stored frame between the first and last labelled
frame belongs to the test split (no memory of training images).
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from torch import nn

from tac_ufld.data.dataset import TemporalLaneDataset
from tac_ufld.data.preprocess import normalize_channels
from tac_ufld.data.transforms import load_frame
from tac_ufld.decoding import decode_logits
from tac_ufld.evaluation.predictor import Predictions
from tac_ufld.evaluation.tracking import sequence_order


@torch.no_grad()
def carry_predictions(model: nn.Module, dataset: TemporalLaneDataset, adapter, device: str, img_w: int,
                      amp: bool, temporal_step: int, batch_size: int = 16) -> Predictions:
    if not getattr(model, "recurrent", False):
        raise TypeError("carry-state evaluation needs a recurrent model (ufld_v07, lite_v06)")
    model.eval()
    records, cfg = dataset.records, dataset.cfg
    n = len(records)
    exist_all = np.zeros((n, cfg.num_row_anchors, cfg.num_lanes), dtype=np.float32)
    x_all = np.zeros_like(exist_all)
    autocast = torch.autocast("cuda") if (amp and device.startswith("cuda")) else nullcontext()
    for idx in sequence_order(records):
        seq = records[idx[0]].sequence
        wanted = {records[i].frame_id: i for i in idx}
        first, last = min(wanted), max(wanted)
        frame_ids = [f for f in range(first, last + 1) if adapter.frame_path(seq, f) is not None]
        states: dict[int, torch.Tensor] = {}
        for start in range(0, len(frame_ids), batch_size):
            chunk = frame_ids[start:start + batch_size]
            frames = torch.stack([load_frame(str(adapter.frame_path(seq, f)), cfg.img_w, cfg.img_h) for f in chunk])
            images = normalize_channels(dataset.preprocessor(frames), dataset.mean, dataset.std).to(device)
            with autocast:
                feats = model.encode_frames(images)["current"]
                for j, fid in enumerate(chunk):
                    out = model.recurrent_step(feats[j:j + 1], states.get(fid - temporal_step))
                    states[fid] = out["state"]
                    states.pop(fid - temporal_step, None)
                    if fid in wanted:
                        logits = out["logits"].float()
                        e, bins = decode_logits(logits)
                        grid = logits.shape[1] - 1
                        exist_all[wanted[fid]] = e[0].cpu().numpy()
                        x_all[wanted[fid]] = (bins[0] * (img_w - 1) / (grid - 1)).cpu().numpy()
    return Predictions(exist=exist_all, x=x_all, current_weight=np.full(n, np.nan, dtype=np.float32))
