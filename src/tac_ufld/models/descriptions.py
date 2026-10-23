"""Plain-language descriptions of every model, shared by the results page and
the UI. Facts only (architecture, what changes, what it costs); results are
never written here, they come from the run folders."""

from __future__ import annotations

# role: baseline | control | temporal ; origin: who designed it
MODEL_NOTES: dict[str, dict[str, str]] = {
    "ufld_baseline": {
        "role": "baseline", "origin": "Official UFLD (Qin et al., ECCV 2020), supervisor's notebook",
        "frames": "t", "history": "none", "aligns": "no", "memory": "none",
        "what": "Ultra-Fast Lane Detection with an ImageNet-pretrained ResNet-18. Lane detection is row-wise "
                "classification: for each of 18 image rows and each lane slot the head picks one of 100 "
                "horizontal cells, or \"no lane\". One frame in, no memory.",
        "differs": "The reference of the UFLD family. Every UFLD temporal model starts from its trained weights "
                   "and is compared with it on the same seeds.",
    },
    "ufld_baseline_ct": {
        "role": "control", "origin": "v0.4 control",
        "frames": "t", "history": "none", "aligns": "no", "memory": "none",
        "what": "The same network trained a second time from its own best checkpoint, with exactly the schedule "
                "the temporal variants get after their warm start.",
        "differs": "A control, not a candidate. A temporal model that beats the baseline but not this control "
                   "gained from extra training, not from time.",
    },
    "ufld_v02": {
        "role": "temporal", "origin": "Supervisor's notebook",
        "frames": "t-4, t-2, t", "history": "weighted average", "aligns": "no", "memory": "3-frame window",
        "what": "Encodes the three frames with the shared ResNet and averages the feature maps with three learned "
                "weights (starting at 0.1 / 0.2 / 0.7) before the UFLD head. Trained with an extra existence "
                "loss and a temporal-consistency loss.",
        "differs": "The simplest fusion: one weight per frame, the same at every pixel. Lane markings that moved "
                   "between frames are averaged at different positions.",
    },
    "ufld_v03": {
        "role": "temporal", "origin": "Supervisor's notebook",
        "frames": "t-4, t-2, t", "history": "per-cell gates", "aligns": "no", "memory": "3-frame window",
        "what": "Like v0.2, but a small convolutional network predicts a softmax gate per frame and per feature "
                "cell, biased towards the current frame.",
        "differs": "Can use history only where the current frame is weak, such as an occluded cell. Like v0.2 "
                   "it fuses feature maps that are not aligned.",
    },
    "ufld_v04": {
        "role": "temporal", "origin": "Supervisor's notebook",
        "frames": "t-4, t-2, t", "history": "weighted average", "aligns": "no", "memory": "3-frame window",
        "what": "v0.2's network with a different loss: a soft-argmax coordinate term penalises the lane position "
                "error in pixels, and the existence loss is re-weighted.",
        "differs": "Same fusion as v0.2. It tests whether a position-aware loss helps; the network itself is "
                   "unchanged.",
    },
    "ufld_v06": {
        "role": "temporal", "origin": "New in v0.4 (port of lite v0.5)",
        "frames": "t-4, t-2, t", "history": "warp + gate", "aligns": "yes (learned flow)", "memory": "3-frame window",
        "what": "A learned flow field (up to 4 cells of the 12x16 feature grid, about 128 px) warps each history "
                "map onto the current frame. The aligned history is compressed and blended in through a per-cell "
                "gate that starts almost closed.",
        "differs": "The only UFLD model that aligns frames before fusing, so moving markings are not blurred. It "
                   "adds about 1.7 M parameters to the baseline.",
    },
    "ufld_v07": {
        "role": "temporal", "origin": "New in v0.4",
        "frames": "t-4, t-2, t", "history": "ConvGRU", "aligns": "no", "memory": "recurrent state",
        "what": "A convolutional GRU (64 channels) runs over the reduced feature maps from the oldest frame to the "
                "current one. Its final state is projected back and added to the current features. The "
                "projection starts at zero, so the untrained model is exactly the baseline.",
        "differs": "Remembers through a gated recurrence instead of a weighted sum. A deployment can carry one "
                   "hidden state per camera instead of storing past feature maps.",
    },
    "ufld_v08": {
        "role": "temporal", "origin": "New on 2026-10-05",
        "frames": "t-4, t-2, t", "history": "cross-frame attention", "aligns": "implicitly (attention)",
        "memory": "3-frame window",
        "what": "Every cell of the current feature map attends (4 heads, 128 channels) to the cells of all three "
                "frames, each frame with a learned embedding; the result is projected back and added to the "
                "current features. The projection starts at zero, so the untrained model is exactly the baseline.",
        "differs": "Takes context from any place and time in the window, the approach of TGC-Net and LaneTCA, "
                   "instead of the same cell (v0.2, v0.3), a learned displacement (v0.6) or a recurrent state "
                   "(v0.7).",
    },
    "lite_baseline": {
        "role": "baseline", "origin": "ELAS development script",
        "frames": "t", "history": "none", "aligns": "no", "memory": "none",
        "what": "A small CNN: four stride-2 convolution blocks (32 to 128 channels) and a head that pools the "
                "features to the 18x100 row-anchor grid and classifies every row with a shared MLP. Trained from "
                "scratch, without ImageNet.",
        "differs": "The reference of the lite family: about 8 times fewer parameters and 3 to 4 times less "
                   "compute than UFLD.",
    },
    "lite_baseline_ct": {
        "role": "control", "origin": "v0.4 control",
        "frames": "t", "history": "none", "aligns": "no", "memory": "none",
        "what": "The lite baseline trained a second time from its own best checkpoint with the temporal "
                "variants' schedule.",
        "differs": "The equal-training control for lite v0.5 and lite v0.6.",
    },
    "lite_v05": {
        "role": "temporal", "origin": "ELAS development script (refactored)",
        "frames": "t-4, t-2, t", "history": "warp + gate", "aligns": "yes (learned flow)", "memory": "3-frame window",
        "what": "The lite baseline plus warped residual fusion: a learned flow (up to 8 cells at stride 16, about "
                "128 px) aligns each history feature map to the current one, and a per-pixel gate that starts "
                "closed mixes the aligned history in.",
        "differs": "Same fusion idea as UFLD v0.6 on a network that is 8 times smaller.",
    },
    "lite_v05_static": {
        "role": "control", "origin": "v0.4 control",
        "frames": "t, t, t", "history": "warp + gate on copies of t", "aligns": "n/a", "memory": "none",
        "what": "Lite v0.5 exactly (same layers, warm start, loss and budget), but every history frame is replaced "
                "by the current frame, in training and at test time.",
        "differs": "A capacity control. Lite v0.5 has extra fusion layers; if it beats this model, the gain comes "
                   "from the earlier frames and not from the extra layers.",
    },
    "lite_v06": {
        "role": "temporal", "origin": "New in v0.4",
        "frames": "t-4, t-2, t", "history": "ConvGRU", "aligns": "no", "memory": "recurrent state",
        "what": "The lite baseline plus the ConvGRU fusion of UFLD v0.7 (64 hidden channels, read-out starting "
                "at zero).",
        "differs": "Recurrent memory on the smallest network: the candidate for an embedded system that keeps "
                   "one state tensor per camera. Unlike v0.5 it does not align frames explicitly.",
    },
    "lite_v07": {
        "role": "temporal", "origin": "New on 2026-10-05",
        "frames": "t-4, t-2, t", "history": "cross-frame attention", "aligns": "implicitly (attention)",
        "memory": "3-frame window",
        "what": "The lite baseline plus the cross-frame attention of UFLD v0.8; keys and values are pooled to "
                "at most 12x20 cells per frame to keep the cost low.",
        "differs": "Attention-based memory on the smallest network, next to lite v0.5 (flow) and lite v0.6 "
                   "(recurrent).",
    },
    "lite_v06_occ": {
        "role": "temporal", "origin": "New on 2026-10-19 (idea of OMR / SIGMA-Lane)",
        "frames": "t-4, t-2, t", "history": "ConvGRU, occlusion-gated write", "aligns": "no",
        "memory": "recurrent state",
        "what": "Lite v0.6 with one extra 3x3 convolution that estimates, per feature cell, how visible the "
                "incoming frame is. The GRU update gate is multiplied by this visibility, so a cell that is "
                "occluded, blurred or dark in a frame keeps its memory instead of overwriting it.",
        "differs": "Same memory as lite v0.6; only the write is gated. It targets the case where the current "
                   "frame is the bad one, and costs about 600 parameters more.",
    },
    "lite_tsm": {
        "role": "temporal", "origin": "New on 2026-10-19 (TSM, Lin et al. 2019)",
        "frames": "t-4, t-2, t", "history": "temporal channel shift", "aligns": "no",
        "memory": "one channel slice per shifted block",
        "what": "The lite baseline's exact layers, with 1/8 of the channels at the input of blocks 2, 3 and 4 "
                "taken from the previous frame (online temporal shift). Every block then mixes time with its "
                "ordinary convolutions.",
        "differs": "No fusion module at all: zero extra parameters and MACs. A deployment keeps one small slice "
                   "per shifted block from the previous frame, so each frame costs exactly one baseline pass.",
    },
    "lite_tsm_static": {
        "role": "control", "origin": "New on 2026-10-19 (control)",
        "frames": "t, t, t", "history": "shift on copies of t", "aligns": "n/a", "memory": "none",
        "what": "Lite TSM exactly (same layers, warm start, loss and budget), but every history frame is replaced "
                "by the current frame, in training and at test time.",
        "differs": "A capacity control. The shift alone changes the network (it mixes channels and pads the first "
                   "frame with zeros); if lite TSM beats this model, the gain comes from the earlier frames.",
    },
    "lite_tsm_kd": {
        "role": "temporal", "origin": "New on 2026-10-19 (knowledge distillation, Hinton et al. 2015)",
        "frames": "t-4, t-2, t", "history": "temporal channel shift", "aligns": "no",
        "memory": "one channel slice per shifted block",
        "what": "Lite TSM trained with an extra loss that makes its per-anchor distribution over the grid cells "
                "match the softened distribution of a trained UFLD v0.7 temporal model (the teacher) on the same "
                "clip.",
        "differs": "Identical network and cost to lite TSM at inference; only training differs. Compared with "
                   "lite TSM it measures what the teacher's knowledge adds.",
    },
    "lite_baseline_kd": {
        "role": "single-frame", "origin": "New on 2026-10-19 (knowledge distillation from a temporal teacher)",
        "frames": "t", "history": "none at inference", "aligns": "n/a", "memory": "none",
        "what": "The single-frame lite model, continued from the lite baseline, learning from the UFLD v0.7 "
                "temporal teacher, which sees t-4, t-2 and t while the student sees only t.",
        "differs": "Temporal knowledge at zero runtime cost: the student never reads a past frame. Compared with "
                   "lite_baseline_ct (same extra training, no teacher).",
    },
}

KALMAN_NOTE = ("Any model + Kalman tracker: a causal constant-velocity Kalman filter on every lane point, with "
               "existence smoothing, tuned on validation. It costs microseconds per frame and is the bar a "
               "learned temporal model has to clear.")

DIFFERENCES = [
    "Every temporal model keeps its baseline's backbone and head and only adds a fusion step between them. It "
    "is compared with its own baseline on the same seeds.",
    "Three questions separate the temporal models: do they align the frames before fusing (v0.5, v0.6), how do "
    "they remember (a fixed 3-frame window for v0.2 to v0.6, a recurrent state for v0.7 and lite v0.6), and what "
    "do they cost (UFLD about 7 GMACs per frame, lite about 2).",
    "With feature caching each frame is encoded once, so a temporal model costs its baseline plus the fusion "
    "step, not three backbones.",
    "The controls are not candidates. +CT separates a temporal gain from extra training, lite v0.5 static "
    "separates it from the extra fusion layers, and the Kalman tracker shows what simple output filtering already "
    "gives.",
]
