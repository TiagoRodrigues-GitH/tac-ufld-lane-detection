# v0.4: making the temporal models earn their cost

The first pilot (`docs/PILOT_FINDINGS.md`) showed three problems: every UFLD model overfits after one epoch,
the temporal models barely use their history (replacing it by the current frame changes F1 by less than
0.003 for UFLD and 0.017 for lite v0.5), and lite v0.5's lead over its baseline came mostly from extra training.
v0.4 adds one tool per problem. Everything is off by default or opt-in, so every v0.3 configuration trains
exactly as before (the new augmentation draws no random numbers when disabled, and the optimizer keeps one
parameter group when the backbone multiplier is 1; both are tested).

Results of the runs that used these tools: `docs/PILOT_V2_FINDINGS.md` and the results page.

## 1. UFLD overfitting

| Remedy | Setting | Notes |
|---|---|---|
| Geometric augmentation | `data.augmentation.geometric.*` | homography per clip, labels re-encoded exactly (v0.3) |
| Extended photometric augmentation | `gamma`, `hue`, `blur_prob`, `motion_blur_prob`, `shadow_prob` | v0.3 |
| Dropout + label smoothing + weight decay | `model.head_dropout`, `train.loss.label_smoothing`, `train.weight_decay` | v0.3 |
| Lower backbone learning rate | `train.lr_backbone_mult: 0.1` | backbone lr = lr x multiplier; head and fusion keep theirs. Also an HPO parameter (`hpo.search_space.lr_backbone_mult`) |
| Frozen early layers | `train.freeze_backbone_stages: 2` | ResNet: 1 = stem, 2 = + layer1, ... 5 = all; lite: N of 4 blocks. Frozen stages keep their BatchNorm statistics |
| Smaller UFLD head | `model.ufld_head_hidden: 256` | the official head (2048 hidden units) holds about 10 M of the 21.8 M parameters; 256 units give 12.5 M in total. Not official UFLD any more; an overfitting test |
| More ELAS scenes | `data.scenes` (+ 8 scenes) | held-out test scenes unchanged; VV_S01/02/04 left out so that VV_S03 stays an unseen city |
| Pretraining on another dataset | `model.init_checkpoint`, `model.init_scope` | loads every matching tensor from an official UFLD checkpoint (`culane_18.pth`, `tusimple_18.pth`) or one of our checkpoints; not run (no CULane/TuSimple data or weights on the development machine) |

All of them are arms of `configs/ablations/augmentation.yaml` (UFLD baseline, 3 seeds, pilot budget), except
pretraining. `more_scenes` changes the training data on purpose, which the spec must declare
(`allow: [data.scenes]`); the held-out test set is identical in every arm.

## 2. Longer history

`configs/ablations/history.yaml` crosses frame steps {1, 2, 5, 10, 15} with clip lengths {2, 3, 4, 5}
(20 arms, named `s<step>_t<frames>`). Add values to the grid for more options. Two rules keep the arms comparable:

* one split for all arms: the purge gap (60 frames, `common:` in the spec) covers the longest span
  (15 x 4); 180-frame blocks keep about 80 % of the training frames of the 60/10 pilot split;
* one baseline for all arms: the single-frame baseline does not depend on the history setting, so it is trained
  once in the reference arm and copied into the others (`reuse_single_frame`); every temporal arm starts from
  the same weights.

For a new setting in the main configs, remember that `data.split.purge_frames` must be at least
`temporal_step x (num_frames - 1)` (the config check refuses anything smaller), which changes the split.

## 3. Current-frame degradation

`data.augmentation.current_frame_prob: 0.3` degrades only the **current** frame of 30 % of the training clips,
with one of `current_frame_ops`: solid occluding boxes centred in the road band, strong blur, darkening or
noise. The history frames stay clean and the label is unchanged, so the only way to recover the lanes is to use
the history. It is applied to every model, the single-frame baselines included (for them it is extra
augmentation), so the recipe stays identical across each comparison. Tests: the history is untouched, boxes lie
in the configured band, and the option draws no random numbers when it is 0.

## 4. Align before fusing: UFLD v0.6

`ufld_v06` puts lite v0.5's fusion on the UFLD layer-4 features: a learned flow (at most
`model.aligned_max_disp` = 4 cells of the 12 x 16 grid, about 128 px) warps each history map onto the current
frame, a 1 x 1 convolution compresses the warped history and a per-cell gate that starts almost closed blends it
in. About +1.7 M parameters. Warping with the vehicle's speed and yaw rate (CAN bus) instead of a learned flow
would need those signals, which ELAS does not provide; it is left for the in-car system.

## 5. Recurrent state: UFLD v0.7 and lite v0.6

`ConvGRUFusion` reduces the features to `model.recurrent_hidden` (64) channels, runs a convolutional GRU over
the frames and adds the projected final state to the current features. The projection starts at zero, so after
the warm start the model is exactly its baseline (tested). Two ways to run it:

* **window** (training, main results): the GRU restarts from zero on every clip, exactly like the other
  temporal models; streaming caches the per-frame features;
* **carry** (`StreamingLaneDetector(mode="carry")`, `stream --mode carry`): one hidden state per chain is kept
  for the whole stream, updated every `temporal_step` frames as in training; memory then reaches back to the
  start of the stream. It equals window mode until the chain is longer than the training clip (tested). The
  experiment reports carry minus window on the held-out scenes (`report/carry_state.csv`).

Carry mode runs on the PyTorch backend only; exporting the single-step graph (state in, state out) to ONNX /
TensorRT belongs to the embedded phase.

## 6. Equal-training controls

`ufld_baseline_ct` and `lite_baseline_ct` are the baselines trained a second time from their own best
checkpoint with the schedule the temporal variants get after their warm start. The report pairs every temporal
model with its baseline **and** with this control (`budget_reference`), and the control with the baseline
(does extra training alone help?).

## 7. Evaluate where time should help

* **Per-condition F1** (`report/test_conditions.csv`): lane F1 per ELAS scene tag (occlusion, rain, shaky,
  transition) on the held-out scenes, with jitter. The tags are per scene, so with three held-out scenes these
  are scene-level subsets.
* **Kalman reference** (`tac_ufld.evaluation.tracking`): a causal constant-velocity Kalman filter per lane point
  with existence smoothing and a re-initialisation gate, tuned on validation (`evaluation.kalman_grid`) and
  applied to every model (`input = kalman` rows, `report/kalman_reference.csv`,
  `report/paired_tests_kalman.csv`). A temporal model is only worth its cost if it beats
  "its baseline + Kalman". The tracker is also available in streaming (`stream --kalman`).
