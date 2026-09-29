# Preprocessing, augmentation and training controls

All options are **off or identical to v0.2 by default**; existing configurations reproduce the previous behaviour bit for bit (tested: the RGB input pipeline and the original photometric augmentation give identical tensors for identical seeds). Every option is recorded in the report's *Protocol* section and in `report/config_changes.json`.

## Per-item pipeline (training, evaluation and deployment)

1. load the RGB frame and resize it to the model size (PIL, bilinear);
2. geometric augmentation (training only): one homography for the whole clip, targets re-encoded;
3. photometric augmentation (training only), sampled once per clip;
4. preprocessing to the input representation (`data.preprocessing`);
5. per-channel normalisation.

Steps 1, 4 and 5 are the same code in `TemporalLaneDataset`, streaming inference, the ONNX/TensorRT runners and the UI (`FramePreprocessor`); the preprocessing config is stored in every checkpoint and in `deployment.json`, so a model cannot be deployed with a different representation.

## Preprocessing ablations (`data.preprocessing`)

| `mode` | Channels | Content | Model change |
|---|---|---|---|
| `rgb` (default) | 3 | original | none |
| `gray3` | 3 | luminance replicated | none (ImageNet weights used as is) |
| `gray` | 1 | luminance | `conv1` 1 channel; ImageNet filters summed over RGB |
| `edge` | 1 | Sobel magnitude (`sobel_ksize`; `edge_normalization: max | fixed`) | as `gray` |
| `canny` | 1 | Canny map (`canny_low`, `canny_high`, `canny_l2`) | as `gray` |
| `hough` | 1 | probabilistic-Hough segments from the Canny map (`hough_threshold`, `hough_min_line_length`, `hough_max_line_gap`, `hough_thickness`); near-horizontal segments below `hough_min_angle_deg` dropped | as `gray` |
| `rgb_edge` | 4 | RGB + Sobel or Canny (`edge_source`) | `conv1` 4 channels; extra channel initialised with the mean RGB filter |

`pre_ops` runs `blur`, `clahe` and/or `equalize` first, **in the listed order**. Normalisation: ImageNet statistics for RGB/gray3, their channel mean for gray, `feature_mean`/`feature_std` (0.5/0.5) for edge-type channels.

**Architectural implication.** Modes with ≠ 3 channels change the first convolution. With ImageNet initialisation the adapted filters are exact for grey input (a grey image and the same image replicated to RGB give the same first-layer response, tested) but not for edge maps, so those arms start further from a good optimum. They are separate configurations, compared as ablation arms, never substituted for the RGB baseline. The ablation runner prints the input channels of every arm.

These are experimental options; nothing here assumes they help.

## Geometric augmentation (`data.augmentation.geometric`)

Translation (`translate_x/y`, fraction of the size), isotropic scaling (`scale`), small rotations (`rotate_deg` ≤ 30), perspective jitter of the four corners (`perspective`), crop and resize (`crop_scale`), horizontal flip (`hflip_prob`). One homography per clip is applied to every frame (the motion between frames is preserved) and to the lane polylines; the row-anchor targets are re-encoded:

* homographies keep straight segments straight, so transforming the annotated vertices is exact;
* a lane that leaves the image at a row becomes "no lane" there, exactly like the unaugmented encoder;
* a row may be labelled "no lane" only if all its visible content maps inside the annotated band of the source image (`min_row_valid`, default 1.0 = strict); rows that show unannotated content (e.g. above the ELAS ROI after a rotation) are ignored;
* a lane whose transformed points are no longer ordered in y is ignored for that sample;
* flips permute slots with the dataset's `flip_permutation`; a dataset without one cannot enable flipping (config error).

Tests (`tests/test_preprocess_augment.py`): identity transform = original encoder; translation shifts targets exactly; re-encoded targets land on the drawn lane pixels of the warped image (> 95 % of 30 random transforms incl. flips); ROI handling under rotation; folded lanes ignored; deterministic sampling for a fixed seed.

## Photometric augmentation (`data.augmentation`)

Original (v0.2, default on): brightness, contrast, saturation, Gaussian noise, random erasing, history-frame dropout. New, off by default: `gamma`, `hue` (rotation in YIQ space), `shadow_prob`/`shadow_strength` (random cast-shadow band), `blur_prob`/`blur_sigma`, `motion_blur_prob`/`motion_blur_kernel`. All are sampled once per clip. When off they draw no random numbers, so the random stream of existing runs is unchanged.

## Training controls

| Control | Where | Default |
|---|---|---|
| Weight decay | `train.weight_decay` | 1e-4 (tuned by HPO in the full run) |
| Learning-rate schedule | `train.scheduler` (`cos`, `multi`, `none`) + `warmup_iters` | cosine, 100 warm-up iterations |
| Gradient clipping | `train.grad_clip_norm` | 1.0 |
| Early stopping | `train.early_stopping_patience`, on the declared selection metric | 10 (full), 2 (pilot) |
| Best-checkpoint selection | validation `evaluation.selection_metric`, tie broken by validation loss | `lane_f1_iou50` |
| Dropout | `model.head_dropout` (UFLD head, before the last Linear; parameter-free, state dicts unchanged); `model.lite_dropout`, `model.lite_head_dropout` (original values) | 0.0 / 0.05 / 0.15 |
| Label smoothing | `train.loss.label_smoothing` (uniform target mixed into the focal loss; exactly the official loss at 0) | 0.0 |
| Resume | `--resume`: finished variants are skipped (checkpoint **and** history), an interrupted variant continues from `<variant>.last.pt` (model, optimiser, scheduler, AMP scaler, RNG and data-order state), finished HPO studies are reused | `train.save_last: true` |
| Monitoring | per-epoch CSV + TensorBoard + overfitting flags in the log | on |

**Fixed during this work.** `--resume` previously skipped any variant whose best checkpoint existed, but that checkpoint is written during training, so an interrupted variant was treated as finished. Completion is now proven by the history CSV.

`python -m tac_ufld sanity --config <cfg>` checks, without training, that every variant has the right output shape, finite outputs and loss, finite gradients reaching backbone, head and fusion (measured on the second step, because the lite head is zero-initialised and blocks gradients at step 0 by design), a weight update, and strict, equal warm starts (UFLD temporal ← `ufld` of the baseline; lite v0.5 ← `backbone` + `head`).

## Ablation runner

```powershell
python -m tac_ufld ablate --spec configs/ablations/preprocessing.yaml            # plan only (never trains)
python -m tac_ufld ablate --spec configs/ablations/preprocessing.yaml --confirm  # run every arm
python -m tac_ufld ablate --spec configs/ablations/augmentation.yaml --summarize-only
```

An arm may not change the split, seeds, epochs, patience, HPO setting, model family, image size, anchors, temporal context or selection metric (rejected at load time and re-checked on the resolved configs). Each arm is an ordinary, resumable experiment folder; `ABLATION.md` compares every arm with the reference arm on the same seeds (paired Wilcoxon + Holm, `underpowered` flag). The shipped specs use the pilot budget with 3 seeds for screening; point `base_config` at `configs/elas.yaml` and use ≥ 6 seeds for a conclusive comparison.
