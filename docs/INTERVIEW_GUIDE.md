# Interview guide: what each part of TAC-UFLD does

Short texts you can read aloud. Each one says what the part does, why it is there and one honest limitation.
Code paths are given so you can open the class while you explain it.

---

## The project in 30 seconds

> We detect the lane markings of the lane the car is driving in, from a single front camera, fast enough for an
> embedded computer. The starting point is Ultra-Fast Lane Detection (UFLD): instead of labelling every pixel, it
> looks at a few fixed image rows ("row anchors") and, for each row and each lane, picks which horizontal cell
> holds the lane, or "no lane". That is a classification over about 100 cells, which is why it is so fast. Our
> question: if the current frame is occluded (a truck, glare, rain), can the previous frames fill in the missing
> marking? We add temporal-fusion modules to UFLD and to a much smaller "light" network, and we compare each one
> with its own single-frame baseline under strictly equal conditions (same backbone, same training budget, same
> data, same post-processing tuned on validation, 7-8 paired seeds, Holm-corrected Wilcoxon tests).

## How the output is encoded (`data/targets.py`)

> For each of 18 row anchors and 2 lane slots (ego-left, ego-right) the target is a class: 0-99 is the horizontal
> cell where the lane crosses that row, 100 means "no lane at this row". Rows outside the annotated region, and
> lanes whose annotation is incomplete, get an *ignore* label so the network is not taught "no lane" where we
> simply do not know. At inference the position is the expected cell under the softmax (sub-cell precision), and
> existence is 1 - P(no lane), thresholded with a value tuned on validation.

## `UFLDNet` (`models/ufld.py`): the baseline network

> A ResNet-18, pre-trained on ImageNet, turns the 384x512 image into a 512-channel map at 1/32 resolution. A 1x1
> convolution squeezes it to 8 channels, we flatten it, and two fully connected layers output the 18x2x101
> classification scores. The fully connected head is the key design choice of UFLD: it sees the whole image at
> once, so it can "imagine" a lane where the paint is missing, using the global road shape. It is also the
> weakness: about 10 of its 22 million parameters sit in that head, and on a small dataset it overfits.
> This is UFLD **v1** (ECCV 2020). UFLDv2 (TPAMI 2022) adds column anchors for side lanes; see the review document.

## `UFLDSingleFrame` and `UFLDTemporal` (`models/ufld.py`)

> The single-frame model just runs UFLDNet on the current frame. The temporal model runs the SAME backbone on
> the current and two past frames (t-2, t-4), fuses the three feature maps with a fusion module, and runs the same
> head on the fused map. Because it holds the baseline as a sub-module, we initialise it with the trained
> baseline weights, so the only difference between the pair is the fusion. In a car we do not recompute the past
> frames: each frame's features are cached and reused (streaming).

## The fusion modules

**`TemporalWeightedFusion` (UFLD v0.2/v0.4).**
> One learned weight per frame (softmax, starting at 0.1/0.2/0.7). Simple, but it averages maps that are not
> aligned: if the car moves, the marking is in a different place in each frame.

**`GatedTemporalFusion` (UFLD v0.3, primary comparison).**
> A small CNN looks at the three feature maps and predicts, for every cell, how much to trust each frame. So in an
> occluded region it can lean on the past frames, elsewhere on the current one. It still does not align frames.

**`LearnableFlowWarp` + `ResidualTemporalFusion` (lite v0.5, UFLD v0.6).**
> First align, then fuse: a CNN predicts a displacement for each cell between a past map and the current one and
> resamples the past map, like a learned optical flow. Then a gate, closed at the start, mixes the aligned history
> into the current features. The flow is learned only from the lane loss, so it can do things that are not
> motion, which is why we test it against a "static" twin that gets the current frame three times.

**`ConvGRUFusion` (UFLD v0.7, lite v0.6).**
> A recurrent memory per feature-map cell, a GRU with convolutions. Each frame updates the memory, and the
> memory is added to the current features through a projection that starts at zero, so at the start the model
> is exactly its baseline. In deployment we carry only one memory tensor from frame to frame.

**`TemporalAttentionFusion` (UFLD v0.8, lite v0.7).**
> Each cell of the current frame asks "which places in the last frames look like me?" with multi-head attention;
> keys and values are pooled to at most 12x20 cells per frame to keep it cheap. Like the GRU, its output is
> added through a zero-initialised projection.

## The light models (`models/lite.py`)

**`LightweightBackbone`.**
> Four stride-2 convolution blocks (32, 64, 96, 128 channels), 0.46 M parameters, trained from scratch. Each
> output cell sees only about 120 pixels around it, so it is local, not global like the UFLD head.

**`AnchorRowPool` and `LanePixelHead`.**
> The head reduces the features, takes for every row anchor the feature row at that anchor's image height,
> averages it into 100 horizontal cells, and a small MLP shared by all anchors and lanes classifies the cell (or
> "no lane"). Taking the row *at the anchor* matters: we found in the 2026-10-18 review that the original pooling
> spread the anchors over the whole image, while they only cover the road, so the top anchor looked at the sky.
> That bug explained most of the light baseline's poor accuracy and part of the apparent temporal gains.

**`LiteSingleFrame` / `LiteTemporal`.**
> The same pattern as UFLD: the single-frame light model, and its temporal version with the same backbone and
> head, warm-started from it, plus a fusion module (warp, ConvGRU or attention).

## The controls (`models/registry.py`, `VariantSpec`)

> Two controls make the comparison fair. `*_baseline_ct` is the baseline trained again for the same extra epochs
> the temporal model gets after its warm start, so a gain cannot come from more training. `lite_v05_static` is the
> temporal model fed the current frame three times: same layers, same training, no temporal information, so a gain
> over it cannot come from extra capacity. The primary tests (declared before the run) are UFLD v0.3 vs UFLD
> baseline and lite v0.5 vs lite v0.5 static.

## The loss (`losses.py`, `VariantLoss`)

> Every model is trained with the official UFLD focal classification loss over the 101 classes. Some variants
> add terms from the supervisor's notebook: an existence loss, a prior on the gates, a sub-pixel coordinate loss,
> flow smoothness, and a temporal-consistency term that asks the single-frame predictions of consecutive frames to
> agree. Only the focal loss is logged as the comparable number across models.

## Evaluation (`evaluation/`, `metrics.py`)

> The main metric is the CULane-style lane F1: each predicted and ground-truth lane is drawn as a 30-pixel-wide
> line (scaled to the image), they are matched one-to-one (Hungarian), and a match counts if the IoU is at least
> 0.5. The post-processing threshold is tuned on validation for every model, the baseline included, and the test
> set is three (or four) whole scenes never seen in training. Robustness is measured by degrading only the
> current frame (occlusion, blur, darkness, noise), which is exactly the case temporal fusion should help.
>
> F1 alone hides which error a model makes, so we also report: precision and recall; phantom and missed lanes
> per frame; the share of frames that are completely right ("frame accuracy"); at the anchor cells, the
> false-alarm rate (FP / (FP + TN)) and the miss rate, which is where a true negative exists; the lateral position
> error in pixels, near and far; the lane-centre deviation in % of lane width, which is the ELAS paper's own
> metric; lane-change event recall and false alarms per minute; and the frame-to-frame jitter. A phantom lane is
> worse for lane keeping than a short miss, so the error split matters as much as F1.

## Ego-lane and lane-change analysis (`ego.py`)

> From the two ego boundaries we compute where the car is in its lane: the lane centre and width at the bottom of
> the image, and the lateral offset in lane widths (0 = centred, +-0.5 = on a line). Over time, a sustained drift
> toward a line with a short time-to-line-crossing, or a line passing under the camera, means a lane change to that
> side; it ends when the car is centred again. On the ELAS ground truth this finds all 8 annotated lane changes with
> the correct direction and no false alarm (upper bound with perfect lanes). Turning at intersections is not
> covered: the markings disappear, so it needs ego-motion (yaw rate) or a dataset with turn labels.

## ELAS (`data/elas.py`)

> A Brazilian video dataset (Espirito Santo, 29.97 FPS, 640x480, 22 scenes, ~17,000 frames) annotated for the ego
> lane only: four x positions per boundary at fixed fractions (0, 1/4, 1/2, 1) of a region of interest, plus
> frame labels for lane changes, lane-marking types, adjacent lanes, crosswalks and road signs. It has no vehicle
> boxes, so vehicle detection needs another dataset (OpenLane's closest-in-path-object boxes, BDD100K).
