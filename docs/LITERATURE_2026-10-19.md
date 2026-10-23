# Literature for the open problems (2026-10-19)

Goal: a light temporal model that beats the UFLD v1 baseline on every criterion (accuracy, robustness, stability,
lane change, cost, simulation). Each section is one open problem measured in the pilots, the papers that address
it, and the concrete change it suggests. Every reference is in `Artigo_Mega_Super_Ultra_Fast/references.bib`
(Crossref-checked unless marked). Access: IEEE Xplore, ScienceDirect and SpringerLink through Portal de
Periódicos CAPES (UTFPR); arXiv for preprints.

## Where the light models stand (pilot, 1 seed, 4 epochs)

| criterion | UFLD v1 | best light | status |
|---|---|---|---|
| parameters | 21.8 M | 2.5 M (lite_tsm) | won |
| MACs per frame | 7.12 G | 1.99-2.39 G | won |
| streaming latency, RTX 3050 (Python) | 14-16 ms | 6-11 ms | won |
| clean held-out lane F1 | 0.881 | 0.938 | won (pilot) |
| current-frame corruption | not measured | 0.931 | measure UFLD |
| **persistent corruption** | not measured | 0.775 (< lite baseline 0.795) | **open: problem 1** |
| lane-change false alarms | not measured | 1.4 / min (lite v0.5) vs 0 (lite baseline) | **open: problem 2** |
| crossing-frame lane shape | - | bends at the crossing | **open: problem 3** |
| CARLA closed loop | not tested | fails (domain gap) | **open: problem 4** |

## Problem 1: temporal models over-trust the history

When every frame is corrupted, lite_tsm falls below its single-frame baseline. Its fusion never learned when
the past is unreliable.

* **Estimation theory (the mathematical answer).** A Kalman update weights prediction and measurement by their
  variances, `x = x_pred + K (z - x_pred)` with `K = P / (P + R)` (Särkkä and Svensson 2023, chapters 4-5). If
  both are uncertain, the output is uncertain: the filter does not "trust the past" by default. The learned
  gates (v0.3, v0.5, v0.6_occ) approximate K without a calibrated uncertainty. Change: predict a per-anchor
  variance (heteroscedastic regression, Kendall and Gal 2017, or Dirichlet evidence, Sensoy et al. 2018) for
  the current frame and for the propagated history, and fuse with the Kalman gain. This is a principled,
  explainable version of the occlusion gate, and it is cheap (one extra output channel).
* **Occlusion-aware memory**: OMR (Jin and Kim, ECCV 2024) and SIGMA-Lane (ECCV 2026) gate the memory write by an
  occlusion estimate. Already implemented as `lite_v06_occ`, not yet piloted with the fixed code.
* **Training signal**: masked sequential pre-training (Li and Dong, IEEE T-ITS 2023) masks frames or regions of
  a clip and reconstructs them, which teaches what to do when part of the evidence is missing. Combined with
  `current_frame_persistent_share` (implemented), the model sees both cases: clean history, and corrupted history.

## Problem 2: lane-change false alarms

Thresholds on offset and lateral speed (`ego.LaneChangeDetector`) fire on noisy, extrapolated lanes.

* **IMM**: Toledo-Moreo and Zamora-Izquierdo (IEEE T-ITS 2009) run two Kalman filters, "keep lane" and
  "change lane", combined by an interacting-multiple-model mixer; the manoeuvre probability replaces hand-tuned
  thresholds and has built-in hysteresis.
* **HMM + DTW**: Klitzke et al. (IEEE ITSC 2020) identify lane changes in real drives with a hidden Markov
  model, at an F1 of 98 %.
* **Change detection on lane geometry**: Schön, Eidehall and Gustafsson (IEEE IV 2006) detect lane departures
  inside the road-geometry estimator itself.
* **Change**: replace the thresholds with a two-model IMM on (offset, lateral speed), with the transition
  probabilities and the noise fitted on the training scenes. Report the event metrics on held-out scenes. The
  existing `native_elas_lc_*` metrics evaluate it unchanged.

## Problem 3: physically implausible lanes at the crossing (and jitter)

The per-cell Kalman reference (`evaluation/tracking.py`) filters each anchor independently, so nothing forces a
lane to be a smooth curve.

* **Clothoid road model**: Dickmanns and Mysliwetz (IEEE TPAMI 1992) estimate lane geometry recursively with a
  clothoid model (curvature linear in distance) and the vehicle's ego-state. McCall and Trivedi (IEEE T-ITS 2006)
  survey the same family.
* **Change**: an output tracker on (lateral offset, heading, curvature, curvature rate, lane width), with the
  lane change as a known discontinuity (re-initialise the reference lane at the crossing). It yields physically
  plausible lanes, less jitter, and directly the quantities that LKA needs.

## Problem 4: the ELAS -> CARLA gap

* **Domain randomisation**: Tobin et al. (IEEE/RSJ IROS 2017) randomise textures, lighting and camera so that
  the real world becomes "one more variation".
* **Sim-to-real for lanes**: Hu et al. (IEEE IV 2022) adapt with adversarial feature discriminators; the CARLANE
  benchmark (Gebele et al., NeurIPS 2022 Datasets and Benchmarks) gives CARLA-to-real lane domain-adaptation
  baselines.
* **Test-time adaptation**: Tent (Wang et al., ICLR 2021) updates only the BatchNorm affine parameters by entropy
  minimisation on the target stream. Cheap enough for an embedded device, and it needs no target labels.
* **Change** (now running in the CARLA session): fine-tune on CARLA clips from towns other than the test towns,
  then try Tent on the closed-loop stream as an ablation.

## Problem 5: beating UFLD, not only matching it

* **Distillation from a temporal teacher**: CLRKDNet (IEEE ITSC 2024), SAD (Hou et al., ICCV 2019), and the
  grouped distillation of TDNet (Hu et al., CVPR 2020). A trained UFLD temporal model teaching the light model is
  not found in the lane literature (one search, not a systematic review): it is a gap.
* **Zero-cost temporal mixing**: TSM (Lin et al., ICCV 2019; edge version, IEEE TPAMI 2020). Implemented as
  `lite_tsm`; the best light model so far.
* **Sequence models**: SIGMA-Lane (state-space model, ECCV 2026); heavier, a later step.

## Recommended order (cost vs. expected value)

1. Pilot every existing variant with the persistent-share recipe and all robustness protocols, UFLD included, to
   fill the table above (no new code; about 6 h).
2. Uncertainty-weighted (Kalman-gain) fusion for lite_tsm / lite_v06_occ (problem 1; about 1 day of code).
3. IMM lane-change detector (problem 2; about half a day; no retraining).
4. Clothoid output tracker (problem 3; about 1 day; no retraining).
5. Temporal-teacher distillation (problem 5; about 1 day + training).
