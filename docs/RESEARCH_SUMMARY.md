# Research summary: challenges and next steps

Shown in the "Where the research stands" section of the results page, under the key figures and charts. Those numbers are computed from the run folders; the text here is written by hand (30 Sep 2026). Only lines starting with `- ` or `1. ` under a `## ` heading are shown.

## Challenges

- **Small dataset.** The protocol uses 10 ELAS scenes, and only 3 are held out for testing. UFLD memorised lane positions until geometric augmentation was added.
- **Clean frames hide the temporal effect.** On clean held-out frames the temporal UFLD models stay within about ±0.02 lane F1 of the baseline. Two seeds cannot separate that from noise.
- **The lightweight models train slowly and unstably.** Trained from scratch, they need 16 epochs or more. Lite v0.5 without real history (the capacity control) scored 0.877 in one seed and 0.627 in the other, so how much of lite v0.5's gain comes from the earlier frames is still open.
- **Memory over long sequences.** The recurrent models drift when their state is carried across a whole scene (lite v0.6: −0.48 lane F1). They were trained on 3-frame clips only.
- **More history is not better.** Spacing the history frames 2, 5 or 10 frames apart changes nothing; 5 frames instead of 3 costs lane F1.
- **One dataset so far.** CULane, TuSimple and OpenLane are not downloaded yet. Their readers are ready and were tested on synthetic copies of their layouts.
- **No embedded measurement yet.** The Jetson numbers are simulations based on assumed slowdowns, not measurements.

## Next steps

1. Run the full protocol on ELAS: 6 seeds, up to 50 epochs, and the same hyper-parameter search budget for every model (about 4–5 days on the RTX 3050).
2. Make robustness a primary result: the synthetic corruptions now, then natural occlusions, night and rain frames.
3. Download CULane and TuSimple (the download guide is ready), pre-train on them, fine-tune on ELAS, and report their official metrics.
4. Settle the lightweight question: more seeds and longer training for lite v0.5 against its capacity control, and train the recurrent models on long sequences.
5. Last phase, embedded: export the best lightweight temporal model to TensorRT (FP16/INT8) and measure latency, memory and power on a Jetson.
