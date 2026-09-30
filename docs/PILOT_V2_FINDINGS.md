# v0.4 findings: augmentation ablation and second pilot (2026-09-30)

Indicative only: 2 to 3 seeds, at most 4 epochs, no hyper-parameter search. The list items below are shown on
the results page (`python -m tac_ufld site --notes docs/PILOT_V2_FINDINGS.md`).

- **Geometric augmentation removes the UFLD overfitting.** In the augmentation ablation (UFLD baseline, 3 seeds, same split and budget), small random shifts, zooms, rotations and perspective changes applied to the whole clip lift the best validation lane F1 from 0.161 to 0.772, and the held-out lane F1 from 0.068 to 0.858. The best epoch moves from 1 to 4, the last one: the model no longer memorises. Photometric changes alone (extended colour, blur, shadows) reach only 0.150 held-out.
- **Horizontal flipping does not add to it.** Geometric augmentation with flips scores 0.768 on validation and 0.769 held-out, below the version without flips; the recipe for the second pilot was chosen on validation (geometric, no flip).
- **Dropout + label smoothing + weight decay together make UFLD worse** (held-out 0.002). Lanes are still predicted in every frame but misplaced (anchor F1 falls to 0.05 to 0.12). The likely cause is label smoothing: it teaches a flatter distribution over the 100 grid cells, and the soft-argmax decoder turns that into a pull towards the image centre. This is a hypothesis; the arm changes three things at once.
