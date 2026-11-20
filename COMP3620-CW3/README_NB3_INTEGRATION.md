# COMP3620 Coursework 3 — NB3 integrated teaching notebook

## What is included
- `NB1.ipynb`, `NB2.ipynb`: original submitted group baseline and experimental code from the latest uploaded repository (not silently edited).
- `NB3.ipynb`: updated 30-minute teaching notebook, with empirical seed variation, original/sensitivity comparison, a **three-seed fixed-generator control**, exercises and answer guidance.
- `nb3_evaluation.py`: helper implementing **the same feature-space k=5 generative precision/recall method as NB2**, for reproducible NB3 metric regeneration.
- `nb3_cache/*.csv`: recalculated evidence from the actual checkpoints, not manually typed data.
- `results/`: all 30 original and sensitivity checkpoints plus MNIST classifier; `data/MNIST/`: MNIST data for offline evaluation.

## Quick start
Open the project root in the school's `condaenv/dl2026` Jupyter kernel. Run `NB3.ipynb` top-to-bottom with `QUICK = True` in its first code cell. No machine-specific absolute path or online download is used by NB3. The project ZIP has been tested in a separate Python environment, but NOT on a university lab machine.

To recompute all **45 NB3 inference evaluations** (main, sensitivity and fixed-generator; 5 levels × 3 seeds each), use `QUICK = False`. This *does not retrain 100-epoch GANs*. For complete GAN weight retraining and saved model checkpoints, see NB1/NB2's `QUICK=False` code paths. NB3 cannot accurately claim a 12-step demo retrains the experiment.

## Important source consistency
The **saved output** of the latest NB2 evaluation cell corresponds to sensitivity learning rate `0.0002`, while NB2 Section 8's original-run summary table corresponds to `0.002`. NB3 caches *both* directly from their respective checkpoints and labels them explicitly. Main-run caches match NB2's Section 8.2 rounded averages.

## Archive size and data integrity
To fit Gradescope's 100 MB cap, only the old, ancillary `samples` **visual-history tensors** inside the `.pt` checkpoints were converted from `float32` to `float16`; the Generator/Discriminator `state_dict`s, training loss curves, and configuration fields were kept unmodified. This visual-history precision reduction may cause tiny display differences in NB1/NB2's image-history plots, but it does **not** alter learned weights, NB3 inference images, or metrics. No separate original `.pt` files were overwritten in the source ZIP.

## To confirm before submission
1. Run NB1, NB2 and NB3 top to bottom from a clean `condaenv/dl2026` kernel in lab machines; check `QUICK=True` finishes <5 minutes **per notebook**.
2. Ensure NB1 and NB2's saved outputs accurately match their current `SENSITIVITY_CHECK` / `QUICK` flags; latest NB2 output and written summaries are not in the same experimental condition.
3. Check the project's challenge statement, meeting and AI-use log, group video, and all member contributions separately. These are not included here.
4. For literal full GAN retraining, run NB1/NB2 with `QUICK=False`, which can take a long time. NB3 `QUICK=False` runs the full *teaching evaluation* from the saved weights.
