# NB3 cache provenance (2026-11-19)

All CSVs are **computed from the uploaded 2026-11-19 project ZIP's original checkpoint weights**, not hypothetical examples. Full numerical audit is `evaluation_log.txt` outside the submission or regenerable by NB3 `QUICK=False`.

- `nb2_main_seed_results.csv`: 15 comparisons; original learning rate `0.002`; experiment folders `experiment1`, `experiment2`, `baseline`, `experiment3`, `experiment4`; seeds 42, 123, 456. Recomputed with the classifier in `results/mnist_classifier.pt`, MNIST test images, `num_samples=2000`, fixed torch latent RNG=999 and NB2's `k=5` neighbourhood precision/recall. The aggregate exactly matches NB2 §8.2 up to rounding.
- `nb2_sensitivity_seed_results.csv`: same protocol, but checkpoints under `results/sensitivity_lr_0002/` with learning rate `0.0002`. Matches the saved executed output of current NB2 §6.5. **Important:** Current NB2's saved execution outputs correspond to sensitivity mode even though §8.2 text reports original-run results. NB3 intentionally separates these runs.
- `nb3_fixed_generator_results.csv`: baseline GAN `t=1` checkpoint is held fixed at each seed, and inference latent noise alone changes over `t in {0.25,0.5,1.0,1.5,2.0}`. Again 2000 test and 2000 synthetic features per condition, `k=5`, seeds 42, 123, 456. Corresponding `t=1` results reproduce original-run baseline exactly.
- `nb2_main_summary.csv`, `nb2_sensitivity_summary.csv`, `nb3_fixed_generator_summary.csv`: means and sample SDs, derived from their seed-level tables.

The Python module `nb3_evaluation.py` is the auditable/re-runnable implementation. The MNISTClassifier feature extractor, generator network and precision/recall definitions match NB2. NB3's `QUICK=False` regenerates these result tables; NB1/NB2 own retraining the generator checkpoints.
