# Compare Overfit Design

**Goal:** Extend the fixed-batch overfit debug script so it can compare `plain AE` and `iFSQ AE` on the same normalized batch, print both loss trajectories in the terminal, and save both histories to JSON for later analysis.

**Why this helps:** We already know the data loader is healthy. The next useful isolation step is to hold data, batch, optimizer settings, and loss constant while swapping only the quantization path. If `plain AE` overfits quickly and `iFSQ AE` does not, the quantizer path becomes the primary suspect.

**Chosen approach:** Reuse the existing [scripts/debug_plain_ae_overfit.py](/home/humanoid/yzh/TextOp/motion_ae/scripts/debug_plain_ae_overfit.py:1) script instead of creating a separate compare script.

**Scope**

- Keep the current fixed-batch workflow and config reuse.
- Add a second model branch using [motion_ae/models/autoencoder.py](/home/humanoid/yzh/TextOp/motion_ae/motion_ae/models/autoencoder.py:1).
- Train both models independently on the exact same batch for the same number of steps.
- Print labeled progress for each branch in the terminal.
- Save a single JSON file containing config summary, batch metadata, and both histories.

**Output design**

- Terminal:
  - device, split, batch shape
  - `zero_baseline_loss`
  - `init_model_loss` per model
  - stepwise loss logs labeled `plain-ae` and `ifsq-ae`
  - final summary with first/last/min loss per model
- JSON:
  - `meta`: config path, split, steps, batch index, batch shape
  - `plain_ae`: baseline, init loss, history, summary
  - `ifsq_ae`: baseline, init loss, history, summary

**Non-goals**

- No plotting dependency
- No wandb integration
- No changes to the main training loop
- No model architecture changes beyond reusing the existing models
