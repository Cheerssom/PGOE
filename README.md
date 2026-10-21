# PGOE: Physics-Grounded Optical Editing

Reproducibility code, simulation-generated data, and frozen neural checkpoints for **Physics-Grounded Representation and Reversal Symmetry for Cross-Structure Local Editing of Optical Multilayers**, by Junyi Zhang, Shandong University.

This repository accompanies a manuscript prepared for submission to Knowledge-Based Systems. It is not a claim of journal acceptance. The task is single-step selection among signed local thickness edits and no-operation. All evaluated materials follow a lossless real Cauchy model at normal incidence with identical air exterior media. Thickness and wavelength units are micrometres.

## Contents

- `core.py`, `cli.py`: data generation, physical solvers, models, training and evaluation.
- `data/archived_train/`: the original 1,536 geometries (512 each at 8/12/16 layers), four training queries and original simulated gain labels.
- `data/development/`: exposed development domains.
- `data/new_confirmation/`: the separate frozen-model confirmation domains (256 each at 20 layers, 24 layers and strong-dispersion 16 layers).
- `checkpoints/`: 38 frozen neural checkpoints with training-only normalization, target scales and configurations. `CHECKPOINT_INDEX.json` lists all models. RF weights are not included.
- `results/`: per-seed scientific result summaries.
- `verification/single_candidate/`: executable quality replay of single-proposal physical acceptance.
- `verification/shortlist/`: executable quality replay of all fixed-budget shortlist policies on a separate 256-structure 24-layer set, including archived RF predictions, all measured latency trials and tail diagnostics.
- `SHA256SUMS.json`: hashes of public release files.

## Install

Python 3.10+; NumPy and PyTorch. Tested in the originating environment with NumPy 2.1.0 and PyTorch 2.11.0; cross-platform bitwise optimization identity is not guaranteed. CPU evaluation is supported. GPU training requires a compatible PyTorch CUDA installation.

```sh
python -m pip install -r requirements.txt
```

## Evaluate an archived checkpoint

From the repository root:

```sh
python cli.py evaluate --data data/new_confirmation/depth24 --checkpoint checkpoints/s0_full_aug --out runs/evaluation_seed0 --device cpu
```

Outputs must be new directories. Evaluate seeds separately: reported results average separate model decisions, not an ensemble of predicted gains. Normalization and scales are loaded from the checkpoint.

## Reproduce training with the original inputs

```sh
python cli.py train --data data/archived_train --out runs/train_seed0 --arm full_aug --seed 0 --device cuda
python cli.py train --data data/archived_train --out runs/generic_seed0 --arm generic_aug --seed 0 --device cuda
python cli.py train --data data/archived_train --out runs/optical_raw_seed0 --arm full_aug --seed 0 --device cuda --no-reversal
```

Default training: 800 AdamW updates, learning rate 0.001, weight decay 0.01, 64 original structures per batch and 128 object views including reversal. `--no-reversal` duplicates the original view to match object-forward work. All architectures use a shared zero-anchored action-conditioned readout. Input dimension is 24; the baseline uses 73,410 parameters, and both matched nonlinear-stem variants use 77,570. The checkpoint's `config.json` records the actual arm/seed/augmentation.

Additional packaged arms: `generic_mlp`, `full_mlp`, `no_phase_interface`, `no_global`, `prism_nm_rope`, `prism_scaled_rope`, `prism_nm_rope_b1000`, `prism_scaled_rope_b1000`. Individual optical feature groups are also implemented in `core.py`. RoPE entries are adaptations of positional mechanisms to the same editing task, not full reproductions of the PRISM generative model.

## Generate new simulation data

```sh
python cli.py generate --out runs/fresh_train --layers 8 12 16 --count 512
python cli.py generate --out runs/fresh_test --layers 24 --count 256 --seed 2026101771000 --evaluation
```

A generator run creates a new experiment and does not recreate historical inputs bit-for-bit. Data fields are `a,b,d`, `delta`, `truth`, and `lengths`. Training uses four signed queries (3--6 nm and 8--12 nm amplitude ranges); evaluation uses fixed +/-10 nm. The generator checks characteristic-matrix spectra against independent Fresnel recursion.

## Reproduce physical-verification decisions (NumPy only)

```sh
python verification/single_candidate/replay_w1.py
python verification/shortlist/replay_practical.py
```

These scripts run from any working directory and freshly recompute physical gains for archived proposals. They validate decision quality, not hardware-dependent timings or fresh model training. The replay uses full-stack NumPy TMM; the original shortlist timing benchmark used shared prefix/suffix optical states. Its replay runtime must not be treated as the paper's timing measurement. Full raw timing samples and original boundary descriptions remain in the corresponding summary files.

The single-candidate follow-up reuses exposed confirmation structures. The shortlist test froze predictors and fixed K=8 before its separate 256-structure evaluation; K=1/2/4/8/16 are all reported without test-based budget selection. Full-set PGOE quality uses five seeds; timing compares three seeds on a nested 64-structure subset. Do not mix those scopes.

## Interpretation

PGOE learns cross-structure finite-edit decisions; it is not an exact solver. Exact finite-action TMM remains the oracle, and the full-band second-order physical reference retains higher decision quality. Physical verification can reject negative simulated gains but does not establish manufacturing/noise robustness. Strong-dispersion generalization is conditional, and broadcast global statistics do not have a stable benefit. Geometry-resampled intervals condition on the stated fixed models rather than estimating training-seed population uncertainty. Data in this release are now public/exposed and must not be used as unseen confirmation for future tuning.

## Citation and status

Author: Junyi Zhang. Manuscript title is given above. A journal DOI and publication status will be added only when assigned. No DOI or acceptance is implied by this repository.
