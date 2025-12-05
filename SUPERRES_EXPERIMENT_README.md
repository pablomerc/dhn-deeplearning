# Super-Resolution Experiment: Reproducing Figure 14 + Two-Body Extension

This document explains how to reproduce the trajectory interpolation (super-resolution) results from the DHN paper (Section 4.3, Figures 13-14) and extend them to the two-body problem.

## Overview

The experiment demonstrates **4× trajectory super-resolution** via progressive 2× stages:
- **Stage 0**: Sparsest trajectory (every 32nd point known)
- **Stage 1**: 2× super-resolution (every 16th point)
- **Stage 2**: Another 2× (every 8th point → final 4× resolution)

Key parameters matching the paper:
- Block size `b = 2`
- Stride `s = 1`
- Middle-state masking (two side states known, middle unknown)
- Shared global latent code `z` across all stages
- Autodecoder test-time optimization

## Quick Start

### Option 1: Run Everything at Once

```bash
# Full experiment (will take several hours)
python run_full_superres_experiment.py --phase all

# Or with custom epoch counts
python run_full_superres_experiment.py --phase all --train_epochs 200 --extract_epochs 1000
```

### Option 2: Run Phases Individually

```bash
# Step 1: Generate data (takes ~10-20 minutes)
python run_full_superres_experiment.py --phase data

# Step 2: Train models (takes several hours per model)
python run_full_superres_experiment.py --phase train --train_epochs 200

# Step 3: Extract (autodecode) - same-init and diff-init
python run_full_superres_experiment.py --phase extract --extract_epochs 1000

# Step 4: Generate plots
python run_full_superres_experiment.py --phase plot
```

### Option 3: Use Shell Scripts

```bash
# Generate data
bash scripts/generate_superres_data.sh

# Train all 6 models
bash scripts/superres_train_all.sh 200

# Run extraction for all models
bash scripts/superres_extract_all.sh

# Generate Figure 14-style plot
python visualization_scripts/plot_superres_results.py
```

## Quick Test (Verify Setup Works)

Before running the full experiment, verify everything works:

```bash
python run_quick_test.py
```

This generates minimal data and tests model creation, data loading, and forward/backward passes.

## Experiment Structure

### Systems
1. **Single Pendulum** (`q_dim=1`): Angular position θ
2. **Double Pendulum** (`q_dim=2`): Angular positions (θ₁, θ₂)
3. **Two-Body Problem** (`q_dim=2`): Polar coordinates (r, θ)

### Models per System
- **DHN**: Denoising Hamiltonian Network with H⁺/H⁻ blocks
- **CNN Baseline**: Convolutional neural network for comparison

### Evaluation Settings
- **Same-init**: Trajectories with same initial state as training (steps 0-512)
- **Diff-init**: Trajectories with different initial state (steps 512-1024, OOD)

## Expected Results

After running the experiment, you should see results similar to Figure 14 in the paper:

| System          | Method | Same Init (×100) | Diff Init (×100) |
|-----------------|--------|------------------|------------------|
| Single Pendulum | DHN    | ~0.05-0.1        | ~0.1-0.2         |
| Single Pendulum | CNN    | ~0.05-0.1        | ~0.5-1.0         |
| Double Pendulum | DHN    | ~0.1-0.2         | ~0.2-0.4         |
| Double Pendulum | CNN    | ~0.1-0.2         | ~1.0-2.0         |
| Two-Body        | DHN    | ~TBD             | ~TBD             |
| Two-Body        | CNN    | ~TBD             | ~TBD             |

Key insight: DHN generalizes better to OOD trajectories (diff-init) due to physical inductive bias.

## Output Files

After running the experiment:

```
results/superres/
├── sinpend_4x/                    # DHN Single Pendulum
│   ├── checkpoint.pth
│   ├── checkpoint_ep*.pth
│   ├── extract/                   # Same-init results
│   └── extract_ood/               # Diff-init results
├── sinpend_baseline_cnn_4x/       # CNN Single Pendulum
├── doupend_4x/                    # DHN Double Pendulum
├── doupend_baseline_cnn_4x/       # CNN Double Pendulum
├── two_body_4x/                   # DHN Two-Body
└── two_body_baseline_cnn_4x/      # CNN Two-Body

visualization_scripts/
├── superres_comparison.png        # Figure 14-style bar chart
└── superres_comparison.pdf        # PDF version for paper
```

## Configuration Details

### DHN Super-Resolution Config (`configs/superres/*_4x.py`)

```python
model.hamiltonian = 'superres'
model.stage_step_size = (32, 16, 8)    # 4× via 32→16→8
model.stage_block_size = (2, 2, 2)      # b=2 for all stages
model.stage_block_step = (1, 1, 1)      # s=1 for all stages
model.num_noise_scales = 10             # Denoising steps
model.train_step_span = (0, 513)        # Training segment

# Codebook (autodecoder)
codebook.num_embeddings = 1000          # One per trajectory
codebook.embedding_dim = 128            # Latent code dimension

# Transformer network
network.hidden_dim = 128
network.num_heads = 4
network.num_layers = 2
```

### Training Hyperparameters

| Parameter | Train | Extract |
|-----------|-------|---------|
| Epochs | 200 | 1000 |
| Learning Rate | 1e-4 | 1e-2 |
| Batch Size | 32 | 100 |
| Optimizer | Adam (weight_decay=1e-4) | Adam |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |

## For Your Write-Up

### Method Section
- DHN uses discrete Hamiltonian mechanics with H⁺/H⁻ blocks
- 4× super-resolution via repeated 2× stages (b=2, s=1)
- Middle-state masking pattern (side states known)
- Shared global latent code z per trajectory
- Test-time optimization: freeze network, optimize only z

### Key Results to Highlight
1. **DHN vs CNN on same-init**: Similar performance (both can memorize)
2. **DHN vs CNN on diff-init**: DHN generalizes much better
3. **Physical interpretation**: DHN's Hamiltonian structure provides strong inductive bias

### Two-Body Extension
- Same architecture works for different physical systems
- Polar coordinates (r, θ) used for reduced two-body problem
- Demonstrates generality of the DHN approach

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in configs:
```bash
--config.data.batch_size=16
```

### Missing Data
Regenerate data:
```bash
python run_full_superres_experiment.py --phase data
```

### WandB Issues
The trainer uses WandB for logging. If you don't have it configured:
```bash
wandb login
# OR disable wandb:
export WANDB_MODE=disabled
```

## Citation

If you use this code, please cite the original DHN paper:
```bibtex
@article{deng2025denoising,
  title={Denoising Hamiltonian Network for Physical Reasoning},
  author={Deng, Congyue and Feng, Brandon Y. and Garraffo, Cecilia and ...},
  journal={arXiv preprint arXiv:2503.07596},
  year={2025}
}
```
