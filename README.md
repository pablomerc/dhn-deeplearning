# Denoising Hamiltonian Network for Physical Reasoning

Created by
[Congyue Deng](https://cs.stanford.edu/~congyue/),
[Brandon Y. Feng](https://brandonyfeng.github.io/),
[Cecilia Garraffo](https://www.ceciliagarraffo.com/),
[Alan Garbarz](https://scholar.google.com/citations?user=vz9CBV0AAAAJ&hl=en),
[Robin Walters](https://www.robinwalters.com/),
[William T. Freeman](https://billf.mit.edu/),
[Leonidas Guibas](https://geometry.stanford.edu/?member=guibas), and
[Kaiming He](https://people.csail.mit.edu/kaiming/)

[**Paper**](https://arxiv.org/pdf/2503.07596) | [**Project**](https://cs.stanford.edu/~congyue/dhn/)

<img src='visuals/teaser.png' width=90%>

\
This repository contains the full code for all experiments in the paper, including various configurations and flags. For a simpler version with minimal implementation to help you get started quickly, check out [**this repo**](https://github.com/FlyingGiraffe/dhn).

## Installation

### Option 1: Using Conda (Recommended)

Create a conda environment with all dependencies:

```bash
conda env create -f environment.yml
conda activate dhn
```

**Note:** Adjust the `pytorch-cuda` version in `environment.yml` based on your CUDA version (11.8 or 12.1). If you don't have CUDA, remove the `pytorch-cuda` line.

### Option 2: Using pip

1. First, install PyTorch based on your system:
   - For CUDA 11.8: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
   - For CUDA 12.1: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
   - For CPU only: `pip install torch torchvision torchaudio`

2. Then install the remaining dependencies:
```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9+
- PyTorch 2.0+ (required for `torch.func`, `torch.vmap`, and `SDPBackend` features)
- CUDA support (optional but recommended for GPU acceleration)

## Data Preparation

Download the data from [**Google Drive**](https://drive.google.com/file/d/1ry6Lz1Wa77UZmApEghfm9aSydXU2vWWr/view?usp=sharing) and upzip it to the folder `data` into the following format:
```
data/
 ├── single_pendulum/
 │     ├── train/
 │     └── test
 └── double_pendulum/
       ├── train/
       └── test
```

You can also generate the data yourself by running
```
bash scripts/data_gen_train.sh
bash scripts/data_gen_test.sh
```
Change the variable `DATA_NAME` to generate simulated data for different physical systems (`single_pendulum` or `double_pendulum`).

## Experiments

Change the variables in the scripts to run different experiments:
- `EXP_CLASS` for different tasks: `ar` for forward simulation (autoregression and completion), `repn` for representation learning, and `superres` for trajectory interpolation (super-resolution).
- `EXP_NAME` for different config files.

All experimental results, including logs and checkpoints, will be under the directory `results/${EXP_CLASS}/${EXP_NAME}`.

### Forward Simulation

**Fitting known trajectories**\
Step 1: Run `train.sh` with `EXP_CLASS=ar`.\
Step 2: Run `generate.sh` with `EXP_CLASS=ar`.\
The generated sequences will be in a subfolder named `gen_sequence`.

**Completion on novel trajectories**\
Step 1: Run `train.sh` with `EXP_CLASS=ar`.\
Step 2: Run `extract_partial.sh` with `EXP_CLASS=ar`.\
Step 3: Run `generate_partial.sh` with `EXP_CLASS=ar`.\
The generated sequences will be in a subfolder named `extract/gen_sequence`.

### Representation Learning
Step 1: Run `train.sh` with `EXP_CLASS=repn`.\
Step 2: Run `extract.sh` with `EXP_CLASS=repn`.

### Trajectory Interpolation (Super-Resolution)
Step 1: Run `train.sh` with `EXP_CLASS=superres`.\
Step 2: Run `extract.sh` (for in-distribution trajectories with the same initial states) or `extract_ood.sh` (for out-of-distribution trajectories with different initial states) with `EXP_CLASS=superres`.

## License
MIT License
