# Hybrid NAFNet–Mamba Architecture for Fluorescence Microscopy Denoising

This repository provides the official implementation of the 3-fold cross-validation training and evaluation pipeline for fluorescence microscopy image restoration, combining nonlinear activation-free convolutional blocks (NAFNet) with a state-space model bottleneck (MambaIRv2).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[YourUsername]/mamba-microscopy-denoising/blob/main/notebooks/3_Fold_Training_Pipeline.ipynb)

## Quick Start & Interactive Demo

For an end-to-end cloud environment with zero local setup, run our complete interactive notebook on Google Colab:
* **Notebook:** [`notebooks/3_Fold_Training_Pipeline.ipynb`](notebooks/3_Fold_Training_Pipeline.ipynb)
* **Features:** End-to-end data extraction, 3-fold cross-validation, live metric tracking (PSNR, SSIM, SI-PSNR), and qualitative visual comparisons.

## Repository Overview

```text
├── configs/           # Experiment hyperparameter YAML files
├── notebooks/         # Standalone Colab training & inference notebook
├── src/
│   ├── models/        # Hybrid NAFNet-Mamba & CARE U-Net baselines
│   ├── data/          # 2D & 2.5D dataset loaders, transforms, & splitters
│   ├── losses/        # Composite loss (Charbonnier + MSE + Edge)
│   └── utils/         # In-text evaluation metrics (PSNR/SSIM/SI-PSNR) & TTA
├── tests/             # Unit and smoke tests for modular components
├── train.py           # Unified 3-fold cross-validation CLI
├── evaluate.py        # Independent evaluation and inference script
├── requirements.txt   # Pip dependencies
└── environment.yml    # Conda environment configuration
```

## Dataset Acquisition

The datasets used in this work originate from the **AI4Life Microscopy Supervised Denoising Challenge (MDC25)**:

* **Nuclei:** 2D cell nucleus fluorescence microscopy pairs.
* **FMD (Fluorescence Microscopy Dataset):** Real confocal, two-photon, and widefield microscopy images under various Poisson-Gaussian noise conditions.
* **Planaria:** Volumetric confocal imaging of *Schmidtea mediterranea* (processed via sliding 5-slice 2.5D windows).
* **Tribolium:** Volumetric light-sheet microscopy of *Tribolium castaneum* embryos (processed via sliding 5-slice 2.5D windows).

Download the datasets from the official AI4Life MDC25 challenge portal, and place the uncompressed directories inside `./data/` matching the paths specified in `configs/`:

```text
data/
├── AI4Life-MDC25-Nuclei/
├── AI4Life-MDC25-FMD/
├── planaria_2d/
└── tribolium_2d/
```

## Setup & Dependencies

Hardware requirements: NVIDIA GPU with CUDA 12.1 support (e.g., T4/V100/A100).

```bash
# 1. Clone the repository and submodule dependencies
git clone [https://github.com/](https://github.com/)[YourUsername]/mamba-microscopy-denoising.git
cd mamba-microscopy-denoising
git clone [https://github.com/csguoh/MambaIR.git](https://github.com/csguoh/MambaIR.git)

# 2. Set up Conda environment
conda env create -f environment.yml
conda activate mamba-denoising
```

Alternatively, install dependencies via `pip`:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install -r requirements.txt
pip install causal-conv1d==1.4.0
pip install mamba-ssm==2.2.2 --no-build-isolation
```

## Verification Without Data

To verify model forward/backward passes, layer shapes, and loss calculations on synthetic tensors without downloading datasets:

```bash
python -m tests.test_pipeline
```

## Verification Without Data

To verify model forward/backward passes, layer shapes, and loss calculations on synthetic tensors without downloading datasets:

```bash
python -m tests.test_pipeline
```

## Standalone Evaluation

To run inference and compute strict metrics using 8-fold test-time augmentation (dihedral group transforms):

```bash
python evaluate.py \
  --config configs/2d_nuclei.yaml \
  --weights checkpoints/2d_nuclei/model_fold1_best.pth \
  --tta x8
```

## Citation

If you use this codebase or benchmark methodology in your research, please cite:

```bibtex
@mastersthesis{anagnostopoulos2026hybrid,
  author  = {Theodoros Anagnostopoulos},
  title   = {Fluorescence Microscopy Image Restoration Using Hybrid Deep Learning Architectures},
  school  = {University of West Attica},
  year    = {2026}
}
```

## LicenseDistributed under the MIT License. 
See LICENSE for more information.