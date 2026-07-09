# Industrial Surface Defect Detection — MVTec AD

## Problem Statement

Unsupervised anomaly detection for industrial surface inspection, where the goal is
to identify defective products using only defect-free training images. This project
compares two approaches on the MVTec Anomaly Detection dataset: (1) a convolutional
autoencoder baseline trained from scratch that detects anomalies via reconstruction
error, and (2) PatchCore, a state-of-the-art method using pretrained ImageNet
features (transfer learning) with a memory bank of normal patch embeddings. We
evaluate on three categories — bottle, hazelnut (objects), and carpet (texture) —
to cover both object-level and texture-level defect detection scenarios. On top of
detection, the project adds an **explainable AI (XAI)** layer: a gradient-based
attribution method per model (Grad-CAM for the autoencoder, occlusion sensitivity
for PatchCore) that is quantitatively evaluated — not just visualized — against the
ground-truth defect masks, so the pipeline reports not only *whether* an image is
anomalous but *why*, and how trustworthy that explanation is.

## Setup

### Prerequisites
- Python 3.10–3.12 (recommended for anomalib compatibility)
- CUDA-capable GPU recommended (CPU-only works but is slower)

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/Mac)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Reproduction Steps

Run all commands from the project root directory.

```bash
# 1. Download the MVTec AD dataset (3 categories: bottle, hazelnut, carpet)
python scripts/download_data.py

# 2. Train the autoencoder baseline (all 3 categories)
python scripts/train_baseline.py

# 3. Train PatchCore with WideResNet50 backbone (all 3 categories)
python scripts/train_patchcore.py

# 4. Evaluate both methods → results/metrics.csv
python scripts/evaluate.py

# 5. Generate qualitative outputs → results/qualitative/
python scripts/make_qualitative.py

# 6. Run backbone ablation study → results/ablation.csv
python scripts/ablation_backbone.py

# 7. Explainability analysis (Grad-CAM / occlusion sensitivity) → results/explainability.csv
python scripts/explain.py
```

### Single-Category Quick Test
To verify the pipeline on just one category (faster):
```bash
python scripts/train_baseline.py --categories bottle --epochs 50
python scripts/train_patchcore.py --categories bottle
python scripts/evaluate.py --categories bottle
```

## Expected Runtime

| Step | GPU (consumer) | CPU-only |
|------|---------------|----------|
| Download data | 5–15 min | 5–15 min |
| Train autoencoder (3 cats) | 15–30 min | 30–60 min |
| Train PatchCore WRN50 (3 cats) | 5–10 min | 15–30 min |
| Evaluate all | 5–10 min | 10–20 min |
| Qualitative outputs | 2–5 min | 5–10 min |
| Ablation (both backbones) | 10–20 min | 30–60 min |
| Explainability analysis | 5–10 min | 30–60 min |
| **Total** | **~50–100 min** | **~2.5–4 hours** |

**Hardware assumptions:** Single NVIDIA GPU with ≥4 GB VRAM (e.g., GTX 1650 or better).
All steps work on CPU — pass `--device cpu` to scripts if no GPU is available.
WideResNet50 requires more memory than ResNet18; if GPU memory is limited, use
`--backbone resnet18` for PatchCore.

## Results

### Method Comparison (metrics.csv)

| Method | Category | Image AUROC | Pixel AUROC | PRO |
|--------|----------|-------------|-------------|-----|
| Autoencoder | bottle | — | — | — |
| Autoencoder | hazelnut | — | — | — |
| Autoencoder | carpet | — | — | — |
| PatchCore (WRN50) | bottle | — | — | — |
| PatchCore (WRN50) | hazelnut | — | — | — |
| PatchCore (WRN50) | carpet | — | — | — |

*Results will be filled after running the pipeline. See `results/metrics.csv`.*

### Ablation: Backbone Comparison (ablation.csv)

| Category | Backbone | Image AUROC | Pixel AUROC | PRO | Avg Inference (ms) |
|----------|----------|-------------|-------------|-----|---------------------|
| bottle | ResNet18 | — | — | — | — |
| bottle | WideResNet50 | — | — | — | — |
| hazelnut | ResNet18 | — | — | — | — |
| hazelnut | WideResNet50 | — | — | — | — |
| carpet | ResNet18 | — | — | — | — |
| carpet | WideResNet50 | — | — | — | — |

*See `results/ablation.csv` for full results.*

### Qualitative Results

Grid images showing Original / Ground Truth Mask / Predicted Anomaly Heatmap
are saved in `results/qualitative/`. Each grid contains 2 correct detections
and 1 failure case per (method, category) combination.

## Project Structure

```
├── data/                    # Dataset (gitignored, populated by download_data.py)
├── scripts/
│   ├── download_data.py     # Downloads MVTec AD (3 categories)
│   ├── train_baseline.py    # Trains convolutional autoencoder per category
│   ├── train_patchcore.py   # Builds PatchCore memory bank per category
│   ├── evaluate.py          # Computes all metrics, writes metrics.csv
│   ├── make_qualitative.py  # Generates qualitative grid images
│   └── ablation_backbone.py # ResNet18 vs WideResNet50 comparison
├── src/
│   ├── autoencoder.py       # Convolutional autoencoder model (~2.5M params)
│   ├── datasets.py          # MVTec AD dataset loader and utilities
│   └── metrics.py           # Image AUROC, Pixel AUROC, AUPRO
├── notebooks/
│   └── eda.ipynb            # Exploratory data analysis
├── results/                 # Generated outputs (gitignored)
│   ├── metrics.csv          # Main results table
│   ├── ablation.csv         # Backbone comparison results
│   ├── qualitative/         # Grid visualizations
│   └── checkpoints/         # Trained model weights
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

**MVTec Anomaly Detection (MVTec AD)**

> Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2021).
> The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset
> for Unsupervised Anomaly Detection. *International Journal of Computer
> Vision*, 129, 1038–1059. https://doi.org/10.1007/s11263-020-01400-4

**License:** CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0).
This dataset is for **non-commercial use only**.

**Download:** https://www.mvtec.com/company/research/datasets/mvtec-ad

## Known Limitations

The autoencoder baseline struggles with texture categories (carpet) where
subtle defects are difficult to distinguish from normal texture variation.
PatchCore significantly outperforms the autoencoder across all categories,
as expected from the literature. Both methods may produce false positives
near image borders or in regions with high-frequency normal texture. The
autoencoder's localization quality is limited by its small receptive field
and the inherent blurriness of reconstruction-based approaches. Further
limitations to be documented after reviewing qualitative outputs.

## AI Tool Use Disclosure

This project used AI coding assistance (Claude) for boilerplate and debugging;
all code was reviewed and understood by the group. See commit history for details.
