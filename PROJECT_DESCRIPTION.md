# Comprehensive Project Description: Industrial Surface Defect Detection

## 1. Project Purpose

This project addresses automated quality control in manufacturing through unsupervised anomaly detection. The core problem: in real factories, defective products are rare and defect types are unpredictable, so collecting labeled examples of every possible defect is impractical. Instead of training a classifier on "normal vs. defective," this project trains models exclusively on defect-free images and then flags anything that deviates from learned normality at test time.

Two anomaly detection approaches are compared -- a Convolutional Autoencoder (baseline, trained from scratch) and PatchCore (state-of-the-art, using pretrained ImageNet features) -- on three categories from the MVTec Anomaly Detection dataset: Bottle (rigid object), Hazelnut (organic object) and Carpet (texture). This selection deliberately covers both object-level and texture-level defect scenarios, which behave very differently. On top of binary detection ("is this image defective?"), the project performs pixel-level defect localization ("where is the defect?") and adds an explainability layer (Grad-CAM and occlusion sensitivity) that is quantitatively evaluated against ground truth masks.


## 2. Full Pipeline Overview

The project executes in seven sequential stages, each implemented as a standalone script:

**Stage 1: Data Download** (`scripts/download_data.py`)
Downloads the MVTec AD dataset for three categories (bottle, hazelnut, carpet) via a direct URL from mydrive.ch, with an anomalib-based fallback. Each category arrives as a directory containing `train/good/` (only normal images), `test/<defect_type>/` (normal + various defect types), and `ground_truth/<defect_type>/` (pixel-level binary masks marking exact defect regions).

**Stage 2: Baseline Training** (`scripts/train_baseline.py`)
Trains a Convolutional Autoencoder independently per category on defect-free training images. Uses MSE loss, Adam optimizer, ReduceLROnPlateau learning rate scheduler and early stopping with patience=10. Saves the best checkpoint (by validation loss) per category to `results/checkpoints/autoencoder_<category>.pth`.

**Stage 3: PatchCore Training** (`scripts/train_patchcore.py`)
Builds PatchCore memory banks per category using anomalib's Engine. A pretrained WideResNet50 backbone extracts patch-level features from layers 2 and 3 of the network. These features are reduced via coreset subsampling (ratio=0.1) to create a compact memory bank of normal patch embeddings. No gradient-based training occurs -- this is purely feature extraction and storage. Checkpoints are saved as `.ckpt` files under `results/checkpoints/patchcore_wide_resnet50_2/<category>/`.

**Stage 4: Evaluation** (`scripts/evaluate.py`)
Loads trained checkpoints for both methods and runs inference on the test set. Computes three metrics per (method, category) pair: Image AUROC (image-level detection), Pixel AUROC (pixel-level localization) and AUPRO (Per-Region Overlap, which gives equal weight to defects of different sizes). Results are saved incrementally to `results/metrics.csv` via a merge-on-key strategy that allows per-category reruns without losing other results.

**Stage 5: Qualitative Visualization** (`scripts/make_qualitative.py`)
Generates 3-row grid images for each (method, category) combination. Each grid shows 2 correct detections (highest-scoring anomalous images) and 1 failure case (lowest-scoring anomalous image, i.e., the hardest miss). Each row displays: Original Image | Ground Truth Mask | Predicted Anomaly Heatmap. Saved as PNG files in `results/qualitative/`.

**Stage 6: Backbone Ablation** (`scripts/ablation_backbone.py`)
Compares ResNet18 vs. WideResNet50 as PatchCore backbones. For each backbone and category, it builds a memory bank, evaluates on the test set and measures average inference time per image. Results go to `results/ablation.csv`.

**Stage 7: Explainability Analysis** (`scripts/explain.py`)
Applies Grad-CAM to the autoencoder and occlusion sensitivity to PatchCore on 20 anomalous test samples per category. Computes two XAI evaluation metrics (Pointing Game accuracy and Top-5% IoU) against ground truth masks. Generates attribution grid visualizations in `results/explainability/` and saves metrics to `results/explainability.csv`.


## 3. How Each Component Works

### 3.1 Convolutional Autoencoder (`src/autoencoder.py`)

The autoencoder is a symmetric encoder-decoder network with approximately 2.5 million parameters.

**Encoder:** Four convolutional blocks, each consisting of Conv2d (stride=2 for spatial downsampling) followed by BatchNorm2d and ReLU activation. The spatial dimensions halve at each block while channels increase: 3 -> 32 -> 64 -> 128 -> 256. Input images of 256x256x3 are compressed to a 16x16x256 bottleneck representation.

**Decoder:** Four transposed convolutional blocks (ConvTranspose2d with stride=2 for spatial upsampling) that mirror the encoder. Channels decrease: 256 -> 128 -> 64 -> 32 -> 3. The final layer uses Sigmoid activation to constrain output to [0, 1], matching the normalized input range.

**Anomaly Detection Logic:**
- `get_anomaly_map()`: Computes pixel-wise MSE between the input image and its reconstruction. Normal regions reconstruct well (low error); defective regions reconstruct poorly (high error) because the model has never seen defects during training.
- `get_image_score()`: Takes the top-k mean of pixel errors (k=100) rather than a simple maximum. This is more robust than max because it avoids being dominated by a single noisy pixel while still capturing the strongest anomaly signal.

### 3.2 PatchCore (via anomalib)

PatchCore operates on a fundamentally different principle. Instead of learning to reconstruct images, it builds a memory bank of what "normal" looks like in a pretrained feature space.

**Feature Extraction:** A WideResNet50 backbone pretrained on ImageNet extracts intermediate feature maps from layers 2 and 3 of the network. These layers capture mid-level visual features (edges, textures, parts) that transfer well to industrial inspection without any fine-tuning on the target domain.

**Memory Bank Construction:** All patch-level feature vectors from the training set (defect-free images only) are collected. Coreset subsampling reduces this bank to 10% of its original size while preserving its coverage of the feature space. This makes inference tractable without significant accuracy loss.

**Anomaly Scoring:** At test time, each patch's feature vector is compared to its nearest neighbour in the memory bank. The distance to the nearest normal patch becomes the anomaly score for that spatial location. Patches that look unlike anything in the normal memory bank get high scores. The per-patch scores form a spatial anomaly map; the image-level score is derived from the maximum patch distance.

### 3.3 Dataset Handling (`src/datasets.py`)

The `MVTecDataset` class handles the MVTec AD directory structure where training data contains only `good/` images and test data contains both `good/` and multiple defect-type subdirectories with corresponding ground truth masks.

Key constants: `IMG_SIZE = 256`, `CATEGORIES = ["bottle", "hazelnut", "carpet"]`.

Image transforms: `Resize(256, 256)` followed by `ToTensor()` (which normalizes pixel values from [0, 255] to [0, 1]). For the autoencoder, this [0, 1] range matches the Sigmoid output. For PatchCore, anomalib applies its own ImageNet normalization on top.

Mask transforms: `Resize(256, 256)` with `NEAREST` interpolation (to preserve binary mask edges without introducing intermediate gray values) followed by `ToTensor()` and binarization at threshold 0.5.

The `get_dataloaders()` function creates train/validation/test splits with a fixed random seed (42) for reproducibility.

### 3.4 Evaluation Metrics (`src/metrics.py`)

**Image AUROC:** Standard ROC AUC on image-level binary labels (normal=0, anomalous=1) vs. image-level anomaly scores. Measures detection quality.

**Pixel AUROC:** Flattens all pixel-level ground truth masks and anomaly maps across all test images, then computes ROC AUC. Measures localization quality but is biased toward large defects (more pixels = more weight).

**AUPRO (Per-Region Overlap):** Uses anomalib's internal `_AUPRO` class. At each threshold, computes overlap between the predicted binary mask and each connected defect region independently, then averages across regions. This gives equal weight to small and large defects. Integration is limited to FPR < 0.3 (standard in the literature). A scipy-based fallback using `ndimage.label` for connected component analysis is provided if anomalib is unavailable.

**Pointing Game Accuracy:** Checks whether the single highest-attribution pixel in an XAI map falls inside the ground truth defect region.

**Top-5% IoU:** Thresholds the attribution map at the top 5% of values, binarizes it and computes Intersection-over-Union with the ground truth mask.

### 3.5 Explainability (`src/explainability.py`)

**Grad-CAM for the Autoencoder:** A modified version of Grad-CAM (Selvaraju et al., 2017). Standard Grad-CAM applies global average pooling to the gradients before weighting activations, which works for classification but destroys spatial information here. Because the autoencoder's anomaly score is already spatially selective (it is built from per-pixel reconstruction errors), this project keeps the gradient at full spatial resolution: element-wise multiplication of gradient and activation, summed over channels, followed by ReLU. The result is bilinearly upsampled from the bottleneck resolution (16x16) to full image size (256x256).

**Occlusion Sensitivity for PatchCore:** Grad-CAM cannot be applied to PatchCore because anomalib wraps the backbone feature extraction in `torch.no_grad()`, so gradients never reach the input. Occlusion sensitivity (Zeiler and Fergus, 2014) sidesteps this by using only forward passes. A 16x16 patch filled with the image's channel-wise mean (not black or white, which would be out-of-distribution for the memory bank) is slid across the input. At each position, the anomaly map's total response (sum, not max -- to avoid border artifacts) is measured. The drop in anomaly score relative to the un-occluded image gives the attribution for that patch location.

### 3.6 Anomalib Compatibility Fix (`src/anomalib_compat.py`)

Patches a bug where anomalib's `make_mvtec_ad_dataset` compares a pandas Series against a `Split` enum using `==`, which returns all `False` because pandas does not know how to compare with that type. The fix monkey-patches the function to coerce the split enum to a plain string before comparison.

### 3.7 Results I/O (`src/results_io.py`)

`save_or_merge_csv()` enables incremental result accumulation. When a script finishes evaluating one category, it merges the new rows into the existing CSV by matching on key columns (e.g., method + category). This lets individual categories be re-run without wiping results from other categories.


## 4. Where Image Processing and Computer Vision Are Applied

### 4.1 Classical Image Processing Techniques

- **Image resizing** (`src/datasets.py`): All input images are resized to 256x256 using bilinear interpolation. This standardizes spatial dimensions across the dataset (MVTec AD images vary in original resolution per category).

- **Nearest-neighbour interpolation for masks** (`src/datasets.py`): Ground truth binary masks are resized using `InterpolationMode.NEAREST` instead of bilinear. This preserves the sharp binary edges of defect regions without introducing interpolation artifacts (gray pixels at boundaries).

- **Mask binarization via thresholding** (`src/datasets.py`): After resizing and converting masks to tensors, a threshold of 0.5 is applied to ensure strict binary values (0 or 1). Any residual intermediate values from format conversion are eliminated.

- **Pixel normalization** (`src/datasets.py`): `ToTensor()` scales pixel values from [0, 255] to [0, 1]. For PatchCore, additional ImageNet normalization (subtract mean, divide by std per channel) is applied.

- **Anomaly map generation via pixel-wise MSE** (`src/autoencoder.py`): The per-pixel squared difference between input and reconstruction is the core anomaly signal. This is a direct image processing operation -- comparing two images pixel by pixel to find where they disagree.

- **Heatmap overlay for visualization** (`scripts/make_qualitative.py`): Anomaly maps are min-max normalized, mapped to the "inferno" colormap and alpha-blended (alpha=0.5) onto the original image for visual inspection.

- **ImageNet un-normalization** (`scripts/make_qualitative.py`): PatchCore images stored by anomalib are in ImageNet-normalized space. Before display, they are un-normalized by multiplying by std and adding mean, then clipped to [0, 1] to handle floating-point drift.

- **Red overlay for mask visualization** (`notebooks/eda.ipynb`): Defect masks are overlaid on original images using alpha blending (0.6 original + 0.4 red-tinted overlay) for the EDA notebook.

### 4.2 Convolutional Neural Network Operations (Computer Vision)

- **Strided convolutions for spatial downsampling** (`src/autoencoder.py`): The encoder uses Conv2d with stride=2 to reduce spatial dimensions by half at each layer. This is a learned downsampling operation that replaces traditional pooling.

- **Transposed convolutions for spatial upsampling** (`src/autoencoder.py`): The decoder uses ConvTranspose2d with stride=2 to double spatial dimensions at each layer, reconstructing the full-resolution output from the compressed bottleneck.

- **Batch normalization** (`src/autoencoder.py`): BatchNorm2d normalizes activations per channel across the batch, stabilizing training and enabling higher learning rates.

- **Reconstruction-based anomaly detection** (`src/autoencoder.py`): The entire autoencoder architecture is a computer vision approach -- it learns a compressed visual representation of normal appearance and uses reconstruction error as an anomaly signal.

- **Transfer learning via pretrained backbone** (`scripts/train_patchcore.py`): PatchCore uses a WideResNet50 pretrained on ImageNet (1.2M natural images, 1000 classes). The intermediate feature maps from this backbone transfer to industrial inspection without any fine-tuning, exploiting the fact that mid-level visual features (edges, textures, shapes) generalize across domains.

- **Patch-level feature extraction** (PatchCore): Features are extracted at the patch level from layers 2 and 3 of WideResNet50. Each spatial position in these feature maps corresponds to a receptive field (patch) in the input image, capturing local visual patterns.

- **Nearest-neighbour anomaly scoring in feature space** (PatchCore): At test time, each patch feature is compared to the memory bank using L2 distance. This is a computer vision technique -- measuring visual similarity in a learned feature space rather than in raw pixel space.

- **Coreset subsampling** (PatchCore): Reduces the memory bank to 10% while maintaining coverage of the feature space. This is a computational geometry technique applied to high-dimensional visual feature vectors.

### 4.3 Explainability (CV + Image Processing)

- **Grad-CAM: gradient-weighted class activation mapping** (`src/explainability.py`): Hooks into the last convolutional layer of the encoder, backpropagates the anomaly score, and computes gradient-weighted activation maps. The spatial gradient is preserved (not pooled) because the anomaly score is already spatially selective.

- **Bilinear upsampling of attribution maps** (`src/explainability.py`): Grad-CAM output at bottleneck resolution (16x16) is bilinearly interpolated to full image size (256x256) using `F.interpolate(mode='bilinear')`.

- **ReLU activation on attribution maps** (`src/explainability.py`): Negative attributions are zeroed out with ReLU, keeping only regions that positively contribute to the anomaly score.

- **Occlusion sensitivity: perturbation-based attribution** (`src/explainability.py`): Systematically occludes patches of the input and measures anomaly score changes. The baseline fill value is the image's channel-wise mean (not a fixed color) to avoid introducing out-of-distribution artifacts.

### 4.4 Evaluation of Visual Explanations

- **Pointing Game** (`src/explainability.py`): Uses `np.argmax` on the flattened attribution map and `np.unravel_index` to find the 2D coordinates of the highest-attribution pixel, then checks if that pixel falls inside the ground truth mask. This is a spatial evaluation of visual explanation quality.

- **Top-5% IoU** (`src/explainability.py`): Thresholds the attribution map at the 95th percentile (`np.percentile`), binarizes it, and computes intersection-over-union with the ground truth. This evaluates how well the explanation's hottest region aligns with the actual defect.

- **Connected component analysis for AUPRO** (`src/metrics.py`): The PRO metric uses `scipy.ndimage.label` to identify individual connected defect regions in the ground truth mask, then evaluates overlap per region independently.


## 5. Results Summary

### Detection and Localization

| Method | Category | Image AUROC | Pixel AUROC | PRO |
|--------|----------|-------------|-------------|-----|
| Autoencoder | bottle | 0.8849 | 0.7247 | 0.4291 |
| Autoencoder | hazelnut | 0.9664 | 0.9400 | 0.8995 |
| Autoencoder | carpet | 0.3190 | 0.5471 | 0.2271 |
| PatchCore (WRN50) | bottle | 1.0000 | 0.9856 | 0.9445 |
| PatchCore (WRN50) | hazelnut | 1.0000 | 0.9882 | 0.9523 |
| PatchCore (WRN50) | carpet | 0.9864 | 0.9908 | 0.9494 |

PatchCore dominates on every category and metric. The autoencoder catastrophically fails on carpet (Image AUROC 0.319, worse than random) because reconstruction error cannot distinguish defects from normal texture variation in an already irregular-looking category.

### Backbone Ablation

WideResNet50 edges out ResNet18 by at most 1.3 points on any metric but costs 25-35% more inference time. ResNet18 is the better choice under latency constraints.

### Explainability

| Method | Category | Pointing Game | Top-5% IoU |
|--------|----------|---------------|------------|
| Autoencoder (Grad-CAM) | bottle | 0.45 | 0.165 |
| Autoencoder (Grad-CAM) | hazelnut | 0.55 | 0.186 |
| Autoencoder (Grad-CAM) | carpet | 0.00 | 0.017 |
| PatchCore (Occlusion) | bottle | 0.60 | 0.117 |
| PatchCore (Occlusion) | hazelnut | 0.35 | 0.077 |
| PatchCore (Occlusion) | carpet | 0.50 | 0.039 |

Neither method localizes tightly -- both operate at 16x16 resolution. The autoencoder's Grad-CAM scores 0.0 Pointing Game on carpet, mirroring its detection failure: an explanation cannot point at a defect the model is not finding.


## 6. What Has Been Completed

Every stage of the pipeline is fully implemented and has produced results:

1. Dataset download and EDA notebook
2. Autoencoder training (all 3 categories)
3. PatchCore memory bank construction (all 3 categories)
4. Full evaluation with all 5 metrics
5. Qualitative grid visualizations (6 grids: 2 methods x 3 categories)
6. Backbone ablation study (ResNet18 vs. WideResNet50)
7. Explainability analysis with quantitative XAI evaluation
8. Comprehensive README with results tables, method descriptions and known limitations
9. M1 project proposal document

The project is complete from data acquisition through final analysis. All scripts are runnable end-to-end with a single command sequence documented in the README.
