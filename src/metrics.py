"""
Metrics for anomaly detection evaluation.

Computes:
  - Image-level AUROC (binary classification: normal vs anomalous)
  - Pixel-level AUROC (per-pixel anomaly localization quality)
  - AUPRO (Area Under Per-Region Overlap curve) — evaluates localization
    quality while giving equal weight to anomalies of different sizes.

Uses scikit-learn for AUROC and anomalib for AUPRO.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.explainability import pointing_game, topk_iou


def compute_image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute image-level AUROC.

    Args:
        labels: Binary labels [N], 0=normal, 1=anomalous
        scores: Anomaly scores [N], higher = more anomalous

    Returns:
        AUROC score in [0, 1]
    """
    # Need both classes present
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_pixel_auroc(masks: np.ndarray, anomaly_maps: np.ndarray) -> float:
    """
    Compute pixel-level AUROC.

    Args:
        masks: Ground-truth binary masks [N, H, W], 1=defect
        anomaly_maps: Predicted anomaly maps [N, H, W], higher = more anomalous

    Returns:
        Pixel-level AUROC score in [0, 1]
    """
    # Flatten all pixels
    masks_flat = masks.flatten().astype(int)
    maps_flat = anomaly_maps.flatten()

    # Need both classes present
    if len(np.unique(masks_flat)) < 2:
        return float("nan")

    return float(roc_auc_score(masks_flat, maps_flat))


def compute_pro(
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    fpr_limit: float = 0.3,
) -> float:
    """
    Compute AUPRO (Area Under Per-Region Overlap curve).

    Uses anomalib's AUPRO implementation for correctness and efficiency.
    The PRO metric evaluates localization quality while giving equal weight
    to anomalies of different sizes (unlike pixel AUROC which is biased
    toward large defects).

    Args:
        masks: Ground-truth binary masks [N, H, W]
        anomaly_maps: Predicted anomaly maps [N, H, W]
        fpr_limit: FPR integration limit (default 0.3, standard in literature)

    Returns:
        AUPRO score in [0, 1]
    """
    try:
        # Use the internal _AUPRO class which accepts raw tensors directly.
        # The public AUPRO class (anomalib >= 2.5) wraps it with AnomalibMetric
        # which requires Batch objects and a `fields` argument.
        try:
            from anomalib.metrics.aupro import _AUPRO
            aupro_metric = _AUPRO(fpr_limit=fpr_limit)
        except ImportError:
            from anomalib.metrics import AUPRO
            aupro_metric = AUPRO(fpr_limit=fpr_limit)

        # Convert to torch tensors if needed
        if isinstance(masks, np.ndarray):
            masks_t = torch.from_numpy(masks).long()
        else:
            masks_t = masks.long()

        if isinstance(anomaly_maps, np.ndarray):
            maps_t = torch.from_numpy(anomaly_maps).float()
        else:
            maps_t = anomaly_maps.float()

        # AUPRO expects [N, H, W] for both
        if masks_t.ndim == 4:
            masks_t = masks_t.squeeze(1)
        if maps_t.ndim == 4:
            maps_t = maps_t.squeeze(1)

        # Update metric in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(masks_t), batch_size):
            batch_masks = masks_t[i : i + batch_size]
            batch_maps = maps_t[i : i + batch_size]
            aupro_metric.update(batch_maps, batch_masks)

        return float(aupro_metric.compute())

    except ImportError:
        print("WARNING: anomalib not installed, falling back to simplified PRO computation")
        return _compute_pro_fallback(masks, anomaly_maps, fpr_limit)
    except Exception as e:
        print(f"WARNING: AUPRO computation failed ({e}), using fallback")
        return _compute_pro_fallback(masks, anomaly_maps, fpr_limit)


def _compute_pro_fallback(
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    fpr_limit: float = 0.3,
) -> float:
    """
    Simplified PRO fallback when anomalib is unavailable.

    This is a basic approximation — the anomalib version is preferred.
    Computes per-region overlap at multiple thresholds and integrates.
    """
    from scipy import ndimage

    # Only consider images with defects
    has_defect = masks.reshape(masks.shape[0], -1).max(axis=1) > 0
    if not has_defect.any():
        return float("nan")

    defect_masks = masks[has_defect]
    defect_maps = anomaly_maps[has_defect]

    # Sample thresholds from the anomaly maps
    thresholds = np.percentile(anomaly_maps.flatten(), np.linspace(0, 100, 200))
    thresholds = np.unique(thresholds)[::-1]  # Descending

    pro_values = []
    fpr_values = []

    # Normal pixels for FPR computation
    normal_pixels = masks.flatten() == 0
    total_normal = normal_pixels.sum()

    for thresh in thresholds:
        binary_pred = (anomaly_maps >= thresh).astype(float)

        # FPR on normal pixels
        fp = ((binary_pred.flatten() > 0) & normal_pixels).sum()
        fpr = fp / max(total_normal, 1)

        if fpr > fpr_limit:
            continue

        # Per-region overlap
        overlaps = []
        for i in range(len(defect_masks)):
            labeled_mask, num_regions = ndimage.label(defect_masks[i])
            for region_id in range(1, num_regions + 1):
                region = (labeled_mask == region_id)
                region_size = region.sum()
                if region_size == 0:
                    continue
                overlap = (binary_pred[i] * region).sum() / region_size
                overlaps.append(overlap)

        if overlaps:
            pro_values.append(np.mean(overlaps))
            fpr_values.append(fpr)

    if len(pro_values) < 2:
        return float("nan")

    # Sort by FPR and integrate
    sorted_idx = np.argsort(fpr_values)
    fpr_sorted = np.array(fpr_values)[sorted_idx]
    pro_sorted = np.array(pro_values)[sorted_idx]

    aupro = float(np.trapezoid(pro_sorted, fpr_sorted) / fpr_limit)
    return max(0.0, min(1.0, aupro))


def compute_pointing_game(masks: np.ndarray, attributions: np.ndarray) -> float:
    """
    Compute Pointing Game accuracy over a set of explanation attribution maps.

    Only images with a non-empty ground-truth mask are scored (there is
    nothing to "point at" for normal images).

    Args:
        masks: Ground-truth binary masks [N, H, W]
        attributions: Explanation attribution maps [N, H, W]

    Returns:
        Fraction of images where the highest-attribution pixel falls inside
        the defect mask. NaN if no image has a non-empty mask.
    """
    hits = [
        pointing_game(attributions[i], masks[i])
        for i in range(len(masks))
    ]
    hits = [h for h in hits if h is not None]
    return float(np.mean(hits)) if hits else float("nan")


def compute_topk_iou(
    masks: np.ndarray,
    attributions: np.ndarray,
    top_frac: float = 0.05,
) -> float:
    """
    Compute mean top-k% attribution IoU over a set of explanation maps.

    Args:
        masks: Ground-truth binary masks [N, H, W]
        attributions: Explanation attribution maps [N, H, W]
        top_frac: Fraction of highest-attribution pixels to threshold at

    Returns:
        Mean IoU between the top-`top_frac` attribution region and the
        ground-truth mask, over images with a non-empty mask. NaN if none.
    """
    ious = [
        topk_iou(attributions[i], masks[i], top_frac=top_frac)
        for i in range(len(masks))
    ]
    ious = [v for v in ious if v is not None]
    return float(np.mean(ious)) if ious else float("nan")


def compute_all_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
) -> dict[str, float]:
    """
    Compute all metrics for a method on a category.

    Args:
        labels: Image-level binary labels [N]
        scores: Image-level anomaly scores [N]
        masks: Ground-truth masks [N, H, W]
        anomaly_maps: Predicted anomaly maps [N, H, W]

    Returns:
        Dict with keys: image_auroc, pixel_auroc, pro
    """
    return {
        "image_auroc": compute_image_auroc(labels, scores),
        "pixel_auroc": compute_pixel_auroc(masks, anomaly_maps),
        "pro": compute_pro(masks, anomaly_maps),
    }


if __name__ == "__main__":
    # Quick test with random data
    np.random.seed(42)
    n = 50
    labels = np.array([0] * 25 + [1] * 25)
    scores = np.random.rand(n)
    scores[25:] += 0.5  # Anomalous images get higher scores

    masks = np.zeros((n, 64, 64))
    masks[25:, 20:40, 20:40] = 1  # Simple square defects

    anomaly_maps = np.random.rand(n, 64, 64) * 0.3
    anomaly_maps[25:, 15:45, 15:45] += 0.5  # Higher values near defects

    metrics = compute_all_metrics(labels, scores, masks, anomaly_maps)
    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
