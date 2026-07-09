"""
Explainability (XAI) methods for the two anomaly detection models.

Two different gradient/perturbation techniques are used because the two models
have different constraints:

  - Autoencoder: a plain nn.Module we fully control, so we can run true Grad-CAM
    (Selvaraju et al., 2017) on the encoder's final feature map, using the
    image-level anomaly score as the target signal.
  - PatchCore (via anomalib): the backbone feature extraction is wrapped in
    torch.no_grad() inside anomalib's own forward pass, so gradients from the
    anomaly score never reach the input pixels. We use occlusion sensitivity
    (Zeiler & Fergus, 2014) instead: a model-agnostic, gradient-free method that
    only needs forward passes, so it works regardless of that constraint.

Both methods produce a single-channel attribution map the same size as the
input image, which `pointing_game` and `topk_iou` can then score against a
ground-truth defect mask.
"""

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


def grad_cam_autoencoder(model, image: torch.Tensor) -> np.ndarray:
    """
    Grad-CAM for the convolutional autoencoder.

    Uses the encoder's final feature map (the latent representation) as the
    target layer, and the image-level anomaly score (top-k reconstruction
    error) as the scalar target for backpropagation.

    Args:
        model: A ConvAutoencoder instance (in eval mode).
        image: Input tensor [1, 3, H, W].

    Returns:
        Attribution map [H, W], normalized to [0, 1].
    """
    model.zero_grad()

    latent = model.encoder(image)
    latent.retain_grad()

    reconstruction = model.decoder(latent)
    anomaly_map = torch.mean((image - reconstruction) ** 2, dim=1, keepdim=True)
    score = model.get_image_score(anomaly_map)
    score.sum().backward()

    # Classic Grad-CAM global-average-pools the gradient over the whole feature
    # map to get a single per-channel importance weight, which is the right
    # move when explaining one global classification logit. Here the target
    # score is already spatially selective (top-k over per-pixel errors), so
    # pooling away the spatial gradient pattern would throw away exactly the
    # localization signal we want and leave only generic "high-contrast
    # region" channel importance. We keep the gradient at full spatial
    # resolution instead (gradient x activation, summed over channels).
    gradients = latent.grad  # [1, C, h, w]
    cam = F.relu((gradients * latent.detach()).sum(dim=1, keepdim=True))  # [1, 1, h, w]
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze(0).squeeze(0).cpu().numpy()

    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def occlusion_attribution(
    score_fn: Callable[[torch.Tensor], float],
    image: torch.Tensor,
    patch_size: int = 32,
    stride: int = 32,
    baseline_value: float | None = None,
) -> np.ndarray:
    """
    Occlusion sensitivity attribution.

    Slides a baseline-filled patch over the image and measures how much each
    occlusion drops the anomaly score. A large drop means that region was
    important for the model's anomaly decision. Requires no gradients, so it
    works for any model exposed only through a forward-pass scoring function
    (e.g. PatchCore via anomalib, where gradients to the input are blocked).

    Args:
        score_fn: Callable taking an image batch [1, 3, H, W] and returning a
            scalar anomaly score (float).
        image: Input tensor [1, 3, H, W].
        patch_size: Side length of the square occlusion patch.
        stride: Step size between patches (use stride == patch_size for a
            non-overlapping grid).
        baseline_value: Pixel value used to fill occluded regions. Defaults to
            the image's own mean pixel value (Zeiler & Fergus, 2014) rather
            than a fixed value like 0 (black) or 1 (white) — filling with a
            fixed extreme value creates a patch that is itself far
            out-of-distribution for models sensitive to it (e.g. PatchCore's
            memory bank, built from real texture statistics), which can make
            the occluded patch look "anomalous" regardless of what was
            originally there and corrupt the attribution.

    Returns:
        Attribution map [H, W] (non-negative; higher = more important).
    """
    with torch.no_grad():
        if baseline_value is None:
            baseline_value = float(image.mean())
        base_score = float(score_fn(image))
        _, _, h, w = image.shape
        attribution = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2, x2 = min(y + patch_size, h), min(x + patch_size, w)
                occluded = image.clone()
                occluded[:, :, y:y2, x:x2] = baseline_value
                occluded_score = float(score_fn(occluded))
                attribution[y:y2, x:x2] = max(base_score - occluded_score, 0.0)

    return attribution


def pointing_game(attribution: np.ndarray, mask: np.ndarray) -> bool | None:
    """
    Pointing Game (Zhang et al., 2016): does the highest-attribution pixel
    fall inside the ground-truth defect region?

    Returns None if the mask has no defect pixels (nothing to point at).
    """
    if mask.sum() == 0:
        return None
    y, x = np.unravel_index(np.argmax(attribution), attribution.shape)
    return bool(mask[y, x] > 0.5)


def topk_iou(attribution: np.ndarray, mask: np.ndarray, top_frac: float = 0.05) -> float | None:
    """
    IoU between the top-`top_frac` highest-attribution region and the
    ground-truth defect mask.

    Returns None if the mask has no defect pixels.
    """
    if mask.sum() == 0:
        return None
    threshold = np.percentile(attribution, 100 * (1 - top_frac))
    pred_region = attribution >= threshold
    gt_region = mask > 0.5
    intersection = np.logical_and(pred_region, gt_region).sum()
    union = np.logical_or(pred_region, gt_region).sum()
    return float(intersection / union) if union > 0 else 0.0
