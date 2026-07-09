"""
Explainability (XAI) analysis for both methods (autoencoder + PatchCore).

For each (method, category):
  - Computes a gradient/perturbation-based attribution map for a capped subset
    of anomalous test images (Grad-CAM for the autoencoder, occlusion
    sensitivity for PatchCore — see src/explainability.py for why they differ).
  - Scores those attribution maps against the ground-truth defect masks using
    Pointing Game accuracy and top-5% attribution IoU.
  - Saves the aggregated scores to results/explainability.csv.
  - Saves a few 4-panel qualitative grids per (method, category) to
    results/explainability/: [Original | GT Mask | Anomaly Heatmap | Explanation].

Usage:
    python scripts/explain.py
    python scripts/explain.py --categories bottle --device cpu --max_samples 10
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder import ConvAutoencoder
from src.datasets import CATEGORIES, MVTecDataset
from src.explainability import grad_cam_autoencoder, occlusion_attribution
from src.metrics import compute_pointing_game, compute_topk_iou
from src.results_io import save_or_merge_csv


def select_anomalous_indices(dataset, max_samples):
    """Indices of defective (label==1) test samples, capped for runtime."""
    indices = [i for i, s in enumerate(dataset.samples) if s["label"] == 1]
    return indices[:max_samples]


def load_autoencoder_model(category, device, checkpoint_dir):
    ckpt_path = Path(checkpoint_dir) / f"autoencoder_{category}.pth"
    if not ckpt_path.exists():
        print(f"    Checkpoint not found: {ckpt_path}")
        return None
    model = ConvAutoencoder().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_patchcore_model(category, backbone, device, checkpoint_dir):
    from anomalib.models import Patchcore

    ckpt_base = Path(checkpoint_dir) / f"patchcore_{backbone}" / category
    ckpt_files = list(ckpt_base.rglob("*.ckpt"))
    if not ckpt_files:
        print(f"    No checkpoint found in {ckpt_base}")
        return None
    ckpt_path = str(ckpt_files[0])

    try:
        model = Patchcore.load_from_checkpoint(ckpt_path, map_location=device)
    except Exception as e:
        print(f"    load_from_checkpoint failed ({e}); falling back to manual state_dict load")
        model = Patchcore(backbone=backbone, layers=["layer2", "layer3"], coreset_sampling_ratio=0.1)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

    model.eval()
    model.to(device)
    return model


def explain_autoencoder(category, data_root, device, checkpoint_dir, max_samples):
    """Grad-CAM attribution + anomaly maps for a capped set of anomalous test images."""
    model = load_autoencoder_model(category, device, checkpoint_dir)
    if model is None:
        return None

    dataset = MVTecDataset(data_root, category, split="test")
    indices = select_anomalous_indices(dataset, max_samples)

    entries = []
    for idx in indices:
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"][0].numpy()

        with torch.no_grad():
            anomaly_map = model.get_anomaly_map(image)
            score = model.get_image_score(anomaly_map).item()
        cam = grad_cam_autoencoder(model, image)

        entries.append({
            "image": image[0].detach().cpu().permute(1, 2, 0).numpy(),
            "mask": mask,
            "anomaly_map": anomaly_map[0, 0].cpu().numpy(),
            "attribution": cam,
            "score": score,
            "defect_type": sample["defect_type"],
        })
    return entries


def explain_patchcore(category, data_root, backbone, device, checkpoint_dir, max_samples, patch_size, stride):
    """Occlusion-sensitivity attribution + anomaly maps for anomalous test images."""
    model = load_patchcore_model(category, backbone, device, checkpoint_dir)
    if model is None:
        return None

    def score_fn(image_batch):
        # Occlusion attribution targets the *sum* of the spatial anomaly map,
        # not pred_score or the map's max. pred_score is a single
        # argmax-over-patches distance (see PatchcoreModel.compute_anomaly_score)
        # that's easily hijacked by the occlusion patch itself looking
        # synthetic, and the map's max is often dominated by a border/edge
        # artifact (this project's own README already notes both methods see
        # false positives near image borders) rather than the true defect.
        # Summing over the whole map means occluding the (larger) true defect
        # region reduces the target by more than occluding a small border
        # blip, giving a much more robust localization signal to attribute.
        with torch.no_grad():
            output = model(image_batch)
        return output.anomaly_map.sum()

    dataset = MVTecDataset(data_root, category, split="test")
    indices = select_anomalous_indices(dataset, max_samples)

    entries = []
    for idx in indices:
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"][0].numpy()

        with torch.no_grad():
            output = model(image)
            score = float(output.pred_score)
            amap = output.anomaly_map[0, 0].cpu().numpy()

        attribution = occlusion_attribution(score_fn, image, patch_size=patch_size, stride=stride)

        entries.append({
            "image": image[0].detach().cpu().permute(1, 2, 0).numpy(),
            "mask": mask,
            "anomaly_map": amap,
            "attribution": attribution,
            "score": score,
            "defect_type": sample["defect_type"],
        })
    return entries


def aggregate_metrics(entries):
    masks = np.stack([e["mask"] for e in entries])
    attributions = np.stack([e["attribution"] for e in entries])
    return {
        "pointing_game_acc": compute_pointing_game(masks, attributions),
        "topk_iou": compute_topk_iou(masks, attributions, top_frac=0.05),
    }


def _percentile_normalize(arr, low=50, high=99.5):
    """
    Contrast-stretch a map for display using percentiles rather than true
    min/max. When most of the attribution mass sits in a handful of patches
    (common for occlusion sensitivity), a plain min-max normalization renders
    almost everything as background and hides the real signal.
    """
    lo, hi = np.percentile(arr, [low, high])
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def save_grid(entries, method, category, explanation_label, output_path, n_examples=2):
    samples = entries[:n_examples]
    if not samples:
        return
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, sample in enumerate(samples):
        img = np.clip(sample["image"], 0, 1)
        mask = sample["mask"]
        amap = sample["anomaly_map"]
        attr = sample["attribution"]

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original ({sample['defect_type']})\nscore={sample['score']:.4f}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img)
        axes[i, 2].imshow(_percentile_normalize(amap), cmap="inferno", alpha=0.5)
        axes[i, 2].set_title("Anomaly Heatmap")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(img)
        axes[i, 3].imshow(_percentile_normalize(attr), cmap="jet", alpha=0.5)
        axes[i, 3].set_title(explanation_label)
        axes[i, 3].axis("off")

    fig.suptitle(f"{method.upper()} - {category}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Explainability analysis for both methods")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--data_root", type=str, default="data/mvtec_ad")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--max_samples", type=int, default=20,
                        help="Max anomalous test images per category used for XAI evaluation")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Occlusion patch size (PatchCore)")
    parser.add_argument("--stride", type=int, default=16,
                        help="Occlusion stride (PatchCore)")
    args = parser.parse_args()

    data_root = args.data_root
    if not Path(data_root).is_absolute():
        data_root = str(PROJECT_ROOT / data_root)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    checkpoint_dir = str(PROJECT_ROOT / "results" / "checkpoints")
    output_dir = PROJECT_ROOT / "results" / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for category in args.categories:
        print(f"\n{'='*60}")
        print(f"Explainability: {category}")
        print(f"{'='*60}")

        print("  [Autoencoder] Grad-CAM...")
        ae_entries = explain_autoencoder(category, data_root, device, checkpoint_dir, args.max_samples)
        if ae_entries:
            metrics = aggregate_metrics(ae_entries)
            rows.append({"method": "autoencoder", "category": category, "n_samples": len(ae_entries), **metrics})
            print(f"    Pointing Game Acc: {metrics['pointing_game_acc']:.4f}")
            print(f"    Top-5% IoU:        {metrics['topk_iou']:.4f}")
            save_grid(ae_entries, "autoencoder", category, "Grad-CAM Explanation",
                     output_dir / f"autoencoder_{category}.png")

        print("  [PatchCore] Occlusion sensitivity...")
        pc_entries = explain_patchcore(category, data_root, args.backbone, device, checkpoint_dir,
                                       args.max_samples, args.patch_size, args.stride)
        if pc_entries:
            metrics = aggregate_metrics(pc_entries)
            rows.append({"method": "patchcore", "category": category, "n_samples": len(pc_entries), **metrics})
            print(f"    Pointing Game Acc: {metrics['pointing_game_acc']:.4f}")
            print(f"    Top-5% IoU:        {metrics['topk_iou']:.4f}")
            save_grid(pc_entries, "patchcore", category, "Occlusion Explanation",
                     output_dir / f"patchcore_{category}.png")

    csv_path = PROJECT_ROOT / "results" / "explainability.csv"
    df = save_or_merge_csv(rows, csv_path, key_cols=["method", "category"])
    print(f"\n{'='*60}")
    print("EXPLAINABILITY RESULTS")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
