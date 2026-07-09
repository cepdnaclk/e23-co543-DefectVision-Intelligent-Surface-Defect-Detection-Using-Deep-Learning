"""
Generate qualitative output grids for each (method, category).

For each combination, produces a grid showing:
  - 2 correct detections (true positives with good localization)
  - 1 failure case (false negative or false positive)

Each row: [Original Image | Ground Truth Mask | Predicted Anomaly Heatmap]

Usage:
    python scripts/make_qualitative.py
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder import ConvAutoencoder
from src.datasets import CATEGORIES, IMAGENET_MEAN, IMAGENET_STD, MVTecDataset


def get_autoencoder_predictions(category, data_root, device, checkpoint_dir):
    """Get anomaly maps and scores for autoencoder on test set."""
    ckpt_path = Path(checkpoint_dir) / f"autoencoder_{category}.pth"
    if not ckpt_path.exists():
        print(f"  Checkpoint not found: {ckpt_path}")
        return None

    model = ConvAutoencoder().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_dataset = MVTecDataset(data_root, category, split="test")
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                          shuffle=False, num_workers=0)

    results = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            anomaly_map = model.get_anomaly_map(images)
            score = model.get_image_score(anomaly_map).item()

            img_np = images[0].cpu().permute(1, 2, 0).numpy()
            mask_np = batch["mask"][0, 0].numpy()
            amap_np = anomaly_map[0, 0].cpu().numpy()

            results.append({
                "image": img_np,
                "mask": mask_np,
                "anomaly_map": amap_np,
                "score": score,
                "label": batch["label"][0].item(),
                "defect_type": batch["defect_type"][0],
                "image_path": batch["image_path"][0],
            })
    return results


def get_patchcore_predictions(category, data_root, backbone, device_str):
    """Get anomaly maps and scores for PatchCore on test set."""
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    from src.anomalib_compat import patch_split_enum_bug
    patch_split_enum_bug()

    # num_workers=0: Windows multiprocessing DataLoader workers have been
    # observed to crash on torch DLL init in this environment; loading in the
    # main process avoids that entirely.
    datamodule = MVTecAD(root=data_root, category=category, num_workers=0)
    model = Patchcore(backbone=backbone, layers=["layer2", "layer3"],
                      coreset_sampling_ratio=0.1)

    ckpt_base = PROJECT_ROOT / "results" / "checkpoints" / f"patchcore_{backbone}" / category
    ckpt_files = list(ckpt_base.rglob("*.ckpt"))
    ckpt_path = str(ckpt_files[0]) if ckpt_files else None

    if ckpt_path is None:
        print(f"  No checkpoint found, training on the fly...")
        accelerator = device_str if device_str != "auto" else "auto"
        engine = Engine(max_epochs=1, accelerator=accelerator,
                        default_root_dir=str(ckpt_base), devices=1)
        engine.fit(model=model, datamodule=datamodule)
        ckpt_files = list(ckpt_base.rglob("*.ckpt"))
        ckpt_path = str(ckpt_files[0]) if ckpt_files else None

    accelerator = device_str if device_str != "auto" else "auto"
    engine = Engine(max_epochs=1, accelerator=accelerator,
                    default_root_dir=str(ckpt_base), devices=1)

    predictions = engine.predict(model=model, datamodule=datamodule,
                                  ckpt_path=ckpt_path)

    results = []
    if predictions is not None:
        for pred in predictions:
            batch_size = pred.image.shape[0] if hasattr(pred, 'image') else 1
            for i in range(batch_size):
                entry = {}
                if hasattr(pred, 'image') and pred.image is not None:
                    img = pred.image[i].cpu()
                    if img.shape[0] == 3:
                        img = img.permute(1, 2, 0)
                    # anomalib's pre_processor normalizes with ImageNet
                    # mean/std before the model sees it; pred.image is that
                    # normalized tensor, not the raw [0, 1] image, so it must
                    # be un-normalized before display (otherwise almost every
                    # pixel clips to black).
                    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
                    std = np.array(IMAGENET_STD).reshape(1, 1, 3)
                    # Clip to [0, 1]: un-normalized values can drift a hair
                    # above 1.0 from floating-point rounding, which would
                    # trip create_grid's "img.max() <= 1.0" branch check into
                    # treating this as a [0, 255]-range image and dividing by
                    # 255 — crushing it to near-black.
                    entry["image"] = np.clip(img.numpy() * std + mean, 0, 1)
                if hasattr(pred, 'anomaly_map') and pred.anomaly_map is not None:
                    amap = pred.anomaly_map[i].cpu().numpy()
                    if amap.ndim == 3:
                        amap = amap.squeeze(0)
                    entry["anomaly_map"] = amap
                if hasattr(pred, 'pred_score') and pred.pred_score is not None:
                    s = pred.pred_score[i] if pred.pred_score.ndim > 0 else pred.pred_score
                    entry["score"] = float(s.cpu())
                if hasattr(pred, 'gt_mask') and pred.gt_mask is not None:
                    m = pred.gt_mask[i].cpu().numpy()
                    if m.ndim == 3:
                        m = m.squeeze(0)
                    entry["mask"] = m
                else:
                    entry["mask"] = np.zeros_like(entry.get("anomaly_map", np.zeros((256, 256))))
                if hasattr(pred, 'gt_label') and pred.gt_label is not None:
                    entry["label"] = int(pred.gt_label[i].cpu())
                else:
                    entry["label"] = 0
                entry["defect_type"] = "unknown"
                if "image" in entry and "anomaly_map" in entry:
                    results.append(entry)
    return results


def select_samples(results):
    """Select 2 correct detections + 1 failure case."""
    anomalous = [r for r in results if r["label"] == 1]
    normal = [r for r in results if r["label"] == 0]

    selected = []

    # 2 correct detections: anomalous images with highest scores
    if anomalous:
        anomalous_sorted = sorted(anomalous, key=lambda x: x["score"], reverse=True)
        selected.extend(anomalous_sorted[:2])

    # 1 failure case: anomalous image with lowest score (false negative)
    # or normal image with highest score (false positive)
    if anomalous:
        worst_fn = sorted(anomalous, key=lambda x: x["score"])[0]
        selected.append(worst_fn)
    elif normal:
        worst_fp = sorted(normal, key=lambda x: x["score"], reverse=True)[0]
        selected.append(worst_fp)

    # Pad if we don't have enough
    while len(selected) < 3 and results:
        selected.append(results[len(selected) % len(results)])

    return selected[:3]


def create_grid(samples, method, category, output_path):
    """Create a 3-row grid: each row = [Original | GT Mask | Heatmap]."""
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    row_labels = ["Correct Detection 1", "Correct Detection 2", "Failure Case"]

    for i, sample in enumerate(samples):
        img = sample["image"]
        mask = sample["mask"]
        amap = sample["anomaly_map"]

        # Normalize image for display if needed
        if img.max() <= 1.0:
            img_display = np.clip(img, 0, 1)
        else:
            img_display = np.clip(img / 255.0, 0, 1)

        # Original image
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title(f"{row_labels[i]}\n(score={sample['score']:.4f})")
        axes[i, 0].axis("off")

        # Ground truth mask
        axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        # Anomaly heatmap overlaid on image
        axes[i, 2].imshow(img_display)
        amap_normalized = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        axes[i, 2].imshow(amap_normalized, cmap="inferno", alpha=0.5)
        axes[i, 2].set_title("Anomaly Heatmap")
        axes[i, 2].axis("off")

    fig.suptitle(f"{method.upper()} - {category}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative output grids")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--data_root", type=str, default="data/mvtec_ad")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    args = parser.parse_args()

    data_root = args.data_root
    if not Path(data_root).is_absolute():
        data_root = str(PROJECT_ROOT / data_root)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = PROJECT_ROOT / "results" / "qualitative"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = str(PROJECT_ROOT / "results" / "checkpoints")

    for category in args.categories:
        print(f"\n{'='*60}")
        print(f"Qualitative: {category}")
        print(f"{'='*60}")

        # Autoencoder
        print("  Autoencoder predictions...")
        ae_results = get_autoencoder_predictions(category, data_root, device, checkpoint_dir)
        if ae_results:
            samples = select_samples(ae_results)
            create_grid(samples, "autoencoder", category,
                       output_dir / f"autoencoder_{category}.png")

        # PatchCore
        print("  PatchCore predictions...")
        pc_results = get_patchcore_predictions(category, data_root, args.backbone, args.device)
        if pc_results:
            samples = select_samples(pc_results)
            create_grid(samples, "patchcore", category,
                       output_dir / f"patchcore_{category}.png")


if __name__ == "__main__":
    main()
