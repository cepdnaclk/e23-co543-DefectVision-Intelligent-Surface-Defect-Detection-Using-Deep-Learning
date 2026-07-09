"""
Evaluate both methods (autoencoder + PatchCore) on all categories.

Computes Image-AUROC, Pixel-AUROC, and PRO (AUPRO) for each combination.
Writes results to results/metrics.csv.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder import ConvAutoencoder
from src.datasets import CATEGORIES, MVTecDataset, get_transforms, get_mask_transform
from src.metrics import compute_all_metrics
from src.results_io import save_or_merge_csv


def evaluate_autoencoder(category, data_root, device, checkpoint_dir):
    """Evaluate trained autoencoder on a category's test set."""
    print(f"\n  [Autoencoder] Evaluating on {category}...")

    ckpt_path = Path(checkpoint_dir) / f"autoencoder_{category}.pth"
    if not ckpt_path.exists():
        print(f"    Checkpoint not found: {ckpt_path}")
        return None

    model = ConvAutoencoder().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_dataset = MVTecDataset(data_root, category, split="test")
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=16,
                                          shuffle=False, num_workers=0)

    all_labels, all_scores = [], []
    all_masks, all_maps = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"    {category}", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()
            labels = np.array(batch["label"])

            anomaly_map = model.get_anomaly_map(images)
            scores = model.get_image_score(anomaly_map)

            amap_np = anomaly_map.squeeze(1).cpu().numpy()
            scores_np = scores.cpu().numpy()

            all_labels.append(labels)
            all_scores.append(scores_np)
            all_masks.append(masks.squeeze(1) if masks.ndim == 4 else masks)
            all_maps.append(amap_np)

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    all_masks = np.concatenate(all_masks)
    all_maps = np.concatenate(all_maps)

    metrics = compute_all_metrics(all_labels, all_scores, all_masks, all_maps)
    print(f"    Image AUROC: {metrics['image_auroc']:.4f}")
    print(f"    Pixel AUROC: {metrics['pixel_auroc']:.4f}")
    print(f"    PRO:         {metrics['pro']:.4f}")
    return metrics


def evaluate_patchcore(category, data_root, backbone="wide_resnet50_2", device_str="auto"):
    """Evaluate PatchCore on a category using anomalib."""
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    from src.anomalib_compat import patch_split_enum_bug
    patch_split_enum_bug()

    print(f"\n  [PatchCore/{backbone}] Evaluating on {category}...")

    # num_workers=0: Windows multiprocessing DataLoader workers have been
    # observed to crash on torch DLL init in this environment; loading in the
    # main process avoids that entirely.
    datamodule = MVTecAD(root=data_root, category=category, num_workers=0)
    model = Patchcore(backbone=backbone, layers=["layer2", "layer3"],
                      coreset_sampling_ratio=0.1)

    ckpt_base = PROJECT_ROOT / "results" / "checkpoints" / f"patchcore_{backbone}" / category
    ckpt_files = list(ckpt_base.rglob("*.ckpt"))
    if not ckpt_files:
        print(f"    No checkpoint found in {ckpt_base}")
        print(f"    Training PatchCore on the fly...")
        accelerator = device_str if device_str != "auto" else "auto"
        engine = Engine(max_epochs=1, accelerator=accelerator,
                        default_root_dir=str(ckpt_base), devices=1)
        engine.fit(model=model, datamodule=datamodule)
        ckpt_files = list(ckpt_base.rglob("*.ckpt"))

    if ckpt_files:
        ckpt_path = str(ckpt_files[0])
        print(f"    Using checkpoint: {ckpt_path}")
    else:
        ckpt_path = None

    accelerator = device_str if device_str != "auto" else "auto"
    engine = Engine(max_epochs=1, accelerator=accelerator,
                    default_root_dir=str(ckpt_base), devices=1)

    # Get predictions
    datamodule.setup()
    test_dl = datamodule.test_dataloader()

    # Collect predictions
    all_labels, all_scores = [], []
    all_masks, all_maps = [], []

    predictions = engine.predict(model=model, datamodule=datamodule,
                                  ckpt_path=ckpt_path)

    if predictions is not None:
        for pred in predictions:
            if hasattr(pred, 'pred_score') and pred.pred_score is not None:
                scores = pred.pred_score.cpu().numpy()
                all_scores.append(scores.flatten())
            if hasattr(pred, 'anomaly_map') and pred.anomaly_map is not None:
                amap = pred.anomaly_map.cpu().numpy()
                if amap.ndim == 4:
                    amap = amap.squeeze(1)
                all_maps.append(amap)
            if hasattr(pred, 'gt_label') and pred.gt_label is not None:
                labels = pred.gt_label.cpu().numpy()
                all_labels.append(labels.flatten())
            if hasattr(pred, 'gt_mask') and pred.gt_mask is not None:
                masks = pred.gt_mask.cpu().numpy()
                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                all_masks.append(masks)

    if not all_labels or not all_scores:
        print("    WARNING: Could not extract predictions, using anomalib test metrics")
        test_results = engine.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        if test_results:
            r = test_results[0]
            return {
                "image_auroc": r.get("image_AUROC", r.get("test/image_AUROC", float("nan"))),
                "pixel_auroc": r.get("pixel_AUROC", r.get("test/pixel_AUROC", float("nan"))),
                "pro": float("nan"),
            }
        return None

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    all_masks = np.concatenate(all_masks)
    all_maps = np.concatenate(all_maps)

    metrics = compute_all_metrics(all_labels, all_scores, all_masks, all_maps)
    print(f"    Image AUROC: {metrics['image_auroc']:.4f}")
    print(f"    Pixel AUROC: {metrics['pixel_auroc']:.4f}")
    print(f"    PRO:         {metrics['pro']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate all methods on MVTec AD")
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
    print(f"Device: {device}")

    checkpoint_dir = str(PROJECT_ROOT / "results" / "checkpoints")
    rows = []

    for category in args.categories:
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}")

        # Autoencoder
        ae_metrics = evaluate_autoencoder(category, data_root, device, checkpoint_dir)
        if ae_metrics:
            rows.append({"method": "autoencoder", "category": category, **ae_metrics})

        # PatchCore
        pc_metrics = evaluate_patchcore(category, data_root, args.backbone, args.device)
        if pc_metrics:
            rows.append({"method": "patchcore", "category": category, **pc_metrics})

    # Save results (merged with any existing rows from prior runs, e.g. when
    # categories are run one at a time as separate processes)
    results_dir = PROJECT_ROOT / "results"
    csv_path = results_dir / "metrics.csv"

    df = save_or_merge_csv(rows, csv_path, key_cols=["method", "category"])
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
