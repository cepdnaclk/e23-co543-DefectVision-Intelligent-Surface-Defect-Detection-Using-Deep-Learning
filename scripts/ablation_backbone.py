"""
Ablation study: PatchCore backbone comparison (ResNet18 vs WideResNet50).

Reports metric deltas and inference time differences per category.
Saves results to results/ablation.csv.

Usage:
    python scripts/ablation_backbone.py
    python scripts/ablation_backbone.py --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import CATEGORIES
from src.results_io import save_or_merge_csv


def run_patchcore_eval(category, data_root, backbone, device_str):
    """Train (if needed) and evaluate PatchCore, measuring inference time."""
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    from src.anomalib_compat import patch_split_enum_bug
    from src.metrics import compute_all_metrics

    patch_split_enum_bug()

    # num_workers=0: Windows multiprocessing DataLoader workers have been
    # observed to crash on torch DLL init in this environment; loading in the
    # main process avoids that entirely.
    datamodule = MVTecAD(root=data_root, category=category, num_workers=0)
    model = Patchcore(backbone=backbone, layers=["layer2", "layer3"],
                      coreset_sampling_ratio=0.1)

    ckpt_base = PROJECT_ROOT / "results" / "checkpoints" / f"patchcore_{backbone}" / category
    ckpt_base.mkdir(parents=True, exist_ok=True)
    ckpt_files = list(ckpt_base.rglob("*.ckpt"))

    accelerator = device_str if device_str != "auto" else "auto"
    engine = Engine(max_epochs=1, accelerator=accelerator,
                    default_root_dir=str(ckpt_base), devices=1)

    if not ckpt_files:
        print(f"    Training {backbone} on {category}...")
        engine.fit(model=model, datamodule=datamodule)
        ckpt_files = list(ckpt_base.rglob("*.ckpt"))

    ckpt_path = str(ckpt_files[0]) if ckpt_files else None

    # Predict and measure time
    start = time.time()
    predictions = engine.predict(model=model, datamodule=datamodule,
                                  ckpt_path=ckpt_path)
    total_time = time.time() - start

    all_labels, all_scores, all_masks, all_maps = [], [], [], []
    n_images = 0

    if predictions is not None:
        for pred in predictions:
            bs = pred.image.shape[0] if hasattr(pred, 'image') else 1
            n_images += bs
            if hasattr(pred, 'pred_score') and pred.pred_score is not None:
                all_scores.append(pred.pred_score.cpu().numpy().flatten())
            if hasattr(pred, 'anomaly_map') and pred.anomaly_map is not None:
                amap = pred.anomaly_map.cpu().numpy()
                if amap.ndim == 4:
                    amap = amap.squeeze(1)
                all_maps.append(amap)
            if hasattr(pred, 'gt_label') and pred.gt_label is not None:
                all_labels.append(pred.gt_label.cpu().numpy().flatten())
            if hasattr(pred, 'gt_mask') and pred.gt_mask is not None:
                m = pred.gt_mask.cpu().numpy()
                if m.ndim == 4:
                    m = m.squeeze(1)
                all_masks.append(m)

    avg_time_ms = (total_time / max(n_images, 1)) * 1000

    if all_labels and all_scores and all_masks and all_maps:
        labels = np.concatenate(all_labels)
        scores = np.concatenate(all_scores)
        masks = np.concatenate(all_masks)
        maps_ = np.concatenate(all_maps)
        metrics = compute_all_metrics(labels, scores, masks, maps_)
    else:
        # Fallback: use anomalib's test
        test_results = engine.test(model=model, datamodule=datamodule,
                                    ckpt_path=ckpt_path)
        if test_results:
            r = test_results[0]
            metrics = {
                "image_auroc": r.get("image_AUROC", float("nan")),
                "pixel_auroc": r.get("pixel_AUROC", float("nan")),
                "pro": float("nan"),
            }
        else:
            metrics = {"image_auroc": float("nan"), "pixel_auroc": float("nan"),
                      "pro": float("nan")}

    return {**metrics, "avg_inference_time_ms": avg_time_ms}


def main():
    parser = argparse.ArgumentParser(description="Ablation: ResNet18 vs WideResNet50")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--data_root", type=str, default="data/mvtec_ad")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    data_root = args.data_root
    if not Path(data_root).is_absolute():
        data_root = str(PROJECT_ROOT / data_root)

    backbones = ["resnet18", "wide_resnet50_2"]
    rows = []

    for category in args.categories:
        for backbone in backbones:
            print(f"\n{'='*60}")
            print(f"Ablation: {category} / {backbone}")
            print(f"{'='*60}")
            result = run_patchcore_eval(category, data_root, backbone, args.device)
            rows.append({"category": category, "backbone": backbone, **result})
            print(f"  Image AUROC: {result['image_auroc']:.4f}")
            print(f"  Pixel AUROC: {result['pixel_auroc']:.4f}")
            print(f"  PRO:         {result['pro']:.4f}")
            print(f"  Avg time:    {result['avg_inference_time_ms']:.1f} ms/image")

    csv_path = PROJECT_ROOT / "results" / "ablation.csv"
    df = save_or_merge_csv(rows, csv_path, key_cols=["category", "backbone"])

    # Print deltas
    print(f"\n{'='*60}")
    print("ABLATION RESULTS")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    print(f"\nDeltas (WideResNet50 - ResNet18):")
    for category in args.categories:
        r18 = df[(df["category"] == category) & (df["backbone"] == "resnet18")]
        wrn = df[(df["category"] == category) & (df["backbone"] == "wide_resnet50_2")]
        if not r18.empty and not wrn.empty:
            r18, wrn = r18.iloc[0], wrn.iloc[0]
            print(f"  {category}:")
            print(f"    Image AUROC: {wrn['image_auroc'] - r18['image_auroc']:+.4f}")
            print(f"    Pixel AUROC: {wrn['pixel_auroc'] - r18['pixel_auroc']:+.4f}")
            print(f"    PRO:         {wrn['pro'] - r18['pro']:+.4f}")
            print(f"    Time delta:  {wrn['avg_inference_time_ms'] - r18['avg_inference_time_ms']:+.1f} ms")

    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
