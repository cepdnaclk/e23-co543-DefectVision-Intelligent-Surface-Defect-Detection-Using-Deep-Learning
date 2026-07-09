"""
Train PatchCore on MVTec AD using anomalib.

PatchCore builds a memory bank of patch-level features from a frozen
ImageNet-pretrained backbone. No gradient-based training needed.

Usage:
    python scripts/train_patchcore.py
    python scripts/train_patchcore.py --backbone resnet18
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import CATEGORIES


def train_patchcore_category(category, data_root, backbone="wide_resnet50_2",
                              coreset_ratio=0.1, device="auto", results_dir="results"):
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    from src.anomalib_compat import patch_split_enum_bug
    patch_split_enum_bug()

    print(f"\n{'='*60}")
    print(f"Training PatchCore - {category} (backbone={backbone})")
    print(f"{'='*60}")

    start_time = time.time()

    # num_workers=0: Windows multiprocessing DataLoader workers have been
    # observed to crash on torch DLL init in this environment; loading in the
    # main process avoids that entirely.
    datamodule = MVTecAD(root=data_root, category=category, num_workers=0)
    model = Patchcore(backbone=backbone, layers=["layer2", "layer3"],
                      coreset_sampling_ratio=coreset_ratio)

    ckpt_dir = Path(results_dir) / "checkpoints" / f"patchcore_{backbone}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    accelerator = device if device != "auto" else "auto"
    engine = Engine(max_epochs=1, accelerator=accelerator,
                    default_root_dir=str(ckpt_dir / category), devices=1)

    engine.fit(model=model, datamodule=datamodule)
    elapsed = time.time() - start_time
    print(f"  Memory bank built in {elapsed:.1f}s")

    test_results = engine.test(model=model, datamodule=datamodule)
    print(f"  Test results: {test_results}")

    return {"category": category, "backbone": backbone,
            "training_time": elapsed, "checkpoint_dir": str(ckpt_dir / category)}


def main():
    parser = argparse.ArgumentParser(description="Train PatchCore on MVTec AD")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2",
                        choices=["resnet18", "wide_resnet50_2"])
    parser.add_argument("--coreset_ratio", type=float, default=0.1)
    parser.add_argument("--data_root", type=str, default="data/mvtec_ad")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "gpu", "cpu"])
    args = parser.parse_args()

    data_root = args.data_root
    if not Path(data_root).is_absolute():
        data_root = str(PROJECT_ROOT / data_root)

    results_dir = str(PROJECT_ROOT / "results")
    results = []
    for category in args.categories:
        result = train_patchcore_category(category, data_root, args.backbone,
                                           args.coreset_ratio, args.device, results_dir)
        results.append(result)

    print(f"\n{'='*60}")
    print("PATCHCORE TRAINING SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['category']:12s} | backbone={r['backbone']} | time={r['training_time']:.1f}s")


if __name__ == "__main__":
    main()
