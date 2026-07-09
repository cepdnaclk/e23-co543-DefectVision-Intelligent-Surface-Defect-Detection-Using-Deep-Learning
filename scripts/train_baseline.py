"""
Train the convolutional autoencoder baseline on MVTec AD.

Trains a separate model per category on defect-free images only.
Uses MSE reconstruction loss with Adam optimizer and early stopping.

Usage:
    python scripts/train_baseline.py
    python scripts/train_baseline.py --categories bottle --epochs 50
    python scripts/train_baseline.py --device cpu
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder import ConvAutoencoder
from src.datasets import CATEGORIES, get_dataloaders


def train_one_epoch(
    model: ConvAutoencoder,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device)

        optimizer.zero_grad()
        reconstruction, _ = model(images)
        loss = criterion(reconstruction, images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: ConvAutoencoder,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        reconstruction, _ = model(images)
        loss = criterion(reconstruction, images)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_category(
    category: str,
    data_root: str,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 10,
    checkpoint_dir: str = "results/checkpoints",
) -> dict:
    """
    Train autoencoder for a single category.

    Returns dict with training metadata.
    """
    print(f"\n{'='*60}")
    print(f"Training Autoencoder - {category}")
    print(f"{'='*60}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        category, data_root, batch_size=batch_size, num_workers=0
    )
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples:   {len(val_loader.dataset)}")

    # Model
    model = ConvAutoencoder().to(device)
    print(f"  Parameters:    {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    # Early stopping
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = 0

    # Checkpoint directory
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"autoencoder_{category}.pth"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        # Progress
        lr_current = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {lr_current:.2e}",
            end=""
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "category": category,
            }, ckpt_path)
            print(" <- saved", end="")
        else:
            epochs_no_improve += 1

        print()

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - start_time
    print(f"  Best epoch: {best_epoch} (val_loss={best_val_loss:.6f})")
    print(f"  Training time: {elapsed:.1f}s")
    print(f"  Checkpoint: {ckpt_path}")

    return {
        "category": category,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_time": elapsed,
        "checkpoint": str(ckpt_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train autoencoder baseline on MVTec AD")
    parser.add_argument(
        "--categories", nargs="+", default=CATEGORIES,
        help=f"Categories to train on (default: {CATEGORIES})"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--data_root", type=str, default="data/mvtec_ad", help="Dataset root path")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Resolve data root relative to project root
    data_root = args.data_root
    if not Path(data_root).is_absolute():
        data_root = str(PROJECT_ROOT / data_root)

    # Train each category
    results = []
    for category in args.categories:
        result = train_category(
            category=category,
            data_root=data_root,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            checkpoint_dir=str(PROJECT_ROOT / "results" / "checkpoints"),
        )
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['category']:12s} | "
              f"epoch {r['best_epoch']:3d} | "
              f"val_loss {r['best_val_loss']:.6f} | "
              f"time {r['training_time']:.1f}s")


if __name__ == "__main__":
    main()
