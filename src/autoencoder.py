"""
Convolutional Autoencoder for unsupervised anomaly detection.

Architecture: Simple encoder-decoder with 4 conv/deconv blocks.
Input: 256x256 RGB images. Latent: 16x16x256.
Parameters: ~2.5M (small, this is a baseline).

Anomaly detection approach:
  - Train on defect-free images only (reconstruction task)
  - At test time, anomaly score = pixel-wise reconstruction error
  - Defective regions produce higher reconstruction error
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encodes 256x256x3 → 16x16x256."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Block 1: 256x256x3 → 128x128x32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Block 2: 128x128x32 → 64x64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Block 3: 64x64x64 → 32x32x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Block 4: 32x32x128 → 16x16x256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Decoder(nn.Module):
    """Decodes 16x16x256 → 256x256x3."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Block 1: 16x16x256 → 32x32x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Block 2: 32x32x128 → 64x64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Block 3: 64x64x64 → 128x128x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Block 4: 128x128x32 → 256x256x3
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output in [0, 1] range to match normalized input
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for anomaly detection baseline.

    Trained on defect-free images to reconstruct normal appearances.
    Anomalies produce higher reconstruction error since the model
    has never seen defective patterns during training.
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image tensor [B, 3, 256, 256], values in [0, 1].

        Returns:
            reconstruction: Reconstructed image [B, 3, 256, 256]
            latent: Latent representation [B, 256, 16, 16]
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def get_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pixel-wise anomaly map as mean squared error per pixel.

        Args:
            x: Input image tensor [B, 3, 256, 256]

        Returns:
            anomaly_map: Per-pixel anomaly score [B, 1, 256, 256]
        """
        reconstruction, _ = self.forward(x)
        # MSE across channels → single-channel anomaly map
        anomaly_map = torch.mean((x - reconstruction) ** 2, dim=1, keepdim=True)
        return anomaly_map

    def get_image_score(self, anomaly_map: torch.Tensor, top_k: int = 100) -> torch.Tensor:
        """
        Compute image-level anomaly score as mean of top-k pixel errors.

        Using top-k mean instead of max for robustness to noise.
        Literature standard: top-k with k~100 works well for 256x256.

        Args:
            anomaly_map: Per-pixel anomaly scores [B, 1, H, W]
            top_k: Number of highest-error pixels to average

        Returns:
            scores: Image-level anomaly scores [B]
        """
        batch_size = anomaly_map.shape[0]
        flat = anomaly_map.view(batch_size, -1)
        topk_vals, _ = torch.topk(flat, k=min(top_k, flat.shape[1]), dim=1)
        return topk_vals.mean(dim=1)

    @staticmethod
    def count_parameters() -> str:
        """Return a human-readable parameter count."""
        model = ConvAutoencoder()
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return f"Total: {total:,} | Trainable: {trainable:,}"


if __name__ == "__main__":
    # Quick sanity check
    model = ConvAutoencoder()
    print(f"Model parameters: {ConvAutoencoder.count_parameters()}")

    x = torch.randn(2, 3, 256, 256)
    recon, latent = model(x)
    print(f"Input shape:          {x.shape}")
    print(f"Latent shape:         {latent.shape}")
    print(f"Reconstruction shape: {recon.shape}")

    amap = model.get_anomaly_map(x)
    scores = model.get_image_score(amap)
    print(f"Anomaly map shape:    {amap.shape}")
    print(f"Image scores shape:   {scores.shape}")
