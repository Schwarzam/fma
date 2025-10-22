"""
Step 5: Train Codecs

Train each codec independently on reconstruction task.

Goal: Learn to compress and reconstruct each modality type.

Training objective:
    Loss = Reconstruction Loss + Commitment Loss (for VQ)

For images: Also use perceptual loss for better quality.

This step shows:
1. How to create datasets for each modality
2. Training loop for each codec type
3. Evaluation and checkpointing
4. Monitoring codebook usage (for VQ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from typing import Optional
from tqdm import tqdm

# Import from previous steps
from step1_define_modalities import MyImage, MyTimeSeries, MyScalar, MyTabular
from step3_implement_codecs import ImageCodec, TimeSeriesCodec, ScalarCodec, TabularCodec


# ============================================================================
# Dataset Classes
# ============================================================================

class ImageDataset(Dataset):
    """
    Simple image dataset for codec training.

    In practice, you would load from disk or use existing datasets.
    This generates random images for demonstration.
    """
    def __init__(self, num_samples: int = 1000, image_size: int = 224):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random image (in practice, load real images)
        # Return 3D tensor - DataLoader will add batch dimension
        pixels = torch.randn(3, self.image_size, self.image_size)
        return pixels


class TimeSeriesDataset(Dataset):
    """
    Time series dataset for codec training.

    Generates synthetic sine waves with noise.
    """
    def __init__(self, num_samples: int = 1000, seq_length: int = 1000):
        self.num_samples = num_samples
        self.seq_length = seq_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic time series (sine wave + noise)
        t = torch.linspace(0, 10, self.seq_length)
        freq = torch.rand(1) * 5 + 1  # Random frequency 1-6 Hz
        values = torch.sin(2 * torch.pi * freq * t) + torch.randn(self.seq_length) * 0.1

        return {
            'values': values,  # [seq_length]
            'timestamps': t,
            'mask': torch.ones(self.seq_length, dtype=torch.bool)
        }


class ScalarDataset(Dataset):
    """
    Scalar dataset for codec training.

    Generates random scalars in range [0, 100].
    """
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        value = torch.rand(1) * 100.0  # Random value 0-100
        return value


class TabularDataset(Dataset):
    """
    Tabular dataset for codec training.

    Generates random feature vectors.
    """
    def __init__(self, num_samples: int = 1000, num_features: int = 32):
        self.num_samples = num_samples
        self.num_features = num_features

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        features = torch.randn(self.num_features)
        return features


# ============================================================================
# Perceptual Loss (for images)
# ============================================================================

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.

    Helps images look more natural by comparing high-level features
    instead of just pixel values.

    Note: Requires torchvision. If not available, falls back to MSE.
    """
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.features = vgg.features[:16].eval()  # Use first 16 layers

            # Freeze parameters
            for param in self.features.parameters():
                param.requires_grad = False

            self.available = True
        except ImportError:
            print("Warning: torchvision not available, perceptual loss disabled")
            self.available = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.available:
            return F.mse_loss(x, y)

        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)

        x_norm = (x - mean) / std
        y_norm = (y - mean) / std

        # Extract features
        x_feat = self.features(x_norm)
        y_feat = self.features(y_norm)

        return F.mse_loss(x_feat, y_feat)


# ============================================================================
# Training Functions
# ============================================================================

def train_image_codec(
    codec: ImageCodec,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cpu",
    checkpoint_dir: str = "./checkpoints",
    use_perceptual_loss: bool = True
):
    """
    Train image codec on reconstruction task.

    Args:
        codec: ImageCodec instance
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        use_perceptual_loss: Whether to use perceptual loss
    """
    print("=" * 80)
    print("Training Image Codec")
    print("=" * 80)

    # Setup
    codec = codec.to(device)
    codec.train()
    optimizer = torch.optim.Adam(codec.parameters(), lr=lr)

    # Losses
    perceptual_loss = PerceptualLoss().to(device) if use_perceptual_loss else None

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 80)

        # Training
        total_loss = 0
        total_recon_loss = 0
        total_quant_loss = 0
        total_percept_loss = 0

        codec.train()
        for batch_idx, pixels in enumerate(tqdm(train_loader, desc="Training")):
            # Create modality from batched pixels
            image = MyImage(
                pixels=pixels.to(device),  # Already [B, C, H, W] from DataLoader
                metadata={}
            )

            # Normalize to [0, 1]
            if image.pixels.max() > 1.0:
                pixels_normalized = image.pixels / 255.0
            else:
                pixels_normalized = image.pixels

            # Forward pass
            # 1. Encode to continuous
            z_e = codec._encode(image)

            # 2. Quantize
            z_q, quant_loss, usage = codec.quantizer(z_e)

            # 3. Decode
            reconstructed = codec._decode(z_q, metadata=image.metadata)

            # Normalize reconstructed
            recon_normalized = reconstructed.pixels

            # Compute losses
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(recon_normalized, pixels_normalized)

            # Perceptual loss (if available)
            if perceptual_loss is not None:
                percept_loss = perceptual_loss(recon_normalized, pixels_normalized)
                loss = recon_loss + 0.1 * percept_loss + 0.25 * quant_loss
            else:
                percept_loss = torch.tensor(0.0)
                loss = recon_loss + 0.25 * quant_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_quant_loss += quant_loss.item()
            total_percept_loss += percept_loss.item()

        # Print epoch metrics
        num_batches = len(train_loader)
        print(f"\nTraining Metrics:")
        print(f"  Total Loss: {total_loss / num_batches:.4f}")
        print(f"  Recon Loss: {total_recon_loss / num_batches:.4f}")
        print(f"  Quant Loss: {total_quant_loss / num_batches:.4f}")
        if perceptual_loss is not None:
            print(f"  Percept Loss: {total_percept_loss / num_batches:.4f}")

        # Validation
        if val_loader is not None:
            codec.eval()
            val_loss = 0

            with torch.no_grad():
                for pixels in val_loader:
                    image = MyImage(
                        pixels=pixels.to(device),
                        metadata={}
                    )

                    # Encode and decode
                    tokens = codec.encode(image)
                    reconstructed = codec.decode(tokens, metadata=image.metadata)

                    # Normalize
                    if image.pixels.max() > 1.0:
                        pixels_normalized = image.pixels / 255.0
                    else:
                        pixels_normalized = image.pixels

                    # Loss
                    loss = F.mse_loss(reconstructed.pixels, pixels_normalized)
                    val_loss += loss.item()

            print(f"  Val Loss: {val_loss / len(val_loader):.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"image_codec_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': codec.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

    print("\n✓ Image codec training complete!")
    return codec


def train_timeseries_codec(
    codec: TimeSeriesCodec,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cpu",
    checkpoint_dir: str = "./checkpoints"
):
    """
    Train time series codec on reconstruction task.
    """
    print("=" * 80)
    print("Training Time Series Codec")
    print("=" * 80)

    # Setup
    codec = codec.to(device)
    codec.train()
    optimizer = torch.optim.Adam(codec.parameters(), lr=lr)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 80)

        total_loss = 0
        total_recon_loss = 0
        total_quant_loss = 0

        codec.train()
        for batch in tqdm(train_loader, desc="Training"):
            # Create modality from batched data
            timeseries = MyTimeSeries(
                values=batch['values'].to(device),  # [B, seq_length]
                timestamps=batch['timestamps'][0].to(device),  # Use first sample's timestamps
                mask=batch['mask'].to(device)
            )

            # Forward pass
            z_e = codec._encode(timeseries)
            z_q, quant_loss, usage = codec.quantizer(z_e)
            reconstructed = codec._decode(
                z_q,
                timestamps=timeseries.timestamps,
                mask=timeseries.mask
            )

            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed.values, timeseries.values)
            loss = recon_loss + 0.25 * quant_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_quant_loss += quant_loss.item()

        # Print metrics
        num_batches = len(train_loader)
        print(f"\nTraining Metrics:")
        print(f"  Total Loss: {total_loss / num_batches:.4f}")
        print(f"  Recon Loss: {total_recon_loss / num_batches:.4f}")
        print(f"  Quant Loss: {total_quant_loss / num_batches:.4f}")

        # Validation
        if val_loader is not None:
            codec.eval()
            val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    timeseries = MyTimeSeries(
                        values=batch['values'].to(device),
                        timestamps=batch['timestamps'][0].to(device),
                        mask=batch['mask'].to(device)
                    )

                    tokens = codec.encode(timeseries)
                    reconstructed = codec.decode(
                        tokens,
                        timestamps=timeseries.timestamps,
                        mask=timeseries.mask
                    )

                    loss = F.mse_loss(reconstructed.values, timeseries.values)
                    val_loss += loss.item()

            print(f"  Val Loss: {val_loss / len(val_loader):.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"timeseries_codec_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': codec.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

    print("\n✓ Time series codec training complete!")
    return codec


def train_tabular_codec(
    codec: TabularCodec,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cpu",
    checkpoint_dir: str = "./checkpoints"
):
    """
    Train tabular codec on reconstruction task.
    """
    print("=" * 80)
    print("Training Tabular Codec")
    print("=" * 80)

    # Setup
    codec = codec.to(device)
    codec.train()
    optimizer = torch.optim.Adam(codec.parameters(), lr=lr)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 80)

        total_loss = 0
        total_recon_loss = 0
        total_quant_loss = 0

        codec.train()
        for features in tqdm(train_loader, desc="Training"):
            # Create modality from batched features
            tabular = MyTabular(
                features=features.to(device),  # [B, num_features]
                feature_names=[f"feature_{i}" for i in range(features.shape[1])]
            )

            # Forward pass
            z_e = codec._encode(tabular)
            z_q, quant_loss, usage = codec.quantizer(z_e)
            reconstructed = codec._decode(z_q, feature_names=tabular.feature_names)

            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed.features, tabular.features)
            loss = recon_loss + 0.25 * quant_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_quant_loss += quant_loss.item()

        # Print metrics
        num_batches = len(train_loader)
        print(f"\nTraining Metrics:")
        print(f"  Total Loss: {total_loss / num_batches:.4f}")
        print(f"  Recon Loss: {total_recon_loss / num_batches:.4f}")
        print(f"  Quant Loss: {total_quant_loss / num_batches:.4f}")

        # Validation
        if val_loader is not None:
            codec.eval()
            val_loss = 0

            with torch.no_grad():
                for features in val_loader:
                    tabular = MyTabular(
                        features=features.to(device),
                        feature_names=[f"feature_{i}" for i in range(features.shape[1])]
                    )

                    tokens = codec.encode(tabular)
                    reconstructed = codec.decode(tokens, feature_names=tabular.feature_names)

                    loss = F.mse_loss(reconstructed.features, tabular.features)
                    val_loss += loss.item()

            print(f"  Val Loss: {val_loss / len(val_loader):.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"tabular_codec_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': codec.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

    print("\n✓ Tabular codec training complete!")
    return codec


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("STEP 5: Training Codecs")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # ========================================================================
    # Train Image Codec
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. Training Image Codec")
    print("=" * 80)

    # Create datasets
    train_dataset = ImageDataset(num_samples=100, image_size=224)
    val_dataset = ImageDataset(num_samples=20, image_size=224)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Create codec
    image_codec = ImageCodec(in_channels=3, embedding_dim=64)

    # Train
    image_codec = train_image_codec(
        codec=image_codec,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,
        lr=1e-4,
        device=device,
        checkpoint_dir="./checkpoints/image",
        use_perceptual_loss=False  # Set to True if torchvision available
    )

    # ========================================================================
    # Train Time Series Codec
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. Training Time Series Codec")
    print("=" * 80)

    # Create datasets
    train_dataset = TimeSeriesDataset(num_samples=100, seq_length=1000)
    val_dataset = TimeSeriesDataset(num_samples=20, seq_length=1000)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Create codec
    ts_codec = TimeSeriesCodec(embedding_dim=64)

    # Train
    ts_codec = train_timeseries_codec(
        codec=ts_codec,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,
        lr=1e-4,
        device=device,
        checkpoint_dir="./checkpoints/timeseries"
    )

    # ========================================================================
    # Train Tabular Codec
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. Training Tabular Codec")
    print("=" * 80)

    # Create datasets
    train_dataset = TabularDataset(num_samples=100, num_features=32)
    val_dataset = TabularDataset(num_samples=20, num_features=32)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Create codec
    tabular_codec = TabularCodec(num_features=32, embedding_dim=64)

    # Train
    tabular_codec = train_tabular_codec(
        codec=tabular_codec,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,
        lr=1e-4,
        device=device,
        checkpoint_dir="./checkpoints/tabular"
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("✓ Step 5 Complete: Codec Training")
    print("=" * 80)
    print("\nTrained Codecs:")
    print("  • Image Codec: Saved in ./checkpoints/image/")
    print("  • Time Series Codec: Saved in ./checkpoints/timeseries/")
    print("  • Tabular Codec: Saved in ./checkpoints/tabular/")

    print("\nKey Training Details:")
    print("  • Loss: Reconstruction MSE + Commitment Loss")
    print("  • Optimizer: Adam with lr=1e-4")
    print("  • Image codec can use perceptual loss for better quality")
    print("  • Codecs trained separately, independently")

    print("\n" + "=" * 80)
    print("Next Steps:")
    print("  → Step 6: Train transformer on masked modeling")
    print("  → Use these trained codecs as frozen tokenizers")
    print("  → Transformer learns to predict masked modalities")
    print("=" * 80)
