"""
Galaxy Image Segmentation Example

This shows how to adapt the multimodal framework for segmentation tasks like:
- Galaxy component segmentation (bulge, disk, spiral arms)
- Gravitational lensing arc detection
- Background object separation
"""

import torch
import torch.nn as nn
from step1_define_modalities import BaseModality
from astronomical_dataset import GalaxyImage
from astronomical_codecs import GalaxyImageCodec


# ============================================================================
# 1. Define Segmentation Mask Modality
# ============================================================================

class SegmentationMask(BaseModality):
    """
    Segmentation mask for galaxy images.

    Classes could be:
    0 = background
    1 = galaxy bulge
    2 = galaxy disk
    3 = spiral arms
    4 = foreground stars
    5 = background galaxies
    """

    token_key = "tok_segmentation"
    num_tokens = 64  # Same spatial resolution as encoded image (8x8 or 16x16)

    def __init__(self, mask: torch.Tensor, num_classes: int = 6, metadata: dict = None):
        """
        Args:
            mask: [B, H, W] with integer class labels 0-5
            num_classes: Number of segmentation classes
        """
        super().__init__(metadata)
        self.mask = mask
        self.num_classes = num_classes

    def validate(self):
        assert self.mask.ndim == 3, f"Expected [B, H, W], got {self.mask.shape}"
        assert self.mask.dtype in [torch.long, torch.int], "Mask should be integer type"
        assert self.mask.min() >= 0 and self.mask.max() < self.num_classes, \
            f"Mask values must be in [0, {self.num_classes-1}]"


# ============================================================================
# 2. Segmentation Codec (Image → Segmentation Mask)
# ============================================================================

class GalaxySegmentationCodec(nn.Module):
    """
    Encode image to spatial tokens, decode to segmentation mask.

    This is different from reconstruction - instead of reconstructing the image,
    we output class predictions for each pixel.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 6, embedding_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Reuse the trained encoder from GalaxyImageCodec
        self.encoder = SimpleImageEncoder(in_channels, embedding_dim)

        # Decoder outputs segmentation logits instead of RGB
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 256, kernel_size=4, stride=2, padding=1),  # 12×12
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),           # 24×24
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),            # 48×48
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=2, padding=1),    # 96×96
        )

    def encode(self, image: GalaxyImage) -> torch.Tensor:
        """Encode image to spatial feature map"""
        pixels = image.flux
        if pixels.max() > 1.0:
            pixels = pixels / 255.0
        pixels = pixels * 2.0 - 1.0

        z = self.encoder(pixels)  # [B, embedding_dim, 6, 6]
        # Flatten spatial for transformer
        B, C, H, W = z.shape
        z_flat = z.reshape(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        return z_flat

    def decode(self, z: torch.Tensor) -> SegmentationMask:
        """
        Decode spatial features to segmentation mask

        Args:
            z: [B, H*W, C] from transformer or [B, C, H, W] from encoder

        Returns:
            SegmentationMask with [B, 96, 96] class predictions
        """
        if z.ndim == 3:
            # Reshape from [B, H*W, C] to [B, C, H, W]
            B, N, C = z.shape
            H = W = int(N ** 0.5)
            z = z.permute(0, 2, 1).reshape(B, C, H, W)

        logits = self.decoder(z)  # [B, num_classes, 96, 96]
        mask = logits.argmax(dim=1)  # [B, 96, 96]

        return SegmentationMask(mask=mask, num_classes=self.num_classes)


# ============================================================================
# 3. Training Loop for Segmentation
# ============================================================================

def train_segmentation(model, train_loader, num_epochs=50, device='cuda'):
    """
    Train segmentation model.

    Loss: Cross-entropy between predicted and true segmentation masks
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            images = batch['galaxy_image']  # GalaxyImage objects
            masks = batch['segmentation_mask']  # Ground truth masks [B, H, W]

            # Stack images
            image_batch = torch.cat([img.flux for img in images], dim=0).to(device)
            mask_batch = torch.cat([m for m in masks], dim=0).to(device)

            # Forward pass
            # Option 1: Direct encoder→decoder (no transformer)
            z = model.encoder(image_batch * 2.0 - 1.0)
            logits = model.decoder(z)  # [B, num_classes, H, W]

            # Compute loss
            loss = criterion(logits, mask_batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

    return model


# ============================================================================
# 4. Multimodal Segmentation (Use Other Modalities)
# ============================================================================

def multimodal_segmentation_example():
    """
    Example: Predict segmentation mask from image + spectrum + redshift

    This uses the transformer to learn cross-modal relationships.
    """
    print("Example: Multimodal Segmentation")
    print("=" * 80)

    # You would modify your MultimodalTransformer to output segmentation tokens
    # Then decode those tokens to a segmentation mask

    # Pseudocode:
    """
    # Encode all modalities
    tokens = {
        'tok_galaxy_image': image_codec.encode(image),
        'tok_gaia_spectrum': spectrum_codec.encode(spectrum),
        'tok_redshift': redshift_codec.encode(redshift)
    }

    # Predict segmentation tokens
    predicted_tokens = transformer(tokens, predict=['tok_segmentation'])

    # Decode to mask
    segmentation_mask = segmentation_codec.decode(predicted_tokens['tok_segmentation'])
    """
    pass


# ============================================================================
# 5. Usage Example
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("GALAXY SEGMENTATION EXAMPLE")
    print("=" * 80)

    # Create model
    seg_model = GalaxySegmentationCodec(
        in_channels=3,
        num_classes=6,  # background, bulge, disk, arms, stars, bg_galaxies
        embedding_dim=64
    )

    print(f"Model parameters: {sum(p.numel() for p in seg_model.parameters()):,}")

    # Example forward pass
    dummy_image = GalaxyImage(
        flux=torch.randn(2, 3, 96, 96),  # 2 images
        metadata={}
    )

    # Encode
    z = seg_model.encode(dummy_image)
    print(f"Encoded shape: {z.shape}")  # [2, 36, 64]

    # Decode
    mask = seg_model.decode(z)
    print(f"Segmentation mask shape: {mask.mask.shape}")  # [2, 96, 96]
    print(f"Unique classes in mask: {mask.mask.unique().tolist()}")

    print("\n✓ Segmentation example complete!")
    print("\nTo use this for real:")
    print("1. Create ground truth segmentation masks (manual annotation or simulated)")
    print("2. Add SegmentationMask to your dataset")
    print("3. Train the segmentation codec")
    print("4. Integrate with your transformer for multimodal segmentation")
