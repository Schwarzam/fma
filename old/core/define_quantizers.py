"""
Step 2: Define Quantizers

Quantizers convert continuous embeddings into discrete tokens (and back).

Key concepts:
- quantize(): Continuous → Discrete tokens
- decode(): Discrete tokens → Continuous (lookup codebook)
- forward(): Training interface with commitment loss

Different quantizers for different data types:
1. VectorQuantizer (VQ-VAE): Learned codebook with nearest neighbor
2. FiniteScalarQuantizer (FSQ): Fixed levels, no codebook
3. ScalarLinearQuantizer: Fixed bins for bounded ranges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple
from jaxtyping import Float
from torch import Tensor


# ============================================================================
# Abstract Base Quantizer
# ============================================================================

class Quantizer(ABC, nn.Module):
    """
    Abstract base class for all quantizers.

    A quantizer discretizes continuous embeddings into tokens.
    """

    @abstractmethod
    def quantize(self, x: Float[Tensor, "batch channels *shape"]) -> Float[Tensor, "batch channels *shape"]:
        """
        Quantize continuous embeddings to discrete tokens.

        Args:
            x: Continuous embeddings

        Returns:
            Quantized embeddings (discrete)
        """
        pass

    @abstractmethod
    def decode(self, z: Float[Tensor, "batch channels *shape"]) -> Float[Tensor, "batch channels *shape"]:
        """
        Convert discrete tokens back to continuous embeddings.

        Args:
            z: Discrete token indices

        Returns:
            Continuous embeddings from codebook
        """
        pass

    @abstractmethod
    def forward(self, z_e: Float[Tensor, "..."]) -> Tuple[Tensor, Tensor, float]:
        """
        Training interface.

        Args:
            z_e: Encoder output (continuous)

        Returns:
            z_q: Quantized embeddings
            loss: Commitment/codebook loss
            usage: Codebook usage percentage
        """
        pass


# ============================================================================
# 1. Vector Quantizer (VQ-VAE)
# ============================================================================

class VectorQuantizer(Quantizer):
    """
    Vector Quantizer from VQ-VAE (van den Oord et al. 2017).

    Learns a discrete codebook and maps embeddings to nearest codebook entry.

    Good for: Images, audio, any rich continuous embeddings

    Args:
        num_embeddings: Size of codebook (e.g., 8192)
        embedding_dim: Dimension of each code (e.g., 256)
        commitment_cost: Weight for commitment loss (default: 0.25)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Learnable codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def get_indices(self, x: Float[Tensor, "batch channels *shape"]) -> Tensor:
        """
        Get discrete codebook indices for encoder output.

        Args:
            x: Encoder output [batch, channels, spatial...]

        Returns:
            indices: [batch, spatial...] discrete token IDs
        """
        # Flatten spatial dimensions
        batch_size = x.shape[0]
        x_flat = x.reshape(-1, self.embedding_dim)  # [batch*spatial, dim]

        # Compute distances to all codebook entries
        distances = torch.cdist(x_flat, self.embedding.weight)  # [batch*spatial, num_embeddings]

        # Find nearest
        indices = torch.argmin(distances, dim=1)  # [batch*spatial]

        # Reshape to [batch, spatial...]
        spatial_shape = x.shape[2:]
        indices = indices.reshape(batch_size, *spatial_shape)

        return indices

    def quantize(self, x: Float[Tensor, "batch channels *shape"]) -> Float[Tensor, "batch channels *shape"]:
        """Find nearest codebook entry"""
        # Flatten spatial dimensions
        original_shape = x.shape
        x_flat = x.reshape(-1, self.embedding_dim)  # [batch*spatial, dim]

        # Compute distances to all codebook entries
        distances = torch.cdist(x_flat, self.embedding.weight)  # [batch*spatial, num_embeddings]

        # Find nearest
        indices = torch.argmin(distances, dim=1)  # [batch*spatial]

        # Lookup quantized values
        quantized = self.embedding(indices)  # [batch*spatial, dim]

        # Reshape back
        quantized = quantized.reshape(original_shape)

        return quantized

    def decode(self, z: Float[Tensor, "batch *shape"]) -> Float[Tensor, "batch channels *shape"]:
        """Lookup codebook entries"""
        indices = z.long()
        quantized = self.embedding(indices)  # [batch, spatial, dim]

        # Rearrange to [batch, dim, spatial]
        if quantized.ndim == 3:  # [batch, spatial, dim]
            quantized = quantized.permute(0, 2, 1)  # [batch, dim, spatial]

        return quantized

    def forward(self, z_e: Float[Tensor, "batch channels *shape"]) -> Tuple[Tensor, Tensor, float]:
        """
        Training forward pass with straight-through estimator.

        Returns:
            z_q: Quantized embeddings
            loss: VQ loss (commitment + codebook)
            usage: Fraction of codebook used
        """
        # Flatten
        original_shape = z_e.shape
        z_e_flat = z_e.reshape(-1, self.embedding_dim)

        # Find nearest
        distances = torch.cdist(z_e_flat, self.embedding.weight)
        indices = torch.argmin(distances, dim=1)
        z_q_flat = self.embedding(indices)

        # Reshape
        z_q = z_q_flat.reshape(original_shape)

        # Straight-through estimator: copy gradients
        z_q = z_e + (z_q - z_e).detach()

        # VQ loss
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Codebook usage
        unique_indices = torch.unique(indices)
        usage = len(unique_indices) / self.num_embeddings

        return z_q, loss, usage


# ============================================================================
# 2. Finite Scalar Quantizer (FSQ)
# ============================================================================

class FiniteScalarQuantizer(Quantizer):
    """
    Finite Scalar Quantization (Mentzer et al. 2023).

    No learned codebook - uses fixed quantization levels.
    Deterministic and simple.

    Good for: Images, any data where you want fixed discretization

    Args:
        levels: List of levels per dimension (e.g., [8, 5, 5, 5] = 5000 codes)
    """

    def __init__(self, levels: list[int]):
        super().__init__()
        self.levels = levels
        self.num_channels = len(levels)
        self.codebook_size = int(torch.prod(torch.tensor(levels)).item())

        # Precompute level boundaries
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.float32))

    def _scale_to_levels(self, x: Tensor, level: int) -> Tensor:
        """Map continuous value to discrete level"""
        # Input x in [-1, 1], map to [0, level-1]
        x = torch.clamp(x, -1.0, 1.0)
        x = (x + 1.0) / 2.0  # [0, 1]
        x = x * (level - 1)  # [0, level-1]
        return torch.round(x)

    def _unscale_from_levels(self, x: Tensor, level: int) -> Tensor:
        """Map discrete level back to continuous"""
        x = x / (level - 1)  # [0, 1]
        x = x * 2.0 - 1.0  # [-1, 1]
        return x

    def get_indices(self, x: Float[Tensor, "batch channels *shape"]) -> Tensor:
        """
        Convert quantized levels to flat token indices.

        For levels [8,5,5,5], maps 4D indices to single index in [0, 999].
        Example: indices [2,3,1,4] -> 2*5*5*5 + 3*5*5 + 1*5 + 4 = 334
        """
        # Quantize to discrete levels
        quantized = self.quantize(x)  # [B, 4, H, W] with values in [0, level-1]

        # Flatten spatial dimensions
        B, C = quantized.shape[:2]
        spatial_shape = quantized.shape[2:]
        quantized_flat = quantized.reshape(B, C, -1)  # [B, 4, H*W]

        # Convert multi-dimensional indices to flat index
        # For [8,5,5,5]: index = i0*5*5*5 + i1*5*5 + i2*5 + i3
        multipliers = [1]
        for level in reversed(self.levels[1:]):
            multipliers.insert(0, multipliers[0] * level)
        multipliers = torch.tensor(multipliers, device=x.device).view(C, 1)

        # Compute flat indices
        flat_indices = (quantized_flat * multipliers).sum(dim=1).long()  # [B, H*W]

        # Reshape to spatial
        flat_indices = flat_indices.reshape(B, *spatial_shape)

        return flat_indices

    def quantize(self, x: Float[Tensor, "batch channels *shape"]) -> Float[Tensor, "batch channels *shape"]:
        """Quantize each channel independently"""
        quantized = torch.zeros_like(x)

        for i, level in enumerate(self.levels):
            quantized[:, i] = self._scale_to_levels(x[:, i], level)

        return quantized

    def indices_to_levels(self, indices: Tensor) -> Tensor:
        """
        Convert flat indices back to multi-dimensional levels.

        Args:
            indices: [B, *spatial] with values in [0, codebook_size-1]

        Returns:
            levels: [B, num_channels, *spatial] with discrete levels
        """
        # Flatten spatial
        B = indices.shape[0]
        spatial_shape = indices.shape[1:]
        indices_flat = indices.reshape(B, -1)  # [B, H*W]

        # Decompose flat index into per-channel indices
        # For [8,5,5,5]: 334 -> [2,3,1,4]
        levels_list = []
        remaining = indices_flat.long()

        for i, level in enumerate(self.levels):
            if i == 0:
                # First channel
                divisor = int(torch.prod(torch.tensor(self.levels[1:])).item())
                channel_indices = remaining // divisor
                remaining = remaining % divisor
            elif i < len(self.levels) - 1:
                # Middle channels
                divisor = int(torch.prod(torch.tensor(self.levels[i+1:])).item())
                channel_indices = remaining // divisor
                remaining = remaining % divisor
            else:
                # Last channel
                channel_indices = remaining

            levels_list.append(channel_indices)

        # Stack to [B, C, H*W]
        levels = torch.stack(levels_list, dim=1).float()

        # Reshape to [B, C, *spatial]
        levels = levels.reshape(B, len(self.levels), *spatial_shape)

        return levels

    def decode(self, z: Float[Tensor, "batch *shape"]) -> Float[Tensor, "batch channels *shape"]:
        """
        Dequantize back to continuous.

        Args:
            z: Can be either:
               - [B, C, *spatial] discrete levels (old behavior)
               - [B, *spatial] flat indices (new behavior for transformer)

        Returns:
            continuous: [B, C, *spatial] continuous values in [-1, 1]
        """
        # Check if input is flat indices or multi-channel levels
        if z.ndim >= 2 and z.shape[1] == len(self.levels):
            # Old behavior: already has channel dimension [B, 4, H, W]
            levels = z
        else:
            # New behavior: flat indices [B, H, W]
            levels = self.indices_to_levels(z)

        continuous = torch.zeros_like(levels, dtype=torch.float32)

        for i, level in enumerate(self.levels):
            continuous[:, i] = self._unscale_from_levels(levels[:, i].float(), level)

        return continuous

    def forward(self, z_e: Float[Tensor, "batch channels *shape"]) -> Tuple[Tensor, Tensor, float]:
        """
        Training forward pass.

        FSQ has no learned parameters, so no loss.
        Uses straight-through estimator for gradients.
        """
        z_q = self.quantize(z_e)
        z_q_continuous = self.decode(z_q)

        # Straight-through: copy gradients
        z_q_st = z_e + (z_q_continuous - z_e).detach()

        # No loss for FSQ
        loss = torch.tensor(0.0, device=z_e.device)

        # Usage is always 100% (deterministic)
        usage = 1.0

        return z_q_st, loss, usage


# ============================================================================
# 3. Scalar Linear Quantizer
# ============================================================================

class ScalarLinearQuantizer(Quantizer):
    """
    Simple linear quantization for scalars with known range.

    Maps continuous range [min_val, max_val] to discrete bins.

    Good for: Bounded scalars (redshift, angles, percentages)

    Args:
        num_bins: Number of discrete bins (e.g., 256)
        min_val: Minimum value of range
        max_val: Maximum value of range
    """

    def __init__(self, num_bins: int, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__()
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        self.bin_width = (max_val - min_val) / num_bins

    def quantize(self, x: Float[Tensor, "batch 1"]) -> Float[Tensor, "batch 1"]:
        """Map to bin indices"""
        # Clamp to range
        x_clamped = torch.clamp(x, self.min_val, self.max_val)

        # Map to [0, num_bins-1]
        x_normalized = (x_clamped - self.min_val) / (self.max_val - self.min_val)
        bin_indices = torch.floor(x_normalized * self.num_bins)
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)

        return bin_indices

    def decode(self, z: Float[Tensor, "batch 1"]) -> Float[Tensor, "batch 1"]:
        """Map bin indices back to continuous"""
        # Bin center
        continuous = self.min_val + (z + 0.5) * self.bin_width
        return continuous

    def forward(self, z_e: Float[Tensor, "batch 1"]) -> Tuple[Tensor, Tensor, float]:
        """Training forward with straight-through"""
        z_q = self.quantize(z_e)
        z_q_continuous = self.decode(z_q)

        # Straight-through
        z_q_st = z_e + (z_q_continuous - z_e).detach()

        # No learned parameters
        loss = torch.tensor(0.0, device=z_e.device)
        usage = 1.0

        return z_q_st, loss, usage


# ============================================================================
# 4. Identity Quantizer (No Quantization)
# ============================================================================

class IdentityQuantizer(Quantizer):
    """
    Pass-through quantizer (no quantization).

    Good for: Testing, or when you want continuous tokens

    Args:
        num_codes: Virtual codebook size (not used, for compatibility)
    """

    def __init__(self, num_codes: int = 1):
        super().__init__()
        self.num_codes = num_codes

    def quantize(self, x: Tensor) -> Tensor:
        """Pass through"""
        return x

    def decode(self, z: Tensor) -> Tensor:
        """Pass through"""
        return z

    def forward(self, z_e: Tensor) -> Tuple[Tensor, Tensor, float]:
        """No quantization, no loss"""
        return z_e, torch.tensor(0.0, device=z_e.device), 1.0


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("STEP 2: Defining Quantizers")
    print("=" * 80)

    batch_size = 4
    spatial_size = 16

    # ========================================================================
    # Test 1: Vector Quantizer (for images)
    # ========================================================================
    print("\n1. Vector Quantizer (VQ-VAE)")
    print("-" * 80)

    vq = VectorQuantizer(num_embeddings=512, embedding_dim=64, commitment_cost=0.25)
    print(f"✓ Created VQ with {vq.num_embeddings} codes, dim={vq.embedding_dim}")

    # Simulate encoder output: [batch, channels, spatial]
    z_e = torch.randn(batch_size, 64, spatial_size)
    print(f"  - Input shape: {z_e.shape}")

    # Quantize
    z_q, loss, usage = vq(z_e)
    print(f"  - Quantized shape: {z_q.shape}")
    print(f"  - VQ loss: {loss.item():.4f}")
    print(f"  - Codebook usage: {usage:.2%}")

    # Decode (simulate discrete tokens)
    tokens = torch.randint(0, 512, (batch_size, spatial_size))
    decoded = vq.decode(tokens)
    print(f"  - Decoded shape: {decoded.shape}")

    # ========================================================================
    # Test 2: Finite Scalar Quantizer (for images)
    # ========================================================================
    print("\n2. Finite Scalar Quantizer (FSQ)")
    print("-" * 80)

    fsq = FiniteScalarQuantizer(levels=[8, 5, 5, 5])
    print(f"✓ Created FSQ with levels {fsq.levels}")
    print(f"  - Codebook size: {fsq.codebook_size} codes")

    # Simulate encoder output: [batch, 4 channels, spatial, spatial]
    z_e = torch.randn(batch_size, 4, 16, 16) * 0.5  # Keep in reasonable range
    print(f"  - Input shape: {z_e.shape}")

    # Quantize
    z_q, loss, usage = fsq(z_e)
    print(f"  - Quantized shape: {z_q.shape}")
    print(f"  - FSQ loss: {loss.item():.4f}")
    print(f"  - Usage: {usage:.2%}")

    # Decode
    decoded = fsq.decode(z_q)
    print(f"  - Decoded shape: {decoded.shape}")
    print(f"  - Reconstruction error: {F.mse_loss(decoded, z_e).item():.4f}")

    # ========================================================================
    # Test 3: Scalar Linear Quantizer (for scalars)
    # ========================================================================
    print("\n3. Scalar Linear Quantizer")
    print("-" * 80)

    slq = ScalarLinearQuantizer(num_bins=256, min_val=0.0, max_val=1.0)
    print(f"✓ Created Linear Quantizer with {slq.num_bins} bins")
    print(f"  - Range: [{slq.min_val}, {slq.max_val}]")

    # Simulate scalar values (e.g., redshift)
    scalars = torch.rand(batch_size, 1)  # Random values in [0, 1]
    print(f"  - Input shape: {scalars.shape}")
    print(f"  - Input values: {scalars.squeeze().tolist()}")

    # Quantize
    z_q, loss, usage = slq(scalars)
    print(f"  - Quantized (bins): {z_q.squeeze().tolist()}")
    print(f"  - Loss: {loss.item():.4f}")

    # Decode
    decoded = slq.decode(z_q)
    print(f"  - Decoded values: {decoded.squeeze().tolist()}")
    print(f"  - Reconstruction error: {F.mse_loss(decoded, scalars).item():.6f}")

    # ========================================================================
    # Test 4: Identity Quantizer
    # ========================================================================
    print("\n4. Identity Quantizer (No Quantization)")
    print("-" * 80)

    identity = IdentityQuantizer(num_codes=1)
    print(f"✓ Created Identity Quantizer")

    # Pass through
    x = torch.randn(batch_size, 10)
    z_q, loss, usage = identity(x)
    print(f"  - Input equals output: {torch.allclose(x, z_q)}")
    print(f"  - Loss: {loss.item():.4f}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("✓ Step 2 Complete: Quantizers Defined")
    print("=" * 80)
    print("\nQuantizer Selection Guide:")
    print("  • VectorQuantizer (VQ-VAE):")
    print("    - Rich continuous embeddings (images, audio)")
    print("    - Learned codebook")
    print("    - Requires commitment loss")
    print("\n  • FiniteScalarQuantizer (FSQ):")
    print("    - Images, structured data")
    print("    - No learned parameters")
    print("    - Deterministic")
    print("\n  • ScalarLinearQuantizer:")
    print("    - Bounded scalars (age, price, angles)")
    print("    - Fixed bins in known range")
    print("\n  • IdentityQuantizer:")
    print("    - Testing or continuous tokens")
    print("    - No discretization")

    print("\n" + "=" * 80)
    print("Next Steps:")
    print("  → Step 3: Implement codecs (encoder + decoder + quantizer)")
    print("  → Step 4: Create codec manager")
    print("  → Step 5: Train codecs")
    print("=" * 80)
