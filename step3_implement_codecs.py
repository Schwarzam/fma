"""
Step 3: Implement Codecs

Codecs combine encoders, decoders, and quantizers to transform modalities into tokens.

Architecture:
    Input Modality → Encoder → Continuous Embedding → Quantizer → Discrete Tokens
    Discrete Tokens → Quantizer Decode → Continuous Embedding → Decoder → Output Modality

Each codec must implement:
- _encode(): Modality → Continuous embedding
- _decode(): Continuous embedding → Modality
- quantizer property: Returns the quantizer instance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Type
from jaxtyping import Float
from torch import Tensor

# Import from previous steps
from step1_define_modalities import BaseModality, MyImage, MyTimeSeries, MyScalar, MyTabular
from step2_define_quantizers import Quantizer, VectorQuantizer, FiniteScalarQuantizer, ScalarLinearQuantizer


# ============================================================================
# Abstract Base Codec
# ============================================================================

class Codec(ABC, nn.Module):
    """
    Abstract base class for all codecs.

    A codec transforms a modality into discrete tokens and back.
    """

    @property
    @abstractmethod
    def modality(self) -> Type[BaseModality]:
        """Return the modality class this codec handles"""
        pass

    @abstractmethod
    def _encode(self, x: BaseModality) -> Float[Tensor, "batch channels *shape"]:
        """
        Encode modality to continuous embedding.

        Args:
            x: Input modality

        Returns:
            Continuous embedding tensor
        """
        pass

    @abstractmethod
    def _decode(self, z: Float[Tensor, "batch channels *shape"], **metadata) -> BaseModality:
        """
        Decode continuous embedding back to modality.

        Args:
            z: Continuous embedding
            **metadata: Optional metadata for reconstruction

        Returns:
            Reconstructed modality
        """
        pass

    @property
    @abstractmethod
    def quantizer(self) -> Quantizer:
        """Return the quantizer instance"""
        pass

    def encode(self, x: BaseModality) -> Float[Tensor, "batch num_tokens"]:
        """
        Public interface: Encode modality to discrete tokens.

        Pipeline: Modality → _encode() → Continuous → Quantizer → Discrete tokens
        """
        # Validate input type
        if not isinstance(x, self.modality):
            raise TypeError(f"Expected {self.modality.__name__}, got {type(x).__name__}")

        # Encode to continuous
        z_continuous = self._encode(x)

        # Get discrete token indices if quantizer supports it (e.g., VectorQuantizer)
        # Otherwise use quantized continuous values (e.g., FSQ, ScalarLinearQuantizer)
        if hasattr(self.quantizer, 'get_indices'):
            # VectorQuantizer: get discrete indices [batch, spatial...]
            z_quantized = self.quantizer.get_indices(z_continuous)
        else:
            # Other quantizers: use continuous quantized values
            z_quantized, _, _ = self.quantizer(z_continuous)

        # Flatten if needed
        if z_quantized.ndim > 2:
            batch_size = z_quantized.shape[0]
            z_quantized = z_quantized.reshape(batch_size, -1)

        return z_quantized

    def decode(self, z: Float[Tensor, "batch num_tokens"], **metadata) -> BaseModality:
        """
        Public interface: Decode discrete tokens back to modality.

        Pipeline: Discrete tokens → Dequantize → Continuous → _decode() → Modality
        """
        # Dequantize to continuous
        z_continuous = self.quantizer.decode(z)

        # Decode to modality
        return self._decode(z_continuous, **metadata)


# ============================================================================
# 1. Image Codec
# ============================================================================

class SimpleImageEncoder(nn.Module):
    """
    Simple CNN encoder for images.

    Downsamples 224×224 → 16×16 with 4× spatial downsampling
    """
    def __init__(self, in_channels: int = 3, embedding_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)   # 112×112
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)            # 56×56
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)           # 28×28
        self.conv4 = nn.Conv2d(256, embedding_dim, kernel_size=4, stride=2, padding=1) # 14×14

        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.conv4(x)  # No activation on output
        return x


class SimpleImageDecoder(nn.Module):
    """
    Simple CNN decoder for images.

    Upsamples 14×14 → 224×224
    """
    def __init__(self, embedding_dim: int = 64, out_channels: int = 3):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(embedding_dim, 256, kernel_size=4, stride=2, padding=1)  # 28×28
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)            # 56×56
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)             # 112×112
        self.conv4 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)    # 224×224

        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = torch.tanh(self.conv4(x))  # Tanh to output [-1, 1]
        return x


class ImageCodec(Codec):
    """
    Codec for image modality.

    Uses CNN encoder/decoder with FSQ quantization.
    """

    def __init__(self, in_channels: int = 3, embedding_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        # Encoder and decoder
        self.encoder = SimpleImageEncoder(in_channels, embedding_dim)
        self.decoder = SimpleImageDecoder(embedding_dim, in_channels)

        # Quantizer: FSQ with 4 channels
        self.projection = nn.Conv2d(embedding_dim, 4, kernel_size=1)  # Project to 4 channels for FSQ
        self.unprojection = nn.Conv2d(4, embedding_dim, kernel_size=1)  # Project back
        self._quantizer = FiniteScalarQuantizer(levels=[8, 5, 5, 5])

    @property
    def modality(self) -> Type[BaseModality]:
        return MyImage

    @property
    def quantizer(self) -> Quantizer:
        return self._quantizer

    def _encode(self, x: MyImage) -> Float[Tensor, "batch 4 h w"]:
        """Encode image to continuous embedding"""
        # Normalize to [-1, 1]
        pixels = x.pixels
        if pixels.max() > 1.0:
            pixels = pixels / 255.0
        pixels = pixels * 2.0 - 1.0

        # Encode
        z = self.encoder(pixels)  # [batch, embedding_dim, 14, 14]
        z = self.projection(z)     # [batch, 4, 14, 14]

        return z

    def _decode(self, z: Float[Tensor, "batch *"], **metadata) -> MyImage:
        """
        Decode embedding back to image.
        Accepts either flat tokens [B, 4*H*W] or shaped [B, 4, H, W].
        """
        # If flat, reshape back to [B, 4, H, W]
        if z.ndim == 2:
            B, N = z.shape
            # FSQ channels = 4; assume square spatial map
            C = 4
            assert N % C == 0, f"Token length {N} not divisible by {C}"
            HW = N // C
            H = W = int(HW**0.5)
            assert H * W == HW, f"Cannot infer square spatial size from {HW}"
            z = z.view(B, C, H, W)

        # Unproject to model embedding dim and decode
        z = self.unprojection(z)                 # [B, embedding_dim, H, W]
        pixels = self.decoder(z)                 # [B, C_out, 224, 224]
        pixels = (pixels + 1.0) / 2.0            # [-1,1] -> [0,1]
        return MyImage(pixels=pixels, metadata=metadata.get('metadata', None))


# ============================================================================
# 2. Time Series Codec
# ============================================================================

class TimeSeriesEncoder(nn.Module):
    """
    1D CNN encoder for time series.

    Downsamples temporal dimension: 1000 → 128
    """
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=8, stride=2, padding=3)    # 500
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=3)   # 250
        self.conv3 = nn.Conv1d(64, embedding_dim, kernel_size=8, stride=2, padding=3)  # 125

        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        return x


class TimeSeriesDecoder(nn.Module):
    """
    1D CNN decoder for time series.

    Upsamples: 125 → 1000
    """
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(embedding_dim, 64, kernel_size=8, stride=2, padding=3)  # 250
        self.conv2 = nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2, padding=3)             # 500
        self.conv3 = nn.ConvTranspose1d(32, 1, kernel_size=8, stride=2, padding=3)              # 1000

        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        return x


class TimeSeriesCodec(Codec):
    """
    Codec for time series modality.

    Uses 1D CNN with VQ quantization.
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.encoder = TimeSeriesEncoder(embedding_dim)
        self.decoder = TimeSeriesDecoder(embedding_dim)
        self._quantizer = VectorQuantizer(num_embeddings=512, embedding_dim=embedding_dim)

    @property
    def modality(self) -> Type[BaseModality]:
        return MyTimeSeries

    @property
    def quantizer(self) -> Quantizer:
        return self._quantizer

    def _encode(self, x: MyTimeSeries) -> Float[Tensor, "batch embedding_dim temporal"]:
        """Encode time series to continuous embedding"""
        # Add channel dimension
        values = x.values.unsqueeze(1)  # [batch, 1, timesteps]

        # Encode
        z = self.encoder(values)  # [batch, embedding_dim, ~125]

        return z

    def _decode(self, z: Float[Tensor, "batch *"], **metadata) -> MyTimeSeries:
        """
        Decode embedding back to time series.
        Accepts:
        - flat tokens [B, E*T]
        - shaped [B, E, T]
        - shaped [B, T, E]
        Ensures decoder sees [B, E, T] and returns values [B, T_out].
        If T_out ≠ len(timestamps/mask), resamples to match.
        """
        B = z.size(0)

        # 1) Normalize shape to [B, E, T]
        if z.ndim == 2:
            E = self.embedding_dim
            N = z.size(1)
            assert N % E == 0, f"Token length {N} not divisible by embedding dim {E}"
            T = N // E
            z = z.view(B, E, T)                     # [B, E, T]
        elif z.ndim == 3:
            # Either [B, E, T] (good) or [B, T, E] (transpose)
            if z.shape[1] != self.embedding_dim and z.shape[2] == self.embedding_dim:
                z = z.transpose(1, 2)               # [B, E, T]
            elif z.shape[1] != self.embedding_dim:
                raise ValueError(f"Unexpected z shape {tuple(z.shape)} for embedding_dim={self.embedding_dim}")
        else:
            raise ValueError(f"Expected 2D or 3D z, got {z.ndim}D")

        # 2) Decode -> [B, 1, T_out] (by design of your decoder)
        out = self.decoder(z)

        # Defensive: if channels ≠ 1, collapse channels (shouldn't happen, but avoids silent blow-ups)
        if out.ndim == 3 and out.shape[1] != 1:
            out = out.mean(dim=1, keepdim=True)     # [B, 1, T_out]

        values = out.squeeze(1)                      # [B, T_out]

        # 3) Match original length if metadata provided
        target_len = None
        ts = metadata.get('timestamps', None)
        mk = metadata.get('mask', None)
        if ts is not None:
            target_len = ts.shape[-1]
        elif mk is not None:
            target_len = mk.shape[-1]

        if target_len is not None and values.shape[-1] != target_len:
            # Linear resample to match (keeps codepath simple for the demo)
            values = F.interpolate(values.unsqueeze(1), size=target_len, mode="linear", align_corners=False).squeeze(1)

        return MyTimeSeries(values=values, timestamps=ts, mask=mk)


# ============================================================================
# 3. Scalar Codec
# ============================================================================

class ScalarCodec(Codec):
    """
    Codec for scalar modality.

    Uses identity mapping (no encoder/decoder) with linear quantization.
    Simple scalars don't need learned embeddings.
    """

    def __init__(self, num_bins: int = 256, min_val: float = 0.0, max_val: float = 100.0):
        super().__init__()
        self._quantizer = ScalarLinearQuantizer(num_bins=num_bins, min_val=min_val, max_val=max_val)

    @property
    def modality(self) -> Type[BaseModality]:
        return MyScalar

    @property
    def quantizer(self) -> Quantizer:
        return self._quantizer

    def _encode(self, x: MyScalar) -> Float[Tensor, "batch 1"]:
        """Identity encoding for scalars"""
        return x.value

    def _decode(self, z: Float[Tensor, "batch 1"], **metadata) -> MyScalar:
        """Identity decoding for scalars"""
        name = metadata.get('name', 'unnamed_scalar')
        return MyScalar(value=z, name=name)


# ============================================================================
# 4. Tabular Codec
# ============================================================================

class TabularEncoder(nn.Module):
    """Simple MLP encoder for tabular data"""
    def __init__(self, num_features: int, embedding_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TabularDecoder(nn.Module):
    """Simple MLP decoder for tabular data"""
    def __init__(self, embedding_dim: int, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_features)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TabularCodec(Codec):
    """
    Codec for tabular/multi-feature data.

    Uses MLP encoder/decoder with VQ quantization.
    """

    def __init__(self, num_features: int = 32, embedding_dim: int = 64):
        super().__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim

        self.encoder = TabularEncoder(num_features, embedding_dim)
        self.decoder = TabularDecoder(embedding_dim, num_features)
        self._quantizer = VectorQuantizer(num_embeddings=256, embedding_dim=embedding_dim)

    @property
    def modality(self) -> Type[BaseModality]:
        return MyTabular

    @property
    def quantizer(self) -> Quantizer:
        return self._quantizer

    def _encode(self, x: MyTabular) -> Float[Tensor, "batch embedding_dim"]:
        """Encode tabular features to continuous embedding"""
        z = self.encoder(x.features)  # [batch, embedding_dim]
        return z.unsqueeze(-1)  # [batch, embedding_dim, 1] for compatibility

    def _decode(self, z: Float[Tensor, "batch *"], **metadata) -> MyTabular:
        """
        Decode embedding back to tabular features.
        Accepts flat [B, E], or shaped [B, E, 1] / [B, 1, E] / [B, E, T].
        Returns strictly 2D features [B, F] as required by MyTabular.
        """
        if z.ndim == 3:
            B = z.size(0)
            # Common singleton cases first
            if z.shape[-1] == 1:               # [B, E, 1] -> [B, E]
                z = z[..., 0].contiguous()
            elif z.shape[1] == 1:              # [B, 1, E] -> [B, E]
                z = z[:, 0, :].contiguous()
            elif z.shape[1] == self.embedding_dim:  # [B, E, T] -> collapse T
                z = z.reshape(B, self.embedding_dim, -1).mean(-1).contiguous()
            elif z.shape[2] == self.embedding_dim:  # [B, T, E] -> transpose, collapse T
                z = z.transpose(1, 2).reshape(B, self.embedding_dim, -1).mean(-1).contiguous()
            else:
                # Fallback: flatten safely; will match decoder only if width==embedding_dim
                z = z.reshape(B, -1).contiguous()
        elif z.ndim == 2:
            z = z.contiguous()
        else:
            raise ValueError(f"Expected 2D/3D z, got {z.ndim}D")

        # Now z is [B, E]; decoder expects in_features == embedding_dim
        if z.size(1) != self.embedding_dim:
            raise RuntimeError(f"Decoded width {z.size(1)} != embedding_dim {self.embedding_dim}")

        features = self.decoder(z)  # [B, num_features]
        if features.ndim != 2:
            features = features.reshape(features.size(0), -1)

        return MyTabular(
            features=features,
            feature_names=metadata.get('feature_names', None),
            categorical_mask=metadata.get('categorical_mask', None),
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("STEP 3: Implementing Codecs")
    print("=" * 80)

    # ========================================================================
    # Test 1: Image Codec
    # ========================================================================
    print("\n1. Image Codec")
    print("-" * 80)

    image_codec = ImageCodec(in_channels=3, embedding_dim=64)
    print(f"✓ Created Image Codec")
    print(f"  - Encoder: CNN with 4 layers")
    print(f"  - Decoder: CNN with 4 layers")
    print(f"  - Quantizer: FSQ with levels [8, 5, 5, 5]")

    # Create sample image
    image = MyImage(
        pixels=torch.randn(2, 3, 224, 224),
        metadata={"source": "test"}
    )
    print(f"  - Input shape: {image.pixels.shape}")

    # Encode to tokens
    tokens = image_codec.encode(image)
    print(f"  - Token shape: {tokens.shape}")
    print(f"  - Num tokens: {tokens.shape[1]}")

    # Decode back
    reconstructed = image_codec.decode(tokens, metadata={"source": "test"})
    print(f"  - Reconstructed shape: {reconstructed.pixels.shape}")

    # Compute reconstruction error
    recon_error = F.mse_loss(reconstructed.pixels, (image.pixels / 255.0 if image.pixels.max() > 1 else image.pixels))
    print(f"  - Reconstruction MSE: {recon_error.item():.6f}")

    # ========================================================================
    # Test 2: Time Series Codec
    # ========================================================================
    print("\n2. Time Series Codec")
    print("-" * 80)

    ts_codec = TimeSeriesCodec(embedding_dim=64)
    print(f"✓ Created Time Series Codec")
    print(f"  - Encoder: 1D CNN with 3 layers")
    print(f"  - Decoder: 1D CNN with 3 layers")
    print(f"  - Quantizer: VQ with 512 codes")

    # Create sample time series
    timeseries = MyTimeSeries(
        values=torch.randn(2, 1000),
        timestamps=torch.linspace(0, 10, 1000),
        mask=torch.ones(2, 1000, dtype=torch.bool)
    )
    print(f"  - Input shape: {timeseries.values.shape}")

    # Encode to tokens
    tokens = ts_codec.encode(timeseries)
    print(f"  - Token shape: {tokens.shape}")
    print(f"  - Num tokens: {tokens.shape[1]}")

    # Decode back
    reconstructed = ts_codec.decode(
        tokens,
        timestamps=timeseries.timestamps,
        mask=timeseries.mask
    )
    print(f"  - Reconstructed shape: {reconstructed.values.shape}")

    recon_error = F.mse_loss(reconstructed.values, timeseries.values)
    print(f"  - Reconstruction MSE: {recon_error.item():.6f}")

    # ========================================================================
    # Test 3: Scalar Codec
    # ========================================================================
    print("\n3. Scalar Codec")
    print("-" * 80)

    scalar_codec = ScalarCodec(num_bins=256, min_val=0.0, max_val=100.0)
    print(f"✓ Created Scalar Codec")
    print(f"  - Encoder: Identity (no learning)")
    print(f"  - Decoder: Identity (no learning)")
    print(f"  - Quantizer: Linear with 256 bins")

    # Create sample scalars
    scalar = MyScalar(
        value=torch.tensor([[23.5], [67.2], [89.1], [12.4]]),
        name="temperature"
    )
    print(f"  - Input shape: {scalar.value.shape}")
    print(f"  - Input values: {scalar.value.squeeze().tolist()}")

    # Encode to tokens
    tokens = scalar_codec.encode(scalar)
    print(f"  - Token shape: {tokens.shape}")
    print(f"  - Token values (bins): {tokens.squeeze().tolist()}")

    # Decode back
    reconstructed = scalar_codec.decode(tokens, name="temperature")
    print(f"  - Reconstructed values: {reconstructed.value.squeeze().tolist()}")

    recon_error = F.mse_loss(reconstructed.value, scalar.value)
    print(f"  - Reconstruction MSE: {recon_error.item():.6f}")

    # ========================================================================
    # Test 4: Tabular Codec
    # ========================================================================
    print("\n4. Tabular Codec")
    print("-" * 80)

    tabular_codec = TabularCodec(num_features=32, embedding_dim=64)
    print(f"✓ Created Tabular Codec")
    print(f"  - Encoder: MLP with 3 layers")
    print(f"  - Decoder: MLP with 3 layers")
    print(f"  - Quantizer: VQ with 256 codes")

    # Create sample tabular data
    tabular = MyTabular(
        features=torch.randn(2, 32),
        feature_names=[f"feature_{i}" for i in range(32)],
        categorical_mask=torch.zeros(32, dtype=torch.bool)
    )
    print(f"  - Input shape: {tabular.features.shape}")

    # Encode to tokens
    tokens = tabular_codec.encode(tabular)
    print(f"  - Token shape: {tokens.shape}")

    # Decode back
    reconstructed = tabular_codec.decode(
        tokens,
        feature_names=tabular.feature_names,
        categorical_mask=tabular.categorical_mask
    )
    print(f"  - Reconstructed shape: {reconstructed.features.shape}")

    recon_error = F.mse_loss(reconstructed.features, tabular.features)
    print(f"  - Reconstruction MSE: {recon_error.item():.6f}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("✓ Step 3 Complete: Codecs Implemented")
    print("=" * 80)
    print("\nCodec Architecture Summary:")
    print("  • Image Codec: CNN encoder/decoder + FSQ")
    print("  • Time Series Codec: 1D CNN + VQ")
    print("  • Scalar Codec: Identity mapping + Linear quantizer")
    print("  • Tabular Codec: MLP encoder/decoder + VQ")

    print("\n" + "=" * 80)
    print("Next Steps:")
    print("  → Step 4: Create codec manager for batched operations")
    print("  → Step 5: Train codecs on reconstruction task")
    print("=" * 80)
