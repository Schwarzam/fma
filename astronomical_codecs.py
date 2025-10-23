"""
astronomical_codecs.py - Codecs for astronomical modalities

Creates codecs that work with the astronomical modality types.
"""

import torch
import torch.nn as nn
from typing import Type

from step3_implement_codecs import Codec, SimpleImageEncoder, SimpleImageDecoder
from step3_implement_codecs import TimeSeriesEncoder, TimeSeriesDecoder
from step2_define_quantizers import FiniteScalarQuantizer, VectorQuantizer, ScalarLinearQuantizer
from astronomical_dataset import (
    GalaxyImage, GaiaSpectrum, ZTFLightCurve,
    Redshift, StellarMass, StarFormationRate
)


# ============================================================================
# Galaxy Image Codec
# ============================================================================

class GalaxyImageCodec(Codec):
    """Codec for galaxy images"""

    def __init__(self, in_channels: int = 3, embedding_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.encoder = SimpleImageEncoder(in_channels, embedding_dim)
        self.decoder = SimpleImageDecoder(embedding_dim, in_channels)
        self.projection = nn.Conv2d(embedding_dim, 4, kernel_size=1)
        self.unprojection = nn.Conv2d(4, embedding_dim, kernel_size=1)
        self._quantizer = FiniteScalarQuantizer(levels=[8, 5, 5, 5])

    @property
    def modality(self) -> Type:
        return GalaxyImage

    @property
    def quantizer(self):
        return self._quantizer

    def _encode(self, x: GalaxyImage):
        pixels = x.flux
        if pixels.max() > 1.0:
            pixels = pixels / 255.0
        pixels = pixels * 2.0 - 1.0

        z = self.encoder(pixels)
        z = self.projection(z)
        # Scale up and normalize to [-1, 1] for FSQ
        # Without training, encoder outputs small values (~0.1)
        # Scale by 10x then tanh to spread across [-1, 1]
        z = torch.tanh(z * 10.0)
        return z

    def decode(self, z, **metadata):
        """
        Override decode to handle flat token sequence from transformer.

        Transformer outputs [B, 36] but FSQ needs [B, 6, 6] spatial shape.
        """
        # If z is flat [B, 36], reshape to [B, 6, 6] for FSQ
        if z.ndim == 2 and z.shape[1] == 36:
            B = z.shape[0]
            z = z.view(B, 6, 6)

        # Now call quantizer.decode to get [B, 4, 6, 6]
        z_continuous = self.quantizer.decode(z)

        # Call _decode to reconstruct image
        return self._decode(z_continuous, **metadata)

    def _decode(self, z, **metadata):
        # Handle both [B, 4, H, W] from training and [B, 4, H, W] from quantizer.decode()
        # The z here should already be [B, 4, H, W] from quantizer.decode()
        # But just in case it's flat [B, C*H*W], reshape it
        if z.ndim == 2:
            B, N = z.shape
            C = 4
            assert N % C == 0, f"Expected N={N} to be divisible by C={C}"
            HW = N // C
            H = W = int(HW ** 0.5)
            assert H * W == HW, f"Expected perfect square, got HW={HW}"
            z = z.view(B, C, H, W)

        z = self.unprojection(z)
        pixels = self.decoder(z)
        pixels = (pixels + 1.0) / 2.0
        return GalaxyImage(flux=pixels, metadata=metadata.get('metadata', None))


# ============================================================================
# Gaia Spectrum Codec
# ============================================================================

class GaiaSpectrumCodec(Codec):
    """Codec for Gaia BP/RP spectra"""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.encoder = TimeSeriesEncoder(embedding_dim)
        self.decoder = TimeSeriesDecoder(embedding_dim)
        self._quantizer = VectorQuantizer(num_embeddings=512, embedding_dim=embedding_dim)

    @property
    def modality(self) -> Type:
        return GaiaSpectrum

    @property
    def quantizer(self):
        return self._quantizer

    def _encode(self, x: GaiaSpectrum):
        values = x.flux.unsqueeze(1)  # [batch, 1, wavelength]
        z = self.encoder(values)
        return z

    def _decode(self, z, **metadata):
        B = z.size(0)

        if z.ndim == 2:
            E = self.embedding_dim
            N = z.size(1)
            assert N % E == 0
            T = N // E
            z = z.view(B, E, T)
        elif z.ndim == 3:
            if z.shape[1] != self.embedding_dim and z.shape[2] == self.embedding_dim:
                z = z.transpose(1, 2)

        out = self.decoder(z)
        if out.ndim == 3 and out.shape[1] != 1:
            out = out.mean(dim=1, keepdim=True)

        flux = out.squeeze(1)

        wavelength = metadata.get('wavelength', None)
        ivar = metadata.get('ivar', None)
        mask = metadata.get('mask', None)

        return GaiaSpectrum(
            flux=flux,
            ivar=ivar if ivar is not None else torch.ones_like(flux),
            mask=mask if mask is not None else torch.ones_like(flux, dtype=torch.bool),
            wavelength=wavelength if wavelength is not None else torch.arange(flux.shape[1]).float(),
            metadata=metadata.get('metadata', None)
        )


# ============================================================================
# ZTF Light Curve Codec
# ============================================================================

class ZTFLightCurveCodec(Codec):
    """Codec for ZTF light curves"""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.encoder = TimeSeriesEncoder(embedding_dim)
        self.decoder = TimeSeriesDecoder(embedding_dim)
        self._quantizer = VectorQuantizer(num_embeddings=512, embedding_dim=embedding_dim)

    @property
    def modality(self) -> Type:
        return ZTFLightCurve

    @property
    def quantizer(self):
        return self._quantizer

    def _encode(self, x: ZTFLightCurve):
        values = x.flux.unsqueeze(1)  # [batch, 1, time]
        z = self.encoder(values)
        return z

    def _decode(self, z, **metadata):
        B = z.size(0)

        if z.ndim == 2:
            E = self.embedding_dim
            N = z.size(1)
            assert N % E == 0
            T = N // E
            z = z.view(B, E, T)
        elif z.ndim == 3:
            if z.shape[1] != self.embedding_dim and z.shape[2] == self.embedding_dim:
                z = z.transpose(1, 2)

        out = self.decoder(z)
        if out.ndim == 3 and out.shape[1] != 1:
            out = out.mean(dim=1, keepdim=True)

        flux = out.squeeze(1)

        mjd = metadata.get('mjd', None)
        flux_err = metadata.get('flux_err', None)

        return ZTFLightCurve(
            flux=flux,
            flux_err=flux_err if flux_err is not None else torch.ones_like(flux) * 0.1,
            mjd=mjd if mjd is not None else torch.arange(flux.shape[1]).float(),
            metadata=metadata.get('metadata', None)
        )


# ============================================================================
# Scalar Codecs
# ============================================================================

class RedshiftCodec(Codec):
    """Codec for redshift (0-2 range)"""

    def __init__(self):
        super().__init__()
        self._quantizer = ScalarLinearQuantizer(num_bins=256, min_val=0.0, max_val=2.0)

    @property
    def modality(self) -> Type:
        return Redshift

    @property
    def quantizer(self):
        return self._quantizer

    def _encode(self, x: Redshift):
        return x.value

    def _decode(self, z, **metadata):
        return Redshift(value=z, metadata=metadata.get('metadata', None))


class StellarMassCodec(Codec):
    """Codec for stellar mass (log scale, 9-12 range)"""

    def __init__(self):
        super().__init__()
        self._quantizer = ScalarLinearQuantizer(num_bins=256, min_val=9.0, max_val=12.0)

    @property
    def modality(self) -> Type:
        return StellarMass

    @property
    def quantizer(self):
        return self._quantizer

    def _encode(self, x: StellarMass):
        return x.value

    def _decode(self, z, **metadata):
        return StellarMass(value=z, metadata=metadata.get('metadata', None))


class StarFormationRateCodec(Codec):
    """Codec for SFR (log scale, 0.1-100 range)"""

    def __init__(self):
        super().__init__()
        # Use log scale for SFR (large dynamic range)
        self._quantizer = ScalarLinearQuantizer(num_bins=256, min_val=-1.0, max_val=2.0)  # log10(0.1) to log10(100)

    @property
    def modality(self) -> Type:
        return StarFormationRate

    @property
    def quantizer(self):
        return self._quantizer

    def _encode(self, x: StarFormationRate):
        # Convert to log scale
        return torch.log10(x.value + 1e-10)

    def _decode(self, z, **metadata):
        # Convert back from log scale
        value = 10 ** z
        return StarFormationRate(value=value, metadata=metadata.get('metadata', None))
