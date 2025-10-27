"""
Step 4: Codec Manager

The Codec Manager orchestrates encoding and decoding for multiple modalities.

Key features:
- Dynamic codec loading based on modality type
- Batched encoding of multiple modalities at once
- Token routing using token_key
- Caching for efficiency

Usage:
    manager = CodecManager()

    # Encode multiple modalities
    tokens = manager.encode(image, timeseries, scalar)
    # Returns: {"tok_my_image": [...], "tok_timeseries": [...], "tok_scalar": [...]}

    # Decode specific modality
    reconstructed_image = manager.decode(tokens, MyImage)

    # Inference convenience:
    # You can also pass a tensor of tokens instead of a dict:
    reconstructed_image = manager.decode(pred_tokens, MyImage)
"""

from typing import Dict, Type, List, Union
import torch
import torch.nn as nn

# Import from previous steps (base/example modalities & codecs)
from core.define_modalities import BaseModality, MyImage, MyTimeSeries, MyScalar, MyTabular
from core.implement_codecs import Codec, ImageCodec, TimeSeriesCodec, ScalarCodec, TabularCodec

# Try to import astronomical modalities/codecs (if present)
_HAS_ASTRO = False
try:
    from astronomical_dataset import (
        GalaxyImage, GaiaSpectrum, ZTFLightCurve, Redshift, StellarMass, StarFormationRate
    )
    from astronomical_codecs import (
        GalaxyImageCodec, GaiaSpectrumCodec, ZTFLightCurveCodec,
        RedshiftCodec, StellarMassCodec, StarFormationRateCodec
    )
    _HAS_ASTRO = True
except Exception:
    _HAS_ASTRO = False


# ============================================================================
# Custom Exceptions
# ============================================================================

class ModalityTypeError(Exception):
    """Raised when an invalid modality type is provided"""
    pass


class TokenKeyError(Exception):
    """Raised when a required token key is missing"""
    pass


# ============================================================================
# Codec Registry
# ============================================================================

# Mapping from modality class to codec class
MODALITY_CODEC_MAPPING: Dict[Type[BaseModality], Type[Codec]] = {
    MyImage: ImageCodec,
    MyTimeSeries: TimeSeriesCodec,
    MyScalar: ScalarCodec,
    MyTabular: TabularCodec,
}

# Register astronomical codecs if available
if _HAS_ASTRO:
    MODALITY_CODEC_MAPPING.update({
        GalaxyImage: GalaxyImageCodec,
        GaiaSpectrum: GaiaSpectrumCodec,
        ZTFLightCurve: ZTFLightCurveCodec,
        Redshift: RedshiftCodec,
        StellarMass: StellarMassCodec,
        StarFormationRate: StarFormationRateCodec,
    })


# ============================================================================
# Codec Manager
# ============================================================================

class CodecManager:
    """
    Manages encoding and decoding of multiple modalities.

    Handles:
    - Dynamic codec loading
    - Batched encoding/decoding
    - Device management
    - Caching for efficiency
    """

    def __init__(self, device: str = "cpu", codec_kwargs: Dict = None):
        """
        Initialize Codec Manager.

        Args:
            device: Device to run codecs on ("cpu", "cuda", "mps")
            codec_kwargs: Optional kwargs for codec initialization
                         Format: {modality_class: {kwarg: value}}
        """
        self.device = device
        self.codec_kwargs = codec_kwargs or {}
        self._codec_cache: Dict[Type[BaseModality], Codec] = {}

    # ------------------------------ internal ------------------------------

    def _get_codec_class(self, modality_type: Type[BaseModality]) -> Type[Codec]:
        """
        Get codec class for a given modality type.
        """
        if modality_type not in MODALITY_CODEC_MAPPING:
            available = ", ".join(cls.__name__ for cls in MODALITY_CODEC_MAPPING.keys())
            raise ModalityTypeError(
                f"No codec registered for modality type: {modality_type}. "
                f"Available modalities: [{available}]"
            )
        return MODALITY_CODEC_MAPPING[modality_type]

    def _load_codec(self, modality_type: Type[BaseModality]) -> Codec:
        """
        Load codec for a modality type (with caching).
        """
        if modality_type in self._codec_cache:
            return self._codec_cache[modality_type]

        codec_class = self._get_codec_class(modality_type)
        kwargs = self.codec_kwargs.get(modality_type, {})
        codec = codec_class(**kwargs).to(self.device).eval()

        # Load trained weights for GalaxyImageCodec
        if modality_type.__name__ == 'GalaxyImage':
            import os
            checkpoint_path = 'image_codec_final.pt'
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    codec.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print(f"✓ Loaded trained image codec from {checkpoint_path}")
                except Exception as e:
                    print(f"⚠ Could not load trained codec: {e}")
            else:
                print(f"⚠ Trained codec not found at {checkpoint_path}, using random init")

        self._codec_cache[modality_type] = codec
        return codec

    def _move_to_device(self, modality: BaseModality) -> BaseModality:
        """
        Move modality tensors to the correct device.
        Handles both example (My*) and astronomical modalities.
        """
        # Example modalities
        if isinstance(modality, MyImage):
            return MyImage(
                pixels=modality.pixels.to(self.device),
                metadata=getattr(modality, "metadata", None),
            )
        if isinstance(modality, MyTimeSeries):
            return MyTimeSeries(
                values=modality.values.to(self.device),
                timestamps=modality.timestamps.to(self.device) if modality.timestamps is not None else None,
                mask=modality.mask.to(self.device) if modality.mask is not None else None,
            )
        if isinstance(modality, MyScalar):
            return MyScalar(
                value=modality.value.to(self.device),
                name=getattr(modality, "name", None),
            )
        if isinstance(modality, MyTabular):
            return MyTabular(
                features=modality.features.to(self.device),
                feature_names=getattr(modality, "feature_names", None),
                categorical_mask=modality.categorical_mask.to(self.device) if getattr(modality, "categorical_mask", None) is not None else None,
            )

        # Astronomical modalities (if available)
        if _HAS_ASTRO:
            if isinstance(modality, GalaxyImage):
                return GalaxyImage(flux=modality.flux.to(self.device), metadata=getattr(modality, "metadata", None))
            if isinstance(modality, GaiaSpectrum):
                return GaiaSpectrum(
                    flux=modality.flux.to(self.device),
                    ivar=modality.ivar.to(self.device),
                    mask=modality.mask.to(self.device),
                    wavelength=modality.wavelength.to(self.device),
                    metadata=getattr(modality, "metadata", None),
                )
            if isinstance(modality, ZTFLightCurve):
                return ZTFLightCurve(
                    flux=modality.flux.to(self.device),
                    flux_err=modality.flux_err.to(self.device),
                    mjd=modality.mjd.to(self.device),
                    metadata=getattr(modality, "metadata", None),
                )
            if isinstance(modality, Redshift):
                return Redshift(value=modality.value.to(self.device), metadata=getattr(modality, "metadata", None))
            if isinstance(modality, StellarMass):
                return StellarMass(value=modality.value.to(self.device), metadata=getattr(modality, "metadata", None))
            if isinstance(modality, StarFormationRate):
                return StarFormationRate(value=modality.value.to(self.device), metadata=getattr(modality, "metadata", None))

        # Default: return as-is
        return modality

    # ------------------------------ public API ----------------------------

    def encode(self, *modalities: BaseModality) -> Dict[str, torch.Tensor]:
        """
        Encode multiple modalities to discrete tokens.

        Returns:
            Dict[token_key, tokens] with shapes [B, T] (or [B, 1] for scalars)
        """
        if len(modalities) == 0:
            raise ValueError("Must provide at least one modality to encode")

        tokens: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for modality in modalities:
                modality_type = type(modality)
                codec = self._load_codec(modality_type)
                modality_on_dev = self._move_to_device(modality)
                tokenized = codec.encode(modality_on_dev)   # expected [B, T] long/int tensor
                token_key = modality_type.token_key
                tokens[token_key] = tokenized
        return tokens

    def decode(
        self,
        tokens_or_tensor: Union[Dict[str, torch.Tensor], torch.Tensor],
        modality_type: Type[BaseModality],
        **metadata
    ) -> BaseModality:
        """
        Decode tokens back to a specific modality.

        Accepts either:
          • a dict of tokens (from encode()), OR
          • a single tensor of tokens for the target modality.

        If an integer tensor of token IDs is given, this will convert to a
        continuous latent (float) using the codec's available helpers before
        calling the decoder.
        """
        codec = self._load_codec(modality_type)
        token_key = modality_type.token_key

        # 1) Resolve the token tensor for this modality
        if isinstance(tokens_or_tensor, dict):
            if token_key not in tokens_or_tensor:
                available = ", ".join(tokens_or_tensor.keys())
                raise TokenKeyError(
                    f"Token key '{token_key}' not found in tokens dict. Available: [{available}]"
                )
            token_tensor = tokens_or_tensor[token_key]
        else:
            token_tensor = tokens_or_tensor

        token_tensor = token_tensor.to(self.device)

        # 2) Call codec.decode() directly - it will handle quantizer decode internally
        # The Codec base class handles the full pipeline: tokens -> quantizer.decode() -> _decode()
        with torch.no_grad():
            decoded_modality = codec.decode(token_tensor, **metadata)

        return decoded_modality

    def encode_batch(
        self,
        modality_batches: Dict[Type[BaseModality], List[BaseModality]]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode batches of modalities of the same type.
        """
        all_tokens: Dict[str, torch.Tensor] = {}
        for modality_type, modality_list in modality_batches.items():
            if len(modality_list) == 0:
                continue
            token_key = modality_type.token_key
            tok_list = []
            for m in modality_list:
                out = self.encode(m)  # dict with a single key
                tok_list.append(out[token_key])
            all_tokens[token_key] = torch.cat(tok_list, dim=0)
        return all_tokens

    def list_available_modalities(self) -> List[str]:
        """Return list of available modality class names"""
        return [mod.__name__ for mod in MODALITY_CODEC_MAPPING.keys()]

    def clear_cache(self):
        """Clear codec cache (useful for memory management)"""
        self._codec_cache.clear()


# ============================================================================
# Example Self-Test (optional)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("STEP 4: Codec Manager (quick self-test)")
    print("=" * 80)

    mgr = CodecManager(device="cpu")
    print(f"✓ Device: {mgr.device}")
    print(f"Available modalities: {mgr.list_available_modalities()}")