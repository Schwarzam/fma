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
"""

import torch
import torch.nn as nn
from typing import Dict, Type, List
from functools import lru_cache

# Import from previous steps
from step1_define_modalities import BaseModality, MyImage, MyTimeSeries, MyScalar, MyTabular
from step3_implement_codecs import Codec, ImageCodec, TimeSeriesCodec, ScalarCodec, TabularCodec


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

    def _get_codec_class(self, modality_type: Type[BaseModality]) -> Type[Codec]:
        """
        Get codec class for a given modality type.

        Args:
            modality_type: Modality class

        Returns:
            Codec class

        Raises:
            ModalityTypeError: If modality type is not registered
        """
        if modality_type not in MODALITY_CODEC_MAPPING:
            raise ModalityTypeError(
                f"No codec registered for modality type: {modality_type.__name__}. "
                f"Available modalities: {list(MODALITY_CODEC_MAPPING.keys())}"
            )
        return MODALITY_CODEC_MAPPING[modality_type]

    def _load_codec(self, modality_type: Type[BaseModality]) -> Codec:
        """
        Load codec for a modality type (with caching).

        Args:
            modality_type: Modality class

        Returns:
            Initialized codec on correct device
        """
        # Check cache first
        if modality_type in self._codec_cache:
            return self._codec_cache[modality_type]

        # Get codec class
        codec_class = self._get_codec_class(modality_type)

        # Get kwargs for this codec (if any)
        kwargs = self.codec_kwargs.get(modality_type, {})

        # Initialize codec
        codec = codec_class(**kwargs)

        # Move to device
        codec = codec.to(self.device)
        codec.eval()  # Set to eval mode by default

        # Cache it
        self._codec_cache[modality_type] = codec

        return codec

    def encode(self, *modalities: BaseModality) -> Dict[str, torch.Tensor]:
        """
        Encode multiple modalities to discrete tokens.

        Args:
            *modalities: Variable number of modality instances

        Returns:
            Dictionary mapping token_key -> tokens tensor
            Format: {"tok_my_image": [batch, 784], "tok_scalar": [batch, 1], ...}

        Example:
            >>> manager = CodecManager()
            >>> image = MyImage(pixels=torch.randn(4, 3, 224, 224))
            >>> scalar = MyScalar(value=torch.tensor([[23.5], [25.1], [22.8], [24.3]]))
            >>> tokens = manager.encode(image, scalar)
            >>> print(tokens.keys())  # dict_keys(['tok_my_image', 'tok_scalar'])
        """
        if len(modalities) == 0:
            raise ValueError("Must provide at least one modality to encode")

        tokens = {}

        for modality in modalities:
            # Get modality type
            modality_type = type(modality)

            # Load codec
            codec = self._load_codec(modality_type)

            # Move modality data to device
            modality = self._to_device(modality)

            # Encode
            with torch.no_grad():  # No gradients for encoding
                tokenized = codec.encode(modality)

            # Store with token_key
            token_key = modality_type.token_key
            tokens[token_key] = tokenized

        return tokens

    def decode(
        self,
        tokens: Dict[str, torch.Tensor],
        modality_type: Type[BaseModality],
        **metadata
    ) -> BaseModality:
        """
        Decode tokens back to a specific modality.

        Args:
            tokens: Dictionary of tokens (from encode())
            modality_type: Which modality to reconstruct
            **metadata: Optional metadata for reconstruction
                       (e.g., bands for images, timestamps for time series)

        Returns:
            Reconstructed modality instance

        Raises:
            TokenKeyError: If required token_key not in tokens dict

        Example:
            >>> tokens = {"tok_my_image": torch.randn(4, 784)}
            >>> image = manager.decode(tokens, MyImage)
        """
        # Get token key for this modality
        token_key = modality_type.token_key

        # Check if token_key exists
        if token_key not in tokens:
            raise TokenKeyError(
                f"Token key '{token_key}' not found in tokens dict. "
                f"Available keys: {list(tokens.keys())}"
            )

        # Load codec
        codec = self._load_codec(modality_type)

        # Get tokens
        token_tensor = tokens[token_key].to(self.device)

        # Decode
        with torch.no_grad():
            decoded_modality = codec.decode(token_tensor, **metadata)

        return decoded_modality

    def encode_batch(
        self,
        modality_batches: Dict[Type[BaseModality], List[BaseModality]]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode batches of modalities of the same type.

        Useful when you have multiple samples and want to batch process them.

        Args:
            modality_batches: Dict mapping modality type to list of instances

        Returns:
            Dictionary of concatenated tokens

        Example:
            >>> batches = {
            ...     MyImage: [image1, image2, image3],
            ...     MyScalar: [scalar1, scalar2, scalar3]
            ... }
            >>> tokens = manager.encode_batch(batches)
        """
        all_tokens = {}

        for modality_type, modality_list in modality_batches.items():
            if len(modality_list) == 0:
                continue

            # Encode all modalities of this type
            token_list = []
            for modality in modality_list:
                result = self.encode(modality)
                token_key = modality_type.token_key
                token_list.append(result[token_key])

            # Concatenate along batch dimension
            token_key = modality_type.token_key
            all_tokens[token_key] = torch.cat(token_list, dim=0)

        return all_tokens

    def _to_device(self, modality: BaseModality) -> BaseModality:
        """
        Move modality tensors to the correct device.

        Args:
            modality: Modality instance

        Returns:
            Modality with tensors on correct device
        """
        # This is a simplified version - in practice you'd need to handle
        # each modality type's specific tensor fields
        if isinstance(modality, MyImage):
            return MyImage(
                pixels=modality.pixels.to(self.device),
                metadata=modality.metadata
            )
        elif isinstance(modality, MyTimeSeries):
            return MyTimeSeries(
                values=modality.values.to(self.device),
                timestamps=modality.timestamps.to(self.device) if modality.timestamps is not None else None,
                mask=modality.mask.to(self.device) if modality.mask is not None else None
            )
        elif isinstance(modality, MyScalar):
            return MyScalar(
                value=modality.value.to(self.device),
                name=modality.name
            )
        elif isinstance(modality, MyTabular):
            return MyTabular(
                features=modality.features.to(self.device),
                feature_names=modality.feature_names,
                categorical_mask=modality.categorical_mask.to(self.device) if modality.categorical_mask is not None else None
            )
        else:
            return modality

    def list_available_modalities(self) -> List[str]:
        """Return list of available modality names"""
        return [mod.__name__ for mod in MODALITY_CODEC_MAPPING.keys()]

    def clear_cache(self):
        """Clear codec cache (useful for memory management)"""
        self._codec_cache.clear()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("STEP 4: Codec Manager")
    print("=" * 80)

    # ========================================================================
    # Initialize Manager
    # ========================================================================
    print("\n1. Initialize Codec Manager")
    print("-" * 80)

    manager = CodecManager(device="cpu")
    print(f"✓ Created Codec Manager on device: {manager.device}")
    print(f"  Available modalities: {manager.list_available_modalities()}")

    # ========================================================================
    # Test 1: Encode Multiple Modalities
    # ========================================================================
    print("\n2. Encode Multiple Modalities")
    print("-" * 80)

    # Create sample modalities
    image = MyImage(
        pixels=torch.randn(2, 3, 224, 224),
        metadata={"source": "camera_01"}
    )

    timeseries = MyTimeSeries(
        values=torch.randn(2, 1000),
        timestamps=torch.linspace(0, 10, 1000),
        mask=torch.ones(2, 1000, dtype=torch.bool)
    )

    scalar = MyScalar(
        value=torch.tensor([[23.5], [67.2]]),
        name="temperature"
    )

    tabular = MyTabular(
        features=torch.randn(2, 32),
        feature_names=[f"feature_{i}" for i in range(32)]
    )

    print(f"✓ Created 4 modalities")
    print(f"  - Image: {image.pixels.shape}")
    print(f"  - TimeSeries: {timeseries.values.shape}")
    print(f"  - Scalar: {scalar.value.shape}")
    print(f"  - Tabular: {tabular.features.shape}")

    # Encode all at once
    print("\n  Encoding all modalities...")
    tokens = manager.encode(image, timeseries, scalar, tabular)

    print(f"✓ Encoded to tokens:")
    for token_key, token_tensor in tokens.items():
        print(f"  - {token_key}: {token_tensor.shape}")

    # ========================================================================
    # Test 2: Decode Specific Modalities
    # ========================================================================
    print("\n3. Decode Specific Modalities")
    print("-" * 80)

    # Decode image
    print("  Decoding image...")
    reconstructed_image = manager.decode(tokens, MyImage, metadata={"source": "camera_01"})
    print(f"✓ Reconstructed image shape: {reconstructed_image.pixels.shape}")

    # Decode time series
    print("  Decoding time series...")
    reconstructed_ts = manager.decode(
        tokens,
        MyTimeSeries,
        timestamps=timeseries.timestamps,
        mask=timeseries.mask
    )
    print(f"✓ Reconstructed time series shape: {reconstructed_ts.values.shape}")

    # Decode scalar
    print("  Decoding scalar...")
    reconstructed_scalar = manager.decode(tokens, MyScalar, name="temperature")
    print(f"✓ Reconstructed scalar values: {reconstructed_scalar.value.squeeze().tolist()}")

    # Decode tabular
    print("  Decoding tabular...")
    reconstructed_tabular = manager.decode(
        tokens,
        MyTabular,
        feature_names=tabular.feature_names
    )
    print(f"✓ Reconstructed tabular shape: {reconstructed_tabular.features.shape}")

    # ========================================================================
    # Test 3: Reconstruction Quality
    # ========================================================================
    print("\n4. Reconstruction Quality")
    print("-" * 80)

    import torch.nn.functional as F

    # Image reconstruction error
    image_mse = F.mse_loss(
        reconstructed_image.pixels,
        (image.pixels / 255.0 if image.pixels.max() > 1 else image.pixels)
    )
    print(f"  Image MSE: {image_mse.item():.6f}")

    # Time series reconstruction error
    ts_mse = F.mse_loss(reconstructed_ts.values, timeseries.values)
    print(f"  Time Series MSE: {ts_mse.item():.6f}")

    # Scalar reconstruction error
    scalar_mse = F.mse_loss(reconstructed_scalar.value, scalar.value)
    print(f"  Scalar MSE: {scalar_mse.item():.6f}")

    # Tabular reconstruction error
    tabular_mse = F.mse_loss(reconstructed_tabular.features, tabular.features)
    print(f"  Tabular MSE: {tabular_mse.item():.6f}")

    # ========================================================================
    # Test 4: Token Key Routing
    # ========================================================================
    print("\n5. Token Key Routing")
    print("-" * 80)

    print("  Token keys act as routing identifiers:")
    for modality_type in [MyImage, MyTimeSeries, MyScalar, MyTabular]:
        print(f"  - {modality_type.__name__:15} → {modality_type.token_key}")

    print("\n  This allows flexible multimodal batching:")
    print("  - Encode any combination of modalities")
    print("  - Store all tokens in one dict")
    print("  - Decode specific modalities as needed")

    # ========================================================================
    # Test 5: Error Handling
    # ========================================================================
    print("\n6. Error Handling")
    print("-" * 80)

    # Try to decode with missing token key
    try:
        fake_tokens = {"tok_nonexistent": torch.randn(2, 100)}
        manager.decode(fake_tokens, MyImage)
    except TokenKeyError as e:
        print(f"✓ Caught TokenKeyError: {e}")

    # ========================================================================
    # Test 6: Batch Processing
    # ========================================================================
    print("\n7. Batch Processing")
    print("-" * 80)

    # Create multiple images
    images = [
        MyImage(pixels=torch.randn(1, 3, 224, 224), metadata={"id": i})
        for i in range(3)
    ]

    scalars = [
        MyScalar(value=torch.tensor([[float(i * 10)]]), name="measurement")
        for i in range(3)
    ]

    print(f"✓ Created 3 images and 3 scalars")

    # Encode in batch
    batch_tokens = manager.encode_batch({
        MyImage: images,
        MyScalar: scalars
    })

    print(f"✓ Batch encoded:")
    for token_key, token_tensor in batch_tokens.items():
        print(f"  - {token_key}: {token_tensor.shape}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("✓ Step 4 Complete: Codec Manager Implemented")
    print("=" * 80)
    print("\nCodec Manager Features:")
    print("  • Dynamic codec loading with caching")
    print("  • Multi-modality encoding in single call")
    print("  • Token-key based routing")
    print("  • Device management")
    print("  • Batch processing support")
    print("  • Error handling for missing modalities/tokens")

    print("\n" + "=" * 80)
    print("Next Steps:")
    print("  → Step 5: Train codecs on reconstruction task")
    print("  → Step 6: Train transformer on masked modeling task")
    print("=" * 80)
