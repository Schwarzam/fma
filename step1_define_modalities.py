"""
Step 1: Define Modalities

This is the first step in building a tokenizer system like AION.
We define the data structures (modalities) that our model will work with.

Each modality is a dataclass that specifies:
- token_key: Unique identifier for routing in the model
- num_tokens: Expected number of discrete tokens after tokenization
- Data fields: The actual data (tensors, metadata, etc.)
"""

from dataclasses import dataclass
from typing import ClassVar
from jaxtyping import Float, Int, Bool
import torch
from torch import Tensor


# ============================================================================
# Base Modality Class
# ============================================================================

@dataclass
class BaseModality:
    """
    Abstract base class for all modalities.

    Each modality must define:
    - token_key: Unique string identifier
    - num_tokens: Number of discrete tokens after encoding
    """
    token_key: ClassVar[str]
    num_tokens: ClassVar[int]


# ============================================================================
# Image Modality
# ============================================================================

@dataclass
class MyImage(BaseModality):
    """
    Image modality for visual data.

    Example use case: Medical images, satellite imagery, photographs

    Attributes:
        pixels: Image tensor [batch, channels, height, width]
        metadata: Optional metadata (image_id, timestamp, etc.)
    """
    token_key: ClassVar[str] = "tok_my_image"
    num_tokens: ClassVar[int] = 256  # 16x16 spatial tokens

    pixels: Float[Tensor, "batch channels height width"]
    metadata: dict = None

    def __post_init__(self):
        """Validate image dimensions"""
        assert self.pixels.ndim == 4, f"Expected 4D tensor, got {self.pixels.ndim}D"
        assert self.pixels.shape[1] in [1, 3, 4], f"Expected 1, 3, or 4 channels, got {self.pixels.shape[1]}"


# ============================================================================
# Time Series Modality
# ============================================================================

@dataclass
class MyTimeSeries(BaseModality):
    """
    Time series modality for sequential data.

    Example use case: Stock prices, sensor readings, EEG signals

    Attributes:
        values: Time series values [batch, timesteps]
        timestamps: Optional timestamp values [timesteps]
        mask: Optional mask for missing values [batch, timesteps]
    """
    token_key: ClassVar[str] = "tok_timeseries"
    num_tokens: ClassVar[int] = 128  # 128 temporal tokens

    values: Float[Tensor, "batch timesteps"]
    timestamps: Float[Tensor, "timesteps"] = None
    mask: Bool[Tensor, "batch timesteps"] = None

    def __post_init__(self):
        """Validate time series dimensions"""
        assert self.values.ndim == 2, f"Expected 2D tensor, got {self.values.ndim}D"

        if self.mask is not None:
            assert self.mask.shape == self.values.shape, \
                f"Mask shape {self.mask.shape} doesn't match values shape {self.values.shape}"


# ============================================================================
# Scalar Modality
# ============================================================================

@dataclass
class MyScalar(BaseModality):
    """
    Scalar modality for single-value measurements.

    Example use case: Temperature, age, price, measurement

    Attributes:
        value: Scalar value [batch, 1]
        name: Name of the scalar (e.g., "temperature", "age")
    """
    token_key: ClassVar[str] = "tok_scalar"
    num_tokens: ClassVar[int] = 1  # Single token

    value: Float[Tensor, "batch 1"]
    name: str = "unnamed_scalar"

    def __post_init__(self):
        """Validate scalar dimensions"""
        assert self.value.ndim == 2, f"Expected 2D tensor [batch, 1], got {self.value.ndim}D"
        assert self.value.shape[1] == 1, f"Expected single value, got {self.value.shape[1]} values"


# ============================================================================
# Text Modality (Token IDs)
# ============================================================================

@dataclass
class MyText(BaseModality):
    """
    Text modality (already tokenized).

    Example use case: Captions, descriptions, documents

    Attributes:
        token_ids: Token IDs from existing tokenizer [batch, seq_len]
        attention_mask: Mask for padding [batch, seq_len]
        vocab_size: Size of token vocabulary
    """
    token_key: ClassVar[str] = "tok_text"
    num_tokens: ClassVar[int] = 64  # 64 text tokens

    token_ids: Int[Tensor, "batch seq_len"]
    attention_mask: Bool[Tensor, "batch seq_len"] = None
    vocab_size: int = 50000

    def __post_init__(self):
        """Validate text dimensions"""
        assert self.token_ids.ndim == 2, f"Expected 2D tensor, got {self.token_ids.ndim}D"

        if self.attention_mask is not None:
            assert self.attention_mask.shape == self.token_ids.shape, \
                f"Attention mask shape {self.attention_mask.shape} doesn't match token_ids shape {self.token_ids.shape}"


# ============================================================================
# Tabular/Multi-Scalar Modality
# ============================================================================

@dataclass
class MyTabular(BaseModality):
    """
    Tabular modality for structured data with multiple features.

    Example use case: Patient records, financial data, survey responses

    Attributes:
        features: Feature tensor [batch, num_features]
        feature_names: List of feature names
        categorical_mask: Boolean mask indicating categorical features [num_features]
    """
    token_key: ClassVar[str] = "tok_tabular"
    num_tokens: ClassVar[int] = 32  # One token per feature (example with 32 features)

    features: Float[Tensor, "batch num_features"]
    feature_names: list[str] = None
    categorical_mask: Bool[Tensor, "num_features"] = None

    def __post_init__(self):
        """Validate tabular dimensions"""
        assert self.features.ndim == 2, f"Expected 2D tensor, got {self.features.ndim}D"

        if self.feature_names is not None:
            assert len(self.feature_names) == self.features.shape[1], \
                f"Number of feature names {len(self.feature_names)} doesn't match features {self.features.shape[1]}"


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("STEP 1: Defining Modalities")
    print("=" * 80)

    # Example 1: Create an image modality
    print("\n1. Creating Image Modality")
    print("-" * 80)
    image = MyImage(
        pixels=torch.randn(4, 3, 224, 224),  # 4 RGB images of 224x224
        metadata={"source": "camera_01", "timestamp": "2024-01-15"}
    )
    print(f"✓ Created image modality")
    print(f"  - Shape: {image.pixels.shape}")
    print(f"  - Token key: {image.token_key}")
    print(f"  - Expected tokens: {image.num_tokens}")
    print(f"  - Metadata: {image.metadata}")

    # Example 2: Create a time series modality
    print("\n2. Creating Time Series Modality")
    print("-" * 80)
    timeseries = MyTimeSeries(
        values=torch.randn(4, 1000),  # 4 time series with 1000 timesteps
        timestamps=torch.linspace(0, 10, 1000),  # 0 to 10 seconds
        mask=torch.ones(4, 1000, dtype=torch.bool)  # All valid
    )
    print(f"✓ Created time series modality")
    print(f"  - Shape: {timeseries.values.shape}")
    print(f"  - Token key: {timeseries.token_key}")
    print(f"  - Expected tokens: {timeseries.num_tokens}")
    print(f"  - Time range: [{timeseries.timestamps[0]:.2f}, {timeseries.timestamps[-1]:.2f}]")

    # Example 3: Create a scalar modality
    print("\n3. Creating Scalar Modality")
    print("-" * 80)
    temperature = MyScalar(
        value=torch.tensor([[23.5], [25.1], [22.8], [24.3]]),  # 4 temperature readings
        name="temperature_celsius"
    )
    print(f"✓ Created scalar modality")
    print(f"  - Shape: {temperature.value.shape}")
    print(f"  - Token key: {temperature.token_key}")
    print(f"  - Expected tokens: {temperature.num_tokens}")
    print(f"  - Name: {temperature.name}")
    print(f"  - Values: {temperature.value.squeeze().tolist()}")

    # Example 4: Create a text modality
    print("\n4. Creating Text Modality")
    print("-" * 80)
    text = MyText(
        token_ids=torch.randint(0, 50000, (4, 64)),  # 4 sequences of 64 tokens
        attention_mask=torch.ones(4, 64, dtype=torch.bool),
        vocab_size=50000
    )
    print(f"✓ Created text modality")
    print(f"  - Shape: {text.token_ids.shape}")
    print(f"  - Token key: {text.token_key}")
    print(f"  - Expected tokens: {text.num_tokens}")
    print(f"  - Vocab size: {text.vocab_size}")

    # Example 5: Create a tabular modality
    print("\n5. Creating Tabular Modality")
    print("-" * 80)
    tabular = MyTabular(
        features=torch.randn(4, 32),  # 4 samples with 32 features
        feature_names=[f"feature_{i}" for i in range(32)],
        categorical_mask=torch.tensor([i % 5 == 0 for i in range(32)])  # Every 5th is categorical
    )
    print(f"✓ Created tabular modality")
    print(f"  - Shape: {tabular.features.shape}")
    print(f"  - Token key: {tabular.token_key}")
    print(f"  - Expected tokens: {tabular.num_tokens}")
    print(f"  - Num features: {len(tabular.feature_names)}")
    print(f"  - Categorical features: {tabular.categorical_mask.sum().item()}")

    print("\n" + "=" * 80)
    print("✓ Step 1 Complete: Modalities Defined")
    print("=" * 80)
    print("\nNext Steps:")
    print("  → Step 2: Define quantizers for each modality type")
    print("  → Step 3: Implement codecs (encoder + decoder + quantizer)")
    print("  → Step 4: Create codec manager for batched operations")
    print("  → Step 5: Train codecs on reconstruction task")
