# Model Inference & Output Guide

Understanding what your trained model outputs and how to use it for predictions.

---

## Table of Contents

1. [What the Model Outputs](#what-the-model-outputs)
2. [Inference Modes](#inference-modes)
3. [Complete Inference Pipeline](#complete-inference-pipeline)
4. [Use Cases](#use-cases)
5. [Advanced: Conditional Generation](#advanced-conditional-generation)

---

## What the Model Outputs

### During Training

The model outputs **discrete token predictions** for masked modalities:

```python
# Input: Some modalities visible, some masked
input_tokens = {
    "tok_image_legacy": [4, 256],      # Visible
    "tok_demographics": [4, 1],         # Visible
    "tok_lab_results": [4, 1],          # MASKED (model will predict)
    "tok_timeseries": [4, 128],         # MASKED (model will predict)
}

# Output: Logits for masked modalities
predictions = model(input_tokens, masked=["tok_lab_results", "tok_timeseries"])

# predictions = {
#     "tok_lab_results": [4, 1, 256],    # [batch, num_tokens, vocab_size]
#     "tok_timeseries": [4, 128, 512],   # [batch, num_tokens, vocab_size]
# }
```

**Each prediction is:**
- Shape: `[batch, num_tokens, vocab_size]`
- Logits (unnormalized scores) for each possible token
- Higher logit = more confident prediction

### Converting to Actual Data

To get actual modalities (images, time series, etc.), you need to:

1. **Sample tokens** from logits
2. **Decode tokens** back to continuous space (using codec)
3. **Reconstruct modality** from continuous space

---

## Inference Modes

### Mode 1: Predict Missing Modality

**Use case:** Patient has image + demographics, predict lab results

```python
"""
inference.py - Inference pipeline
"""

import torch
from step4_codec_manager import CodecManager
from step6_train_transformer import MultimodalTransformer
from step1_define_modalities import *


class MultimodalPredictor:
    """Inference wrapper for trained model"""

    def __init__(
        self,
        model: MultimodalTransformer,
        codec_manager: CodecManager,
        device: str = "cuda"
    ):
        self.model = model.to(device).eval()
        self.codec_manager = codec_manager
        self.device = device

    @torch.no_grad()
    def predict_missing(
        self,
        observed_modalities: dict,
        target_modality_type: type,
        **decode_kwargs
    ):
        """
        Predict a missing modality from observed ones.

        Args:
            observed_modalities: Dict of observed modality instances
                                 e.g., {'image': MyImage(...), 'demographics': PatientDemographics(...)}
            target_modality_type: Type of modality to predict (e.g., LabResults)
            **decode_kwargs: Metadata needed for decoding (e.g., timestamps, bands)

        Returns:
            Predicted modality instance
        """

        # 1. Encode observed modalities
        observed_tokens = {}
        for name, modality in observed_modalities.items():
            tokens = self.codec_manager.encode(modality)
            observed_tokens.update(tokens)

        # Move to device
        observed_tokens = {k: v.to(self.device) for k, v in observed_tokens.items()}

        # 2. Get target token key
        target_token_key = target_modality_type.token_key

        # 3. Forward pass - predict target
        predictions = self.model(observed_tokens, masked=[target_token_key])

        # predictions[target_token_key] has shape [batch, num_tokens, vocab_size]

        # 4. Sample tokens from predictions
        pred_logits = predictions[target_token_key]  # [B, N, V]

        # Greedy sampling (take argmax)
        pred_tokens = pred_logits.argmax(dim=-1)  # [B, N]

        # 5. Decode tokens back to modality
        predicted_modality = self.codec_manager.decode(
            {target_token_key: pred_tokens},
            target_modality_type,
            **decode_kwargs
        )

        return predicted_modality


# Example usage
if __name__ == "__main__":
    # Load trained model
    model = MultimodalTransformer(...)
    model.load_state_dict(torch.load("checkpoints/best_model.pt")['model_state_dict'])

    codec_manager = CodecManager(device="cuda")

    predictor = MultimodalPredictor(model, codec_manager, device="cuda")

    # Observed data
    image = MyImage(
        pixels=torch.randn(1, 3, 224, 224),  # [1, C, H, W]
        metadata={}
    )

    demographics = PatientDemographics(
        age=torch.tensor([[65.0]]),
        sex=torch.tensor([[1.0]]),  # Male
        bmi=torch.tensor([[28.5]]),
        metadata={}
    )

    # Predict lab results
    predicted_labs = predictor.predict_missing(
        observed_modalities={'image': image, 'demographics': demographics},
        target_modality_type=LabResults
    )

    print(f"Predicted glucose: {predicted_labs.glucose.item():.1f}")
    print(f"Predicted cholesterol: {predicted_labs.cholesterol.item():.1f}")
    print(f"Predicted hemoglobin: {predicted_labs.hemoglobin.item():.1f}")
```

### Mode 2: Predict Multiple Missing Modalities

**Use case:** Given image, predict everything else

```python
class MultimodalPredictor:
    # ... previous code ...

    @torch.no_grad()
    def predict_multiple(
        self,
        observed_modalities: dict,
        target_modality_types: list,
        decode_kwargs: dict = None
    ):
        """
        Predict multiple missing modalities at once.

        Args:
            observed_modalities: Dict of observed modalities
            target_modality_types: List of modality types to predict
            decode_kwargs: Dict mapping modality_type -> decode kwargs

        Returns:
            Dict mapping modality type -> predicted instance
        """
        decode_kwargs = decode_kwargs or {}

        # Encode observed
        observed_tokens = {}
        for name, modality in observed_modalities.items():
            tokens = self.codec_manager.encode(modality)
            observed_tokens.update(tokens)

        observed_tokens = {k: v.to(self.device) for k, v in observed_tokens.items()}

        # Get target token keys
        target_token_keys = [mod_type.token_key for mod_type in target_modality_types]

        # Predict all targets
        predictions = self.model(observed_tokens, masked=target_token_keys)

        # Decode each predicted modality
        predicted_modalities = {}

        for mod_type in target_modality_types:
            token_key = mod_type.token_key

            # Sample tokens
            pred_logits = predictions[token_key]
            pred_tokens = pred_logits.argmax(dim=-1)

            # Decode
            kwargs = decode_kwargs.get(mod_type, {})
            predicted = self.codec_manager.decode(
                {token_key: pred_tokens},
                mod_type,
                **kwargs
            )

            predicted_modalities[mod_type] = predicted

        return predicted_modalities


# Example: Predict everything from just an image
if __name__ == "__main__":
    predictor = MultimodalPredictor(model, codec_manager, device="cuda")

    # Only have an image
    image = MyImage(pixels=torch.randn(1, 3, 224, 224), metadata={})

    # Predict demographics, labs, vitals, and time series
    predictions = predictor.predict_multiple(
        observed_modalities={'image': image},
        target_modality_types=[
            PatientDemographics,
            LabResults,
            VitalSigns,
            MyTimeSeries
        ],
        decode_kwargs={
            MyTimeSeries: {'timestamps': torch.arange(1000).float()}
        }
    )

    # Access predictions
    demo = predictions[PatientDemographics]
    print(f"Predicted age: {demo.age.item():.1f}")
    print(f"Predicted sex: {demo.sex.item():.1f}")

    labs = predictions[LabResults]
    print(f"Predicted glucose: {labs.glucose.item():.1f}")

    vitals = predictions[VitalSigns]
    print(f"Predicted heart rate: {vitals.heart_rate.item():.1f}")

    timeseries = predictions[MyTimeSeries]
    print(f"Predicted time series shape: {timeseries.values.shape}")
```

### Mode 3: Iterative Refinement

**Use case:** Predict one modality, then use it to predict another

```python
class MultimodalPredictor:
    # ... previous code ...

    @torch.no_grad()
    def predict_iterative(
        self,
        initial_modalities: dict,
        prediction_sequence: list
    ):
        """
        Iteratively predict modalities, using previous predictions.

        Args:
            initial_modalities: Starting modalities
            prediction_sequence: List of (modality_type, decode_kwargs) tuples

        Returns:
            Dict with all modalities (initial + predicted)
        """
        current_modalities = initial_modalities.copy()

        for mod_type, decode_kwargs in prediction_sequence:
            # Predict next modality using all current modalities
            predicted = self.predict_missing(
                current_modalities,
                mod_type,
                **decode_kwargs
            )

            # Add to current modalities
            current_modalities[mod_type.__name__.lower()] = predicted

        return current_modalities


# Example: Image → Demographics → Labs → Vitals
if __name__ == "__main__":
    predictor = MultimodalPredictor(model, codec_manager)

    image = MyImage(pixels=torch.randn(1, 3, 224, 224), metadata={})

    # Predict in sequence
    all_modalities = predictor.predict_iterative(
        initial_modalities={'image': image},
        prediction_sequence=[
            (PatientDemographics, {}),
            (LabResults, {}),
            (VitalSigns, {}),
            (MyTimeSeries, {'timestamps': torch.arange(1000).float()})
        ]
    )

    print("Final modalities:", all_modalities.keys())
```

---

## Complete Inference Pipeline

### Step-by-Step Example

```python
"""
complete_inference.py - Full inference pipeline
"""

import torch
from pathlib import Path
from PIL import Image
import numpy as np

from step4_codec_manager import CodecManager
from step6_train_transformer import MultimodalTransformer
from step1_define_modalities import *


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained model from checkpoint"""

    # Define model architecture (must match training)
    vocab_sizes = {
        "tok_my_image": 10000,
        "tok_demographics": 256,
        "tok_lab_results": 256,
        "tok_vitals": 256,
        "tok_timeseries": 512,
    }

    num_tokens = {
        "tok_my_image": 784,
        "tok_demographics": 1,
        "tok_lab_results": 1,
        "tok_vitals": 1,
        "tok_timeseries": 128,
    }

    model = MultimodalTransformer(
        vocab_sizes=vocab_sizes,
        num_tokens=num_tokens,
        d_model=512,
        nhead=8,
        num_layers=12
    )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()

    return model


def load_image_from_file(image_path: str) -> MyImage:
    """Load and preprocess image from file"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))

    # Convert to tensor
    pixels = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    pixels = pixels.unsqueeze(0)  # [1, 3, 224, 224]

    return MyImage(pixels=pixels, metadata={'path': image_path})


def predict_patient_data(image_path: str, checkpoint_path: str):
    """
    Complete inference: Given patient image, predict all other data.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load model
    print("Loading model...")
    model = load_model(checkpoint_path, device)
    codec_manager = CodecManager(device=device)

    # 2. Load image
    print(f"Loading image from {image_path}...")
    image = load_image_from_file(image_path)
    image = MyImage(pixels=image.pixels.to(device), metadata=image.metadata)

    # 3. Create predictor
    predictor = MultimodalPredictor(model, codec_manager, device)

    # 4. Predict all modalities
    print("Predicting patient data...")
    predictions = predictor.predict_multiple(
        observed_modalities={'image': image},
        target_modality_types=[
            PatientDemographics,
            LabResults,
            VitalSigns,
            MyTimeSeries
        ],
        decode_kwargs={
            MyTimeSeries: {'timestamps': torch.arange(1000).float()}
        }
    )

    # 5. Extract and display results
    demo = predictions[PatientDemographics]
    labs = predictions[LabResults]
    vitals = predictions[VitalSigns]
    ts = predictions[MyTimeSeries]

    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)

    print("\nDemographics:")
    print(f"  Age: {demo.age.item():.1f} years")
    print(f"  Sex: {'Male' if demo.sex.item() > 0.5 else 'Female'}")
    print(f"  BMI: {demo.bmi.item():.1f}")

    print("\nLab Results:")
    print(f"  Glucose: {labs.glucose.item():.1f} mg/dL")
    print(f"  Cholesterol: {labs.cholesterol.item():.1f} mg/dL")
    print(f"  Hemoglobin: {labs.hemoglobin.item():.1f} g/dL")

    print("\nVital Signs:")
    print(f"  Heart Rate: {vitals.heart_rate.item():.1f} bpm")
    print(f"  Blood Pressure: {vitals.blood_pressure_sys.item():.0f}/{vitals.blood_pressure_dia.item():.0f} mmHg")
    print(f"  Temperature: {vitals.temperature.item():.1f} °F")

    print("\nTime Series:")
    print(f"  Length: {ts.values.shape[1]} timesteps")
    print(f"  Mean value: {ts.values.mean().item():.3f}")
    print(f"  Std value: {ts.values.std().item():.3f}")

    return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to patient image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')

    args = parser.parse_args()

    predictions = predict_patient_data(args.image, args.checkpoint)
```

**Run:**
```bash
python complete_inference.py \
    --image /path/to/patient_image.jpg \
    --checkpoint checkpoints/best_model.pt
```

---

## Use Cases

### 1. Missing Data Imputation

**Scenario:** Patient has incomplete records

```python
# Have: Image + Demographics
# Missing: Lab results, Vitals

# Predict missing
predictions = predictor.predict_multiple(
    observed_modalities={
        'image': patient_image,
        'demographics': patient_demographics
    },
    target_modality_types=[LabResults, VitalSigns]
)

# Fill in missing values
complete_record = {
    'image': patient_image,
    'demographics': patient_demographics,
    'labs': predictions[LabResults],
    'vitals': predictions[VitalSigns]
}
```

### 2. Cross-Modal Retrieval

**Scenario:** Find similar patients based on predicted characteristics

```python
# Given only an image, predict demographics
predicted_demo = predictor.predict_missing(
    observed_modalities={'image': query_image},
    target_modality_type=PatientDemographics
)

# Search database for similar age/sex/BMI
similar_patients = database.search(
    age=predicted_demo.age.item(),
    sex=predicted_demo.sex.item(),
    bmi=predicted_demo.bmi.item()
)
```

### 3. Anomaly Detection

**Scenario:** Check if predicted values match observations

```python
# Predict lab results from image
predicted_labs = predictor.predict_missing(
    observed_modalities={'image': patient_image},
    target_modality_type=LabResults
)

# Compare with actual lab results
actual_labs = patient_record['labs']

glucose_diff = abs(predicted_labs.glucose - actual_labs.glucose)
if glucose_diff > threshold:
    print("⚠️  Anomaly detected: Glucose mismatch")
    print(f"   Predicted: {predicted_labs.glucose.item():.1f}")
    print(f"   Actual: {actual_labs.glucose.item():.1f}")
```

### 4. Multimodal Embeddings

**Scenario:** Get unified representation of patient

```python
class MultimodalPredictor:
    # ... previous code ...

    @torch.no_grad()
    def get_embedding(self, modalities: dict) -> torch.Tensor:
        """
        Get multimodal embedding for input modalities.

        Returns:
            Embedding tensor [batch, d_model]
        """
        # Encode modalities
        tokens = {}
        for name, modality in modalities.items():
            token_dict = self.codec_manager.encode(modality)
            tokens.update(token_dict)

        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        # Embed tokens (no masking)
        embedded, _ = self.model.embed_tokens(tokens, masked_modalities=[])

        # Pool to get single embedding (mean over sequence)
        embedding = embedded.mean(dim=1)  # [batch, d_model]

        return embedding


# Use embeddings for similarity search
embedding1 = predictor.get_embedding({'image': patient1_image})
embedding2 = predictor.get_embedding({'image': patient2_image})

similarity = F.cosine_similarity(embedding1, embedding2)
print(f"Similarity: {similarity.item():.3f}")
```

### 5. Conditional Generation

**Scenario:** Generate synthetic data with specific characteristics

```python
# Generate lab results for 65-year-old male with BMI 30
target_demographics = PatientDemographics(
    age=torch.tensor([[65.0]]),
    sex=torch.tensor([[1.0]]),
    bmi=torch.tensor([[30.0]]),
    metadata={}
)

# Predict what labs would look like
predicted_labs = predictor.predict_missing(
    observed_modalities={'demographics': target_demographics},
    target_modality_type=LabResults
)

print(f"Expected glucose for 65yo male, BMI 30: {predicted_labs.glucose.item():.1f}")
```

---

## Advanced: Conditional Generation

### Temperature Sampling

Instead of greedy (argmax), sample with temperature for diversity:

```python
class MultimodalPredictor:
    # ... previous code ...

    def sample_tokens(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample tokens from logits with temperature.

        Args:
            logits: [batch, num_tokens, vocab_size]
            temperature: Higher = more random, lower = more deterministic

        Returns:
            Sampled tokens [batch, num_tokens]
        """
        # Apply temperature
        logits = logits / temperature

        # Sample from categorical distribution
        probs = torch.softmax(logits, dim=-1)  # [B, N, V]

        # Sample for each position
        batch_size, num_tokens, vocab_size = logits.shape
        sampled = []

        for i in range(num_tokens):
            token_probs = probs[:, i, :]  # [B, V]
            token_samples = torch.multinomial(token_probs, num_samples=1)  # [B, 1]
            sampled.append(token_samples)

        sampled_tokens = torch.cat(sampled, dim=1)  # [B, N]
        return sampled_tokens

    @torch.no_grad()
    def predict_with_sampling(
        self,
        observed_modalities: dict,
        target_modality_type: type,
        temperature: float = 1.0,
        num_samples: int = 1,
        **decode_kwargs
    ):
        """
        Predict with sampling - generate multiple diverse predictions.

        Returns:
            List of predicted modalities (length = num_samples)
        """
        # Encode observed
        observed_tokens = {}
        for name, modality in observed_modalities.items():
            tokens = self.codec_manager.encode(modality)
            observed_tokens.update(tokens)

        observed_tokens = {k: v.to(self.device) for k, v in observed_tokens.items()}

        target_key = target_modality_type.token_key

        predictions = []

        for _ in range(num_samples):
            # Forward pass
            pred_dict = self.model(observed_tokens, masked=[target_key])
            pred_logits = pred_dict[target_key]

            # Sample tokens
            sampled_tokens = self.sample_tokens(pred_logits, temperature)

            # Decode
            predicted = self.codec_manager.decode(
                {target_key: sampled_tokens},
                target_modality_type,
                **decode_kwargs
            )

            predictions.append(predicted)

        return predictions


# Generate 10 possible lab results given demographics
samples = predictor.predict_with_sampling(
    observed_modalities={'demographics': patient_demo},
    target_modality_type=LabResults,
    temperature=0.8,  # Some randomness
    num_samples=10
)

# Analyze distribution of predictions
glucose_values = [s.glucose.item() for s in samples]
print(f"Mean predicted glucose: {np.mean(glucose_values):.1f} ± {np.std(glucose_values):.1f}")
```

### Top-K / Top-P Sampling

```python
def sample_top_k(self, logits: torch.Tensor, k: int = 50) -> torch.Tensor:
    """Sample from top-k most likely tokens"""
    # Get top k values and indices
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)

    # Sample from top k
    probs = torch.softmax(top_k_logits, dim=-1)
    sampled_indices = torch.multinomial(
        probs.view(-1, k),
        num_samples=1
    ).view(logits.shape[0], logits.shape[1])

    # Map back to original vocabulary
    sampled_tokens = torch.gather(top_k_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)

    return sampled_tokens
```

---

## Summary

### Model Outputs

**During Training:**
- Token logits for masked modalities
- Shape: `[batch, num_tokens, vocab_size]`

**During Inference:**
1. Sample discrete tokens from logits
2. Decode tokens → continuous embeddings (codec)
3. Decode embeddings → actual modalities (codec)

### Inference Modes

1. **Predict one missing modality** - Given some, predict one
2. **Predict multiple modalities** - Given some, predict many
3. **Iterative refinement** - Chain predictions
4. **Generate embeddings** - Get unified representation
5. **Sample with diversity** - Temperature/top-k sampling

### Use Cases

- Missing data imputation
- Cross-modal retrieval
- Anomaly detection
- Similarity search
- Conditional generation
- Synthetic data generation

The model is essentially a **multimodal autoencoder + predictor** that learns correlations between different data types! 🎯
