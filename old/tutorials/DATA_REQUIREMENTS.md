# Data Requirements & Limited Data Strategies

How much data do you actually need, and what to do if you don't have much.

---

## Table of Contents

1. [Quick Answer](#quick-answer)
2. [Data Requirements by Stage](#data-requirements-by-stage)
3. [Limited Data Strategies](#limited-data-strategies)
4. [Transfer Learning](#transfer-learning)
5. [Synthetic Data](#synthetic-data)
6. [Practical Examples](#practical-examples)

---

## Quick Answer

### Minimum Data Requirements

**For a working proof-of-concept:**

| Component | Minimum | Comfortable | Ideal |
|-----------|---------|-------------|-------|
| **Codec Training** | 1K-10K samples | 100K samples | 1M+ samples |
| **Transformer Training** | 10K samples | 100K samples | 1M+ samples |

**But:** You can start training with much less using the strategies below!

### Reality Check

**AION's dataset:**
- ~10 million astronomical objects
- But many have missing modalities
- Effective training set: ~1-2 million complete samples

**You can start smaller:**
- 10K samples: Will train, but may overfit
- 50K samples: Good starting point
- 100K+ samples: Comfortable for most applications

---

## Data Requirements by Stage

### Stage 1: Codec Pre-training

**Purpose:** Learn to compress and reconstruct each modality

**Requirements per modality:**

```python
# Image Codec
Minimum:  1,000 images      # Will work, may not generalize well
Good:     10,000 images     # Decent compression learned
Better:   100,000 images    # Good generalization
Best:     1,000,000+ images # Publication-quality

# Time Series Codec
Minimum:  1,000 sequences
Good:     10,000 sequences
Better:   100,000 sequences

# Scalar/Tabular Codecs
Minimum:  10,000 samples    # For reservoir quantizers to build good CDF
Good:     100,000 samples
```

**Why codecs need less data:**
- Single modality focus (simpler task)
- Reconstruction objective (self-supervised)
- Can use data augmentation aggressively

### Stage 2: Transformer Training

**Purpose:** Learn correlations between modalities

**Requirements:**

```python
# Multimodal samples (aligned across modalities)
Minimum:  10,000 samples    # Will train, expect overfitting
Good:     50,000 samples    # Reasonable generalization
Better:   100,000 samples   # Good performance
Best:     1,000,000+ samples # Excellent performance

# Not all samples need all modalities!
# Model handles missing data naturally
```

**Why transformer needs more data:**
- Multi-modal correlations more complex
- More parameters to train
- Benefits from diversity

---

## Limited Data Strategies

### Strategy 1: Start with Pre-trained Codecs

**Use existing image encoders:**

```python
"""
Use pre-trained image encoder (e.g., from ImageNet)
"""

import torchvision.models as models

class PretrainedImageCodec(Codec):
    """Image codec using pre-trained ResNet"""

    def __init__(self):
        super().__init__()

        # Use pre-trained ResNet (FROZEN)
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove classifier

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Only train projection and quantizer
        self.projection = nn.Conv2d(2048, 64, kernel_size=1)
        self.unprojection = nn.Conv2d(64, 2048, kernel_size=1)

        # Learnable decoder
        self.decoder = SimpleImageDecoder(embedding_dim=2048, out_channels=3)

        self._quantizer = FiniteScalarQuantizer(levels=[8, 5, 5, 5])

    def _encode(self, x: MyImage):
        # Use frozen pre-trained encoder
        with torch.no_grad():
            z = self.encoder(x.pixels)  # [B, 2048, 1, 1]

        z = z.squeeze(-1).squeeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, 2048, 1, 1]
        z = self.projection(z)  # [B, 64, 1, 1]
        return z

    # ... rest of codec
```

**Benefits:**
- Train with 10x less image data
- Better generalization
- Faster training

**Available pre-trained models:**
- Images: ResNet, ViT, CLIP, DINOv2
- Time series: Pre-trained on UCR archive
- Text: BERT, GPT embeddings

### Strategy 2: Heavy Data Augmentation

**For images:**

```python
import torchvision.transforms as T

class AugmentedImageDataset(Dataset):
    """Dataset with aggressive augmentation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomErasing(p=0.1),
        ])

    def __getitem__(self, idx):
        image = self._load_image(idx)

        # Apply augmentation
        if self.split == 'train':
            image = self.augment(image)

        return image
```

**Effective data multiplication:**
- 1,000 images × 10 augmentations = 10,000 effective samples

**For time series:**

```python
def augment_timeseries(ts: torch.Tensor) -> torch.Tensor:
    """Augment time series"""

    # Time warping
    if torch.rand(1) > 0.5:
        ts = time_warp(ts)

    # Magnitude warping
    if torch.rand(1) > 0.5:
        scale = 0.8 + torch.rand(1) * 0.4  # [0.8, 1.2]
        ts = ts * scale

    # Gaussian noise
    if torch.rand(1) > 0.5:
        noise = torch.randn_like(ts) * 0.1
        ts = ts + noise

    # Window slicing (use random subsequence)
    if torch.rand(1) > 0.5:
        start = torch.randint(0, len(ts) // 4, (1,))
        ts = ts[start:start + len(ts) // 2]
        ts = F.interpolate(ts.unsqueeze(0).unsqueeze(0), size=len(ts), mode='linear')
        ts = ts.squeeze()

    return ts
```

### Strategy 3: Self-Supervised Pre-training

**Pre-train codecs with self-supervision:**

```python
"""
Pre-train image codec on unlabeled images
"""

class MoCoImageCodec(Codec):
    """Image codec with MoCo self-supervised pre-training"""

    def pretrain_moco(self, unlabeled_images: DataLoader, num_epochs: int = 100):
        """
        Pre-train with momentum contrast.

        Learns good representations from unlabeled data.
        """
        # Implement MoCo/SimCLR/BYOL
        # See: https://arxiv.org/abs/1911.05722

        # Query encoder
        query_encoder = self.encoder

        # Key encoder (momentum updated)
        key_encoder = copy.deepcopy(self.encoder)

        for epoch in range(num_epochs):
            for images in unlabeled_images:
                # Two augmented views
                view1 = augment(images)
                view2 = augment(images)

                # Encode
                q = query_encoder(view1)  # Queries
                k = key_encoder(view2)    # Keys

                # Contrastive loss
                loss = contrastive_loss(q, k)

                # Update
                loss.backward()
                optimizer.step()

                # Momentum update key encoder
                update_momentum(query_encoder, key_encoder, momentum=0.999)

        # Now encoder has learned good features!
        # Fine-tune with reconstruction task
```

**Benefits:**
- Use unlimited unlabeled data
- Better representations
- Less labeled data needed

### Strategy 4: Progressive Training

**Start small, grow gradually:**

```python
"""
Progressive training strategy
"""

def train_progressive(model, data, num_stages=3):
    """
    Train progressively with increasing data and complexity.

    Stage 1: Small model, small data
    Stage 2: Medium model, more data
    Stage 3: Full model, all data
    """

    # Stage 1: Small model, 10K samples
    print("Stage 1: Training small model...")
    small_model = MultimodalTransformer(
        d_model=128,
        nhead=4,
        num_layers=4
    )
    train(small_model, data[:10000], num_epochs=50)

    # Stage 2: Medium model, 50K samples
    print("Stage 2: Growing model...")
    medium_model = MultimodalTransformer(
        d_model=256,
        nhead=8,
        num_layers=6
    )
    # Initialize from small model
    copy_weights(small_model, medium_model)
    train(medium_model, data[:50000], num_epochs=30)

    # Stage 3: Full model, all data
    print("Stage 3: Full model...")
    full_model = MultimodalTransformer(
        d_model=512,
        nhead=8,
        num_layers=12
    )
    copy_weights(medium_model, full_model)
    train(full_model, data, num_epochs=20)

    return full_model
```

### Strategy 5: Few-Shot Learning

**Train with minimal data per class:**

```python
"""
Meta-learning for few-shot multimodal learning
"""

class FewShotMultimodalModel:
    """
    Train to predict with few examples.

    Uses MAML (Model-Agnostic Meta-Learning).
    """

    def meta_train(self, support_sets: list, query_sets: list):
        """
        Meta-training loop.

        For each task:
        1. Sample support set (few examples)
        2. Adapt model to task
        3. Test on query set
        4. Update meta-parameters
        """

        for task in range(num_tasks):
            support = support_sets[task]  # e.g., 5 samples
            query = query_sets[task]

            # Inner loop: adapt to task
            adapted_model = copy.deepcopy(self.model)
            for _ in range(inner_steps):
                loss = compute_loss(adapted_model, support)
                adapted_model.adapt(loss)

            # Outer loop: meta-update
            meta_loss = compute_loss(adapted_model, query)
            meta_loss.backward()
            self.meta_optimizer.step()

    def predict_with_few_examples(self, examples: list, query):
        """Predict using just a few examples"""
        # Adapt model to examples
        adapted_model = self.adapt_to_examples(examples)

        # Predict on query
        return adapted_model(query)
```

---

## Transfer Learning

### Use Pre-trained AION Models

**Start from AION's weights:**

```python
"""
Transfer learning from AION
"""

# 1. Load pre-trained AION codecs
from aion.codecs.manager import CodecManager as AIONCodecManager

aion_codec_manager = AIONCodecManager(
    device="cuda",
    hf_repo="polymathic-ai/aion-base"
)

# 2. Use AION's image codec for your data
aion_image_codec = aion_codec_manager._load_codec(LegacySurveyImage)

# 3. Fine-tune on your domain
for images in your_dataloader:
    # Encode with AION codec
    z_e = aion_image_codec._encode(images)
    z_q, loss, _ = aion_image_codec.quantizer(z_e)
    reconstructed = aion_image_codec._decode(z_q)

    # Fine-tune loss
    loss = F.mse_loss(reconstructed.pixels, images.pixels)
    loss.backward()
    optimizer.step()

# 4. Now you have a domain-adapted codec!
```

### Cross-Domain Transfer

**Medical images → Astronomical images:**

```python
class DomainAdaptiveCodec(Codec):
    """
    Transfer from one domain to another.

    Uses domain adversarial training.
    """

    def __init__(self, source_codec: Codec):
        super().__init__()

        # Copy source encoder (pre-trained)
        self.encoder = copy.deepcopy(source_codec.encoder)

        # Domain discriminator
        self.domain_discriminator = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Source vs Target domain
        )

        # New decoder for target domain
        self.decoder = SimpleImageDecoder(...)

    def train_domain_adaptive(self, source_data, target_data):
        """
        Train to transfer from source to target domain.

        Encoder learns domain-invariant features.
        """

        for source_batch, target_batch in zip(source_loader, target_loader):
            # Encode both domains
            source_z = self.encoder(source_batch)
            target_z = self.encoder(target_batch)

            # Reconstruction loss (target domain)
            recon = self.decoder(target_z)
            recon_loss = F.mse_loss(recon, target_batch)

            # Domain classification
            source_domain = self.domain_discriminator(source_z)
            target_domain = self.domain_discriminator(target_z)

            # Domain adversarial loss (encoder tries to fool discriminator)
            domain_loss = domain_adversarial_loss(source_domain, target_domain)

            # Total loss
            loss = recon_loss - 0.1 * domain_loss  # Negative for adversarial
            loss.backward()
```

---

## Synthetic Data

### Generate Synthetic Training Data

**When you have very little real data:**

```python
"""
Generate synthetic multimodal data
"""

class SyntheticDataGenerator:
    """Generate synthetic paired multimodal data"""

    def __init__(self, num_samples: int = 10000):
        self.num_samples = num_samples

    def generate_correlated_data(self):
        """
        Generate synthetic data with known correlations.

        Useful for:
        1. Initial training
        2. Testing pipeline
        3. Augmenting real data
        """

        synthetic_data = []

        for i in range(self.num_samples):
            # Sample latent factors
            age = np.random.normal(50, 15)  # Mean 50, std 15
            disease_severity = np.random.uniform(0, 1)

            # Generate correlated modalities
            # Image: darker with disease
            image = self._generate_image(disease_severity)

            # Lab values: correlate with age and disease
            glucose = 90 + 0.5 * age + 50 * disease_severity + np.random.normal(0, 10)
            cholesterol = 150 + 0.8 * age + 30 * disease_severity + np.random.normal(0, 20)

            # Time series: noisier with disease
            heart_rate = 70 + 10 * disease_severity + np.random.normal(0, 5)
            ecg = self._generate_ecg(heart_rate, disease_severity)

            synthetic_data.append({
                'image': image,
                'demographics': {'age': age},
                'labs': {'glucose': glucose, 'cholesterol': cholesterol},
                'timeseries': ecg
            })

        return synthetic_data

    def _generate_image(self, disease_severity):
        """Generate synthetic medical image"""
        # Base image
        img = np.random.randn(3, 224, 224) * 0.1 + 0.5

        # Add disease patterns
        if disease_severity > 0.5:
            # Add dark spots
            center_x, center_y = 112, 112
            xx, yy = np.meshgrid(range(224), range(224))
            mask = ((xx - center_x)**2 + (yy - center_y)**2) < (50 * disease_severity)**2
            img[:, mask] *= 0.5

        return torch.from_numpy(img).float()

    def _generate_ecg(self, heart_rate, disease_severity):
        """Generate synthetic ECG signal"""
        t = np.linspace(0, 10, 1000)
        freq = heart_rate / 60  # Hz

        # Normal ECG: sum of sinusoids
        ecg = np.sin(2 * np.pi * freq * t)

        # Add disease artifacts
        if disease_severity > 0.5:
            ecg += 0.3 * np.sin(2 * np.pi * 5 * freq * t) * disease_severity

        # Add noise
        ecg += np.random.normal(0, 0.1 * (1 + disease_severity), 1000)

        return torch.from_numpy(ecg).float()


# Use synthetic data to bootstrap training
synthetic_gen = SyntheticDataGenerator(num_samples=50000)
synthetic_data = synthetic_gen.generate_correlated_data()

# Pre-train on synthetic
train(model, synthetic_data, num_epochs=20)

# Fine-tune on real (even if just 1000 samples)
train(model, real_data, num_epochs=50)
```

### Mix Synthetic and Real

**Curriculum learning: synthetic → real**

```python
def train_with_synthetic_curriculum(model, synthetic_data, real_data):
    """
    Gradually shift from synthetic to real data.

    Epochs 1-20:   100% synthetic
    Epochs 21-40:   50% synthetic, 50% real
    Epochs 41-60:   25% synthetic, 75% real
    Epochs 61+:      0% synthetic, 100% real
    """

    for epoch in range(100):
        # Determine mixing ratio
        if epoch < 20:
            synthetic_ratio = 1.0
        elif epoch < 40:
            synthetic_ratio = 0.5
        elif epoch < 60:
            synthetic_ratio = 0.25
        else:
            synthetic_ratio = 0.0

        # Create mixed dataset
        num_synthetic = int(len(real_data) * synthetic_ratio)
        num_real = len(real_data) - num_synthetic

        mixed_data = (
            random.sample(synthetic_data, num_synthetic) +
            random.sample(real_data, num_real)
        )

        # Train on mixed data
        train_epoch(model, mixed_data)
```

---

## Practical Examples

### Example 1: Small Medical Dataset (5K samples)

**Setup:**
- 5,000 patients with X-rays and lab results
- Want to predict labs from X-rays

**Strategy:**

```python
# 1. Use pre-trained image encoder (ImageNet ResNet)
image_codec = PretrainedImageCodec()  # Frozen ResNet encoder

# 2. Train only projection + quantizer (10x less data needed)
train_codec(image_codec, images, num_epochs=30)

# 3. Heavy augmentation for images
augmented_loader = DataLoader(
    AugmentedImageDataset(images),
    batch_size=32
)

# 4. Simple scalar codec (no neural net needed)
lab_codec = ScalarCodec()  # Just quantization

# 5. Small transformer (less data needed)
model = MultimodalTransformer(
    d_model=256,      # Smaller
    nhead=4,
    num_layers=4,     # Fewer layers
    dropout=0.2       # More regularization
)

# 6. Train with regularization
train(model, data, num_epochs=100, weight_decay=0.1)
```

**Result:** Works reasonably well with 5K samples!

### Example 2: Tiny Astronomy Dataset (1K objects)

**Setup:**
- 1,000 galaxies with images and spectra
- Want to predict spectrum from image

**Strategy:**

```python
# 1. Use AION's pre-trained codecs (transfer learning)
from aion.codecs import ImageCodec as AIONImageCodec

image_codec = AIONImageCodec.from_pretrained("polymathic-ai/aion-base")

# 2. Fine-tune only decoder (encoder frozen)
for param in image_codec.encoder.parameters():
    param.requires_grad = False

# Fine-tune decoder
train_codec(image_codec, your_galaxies, num_epochs=50)

# 3. Use AION's spectrum codec (or train new one)
spectrum_codec = AIONSpectrumCodec.from_pretrained("polymathic-ai/aion-base")

# 4. Tiny transformer
model = MultimodalTransformer(
    d_model=128,
    nhead=2,
    num_layers=3
)

# 5. Few-shot learning
model = train_few_shot(model, your_galaxies)
```

**Result:** Reasonable predictions even with 1K samples!

### Example 3: Zero-Shot (No paired data!)

**Setup:**
- 10,000 images (no labels)
- 10,000 lab results (not paired with images)
- Want to learn relationship

**Strategy:**

```python
"""
Use cycle-consistency loss (like CycleGAN)
"""

# Train bidirectional prediction
# Image → Labs → Image
# Labs → Image → Labs

for images, labs in zip(image_loader, lab_loader):
    # Forward cycle
    predicted_labs = model.predict(image, target=Labs)
    reconstructed_image = model.predict(predicted_labs, target=Image)

    cycle_loss_1 = F.mse_loss(reconstructed_image, image)

    # Backward cycle
    predicted_image = model.predict(labs, target=Image)
    reconstructed_labs = model.predict(predicted_image, target=Labs)

    cycle_loss_2 = F.mse_loss(reconstructed_labs, labs)

    # Total loss
    loss = cycle_loss_1 + cycle_loss_2
    loss.backward()
```

---

## Summary

### You Can Start With Less Data Than You Think!

**Minimum viable:**
- 1K-10K samples: Possible with heavy transfer learning
- 10K-50K samples: Good with pre-trained encoders + augmentation
- 50K+ samples: Comfortable for training from scratch

**Key strategies:**
1. ✅ Use pre-trained encoders (ImageNet, CLIP, etc.)
2. ✅ Heavy data augmentation
3. ✅ Self-supervised pre-training
4. ✅ Transfer learning from AION
5. ✅ Start with smaller models
6. ✅ Progressive training
7. ✅ Synthetic data bootstrapping
8. ✅ Few-shot learning

**The good news:**
- Codecs need less data (single modality)
- Can use unlabeled data
- Transfer learning is very effective
- Not all samples need all modalities

**Start small, iterate fast, scale up gradually!** 🚀
