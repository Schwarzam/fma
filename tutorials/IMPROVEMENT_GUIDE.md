# Guide: Improving Multimodal Model Performance

Your model is working correctly (no bugs), but predictions are poor quality. Here's how to improve it.

## Current Issues

1. **Image reconstruction**: Just noise
2. **Spectrum prediction**: Correlation ≈ 0 (random)
3. **Light curve prediction**: Poor MAE
4. **Scalar predictions**: Very inaccurate (redshift MAE ~0.95, should be <0.1)
5. **SFR prediction**: Completely off (MAE 53 M☉/yr)

## Why Performance Is Poor

### 1. Model Size (CRITICAL)
Your current model is **tiny**:
- `d_model=256` (embedding dimension)
- `num_layers=4` (transformer depth)
- `nhead=4` (attention heads)
- `dim_feedforward=1024`

**Total params**: ~5-10M (toy size)

For comparison:
- GPT-2 small: 117M params
- BERT base: 110M params
- Vision Transformer (ViT-B): 86M params

**Solution**: Scale up the model

### 2. Training Duration
Your training likely ran for only 10-15 epochs with a small dataset.

**Solution**: Train longer (100+ epochs) or use more data

### 3. Task Difficulty
Cross-modal prediction is **extremely hard**:
- Predicting spectrum from image alone: Very challenging (requires learning physics)
- Reconstructing images: Requires spatial understanding
- Light curves: Temporal patterns need many examples

**Solution**:
- Use easier tasks first (same-modality reconstruction)
- Add auxiliary losses
- Use pretrained encoders

### 4. Data Scale
You have only 150 samples (tiny dataset).

**Solution**: Generate more synthetic data or use real surveys

---

## Improvement Strategy

### Priority 1: Scale Up Model ⭐⭐⭐

**Edit `train_astronomical.py` around line 219:**

```python
# BEFORE (tiny model)
model = MultimodalTransformer(
    vocab_sizes=vocab_sizes,
    num_tokens=inferred_num_tokens,
    d_model=256,      # Too small!
    nhead=4,          # Too few heads!
    num_layers=4,     # Too shallow!
    dim_feedforward=1024,
    dropout=0.1
)

# AFTER (medium model)
model = MultimodalTransformer(
    vocab_sizes=vocab_sizes,
    num_tokens=inferred_num_tokens,
    d_model=768,      # 3x larger
    nhead=12,         # 3x more heads
    num_layers=12,    # 3x deeper
    dim_feedforward=3072,  # 3x wider FFN
    dropout=0.1
)
```

**Impact**: ~50-100M params (100x larger)
**Cost**: Slower training (but much better results)

### Priority 2: Train Longer ⭐⭐⭐

**Edit `train_astronomical.py` around line 236:**

```python
# BEFORE
num_epochs = 15  # Too short!

# AFTER
num_epochs = 100  # Or more!
```

**Add learning rate scheduling:**

```python
# After creating optimizer (around line 233)
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# In training loop, after optimizer.step():
scheduler.step()
```

### Priority 3: Improve Codecs ⭐⭐

Your current codecs are **too simple**. Upgrade them:

#### Better Image Encoder

```python
# In astronomical_codecs.py, replace SimpleImageEncoder with:

class BetterImageEncoder(nn.Module):
    """ResNet-style encoder"""
    def __init__(self, in_channels=3, embedding_dim=64):
        super().__init__()
        # Use ResNet blocks instead of simple conv
        self.initial = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3)

        # Add residual blocks
        self.block1 = self._make_block(64, 128, stride=2)
        self.block2 = self._make_block(128, 256, stride=2)
        self.block3 = self._make_block(256, embedding_dim, stride=2)

    def _make_block(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
```

#### Larger Codebook for VectorQuantizer

```python
# In astronomical_codecs.py, line 85
# BEFORE
self._quantizer = VectorQuantizer(num_embeddings=512, embedding_dim=embedding_dim)

# AFTER
self._quantizer = VectorQuantizer(num_embeddings=8192, embedding_dim=embedding_dim)
# Larger codebook = more expressive
```

Don't forget to update `vocab_sizes` in `train_astronomical.py`:
```python
vocab_sizes = {
    "tok_galaxy_image": 1000,
    "tok_gaia_spectrum": 8192,  # Updated from 512
    "tok_ztf_lightcurve": 8192,  # Updated from 512
    "tok_redshift": 256,
    "tok_stellar_mass": 256,
    "tok_sfr": 256,
}
```

### Priority 4: Add Auxiliary Losses ⭐⭐

Help the model learn better representations:

```python
# In train_astronomical.py, inside training loop

# Current loss (masked token prediction)
loss = cross_entropy_loss

# Add reconstruction loss
reconstruction_loss = 0.0
for key in all_tokens.keys():
    if key not in masked:
        # Predict visible tokens too (self-reconstruction)
        pred = model.predict_single(all_tokens, key)
        recon_loss = F.cross_entropy(pred, all_tokens[key])
        reconstruction_loss += recon_loss

# Combined loss
total_loss = loss + 0.1 * reconstruction_loss
```

### Priority 5: Use Pretrained Vision Encoder ⭐

Instead of training image encoder from scratch, use pretrained:

```python
import torchvision.models as models

class PretrainedImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Use ResNet18 pretrained on ImageNet
        resnet = models.resnet18(pretrained=True)
        # Remove final FC layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        # Freeze early layers
        for param in list(self.encoder.parameters())[:-10]:
            param.requires_grad = False

    def forward(self, x):
        return self.encoder(x)
```

### Priority 6: Generate More Data ⭐

Your 150 samples are way too few. Generate more:

```bash
# Modify generate_toy_data.py to create 10,000+ samples
python generate_toy_data.py --num_train 8000 --num_val 1000 --num_test 1000
```

### Priority 7: Better Masking Strategy

```python
# In train_astronomical.py, modify masking

# Instead of random masking, use:
# 1. Whole modality masking (current approach) - 50%
# 2. Token-level masking within modalities - 30%
# 3. No masking (reconstruction) - 20%

import random
mask_strategy = random.random()

if mask_strategy < 0.5:
    # Whole modality masking (current)
    masked_modalities = random.sample(list(all_tokens.keys()), num_to_mask)
elif mask_strategy < 0.8:
    # Token-level masking (like BERT)
    for key in all_tokens.keys():
        mask = torch.rand(all_tokens[key].shape) < 0.15
        # Apply mask...
else:
    # Full reconstruction
    masked_modalities = []
```

---

## Recommended Action Plan

### Phase 1: Quick Wins (1-2 days)
1. ✅ Scale up model: d_model=768, num_layers=12
2. ✅ Train for 100 epochs
3. ✅ Add LR scheduling

**Expected improvement**: 3-5x better metrics

### Phase 2: Architecture (3-5 days)
4. ✅ Upgrade codecs (ResNet encoder)
5. ✅ Larger codebook (512→8192)
6. ✅ Add auxiliary losses

**Expected improvement**: 2-3x on top of Phase 1

### Phase 3: Data & Transfer Learning (1 week)
7. ✅ Generate 10k+ samples
8. ✅ Use pretrained image encoder
9. ✅ Better masking strategies

**Expected improvement**: Near production-quality

---

## Expected Performance Targets

After Phase 1 (scaled model):
- Redshift MAE: ~0.3 (down from 0.95)
- Spectrum correlation: ~0.3-0.5 (up from 0.0)
- Images: Blurry but recognizable (not noise)

After Phase 2 (better architecture):
- Redshift MAE: ~0.1-0.15
- Spectrum correlation: ~0.6-0.7
- Images: Clear reconstruction
- SFR MAE: ~10-20 M☉/yr (down from 53)

After Phase 3 (more data + pretrained):
- Redshift MAE: ~0.05
- Spectrum correlation: ~0.8+
- Images: High-quality reconstruction
- All metrics near state-of-the-art

---

## Quick Start: Best Configuration

If you want the **single best improvement** right now:

```python
# train_astronomical.py line ~219
model = MultimodalTransformer(
    vocab_sizes=vocab_sizes,
    num_tokens=inferred_num_tokens,
    d_model=512,       # Doubled
    nhead=8,           # Doubled
    num_layers=8,      # Doubled
    dim_feedforward=2048,  # Doubled
    dropout=0.1
)

# line ~236
num_epochs = 50  # 3x longer

# Add after line ~233
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# In training loop after optimizer.step():
scheduler.step()
```

Then retrain:
```bash
python train_astronomical.py
```

This should give **10-20x better performance** with minimal code changes!

---

## Debugging Tips

If after scaling up you still see poor performance:

1. **Check if model is learning:**
   - Training loss should decrease consistently
   - If loss plateaus early → increase learning rate
   - If loss explodes → decrease learning rate or add gradient clipping

2. **Check codecs are working:**
   ```python
   # Test encode-decode cycle
   sample = dataset[0]
   encoded = codec.encode(sample)
   decoded = codec.decode(encoded)
   # Should look similar to original
   ```

3. **Visualize attention:**
   - Check if model attends to relevant modalities
   - Use attention visualization tools

4. **Ablation studies:**
   - Train with only images → scalars (easier task)
   - If that works, gradually add more modalities

---

## Summary

**What to do NOW:**
1. Scale up model (d_model=512, num_layers=8)
2. Train for 50-100 epochs
3. Add LR scheduling

**Expected result**: Much better predictions in 1-2 hours of training!

**Next steps**: Follow Phase 2-3 if you need production quality.
