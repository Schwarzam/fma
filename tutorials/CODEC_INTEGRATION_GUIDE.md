# How to Use the Trained Image Codec

## Step 1: Verify Codec Quality

Check the reconstruction quality:

```bash
# Look at the visualization
xdg-open codec_reconstruction_samples.png  # or open it manually
```

The reconstructions should show clear galaxy structure (not flat/noisy).

Expected MSE: 0.005-0.015 (lower is better, but 0.21 was the baseline)

---

## Step 2: Update Codec Manager to Load Trained Codec

The codec manager needs to load the trained weights instead of using random initialization.

**Edit `step4_codec_manager.py`:**

Find the `_load_codec` method (around line 130) and modify it:

```python
def _load_codec(self, modality_type: Type[BaseModality]) -> Codec:
    """
    Load codec for a modality type (with caching).
    """
    if modality_type in self._codec_cache:
        return self._codec_cache[modality_type]

    codec_class = self._get_codec_class(modality_type)
    kwargs = self.codec_kwargs.get(modality_type, {})
    codec = codec_class(**kwargs).to(self.device).eval()

    # NEW: Load trained weights for GalaxyImageCodec
    if modality_type.__name__ == 'GalaxyImage':
        try:
            checkpoint = torch.load('image_codec_final.pt', map_location=self.device)
            codec.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"✓ Loaded trained image codec from image_codec_final.pt")
        except FileNotFoundError:
            print(f"⚠ Warning: image_codec_final.pt not found, using untrained codec")

    self._codec_cache[modality_type] = codec
    return codec
```

---

## Step 3: Update Training Script

**Option A: Simple Approach (Freeze Codec)**

This keeps the codec frozen during transformer training:

**Edit `train_astronomical.py`, add after line 179:**

```python
# After creating codec_manager
codec_manager = CodecManager(device=device)

# Load and freeze trained image codec
print("Loading trained image codec...")
try:
    from astronomical_dataset import GalaxyImage
    image_codec = codec_manager._load_codec(GalaxyImage)
    # Freeze codec parameters
    for param in image_codec.parameters():
        param.requires_grad = False
    print("✓ Image codec loaded and frozen")
except Exception as e:
    print(f"⚠ Warning: Could not load trained codec: {e}")
```

**Option B: Fine-tune Codec** (Advanced)

Allow codec to adapt during transformer training:

```python
# After creating codec_manager
codec_manager = CodecManager(device=device)

# Load trained image codec but keep trainable
print("Loading trained image codec (fine-tuning mode)...")
try:
    from astronomical_dataset import GalaxyImage
    image_codec = codec_manager._load_codec(GalaxyImage)
    # Set lower learning rate for codec
    codec_params = list(image_codec.parameters())
    transformer_params = list(model.parameters())

    # Update optimizer to use different learning rates
    optimizer = torch.optim.AdamW([
        {'params': transformer_params, 'lr': 1e-4},
        {'params': codec_params, 'lr': 1e-5}  # 10x lower for codec
    ], weight_decay=0.05)
    print("✓ Image codec loaded with fine-tuning")
except Exception as e:
    print(f"⚠ Warning: Could not load trained codec: {e}")
```

I recommend **Option A** for simplicity.

---

## Step 4: Update Inference Script

**Edit `inference_astronomical.py`, in the `__init__` method around line 150:**

```python
# After initializing codec manager (line ~151)
print("Initializing codecs...")
self.codec_manager = CodecManager(device=device)

# NEW: Load trained image codec
print("Loading trained image codec for inference...")
try:
    from astronomical_dataset import GalaxyImage
    image_codec = self.codec_manager._load_codec(GalaxyImage)
    print("✓ Trained image codec loaded")
except Exception as e:
    print(f"⚠ Warning: Could not load trained codec: {e}")
```

---

## Step 5: Quick Implementation (Copy-Paste Ready)

### For `step4_codec_manager.py`

Add this import at the top:
```python
import torch
```

Replace the `_load_codec` method with:

```python
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
```

### For `train_astronomical.py`

Add after codec_manager creation (around line 179):

```python
# Freeze trained image codec
print("Configuring image codec...")
try:
    from astronomical_dataset import GalaxyImage
    image_codec = codec_manager._load_codec(GalaxyImage)
    for param in image_codec.parameters():
        param.requires_grad = False
    print("✓ Image codec frozen for training")
except Exception as e:
    print(f"⚠ Warning: {e}")
```

### For `inference_astronomical.py`

Add after codec_manager initialization (around line 151):

```python
# Load trained image codec
try:
    from astronomical_dataset import GalaxyImage
    _ = self.codec_manager._load_codec(GalaxyImage)
except Exception as e:
    print(f"⚠ Warning: {e}")
```

---

## Step 6: Retrain the Transformer

Now retrain with the working codec:

```bash
/home/gustavo/miniconda3/bin/python3 train_astronomical.py
```

**Expected improvements:**
- Images should now be recognizable (not flat)
- Image reconstruction from other modalities will still be challenging but at least possible
- Training should converge better

---

## Step 7: Test Inference

After retraining:

```bash
/home/gustavo/miniconda3/bin/python3 inference_astronomical.py
```

Check:
- `prediction_example1_image_to_scalars.png` - scalars should still work
- `prediction_example2_image_to_spectrum.png` - spectrum should have some correlation
- `prediction_example4_reconstruct_image.png` - **THIS SHOULD NOW SHOW STRUCTURE!**

---

## Troubleshooting

### If codec doesn't load:
- Check `image_codec_final.pt` exists in current directory
- Try absolute path: `/home/gustavo/fma/image_codec_final.pt`

### If images still flat after retraining:
- Verify codec loaded: Look for "✓ Loaded trained image codec" in training output
- Check `codec_reconstruction_samples.png` to verify codec quality
- If codec MSE > 0.02, retrain codec for more epochs

### If training crashes:
- Make sure codec is frozen (Option A)
- Check GPU memory (codec + transformer might be too large)

---

## Summary

**What changed:**
1. ✅ Trained image codec can now compress/decompress images properly
2. 🔄 Codec manager loads trained weights automatically
3. 🔄 Training uses frozen codec (or fine-tunes with low LR)
4. 🔄 Inference uses trained codec

**What you need to do:**
1. Update `step4_codec_manager.py` (one method)
2. Update `train_astronomical.py` (3 lines)
3. Update `inference_astronomical.py` (3 lines)
4. Retrain transformer
5. Test inference

After this, your images should have structure instead of being flat noise!
