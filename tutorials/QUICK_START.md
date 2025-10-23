# Quick Start: Astronomical Multimodal Training

Complete example with synthetic astronomical data (galaxies, spectra, light curves).

---

## Setup

### 1. Install Dependencies

```bash
pip install torch torchvision numpy scipy matplotlib tqdm
```

### 2. Generate Synthetic Data

```bash
# Generate 1000 samples (~2 minutes)
python example_data_generator.py --num_samples 1000 --visualize

# Or generate more for better training
python example_data_generator.py --num_samples 10000
```

**This creates:**
```
example_data/
├── images/           # Galaxy images (g,r,i bands)
├── spectra/          # Gaia BP/RP spectra
├── lightcurves/      # ZTF photometric time series
└── metadata/
    ├── manifest.json
    ├── train_manifest.json  (70% of data)
    ├── val_manifest.json    (15% of data)
    └── test_manifest.json   (15% of data)
```

### 3. Test Data Loading

```bash
python astronomical_dataset.py
```

**Expected output:**
```
Testing astronomical dataset...

Dataset size: 700

Testing single sample...
  Image shape: torch.Size([3, 96, 96])
  Spectrum flux shape: torch.Size([343])
  Light curve flux shape: torch.Size([100])
  Redshift: 0.856
  Stellar mass: 10.24 log(M☉)
  SFR: 45.3 M☉/yr

✓ Dataset test complete!
```

---

## Training

### Quick Training (3 epochs, ~5 minutes)

```bash
python train_astronomical.py
```

**What this does:**
1. Loads train/val data
2. Creates codec manager (frozen encoders)
3. Creates small transformer (256 dim, 4 layers)
4. Trains with multimodal masked modeling
5. Saves model to `astronomical_model.pt`

**Expected output:**
```
Using device: cuda

Loading datasets...
Train: 700 samples
Val: 150 samples

Creating transformer...
Model parameters: 8,234,752

Training for 3 epochs...
Epoch 1/3: 100%|████████| 175/175 [01:23<00:00, loss: 2.3451, acc: 0.4521]
Epoch 1 - Loss: 2.3451, Acc: 0.4521

Epoch 2/3: 100%|████████| 175/175 [01:22<00:00, loss: 1.8923, acc: 0.5234]
Epoch 2 - Loss: 1.8923, Acc: 0.5234

Epoch 3/3: 100%|████████| 175/175 [01:21<00:00, loss: 1.5432, acc: 0.5876]
Epoch 3 - Loss: 1.5432, Acc: 0.5876

✓ Training complete!
✓ Saved to: ./astronomical_model.pt
```

---

## Data Generated

### Galaxy Images

**Elliptical galaxies:**
- Smooth, red
- Old stellar populations
- Sersic profile

**Spiral galaxies:**
- Blue with spiral arms
- Young stars, star formation
- Exponential disk + arms

**Realistic features:**
- Dimmer at higher redshift
- Brighter with higher mass
- Random orientations
- Poisson noise

### Gaia BP/RP Spectra

**330-1050 nm, 343 wavelength bins**

**Features:**
- Blackbody continuum (temperature depends on galaxy type)
- 4000Å break (stronger in ellipticals)
- Emission lines (H-alpha, H-beta, [OIII] in spirals)
- Redshifted based on z
- Realistic S/N ~ 50

### ZTF Light Curves

**g-band photometry, ~100 observations over 3 years**

**Features:**
- Irregular cadence (realistic ZTF observing)
- Seasonal variations
- Supernovae transients (in spirals with high SFR)
- Photometric noise
- Magnitude vs flux

### Physical Parameters

**Correlated quantities:**
- Redshift: 0.0 - 2.0
- Stellar mass: 10^9 - 10^12 M☉
- Star formation rate: 0.1 - 100 M☉/yr
- Galaxy type: Elliptical (30%) or Spiral (70%)

---

## Understanding the Pipeline

### What Each File Does

```
example_data_generator.py
    ↓ Generates synthetic data
example_data/
    ↓ Loaded by
astronomical_dataset.py
    ↓ Feeds data to
train_astronomical.py
    ↓ Uses
step1-6_*.py (multimodal system)
    ↓ Outputs
astronomical_model.pt (trained model)
```

### The Modalities

```python
# 6 modalities in this example:
GalaxyImage         → tok_galaxy_image      (784 tokens)
GaiaSpectrum        → tok_gaia_spectrum     (128 tokens)
ZTFLightCurve       → tok_ztf_lightcurve    (100 tokens)
Redshift            → tok_redshift          (1 token)
StellarMass         → tok_stellar_mass      (1 token)
StarFormationRate   → tok_sfr               (1 token)
```

### What the Model Learns

**Training objective:** Predict masked modalities from visible ones

**Example predictions:**
- Image → Spectrum (morphology → stellar population)
- Image → Redshift (appearance → distance)
- Spectrum → SFR (emission lines → star formation)
- Light curve → Galaxy type (variability → morphology)

---

## Next Steps

### 1. Visualize Predictions

```python
from astronomical_inference import predict_from_image

# Load model
model, codec_manager = load_trained_model('./astronomical_model.pt')

# Load test image
image = load_galaxy_image('./example_data/images/galaxy_000000.npy')

# Predict everything else
predictions = predict_from_image(model, codec_manager, image)

print(f"Predicted redshift: {predictions['redshift']:.3f}")
print(f"Predicted mass: {predictions['stellar_mass']:.2f}")
print(f"Predicted SFR: {predictions['sfr']:.1f}")
```

### 2. Train Longer

```bash
# Modify train_astronomical.py
num_epochs = 30  # Instead of 3
d_model = 512    # Bigger model
num_layers = 8

python train_astronomical.py
```

### 3. Add More Data

```bash
# Generate 10K samples
python example_data_generator.py --num_samples 10000

# Train on larger dataset
python train_astronomical.py
```

### 4. Add Your Own Modalities

**Example: Add WISE infrared photometry**

```python
# In astronomical_dataset.py

@dataclass
class WISEPhotometry(BaseModality):
    """WISE W1, W2, W3, W4 bands"""
    token_key: ClassVar[str] = "tok_wise"
    num_tokens: ClassVar[int] = 4

    w1: Float[Tensor, "batch 1"]
    w2: Float[Tensor, "batch 1"]
    w3: Float[Tensor, "batch 1"]
    w4: Float[Tensor, "batch 1"]
```

### 5. Use Real Data

Replace `example_data_generator.py` with your real data loader:

```python
class RealAstronomicalDataset(Dataset):
    def __init__(self, fits_dir, catalog_file):
        # Load FITS images
        # Load spectroscopic catalog
        # Load light curve database
        pass

    def __getitem__(self, idx):
        # Load real FITS file
        # Load real spectrum
        # Load real light curve
        pass
```

---

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
batch_size = 2  # Instead of 4

# Reduce model size
d_model = 128  # Instead of 256
num_layers = 3  # Instead of 4
```

### Slow Training

```python
# Use more workers
DataLoader(..., num_workers=4)

# Use GPU if available
device = "cuda"  # Auto-detected

# Reduce data size for testing
python example_data_generator.py --num_samples 100
```

### Import Errors

```bash
# Make sure you're in the /new directory
cd /Users/gustavoschwarz/Documents/AION/new

# All imports are relative
python train_astronomical.py
```

---

## File Structure

```
new/
├── step1_define_modalities.py       # Base modality classes
├── step2_define_quantizers.py       # Quantization methods
├── step3_implement_codecs.py        # Encoder/decoder networks
├── step4_codec_manager.py           # Codec orchestration
├── step5_train_codecs.py            # Codec training (optional)
├── step6_train_transformer.py       # Transformer training
│
├── example_data_generator.py        # ← Generate synthetic data
├── astronomical_dataset.py          # ← Load astronomical data
├── train_astronomical.py            # ← Train on astronomical data
│
├── QUICK_START.md                   # This file
├── TOKENIZER_INSTRUCTIONS.md        # Architecture deep dive
├── PRODUCTION_GUIDE.md              # Scale to production
├── MULTI_SOURCE_GUIDE.md            # Multiple surveys/tables
├── INFERENCE_GUIDE.md               # Using trained models
└── DATA_REQUIREMENTS.md             # Data needs & strategies
```

---

## Summary

**5-minute pipeline:**

```bash
# 1. Generate data
python example_data_generator.py --num_samples 1000 --visualize

# 2. Test loading
python astronomical_dataset.py

# 3. Train model
python train_astronomical.py

# Done! You have a trained multimodal model.
```

**What you get:**
- Trained model that understands 6 astronomical modalities
- Can predict missing data from observed data
- Can generate multimodal embeddings
- Ready to scale to real data

🚀 **Now you have a complete working example!**
