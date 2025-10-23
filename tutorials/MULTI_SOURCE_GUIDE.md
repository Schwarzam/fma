# Multi-Source Modality Guide

How to handle multiple images from different surveys and multiple tables in your multimodal system.

This is exactly what AION does - it handles images from Legacy Survey, HSC, and Gaia, plus multiple catalogs!

---

## Table of Contents

1. [Multiple Image Sources](#multiple-image-sources)
2. [Multiple Tabular Sources](#multiple-tabular-sources)
3. [Complete Example](#complete-example)
4. [Advanced: Survey-Specific Processing](#advanced-survey-specific-processing)

---

## Multiple Image Sources

### Approach 1: Separate Modality Per Survey

**Best for:** Different image properties (bands, resolutions, preprocessing)

```python
"""
Define separate modalities for each survey
"""

from dataclasses import dataclass
from typing import ClassVar
from jaxtyping import Float
import torch
from torch import Tensor

@dataclass
class LegacySurveyImage(BaseModality):
    """Images from Legacy Survey (g, r, i, z bands)"""
    token_key: ClassVar[str] = "tok_image_legacy"
    num_tokens: ClassVar[int] = 256  # 16×16 tokens

    flux: Float[Tensor, "batch 4 height width"]  # 4 bands
    bands: list[str]  # ["g", "r", "i", "z"]
    metadata: dict = None

    def __post_init__(self):
        assert self.flux.ndim == 4
        assert self.flux.shape[1] == 4, "Legacy Survey has 4 bands"


@dataclass
class HSCImage(BaseModality):
    """Images from Hyper Suprime-Cam (g, r, i, z, y bands)"""
    token_key: ClassVar[str] = "tok_image_hsc"
    num_tokens: ClassVar[int] = 256

    flux: Float[Tensor, "batch 5 height width"]  # 5 bands
    bands: list[str]  # ["g", "r", "i", "z", "y"]
    metadata: dict = None

    def __post_init__(self):
        assert self.flux.ndim == 4
        assert self.flux.shape[1] == 5, "HSC has 5 bands"


@dataclass
class GaiaImage(BaseModality):
    """Images from Gaia (BP, G, RP bands)"""
    token_key: ClassVar[str] = "tok_image_gaia"
    num_tokens: ClassVar[int] = 256

    flux: Float[Tensor, "batch 3 height width"]  # 3 bands
    bands: list[str]  # ["BP", "G", "RP"]
    metadata: dict = None

    def __post_init__(self):
        assert self.flux.ndim == 4
        assert self.flux.shape[1] == 3, "Gaia has 3 bands"
```

### Create Separate Codecs

```python
"""
One codec per survey type
"""

class LegacySurveyImageCodec(Codec):
    """Codec specifically for Legacy Survey images"""

    def __init__(self):
        super().__init__()
        self.encoder = SimpleImageEncoder(in_channels=4, embedding_dim=64)  # 4 bands
        self.decoder = SimpleImageDecoder(embedding_dim=64, out_channels=4)
        self.projection = nn.Conv2d(64, 4, kernel_size=1)
        self.unprojection = nn.Conv2d(4, 64, kernel_size=1)
        self._quantizer = FiniteScalarQuantizer(levels=[8, 5, 5, 5])

    @property
    def modality(self):
        return LegacySurveyImage

    # ... rest of codec implementation


class HSCImageCodec(Codec):
    """Codec specifically for HSC images"""

    def __init__(self):
        super().__init__()
        self.encoder = SimpleImageEncoder(in_channels=5, embedding_dim=64)  # 5 bands
        self.decoder = SimpleImageDecoder(embedding_dim=64, out_channels=5)
        self.projection = nn.Conv2d(64, 4, kernel_size=1)
        self.unprojection = nn.Conv2d(4, 64, kernel_size=1)
        self._quantizer = FiniteScalarQuantizer(levels=[8, 5, 5, 5])

    @property
    def modality(self):
        return HSCImage

    # ... rest of codec implementation
```

### Update Codec Registry

```python
"""
Register all image codecs
"""

MODALITY_CODEC_MAPPING = {
    # Multiple image sources
    LegacySurveyImage: LegacySurveyImageCodec,
    HSCImage: HSCImageCodec,
    GaiaImage: GaiaImageCodec,

    # Other modalities
    MyTimeSeries: TimeSeriesCodec,
    MyScalar: ScalarCodec,
    MyTabular: TabularCodec,
}
```

### Dataset with Multiple Image Sources

```python
"""
Dataset that can have multiple images per sample
"""

class MultiSurveyDataset(Dataset):
    """Dataset with images from multiple surveys"""

    def __init__(self, data_root: str, manifest_path: str):
        self.data_root = Path(data_root)

        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

    def __getitem__(self, idx):
        sample = self.manifest['samples'][idx]

        modalities = {}

        # Load Legacy Survey image (if available)
        if 'legacy_image' in sample:
            legacy_img = self._load_legacy_image(sample['legacy_image'])
            modalities['legacy_image'] = legacy_img

        # Load HSC image (if available)
        if 'hsc_image' in sample:
            hsc_img = self._load_hsc_image(sample['hsc_image'])
            modalities['hsc_image'] = hsc_img

        # Load Gaia image (if available)
        if 'gaia_image' in sample:
            gaia_img = self._load_gaia_image(sample['gaia_image'])
            modalities['gaia_image'] = gaia_img

        # Load other modalities...
        if 'timeseries' in sample:
            modalities['timeseries'] = self._load_timeseries(sample['timeseries'])

        return modalities

    def _load_legacy_image(self, path: str) -> dict:
        """Load Legacy Survey image"""
        img_path = self.data_root / path
        # Load FITS file or numpy array
        flux = np.load(img_path)  # Shape: [4, H, W]
        return {
            'flux': torch.from_numpy(flux).float(),
            'bands': ['g', 'r', 'i', 'z']
        }

    def _load_hsc_image(self, path: str) -> dict:
        """Load HSC image"""
        img_path = self.data_root / path
        flux = np.load(img_path)  # Shape: [5, H, W]
        return {
            'flux': torch.from_numpy(flux).float(),
            'bands': ['g', 'r', 'i', 'z', 'y']
        }

    def _load_gaia_image(self, path: str) -> dict:
        """Load Gaia image"""
        img_path = self.data_root / path
        flux = np.load(img_path)  # Shape: [3, H, W]
        return {
            'flux': torch.from_numpy(flux).float(),
            'bands': ['BP', 'G', 'RP']
        }
```

### Collate Function for Multiple Images

```python
def collate_multi_survey(batch: List[Dict]) -> Dict[str, List]:
    """Collate function for multiple image sources"""

    collated = {
        'legacy_image': [],
        'hsc_image': [],
        'gaia_image': [],
        'timeseries': [],
        # ... other modalities
    }

    for sample in batch:
        # Legacy Survey images
        if 'legacy_image' in sample:
            flux = sample['legacy_image']['flux'].unsqueeze(0)  # [1, 4, H, W]
            collated['legacy_image'].append(
                LegacySurveyImage(
                    flux=flux,
                    bands=sample['legacy_image']['bands'],
                    metadata={}
                )
            )

        # HSC images
        if 'hsc_image' in sample:
            flux = sample['hsc_image']['flux'].unsqueeze(0)  # [1, 5, H, W]
            collated['hsc_image'].append(
                HSCImage(
                    flux=flux,
                    bands=sample['hsc_image']['bands'],
                    metadata={}
                )
            )

        # Gaia images
        if 'gaia_image' in sample:
            flux = sample['gaia_image']['flux'].unsqueeze(0)  # [1, 3, H, W]
            collated['gaia_image'].append(
                GaiaImage(
                    flux=flux,
                    bands=sample['gaia_image']['bands'],
                    metadata={}
                )
            )

        # ... other modalities

    return collated
```

### Training with Multiple Images

```python
"""
Training loop handles all image sources
"""

for batch in train_loader:
    all_tokens = {}

    with torch.no_grad():
        # Encode each image source separately
        if 'legacy_image' in batch and len(batch['legacy_image']) > 0:
            legacy_flux = torch.cat([m.flux for m in batch['legacy_image']], dim=0)
            legacy_batched = LegacySurveyImage(
                flux=legacy_flux,
                bands=batch['legacy_image'][0].bands,
                metadata={}
            )
            tokens = codec_manager.encode(legacy_batched)
            all_tokens.update(tokens)

        if 'hsc_image' in batch and len(batch['hsc_image']) > 0:
            hsc_flux = torch.cat([m.flux for m in batch['hsc_image']], dim=0)
            hsc_batched = HSCImage(
                flux=hsc_flux,
                bands=batch['hsc_image'][0].bands,
                metadata={}
            )
            tokens = codec_manager.encode(hsc_batched)
            all_tokens.update(tokens)

        if 'gaia_image' in batch and len(batch['gaia_image']) > 0:
            gaia_flux = torch.cat([m.flux for m in batch['gaia_image']], dim=0)
            gaia_batched = GaiaImage(
                flux=gaia_flux,
                bands=batch['gaia_image'][0].bands,
                metadata={}
            )
            tokens = codec_manager.encode(gaia_batched)
            all_tokens.update(tokens)

    # Now all_tokens contains:
    # {
    #     "tok_image_legacy": [B, 256],
    #     "tok_image_hsc": [B, 256],
    #     "tok_image_gaia": [B, 256],
    #     ... other modalities
    # }

    # Train transformer on all tokens
    _, masked = apply_masking(all_tokens, mask_ratio=0.3)
    predictions = model(all_tokens, masked)
    # ... compute loss
```

---

## Multiple Tabular Sources

### Approach: Separate Modality Per Table

```python
"""
Define separate modalities for each tabular source
"""

@dataclass
class PatientDemographics(BaseModality):
    """Patient demographic information"""
    token_key: ClassVar[str] = "tok_demographics"
    num_tokens: ClassVar[int] = 1

    age: Float[Tensor, "batch 1"]
    sex: Float[Tensor, "batch 1"]  # 0=F, 1=M
    bmi: Float[Tensor, "batch 1"]
    metadata: dict = None

    def __post_init__(self):
        assert self.age.ndim == 2 and self.age.shape[1] == 1


@dataclass
class LabResults(BaseModality):
    """Laboratory test results"""
    token_key: ClassVar[str] = "tok_lab_results"
    num_tokens: ClassVar[int] = 1

    glucose: Float[Tensor, "batch 1"]
    cholesterol: Float[Tensor, "batch 1"]
    hemoglobin: Float[Tensor, "batch 1"]
    metadata: dict = None


@dataclass
class VitalSigns(BaseModality):
    """Vital signs measurements"""
    token_key: ClassVar[str] = "tok_vitals"
    num_tokens: ClassVar[int] = 1

    heart_rate: Float[Tensor, "batch 1"]
    blood_pressure_sys: Float[Tensor, "batch 1"]
    blood_pressure_dia: Float[Tensor, "batch 1"]
    temperature: Float[Tensor, "batch 1"]
    metadata: dict = None
```

### Create Codecs for Each Table

```python
"""
Separate codec for each tabular source
"""

class DemographicsCodec(Codec):
    """Codec for demographic data"""

    def __init__(self):
        super().__init__()
        # Encode 3 features (age, sex, bmi) together
        self.encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self._quantizer = VectorQuantizer(num_embeddings=256, embedding_dim=64)

    @property
    def modality(self):
        return PatientDemographics

    def _encode(self, x: PatientDemographics):
        # Concatenate all features
        features = torch.cat([x.age, x.sex, x.bmi], dim=1)  # [B, 3]
        z = self.encoder(features)  # [B, 64]
        return z.unsqueeze(-1)  # [B, 64, 1]

    def _decode(self, z, **metadata):
        z = z.squeeze(-1)  # [B, 64]
        features = self.decoder(z)  # [B, 3]
        return PatientDemographics(
            age=features[:, 0:1],
            sex=features[:, 1:2],
            bmi=features[:, 2:3],
            metadata=metadata
        )

    @property
    def quantizer(self):
        return self._quantizer


class LabResultsCodec(Codec):
    """Codec for lab results"""
    # Similar structure, encoding 3 lab values
    # ...


class VitalSignsCodec(Codec):
    """Codec for vital signs"""
    # Similar structure, encoding 4 vital measurements
    # ...
```

### Register All Codecs

```python
MODALITY_CODEC_MAPPING = {
    # Images from different surveys
    LegacySurveyImage: LegacySurveyImageCodec,
    HSCImage: HSCImageCodec,
    GaiaImage: GaiaImageCodec,

    # Multiple tabular sources
    PatientDemographics: DemographicsCodec,
    LabResults: LabResultsCodec,
    VitalSigns: VitalSignsCodec,

    # Other modalities
    MyTimeSeries: TimeSeriesCodec,
}
```

### Dataset with Multiple Tables

```python
class MultiTableDataset(Dataset):
    """Dataset with multiple tabular sources"""

    def __init__(self, data_root: str, manifest_path: str):
        self.data_root = Path(data_root)

        # Load all tables
        self.demographics_df = pd.read_csv(data_root + '/demographics.csv')
        self.lab_results_df = pd.read_csv(data_root + '/lab_results.csv')
        self.vitals_df = pd.read_csv(data_root + '/vital_signs.csv')

        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

    def __getitem__(self, idx):
        sample = self.manifest['samples'][idx]

        modalities = {}

        # Demographics
        if 'demographics_row' in sample:
            demo_row = self.demographics_df.iloc[sample['demographics_row']]
            modalities['demographics'] = {
                'age': demo_row['age'],
                'sex': demo_row['sex'],
                'bmi': demo_row['bmi']
            }

        # Lab results
        if 'lab_results_row' in sample:
            lab_row = self.lab_results_df.iloc[sample['lab_results_row']]
            modalities['lab_results'] = {
                'glucose': lab_row['glucose'],
                'cholesterol': lab_row['cholesterol'],
                'hemoglobin': lab_row['hemoglobin']
            }

        # Vital signs
        if 'vitals_row' in sample:
            vital_row = self.vitals_df.iloc[sample['vitals_row']]
            modalities['vitals'] = {
                'heart_rate': vital_row['heart_rate'],
                'blood_pressure_sys': vital_row['bp_sys'],
                'blood_pressure_dia': vital_row['bp_dia'],
                'temperature': vital_row['temperature']
            }

        # Images
        if 'image' in sample:
            modalities['image'] = self._load_image(sample['image'])

        # Time series
        if 'timeseries' in sample:
            modalities['timeseries'] = self._load_timeseries(sample['timeseries'])

        return modalities
```

### Collate for Multiple Tables

```python
def collate_multi_table(batch: List[Dict]) -> Dict[str, List]:
    """Collate function for multiple tables"""

    collated = {
        'demographics': [],
        'lab_results': [],
        'vitals': [],
        'image': [],
        'timeseries': []
    }

    for sample in batch:
        # Demographics
        if 'demographics' in sample:
            collated['demographics'].append(
                PatientDemographics(
                    age=torch.tensor([[sample['demographics']['age']]], dtype=torch.float32),
                    sex=torch.tensor([[sample['demographics']['sex']]], dtype=torch.float32),
                    bmi=torch.tensor([[sample['demographics']['bmi']]], dtype=torch.float32),
                    metadata={}
                )
            )

        # Lab results
        if 'lab_results' in sample:
            collated['lab_results'].append(
                LabResults(
                    glucose=torch.tensor([[sample['lab_results']['glucose']]], dtype=torch.float32),
                    cholesterol=torch.tensor([[sample['lab_results']['cholesterol']]], dtype=torch.float32),
                    hemoglobin=torch.tensor([[sample['lab_results']['hemoglobin']]], dtype=torch.float32),
                    metadata={}
                )
            )

        # Vital signs
        if 'vitals' in sample:
            collated['vitals'].append(
                VitalSigns(
                    heart_rate=torch.tensor([[sample['vitals']['heart_rate']]], dtype=torch.float32),
                    blood_pressure_sys=torch.tensor([[sample['vitals']['blood_pressure_sys']]], dtype=torch.float32),
                    blood_pressure_dia=torch.tensor([[sample['vitals']['blood_pressure_dia']]], dtype=torch.float32),
                    temperature=torch.tensor([[sample['vitals']['temperature']]], dtype=torch.float32),
                    metadata={}
                )
            )

        # Images, time series, etc.
        # ...

    return collated
```

---

## Complete Example

### Manifest File with Multiple Sources

```json
{
    "samples": [
        {
            "id": "patient_001",
            "legacy_image": "images/legacy/patient_001.npy",
            "hsc_image": "images/hsc/patient_001.npy",
            "demographics_row": 0,
            "lab_results_row": 0,
            "vitals_row": 0,
            "timeseries": "timeseries/patient_001_ecg.npy"
        },
        {
            "id": "patient_002",
            "legacy_image": "images/legacy/patient_002.npy",
            "gaia_image": "images/gaia/patient_002.npy",
            "demographics_row": 1,
            "lab_results_row": 1,
            "vitals_row": 1,
            "timeseries": "timeseries/patient_002_ecg.npy"
        },
        {
            "id": "patient_003",
            "hsc_image": "images/hsc/patient_003.npy",
            "demographics_row": 2,
            "vitals_row": 2,
            "timeseries": "timeseries/patient_003_ecg.npy"
        }
    ]
}
```

Note: Patient 3 doesn't have lab results or Legacy Survey image - that's fine!

### Training with All Sources

```python
"""
Complete training with multiple images and tables
"""

# Create transformer with all modalities
vocab_sizes = {
    # Images
    "tok_image_legacy": 10000,
    "tok_image_hsc": 10000,
    "tok_image_gaia": 10000,

    # Tables
    "tok_demographics": 256,
    "tok_lab_results": 256,
    "tok_vitals": 256,

    # Other
    "tok_timeseries": 512,
}

num_tokens = {
    # Images
    "tok_image_legacy": 256,
    "tok_image_hsc": 256,
    "tok_image_gaia": 256,

    # Tables
    "tok_demographics": 1,
    "tok_lab_results": 1,
    "tok_vitals": 1,

    # Other
    "tok_timeseries": 128,
}

model = MultimodalTransformer(
    vocab_sizes=vocab_sizes,
    num_tokens=num_tokens,
    d_model=512,
    nhead=8,
    num_layers=12
)

# Training loop
for batch in train_loader:
    all_tokens = {}

    with torch.no_grad():
        # Encode all modalities that are present
        for modality_name, modality_list in batch.items():
            if len(modality_list) == 0:
                continue  # Skip if no samples have this modality

            # Concatenate and encode
            if modality_name == 'legacy_image':
                flux = torch.cat([m.flux for m in modality_list], dim=0)
                batched = LegacySurveyImage(flux=flux, bands=modality_list[0].bands)
            elif modality_name == 'hsc_image':
                flux = torch.cat([m.flux for m in modality_list], dim=0)
                batched = HSCImage(flux=flux, bands=modality_list[0].bands)
            elif modality_name == 'demographics':
                age = torch.cat([m.age for m in modality_list], dim=0)
                sex = torch.cat([m.sex for m in modality_list], dim=0)
                bmi = torch.cat([m.bmi for m in modality_list], dim=0)
                batched = PatientDemographics(age=age, sex=sex, bmi=bmi)
            # ... handle all other modalities

            tokens = codec_manager.encode(batched)
            all_tokens.update(tokens)

    # Now train on whatever modalities are present
    _, masked = apply_masking(all_tokens, mask_ratio=0.3)
    predictions = model(all_tokens, masked)

    # Compute loss
    loss = 0
    for key in masked:
        if key in predictions:
            pred = predictions[key]
            target = all_tokens[key]
            loss += F.cross_entropy(
                pred.reshape(-1, pred.size(-1)),
                target.reshape(-1).long()
            )

    # Backward pass
    loss.backward()
    optimizer.step()
```

---

## Advanced: Survey-Specific Processing

### Different Preprocessing Per Survey

```python
class LegacySurveyImageCodec(Codec):
    """Codec with Legacy Survey-specific preprocessing"""

    def __init__(self):
        super().__init__()
        self.encoder = SimpleImageEncoder(in_channels=4, embedding_dim=64)
        self.decoder = SimpleImageDecoder(embedding_dim=64, out_channels=4)

        # Legacy Survey-specific preprocessing
        self.legacy_preprocessor = LegacySurveyPreprocessor()

        self.projection = nn.Conv2d(64, 4, kernel_size=1)
        self.unprojection = nn.Conv2d(4, 64, kernel_size=1)
        self._quantizer = FiniteScalarQuantizer(levels=[8, 5, 5, 5])

    def _encode(self, x: LegacySurveyImage):
        # Apply survey-specific preprocessing
        flux = self.legacy_preprocessor.preprocess(x.flux, x.bands)

        # Encode
        z = self.encoder(flux)
        z = self.projection(z)
        return z


class LegacySurveyPreprocessor(nn.Module):
    """Survey-specific preprocessing"""

    def __init__(self):
        super().__init__()
        # Legacy Survey zeropoints (magnitudes)
        self.zeropoints = {
            'g': 22.5,
            'r': 22.5,
            'i': 22.5,
            'z': 22.5
        }

    def preprocess(self, flux: torch.Tensor, bands: list[str]) -> torch.Tensor:
        """
        Apply Legacy Survey-specific preprocessing:
        1. Convert to nanomaggies
        2. Apply zeropoint calibration
        3. Arcsinh stretch
        """
        processed = []

        for i, band in enumerate(bands):
            band_flux = flux[:, i:i+1]

            # Convert to magnitudes if needed
            # mag = -2.5 * log10(flux) + zeropoint

            # Arcsinh stretch (better for astronomical data)
            stretched = torch.arcsinh(10 * band_flux) / 3

            processed.append(stretched)

        return torch.cat(processed, dim=1)
```

### Cross-Survey Alignment

If you want the transformer to learn that images from different surveys show the same sky:

```python
"""
Align different survey images to same resolution/pixel scale
"""

class SurveyAligner:
    """Align images from different surveys"""

    def __init__(self):
        self.target_pixel_scale = 0.262  # arcsec/pixel
        self.target_size = 224  # pixels

    def align_legacy(self, image: np.ndarray) -> np.ndarray:
        """Legacy Survey has 0.262 arcsec/pixel - already aligned"""
        return image

    def align_hsc(self, image: np.ndarray) -> np.ndarray:
        """HSC has 0.168 arcsec/pixel - resample to Legacy scale"""
        scale_factor = 0.168 / 0.262  # ~0.64
        # Resample using scipy or torch.nn.functional.interpolate
        resampled = self._resample(image, scale_factor)
        return resampled

    def align_gaia(self, image: np.ndarray) -> np.ndarray:
        """Gaia has different pixel scale - resample"""
        # Implement resampling
        pass
```

---

## Benefits of This Approach

### 1. **Flexible Data Availability**
- Not all samples need all modalities
- Model learns from whatever is available
- Can predict missing surveys from present ones

### 2. **Survey-Specific Learning**
- Each codec learns survey-specific characteristics
- Different preprocessing per survey
- Maintains data fidelity

### 3. **Cross-Survey Correlations**
- Transformer learns relationships between surveys
- Can predict HSC image from Legacy image
- Can impute missing data

### 4. **Easy to Extend**
- Add new survey: define new modality + codec
- Add new table: define new modality + codec
- No need to retrain existing codecs

---

## Summary

**For Multiple Images:**
1. Define separate modality class per survey
2. Create separate codec per survey (different bands/preprocessing)
3. Each gets unique `token_key`
4. Transformer sees all as separate token streams

**For Multiple Tables:**
1. Define separate modality class per table
2. Create separate codec per table
3. Each gets unique `token_key`
4. Can encode related fields together (demographics as one modality)

**Key Insight:**
The transformer doesn't care about modality types - it just sees token streams with different `token_key` identifiers. You can have as many as you want!

This is exactly how AION handles:
- Legacy Survey images (g,r,i,z)
- HSC images (g,r,i,z,y)
- Gaia photometry (G,BP,RP)
- Gaia spectra
- SDSS/DESI spectra
- 33 different scalar measurements
- Multiple catalogs

All in one model! 🎉
