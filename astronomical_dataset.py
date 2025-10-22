"""
astronomical_dataset.py - Dataset loader for astronomical data

Loads generated astronomical data (images, spectra, light curves)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import ClassVar
from jaxtyping import Float, Bool
from torch import Tensor

from step1_define_modalities import BaseModality


# ============================================================================
# Astronomical Modalities
# ============================================================================

@dataclass
class GalaxyImage(BaseModality):
    """Galaxy image (g, r, i bands)"""
    token_key: ClassVar[str] = "tok_galaxy_image"
    num_tokens: ClassVar[int] = 256  # Will be 14*14*4 = 784 after encoding

    flux: Float[Tensor, "batch 3 height width"]  # 3 bands
    metadata: dict = None

    def __post_init__(self):
        assert self.flux.ndim == 4, f"Expected 4D, got {self.flux.ndim}D"
        assert self.flux.shape[1] == 3, f"Expected 3 bands, got {self.flux.shape[1]}"


@dataclass
class GaiaSpectrum(BaseModality):
    """Gaia BP/RP spectrum"""
    token_key: ClassVar[str] = "tok_gaia_spectrum"
    num_tokens: ClassVar[int] = 128  # Downsampled

    flux: Float[Tensor, "batch wavelength"]
    ivar: Float[Tensor, "batch wavelength"]
    mask: Bool[Tensor, "batch wavelength"]
    wavelength: Float[Tensor, "wavelength"]
    metadata: dict = None

    def __post_init__(self):
        assert self.flux.ndim == 2


@dataclass
class ZTFLightCurve(BaseModality):
    """ZTF photometric light curve"""
    token_key: ClassVar[str] = "tok_ztf_lightcurve"
    num_tokens: ClassVar[int] = 100  # Max 100 observations

    flux: Float[Tensor, "batch time"]
    flux_err: Float[Tensor, "batch time"]
    mjd: Float[Tensor, "time"]
    metadata: dict = None

    def __post_init__(self):
        assert self.flux.ndim == 2


@dataclass
class Redshift(BaseModality):
    """Cosmological redshift"""
    token_key: ClassVar[str] = "tok_redshift"
    num_tokens: ClassVar[int] = 1

    value: Float[Tensor, "batch 1"]
    metadata: dict = None

    def __post_init__(self):
        assert self.value.ndim == 2


@dataclass
class StellarMass(BaseModality):
    """Stellar mass (log10 solar masses)"""
    token_key: ClassVar[str] = "tok_stellar_mass"
    num_tokens: ClassVar[int] = 1

    value: Float[Tensor, "batch 1"]
    metadata: dict = None

    def __post_init__(self):
        assert self.value.ndim == 2


@dataclass
class StarFormationRate(BaseModality):
    """Star formation rate (solar masses per year)"""
    token_key: ClassVar[str] = "tok_sfr"
    num_tokens: ClassVar[int] = 1

    value: Float[Tensor, "batch 1"]
    metadata: dict = None

    def __post_init__(self):
        assert self.value.ndim == 2


# ============================================================================
# Dataset
# ============================================================================

class AstronomicalDataset(Dataset):
    """Dataset for astronomical multimodal data"""

    def __init__(
        self,
        data_root: str,
        manifest_path: str,
        image_size: int = 96
    ):
        """
        Args:
            data_root: Root directory with data
            manifest_path: Path to manifest JSON
            image_size: Target image size
        """
        self.data_root = Path(data_root)
        self.image_size = image_size

        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        self.samples = manifest['samples']

        print(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            # Load image
            image_data = self._load_image(sample['image'])

            # Load spectrum
            spectrum_data = self._load_spectrum(sample['spectrum'])

            # Load light curve
            lightcurve_data = self._load_lightcurve(sample['lightcurve'])

            # Physical parameters
            redshift = torch.tensor([sample['redshift']], dtype=torch.float32)
            stellar_mass = torch.tensor([sample['stellar_mass']], dtype=torch.float32)
            sfr = torch.tensor([sample['sfr']], dtype=torch.float32)

            return {
                'image': image_data,
                'spectrum': spectrum_data,
                'lightcurve': lightcurve_data,
                'redshift': redshift,
                'stellar_mass': stellar_mass,
                'sfr': sfr,
                'sample_id': sample['id']
            }

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return next sample
            return self.__getitem__((idx + 1) % len(self))

    def _load_image(self, rel_path: str) -> torch.Tensor:
        """Load galaxy image"""
        img_path = self.data_root / rel_path
        flux = np.load(img_path)  # [3, H, W]

        # Convert to tensor
        flux = torch.from_numpy(flux).float()

        # Resize if needed
        if flux.shape[1] != self.image_size or flux.shape[2] != self.image_size:
            flux = torch.nn.functional.interpolate(
                flux.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        return flux

    def _load_spectrum(self, rel_path: str) -> dict:
        """Load Gaia spectrum"""
        spec_path = self.data_root / rel_path
        spec_data = np.load(spec_path, allow_pickle=True).item()

        return {
            'flux': torch.from_numpy(spec_data['flux']).float(),
            'ivar': torch.from_numpy(spec_data['ivar']).float(),
            'mask': torch.from_numpy(spec_data['mask']),
            'wavelength': torch.from_numpy(spec_data['wavelength']).float()
        }

    def _load_lightcurve(self, rel_path: str) -> dict:
        """Load ZTF light curve"""
        lc_path = self.data_root / rel_path
        lc_data = np.load(lc_path, allow_pickle=True).item()

        return {
            'mjd': torch.from_numpy(lc_data['mjd']).float(),
            'flux': torch.from_numpy(lc_data['flux']).float(),
            'flux_err': torch.from_numpy(lc_data['flux_err']).float()
        }


def collate_astronomical(batch):
    """Collate function for astronomical data"""

    images = []
    spectra = []
    lightcurves = []
    redshifts = []
    stellar_masses = []
    sfrs = []
    sample_ids = []

    for sample in batch:
        # Image
        images.append(sample['image'].unsqueeze(0))  # [1, 3, H, W]

        # Spectrum
        spectra.append({
            'flux': sample['spectrum']['flux'].unsqueeze(0),  # [1, wavelength]
            'ivar': sample['spectrum']['ivar'].unsqueeze(0),
            'mask': sample['spectrum']['mask'].unsqueeze(0),
            'wavelength': sample['spectrum']['wavelength']
        })

        # Light curve (pad to max length in batch)
        lightcurves.append({
            'flux': sample['lightcurve']['flux'].unsqueeze(0),
            'flux_err': sample['lightcurve']['flux_err'].unsqueeze(0),
            'mjd': sample['lightcurve']['mjd']
        })

        # Scalars
        redshifts.append(sample['redshift'].unsqueeze(0))
        stellar_masses.append(sample['stellar_mass'].unsqueeze(0))
        sfrs.append(sample['sfr'].unsqueeze(0))

        sample_ids.append(sample['sample_id'])

    # Create modality objects
    image_modalities = [
        GalaxyImage(flux=img, metadata={'id': sid})
        for img, sid in zip(images, sample_ids)
    ]

    spectrum_modalities = [
        GaiaSpectrum(
            flux=spec['flux'],
            ivar=spec['ivar'],
            mask=spec['mask'],
            wavelength=spec['wavelength'],
            metadata={}
        )
        for spec in spectra
    ]

    # For light curves, pad to same length
    max_len = max(lc['flux'].shape[1] for lc in lightcurves)

    lightcurve_modalities = []
    for lc in lightcurves:
        flux = lc['flux']
        flux_err = lc['flux_err']
        mjd = lc['mjd']

        # Pad
        if flux.shape[1] < max_len:
            pad_len = max_len - flux.shape[1]
            flux = torch.cat([flux, torch.zeros(1, pad_len)], dim=1)
            flux_err = torch.cat([flux_err, torch.ones(1, pad_len) * 1e10], dim=1)  # High error for padding
            mjd = torch.cat([mjd, torch.zeros(pad_len)])

        lightcurve_modalities.append(
            ZTFLightCurve(
                flux=flux,
                flux_err=flux_err,
                mjd=mjd,
                metadata={}
            )
        )

    redshift_modalities = [
        Redshift(value=z, metadata={}) for z in redshifts
    ]

    mass_modalities = [
        StellarMass(value=m, metadata={}) for m in stellar_masses
    ]

    sfr_modalities = [
        StarFormationRate(value=s, metadata={}) for s in sfrs
    ]

    return {
        'galaxy_image': image_modalities,
        'gaia_spectrum': spectrum_modalities,
        'ztf_lightcurve': lightcurve_modalities,
        'redshift': redshift_modalities,
        'stellar_mass': mass_modalities,
        'sfr': sfr_modalities,
        'sample_ids': sample_ids
    }


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("Testing astronomical dataset...")

    # Create dataset
    dataset = AstronomicalDataset(
        data_root="./example_data",
        manifest_path="./example_data/metadata/train_manifest.json"
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test single sample
    print("\nTesting single sample...")
    sample = dataset[0]
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Spectrum flux shape: {sample['spectrum']['flux'].shape}")
    print(f"  Light curve flux shape: {sample['lightcurve']['flux'].shape}")
    print(f"  Redshift: {sample['redshift'].item():.3f}")
    print(f"  Stellar mass: {sample['stellar_mass'].item():.2f} log(M☉)")
    print(f"  SFR: {sample['sfr'].item():.1f} M☉/yr")

    # Test dataloader
    print("\nTesting DataLoader with collate...")
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_astronomical
    )

    batch = next(iter(loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"  Images: {len(batch['galaxy_image'])} samples")
    print(f"    - First image shape: {batch['galaxy_image'][0].flux.shape}")
    print(f"  Spectra: {len(batch['gaia_spectrum'])} samples")
    print(f"    - First spectrum shape: {batch['gaia_spectrum'][0].flux.shape}")
    print(f"  Light curves: {len(batch['ztf_lightcurve'])} samples")
    print(f"    - First LC shape: {batch['ztf_lightcurve'][0].flux.shape}")

    print("\n✓ Dataset test complete!")
