"""
example_data_generator.py - Generate synthetic astronomical data

Creates realistic astronomical data for testing:
- Galaxy images (RGB)
- Gaia BP/RP spectra
- ZTF light curves
- Photometric catalogs
- Physical parameters (redshift, stellar mass, etc.)

Run this to generate a small dataset for testing the pipeline.
"""

import numpy as np
import torch
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


class AstronomicalDataGenerator:
    """Generate synthetic astronomical data"""

    def __init__(self, output_dir: str = "./example_data", num_samples: int = 1000):
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples

        # Create directories
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "spectra").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "lightcurves").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metadata").mkdir(parents=True, exist_ok=True)

        # Physical parameter ranges
        self.redshift_range = (0.0, 2.0)
        self.stellar_mass_range = (9.0, 12.0)  # log10(M_solar)
        self.sfr_range = (0.1, 100.0)  # M_solar/yr

    def generate_all(self):
        """Generate complete dataset"""
        print(f"Generating {self.num_samples} astronomical objects...")

        manifest = {"samples": []}

        for i in tqdm(range(self.num_samples)):
            # Sample physical parameters
            redshift = np.random.uniform(*self.redshift_range)
            stellar_mass = np.random.uniform(*self.stellar_mass_range)
            sfr = np.random.uniform(*self.sfr_range)

            # Galaxy type (0=elliptical, 1=spiral)
            galaxy_type = np.random.choice([0, 1], p=[0.3, 0.7])

            # Generate data
            sample_id = f"galaxy_{i:06d}"

            # 1. Galaxy image
            image_path = self.generate_galaxy_image(
                i, redshift, stellar_mass, galaxy_type
            )

            # 2. Gaia spectrum
            spectrum_path = self.generate_gaia_spectrum(
                i, redshift, stellar_mass, galaxy_type
            )

            # 3. ZTF light curve
            lightcurve_path = self.generate_ztf_lightcurve(
                i, galaxy_type, sfr
            )

            # Add to manifest
            manifest["samples"].append({
                "id": sample_id,
                "image": str(image_path.relative_to(self.output_dir)),
                "spectrum": str(spectrum_path.relative_to(self.output_dir)),
                "lightcurve": str(lightcurve_path.relative_to(self.output_dir)),
                "redshift": float(redshift),
                "stellar_mass": float(stellar_mass),
                "sfr": float(sfr),
                "galaxy_type": int(galaxy_type)
            })

        # Save manifest
        manifest_path = self.output_dir / "metadata" / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"\n✓ Generated {self.num_samples} samples")
        print(f"✓ Saved to: {self.output_dir}")
        print(f"✓ Manifest: {manifest_path}")

        # Generate split manifests
        self._create_splits(manifest)

        return manifest

    def generate_galaxy_image(
        self,
        idx: int,
        redshift: float,
        stellar_mass: float,
        galaxy_type: int
    ) -> Path:
        """
        Generate synthetic galaxy image (96x96, 3 bands: g, r, i)

        Features:
        - Elliptical: smooth, red
        - Spiral: arms, blue
        - Dimmer at higher redshift
        - Brighter with higher mass
        """
        size = 96

        # Base brightness (depends on mass and redshift)
        base_brightness = 10 ** (stellar_mass - 10) / (1 + redshift)

        if galaxy_type == 0:
            # Elliptical galaxy
            image = self._create_elliptical(size, base_brightness)
        else:
            # Spiral galaxy
            image = self._create_spiral(size, base_brightness)

        # Add noise
        noise = np.random.randn(3, size, size) * 0.05
        image = image + noise

        # Clip to [0, 1]
        image = np.clip(image, 0, 1)

        # Save
        save_path = self.output_dir / "images" / f"galaxy_{idx:06d}.npy"
        np.save(save_path, image.astype(np.float32))

        return save_path

    def _create_elliptical(self, size: int, brightness: float) -> np.ndarray:
        """Create elliptical galaxy (smooth, red)"""
        center = size // 2
        y, x = np.ogrid[:size, :size]

        # Elliptical profile
        a, b = size // 4, size // 6  # Semi-major, semi-minor axes
        angle = np.random.uniform(0, np.pi)

        x_rot = (x - center) * np.cos(angle) - (y - center) * np.sin(angle)
        y_rot = (x - center) * np.sin(angle) + (y - center) * np.cos(angle)

        r = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2)

        # Sersic profile (n=4 for ellipticals)
        profile = np.exp(-7.67 * (r ** 0.25 - 1))
        profile = profile * brightness

        # Red color (old stars)
        r_band = profile * 1.2
        g_band = profile * 0.8
        i_band = profile * 0.7

        image = np.stack([g_band, r_band, i_band])

        # Smooth
        for i in range(3):
            image[i] = gaussian_filter(image[i], sigma=2)

        return image

    def _create_spiral(self, size: int, brightness: float) -> np.ndarray:
        """Create spiral galaxy (arms, blue)"""
        center = size // 2
        y, x = np.ogrid[:size, :size]

        # Distance and angle from center
        r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        theta = np.arctan2(y - center, x - center)

        # Exponential disk
        disk = np.exp(-r / (size / 6)) * brightness

        # Spiral arms
        num_arms = 2
        arm_width = 0.3
        arms = np.zeros_like(r)

        for arm in range(num_arms):
            arm_angle = 2 * np.pi * arm / num_arms
            # Logarithmic spiral
            spiral_theta = theta - (r / (size / 4) * 2 * np.pi + arm_angle)
            arm_profile = np.exp(-((spiral_theta % (2 * np.pi) - np.pi) / arm_width) ** 2)
            arms += arm_profile

        # Combine disk and arms
        profile = disk * (1 + 0.5 * arms)

        # Blue color (young stars)
        g_band = profile * 1.2
        r_band = profile * 0.9
        i_band = profile * 0.7

        image = np.stack([g_band, r_band, i_band])

        # Smooth
        for i in range(3):
            image[i] = gaussian_filter(image[i], sigma=1.5)

        return image

    def generate_gaia_spectrum(
        self,
        idx: int,
        redshift: float,
        stellar_mass: float,
        galaxy_type: int
    ) -> Path:
        """
        Generate Gaia BP/RP spectrum (330-1050 nm, 343 wavelength bins)

        Features:
        - Elliptical: red spectrum, strong 4000Å break
        - Spiral: blue spectrum, emission lines
        - Redshifted based on z
        """
        # Wavelength grid (Gaia BP/RP range)
        wavelength_rest = np.linspace(330, 1050, 343)  # nm
        wavelength_obs = wavelength_rest * (1 + redshift)

        # Base continuum (blackbody-ish)
        if galaxy_type == 0:
            # Elliptical: old, red stars
            temperature = 4000  # K (cool)
        else:
            # Spiral: young, blue stars
            temperature = 8000  # K (hot)

        # Planck function (simplified)
        flux = self._planck(wavelength_rest, temperature)

        # Add 4000 Angstrom break (stronger for ellipticals)
        if galaxy_type == 0:
            break_strength = 0.4
        else:
            break_strength = 0.2

        break_idx = np.argmin(np.abs(wavelength_rest - 400))
        break_profile = 1.0 / (1 + np.exp(-(np.arange(len(flux)) - break_idx) / 10))
        flux = flux * (1 - break_strength * (1 - break_profile))

        # Add emission lines (stronger for spirals)
        if galaxy_type == 1:
            # H-alpha (656.3 nm)
            flux += self._add_emission_line(wavelength_rest, 656.3, strength=0.3)
            # H-beta (486.1 nm)
            flux += self._add_emission_line(wavelength_rest, 486.1, strength=0.2)
            # [OIII] (500.7 nm)
            flux += self._add_emission_line(wavelength_rest, 500.7, strength=0.25)

        # Normalize
        flux = flux / np.median(flux)

        # Add noise (S/N ~ 50)
        noise = np.random.randn(len(flux)) * 0.02
        flux = flux + noise

        # Create inverse variance (uncertainty)
        ivar = np.ones_like(flux) / (0.02 ** 2)

        # Mask bad pixels (5% randomly)
        mask = np.random.rand(len(flux)) > 0.05

        # Save
        spectrum_data = {
            'flux': flux.astype(np.float32),
            'ivar': ivar.astype(np.float32),
            'mask': mask.astype(bool),
            'wavelength': wavelength_obs.astype(np.float32)
        }

        save_path = self.output_dir / "spectra" / f"spectrum_{idx:06d}.npy"
        np.save(save_path, spectrum_data)

        return save_path

    def _planck(self, wavelength_nm: np.ndarray, temperature: float) -> np.ndarray:
        """Planck function (blackbody spectrum)"""
        h = 6.626e-34  # Planck constant
        c = 3.0e8      # Speed of light
        k = 1.381e-23  # Boltzmann constant

        wavelength_m = wavelength_nm * 1e-9

        # Avoid division by zero
        with np.errstate(over='ignore', invalid='ignore'):
            intensity = (2 * h * c**2 / wavelength_m**5) / \
                       (np.exp(h * c / (wavelength_m * k * temperature)) - 1)

        return intensity / np.max(intensity)

    def _add_emission_line(
        self,
        wavelength: np.ndarray,
        line_center: float,
        strength: float = 0.2,
        width: float = 5.0
    ) -> np.ndarray:
        """Add Gaussian emission line"""
        line = strength * np.exp(-((wavelength - line_center) / width) ** 2)
        return line

    def generate_ztf_lightcurve(
        self,
        idx: int,
        galaxy_type: int,
        sfr: float
    ) -> Path:
        """
        Generate ZTF light curve (g-band, ~100 observations over 3 years)

        Features:
        - Variable for spirals with high SFR (supernovae)
        - Stable for ellipticals
        - Realistic cadence and noise
        """
        # Time span: 3 years, irregular cadence
        start_mjd = 58000  # Modified Julian Date
        duration_days = 3 * 365

        # ZTF cadence: observations every few days with gaps
        num_obs = np.random.randint(80, 120)
        times = np.sort(np.random.uniform(start_mjd, start_mjd + duration_days, num_obs))

        # Base magnitude (depends on brightness)
        if galaxy_type == 0:
            base_mag = 19.0  # Elliptical (stable)
        else:
            base_mag = 18.5  # Spiral (brighter, variable)

        magnitude = np.ones(num_obs) * base_mag

        # Add variability for spirals with high SFR (SNe, AGN)
        if galaxy_type == 1 and sfr > 50:
            # Add a supernova transient
            sn_time = np.random.uniform(start_mjd + 100, start_mjd + duration_days - 100)
            sn_peak_mag = base_mag - np.random.uniform(2, 4)  # SN is 2-4 mag brighter

            # SN light curve (rise and fall)
            dt = times - sn_time
            rise_time = 20  # days
            fall_time = 60  # days

            for i, t in enumerate(dt):
                if -rise_time < t < 0:
                    # Rising
                    magnitude[i] = base_mag + (sn_peak_mag - base_mag) * (1 + t / rise_time)
                elif 0 <= t < fall_time:
                    # Falling
                    magnitude[i] = sn_peak_mag + (base_mag - sn_peak_mag) * (t / fall_time) ** 0.5

        # Add seasonal variations
        seasonal = 0.05 * np.sin(2 * np.pi * times / 365.25)
        magnitude += seasonal

        # Add photometric noise (depends on brightness)
        mag_err = 0.02 + 0.01 * (magnitude - 18)  # Fainter = more noise
        magnitude += np.random.randn(num_obs) * mag_err

        # Convert to flux (for easier modeling)
        # flux = 10^(-0.4 * mag)
        flux = 10 ** (-0.4 * (magnitude - 20))  # Normalized to mag 20

        flux_err = flux * mag_err * 0.4 * np.log(10)

        # Save
        lightcurve_data = {
            'mjd': times.astype(np.float32),
            'flux': flux.astype(np.float32),
            'flux_err': flux_err.astype(np.float32),
            'filter': np.array(['g'] * num_obs, dtype='<U1')
        }

        save_path = self.output_dir / "lightcurves" / f"lightcurve_{idx:06d}.npy"
        np.save(save_path, lightcurve_data)

        return save_path

    def _create_splits(self, manifest: dict):
        """Create train/val/test splits"""
        samples = manifest['samples']
        np.random.shuffle(samples)

        n = len(samples)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        train_manifest = {"samples": samples[:train_end]}
        val_manifest = {"samples": samples[train_end:val_end]}
        test_manifest = {"samples": samples[val_end:]}

        # Save splits
        with open(self.output_dir / "metadata" / "train_manifest.json", 'w') as f:
            json.dump(train_manifest, f, indent=2)

        with open(self.output_dir / "metadata" / "val_manifest.json", 'w') as f:
            json.dump(val_manifest, f, indent=2)

        with open(self.output_dir / "metadata" / "test_manifest.json", 'w') as f:
            json.dump(test_manifest, f, indent=2)

        print(f"\nSplits:")
        print(f"  Train: {len(train_manifest['samples'])} samples")
        print(f"  Val:   {len(val_manifest['samples'])} samples")
        print(f"  Test:  {len(test_manifest['samples'])} samples")

    def visualize_sample(self, sample_idx: int = 0):
        """Visualize a generated sample"""
        manifest_path = self.output_dir / "metadata" / "manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        sample = manifest['samples'][sample_idx]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Image
        image = np.load(self.output_dir / sample['image'])
        axes[0, 0].imshow(image.transpose(1, 2, 0))
        axes[0, 0].set_title(f"Galaxy Image (z={sample['redshift']:.2f})")
        axes[0, 0].axis('off')

        # 2. Spectrum
        spectrum_data = np.load(self.output_dir / sample['spectrum'], allow_pickle=True).item()
        axes[0, 1].plot(spectrum_data['wavelength'], spectrum_data['flux'])
        axes[0, 1].set_xlabel('Wavelength (nm)')
        axes[0, 1].set_ylabel('Flux')
        axes[0, 1].set_title('Gaia BP/RP Spectrum')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Light curve
        lc_data = np.load(self.output_dir / sample['lightcurve'], allow_pickle=True).item()
        axes[1, 0].errorbar(
            lc_data['mjd'] - lc_data['mjd'][0],
            lc_data['flux'],
            yerr=lc_data['flux_err'],
            fmt='o',
            alpha=0.6
        )
        axes[1, 0].set_xlabel('Days since first observation')
        axes[1, 0].set_ylabel('Flux')
        axes[1, 0].set_title('ZTF Light Curve')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Parameters
        axes[1, 1].axis('off')
        params_text = f"""
        Physical Parameters:

        Redshift: {sample['redshift']:.3f}
        Stellar Mass: {sample['stellar_mass']:.2f} log(M☉)
        SFR: {sample['sfr']:.1f} M☉/yr
        Galaxy Type: {'Elliptical' if sample['galaxy_type'] == 0 else 'Spiral'}
        """
        axes[1, 1].text(0.1, 0.5, params_text, fontsize=12, verticalalignment='center')

        plt.tight_layout()
        plt.savefig(self.output_dir / f"example_sample_{sample_idx}.png", dpi=150)
        print(f"\n✓ Saved visualization: {self.output_dir / f'example_sample_{sample_idx}.png'}")
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic astronomical data")
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='./example_data', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize sample')

    args = parser.parse_args()

    # Generate data
    generator = AstronomicalDataGenerator(
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )

    manifest = generator.generate_all()

    # Visualize
    if args.visualize:
        generator.visualize_sample(0)
        generator.visualize_sample(1)
        generator.visualize_sample(2)

    print("\n" + "=" * 80)
    print("✓ Data generation complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. python astronomical_dataset.py  # Test loading the data")
    print(f"  2. python train_astronomical.py     # Train the model")
