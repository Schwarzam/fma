"""
inference_astronomical.py - Use trained model for predictions

This script shows how to use your trained astronomical_model.pt for:
1. Predicting missing modalities from observed ones
2. Reconstructing images, spectra, and light curves
3. Imputing missing scalar values (redshift, stellar mass, SFR)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from astronomical_dataset import (
    AstronomicalDataset, collate_astronomical,
    GalaxyImage, GaiaSpectrum, ZTFLightCurve, Redshift, StellarMass, StarFormationRate
)
from astronomical_codecs import (
    GalaxyImageCodec, GaiaSpectrumCodec, ZTFLightCurveCodec,
    RedshiftCodec, StellarMassCodec, StarFormationRateCodec
)
from step4_codec_manager import CodecManager, MODALITY_CODEC_MAPPING
from step6_train_transformer import MultimodalTransformer


# -----------------------------------------------------------------------------
# Register astronomical modalities with codecs (must match training)
# -----------------------------------------------------------------------------
MODALITY_CODEC_MAPPING.update({
    GalaxyImage: GalaxyImageCodec,
    GaiaSpectrum: GaiaSpectrumCodec,
    ZTFLightCurve: ZTFLightCurveCodec,
    Redshift: RedshiftCodec,
    StellarMass: StellarMassCodec,
    StarFormationRate: StarFormationRateCodec,
})


# -----------------------------------------------------------------------------
# Helpers (mirrors training script behavior)
# -----------------------------------------------------------------------------
def _batch_to_batched_modalities(batch, device):
    """
    Concatenate same-modality items and build batched modality objects per modality.
    """
    concatenated = {}

    for modality_name, modality_list in batch.items():
        if modality_name == 'sample_ids':
            continue
        if len(modality_list) == 0:
            continue

        if modality_name == 'galaxy_image':
            flux = torch.cat([m.flux for m in modality_list], dim=0)
            concatenated['galaxy_image'] = GalaxyImage(flux=flux.to(device), metadata={})

        elif modality_name == 'gaia_spectrum':
            flux = torch.cat([m.flux for m in modality_list], dim=0)
            ivar = torch.cat([m.ivar for m in modality_list], dim=0)
            mask = torch.cat([m.mask for m in modality_list], dim=0)
            wavelength = modality_list[0].wavelength.to(device)  # assume same grid after collate
            concatenated['gaia_spectrum'] = GaiaSpectrum(
                flux=flux.to(device), ivar=ivar.to(device), mask=mask.to(device),
                wavelength=wavelength, metadata={}
            )

        elif modality_name == 'ztf_lightcurve':
            flux = torch.cat([m.flux for m in modality_list], dim=0)
            flux_err = torch.cat([m.flux_err for m in modality_list], dim=0)
            mjd = modality_list[0].mjd.to(device)
            concatenated['ztf_lightcurve'] = ZTFLightCurve(
                flux=flux.to(device), flux_err=flux_err.to(device), mjd=mjd, metadata={}
            )

        elif modality_name == 'redshift':
            values = torch.cat([m.value for m in modality_list], dim=0)
            concatenated['redshift'] = Redshift(value=values.to(device), metadata={})

        elif modality_name == 'stellar_mass':
            values = torch.cat([m.value for m in modality_list], dim=0)
            concatenated['stellar_mass'] = StellarMass(value=values.to(device), metadata={})

        elif modality_name == 'sfr':
            values = torch.cat([m.value for m in modality_list], dim=0)
            concatenated['sfr'] = StarFormationRate(value=values.to(device), metadata={})

    return concatenated


def _encode_many(codec_manager, batched_modalities):
    """
    Encode dict of {name: modality_instance} by calling codec_manager.encode per modality.
    Returns merged dict {token_key: LongTensor[B, T]}.
    """
    tokens = {}
    for name, modality in batched_modalities.items():
        out = codec_manager.encode(modality)  # expects a single BaseModality
        if not isinstance(out, dict):
            raise RuntimeError(f"CodecManager.encode for '{name}' did not return a dict.")
        tokens.update(out)
    return tokens


def _infer_num_tokens_from_codecs(codec_manager, batched_modalities):
    """
    One encode pass to discover sequence lengths per token_key.
    Returns dict: {token_key: T}
    """
    with torch.no_grad():
        tokens = _encode_many(codec_manager, batched_modalities)
    num_tokens = {}
    for key, seq in tokens.items():
        if seq.ndim == 1:
            num_tokens[key] = 1
        else:
            num_tokens[key] = seq.size(1)
    return num_tokens


# -----------------------------------------------------------------------------
# Predictor
# -----------------------------------------------------------------------------
class AstronomicalPredictor:
    """
    Wrapper for making predictions with the trained model.

    Handles:
    - Loading trained model (with correct num_tokens inferred from codecs)
    - Encoding input modalities
    - Predicting missing modalities (masked protocol)
    - Decoding predictions back to interpretable form
    """

    def __init__(self, model_path: str, device: str, warmup_tokens: dict, vocab_sizes: dict):
        """
        Args:
            model_path: path to saved model checkpoint
            device: 'cuda' or 'cpu'
            warmup_tokens: dict {token_key: LongTensor[B, T]} from a warm-up batch (used only for num_tokens)
            vocab_sizes: dict matching training config
        """
        self.device = device
        print(f"Using device: {device}")

        # Setup codecs
        print("Initializing codecs...")
        self.codec_manager = CodecManager(device=device)

        # Infer num_tokens from warmup_tokens (same shape as training)
        inferred_num_tokens = {k: (v.size(1) if v.ndim > 1 else 1) for k, v in warmup_tokens.items()}
        self.num_tokens = inferred_num_tokens

        print("Inferred token lengths per modality (from warm-up):")
        for k, v in self.num_tokens.items():
            print(f"  {k}: {v}")

        # Create model with SAME arch as training
        print("Loading model...")
        self.model = MultimodalTransformer(
            vocab_sizes=vocab_sizes,
            num_tokens=self.num_tokens,
            d_model=512,       # Updated to match training
            nhead=8,           # Updated to match training
            num_layers=8,      # Updated to match training
            dim_feedforward=2048,  # Updated to match training
            dropout=0.1
        ).to(device)

        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        missing, unexpected = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing or unexpected:
            print(f"Warning: load_state_dict mismatch. missing={len(missing)}, unexpected={len(unexpected)}")
            if missing:
                print("  Missing:", missing)
            if unexpected:
                print("  Unexpected:", unexpected)
        self.model.eval()
        print("✓ Model loaded successfully!\n")

    # --------------- internal ---------------
    def _build_all_tokens(self, input_modalities: dict):
        """
        Encode input modalities to tokens.
        Returns dict {token_key: LongTensor[B,T]} and batch size B.
        """
        with torch.no_grad():
            tokens = {}
            for mod_name, modality in input_modalities.items():
                out = self.codec_manager.encode(modality)
                tokens.update(out)
        # infer batch size from first entry
        some = next(iter(tokens.values()))
        B = some.size(0) if some.ndim > 1 else some.size(0)
        return tokens, B

    def _ensure_targets_present(self, all_tokens: dict, target_token_keys: list, batch_size: int):
        """
        If a target token_key is not present in all_tokens (because it's missing/being predicted),
        create a zero placeholder of shape [B, T_target] so the model can index it.
        """
        for key in target_token_keys:
            if key not in all_tokens:
                T = self.num_tokens.get(key, 1)
                all_tokens[key] = torch.zeros(batch_size, T, dtype=torch.long, device=self.device)
        return all_tokens

    def _make_all_true_masks(self, all_tokens: dict, target_token_keys: list):
        """
        Build a masked dict {key: BoolTensor[B,T]} = True (mask) for all positions in target modalities.
        """
        masked = {}
        for key in target_token_keys:
            if key not in all_tokens:
                continue
            masked[key] = torch.ones_like(all_tokens[key], dtype=torch.bool, device=self.device)
        return masked

    def predict(self, input_modalities: dict, predict_modalities: list):
        """
        Predict missing modalities from observed ones.

        Args:
            input_modalities: dict name -> modality instance (already on self.device)
            predict_modalities: list of token_keys (e.g., ['tok_redshift', 'tok_stellar_mass'])
        Returns:
            dict mapping predicted modality name -> decoded modality object
        """
        # map token_key -> modality class (what CodecManager.decode expects)
        TOKENKEY_TO_CLASS = {
            "tok_galaxy_image": GalaxyImage,
            "tok_gaia_spectrum": GaiaSpectrum,
            "tok_ztf_lightcurve": ZTFLightCurve,
            "tok_redshift": Redshift,
            "tok_stellar_mass": StellarMass,
            "tok_sfr": StarFormationRate,
        }

        with torch.no_grad():
            # 1) Encode observed
            all_tokens, B = self._build_all_tokens(input_modalities)

            # 2) Ensure targets exist (zeros) so we know [B, T_model] for each
            all_tokens = self._ensure_targets_present(all_tokens, predict_modalities, B)

            # 3) ALIGN to model’s expected token lengths
            all_tokens = self._align_to_model_num_tokens(all_tokens)

            # 4) Build mask dict (True everywhere for targets) using aligned shapes
            masked = {k: torch.ones_like(all_tokens[k], dtype=torch.bool, device=self.device)
                    for k in predict_modalities if k in all_tokens}

            # 5) Forward
            predictions = self.model(all_tokens, masked)

            # 5) Decode predictions back to modalities
            decoded_predictions = {}
            for token_key, pred_logits in predictions.items():
                # Determine (B, T) from placeholder/all_tokens
                if token_key not in all_tokens:
                    # safety, but _ensure_targets_present should have added it
                    T = self.num_tokens.get(token_key, 1)
                    target_shape = (B, T)
                else:
                    target_shape = tuple(all_tokens[token_key].shape)  # (B, T)

                # Convert logits -> token ids and reshape to [B, T]
                pred_tokens = pred_logits.argmax(dim=-1)
                if pred_tokens.numel() != (target_shape[0] * target_shape[1]):
                    # if model returned per-position already as [B, T], keep it
                    if pred_tokens.shape == target_shape:
                        pass
                    else:
                        # fallback: trim or pad to expected size
                        N = target_shape[0] * target_shape[1]
                        if pred_tokens.numel() > N:
                            pred_tokens = pred_tokens.reshape(-1)[:N]
                        else:
                            pad = N - pred_tokens.numel()
                            pred_tokens = torch.cat([pred_tokens.reshape(-1),
                                                    torch.zeros(pad, dtype=pred_tokens.dtype, device=pred_tokens.device)],
                                                    dim=0)
                        pred_tokens = pred_tokens.view(*target_shape)
                else:
                    pred_tokens = pred_tokens.view(*target_shape)

                # Prepare decoder metadata if needed
                metadata = {}
                if token_key == 'tok_gaia_spectrum':
                    if 'gaia_spectrum' in input_modalities:
                        metadata['wavelength'] = input_modalities['gaia_spectrum'].wavelength
                    else:
                        T = target_shape[1]
                        metadata['wavelength'] = torch.arange(T, device=self.device).float()
                elif token_key == 'tok_ztf_lightcurve':
                    if 'ztf_lightcurve' in input_modalities:
                        metadata['mjd'] = input_modalities['ztf_lightcurve'].mjd
                    else:
                        T = target_shape[1]
                        metadata['mjd'] = torch.arange(T, device=self.device).float()

                # >>> FIX: pass the modality CLASS, not the token_key string
                modality_cls = TOKENKEY_TO_CLASS[token_key]
                decoded = self.codec_manager.decode(pred_tokens, modality_cls, **metadata)

                # Map 'tok_xxx' -> 'xxx'
                modality_name = token_key.replace('tok_', '')
                decoded_predictions[modality_name] = decoded

            return decoded_predictions

    def _align_to_model_num_tokens(self, tokens_dict: dict) -> dict:
        """
        Ensure each token sequence matches the model's configured length for that token_key.
        If longer -> truncate; if shorter -> pad with zeros (left-append).
        """
        aligned = {}
        for k, v in tokens_dict.items():
            if v.ndim == 1:
                v = v.unsqueeze(0)  # [B=1] -> [1, T] consistency (rare)
            B, T = v.shape[0], v.shape[1]
            T_model = self.num_tokens.get(k, T)  # fallback to current length if unknown
            if T > T_model:
                aligned[k] = v[:, :T_model]
            elif T < T_model:
                pad = torch.zeros(B, T_model - T, dtype=v.dtype, device=v.device)
                aligned[k] = torch.cat([v, pad], dim=1)
            else:
                aligned[k] = v
        return aligned

# -----------------------------------------------------------------------------
# Example Use Cases
# -----------------------------------------------------------------------------
def example1_predict_scalars_from_image(predictor, dataset):
    print("=" * 80)
    print("EXAMPLE 1: Predict Scalars from Image")
    print("=" * 80)

    sample = dataset[0]
    image = GalaxyImage(flux=sample['image'].unsqueeze(0).to(predictor.device), metadata={})

    preds = predictor.predict(
        input_modalities={'galaxy_image': image},
        predict_modalities=['tok_redshift', 'tok_stellar_mass', 'tok_sfr']
    )

    print("\nInput: Galaxy Image")
    print(f"  Image shape: {sample['image'].shape}")

    # Collect values for table
    table_data = []
    if 'redshift' in preds:
        pred_z = preds['redshift'].value.item()
        true_z = sample['redshift'].item()
        table_data.append(['Redshift', f'{true_z:.3f}', f'{pred_z:.3f}', f'{abs(pred_z - true_z):.3f}'])
        print(f"  Redshift: {pred_z:.3f} (true: {true_z:.3f}, error: {abs(pred_z - true_z):.3f})")

    if 'stellar_mass' in preds:
        pred_mass = preds['stellar_mass'].value.item()
        true_mass = sample['stellar_mass'].item()
        table_data.append(['Stellar Mass [log(M☉)]', f'{true_mass:.2f}', f'{pred_mass:.2f}', f'{abs(pred_mass - true_mass):.2f}'])
        print(f"  Stellar Mass: {pred_mass:.2f} log(M☉) (true: {true_mass:.2f}, error: {abs(pred_mass - true_mass):.2f})")

    if 'sfr' in preds:
        pred_sfr = preds['sfr'].value.item()
        true_sfr = sample['sfr'].item()
        table_data.append(['SFR [M☉/yr]', f'{true_sfr:.1f}', f'{pred_sfr:.1f}', f'{abs(pred_sfr - true_sfr):.1f}'])
        print(f"  SFR: {pred_sfr:.1f} M☉/yr (true: {true_sfr:.1f}, error: {abs(pred_sfr - true_sfr):.1f})")

    # Create figure with image and table
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Image
    img = sample['image'].permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    ax1.imshow(img)
    ax1.set_title("Input: Galaxy Image", fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Right: Table
    ax2.axis('tight')
    ax2.axis('off')

    # Create table
    table = ax2.table(
        cellText=table_data,
        colLabels=['Property', 'True Value', 'Predicted', 'Absolute Error'],
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.2, 0.2, 0.25]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')

    ax2.set_title("Scalar Predictions", fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('prediction_example1_image_to_scalars.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization: prediction_example1_image_to_scalars.png")


def example2_predict_spectrum_from_image(predictor, dataset):
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Predict Spectrum from Image")
    print("=" * 80)

    sample = dataset[0]
    image = GalaxyImage(flux=sample['image'].unsqueeze(0).to(predictor.device), metadata={})

    preds = predictor.predict(
        input_modalities={'galaxy_image': image},
        predict_modalities=['tok_gaia_spectrum']
    )

    if 'gaia_spectrum' in preds:
        pred_spectrum = preds['gaia_spectrum']
        true_flux = sample['spectrum']['flux'].cpu().numpy()
        pred_flux = pred_spectrum.flux.cpu().numpy().squeeze()

        wavelength = sample['spectrum']['wavelength'].cpu().numpy()
        # Robust length alignment
        true_flux = np.asarray(true_flux).reshape(-1)
        pred_flux = np.asarray(pred_flux).reshape(-1)
        wavelength = np.asarray(wavelength).reshape(-1)

        n = min(len(wavelength), len(true_flux), len(pred_flux))
        if n == 0:
            print("Warning: empty arrays for plotting; skipping figure.")
            return

        wavelength_n = wavelength[:n]
        true_flux_n = true_flux[:n]
        pred_flux_n = pred_flux[:n]

        # Correlation on overlapped region
        valid = (~np.isnan(true_flux_n)) & (~np.isnan(pred_flux_n))
        corr = np.corrcoef(true_flux_n[valid], pred_flux_n[valid])[0, 1] if valid.sum() > 1 else np.nan
        print(f"  Spectrum lengths -> wavelength: {len(wavelength)}, true: {len(true_flux)}, pred: {len(pred_flux)}")
        print(f"  Using n={n} aligned points; corr={corr:.3f}")

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Image
        img = sample['image'].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        axes[0].imshow(img)
        axes[0].set_title("Input: Galaxy Image")
        axes[0].axis('off')

        # Spectrum
        axes[1].plot(wavelength_n, true_flux_n, label='True Spectrum', alpha=0.7, linewidth=2)
        axes[1].plot(wavelength_n, pred_flux_n, label='Predicted Spectrum', alpha=0.7, linewidth=2)
        axes[1].set_xlabel('Wavelength')
        axes[1].set_ylabel('Flux')
        axes[1].set_title(f'Spectrum Prediction (corr={corr:.3f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('prediction_example2_image_to_spectrum.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved visualization: prediction_example2_image_to_spectrum.png")


def example3_predict_lightcurve(predictor, dataset):
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Predict Light Curve from Image + Redshift")
    print("=" * 80)

    sample = dataset[0]
    image = GalaxyImage(flux=sample['image'].unsqueeze(0).to(predictor.device), metadata={})
    redshift = Redshift(value=sample['redshift'].unsqueeze(0).to(predictor.device), metadata={})

    preds = predictor.predict(
        input_modalities={'galaxy_image': image, 'redshift': redshift},
        predict_modalities=['tok_ztf_lightcurve']
    )

    if 'ztf_lightcurve' in preds:
        pred_lc = preds['ztf_lightcurve']
        true_flux = sample['lightcurve']['flux'].cpu().numpy()
        pred_flux = pred_lc.flux.cpu().numpy().squeeze()
        mjd = sample['lightcurve']['mjd'].cpu().numpy()

        # ---- length-safe alignment before plotting ----
        true_flux = np.asarray(true_flux).reshape(-1)
        pred_flux = np.asarray(pred_flux).reshape(-1)
        mjd = np.asarray(mjd).reshape(-1)

        n = min(len(mjd), len(true_flux), len(pred_flux))
        if n == 0:
            print("Warning: empty arrays for plotting; skipping figure.")
            return

        mjd_n = mjd[:n]
        true_flux_n = true_flux[:n]
        pred_flux_n = pred_flux[:n]

        print(f"  Light curve lengths -> mjd: {len(mjd)}, true: {len(true_flux)}, pred: {len(pred_flux)}")
        print(f"  Using n={n} aligned points")

        # Optional quick metrics on the overlapped window
        valid = (~np.isnan(true_flux_n)) & (~np.isnan(pred_flux_n))
        if valid.sum() > 1:
            mae = np.mean(np.abs(true_flux_n[valid] - pred_flux_n[valid]))
            rmse = np.sqrt(np.mean((true_flux_n[valid] - pred_flux_n[valid])**2))
            print(f"  MAE={mae:.3f}, RMSE={rmse:.3f}")


def example4_reconstruct_image(predictor, dataset):
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Reconstruct Image from Other Modalities")
    print("=" * 80)

    sample = dataset[0]
    spectrum = GaiaSpectrum(
        flux=sample['spectrum']['flux'].unsqueeze(0).to(predictor.device),
        ivar=sample['spectrum']['ivar'].unsqueeze(0).to(predictor.device),
        mask=sample['spectrum']['mask'].unsqueeze(0).to(predictor.device),
        wavelength=sample['spectrum']['wavelength'].to(predictor.device),
        metadata={}
    )
    redshift = Redshift(value=sample['redshift'].unsqueeze(0).to(predictor.device), metadata={})
    stellar_mass = StellarMass(value=sample['stellar_mass'].unsqueeze(0).to(predictor.device), metadata={})

    preds = predictor.predict(
        input_modalities={'gaia_spectrum': spectrum, 'redshift': redshift, 'stellar_mass': stellar_mass},
        predict_modalities=['tok_galaxy_image']
    )

    if 'galaxy_image' in preds:
        pred_image = preds['galaxy_image']
        true_img = sample['image'].permute(1, 2, 0).cpu().numpy()
        pred_img = pred_image.flux.squeeze().permute(1, 2, 0).cpu().numpy()
        true_img = np.clip(true_img, 0, 1); pred_img = np.clip(pred_img, 0, 1)

        mse = np.mean((true_img - pred_img) ** 2)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(true_img); axes[0].set_title("True"); axes[0].axis('off')
        axes[1].imshow(pred_img); axes[1].set_title("Reconstructed"); axes[1].axis('off')
        diff = np.abs(true_img - pred_img)
        im = axes[2].imshow(diff, cmap='hot'); axes[2].set_title(f"Abs Diff\nMSE={mse:.4f}"); axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)
        plt.tight_layout()
        plt.savefig('prediction_example4_reconstruct_image.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved visualization: prediction_example4_reconstruct_image.png")


def example5_batch_predictions(predictor, dataset, num_samples=5):
    print("\n" + "=" * 80)
    print(f"EXAMPLE 5: Batch Predictions ({num_samples} samples)")
    print("=" * 80)

    errors_z, errors_mass, errors_sfr = [], [], []

    for i in range(num_samples):
        sample = dataset[i]
        image = GalaxyImage(flux=sample['image'].unsqueeze(0).to(predictor.device), metadata={})

        preds = predictor.predict(
            input_modalities={'galaxy_image': image},
            predict_modalities=['tok_redshift', 'tok_stellar_mass', 'tok_sfr']
        )

        if 'redshift' in preds:
            pred_z = preds['redshift'].value.item()
            true_z = sample['redshift'].item()
            errors_z.append(abs(pred_z - true_z))
        if 'stellar_mass' in preds:
            pred_mass = preds['stellar_mass'].value.item()
            true_mass = sample['stellar_mass'].item()
            errors_mass.append(abs(pred_mass - true_mass))
        if 'sfr' in preds:
            pred_sfr = preds['sfr'].value.item()
            true_sfr = sample['sfr'].item()
            errors_sfr.append(abs(pred_sfr - true_sfr))

    print(f"\nPrediction Errors (n={num_samples}):")
    print(f"  Redshift MAE: {np.mean(errors_z):.4f} ± {np.std(errors_z):.4f}")
    print(f"  Stellar Mass MAE: {np.mean(errors_mass):.3f} ± {np.std(errors_mass):.3f} log(M☉)")
    print(f"  SFR MAE: {np.mean(errors_sfr):.2f} ± {np.std(errors_sfr):.2f} M☉/yr")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(errors_z, bins=10, alpha=0.7, edgecolor='black'); axes[0].set_xlabel('Abs Error'); axes[0].set_title(f'Redshift\nMAE={np.mean(errors_z):.4f}'); axes[0].grid(True, alpha=0.3)
    axes[1].hist(errors_mass, bins=10, alpha=0.7, edgecolor='black'); axes[1].set_xlabel('Abs Error (log M☉)'); axes[1].set_title(f'Mass\nMAE={np.mean(errors_mass):.3f}'); axes[1].grid(True, alpha=0.3)
    axes[2].hist(errors_sfr, bins=10, alpha=0.7, edgecolor='black'); axes[2].set_xlabel('Abs Error (M☉/yr)'); axes[2].set_title(f'SFR\nMAE={np.mean(errors_sfr):.2f}'); axes[2].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('prediction_example5_error_distributions.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization: prediction_example5_error_distributions.png")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("ASTRONOMICAL MODEL INFERENCE")
    print("=" * 80)
    print("\nThis script demonstrates various use cases for the trained model:\n")
    print("1. Predict scalars (z, mass, SFR) from images")
    print("2. Predict spectra from images")
    print("3. Predict light curves from images + redshift")
    print("4. Reconstruct images from other modalities")
    print("5. Batch predictions with error statistics\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./astronomical_model.pt"
    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}\nRun train_astronomical.py first.")
        return

    # --- Load a validation batch FIRST to infer num_tokens (same as training warm-up) ---
    print("Loading validation dataset...")
    val_dataset = AstronomicalDataset(
        data_root="./example_data",
        manifest_path="./example_data/metadata/val_manifest.json"
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_astronomical,
        num_workers=0
    )
    warmup_batch = next(iter(val_loader))

    # Build batched modalities and infer num_tokens via codecs
    codec_manager = CodecManager(device=device)
    batched_modalities = _batch_to_batched_modalities(warmup_batch, device=device)
    warmup_tokens = _encode_many(codec_manager, batched_modalities)

    vocab_sizes = {
        "tok_galaxy_image": 1000,  # Fixed: FSQ [8,5,5,5] = 1000 codes
        "tok_gaia_spectrum": 512,
        "tok_ztf_lightcurve": 512,
        "tok_redshift": 256,
        "tok_stellar_mass": 256,
        "tok_sfr": 256,
    }

    # Construct predictor with the *inferred* num_tokens
    predictor = AstronomicalPredictor(
        model_path=model_path,
        device=device,
        warmup_tokens=warmup_tokens,
        vocab_sizes=vocab_sizes
    )

    print(f"✓ Loaded {len(val_dataset)} validation samples\n")

    # --- Run examples ---
    example1_predict_scalars_from_image(predictor, val_dataset)
    example2_predict_spectrum_from_image(predictor, val_dataset)
    example3_predict_lightcurve(predictor, val_dataset)
    example4_reconstruct_image(predictor, val_dataset)
    example5_batch_predictions(predictor, val_dataset, num_samples=20)

    print("\n" + "=" * 80)
    print("✓ All examples completed!")
    print("=" * 80)
    print("\nGenerated visualizations:")
    print("  - prediction_example1_image_to_scalars.png")
    print("  - prediction_example2_image_to_spectrum.png")
    print("  - prediction_example3_image_to_lightcurve.png")
    print("  - prediction_example4_reconstruct_image.png")
    print("  - prediction_example5_error_distributions.png")


if __name__ == "__main__":
    main()