"""
train_astronomical.py - Quick training script for astronomical data

Simple example showing how to train on the generated data, with:
- loss computed only on masked positions
- num_tokens inferred dynamically from codec outputs (warm-up batch)
- proper per-modality encode calls (no dict passed to CodecManager.encode)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from astronomical_dataset import AstronomicalDataset, collate_astronomical
from astronomical_dataset import GalaxyImage, GaiaSpectrum, ZTFLightCurve, Redshift, StellarMass, StarFormationRate
from astronomical_codecs import (
    GalaxyImageCodec, GaiaSpectrumCodec, ZTFLightCurveCodec,
    RedshiftCodec, StellarMassCodec, StarFormationRateCodec
)
from core.codec_manager import CodecManager, MODALITY_CODEC_MAPPING
from core.train_transformer import MultimodalTransformer, apply_masking


# Register astronomical modalities with their specific codecs
MODALITY_CODEC_MAPPING.update({
    GalaxyImage: GalaxyImageCodec,
    GaiaSpectrum: GaiaSpectrumCodec,
    ZTFLightCurve: ZTFLightCurveCodec,
    Redshift: RedshiftCodec,
    StellarMass: StellarMassCodec,
    StarFormationRate: StarFormationRateCodec,
})


def _masked_cross_entropy(pred, target, mask=None):
    """
    pred: (..., V) logits for masked positions (usually [M, V] where M=#masked tokens)
    target: [B, T] token ids
    mask:  optional boolean mask [B, T] selecting masked positions
    """
    if mask is not None:
        if mask.dtype != torch.bool:
            raise ValueError("Expected boolean mask for masked positions.")
        target = target[mask]  # select masked target positions

    pred_flat = pred.reshape(-1, pred.size(-1))
    target_flat = target.reshape(-1).long().to(pred.device)

    if pred_flat.size(0) == 0 or target_flat.size(0) == 0:
        return None, None

    if pred_flat.size(0) != target_flat.size(0):
        n = min(pred_flat.size(0), target_flat.size(0))
        pred_flat = pred_flat[:n]
        target_flat = target_flat[:n]

    loss = F.cross_entropy(pred_flat, target_flat)
    acc = (pred_flat.argmax(dim=-1) == target_flat).float().mean()
    return loss, acc


def _batch_to_batched_modalities(batch, device):
    """
    Concatenate same-modality items and build batched modality objects that
    the codec_manager can encode in one go per modality.
    Returns dict[str, BaseModality]
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
    Encode a dict of {name: modality_instance} by calling codec_manager.encode per modality.
    Returns a merged dict {token_key: LongTensor[B, T]}.
    """
    tokens = {}
    for name, modality in batched_modalities.items():
        out = codec_manager.encode(modality)  # expects a single BaseModality instance
        if not isinstance(out, dict):
            raise RuntimeError(f"CodecManager.encode for '{name}' did not return a dict.")
        tokens.update(out)
    return tokens


def _infer_num_tokens_from_codecs(codec_manager, batched_modalities):
    """
    Do one encode pass to discover sequence lengths per token_key.
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


def quick_train():
    """Quick training example"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load datasets
    print("Loading datasets...")
    train_dataset = AstronomicalDataset(
        data_root="./example_data",
        manifest_path="./example_data/metadata/train_manifest.json"
    )

    val_dataset = AstronomicalDataset(
        data_root="./example_data",
        manifest_path="./example_data/metadata/val_manifest.json"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_astronomical,
        num_workers=0  # increase (2-4) if I/O bound
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_astronomical,
        num_workers=0
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples\n")

    # Create codec manager
    print("Initializing codec manager...")
    codec_manager = CodecManager(device=device)

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

    # -----------------------------
    # WARM-UP: infer num_tokens dynamically from codecs
    # -----------------------------
    warmup_batch = next(iter(train_loader))
    batched_modalities = _batch_to_batched_modalities(warmup_batch, device=device)

    try:
        inferred_num_tokens = _infer_num_tokens_from_codecs(codec_manager, batched_modalities)
        print("Inferred token lengths per modality (token_key -> T):")
        for k, v in inferred_num_tokens.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Warm-up encode failed: {e}")
        inferred_num_tokens = {
            GalaxyImage.token_key: GalaxyImage.num_tokens,
            GaiaSpectrum.token_key: GaiaSpectrum.num_tokens,
            ZTFLightCurve.token_key: ZTFLightCurve.num_tokens,
            Redshift.token_key: Redshift.num_tokens,
            StellarMass.token_key: StellarMass.num_tokens,
            StarFormationRate.token_key: StarFormationRate.num_tokens,
        }

    # Vocab sizes (keep consistent with your codecs)
    # IMPORTANT: Must match actual quantizer codebook sizes!
    # - FSQ [8,5,5,5] = 8*5*5*5 = 1000 codes (not 10000!)
    # - VectorQuantizer with num_embeddings=512 = 512 codes
    # - ScalarLinearQuantizer with num_bins=256 = 256 codes
    vocab_sizes = {
        "tok_galaxy_image": 1000,  # Fixed: FSQ [8,5,5,5] = 1000 codes
        "tok_gaia_spectrum": 512,
        "tok_ztf_lightcurve": 512,
        "tok_redshift": 256,
        "tok_stellar_mass": 256,
        "tok_sfr": 256,
    }

    # Create transformer with inferred token lengths
    print("Creating transformer...")
    model = MultimodalTransformer(
        vocab_sizes=vocab_sizes,
        num_tokens=inferred_num_tokens,
        d_model=512,  # Small for demo
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

    # Training loop
    num_epochs = 5
    print(f"Training for {num_epochs} epochs...")
    print("=" * 80)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            # Encode all modalities with frozen codecs
            with torch.no_grad():
                batched = _batch_to_batched_modalities(batch, device=device)

                try:
                    all_tokens = _encode_many(codec_manager, batched)  # {token_key: [B, T]}
                except Exception as e:
                    print(f"Error encoding batch: {e}")
                    continue

            if len(all_tokens) < 2:
                # Need at least 2 modalities for cross-modal masking training
                continue

            # Mask some modalities/tokens
            all_tokens, masked_raw = apply_masking(all_tokens, mask_ratio=0.3)

            # --- Normalize 'masked' into a dict {key: BoolTensor[B, T]} ---
            def _normalize_masked(masked, all_tokens):
                # Case 1: already a dict
                if isinstance(masked, dict):
                    return {k: v.bool() for k, v in masked.items()}

                # Case 2: list of keys => whole modality masked (all positions True)
                if isinstance(masked, list) and (len(masked) == 0 or isinstance(masked[0], str)):
                    out = {}
                    for k in masked:
                        if k in all_tokens:
                            out[k] = torch.ones_like(all_tokens[k], dtype=torch.bool)
                    return out

                # Case 3: list of (key, mask) pairs
                if isinstance(masked, list) and len(masked) > 0 and isinstance(masked[0], (tuple, list)):
                    out = {}
                    for item in masked:
                        if len(item) != 2:
                            continue
                        k, m = item
                        if k in all_tokens:
                            out[k] = m.bool()
                    return out

                # Unknown format
                return {}

            masked = _normalize_masked(masked_raw, all_tokens)
            if len(masked) == 0:
                # nothing masked this step
                continue

            # Forward pass
            try:
                preds_raw = model(all_tokens, masked_raw)

                # --- Normalize 'predictions' to dict {key: logits_for_masked_positions} ---
                def _normalize_predictions(preds, masked):
                    # If dict already, just keep the keys that exist in 'masked'
                    if isinstance(preds, dict):
                        return {k: v for k, v in preds.items() if k in masked}

                    # If list, align by the iteration order of masked keys
                    if isinstance(preds, list):
                        keys = list(masked.keys())
                        out = {}
                        for i, p in enumerate(preds):
                            if i < len(keys):
                                out[keys[i]] = p
                        return out

                    # Unknown format
                    return {}

                predictions = _normalize_predictions(preds_raw, masked)

                loss = 0.0
                acc = 0.0
                n = 0

                for key, mask_pos in masked.items():
                    if key not in predictions or key not in all_tokens:
                        continue

                    pred = predictions[key]          # [M, V] (masked positions)
                    target = all_tokens[key]         # [B, T]

                    # Sanity checks
                    if mask_pos.dtype != torch.bool:
                        mask_pos = mask_pos.bool()
                    if mask_pos.shape != target.shape:
                        print(f"Mask/target shape mismatch for {key}: {mask_pos.shape} vs {target.shape}")
                        continue

                    out = _masked_cross_entropy(pred, target, mask_pos)
                    if out is None or out[0] is None:
                        continue

                    l, a = out
                    loss += l
                    acc += a
                    n += 1

                if n == 0:
                    continue

                loss = loss / n
                acc = acc / n

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                train_acc += acc.item()
                num_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc.item():.4f}'})

            except Exception as e:
                print(f"Error in forward pass: {e}")
                continue

        if num_batches > 0:
            avg_loss = train_loss / num_batches
            avg_acc = train_acc / num_batches
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

    print("\n✓ Training complete!")
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, './astronomical_model.pt')
    print("✓ Saved to: ./astronomical_model.pt")


if __name__ == "__main__":
    quick_train()