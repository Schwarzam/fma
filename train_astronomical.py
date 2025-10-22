"""
train_astronomical.py - Quick training script for astronomical data

Simple example showing how to train on the generated data.
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
from step4_codec_manager import CodecManager, MODALITY_CODEC_MAPPING
from step6_train_transformer import MultimodalTransformer, apply_masking


# Register astronomical modalities with their specific codecs
MODALITY_CODEC_MAPPING.update({
    GalaxyImage: GalaxyImageCodec,
    GaiaSpectrum: GaiaSpectrumCodec,
    ZTFLightCurve: ZTFLightCurveCodec,
    Redshift: RedshiftCodec,
    StellarMass: StellarMassCodec,
    StarFormationRate: StarFormationRateCodec,
})


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
        num_workers=0  # Set to 2-4 for faster loading
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

    # Create transformer
    print("Creating transformer...")
    vocab_sizes = {
        "tok_galaxy_image": 10000,
        "tok_gaia_spectrum": 512,
        "tok_ztf_lightcurve": 512,
        "tok_redshift": 256,
        "tok_stellar_mass": 256,
        "tok_sfr": 256,
    }

    num_tokens = {
        "tok_galaxy_image": 784,
        "tok_gaia_spectrum": 8000,
        "tok_ztf_lightcurve": 8000,
        "tok_redshift": 1,
        "tok_stellar_mass": 1,
        "tok_sfr": 1,
    }

    model = MultimodalTransformer(
        vocab_sizes=vocab_sizes,
        num_tokens=num_tokens,
        d_model=256,  # Small for demo
        nhead=4,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

    # Training loop
    num_epochs = 3
    print(f"Training for {num_epochs} epochs...")
    print("=" * 80)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            # Encode all modalities with frozen codecs
            all_tokens = {}

            with torch.no_grad():
                for modality_name, modality_list in batch.items():
                    if modality_name == 'sample_ids':
                        continue

                    if len(modality_list) == 0:
                        continue

                    # Concatenate modalities
                    if modality_name == 'galaxy_image':
                        flux = torch.cat([m.flux for m in modality_list], dim=0)
                        batched = GalaxyImage(flux=flux.to(device), metadata={})
                    elif modality_name == 'gaia_spectrum':
                        flux = torch.cat([m.flux for m in modality_list], dim=0)
                        wavelength = modality_list[0].wavelength
                        mask = torch.cat([m.mask for m in modality_list], dim=0)
                        batched = GaiaSpectrum(
                            flux=flux.to(device),
                            ivar=torch.cat([m.ivar for m in modality_list], dim=0).to(device),
                            mask=mask.to(device),
                            wavelength=wavelength.to(device),
                            metadata={}
                        )
                    elif modality_name == 'ztf_lightcurve':
                        flux = torch.cat([m.flux for m in modality_list], dim=0)
                        mjd = modality_list[0].mjd
                        batched = ZTFLightCurve(
                            flux=flux.to(device),
                            flux_err=torch.cat([m.flux_err for m in modality_list], dim=0).to(device),
                            mjd=mjd.to(device),
                            metadata={}
                        )
                    elif modality_name == 'redshift':
                        values = torch.cat([m.value for m in modality_list], dim=0)
                        batched = Redshift(value=values.to(device), metadata={})
                    elif modality_name == 'stellar_mass':
                        values = torch.cat([m.value for m in modality_list], dim=0)
                        batched = StellarMass(value=values.to(device), metadata={})
                    elif modality_name == 'sfr':
                        values = torch.cat([m.value for m in modality_list], dim=0)
                        batched = StarFormationRate(value=values.to(device), metadata={})
                    else:
                        continue

                    # Encode
                    try:
                        tokens = codec_manager.encode(batched)
                        all_tokens.update(tokens)
                    except Exception as e:
                        print(f"Error encoding {modality_name}: {e}")
                        continue

            if len(all_tokens) < 2:
                continue  # Need at least 2 modalities

            # Mask some modalities
            _, masked = apply_masking(all_tokens, mask_ratio=0.3)

            if len(masked) == 0:
                continue

            # Forward pass
            try:
                predictions = model(all_tokens, masked)

                # Compute loss
                loss = 0
                acc = 0
                n = 0

                for key in masked:
                    if key in predictions and key in all_tokens:
                        pred = predictions[key]
                        target = all_tokens[key]

                        pred_flat = pred.reshape(-1, pred.size(-1))
                        target_flat = target.reshape(-1).long()

                        loss += F.cross_entropy(pred_flat, target_flat)
                        acc += (pred.argmax(-1) == target.long()).float().mean()
                        n += 1

                if n > 0:
                    loss = loss / n
                    acc = acc / n

                    # Backward
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
