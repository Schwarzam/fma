"""
Train the image codec (encoder + decoder + quantizer) for astronomical images
This must be done BEFORE training the transformer!
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from astronomical_dataset import AstronomicalDataset, collate_astronomical
from astronomical_codecs import GalaxyImageCodec

def train_image_codec(
    codec,
    train_loader,
    num_epochs=50,
    lr=1e-3,
    device='cuda'
):
    """Train image codec to reconstruct images"""

    codec = codec.to(device)
    optimizer = torch.optim.Adam(codec.parameters(), lr=lr)

    print(f"Training image codec for {num_epochs} epochs...")
    print(f"Device: {device}")
    print("=" * 80)

    for epoch in range(num_epochs):
        codec.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_quant_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            # Get images
            images = batch['galaxy_image']

            # Batch images - each img.flux is [1, 3, 96, 96], concat on batch dim
            image_batch = torch.cat([img.flux for img in images], dim=0).to(device)

            # Forward pass through codec
            # 1. Encode
            z_e = codec.encoder(image_batch * 2.0 - 1.0)  # Normalize to [-1,1]
            z_e = codec.projection(z_e)
            z_e = torch.tanh(z_e * 10.0)  # Scale for FSQ

            # 2. Quantize (with straight-through)
            z_q, quant_loss, usage = codec.quantizer(z_e)

            # 3. Decode
            z_unprojected = codec.unprojection(z_q)
            reconstructed = codec.decoder(z_unprojected)
            reconstructed = (reconstructed + 1.0) / 2.0  # Back to [0,1]

            # Compute reconstruction loss
            recon_loss = nn.functional.mse_loss(reconstructed, image_batch)

            # Total loss
            loss = recon_loss + 0.25 * quant_loss  # Weight quantizer loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_quant_loss += quant_loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'quant': f'{quant_loss.item():.4f}'
            })

        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon_loss / num_batches
        avg_quant = epoch_quant_loss / num_batches

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Recon Loss: {avg_recon:.4f}")
        print(f"  Quant Loss: {avg_quant:.4f}")
        print()

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': codec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'image_codec_epoch{epoch+1}.pt')
            print(f"  ✓ Saved checkpoint: image_codec_epoch{epoch+1}.pt")

    # Save final model
    torch.save({
        'model_state_dict': codec.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'image_codec_final.pt')
    print("\n✓ Training complete! Saved: image_codec_final.pt")

    return codec


def test_codec(codec, test_loader, device='cuda'):
    """Visualize codec reconstructions"""
    codec.eval()
    codec = codec.to(device)

    # Get one batch
    batch = next(iter(test_loader))
    images = batch['galaxy_image']

    # Take first 4 images - each is [1, 3, 96, 96]
    test_images = torch.cat([img.flux for img in images[:4]], dim=0).to(device)

    with torch.no_grad():
        # Encode and decode
        z_e = codec.encoder(test_images * 2.0 - 1.0)
        z_e = codec.projection(z_e)
        z_e = torch.tanh(z_e * 10.0)
        z_q, _, _ = codec.quantizer(z_e)
        z_unprojected = codec.unprojection(z_q)
        reconstructed = codec.decoder(z_unprojected)
        reconstructed = (reconstructed + 1.0) / 2.0

    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i in range(4):
        # Original
        orig = test_images[i].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(np.clip(orig, 0, 1))
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')

        # Reconstructed
        recon = reconstructed[i].permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(np.clip(recon, 0, 1))
        mse = nn.functional.mse_loss(
            reconstructed[i],
            test_images[i]
        ).item()
        axes[1, i].set_title(f'Recon (MSE={mse:.3f})')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('codec_reconstruction_samples.png', dpi=150)
    print("✓ Saved: codec_reconstruction_samples.png")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    print("Loading training data...")
    train_dataset = AstronomicalDataset(
        data_root="./example_data",
        manifest_path="./example_data/metadata/train_manifest.json",
        image_size=96
    )

    test_dataset = AstronomicalDataset(
        data_root="./example_data",
        manifest_path="./example_data/metadata/val_manifest.json",
        image_size=96
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_astronomical
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_astronomical
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print()

    # Create codec
    print("Creating image codec...")
    codec = GalaxyImageCodec(in_channels=3, embedding_dim=64)
    print(f"Parameters: {sum(p.numel() for p in codec.parameters()):,}")
    print()

    # Train
    codec = train_image_codec(
        codec,
        train_loader,
        num_epochs=50,
        lr=1e-3,
        device=device
    )

    # Test
    print("\nTesting trained codec...")
    test_codec(codec, test_loader, device=device)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Check codec_reconstruction_samples.png to verify quality")
    print("2. Use the trained codec in transformer training:")
    print("   - Load image_codec_final.pt in train_astronomical.py")
    print("   - Freeze codec parameters during transformer training")
