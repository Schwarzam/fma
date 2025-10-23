"""
Test if the trained model can reconstruct images
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from astronomical_dataset import AstronomicalDataset, collate_astronomical, GalaxyImage
from torch.utils.data import DataLoader
from core.codec_manager import CodecManager
from core.train_transformer import MultimodalTransformer

print("=" * 80)
print("Testing Model's Image Reconstruction Ability")
print("=" * 80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# Load data
dataset = AstronomicalDataset(
    data_root='./example_data',
    manifest_path='./example_data/metadata/val_manifest.json',
    image_size=96
)
loader = DataLoader(dataset, batch_size=1, collate_fn=collate_astronomical)
batch = next(iter(loader))

# Setup codec
codec_manager = CodecManager(device=device)

# Encode all modalities
def encode_batch(batch, codec_manager, device):
    tokens = {}
    for key, modalities in batch.items():
        if key == 'sample_ids':
            continue
        if isinstance(modalities, list) and len(modalities) > 0:
            mod = modalities[0]
            # Move to device
            if hasattr(mod, 'flux'):
                mod.flux = mod.flux.to(device)
            if hasattr(mod, 'ivar'):
                mod.ivar = mod.ivar.to(device)
            if hasattr(mod, 'mask'):
                mod.mask = mod.mask.to(device)
            if hasattr(mod, 'wavelength'):
                mod.wavelength = mod.wavelength.to(device)
            if hasattr(mod, 'mjd'):
                mod.mjd = mod.mjd.to(device)
            if hasattr(mod, 'flux_err'):
                mod.flux_err = mod.flux_err.to(device)
            if hasattr(mod, 'value'):
                mod.value = mod.value.to(device)

            encoded = codec_manager.encode(mod)
            tokens.update(encoded)
    return tokens

all_tokens = encode_batch(batch, codec_manager, device)

print("Encoded tokens:")
for k, v in all_tokens.items():
    print(f"  {k}: {v.shape}")

# Load model
vocab_sizes = {
    "tok_galaxy_image": 1000,
    "tok_gaia_spectrum": 512,
    "tok_ztf_lightcurve": 512,
    "tok_redshift": 256,
    "tok_stellar_mass": 256,
    "tok_sfr": 256,
}

num_tokens = {k: v.shape[1] for k, v in all_tokens.items()}

model = MultimodalTransformer(
    vocab_sizes=vocab_sizes,
    num_tokens=num_tokens,
    d_model=512,
    nhead=8,
    num_layers=8,
    dim_feedforward=2048,
    dropout=0.1
).to(device)

checkpoint = torch.load('astronomical_model.pt', map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

print("\nModel loaded successfully!\n")

# Test 1: Reconstruct image when ALL modalities are visible
print("Test 1: Reconstruct image with NO masking (all visible)")
print("-" * 80)

with torch.no_grad():
    # Mask only the image
    masked = ['tok_galaxy_image']
    predictions = model(all_tokens, masked)

    if 'tok_galaxy_image' in predictions:
        pred_logits = predictions['tok_galaxy_image']  # [B, T, vocab]
        pred_tokens = pred_logits.argmax(dim=-1)  # [B, T]

        print(f"Predicted tokens shape: {pred_tokens.shape}")
        print(f"Original tokens shape: {all_tokens['tok_galaxy_image'].shape}")

        # Decode both
        original_decoded = codec_manager.decode(all_tokens['tok_galaxy_image'], GalaxyImage)
        predicted_decoded = codec_manager.decode(pred_tokens, GalaxyImage)

        # Compute MSE
        mse = torch.nn.functional.mse_loss(original_decoded.flux, predicted_decoded.flux)
        print(f"Reconstruction MSE: {mse.item():.4f}")

        if mse.item() < 0.3:
            print("✓ Model can reconstruct images well!")
        elif mse.item() < 0.5:
            print("⚠ Model reconstructs but with high error")
        else:
            print("✗ Model cannot reconstruct images (untrained or broken)")

        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        orig = original_decoded.flux.squeeze().permute(1, 2, 0).cpu().numpy()
        pred = predicted_decoded.flux.squeeze().permute(1, 2, 0).cpu().numpy()

        axes[0].imshow(np.clip(orig, 0, 1))
        axes[0].set_title('Original (from codec)')
        axes[0].axis('off')

        axes[1].imshow(np.clip(pred, 0, 1))
        axes[1].set_title(f'Model Prediction (MSE={mse.item():.3f})')
        axes[1].axis('off')

        axes[2].imshow(np.clip(np.abs(orig - pred), 0, 1))
        axes[2].set_title('Absolute Difference')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig('model_image_reconstruction_test.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved visualization: model_image_reconstruction_test.png")

print("\n" + "=" * 80)
print("Analysis:")
print("=" * 80)
print("""
If MSE is:
- < 0.3: Model has learned image representations
- 0.3-0.5: Model is learning but needs more training
- > 0.5: Model hasn't learned images yet (need much more training)

Note: Even a well-trained model will have MSE ~0.2-0.3 due to lossy codec.
The codec itself has MSE ~0.21, so perfect reconstruction is impossible.
""")
