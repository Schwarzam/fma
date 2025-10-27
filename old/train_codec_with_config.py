"""
Train image codecs with configuration support.
Supports both pretrained and from-scratch codecs.

Usage:
    python train_codec_with_config.py --codec simple
    python train_codec_with_config.py --codec resnet18_finetune --epochs 100
    python train_codec_with_config.py --codec resnet50 --freeze  # Freeze encoder
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import os

# Import codec implementations
from core.pretrained_codecs import PretrainedImageCodec, SimpleCodec
from core.define_quantizers import FiniteScalarQuantizer, VectorQuantizer


def load_config(config_path: str = "codec_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_codec(codec_name: str, config: dict, quantizer):
    """Create codec from configuration."""
    codec_config = config['codecs'][codec_name]
    codec_type = codec_config['type']

    if codec_type == 'simple':
        codec = SimpleCodec(
            embedding_dim=codec_config['embedding_dim'],
            input_channels=codec_config['input_channels'],
            output_channels=codec_config['output_channels'],
            quantizer=quantizer
        )
    elif codec_type == 'pretrained':
        codec = PretrainedImageCodec(
            backbone=codec_config['backbone'],
            embedding_dim=codec_config['embedding_dim'],
            input_channels=codec_config['input_channels'],
            output_channels=codec_config['output_channels'],
            freeze_encoder=codec_config['freeze_encoder'],
            pretrained=codec_config['pretrained'],
            quantizer=quantizer
        )
    else:
        raise ValueError(f"Unknown codec type: {codec_type}")

    return codec


def create_quantizer(config: dict):
    """Create quantizer from configuration."""
    quant_config = config['quantizer']
    quant_type = quant_config['type']

    if quant_type == 'fsq':
        quantizer = FiniteScalarQuantizer(
            levels=quant_config['fsq']['levels']
        )
    elif quant_type == 'vq':
        quantizer = VectorQuantizer(
            num_embeddings=quant_config['vq']['num_embeddings'],
            embedding_dim=quant_config['vq']['embedding_dim'],
            commitment_cost=quant_config['vq']['commitment_cost']
        )
    elif quant_type == 'none':
        quantizer = None
    else:
        raise ValueError(f"Unknown quantizer type: {quant_type}")

    return quantizer


def get_perceptual_loss_fn(device):
    """Get VGG-based perceptual loss function."""
    try:
        import torchvision.models as models
        vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False

        def perceptual_loss(x, y):
            x_features = vgg((x + 1.0) / 2.0)  # Denormalize to [0, 1]
            y_features = vgg((y + 1.0) / 2.0)
            return F.mse_loss(x_features, y_features)

        return perceptual_loss
    except:
        print("⚠ Could not load VGG for perceptual loss, using MSE only")
        return None


def train_codec(
    codec_name: str,
    config: dict,
    train_loader: DataLoader,
    resume_from: str = None,
    override_epochs: int = None,
):
    """Train a codec with given configuration."""

    # Setup
    device = config['training']['device']
    num_epochs = override_epochs if override_epochs else config['training']['num_epochs']
    lr = float(config['training']['learning_rate'])  # Convert to float in case YAML parsed as string
    gradient_clip = float(config['training']['gradient_clip'])
    save_every = config['training']['save_every']
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)

    # Create quantizer
    quantizer = create_quantizer(config)
    if quantizer is not None:
        quantizer = quantizer.to(device)

    # Create codec
    codec = create_codec(codec_name, config, quantizer)
    codec = codec.to(device)

    # Print model info
    total_params = sum(p.numel() for p in codec.parameters())
    trainable_params = sum(p.numel() for p in codec.parameters() if p.requires_grad)
    print(f"\nCodec: {codec_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(codec.parameters(), lr=lr)

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        codec.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"✓ Resumed from epoch {start_epoch}")

    # Perceptual loss (optional)
    perceptual_loss_fn = None
    if config['loss'].get('use_perceptual', False):
        perceptual_loss_fn = get_perceptual_loss_fn(device)

    # Loss weights
    recon_weight = config['loss']['reconstruction_weight']
    quant_weight = config['loss']['quantizer_weight']
    percep_weight = config['loss'].get('perceptual_weight', 0.1)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training {codec_name} for {num_epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, num_epochs):
        codec.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_quant_loss = 0.0
        epoch_percep_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Extract images from batch
            if isinstance(batch, dict) and 'galaxy_image' in batch:
                # Astronomical dataset format - extract galaxy images
                images = torch.cat([img.flux for img in batch['galaxy_image']], dim=0)
            elif isinstance(batch, dict):
                # Fallback: find any image modality
                images = []
                for modality_key, modality_list in batch.items():
                    if 'image' in modality_key.lower():
                        for item in modality_list:
                            if hasattr(item, 'flux') and item.flux.ndim == 4:
                                images.append(item.flux)
                if images:
                    images = torch.cat(images, dim=0)
                else:
                    continue
            else:
                # Standard tensor batch
                images = batch[0] if isinstance(batch, (list, tuple)) else batch

            images = images.to(device)

            # Normalize to [-1, 1]
            if images.max() > 1.0:
                images = images / 255.0
            images = images * 2.0 - 1.0

            # Forward pass
            reconstructed, quant_loss, usage = codec(images)

            # Compute losses
            recon_loss = F.mse_loss(reconstructed, images)
            loss = recon_weight * recon_loss + quant_weight * quant_loss

            # Add perceptual loss if available
            if perceptual_loss_fn is not None:
                percep_loss = perceptual_loss_fn(reconstructed, images)
                loss = loss + percep_weight * percep_loss
                epoch_percep_loss += percep_loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(codec.parameters(), gradient_clip)

            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_quant_loss += quant_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'quant': f'{quant_loss.item():.4f}',
            })

        # Epoch statistics
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon_loss / num_batches
        avg_quant = epoch_quant_loss / num_batches
        avg_percep = epoch_percep_loss / num_batches if perceptual_loss_fn else 0

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss:  {avg_loss:.6f}")
        print(f"  Recon Loss:  {avg_recon:.6f}")
        print(f"  Quant Loss:  {avg_quant:.6f}")
        if perceptual_loss_fn:
            print(f"  Percep Loss: {avg_percep:.6f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            checkpoint_path = checkpoint_dir / f"{codec_name}_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': codec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = checkpoint_dir / f"{codec_name}_final.pt"
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': codec.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, final_path)
    print(f"\n✓ Training complete! Final model saved to: {final_path}")

    return codec


def main():
    parser = argparse.ArgumentParser(description="Train image codecs")
    parser.add_argument('--codec', type=str, default='simple',
                        help='Codec name from config (e.g., simple, resnet18_finetune)')
    parser.add_argument('--config', type=str, default='codec_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs from config')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--freeze', action='store_true',
                        help='Freeze encoder (for pretrained models)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override freeze setting if specified
    if args.freeze and args.codec in config['codecs']:
        config['codecs'][args.codec]['freeze_encoder'] = True

    # Create dataset
    print("Loading dataset...")
    try:
        from astronomical_dataset import AstronomicalDataset, collate_astronomical

        train_dataset = AstronomicalDataset(
            data_root="./example_data",
            manifest_path="./example_data/metadata/train_manifest.json",
            image_size=224  # Resize to 224 for codecs
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_astronomical
        )
        print(f"✓ Loaded {len(train_dataset)} training samples")

    except Exception as e:
        print(f"⚠ Could not load astronomical dataset: {e}")
        print("Creating synthetic dataset...")

        # Create synthetic dataset
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples=1000):
                self.num_samples = num_samples

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return torch.rand(3, 224, 224)

        train_dataset = SyntheticDataset(config['dataset']['num_train_samples'])
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True
        )
        print(f"✓ Created synthetic dataset with {len(train_dataset)} samples")

    # Train
    codec = train_codec(
        codec_name=args.codec,
        config=config,
        train_loader=train_loader,
        resume_from=args.resume,
        override_epochs=args.epochs,
    )

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"\nTo compare this codec with others, run:")
    print(f"  python compare_codecs.py")


if __name__ == "__main__":
    main()
