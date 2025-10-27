"""
Codec Comparison Script

Compare multiple codec models (pretrained vs from-scratch) on reconstruction quality.
Evaluates MSE, PSNR, SSIM, and generates visual comparisons.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
from tqdm import tqdm

# Import codec implementations
from core.pretrained_codecs import PretrainedImageCodec, SimpleCodec
from core.implement_codecs import ImageCodec
from astronomical_codecs import GalaxyImageCodec
from core.define_quantizers import FiniteScalarQuantizer


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    Simplified implementation for grayscale or assumes RGB channels are averaged.
    """
    # Convert to grayscale if RGB
    if img1.shape[1] == 3:
        img1 = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
        img1 = img1.unsqueeze(1)
    if img2.shape[1] == 3:
        img2 = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
        img2 = img2.unsqueeze(1)

    # Constants
    C1 = (0.01 * 2) ** 2
    C2 = (0.03 * 2) ** 2

    # Mean
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Variance
    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2

    # SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


class CodecComparison:
    """Manages comparison between multiple codec models."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.codecs: Dict[str, torch.nn.Module] = {}
        self.results: Dict[str, Dict] = {}

    def add_codec(self, name: str, codec: torch.nn.Module):
        """Add a codec to compare."""
        codec.to(self.device)
        codec.eval()
        self.codecs[name] = codec
        print(f"✓ Added codec: {name}")

    def load_codec_from_checkpoint(self, name: str, codec: torch.nn.Module, checkpoint_path: str):
        """Load codec from checkpoint and add to comparison."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            codec.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"✓ Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"⚠ Checkpoint not found: {checkpoint_path}, using random init")

        self.add_codec(name, codec)

    def evaluate_on_batch(self, images: torch.Tensor) -> Dict[str, Dict]:
        """Evaluate all codecs on a batch of images."""
        images = images.to(self.device)
        batch_results = {}

        # Normalize to [-1, 1] if needed
        if images.max() > 1.0:
            images = images / 255.0
        images = images * 2.0 - 1.0

        for codec_name, codec in self.codecs.items():
            with torch.no_grad():
                recon, quant_loss, usage = codec(images)

                # Compute metrics
                mse = F.mse_loss(recon, images).item()
                psnr = compute_psnr(recon, images, max_val=2.0)  # Range is [-1, 1]
                ssim = compute_ssim(recon, images)

                # Codebook usage (if applicable)
                if usage is not None:
                    usage_pct = usage.item() if isinstance(usage, torch.Tensor) else usage
                else:
                    usage_pct = None

                batch_results[codec_name] = {
                    'mse': mse,
                    'psnr': psnr,
                    'ssim': ssim,
                    'quant_loss': quant_loss.item() if isinstance(quant_loss, torch.Tensor) else quant_loss,
                    'usage': usage_pct,
                    'reconstruction': recon.cpu(),
                }

        return batch_results

    def evaluate_on_dataloader(self, dataloader: DataLoader, num_batches: Optional[int] = None):
        """Evaluate all codecs on entire dataloader."""
        print(f"\n{'='*60}")
        print(f"Evaluating {len(self.codecs)} codecs on dataset...")
        print(f"{'='*60}\n")

        # Initialize results storage
        for codec_name in self.codecs.keys():
            self.results[codec_name] = {
                'mse': [],
                'psnr': [],
                'ssim': [],
                'quant_loss': [],
                'usage': [],
            }

        # Evaluate
        batch_count = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
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

            # Evaluate
            batch_results = self.evaluate_on_batch(images)

            # Aggregate results
            for codec_name, metrics in batch_results.items():
                self.results[codec_name]['mse'].append(metrics['mse'])
                self.results[codec_name]['psnr'].append(metrics['psnr'])
                self.results[codec_name]['ssim'].append(metrics['ssim'])
                self.results[codec_name]['quant_loss'].append(metrics['quant_loss'])
                if metrics['usage'] is not None:
                    self.results[codec_name]['usage'].append(metrics['usage'])

            batch_count += 1
            if num_batches is not None and batch_count >= num_batches:
                break

        # Compute averages
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*60}\n")

        for codec_name, metrics in self.results.items():
            avg_mse = np.mean(metrics['mse'])
            avg_psnr = np.mean(metrics['psnr'])
            avg_ssim = np.mean(metrics['ssim'])
            avg_quant = np.mean(metrics['quant_loss'])
            avg_usage = np.mean(metrics['usage']) if metrics['usage'] else None

            print(f"{codec_name}:")
            print(f"  MSE:        {avg_mse:.6f}")
            print(f"  PSNR:       {avg_psnr:.2f} dB")
            print(f"  SSIM:       {avg_ssim:.4f}")
            print(f"  Quant Loss: {avg_quant:.6f}")
            if avg_usage is not None:
                print(f"  Usage:      {avg_usage:.2f}%")
            print()

    def visualize_comparison(self, images: torch.Tensor, save_path: str = "codec_comparison.png"):
        """Create visual comparison of reconstructions."""
        images = images.to(self.device)

        # Normalize
        if images.max() > 1.0:
            images = images / 255.0
        images_normalized = images * 2.0 - 1.0

        # Get reconstructions
        reconstructions = {}
        with torch.no_grad():
            for codec_name, codec in self.codecs.items():
                recon, _, _ = codec(images_normalized)
                reconstructions[codec_name] = recon.cpu()

        # Plot
        num_codecs = len(self.codecs)
        num_images = min(4, images.shape[0])

        fig, axes = plt.subplots(num_images, num_codecs + 1, figsize=(4 * (num_codecs + 1), 4 * num_images))

        if num_images == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_images):
            # Original - apply asinh stretch for astronomical images
            img = images[i].cpu().permute(1, 2, 0).numpy()
            # Apply asinh stretch (common in astronomy) to make faint features visible
            img_stretched = np.arcsinh(img * 10) / np.arcsinh(10)
            img_stretched = np.clip(img_stretched, 0, 1)

            axes[i, 0].imshow(img_stretched)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis('off')

            # Reconstructions
            for j, (codec_name, recon) in enumerate(reconstructions.items()):
                recon_img = recon[i].permute(1, 2, 0).numpy()
                # Denormalize from [-1, 1] to [0, 1]
                recon_img = (recon_img + 1.0) / 2.0
                recon_img = np.clip(recon_img, 0, 1)

                # Apply same asinh stretch for fair comparison
                recon_stretched = np.arcsinh(recon_img * 10) / np.arcsinh(10)
                recon_stretched = np.clip(recon_stretched, 0, 1)

                axes[i, j + 1].imshow(recon_stretched)
                axes[i, j + 1].set_title(codec_name)
                axes[i, j + 1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison to {save_path}")
        plt.close()

    def plot_metrics(self, save_path: str = "codec_metrics.png"):
        """Plot comparison metrics as bar charts."""
        codec_names = list(self.results.keys())
        metrics_to_plot = ['mse', 'psnr', 'ssim']

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, metric in enumerate(metrics_to_plot):
            values = [np.mean(self.results[name][metric]) for name in codec_names]

            axes[idx].bar(codec_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(codec_names)])
            axes[idx].set_title(metric.upper())
            axes[idx].set_ylabel(metric.upper())
            axes[idx].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved metrics plot to {save_path}")
        plt.close()


def main():
    """Main comparison script."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize quantizer (shared across all codecs)
    quantizer = FiniteScalarQuantizer(levels=[8, 5, 5, 5])

    # Initialize comparison
    comparison = CodecComparison(device=device)

    # Add codecs to compare
    print("\n" + "="*60)
    print("Setting up codecs...")
    print("="*60 + "\n")

    # 1. Simple baseline codec (from scratch)
    simple_codec = SimpleCodec(embedding_dim=64, quantizer=quantizer)
    comparison.add_codec("Simple (scratch)", simple_codec)

    # 2. ResNet18 pretrained (frozen encoder)
    resnet18_frozen = PretrainedImageCodec(
        backbone="resnet18",
        embedding_dim=64,
        quantizer=quantizer,
        freeze_encoder=True,
        pretrained=True
    )
    comparison.add_codec("ResNet18 (frozen)", resnet18_frozen)

    # 3. ResNet18 pretrained (finetunable)
    resnet18_finetune = PretrainedImageCodec(
        backbone="resnet18",
        embedding_dim=64,
        quantizer=quantizer,
        freeze_encoder=False,
        pretrained=True
    )
    comparison.add_codec("ResNet18 (finetune)", resnet18_finetune)

    # 4. Load trained codec if available
    if os.path.exists('checkpoints/simple_final.pt'):
        trained_codec = SimpleCodec(embedding_dim=64, quantizer=quantizer)
        comparison.load_codec_from_checkpoint("Trained (simple)", trained_codec, 'checkpoints/simple_final.pt')
    elif os.path.exists('image_codec_final.pt'):
        trained_codec = SimpleCodec(embedding_dim=64, quantizer=quantizer)
        comparison.load_codec_from_checkpoint("Trained", trained_codec, 'image_codec_final.pt')

    # Load astronomical dataset
    try:
        from astronomical_dataset import AstronomicalDataset, collate_astronomical

        print("\nLoading test dataset...")
        test_dataset = AstronomicalDataset(
            data_root="./example_data",
            manifest_path="./example_data/metadata/test_manifest.json",
            image_size=224  # Resize to 224 for codecs
        )
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_astronomical)

        # Evaluate
        comparison.evaluate_on_dataloader(test_loader, num_batches=10)

        # Visualize
        test_batch = next(iter(test_loader))
        if isinstance(test_batch, dict) and 'galaxy_image' in test_batch:
            # Extract galaxy images from batch
            images = torch.cat([img.flux for img in test_batch['galaxy_image']], dim=0)
        elif isinstance(test_batch, dict):
            # Fallback: extract any modality with flux
            images = []
            for modality_key, modality_list in test_batch.items():
                for item in modality_list:
                    if hasattr(item, 'flux') and item.flux.ndim == 4:
                        images.append(item.flux)
            if images:
                images = torch.cat(images, dim=0)
        else:
            images = test_batch[0] if isinstance(test_batch, (list, tuple)) else test_batch

        comparison.visualize_comparison(images[:4], save_path="codec_reconstruction_samples.png")
        comparison.plot_metrics(save_path="codec_metrics_comparison.png")

    except Exception as e:
        print(f"\n⚠ Could not load dataset: {e}")
        print("Creating synthetic test data...")

        # Create synthetic test images
        test_images = torch.rand(16, 3, 224, 224)  # Random images
        test_images = test_images.to(device)

        # Evaluate on synthetic data
        batch_results = comparison.evaluate_on_batch(test_images)

        print(f"\n{'='*60}")
        print(f"RESULTS ON SYNTHETIC DATA")
        print(f"{'='*60}\n")

        for codec_name, metrics in batch_results.items():
            print(f"{codec_name}:")
            print(f"  MSE:        {metrics['mse']:.6f}")
            print(f"  PSNR:       {metrics['psnr']:.2f} dB")
            print(f"  SSIM:       {metrics['ssim']:.4f}")
            print(f"  Quant Loss: {metrics['quant_loss']:.6f}")
            print()

        # Visualize
        comparison.visualize_comparison(test_images[:4], save_path="codec_reconstruction_samples.png")

    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)


if __name__ == "__main__":
    main()
