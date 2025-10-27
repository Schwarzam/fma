"""
Pretrained codec implementations for comparison.
Supports various pretrained backbones: ResNet, VGG, EfficientNet, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import torchvision.models as models


class PretrainedImageCodec(nn.Module):
    """
    Image codec using pretrained encoder backbones.
    Supports: resnet18, resnet34, resnet50, vgg16, efficientnet_b0
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 64,
        input_channels: int = 3,
        output_channels: int = 3,
        quantizer=None,
        freeze_encoder: bool = False,
        pretrained: bool = True,
    ):
        """
        Args:
            backbone: Name of pretrained model ('resnet18', 'resnet34', 'resnet50', 'vgg16', 'efficientnet_b0')
            embedding_dim: Dimension of latent space
            input_channels: Number of input channels (3 for RGB)
            output_channels: Number of output channels (3 for RGB)
            quantizer: Quantizer module (FSQ, VQ, etc.)
            freeze_encoder: Whether to freeze encoder weights
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()

        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.freeze_encoder = freeze_encoder

        # Build encoder from pretrained model
        self.encoder, encoder_output_dim, encoder_spatial_size = self._build_encoder(backbone, pretrained)

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Adapter to match embedding dimension
        self.encoder_adapter = nn.Conv2d(encoder_output_dim, embedding_dim, kernel_size=1)

        # Projection for quantizer (4 channels for FSQ with levels [8,5,5,5])
        self.projection = nn.Conv2d(embedding_dim, 4, kernel_size=1)

        # Quantizer
        self.quantizer = quantizer

        # Unprojection from quantizer
        self.unprojection = nn.Conv2d(4, embedding_dim, kernel_size=1)

        # Decoder (mirror of encoder)
        self.decoder = self._build_decoder(embedding_dim, encoder_spatial_size)

        self.encoder_output_dim = encoder_output_dim
        self.encoder_spatial_size = encoder_spatial_size

    def _build_encoder(self, backbone: str, pretrained: bool):
        """Build encoder from pretrained model."""

        if backbone == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            # Remove avgpool and fc layers, keep conv layers
            encoder = nn.Sequential(*list(model.children())[:-2])
            output_dim = 512  # ResNet18 output channels
            spatial_size = 7  # 224/32 = 7

        elif backbone == "resnet34":
            model = models.resnet34(pretrained=pretrained)
            encoder = nn.Sequential(*list(model.children())[:-2])
            output_dim = 512
            spatial_size = 7

        elif backbone == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            encoder = nn.Sequential(*list(model.children())[:-2])
            output_dim = 2048  # ResNet50 output channels
            spatial_size = 7

        elif backbone == "vgg16":
            model = models.vgg16(pretrained=pretrained)
            # Use features (conv layers) only
            encoder = model.features
            output_dim = 512
            spatial_size = 7

        elif backbone == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=pretrained)
            encoder = model.features
            output_dim = 1280  # EfficientNet-B0 output channels
            spatial_size = 7

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        return encoder, output_dim, spatial_size

    def _build_decoder(self, embedding_dim: int, spatial_size: int):
        """Build decoder to reconstruct image from latent."""

        # Calculate upsampling needed: spatial_size -> 224
        # For spatial_size=7: 7 -> 14 -> 28 -> 56 -> 112 -> 224 (5 upsamples)
        # For spatial_size=14: 14 -> 28 -> 56 -> 112 -> 224 (4 upsamples)

        if spatial_size == 7:
            decoder = nn.Sequential(
                # 7x7 -> 14x14
                nn.ConvTranspose2d(embedding_dim, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                # 14x14 -> 28x28
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                # 28x28 -> 56x56
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                # 56x56 -> 112x112
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),

                # 112x112 -> 224x224
                nn.ConvTranspose2d(32, self.output_channels, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
        elif spatial_size == 14:
            decoder = nn.Sequential(
                # 14x14 -> 28x28
                nn.ConvTranspose2d(embedding_dim, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                # 28x28 -> 56x56
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                # 56x56 -> 112x112
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                # 112x112 -> 224x224
                nn.ConvTranspose2d(64, self.output_channels, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
        else:
            raise ValueError(f"Unsupported spatial size: {spatial_size}")

        return decoder

    def _encode(self, x):
        """Encode input to continuous latent."""
        z = self.encoder(x)
        z = self.encoder_adapter(z)
        return z

    def _decode(self, z):
        """Decode latent to reconstruction."""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass: encode -> quantize -> decode."""
        # Encode
        z_e = self._encode(x)

        # Project for quantizer
        z_e = self.projection(z_e)

        # Scale for FSQ (expects input in [-1, 1])
        z_e = torch.tanh(z_e * 10.0)

        # Quantize
        if self.quantizer is not None:
            z_q, quant_loss, usage = self.quantizer(z_e)
        else:
            z_q = z_e
            quant_loss = torch.tensor(0.0, device=x.device)
            usage = None

        # Unproject from quantizer
        z_q = self.unprojection(z_q)

        # Decode
        x_recon = self._decode(z_q)

        return x_recon, quant_loss, usage

    def encode(self, x):
        """Encode and quantize to discrete tokens."""
        z_e = self._encode(x)
        z_e = self.projection(z_e)
        z_e = torch.tanh(z_e * 10.0)

        if self.quantizer is not None:
            z_q, _, _ = self.quantizer(z_e)
            # Convert to token indices if quantizer supports it
            if hasattr(self.quantizer, 'get_codebook_indices'):
                tokens = self.quantizer.get_codebook_indices(z_e)
            else:
                tokens = z_q
        else:
            tokens = z_e

        return tokens

    def decode(self, tokens):
        """Decode from discrete tokens."""
        if self.quantizer is not None and hasattr(self.quantizer, 'decode_tokens'):
            z_q = self.quantizer.decode_tokens(tokens)
        else:
            z_q = tokens

        z_q = self.unprojection(z_q)
        x_recon = self._decode(z_q)

        return x_recon


class SimpleCodec(nn.Module):
    """
    Simple baseline codec without pretrained weights (for comparison).
    This is similar to the original ImageCodec but cleaner.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        input_channels: int = 3,
        output_channels: int = 3,
        quantizer=None,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Simple encoder: 224 -> 112 -> 56 -> 28 -> 14
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # 224 -> 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 112 -> 56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 56 -> 28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, embedding_dim, kernel_size=4, stride=2, padding=1),  # 28 -> 14
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        # Projection for quantizer
        self.projection = nn.Conv2d(embedding_dim, 4, kernel_size=1)

        # Quantizer
        self.quantizer = quantizer

        # Unprojection
        self.unprojection = nn.Conv2d(4, embedding_dim, kernel_size=1)

        # Simple decoder: 14 -> 28 -> 56 -> 112 -> 224
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 256, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 28 -> 56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 56 -> 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),  # 112 -> 224
            nn.Tanh()
        )

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Encode
        z_e = self._encode(x)
        z_e = self.projection(z_e)
        z_e = torch.tanh(z_e * 10.0)

        # Quantize
        if self.quantizer is not None:
            z_q, quant_loss, usage = self.quantizer(z_e)
        else:
            z_q = z_e
            quant_loss = torch.tensor(0.0, device=x.device)
            usage = None

        # Decode
        z_q = self.unprojection(z_q)
        x_recon = self._decode(z_q)

        return x_recon, quant_loss, usage
