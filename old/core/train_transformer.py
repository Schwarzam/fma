"""
Step 6: Train Transformer on Masked Modeling

This is the final step: training a transformer to learn correlations between modalities.

Key concepts:
1. Load PRE-TRAINED codecs (frozen, no gradients)
2. Tokenize all modalities using frozen codecs
3. Randomly mask some modalities
4. Train transformer to predict masked modalities from visible ones
5. Loss: Cross-entropy on discrete token predictions

This is the "4M" (Multimodal Masked Modeling) objective used in AION.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from typing import Dict, List, Optional
from tqdm import tqdm
import random

# Import from previous steps
from core.define_modalities import MyImage, MyTimeSeries, MyScalar, MyTabular, BaseModality
from core.implement_codecs import ImageCodec, TimeSeriesCodec, ScalarCodec, TabularCodec
from core.codec_manager import CodecManager


# ============================================================================
# Multimodal Dataset
# ============================================================================

class MultimodalDataset(Dataset):
    """
    Dataset that returns multiple modalities per sample.

    In practice, this would load aligned multimodal data
    (e.g., medical records with images + time series + tabular features).

    For demonstration, we generate synthetic data.
    """

    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate all modalities for one sample
        # In practice: load from disk/database
        # Return raw tensors - collate_fn will create modality objects

        return {
            'image': torch.randn(3, 224, 224),  # [C, H, W]
            'timeseries': {
                'values': torch.randn(1000),
                'timestamps': torch.linspace(0, 10, 1000),
                'mask': torch.ones(1000, dtype=torch.bool)
            },
            'scalar': torch.rand(1) * 100.0,  # [1]
            'tabular': torch.randn(32)  # [num_features]
        }


def collate_multimodal(batch: List[Dict]) -> Dict[str, List[BaseModality]]:
    """
    Custom collate function for multimodal data.

    Creates batched modality objects from raw tensors.

    Args:
        batch: List of dicts with raw tensors

    Returns:
        Dict mapping modality type to list of modality instances
    """
    batch_size = len(batch)

    # Collect all modalities
    images = []
    timeseries = []
    scalars = []
    tabulars = []

    for sample in batch:
        # Image: add batch dimension
        images.append(sample['image'].unsqueeze(0))  # [1, C, H, W]

        # TimeSeries
        timeseries.append({
            'values': sample['timeseries']['values'].unsqueeze(0),  # [1, seq_len]
            'timestamps': sample['timeseries']['timestamps'],
            'mask': sample['timeseries']['mask'].unsqueeze(0)  # [1, seq_len]
        })

        # Scalar: add batch dimension
        scalars.append(sample['scalar'].unsqueeze(0))  # [1, 1]

        # Tabular: add batch dimension
        tabulars.append(sample['tabular'].unsqueeze(0))  # [1, num_features]

    # Create modality instances (one per sample for now)
    image_modalities = [MyImage(pixels=img, metadata={}) for img in images]

    timeseries_modalities = [
        MyTimeSeries(
            values=ts['values'],
            timestamps=ts['timestamps'],
            mask=ts['mask']
        ) for ts in timeseries
    ]

    scalar_modalities = [
        MyScalar(value=s, name="measurement") for s in scalars
    ]

    tabular_modalities = [
        MyTabular(
            features=t,
            feature_names=[f"feature_{i}" for i in range(t.shape[1])]
        ) for t in tabulars
    ]

    return {
        'image': image_modalities,
        'timeseries': timeseries_modalities,
        'scalar': scalar_modalities,
        'tabular': tabular_modalities
    }


# ============================================================================
# Simple Transformer Model
# ============================================================================

class MultimodalTransformer(nn.Module):
    """
    Simple transformer for multimodal masked modeling.

    Architecture:
    1. Token embeddings for each modality
    2. Positional embeddings
    3. Transformer encoder
    4. Prediction heads for each modality

    Note: This is a simplified version. AION uses a more complex
          encoder-decoder architecture with modality-specific embeddings.
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        num_tokens: Dict[str, int],
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_sizes: Dict mapping token_key -> vocabulary size (codebook size)
            num_tokens: Dict mapping token_key -> number of tokens per modality
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.vocab_sizes = vocab_sizes
        self.num_tokens = num_tokens
        self.d_model = d_model

        # Token embeddings for each modality
        self.token_embeddings = nn.ModuleDict()
        for token_key, vocab_size in vocab_sizes.items():
            self.token_embeddings[token_key] = nn.Embedding(vocab_size, d_model)

        # Modality type embeddings (to distinguish modality types)
        # Use nn.ParameterDict instead of nn.ModuleDict for parameters
        self.modality_embeddings = nn.ParameterDict()
        for token_key in vocab_sizes.keys():
            self.modality_embeddings[token_key] = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embeddings (learned)
        max_seq_len = sum(num_tokens.values())
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction heads for each modality
        self.prediction_heads = nn.ModuleDict()
        for token_key, vocab_size in vocab_sizes.items():
            self.prediction_heads[token_key] = nn.Linear(d_model, vocab_size)

        # Special token for masked positions
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

    def embed_tokens(
        self,
        tokens: Dict[str, torch.Tensor],
        masked_modalities: List[str]
    ) -> torch.Tensor:
        """
        Embed tokens for all modalities into a sequence.

        Args:
            tokens: Dict mapping token_key -> token tensor [B, N]
            masked_modalities: List of token_keys that are masked

        Returns:
            Embedded sequence [B, total_seq_len, d_model]
        """
        batch_size = next(iter(tokens.values())).shape[0]
        embedded_sequence = []
        token_key_order = []

        for token_key in self.vocab_sizes.keys():
            if token_key in tokens and token_key not in masked_modalities:
                # Visible modality - embed tokens
                token_tensor = tokens[token_key]  # [B, N]
                embedded = self.token_embeddings[token_key](token_tensor.long())  # [B, N, d_model]
                embedded = embedded + self.modality_embeddings[token_key]  # Add modality type embedding
                embedded_sequence.append(embedded)
                token_key_order.append(token_key)
            elif token_key in masked_modalities:
                # Masked modality - use mask token
                num_tokens = self.num_tokens[token_key]
                mask_tokens = self.mask_token.expand(batch_size, num_tokens, -1)
                mask_tokens = mask_tokens + self.modality_embeddings[token_key]
                embedded_sequence.append(mask_tokens)
                token_key_order.append(token_key)

        # Concatenate all embeddings
        embedded = torch.cat(embedded_sequence, dim=1)  # [B, total_seq_len, d_model]

        # Add positional embeddings
        embedded = embedded + self.pos_embedding[:, :embedded.shape[1], :]

        return embedded, token_key_order

    def forward(
        self,
        tokens: Dict[str, torch.Tensor],
        masked_modalities: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: predict masked modalities from visible ones.

        Args:
            tokens: Dict of all tokens (both visible and masked)
            masked_modalities: Which modalities to predict

        Returns:
            Dict mapping token_key -> logits [B, N, vocab_size] for masked modalities
        """
        # Embed tokens
        embedded, token_key_order = self.embed_tokens(tokens, masked_modalities)

        # Transformer
        encoded = self.transformer(embedded)  # [B, total_seq_len, d_model]

        # Extract predictions for masked modalities
        predictions = {}
        current_pos = 0

        for token_key in token_key_order:
            num_tokens = self.num_tokens[token_key]

            if token_key in masked_modalities:
                # Get encoded tokens for this modality
                modality_encoded = encoded[:, current_pos:current_pos + num_tokens, :]  # [B, N, d_model]

                # Predict tokens
                logits = self.prediction_heads[token_key](modality_encoded)  # [B, N, vocab_size]
                predictions[token_key] = logits

            current_pos += num_tokens

        return predictions


# ============================================================================
# Training Functions
# ============================================================================

def apply_masking(
    tokens: Dict[str, torch.Tensor],
    mask_ratio: float = 0.3,
    modality_keys: Optional[List[str]] = None
) -> tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Randomly mask modalities for training.

    Args:
        tokens: Dict of all tokens
        mask_ratio: Fraction of modalities to mask (0.0 - 1.0)
        modality_keys: List of modality keys (if None, use all)

    Returns:
        tokens: Same dict (not modified)
        masked_modalities: List of masked modality keys
    """
    if modality_keys is None:
        modality_keys = list(tokens.keys())

    # Randomly select modalities to mask
    num_to_mask = max(1, int(len(modality_keys) * mask_ratio))
    masked_modalities = random.sample(modality_keys, num_to_mask)

    return tokens, masked_modalities


def train_transformer(
    model: MultimodalTransformer,
    codec_manager: CodecManager,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    lr: float = 1e-4,
    mask_ratio: float = 0.3,
    device: str = "cpu",
    checkpoint_dir: str = "./checkpoints"
):
    """
    Train transformer on multimodal masked modeling.

    Args:
        model: MultimodalTransformer instance
        codec_manager: CodecManager with pre-trained codecs
        train_loader: DataLoader for training data
        val_loader: Optional validation loader
        num_epochs: Number of training epochs
        lr: Learning rate
        mask_ratio: Fraction of modalities to mask
        device: Device to train on
        checkpoint_dir: Where to save checkpoints
    """
    print("=" * 80)
    print("Training Multimodal Transformer")
    print("=" * 80)
    print(f"Mask ratio: {mask_ratio}")
    print(f"Device: {device}")

    # Setup
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    # Cosine learning rate schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(train_loader)
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 80)

        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        model.train()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # 1. Tokenize all modalities using FROZEN codecs
            all_tokens = {}
            with torch.no_grad():
                for modality_type, modality_list in batch.items():
                    # Stack modalities into batches
                    # Each modality already has shape [1, ...], so concatenate along batch dim
                    if modality_type == 'image':
                        pixels = torch.cat([m.pixels for m in modality_list], dim=0)  # [B, C, H, W]
                        batched_modality = MyImage(pixels=pixels, metadata={})
                    elif modality_type == 'timeseries':
                        values = torch.cat([m.values for m in modality_list], dim=0)  # [B, seq_len]
                        timestamps = modality_list[0].timestamps
                        mask = torch.cat([m.mask for m in modality_list], dim=0)  # [B, seq_len]
                        batched_modality = MyTimeSeries(values=values, timestamps=timestamps, mask=mask)
                    elif modality_type == 'scalar':
                        values = torch.cat([m.value for m in modality_list], dim=0)  # [B, 1]
                        batched_modality = MyScalar(value=values, name="measurement")
                    elif modality_type == 'tabular':
                        features = torch.cat([m.features for m in modality_list], dim=0)  # [B, num_features]
                        batched_modality = MyTabular(features=features, feature_names=modality_list[0].feature_names)

                    # Encode with frozen codec
                    tokens = codec_manager.encode(batched_modality)
                    all_tokens.update(tokens)

            # Move tokens to device
            all_tokens = {k: v.to(device) for k, v in all_tokens.items()}

            # 2. Randomly mask some modalities
            _, masked_modalities = apply_masking(all_tokens, mask_ratio=mask_ratio)

            # 3. Forward pass: predict masked modalities
            predictions = model(all_tokens, masked_modalities)

            # 4. Compute loss on masked modalities only
            loss = 0
            accuracy = 0
            num_predictions = 0

            for token_key in masked_modalities:
                if token_key in predictions and token_key in all_tokens:
                    pred_logits = predictions[token_key]  # [B, N, vocab_size]
                    target_tokens = all_tokens[token_key]  # [B, N]

                    # Flatten for cross-entropy
                    pred_flat = pred_logits.reshape(-1, pred_logits.size(-1))  # [B*N, vocab_size]
                    target_flat = target_tokens.reshape(-1).long()  # [B*N]

                    # Cross-entropy loss
                    loss += F.cross_entropy(pred_flat, target_flat)

                    # Accuracy
                    pred_tokens = pred_logits.argmax(dim=-1)  # [B, N]
                    correct = (pred_tokens == target_tokens.long()).float().mean()
                    accuracy += correct.item()
                    num_predictions += 1

            if num_predictions > 0:
                loss = loss / num_predictions
                accuracy = accuracy / num_predictions

            # 5. Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Track metrics
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1

        # Print epoch metrics
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        print(f"\nTraining Metrics:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {avg_accuracy:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            val_accuracy = 0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    # Tokenize
                    all_tokens = {}
                    for modality_type, modality_list in batch.items():
                        if modality_type == 'image':
                            pixels = torch.cat([m.pixels for m in modality_list], dim=0)
                            batched_modality = MyImage(pixels=pixels, metadata={})
                        elif modality_type == 'timeseries':
                            values = torch.cat([m.values for m in modality_list], dim=0)
                            timestamps = modality_list[0].timestamps
                            mask = torch.cat([m.mask for m in modality_list], dim=0)
                            batched_modality = MyTimeSeries(values=values, timestamps=timestamps, mask=mask)
                        elif modality_type == 'scalar':
                            values = torch.cat([m.value for m in modality_list], dim=0)
                            batched_modality = MyScalar(value=values, name="measurement")
                        elif modality_type == 'tabular':
                            features = torch.cat([m.features for m in modality_list], dim=0)
                            batched_modality = MyTabular(features=features, feature_names=modality_list[0].feature_names)

                        tokens = codec_manager.encode(batched_modality)
                        all_tokens.update(tokens)

                    all_tokens = {k: v.to(device) for k, v in all_tokens.items()}

                    # Mask
                    _, masked_modalities = apply_masking(all_tokens, mask_ratio=mask_ratio)

                    # Predict
                    predictions = model(all_tokens, masked_modalities)

                    # Loss
                    batch_loss = 0
                    batch_accuracy = 0
                    num_predictions = 0

                    for token_key in masked_modalities:
                        if token_key in predictions and token_key in all_tokens:
                            pred_logits = predictions[token_key]
                            target_tokens = all_tokens[token_key]

                            pred_flat = pred_logits.reshape(-1, pred_logits.size(-1))
                            target_flat = target_tokens.reshape(-1).long()

                            batch_loss += F.cross_entropy(pred_flat, target_flat).item()

                            pred_tokens = pred_logits.argmax(dim=-1)
                            correct = (pred_tokens == target_tokens.long()).float().mean()
                            batch_accuracy += correct.item()
                            num_predictions += 1

                    if num_predictions > 0:
                        val_loss += batch_loss / num_predictions
                        val_accuracy += batch_accuracy / num_predictions
                        val_batches += 1

            if val_batches > 0:
                print(f"  Val Loss: {val_loss / val_batches:.4f}")
                print(f"  Val Accuracy: {val_accuracy / val_batches:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"transformer_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

    print("\n✓ Transformer training complete!")
    return model


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("STEP 6: Training Multimodal Transformer")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # ========================================================================
    # 1. Setup Pre-trained Codecs (frozen)
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. Loading Pre-trained Codecs")
    print("=" * 80)

    # Initialize codecs (in practice, load from checkpoints)
    codec_manager = CodecManager(device=device)

    print("✓ Codec manager initialized")
    print("  Note: In practice, load pre-trained codec weights from Step 5")
    print("  For this demo, using randomly initialized codecs")

    # ========================================================================
    # 2. Create Multimodal Dataset
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. Creating Multimodal Dataset")
    print("=" * 80)

    train_dataset = MultimodalDataset(num_samples=100)
    val_dataset = MultimodalDataset(num_samples=20)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_multimodal
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_multimodal
    )

    print(f"✓ Created datasets")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Batch size: 4")

    # ========================================================================
    # 3. Define Vocabulary Sizes (from quantizers)
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. Defining Model Architecture")
    print("=" * 80)

    # These should match your quantizer codebook sizes
    vocab_sizes = {
        "tok_my_image": 10000,      # FSQ with levels [8,5,5,5] = 5000 codes per channel, 4 channels
        "tok_timeseries": 512,      # VQ with 512 codes
        "tok_scalar": 256,          # Linear quantizer with 256 bins
        "tok_tabular": 256,         # VQ with 256 codes
    }

    num_tokens = {
        "tok_my_image": 784,        # 14*14*4 flattened
        "tok_timeseries": 8000,     # 64*125 flattened
        "tok_scalar": 1,            # 1 token
        "tok_tabular": 64,          # 64*1 flattened
    }

    print(f"✓ Model configuration:")
    print(f"  Vocab sizes: {vocab_sizes}")
    print(f"  Num tokens: {num_tokens}")

    # ========================================================================
    # 4. Create Transformer Model
    # ========================================================================

    model = MultimodalTransformer(
        vocab_sizes=vocab_sizes,
        num_tokens=num_tokens,
        d_model=256,           # Smaller for demo
        nhead=4,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Created transformer with {total_params:,} parameters")

    # ========================================================================
    # 5. Train Transformer
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. Training Transformer")
    print("=" * 80)

    model = train_transformer(
        model=model,
        codec_manager=codec_manager,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,
        lr=1e-4,
        mask_ratio=0.3,
        device=device,
        checkpoint_dir="./checkpoints/transformer"
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("✓ Step 6 Complete: Transformer Training")
    print("=" * 80)
    print("\nComplete Pipeline:")
    print("  1. ✓ Defined modalities (Step 1)")
    print("  2. ✓ Defined quantizers (Step 2)")
    print("  3. ✓ Implemented codecs (Step 3)")
    print("  4. ✓ Created codec manager (Step 4)")
    print("  5. ✓ Trained codecs separately (Step 5)")
    print("  6. ✓ Trained transformer on masked modeling (Step 6)")

    print("\nWhat You've Built:")
    print("  • A complete multimodal tokenizer system")
    print("  • Pre-trained codecs for 4 data types")
    print("  • A transformer that learns correlations between modalities")
    print("  • Can predict missing modalities from observed ones")

    print("\n" + "=" * 80)
    print("Next Steps for Production:")
    print("  → Use real datasets instead of synthetic data")
    print("  → Train codecs for longer (50-100 epochs)")
    print("  → Use larger transformer (6-12 layers, 512-1024 dim)")
    print("  → Add more modalities as needed")
    print("  → Implement inference pipeline for predictions")
    print("  → Add evaluation metrics for your use case")
    print("=" * 80)
