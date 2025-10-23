"""
Debug script to check num_tokens mismatch between training and inference
"""

import torch
from pathlib import Path

# Load checkpoint
checkpoint_path = Path("./astronomical_model.pt")
if not checkpoint_path.exists():
    print(f"Checkpoint not found at {checkpoint_path}")
    exit(1)

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("=" * 80)
print("CHECKPOINT ANALYSIS")
print("=" * 80)

# Check what's in the checkpoint
print("\nCheckpoint keys:")
for key in checkpoint.keys():
    print(f"  - {key}")

# Try to infer num_tokens from model architecture
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']

    # Check positional embedding size
    if 'pos_embedding' in state_dict:
        pos_emb_shape = state_dict['pos_embedding'].shape
        print(f"\nPositional embedding shape: {pos_emb_shape}")
        print(f"  -> max_seq_len = {pos_emb_shape[1]}")
        print(f"  -> This is sum of all num_tokens")

    # Check token embedding sizes
    print("\nToken embedding layers (vocab_sizes):")
    for key in state_dict.keys():
        if 'token_embeddings' in key and 'weight' in key:
            shape = state_dict[key].shape
            token_key = key.split('.')[1]
            print(f"  {token_key}: vocab_size={shape[0]}, d_model={shape[1]}")

    # Check prediction head sizes
    print("\nPrediction head layers (vocab_sizes):")
    for key in state_dict.keys():
        if 'prediction_heads' in key and 'weight' in key:
            shape = state_dict[key].shape
            token_key = key.split('.')[1]
            print(f"  {token_key}: vocab_size={shape[0]}, d_model={shape[1]}")

# Check if num_tokens or vocab_sizes are saved
if 'num_tokens' in checkpoint:
    print("\nnum_tokens from checkpoint:")
    for k, v in checkpoint['num_tokens'].items():
        print(f"  {k}: {v}")

if 'vocab_sizes' in checkpoint:
    print("\nvocab_sizes from checkpoint:")
    for k, v in checkpoint['vocab_sizes'].items():
        print(f"  {k}: {v}")

print("\n" + "=" * 80)
print("Now checking actual encoding with current codecs...")
print("=" * 80)

from astronomical_dataset import AstronomicalDataset, collate_astronomical
from torch.utils.data import DataLoader
from step4_codec_manager import CodecManager

# Load one batch
dataset = AstronomicalDataset(
    data_root="./example_data",
    manifest_path="./example_data/metadata/val_manifest.json",
    image_size=96
)

loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_astronomical)
batch = next(iter(loader))

# Setup codec manager
codec_manager = CodecManager(device='cpu')

# Encode each modality
from train_astronomical import _batch_to_batched_modalities, _encode_many

batched = _batch_to_batched_modalities(batch, device='cpu')
encoded = _encode_many(codec_manager, batched)

print("\nActual encoded token lengths:")
for key, tokens in encoded.items():
    print(f"  {key}: {tokens.shape}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nIf 'Positional embedding' max_seq_len doesn't match the sum of")
print("'Actual encoded token lengths', you have a num_tokens mismatch!")
print("\nThe model was trained with different num_tokens than your current")
print("codec configuration produces. You need to either:")
print("  1. Retrain the model with current codecs")
print("  2. Adjust codecs to match the model's expected num_tokens")
