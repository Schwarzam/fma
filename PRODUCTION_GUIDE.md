# Production Guide: Training with Real Data

This guide shows how to adapt the example code to train with real, large-scale datasets.

---

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Creating Real Datasets](#creating-real-datasets)
3. [Efficient Data Loading](#efficient-data-loading)
4. [Training at Scale](#training-at-scale)
5. [Distributed Training](#distributed-training)
6. [Monitoring & Logging](#monitoring--logging)
7. [Complete Example](#complete-example)

---

## Data Preparation

### 1. Organize Your Data

Structure your data directory:

```
/data/
├── images/
│   ├── train/
│   │   ├── img_000001.jpg
│   │   ├── img_000002.jpg
│   │   └── ...
│   └── val/
│       ├── img_000001.jpg
│       └── ...
├── timeseries/
│   ├── train/
│   │   ├── ts_000001.npy
│   │   └── ...
│   └── val/
├── tabular/
│   ├── train_features.csv
│   └── val_features.csv
├── scalars/
│   ├── train_measurements.csv
│   └── val_measurements.csv
└── metadata/
    ├── train_manifest.json  # Links samples across modalities
    └── val_manifest.json
```

### 2. Create Manifest File

A manifest file links all modalities for each sample:

```json
{
    "samples": [
        {
            "id": "sample_000001",
            "image": "images/train/img_000001.jpg",
            "timeseries": "timeseries/train/ts_000001.npy",
            "tabular_row": 0,
            "scalar_row": 0
        },
        {
            "id": "sample_000002",
            "image": "images/train/img_000002.jpg",
            "timeseries": "timeseries/train/ts_000002.npy",
            "tabular_row": 1,
            "scalar_row": 1
        }
    ]
}
```

---

## Creating Real Datasets

### Example: Medical Imaging Dataset

```python
"""
real_dataset.py - Production dataset for multimodal training
"""

import torch
from torch.utils.data import Dataset
import json
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path

from step1_define_modalities import MyImage, MyTimeSeries, MyScalar, MyTabular


class ProductionMultimodalDataset(Dataset):
    """
    Production dataset that loads real multimodal data.

    Supports:
    - Memory-mapped files for large data
    - Lazy loading
    - Data augmentation
    - Missing modality handling
    """

    def __init__(
        self,
        data_root: str,
        manifest_path: str,
        split: str = 'train',
        image_size: int = 224,
        augment: bool = True,
        cache_images: bool = False
    ):
        """
        Args:
            data_root: Root directory containing all data
            manifest_path: Path to manifest JSON file
            split: 'train' or 'val'
            image_size: Target image size
            augment: Whether to apply data augmentation
            cache_images: Cache images in RAM (only for small datasets)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.cache_images = cache_images

        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        self.samples = manifest['samples']

        # Load tabular data (entire dataframe)
        tabular_path = self.data_root / "tabular" / f"{split}_features.csv"
        self.tabular_df = pd.read_csv(tabular_path)

        # Load scalar data
        scalar_path = self.data_root / "scalars" / f"{split}_measurements.csv"
        self.scalar_df = pd.read_csv(scalar_path)

        # Image cache
        self.image_cache = {} if cache_images else None

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            # Load image
            image = self._load_image(sample['image'])

            # Load time series
            timeseries = self._load_timeseries(sample['timeseries'])

            # Load tabular features
            tabular = self._load_tabular(sample['tabular_row'])

            # Load scalar
            scalar = self._load_scalar(sample['scalar_row'])

            return {
                'image': image,
                'timeseries': timeseries,
                'scalar': scalar,
                'tabular': tabular,
                'sample_id': sample['id']
            }

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a random sample instead of crashing
            return self.__getitem__((idx + 1) % len(self))

    def _load_image(self, rel_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        # Check cache
        if self.image_cache is not None and rel_path in self.image_cache:
            img = self.image_cache[rel_path]
        else:
            img_path = self.data_root / rel_path
            img = Image.open(img_path).convert('RGB')

            # Cache if enabled
            if self.image_cache is not None:
                self.image_cache[rel_path] = img

        # Resize
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert to tensor
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        # Augmentation
        if self.augment and self.split == 'train':
            img = self._augment_image(img)

        return img

    def _augment_image(self, img: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation"""
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            img = torch.flip(img, dims=[2])

        # Random brightness/contrast (simple version)
        if torch.rand(1) > 0.5:
            brightness = 0.8 + torch.rand(1) * 0.4  # [0.8, 1.2]
            img = torch.clamp(img * brightness, 0, 1)

        return img

    def _load_timeseries(self, rel_path: str) -> dict:
        """Load time series from .npy file"""
        ts_path = self.data_root / rel_path

        # Load numpy array
        data = np.load(ts_path)

        # Assume format: [timesteps, features] or just [timesteps]
        if data.ndim == 1:
            values = torch.from_numpy(data).float()
        else:
            values = torch.from_numpy(data[:, 0]).float()  # Take first feature

        timestamps = torch.arange(len(values)).float()
        mask = torch.ones_like(values, dtype=torch.bool)

        return {
            'values': values,
            'timestamps': timestamps,
            'mask': mask
        }

    def _load_tabular(self, row_idx: int) -> torch.Tensor:
        """Load tabular features from dataframe"""
        row = self.tabular_df.iloc[row_idx]

        # Convert to tensor (exclude ID columns if present)
        feature_cols = [col for col in row.index if col not in ['id', 'sample_id']]
        features = torch.tensor(row[feature_cols].values, dtype=torch.float32)

        return features

    def _load_scalar(self, row_idx: int) -> torch.Tensor:
        """Load scalar measurement"""
        row = self.scalar_df.iloc[row_idx]

        # Assume column named 'measurement' or 'value'
        value_col = 'measurement' if 'measurement' in row.index else 'value'
        value = torch.tensor([row[value_col]], dtype=torch.float32)

        return value


def collate_production_multimodal(batch):
    """Collate function for production dataset"""
    from step1_define_modalities import MyImage, MyTimeSeries, MyScalar, MyTabular

    images = []
    timeseries_list = []
    scalars = []
    tabulars = []
    sample_ids = []

    for sample in batch:
        # Image
        images.append(sample['image'].unsqueeze(0))  # [1, C, H, W]

        # TimeSeries
        timeseries_list.append({
            'values': sample['timeseries']['values'].unsqueeze(0),
            'timestamps': sample['timeseries']['timestamps'],
            'mask': sample['timeseries']['mask'].unsqueeze(0)
        })

        # Scalar
        scalars.append(sample['scalar'].unsqueeze(0))  # [1, 1]

        # Tabular
        tabulars.append(sample['tabular'].unsqueeze(0))  # [1, num_features]

        # Sample ID
        sample_ids.append(sample['sample_id'])

    # Create modality objects
    image_modalities = [MyImage(pixels=img, metadata={'id': sid}) for img, sid in zip(images, sample_ids)]

    timeseries_modalities = [
        MyTimeSeries(
            values=ts['values'],
            timestamps=ts['timestamps'],
            mask=ts['mask']
        ) for ts in timeseries_list
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
        'tabular': tabular_modalities,
        'sample_ids': sample_ids
    }
```

---

## Efficient Data Loading

### Use DataLoader with Multiple Workers

```python
from torch.utils.data import DataLoader

# Create dataset
train_dataset = ProductionMultimodalDataset(
    data_root='/data/my_dataset',
    manifest_path='/data/my_dataset/metadata/train_manifest.json',
    split='train',
    augment=True,
    cache_images=False  # Set to True only for small datasets
)

# DataLoader with multiple workers for parallel loading
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    prefetch_factor=2,  # Prefetch batches
    persistent_workers=True,  # Keep workers alive between epochs
    collate_fn=collate_production_multimodal
)
```

### Memory-Mapped Files for Large Data

For very large datasets that don't fit in RAM:

```python
import numpy as np

class MemoryMappedTimeSeriesDataset(Dataset):
    """Use memory-mapped files for huge time series data"""

    def __init__(self, data_path: str, index_path: str):
        # Memory-mapped array
        self.data = np.load(data_path, mmap_mode='r')  # Read-only, no copy to RAM

        # Index file tells us where each sample starts
        with open(index_path, 'r') as f:
            self.index = json.load(f)

    def __getitem__(self, idx):
        start, end = self.index[idx]
        # Only loads this slice into RAM
        timeseries = torch.from_numpy(self.data[start:end].copy())
        return timeseries
```

---

## Training at Scale

### Complete Training Script

```python
"""
train_production.py - Production training script
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb  # For experiment tracking
from tqdm import tqdm
import argparse
from pathlib import Path

from step4_codec_manager import CodecManager
from step6_train_transformer import MultimodalTransformer, apply_masking
from real_dataset import ProductionMultimodalDataset, collate_production_multimodal


def train_production(
    data_root: str,
    output_dir: str,
    batch_size: int = 32,
    num_epochs: int = 100,
    lr: float = 1e-4,
    num_workers: int = 8,
    device: str = 'cuda',
    use_wandb: bool = True
):
    """
    Production training script.

    Args:
        data_root: Root directory with data
        output_dir: Where to save checkpoints
        batch_size: Training batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        num_workers: DataLoader workers
        device: 'cuda' or 'cpu'
        use_wandb: Log to Weights & Biases
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    if use_wandb:
        wandb.init(
            project="multimodal-transformer",
            config={
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "lr": lr,
                "device": device
            }
        )

    # ========================================================================
    # 1. Load Pre-trained Codecs
    # ========================================================================
    print("Loading pre-trained codecs...")
    codec_manager = CodecManager(device=device)

    # Load codec weights from Step 5
    # In production, you'd load from saved checkpoints:
    # codec_manager._load_codec(MyImage).load_state_dict(torch.load('checkpoints/image_codec.pt'))
    # etc.

    # ========================================================================
    # 2. Create Datasets
    # ========================================================================
    print("Creating datasets...")

    train_dataset = ProductionMultimodalDataset(
        data_root=data_root,
        manifest_path=f"{data_root}/metadata/train_manifest.json",
        split='train',
        augment=True
    )

    val_dataset = ProductionMultimodalDataset(
        data_root=data_root,
        manifest_path=f"{data_root}/metadata/val_manifest.json",
        split='val',
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=collate_production_multimodal
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_production_multimodal
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    # ========================================================================
    # 3. Create Model
    # ========================================================================
    print("Creating model...")

    vocab_sizes = {
        "tok_my_image": 10000,
        "tok_timeseries": 512,
        "tok_scalar": 256,
        "tok_tabular": 256,
    }

    num_tokens = {
        "tok_my_image": 784,
        "tok_timeseries": 8000,
        "tok_scalar": 1,
        "tok_tabular": 64,
    }

    model = MultimodalTransformer(
        vocab_sizes=vocab_sizes,
        num_tokens=num_tokens,
        d_model=512,  # Larger for production
        nhead=8,
        num_layers=12,  # Deeper for production
        dim_feedforward=2048,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ========================================================================
    # 4. Setup Training
    # ========================================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.05,
        betas=(0.9, 0.95)
    )

    # Cosine schedule with warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))

    # Mixed precision training for speed
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    # ========================================================================
    # 5. Training Loop
    # ========================================================================

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("=" * 80)

        # Train
        model.train()
        train_loss = 0
        train_acc = 0

        for batch in tqdm(train_loader, desc="Training"):
            # Tokenize with frozen codecs
            all_tokens = {}
            with torch.no_grad():
                for modality_type, modality_list in batch.items():
                    if modality_type == 'sample_ids':
                        continue

                    # Concatenate modalities
                    if modality_type == 'image':
                        pixels = torch.cat([m.pixels for m in modality_list], dim=0)
                        from step1_define_modalities import MyImage
                        batched = MyImage(pixels=pixels, metadata={})
                    elif modality_type == 'timeseries':
                        values = torch.cat([m.values for m in modality_list], dim=0)
                        from step1_define_modalities import MyTimeSeries
                        batched = MyTimeSeries(
                            values=values,
                            timestamps=modality_list[0].timestamps,
                            mask=torch.cat([m.mask for m in modality_list], dim=0)
                        )
                    elif modality_type == 'scalar':
                        values = torch.cat([m.value for m in modality_list], dim=0)
                        from step1_define_modalities import MyScalar
                        batched = MyScalar(value=values, name="measurement")
                    elif modality_type == 'tabular':
                        features = torch.cat([m.features for m in modality_list], dim=0)
                        from step1_define_modalities import MyTabular
                        batched = MyTabular(features=features, feature_names=modality_list[0].feature_names)

                    tokens = codec_manager.encode(batched)
                    all_tokens.update(tokens)

            all_tokens = {k: v.to(device) for k, v in all_tokens.items()}

            # Mask modalities
            _, masked = apply_masking(all_tokens, mask_ratio=0.3)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast() if scaler else torch.enable_grad():
                predictions = model(all_tokens, masked)

                # Compute loss
                loss = 0
                acc = 0
                n = 0

                for key in masked:
                    if key in predictions:
                        pred = predictions[key]
                        target = all_tokens[key]

                        pred_flat = pred.reshape(-1, pred.size(-1))
                        target_flat = target.reshape(-1).long()

                        loss += torch.nn.functional.cross_entropy(pred_flat, target_flat)

                        acc += (pred.argmax(-1) == target.long()).float().mean()
                        n += 1

                if n > 0:
                    loss = loss / n
                    acc = acc / n

            # Backward
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            train_loss += loss.item()
            train_acc += acc.item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Same as training but no gradients
                all_tokens = {}
                for modality_type, modality_list in batch.items():
                    if modality_type == 'sample_ids':
                        continue

                    if modality_type == 'image':
                        pixels = torch.cat([m.pixels for m in modality_list], dim=0)
                        from step1_define_modalities import MyImage
                        batched = MyImage(pixels=pixels, metadata={})
                    elif modality_type == 'timeseries':
                        values = torch.cat([m.values for m in modality_list], dim=0)
                        from step1_define_modalities import MyTimeSeries
                        batched = MyTimeSeries(
                            values=values,
                            timestamps=modality_list[0].timestamps,
                            mask=torch.cat([m.mask for m in modality_list], dim=0)
                        )
                    elif modality_type == 'scalar':
                        values = torch.cat([m.value for m in modality_list], dim=0)
                        from step1_define_modalities import MyScalar
                        batched = MyScalar(value=values, name="measurement")
                    elif modality_type == 'tabular':
                        features = torch.cat([m.features for m in modality_list], dim=0)
                        from step1_define_modalities import MyTabular
                        batched = MyTabular(features=features, feature_names=modality_list[0].feature_names)

                    tokens = codec_manager.encode(batched)
                    all_tokens.update(tokens)

                all_tokens = {k: v.to(device) for k, v in all_tokens.items()}

                _, masked = apply_masking(all_tokens, mask_ratio=0.3)
                predictions = model(all_tokens, masked)

                loss = 0
                acc = 0
                n = 0

                for key in masked:
                    if key in predictions:
                        pred = predictions[key]
                        target = all_tokens[key]

                        pred_flat = pred.reshape(-1, pred.size(-1))
                        target_flat = target.reshape(-1).long()

                        loss += torch.nn.functional.cross_entropy(pred_flat, target_flat)
                        acc += (pred.argmax(-1) == target.long()).float().mean()
                        n += 1

                if n > 0:
                    val_loss += (loss / n).item()
                    val_acc += (acc / n).item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Log to W&B
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': scheduler.get_last_lr()[0]
            })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, output_dir / 'best_model.pt')
            print(f"✓ Saved best model (val_loss={val_loss:.4f})")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'checkpoint_epoch_{epoch + 1}.pt')

    print("\n✓ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Root directory with data')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_wandb', action='store_true')

    args = parser.parse_args()

    train_production(
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        use_wandb=not args.no_wandb
    )
```

---

## Distributed Training

For training on multiple GPUs:

```python
"""
train_distributed.py - Multi-GPU training with DDP
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def setup_ddp(rank, world_size):
    """Initialize distributed training"""
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up distributed training"""
    dist.destroy_process_group()


def train_ddp(rank, world_size, args):
    """Training function for each GPU"""
    # Setup
    setup_ddp(rank, world_size)

    # Create model
    model = MultimodalTransformer(...).to(rank)
    model = DDP(model, device_ids=[rank])

    # Create dataset with DistributedSampler
    train_dataset = ProductionMultimodalDataset(...)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,  # Use sampler instead of shuffle
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Training loop
    for epoch in range(args.num_epochs):
        # Set epoch for proper shuffling
        train_sampler.set_epoch(epoch)

        for batch in train_loader:
            # Training code...
            pass

    cleanup_ddp()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    # Spawn processes for each GPU
    mp.spawn(
        train_ddp,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )
```

**Run with:**
```bash
# Single machine, 4 GPUs
python train_distributed.py --world_size 4

# Multiple machines (use torchrun)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr="192.168.1.1" --master_port=12345 \
    train_distributed.py
```

---

## Monitoring & Logging

### 1. Weights & Biases Integration

```python
import wandb

# Initialize
wandb.init(project="multimodal-transformer", name="experiment_1")

# Log metrics
wandb.log({
    'train_loss': loss,
    'val_loss': val_loss,
    'learning_rate': lr
})

# Log images
wandb.log({"reconstructed_images": [wandb.Image(img) for img in images]})

# Log model
wandb.watch(model)
```

### 2. TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

# Log scalars
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)

# Log histograms
writer.add_histogram('predictions', predictions, epoch)

# Log images
writer.add_images('reconstructions', images, epoch)
```

---

## Complete Example

### Directory Structure

```
project/
├── step1_define_modalities.py
├── step2_define_quantizers.py
├── step3_implement_codecs.py
├── step4_codec_manager.py
├── step5_train_codecs.py
├── step6_train_transformer.py
├── real_dataset.py              # NEW
├── train_production.py           # NEW
├── train_distributed.py          # NEW
├── data/
│   ├── images/
│   ├── timeseries/
│   ├── tabular/
│   ├── scalars/
│   └── metadata/
│       ├── train_manifest.json
│       └── val_manifest.json
└── outputs/
    └── checkpoints/
```

### Running Production Training

```bash
# 1. Prepare your data
python prepare_data.py \
    --input_dir /raw_data \
    --output_dir /data/my_dataset

# 2. Train codecs (Step 5)
python step5_train_codecs.py \
    --data_root /data/my_dataset \
    --output_dir ./checkpoints/codecs \
    --num_epochs 100 \
    --batch_size 64

# 3. Train transformer (Step 6 - Production)
python train_production.py \
    --data_root /data/my_dataset \
    --output_dir ./outputs \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 1e-4 \
    --num_workers 8 \
    --device cuda

# 4. Distributed training (multi-GPU)
torchrun --nproc_per_node=4 train_distributed.py \
    --data_root /data/my_dataset \
    --output_dir ./outputs \
    --batch_size 32
```

---

## Performance Tips

### 1. Data Loading
- Use `num_workers=8` or more for parallel loading
- Enable `pin_memory=True` for faster GPU transfer
- Use `persistent_workers=True` to avoid worker restarts
- Consider `prefetch_factor=2` for prefetching

### 2. Training Speed
- Use mixed precision training (`torch.cuda.amp`)
- Gradient accumulation for larger effective batch sizes
- Profile with `torch.profiler` to find bottlenecks

### 3. Memory Optimization
- Gradient checkpointing for very deep models
- Use `torch.utils.checkpoint` for activation checkpointing
- Clear cache periodically: `torch.cuda.empty_cache()`

### 4. Data Preprocessing
- Precompute and cache augmentations
- Use memory-mapped files for huge datasets
- Store data in efficient formats (HDF5, TFRecord, WebDataset)

---

## Troubleshooting

### Out of Memory
```python
# Reduce batch size
batch_size = 16  # instead of 32

# Gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Slow Data Loading
```python
# Profile data loading
import time

start = time.time()
for batch in train_loader:
    print(f"Batch load time: {time.time() - start:.3f}s")
    start = time.time()
    # Training code...
```

### Debugging
```python
# Set environment variables
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous CUDA for debugging
os.environ['TORCH_USE_CUDA_DSA'] = '1'    # Device-side assertions

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)
```

---

## Next Steps

1. **Prepare your data** using the manifest structure
2. **Adapt `real_dataset.py`** to your specific data format
3. **Train codecs** on your data (Step 5)
4. **Train transformer** with `train_production.py`
5. **Monitor** training with W&B or TensorBoard
6. **Scale up** with distributed training when ready

Good luck with your production training! 🚀
