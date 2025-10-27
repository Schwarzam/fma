"""
Galaxy Image Retrieval Example

This shows how to adapt the framework for retrieval tasks:
- Find similar galaxies based on morphology
- Strong gravitational lens retrieval
- Cross-modal retrieval (find images from spectrum/redshift)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import faiss  # For efficient nearest neighbor search

from astronomical_dataset import GalaxyImage
from astronomical_codecs import GalaxyImageCodec


# ============================================================================
# 1. Contrastive Learning for Embeddings
# ============================================================================

class ContrastiveImageEncoder(nn.Module):
    """
    Encoder that produces embeddings optimized for retrieval.

    Uses contrastive learning: similar galaxies should have similar embeddings.
    """

    def __init__(self, base_encoder, embedding_dim: int = 128):
        super().__init__()
        self.base_encoder = base_encoder
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(64 * 6 * 6, 512),  # Flatten encoder output
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 96, 96] images

        Returns:
            embeddings: [B, embedding_dim] L2-normalized embeddings
        """
        # Encode
        z = self.base_encoder(x)  # [B, 64, 6, 6]
        z_flat = z.reshape(z.shape[0], -1)  # [B, 64*6*6]

        # Project
        embedding = self.projection(z_flat)  # [B, embedding_dim]

        # L2 normalize for cosine similarity
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


def contrastive_loss(embeddings1: torch.Tensor, embeddings2: torch.Tensor,
                     labels: torch.Tensor, temperature: float = 0.1):
    """
    InfoNCE / NT-Xent loss for contrastive learning.

    Args:
        embeddings1: [B, D] embeddings from augmented view 1
        embeddings2: [B, D] embeddings from augmented view 2
        labels: [B] class labels (for positive/negative pairs)
        temperature: Temperature for softmax

    Returns:
        loss: Scalar contrastive loss
    """
    batch_size = embeddings1.shape[0]

    # Concatenate embeddings
    embeddings = torch.cat([embeddings1, embeddings2], dim=0)  # [2B, D]

    # Compute similarity matrix
    sim_matrix = torch.mm(embeddings, embeddings.T) / temperature  # [2B, 2B]

    # Create mask for positive pairs
    # Positive pairs: (i, i+B) for each sample
    mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    mask = mask.repeat(2, 2)  # [2B, 2B]
    mask = ~mask  # Negative pairs

    # For each sample, the positive is at position i+B (or i-B)
    labels_expanded = labels.unsqueeze(1).repeat(2, 1)  # [2B, 1]
    label_matrix = (labels_expanded == labels_expanded.T)  # [2B, 2B]

    # Positive pairs: same label AND different augmentation
    positive_mask = label_matrix & mask

    # Compute loss
    exp_sim = torch.exp(sim_matrix)
    sum_exp = (exp_sim * mask.float()).sum(dim=1)  # Sum over negatives

    # Get positive similarities
    positive_sim = (exp_sim * positive_mask.float()).sum(dim=1)

    loss = -torch.log(positive_sim / (sum_exp + 1e-8))
    return loss.mean()


# ============================================================================
# 2. Build Search Index
# ============================================================================

class GalaxyRetrievalSystem:
    """
    Retrieval system for finding similar galaxies.

    Uses FAISS for efficient nearest neighbor search.
    """

    def __init__(self, encoder: nn.Module, embedding_dim: int = 128):
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.index = None
        self.galaxy_metadata = []

    def build_index(self, image_dataset, device='cuda', batch_size=32):
        """
        Build FAISS index from all images in dataset.

        Args:
            image_dataset: Dataset containing galaxy images
        """
        print("Building retrieval index...")
        self.encoder.eval()

        embeddings_list = []
        metadata_list = []

        with torch.no_grad():
            for i in range(0, len(image_dataset), batch_size):
                batch = image_dataset[i:min(i+batch_size, len(image_dataset))]

                # Stack images
                if isinstance(batch, list):
                    images = torch.stack([sample['image'] for sample in batch])
                else:
                    images = batch['image']

                images = images.to(device)

                # Normalize
                images = images * 2.0 - 1.0

                # Get embeddings
                embeddings = self.encoder(images)  # [B, embedding_dim]
                embeddings_list.append(embeddings.cpu().numpy())

                # Store metadata
                for j, sample in enumerate(batch if isinstance(batch, list) else [batch]):
                    metadata_list.append({
                        'index': i + j,
                        'redshift': sample.get('redshift', None),
                        'stellar_mass': sample.get('stellar_mass', None),
                        'sfr': sample.get('sfr', None),
                    })

        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings_list).astype('float32')
        print(f"Total embeddings: {all_embeddings.shape}")

        # Build FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine sim)
        self.index.add(all_embeddings)
        self.galaxy_metadata = metadata_list

        print(f"✓ Index built with {self.index.ntotal} galaxies")

    def search(self, query_image: torch.Tensor, k: int = 10, device='cuda') -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k most similar galaxies.

        Args:
            query_image: [1, 3, 96, 96] or [3, 96, 96] query image
            k: Number of results to return

        Returns:
            similarities: [k] cosine similarities
            indices: [k] indices of most similar galaxies
        """
        if query_image.ndim == 3:
            query_image = query_image.unsqueeze(0)

        self.encoder.eval()
        with torch.no_grad():
            query_image = query_image.to(device) * 2.0 - 1.0
            query_embedding = self.encoder(query_image).cpu().numpy().astype('float32')

        # Search
        similarities, indices = self.index.search(query_embedding, k)

        return similarities[0], indices[0]

    def search_by_properties(self, redshift: float = None, mass_range: Tuple[float, float] = None,
                            k: int = 100) -> List[int]:
        """
        Filter galaxies by physical properties before retrieval.

        Args:
            redshift: Target redshift (±0.1)
            mass_range: (min_mass, max_mass) in log(M☉)
            k: Max number of results

        Returns:
            indices: List of matching galaxy indices
        """
        matches = []

        for idx, meta in enumerate(self.galaxy_metadata):
            # Check redshift
            if redshift is not None:
                if meta['redshift'] is None or abs(meta['redshift'] - redshift) > 0.1:
                    continue

            # Check mass range
            if mass_range is not None:
                if meta['stellar_mass'] is None:
                    continue
                if not (mass_range[0] <= meta['stellar_mass'] <= mass_range[1]):
                    continue

            matches.append(idx)

            if len(matches) >= k:
                break

        return matches


# ============================================================================
# 3. Strong Gravitational Lens Retrieval
# ============================================================================

class StrongLensRetrieval:
    """
    Specialized retrieval for strong gravitational lenses.

    Strong lenses have distinctive features:
    - Einstein rings
    - Multiple lensed images
    - Arcs around massive foreground galaxy
    """

    def __init__(self, encoder: nn.Module):
        self.encoder = encoder
        self.lens_detector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def is_strong_lens(self, image: torch.Tensor, device='cuda') -> float:
        """
        Predict probability that image contains a strong lens.

        Args:
            image: [1, 3, 96, 96] galaxy image

        Returns:
            probability: Float in [0, 1]
        """
        self.encoder.eval()
        self.lens_detector.eval()

        with torch.no_grad():
            image = image.to(device) * 2.0 - 1.0
            embedding = self.encoder(image)
            prob = self.lens_detector(embedding)

        return prob.item()

    def find_strong_lenses(self, dataset, threshold: float = 0.9, device='cuda') -> List[Tuple[int, float]]:
        """
        Scan dataset for likely strong lenses.

        Args:
            dataset: Galaxy image dataset
            threshold: Minimum probability to consider

        Returns:
            candidates: List of (index, probability) for strong lens candidates
        """
        candidates = []

        for i in range(len(dataset)):
            sample = dataset[i]
            image = sample['image'].unsqueeze(0)

            prob = self.is_strong_lens(image, device)

            if prob >= threshold:
                candidates.append((i, prob))

        # Sort by probability
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates


# ============================================================================
# 4. Cross-Modal Retrieval
# ============================================================================

def cross_modal_retrieval_example():
    """
    Find images given spectrum or other modalities.

    Uses the multimodal transformer to bridge modalities.
    """
    print("\nCross-Modal Retrieval Example")
    print("=" * 80)

    # Pseudocode:
    """
    # Given a spectrum, find similar galaxy images

    # 1. Encode spectrum to tokens
    spectrum_tokens = spectrum_codec.encode(query_spectrum)

    # 2. Predict image tokens using transformer
    predicted_image_tokens = transformer(
        input_modalities={'tok_gaia_spectrum': spectrum_tokens},
        predict_modalities=['tok_galaxy_image']
    )

    # 3. Decode to image embedding
    image_embedding = image_codec.encode_to_embedding(predicted_image_tokens)

    # 4. Search in image database
    similar_images = retrieval_system.search_by_embedding(image_embedding, k=10)
    """
    pass


# ============================================================================
# 5. Usage Example
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("GALAXY RETRIEVAL EXAMPLE")
    print("=" * 80)

    # Create encoder (reuse trained image codec)
    from astronomical_codecs import GalaxyImageCodec
    base_codec = GalaxyImageCodec(in_channels=3, embedding_dim=64)

    # Wrap with contrastive encoder
    retrieval_encoder = ContrastiveImageEncoder(
        base_encoder=base_codec.encoder,
        embedding_dim=128
    )

    print(f"Encoder parameters: {sum(p.numel() for p in retrieval_encoder.parameters()):,}")

    # Create retrieval system
    retrieval_system = GalaxyRetrievalSystem(
        encoder=retrieval_encoder,
        embedding_dim=128
    )

    print("\n✓ Retrieval system created!")

    print("\n" + "=" * 80)
    print("TO USE THIS FOR REAL:")
    print("=" * 80)
    print("""
1. TRAIN CONTRASTIVE ENCODER:
   - Use augmented image pairs (rotation, flip, color jitter)
   - Train with contrastive loss
   - Or use SimCLR / MoCo training strategy

2. BUILD INDEX:
   retrieval_system.build_index(your_dataset)

3. SEARCH:
   query_image = dataset[0]['image']
   similarities, indices = retrieval_system.search(query_image, k=10)

4. STRONG LENS DETECTION:
   - Manually label ~1000 lenses vs non-lenses
   - Train lens_detector classifier
   - Scan large survey for candidates

5. CROSS-MODAL:
   - Use trained transformer to bridge modalities
   - Search images by spectrum, redshift, etc.
""")

    print("\nKey Dependencies:")
    print("  pip install faiss-gpu  # or faiss-cpu")
    print("  pip install torch torchvision")
