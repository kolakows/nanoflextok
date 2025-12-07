import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path


def load_and_split_cached_latents(cache_dir: str, train_split: float = 0.9):
    """
    Load cached latents and split into train/test sets.

    Args:
        cache_dir: Directory containing the cached latents.pt file
        train_split: Fraction of data to use for training (default: 0.9)

    Returns:
        Tuple of (train_latents, train_labels, test_latents, test_labels)
    """
    cache_path = Path(cache_dir) / "latents.pt"

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cached latents not found at {cache_path}. "
            f"Run precompute_latents_anime.py first."
        )

    print(f"Loading cached latents from {cache_path}...")
    data = torch.load(cache_path, weights_only=True)

    latents = data['latents'].to(torch.float)
    labels = data['labels']

    # Create train/test split
    total_samples = len(latents)
    train_size = int(total_samples * train_split)

    # Use a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_samples, generator=generator)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_latents = latents[train_indices]
    train_labels = labels[train_indices]
    test_latents = latents[test_indices]
    test_labels = labels[test_indices]

    print(f"Split into {len(train_latents)} train and {len(test_latents)} test samples")
    print(f"Latent shape: {latents.shape}")

    return train_latents, train_labels, test_latents, test_labels


class CachedGochiusaDataset(Dataset):
    """Dataset that holds precomputed latents."""

    def __init__(self, latents: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            latents: Precomputed latent tensors
            labels: Corresponding labels
        """
        self.latents = latents
        self.labels = labels

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        latent = self.latents[idx]
        label = self.labels[idx]

        # TODO: add dino features
        return latent, 0, label


def get_cached_gochiusa_dataloaders(cache_dir, train_epochs, batch_size, num_workers, train_split=0.9):
    """
    Get dataloaders for cached Gochiusa latents with train/test split.

    Args:
        cache_dir: Directory containing the cached latents
        train_epochs: Number of training epochs
        batch_size: Batch size for training
        num_workers: Number of dataloader workers
        train_split: Fraction of data to use for training (default: 0.9)

    Returns:
        train_loader: Generator that yields (latents, labels) batches
        test_loader: DataLoader for validation
        total_train_steps: Total number of training steps
    """
    # Load and split data once
    train_latents, train_labels, test_latents, test_labels = load_and_split_cached_latents(
        cache_dir, train_split=train_split
    )

    # Create datasets
    train_dataset = CachedGochiusaDataset(train_latents, train_labels)
    test_dataset = CachedGochiusaDataset(test_latents, test_labels)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2,
        pin_memory=True,
    )

    total_train_steps = len(train_dataset) // batch_size * train_epochs

    def train_loader_gen():
        i = 0
        while True:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
            for batch in train_loader:
                if i >= total_train_steps:
                    return
                yield batch
                i += 1

    return train_loader_gen(), test_loader, total_train_steps
