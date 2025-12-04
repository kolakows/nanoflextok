import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms


from torch.utils.data import Dataset
from pathlib import Path


class CachedLatentDataset(Dataset):
    """Dataset that loads precomputed VAE latents."""
    
    def __init__(self, cache_dir: str, split: str = "train"):
        cache_path = Path(cache_dir) / split / "latents.pt"
        
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cached latents not found at {cache_path}. "
                f"Run precompute_latents.py first."
            )
        
        print(f"Loading cached latents from {cache_path}...")
        data = torch.load(cache_path, weights_only=True)
        self.latents = data['latents']
        self.dino_features = data['dino_features']
        self.labels = data['labels']
        
        print(f"Loaded {len(self.latents)} latents, shape: {self.latents.shape}")
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        latent = self.latents[idx]
        dino_feature = self.dino_features[idx]
        label = self.labels[idx]
        
        return latent, dino_feature, label


def get_cached_dataloaders(cache_dir, train_epochs, batch_size, num_workers):
    """
    Drop-in replacement for get_imagenet_dataloaders() using cached latents.
    
    Returns:
        train_loader: Generator that yields (latents, labels) batches
        test_loader: DataLoader for validation
        total_train_steps: Total number of training steps
    """
    train_dataset = CachedLatentDataset(cache_dir, split="train")
    test_dataset = CachedLatentDataset(cache_dir, split="validation")
    
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


def get_imagenet_dataloaders(train_epochs, batch_size, num_workers, image_size=256):
    train_dataset = load_dataset("clane9/imagenet-100", split="train")
    test_dataset = load_dataset("clane9/imagenet-100", split="validation")

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def collate_fn(batch):
        images = [transform(item["image"].convert("RGB")) for item in batch]
        labels = [item["label"] for item in batch]
        return torch.stack(images), torch.tensor(labels)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers // 2,
        pin_memory=True
    )

    total_train_steps = len(train_dataset) // batch_size * train_epochs

    def train_loader_gen():
        i = 0
        while True:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            )
            for batch in train_loader:
                if i >= total_train_steps:
                    return
                yield batch
                i += 1

    return train_loader_gen(), test_loader, total_train_steps