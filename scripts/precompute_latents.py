"""
Precompute VAE latents for ImageNet-100 dataset.
Run once, then use CachedLatentDataset for training.

Usage:
    python precompute_latents.py
"""

import os
import torch
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers.models import AutoencoderKL

# ============ Configuration ============
USE_C16_VAE = False
BATCH_SIZE = 64  # Can use larger batch for inference
NUM_WORKERS = 8
IMAGE_SIZE = 256
DEVICE = "cuda"

if USE_C16_VAE:
    VAE_MODEL_NAME = 'EPFL-VILAB/flextok_vae_c16'
    CACHE_DIR = Path("latent_cache_c16")
else:
    VAE_MODEL_NAME = 'EPFL-VILAB/flextok_vae_c4'
    CACHE_DIR = Path("latent_cache_c4")


def main():
    # Load VAE
    print(f"Loading VAE: {VAE_MODEL_NAME}")
    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL_NAME, low_cpu_mem_usage=False
    ).eval().to(DEVICE)
    
    # Compile for faster inference
    vae_encode = torch.compile(vae.encode)
    
    # Transform (no random flip - we'll do augmentation at training time on latents)
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    def collate_fn(batch):
        images = [transform(item["image"].convert("RGB")) for item in batch]
        labels = [item["label"] for item in batch]
        return torch.stack(images), torch.tensor(labels)
    
    # Process both splits
    for split in ["train", "validation"]:
        print(f"\nProcessing {split} split...")
        
        dataset = load_dataset("clane9/imagenet-100", split=split)
        
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,  # Keep order for reproducibility
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        
        save_dir = CACHE_DIR / split
        save_dir.mkdir(parents=True, exist_ok=True)
        
        all_latents = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f"Encoding {split}"):
                images = images.to(DEVICE)
                latents = vae_encode(images).latent_dist.mean
                all_latents.append(latents.cpu())
                all_labels.append(labels)
        
        # Concatenate and save as single file (faster loading)
        all_latents = torch.cat(all_latents, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        torch.save({
            'latents': all_latents,
            'labels': all_labels,
        }, save_dir / "latents.pt")
        
        print(f"Saved {len(all_latents)} latents to {save_dir / 'latents.pt'}")
        print(f"Latent shape: {all_latents.shape}")
        print(f"File size: {(save_dir / 'latents.pt').stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()