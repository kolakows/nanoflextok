"""
Precompute DINOv2 patch features for ImageNet-100 dataset.
Run once, then use cached features for training.
"""

import torch
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from diffusers.models import AutoencoderKL

# ============ Configuration ============
DINO_MODEL_NAME = "facebook/dinov2-large"  # Large model (1024 dim)
VAE_MODEL_NAME = "EPFL-VILAB/flextok_vae_c8"
VAE_IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 8
DEVICE = "cuda"
CACHE_DIR = Path("vae_c8_dino_cache")

def main():
    # Load DINOv2 model and processor
    print(f"Loading DINOv2 model: {DINO_MODEL_NAME}")
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
    dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME).eval().to(DEVICE)
    
    # Load vae
    print(f"Loading VAE: {VAE_MODEL_NAME}")
    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL_NAME, low_cpu_mem_usage=False
    ).eval().to(DEVICE)

    vae_transform = transforms.Compose([
        transforms.Resize(VAE_IMAGE_SIZE),
        transforms.CenterCrop(VAE_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Compile for faster inference
    dino_model = torch.compile(dino_model)
    vae_encode = torch.compile(vae.encode)
    
    def collate_fn(batch):
        images = [item["image"].convert("RGB") for item in batch]
        labels = [item["label"] for item in batch]

        vae_images = torch.stack([vae_transform(img) for img in images])
        dino_inputs = processor(images=images, return_tensors="pt")
        
        return vae_images, dino_inputs, torch.tensor(labels)
    
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
        all_dino_features = []
        all_labels = []
        
        with torch.no_grad():
            for vae_images, dino_inputs, labels in tqdm(loader, desc=f"Encoding {split}"):
                # VAE encoding
                vae_images = vae_images.to(DEVICE)
                latents = vae_encode(vae_images).latent_dist.mean
                all_latents.append(latents.cpu())
                
                # DINOv2 encoding
                dino_inputs = {k: v.to(DEVICE) for k, v in dino_inputs.items()}
                dino_outputs = dino_model(**dino_inputs)
                patch_features = dino_outputs.last_hidden_state[:, 1:, :]  # Skip CLS token
                all_dino_features.append(patch_features.cpu().to(torch.bfloat16))
                
                all_labels.append(labels)

        # Concatenate and save
        all_latents = torch.cat(all_latents, dim=0)
        all_dino_features = torch.cat(all_dino_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        torch.save({
            'latents': all_latents,
            'dino_features': all_dino_features,
            'labels': all_labels,
            'vae_model_name': VAE_MODEL_NAME,
            'dino_model_name': DINO_MODEL_NAME,
        }, save_dir / "latents.pt")
        
        print(f"Saved {len(all_latents)} samples to {save_dir / 'latents.pt'}")
        print(f"Latent shape: {all_latents.shape}")
        print(f"DINOv2 features shape: {all_dino_features.shape} ({all_dino_features.dtype})")
        print(f"File size: {(save_dir / 'latents.pt').stat().st_size / 1e9:.2f} GB")

if __name__ == "__main__":
    main()