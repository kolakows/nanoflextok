"""
Precompute latents.
Run once, then use cached features for training.
"""
import os
import torch
from tqdm import tqdm
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from flux2_tiny_autoencoder import Flux2TinyAutoEncoder

# ============ Configuration ============
DINO_MODEL_NAME = "facebook/dinov2-large"  # Large model (1024 dim)
AE_MODEL_NAME = "fal/FLUX.2-Tiny-AutoEncoder"
DATASET_PATH = "/home/nekoneko/diffusion/data/ANIME"
AE_IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 8
DEVICE = "cuda"
SAVE_DIR = Path("ae_dino_gochiusa_cache")
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    # Load DINOv2 model and processor
    print(f"Loading DINOv2 model: {DINO_MODEL_NAME}")
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
    dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME).eval().to(DEVICE)
    
    # Load vae
    print(f"Loading Autoencoder: {AE_MODEL_NAME}")
    tiny_ae = Flux2TinyAutoEncoder.from_pretrained(
        "fal/FLUX.2-Tiny-AutoEncoder",
    ).eval().to(device=DEVICE, dtype=torch.bfloat16)

    ae_transform = transforms.Compose([
        transforms.Resize(AE_IMAGE_SIZE),
        transforms.CenterCrop(AE_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Compile for faster inference
    dino_model = torch.compile(dino_model)
    @torch.compile
    def ae_encode(x):
        return tiny_ae.encode(x, return_dict=False)
    
    def collate_fn(batch):
        images, labels = zip(*batch)
        ae_images = torch.stack([ae_transform(img.convert("RGB") ) for img in images])
        dino_inputs = processor(images=images, return_tensors="pt")
        
        return ae_images, dino_inputs, torch.tensor(labels)
    
        
    dataset = ImageFolder(DATASET_PATH)
    
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Keep order for reproducibility
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    
    all_latents = []
    # all_dino_features = []
    all_labels = []
    
    with torch.inference_mode():
        for ae_images, dino_inputs, labels in tqdm(loader, desc="Encoding"):
            # VAE encoding
            ae_images = ae_images.to(DEVICE, dtype=tiny_ae.dtype)
            latents = ae_encode(ae_images)
            all_latents.append(latents.cpu())
            
            # # DINOv2 encoding
            # dino_inputs = {k: v.to(DEVICE) for k, v in dino_inputs.items()}
            # dino_outputs = dino_model(**dino_inputs)
            # patch_features = dino_outputs.last_hidden_state[:, 1:, :]  # Skip CLS token
            # all_dino_features.append(patch_features.cpu().to(torch.bfloat16))
            
            all_labels.append(labels)

    # Concatenate and save
    all_latents = torch.cat(all_latents, dim=0)
    # all_dino_features = torch.cat(all_dino_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    torch.save({
        'latents': all_latents,
        # 'dino_features': all_dino_features,
        'labels': all_labels,
        'ae_model_name': AE_MODEL_NAME,
        # 'dino_model_name': DINO_MODEL_NAME,
    }, SAVE_DIR / "latents.pt")
    
    print(f"Saved {len(all_latents)} samples to {SAVE_DIR / 'latents.pt'}")
    print(f"Latent shape: {all_latents.shape}")
    # print(f"DINOv2 features shape: {all_dino_features.shape} ({all_dino_features.dtype})")
    print(f"File size: {(SAVE_DIR / 'latents.pt').stat().st_size / 1e9:.2f} GB")

if __name__ == "__main__":
    main()