import os
import wandb
import webdataset as wds
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from tqdm import tqdm

from dataclasses import fields
from model.flextok import FlexTokCc3mConfig, FlexTok
from utils.utils import warp_time, image_to_patches
from utils.cc3m_logging import log_reconstructed_images, log_test_mse

from diffusers.models import AutoencoderKL

# ============ Configuration ============
USE_C16_VAE = False
TORCH_COMPILE = True

# ============ Updated hyperparameters for 224x224 images ============
# input_h = 224
# input_w = 224
# input_channels = 3  # RGB instead of grayscale
# patch_size = 16     # Common for ViT-style models on 224x224
# patch_dim = patch_size ** 2 * input_channels  # 16*16*3 = 768
# num_patches = (input_h // patch_size) * (input_w // patch_size)  # 14*14 = 196

# vae latents
input_h = 32
input_w = 32

if USE_C16_VAE:
    input_channels = 16
    vae_model_name = 'EPFL-VILAB/flextok_vae_c16'
else:
    input_channels = 4
    vae_model_name = 'EPFL-VILAB/flextok_vae_c4'
patch_size = 2
patch_dim = patch_size ** 2 * input_channels  # 16*16*3 = 768
num_patches = (input_h // patch_size) * (input_w // patch_size)  # 14*14 = 196

d: int = 768                 # Increased model dimension
cond_d: int = 128            # Increased conditioning dimension
nh: int = 12                  # More attention heads
n_layers: int = 12            # More layers
n_registers: int = 256
register_subset_lengths = (1, 2, 4, 8, 16, 32, 64, 128, 256)
fsq_n_levels: int = 5
fsq_l: int = 8
time_dim: int = 128
device: str = "cuda"
lr = 3e-4
betas = (0.9, 0.95)

validation_sample_size = 5
batch_size = 4
cc3m_train_total = 2_905_954

assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype =  torch.bfloat16
ctx = torch.amp.autocast(device_type=device, dtype=dtype)

# HuggingFace token (set this or use environment variable)
hf_token = os.environ.get("HF_TOKEN")

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, tuple))]
user_config = {k: globals()[k] for k in config_keys}

valid_fields = {f.name for f in fields(FlexTokCc3mConfig)}
filtered_cfg_dict = {k: v for k, v in user_config.items() if k in valid_fields}
cfg = FlexTokCc3mConfig(**filtered_cfg_dict)

wandb_run = wandb.init(project="nanoflextok-cc3m", config=user_config, dir='wandb_logs', mode="offline")

# ============ CC3M Dataset Setup ============
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def make_dataloader(shards, batch_size, shuffle):
    url = f"https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main/{shards}.tar"
    url = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}'"
    
    dataset = (
        wds.WebDataset(url, shardshuffle=shuffle)
        .decode("pil")
        .to_tuple("jpg", "txt")
        .map_tuple(transform, lambda x: x)
    )
    
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)


# TODO: benchmark if data fetching keeps up with the training loop
train_loader = make_dataloader("cc3m-train-{0000..0575}", batch_size=batch_size, shuffle=True)
test_loader = make_dataloader("cc3m-validation-{0000..0016}", batch_size=batch_size, shuffle=False)


vae = AutoencoderKL.from_pretrained(
    vae_model_name, low_cpu_mem_usage=False
).eval().to(device)

def encode_fn(vae, x):
    return vae.encode(x).latent_dist.sample()

def train(model, dataloader, lr, device='cuda'):
    model = model.to(device)
    model.train()
    # TODO: exclude embedding layers from weight decay, also learned sinusidoal?
    optimizer = AdamW(model.parameters(), lr=lr, betas=betas)
    
    # log_test_mse(model, test_loader, validation_sample_size, cfg, wandb_run)
    # log_reconstructed_images(model, test_loader, cfg, wandb_run)
    
    running_loss = []
    pbar = tqdm(dataloader, total=cc3m_train_total//batch_size)
    for step, (x, captions) in enumerate(pbar):
        x = x.to(device)
        
        with torch.no_grad():
            latents = encode_fn(vae, x)
            x = latents

        x_patches = image_to_patches(x, cfg.patch_size)
        
        t = torch.rand(x.shape[0], device=device)
        t = warp_time(t)
        
        with ctx:
            flow, noise = model(x_patches, t)
            target_flow = noise - x_patches
            loss = ((flow - target_flow) ** 2).mean()
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            running_loss.append(loss.item())
            pbar.set_postfix({'loss': f'{running_loss[-1]:.4f}'})
            wandb_run.log({"loss": loss.item()})
 
        if step % 100 == 0:
            # TODO: add model checkpointing
            continue
            log_test_mse(model, test_loader, validation_sample_size, cfg, wandb_run)
            log_reconstructed_images(model, test_loader, cfg, wandb_run)

# ============ Main ============
if __name__ == "__main__":
    model = FlexTok(cfg)
    if TORCH_COMPILE:
        model = torch.compile(model)
        encode_fn = torch.compile(encode_fn)
    print(f"Trainable parameter count: {model.parameter_count()}")
    model.to(device)
    train(model, train_loader, lr=lr)
    #torch.save(model.state_dict(), "cc3m_flow.pt")