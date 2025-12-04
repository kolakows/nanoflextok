import os
import wandb
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from dataclasses import fields
from model.flextok import FlexTokCc3mConfig, FlexTok
from utils.utils import warp_time, image_to_patches
from utils.imagenet import get_cached_dataloaders
from utils.logging import log_reconstructed_images, log_test_mse

from diffusers.models import AutoencoderKL

# ============ Configuration ============
TORCH_COMPILE = True
ADAM_FUSED = False
SAVE_CKPT = True

# ============ Hyperparameters ============
# vae latents
input_h = 32
input_w = 32

out_dir = "checkpoints"
os.makedirs(out_dir, exist_ok=True)
input_channels = 8
vae_model_name = "EPFL-VILAB/flextok_vae_c8"
ckpt_base_name = "imagenet_c8_repa"
cache_dir = "vae_c8_dino_cache"
patch_size = 2
patch_dim = patch_size ** 2 * input_channels  # 32
num_patches = (input_h // patch_size) * (input_w // patch_size)  # 16*16 = 256

image_size = 256

d: int = 768                 # Increased model dimension
cond_d: int = 128            # Increased conditioning dimension
nh: int = 12                  # More attention heads
n_layers: int = 12            # More layers
n_registers: int = 256
# register_subset_lengths = (1, 2, 4, 8, 16, 32, 64, 128, 256)
register_subset_lengths = (256,)
fsq_n_levels: int = 5
fsq_l: int = 8
time_dim: int = 128
device: str = "cuda"
lr = 3e-4
betas = (0.9, 0.95)
weight_decay = 0.01

train_epochs = 50
val_img_dec_samples = 8
val_mse_batches = 1
batch_size = 32
dataloader_num_workers = 8

assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype =  torch.bfloat16
ctx = torch.amp.autocast(device_type=device, dtype=dtype)

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, tuple))]
user_config = {k: globals()[k] for k in config_keys}

hf_token = os.environ.get("HF_TOKEN")

valid_fields = {f.name for f in fields(FlexTokCc3mConfig)}
filtered_cfg_dict = {k: v for k, v in user_config.items() if k in valid_fields}
cfg = FlexTokCc3mConfig(**filtered_cfg_dict)

wandb_run = wandb.init(project="nanoflextok-imagenet-100", config=user_config, dir='wandb_logs')

train_loader, test_loader, total_train_steps = get_cached_dataloaders(cache_dir, train_epochs=train_epochs, batch_size=batch_size, num_workers=dataloader_num_workers)
print(f"Total train steps: {total_train_steps}")

vae = AutoencoderKL.from_pretrained(
    vae_model_name, low_cpu_mem_usage=False
).eval().to(device)

def vae_encode_fn(x):
    return vae.encode(x).latent_dist.mean

def vae_decode_fn(latents):
    return vae.decode(latents).sample

def train(model, raw_model, vae_encode_fn, vae_decode_fn, dataloader, lr, device='cuda'):
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.get_parameter_groups(weight_decay), lr=lr, betas=betas, fused=ADAM_FUSED)

    running_loss = []
    pbar = tqdm(dataloader, total=total_train_steps)
    for step, (lat, dino, label) in enumerate(pbar):
        lat = lat.to(device)
        x_patches = image_to_patches(lat, cfg.patch_size)
        
        t = torch.rand(lat.shape[0], device=device)
        t = warp_time(t)
        
        with ctx:
            flow, noise, repa_features = model(x_patches, t)
            target_flow = noise - x_patches
            rf_loss = ((flow - target_flow) ** 2).mean()

            cosine_sim = F.cosine_similarity(repa_features, dino, dim=-1)
            repa_loss = (1 - cosine_sim).mean()

            loss = rf_loss + repa_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            running_loss.append(loss.item())
            pbar.set_postfix({'loss': f'{running_loss[-1]:.4f}', 'rf': f'{rf_loss.item():.4f}', 'repa': f'{repa_loss.item():.4f}', 'iter': step})
            wandb_run.log({
                "loss": loss.item(),
                "rf_loss": rf_loss.item(),
                "repa_loss": repa_loss.item(),
                'iter': step
            })
 
        if step % 5000 == 0:
            log_test_mse(model, vae_encode_fn, test_loader, val_mse_batches, cfg, wandb_run, step)
            log_reconstructed_images(model, vae_encode_fn, vae_decode_fn, test_loader, val_img_dec_samples, cfg, wandb_run, step)

        if step % 10_000 == 0 and SAVE_CKPT:
            checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': filtered_cfg_dict,
                    'step': step
                }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'{ckpt_base_name}_{step}.pt'))

# ============ Main ============
if __name__ == "__main__":
    model = FlexTok(cfg)
    raw_model = model
    if TORCH_COMPILE:
        model = torch.compile(model)
        vae_encode_fn = torch.compile(vae_encode_fn)
        vae_decode_fn = torch.compile(vae_decode_fn)
    print(f"Trainable parameter count: {model.parameter_count()}")
    train(model, raw_model, vae_encode_fn, vae_decode_fn, train_loader, lr=lr)
    if SAVE_CKPT:
        checkpoint = {
            'model': raw_model.state_dict(),
            'step': total_train_steps
        }
        torch.save(checkpoint, os.path.join(out_dir,f"{ckpt_base_name}_{total_train_steps}_final.pt"))