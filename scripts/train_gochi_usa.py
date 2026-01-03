import os
import wandb
import torch
from torch.optim import AdamW
from tqdm import tqdm

from dataclasses import fields
from model.flextok import FlexTokCc3mConfig, FlexTok
from utils.utils import warp_time, image_to_patches, get_lr_fn
from utils.gochiusa import get_cached_gochiusa_dataloaders
from utils.logging import log_reconstructed_images, log_test_mse, fsq_codebook_usage_stats
from flux2_tiny_autoencoder import Flux2TinyAutoEncoder

# ============ Configuration ============
TORCH_COMPILE = True
ADAM_FUSED = False
SAVE_CKPT = True

# ============ Hyperparameters ============
# vae latents - using FLUX.2-Tiny-AutoEncoder from precompute_latents_anime
input_h = 16
input_w = 16
input_channels = 128  # FLUX.2-Tiny-AutoEncoder outputs [128, 16, 16]
vae_model_name = "fal/FLUX.2-Tiny-AutoEncoder"
ckpt_base_name = "gochiusa_flux2ae"
cache_dir = "ae_dino_gochiusa_cache"

out_dir = "checkpoints_better_init_12.12"
os.makedirs(out_dir, exist_ok=True)
patch_size = 2
patch_dim = patch_size ** 2 * input_channels 
num_patches = (input_h // patch_size) * (input_w // patch_size)

image_size = 256

d: int = 768                 # Increased model dimension
cond_d: int = 128            # Increased conditioning dimension
nh: int = 12                  # More attention heads
n_layers: int = 4            
n_registers: int = 128
register_subset_lengths = (128,)
fsq_n_levels: int = 5
fsq_l: int = 8
time_dim: int = 128
device: str = "cuda"
lr = 3e-4
if LR_DECAY:
    min_lr = 3e-5
    lr_warmup_iters = 1000
    lr_decay_iters = 83750
    get_lr = get_lr_fn(lr, min_lr, lr_decay_iters, lr_warmup_iters)
else:
    def get_lr(step):
        return lr
betas = (0.9, 0.95)
weight_decay = 0.01
grad_clip = 1.0

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

wandb_run = wandb.init(project="nanoflextok-gochiusa", config=user_config, dir='wandb_logs')

train_loader, test_loader, total_train_steps = get_cached_gochiusa_dataloaders(cache_dir, train_epochs=train_epochs, batch_size=batch_size, num_workers=dataloader_num_workers)
print(f"Total train steps: {total_train_steps}")

vae = Flux2TinyAutoEncoder.from_pretrained(
    vae_model_name,
).eval().to(device=device, dtype=dtype)

def vae_encode_fn(x):
    x = x.to(vae.dtype)
    return vae.encode(x, return_dict=False)

def vae_decode_fn(latents):
    latents = latents.to(vae.dtype)
    return vae.decode(latents, return_dict=False)

def train(model, raw_model, vae_encode_fn, vae_decode_fn, dataloader, lr, device='cuda'):
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.get_parameter_groups(weight_decay), lr=lr, betas=betas, fused=ADAM_FUSED)

    registers_log = []
    running_loss = []
    pbar = tqdm(dataloader, total=total_train_steps)
    for step, (latents, _, label) in enumerate(pbar):

        # set learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        latents = latents.to(device)

        x_patches = image_to_patches(latents, cfg.patch_size)
        
        t = torch.rand(latents.shape[0], device=device)
        t = warp_time(t)
        
        with ctx:
            flow, noise, repa_features, registers = model(x_patches, t)
            target_flow = noise - x_patches
            rf_loss = ((flow - target_flow) ** 2).mean()

            loss = rf_loss

            # cosine_sim = F.cosine_similarity(repa_features, dino, dim=-1)
            # repa_loss = (1 - cosine_sim).mean()
            # loss = rf_loss + repa_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        
        registers_log.append(registers.detach())

        if step % 10 == 0:
            running_loss.append(loss.item())
            pbar.set_postfix({'loss': f'{running_loss[-1]:.4f}', 'iter': step, "lr": lr})

            fsq_stats = fsq_codebook_usage_stats(registers_log, cfg)
            registers_log = []

            wandb_run.log({"loss": loss.item(), 'iter': step, "lr": lr} | fsq_stats)
 
        if step % 1000 == 0:
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
            # torch.save(model.state_dict(), f"{ckpt_base_name}_{step}.pt")


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
            'model_args': filtered_cfg_dict,
            'step': total_train_steps
        }
        torch.save(checkpoint, os.path.join(out_dir,f"{ckpt_base_name}_{total_train_steps}_final.pt"))