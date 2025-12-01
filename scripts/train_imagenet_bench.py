import os
import wandb
import torch
from torch.optim import AdamW
from tqdm import tqdm

from dataclasses import fields
from model.flextok import FlexTokCc3mConfig, FlexTok
from utils.utils import warp_time, image_to_patches
from utils.imagenet import get_imagenet_dataloaders, get_cached_dataloaders
from utils.logging import log_reconstructed_images, log_test_mse

from diffusers.models import AutoencoderKL

# ============ Configuration ============
USE_C16_VAE = False
TORCH_COMPILE = True
ADAM_FUSED = False
SAVE_CKPT = False

# ============ Hyperparameters ============
# vae latents
input_h = 32
input_w = 32

if USE_C16_VAE:
    input_channels = 16
    vae_model_name = 'EPFL-VILAB/flextok_vae_c16'
    ckpt_base_name = "imagenet_c16"
else:
    input_channels = 4
    vae_model_name = 'EPFL-VILAB/flextok_vae_c4'
    ckpt_base_name = "imagenet_c4"
    cache_dir = "latent_cache_c4"
patch_size = 2
patch_dim = patch_size ** 2 * input_channels  # 16*16*3 = 768
num_patches = (input_h // patch_size) * (input_w // patch_size)  # 14*14 = 196

image_size = 256

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
weight_decay = 0.01

train_epochs = 50
val_img_dec_samples = 8
val_mse_batches = 1
batch_size = 32
dataloader_num_workers = 8

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

wandb_run = wandb.init(project="nanoflextok-imagenet-100", config=user_config, dir='wandb_logs', mode="offline")

train_loader, test_loader, total_train_steps = get_cached_dataloaders(cache_dir, train_epochs=train_epochs, batch_size=batch_size, num_workers=dataloader_num_workers)
print(f"Total train steps: {total_train_steps}")

vae = AutoencoderKL.from_pretrained(
    vae_model_name, low_cpu_mem_usage=False
).eval().to(device)

def vae_encode_fn(x):
    return vae.encode(x).latent_dist.mean

def vae_decode_fn(latents):
    return vae.decode(latents).sample

import time
from collections import defaultdict

class TimeBenchmark:
    """Simple timer for benchmarking different parts of training."""
    
    def __init__(self, warmup_steps=50, log_interval=100):
        self.warmup_steps = warmup_steps
        self.log_interval = log_interval
        self.timings = defaultdict(list)
        self.step = 0
        self.current_timers = {}
        
    def start(self, name):
        """Start timing a named section."""
        torch.cuda.synchronize()
        self.current_timers[name] = time.perf_counter()
    
    def stop(self, name):
        """Stop timing a named section and record the duration."""
        torch.cuda.synchronize()
        if name in self.current_timers:
            elapsed = (time.perf_counter() - self.current_timers[name]) * 1000  # ms
            if self.step >= self.warmup_steps:
                self.timings[name].append(elapsed)
            del self.current_timers[name]
    
    def step_done(self):
        """Mark end of a training step."""
        self.step += 1
    
    def get_stats(self):
        """Get timing statistics."""
        stats = {}
        for name, times in self.timings.items():
            if times:
                stats[name] = {
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'last': times[-1],
                    'count': len(times)
                }
        return stats
    
    def get_summary_string(self):
        """Get a formatted summary string."""
        stats = self.get_stats()
        if not stats:
            return "No timing data yet (still in warmup)"
        
        lines = ["\n" + "="*60, "TIMING BREAKDOWN (ms)", "="*60]
        
        # Calculate total time per step
        total_mean = sum(s['mean'] for s in stats.values())
        
        # Sort by mean time descending
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        for name, s in sorted_stats:
            pct = (s['mean'] / total_mean * 100) if total_mean > 0 else 0
            lines.append(f"{name:25s}: {s['mean']:8.2f} (min: {s['min']:6.2f}, max: {s['max']:6.2f}) [{pct:5.1f}%]")
        
        lines.append("-"*60)
        lines.append(f"{'TOTAL':25s}: {total_mean:8.2f} ms/step")
        lines.append(f"{'Throughput':25s}: {1000/total_mean:.1f} steps/sec" if total_mean > 0 else "")
        lines.append("="*60 + "\n")
        
        return "\n".join(lines)
    

def train2(model, vae_encode_fn, vae_decode_fn, dataloader, lr, device='cuda'):
    """Training function with comprehensive time benchmarking."""
    
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.get_parameter_groups(weight_decay), lr=lr, betas=betas, fused=ADAM_FUSED)
    
    timer = TimeBenchmark(warmup_steps=50, log_interval=100)
    
    running_loss = []
    pbar = tqdm(dataloader, total=total_train_steps)
    
    # Pre-fetch iterator to measure dataloader time properly
    data_iter = iter(dataloader)
    
    for step in range(total_train_steps):
        # ===== DATALOADER =====
        timer.start('1_dataloader')
        try:
            x, captions = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, captions = next(data_iter)
        timer.stop('1_dataloader')
        
        # ===== DATA TRANSFER TO GPU =====
        timer.start('2_to_device')
        x = x.to(device, non_blocking=True)
        torch.cuda.synchronize()  # Ensure transfer is complete
        timer.stop('2_to_device')
        
        # # ===== VAE ENCODING =====
        # timer.start('3_vae_encode')
        # with torch.no_grad():
        #     latents = vae_encode_fn(x)
        #     x = latents
        # timer.stop('3_vae_encode')
        
        # ===== PREPROCESSING (patches, time sampling) =====
        timer.start('4_preprocess')
        x_patches = image_to_patches(x, cfg.patch_size)
        t = torch.rand(x.shape[0], device=device)
        t = warp_time(t)
        timer.stop('4_preprocess')
        
        # ===== FORWARD PASS =====
        timer.start('5_forward')
        with ctx:
            flow, noise = model(x_patches, t)
            target_flow = noise - x_patches
            loss = ((flow - target_flow) ** 2).mean()
        timer.stop('5_forward')
        
        # ===== BACKWARD PASS =====
        timer.start('6_backward')
        optimizer.zero_grad()
        loss.backward()
        timer.stop('6_backward')
        
        # ===== OPTIMIZER STEP =====
        timer.start('7_optimizer')
        optimizer.step()
        timer.stop('7_optimizer')
        
        timer.step_done()
        
        # ===== LOGGING =====
        if step % 10 == 0:
            running_loss.append(loss.item())
            pbar.set_postfix({'loss': f'{running_loss[-1]:.4f}', 'iter': step})
            wandb_run.log({"loss": loss.item(), 'iter': step})
        
        # Log timing stats periodically
        if step % 110 == 0 and step > timer.warmup_steps:
            print(timer.get_summary_string())
        
        # if step % 1000 == 0:
        #     log_test_mse(model, vae_encode_fn, test_loader, val_mse_batches, cfg, wandb_run, step)
        #     log_reconstructed_images(model, vae_encode_fn, vae_decode_fn, test_loader, val_img_dec_samples, cfg, wandb_run, step)

        pbar.update(1)
    
    # Final timing summary
    print("\n" + "="*60)
    print("FINAL TIMING SUMMARY")
    print(timer.get_summary_string())
    
    return timer.get_stats()


# ============ Main ============
if __name__ == "__main__":
    model = FlexTok(cfg)
    if TORCH_COMPILE:
        model = torch.compile(model)
        vae_encode_fn = torch.compile(vae_encode_fn)
        vae_decode_fn = torch.compile(vae_decode_fn)
    print(f"Trainable parameter count: {model.parameter_count()}")
    
    timing_stats = train2(model, vae_encode_fn, vae_decode_fn, train_loader, lr=lr)