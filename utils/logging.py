import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm
from utils.utils import image_to_patches, patches_to_image

import wandb

@torch.no_grad()
def fsq_codebook_usage_stats(registers_log, cfg):

    register_codes = torch.cat(registers_log).view(-1, cfg.fsq_n_levels)
    quantized_codes = register_codes.detach().cpu()

    total_codes_len, levels = quantized_codes.shape
    unique_codes = torch.unique(quantized_codes, dim=0)
    unique_codes_len, _ = unique_codes.shape

    values, counts = torch.unique(quantized_codes, return_counts=True)
    values, counts = torch.round(values.float(), decimals=2).tolist(), counts.float().tolist()
    values_distrib = {v:c for v,c in zip(values, counts)}

    # table = wandb.Table(data=[[k, v] for k, v in values_distrib.items()], 
    #                 columns=["code_value", "count"])
    # code_values_plot = wandb.plot.bar(table, "code_value", "count", title="Code values Distribution")

    desc_distrib = {f"codes_dist/code_val_{v: .2f}": c for v, c in values_distrib.items()}

    return {
        "total_codes_len": total_codes_len,
        "unique_codes_len": unique_codes_len,
        "unique_percentage": unique_codes_len/total_codes_len,
        # "codes_dist/code_val_{k}": code_values_plot,
    } | desc_distrib


@torch.no_grad()
def log_test_mse(model, vae_encode_fn, dataloader, sample_size, cfg, wandb_run, step):
    model.eval()
    mse_acc = []
    
    for i, (x, _, _) in enumerate(tqdm(dataloader, total=sample_size, desc="Validation step")):
        if i >= sample_size:
            break
        x = x.to(cfg.device)
        # x = vae_encode_fn(x)
        x_patches = image_to_patches(x, cfg.patch_size)
        x_hat_patches = model.reconstruct(x_patches, denoising_steps=25)
        mse = ((x_patches - x_hat_patches) ** 2).mean()
        mse_acc.append(mse.item())
    
    mean_mse = sum(mse_acc) / len(mse_acc)
    wandb_run.log({"reconstruction_mse": mean_mse, "iter": step})
    print(f"Validation mse: {mean_mse:.4f}")
    model.train()


@torch.no_grad()
def log_reconstructed_images(model, vae_encode_fn, vae_decode_fn, dataloader, sample_size, cfg, wandb_run, step):
    model.eval()
    x, _, _ = next(iter(dataloader))
    x = x[:sample_size].to(cfg.device)
    
    # x_lat = vae_encode_fn(x)
    x_lat = x
    x_patches = image_to_patches(x_lat, cfg.patch_size)
    x_hat_patches = model.reconstruct(x_patches, denoising_steps=25)
    x_hat = patches_to_image(x_hat_patches, cfg.patch_size, cfg.input_h)
    x_dec = vae_decode_fn(x_lat)
    x_gen = vae_decode_fn(x_hat)

    # Create side-by-side comparison: original | reconstructed
    comparison = torch.cat([x_dec, x_gen], dim=3)  # Concatenate along width
    
    # Denormalize for visualization
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(cfg.device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(cfg.device)
    comparison = comparison.clamp(-1, 1) * std + mean

    # Create grid
    grid = rearrange(comparison, "(r c) ch h w -> (r h) (c w) ch", r=1, c=sample_size)
    grid = (grid.cpu().numpy() * 255).astype(np.uint8)
    
    wandb_run.log({"Sample reconstructed images": wandb.Image(grid), "iter": step})
    model.train()