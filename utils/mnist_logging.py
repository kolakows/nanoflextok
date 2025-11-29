import wandb
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
from PIL import Image

from utils import image_to_patches_mnist, patches_to_image_mnist

@torch.no_grad
def log_test_mse(model, dataloader, sample_size, cfg, wandb_run):
    model.eval()
    mse_acc = []
    for _, (x, y) in tqdm(zip(range(sample_size), dataloader), total=sample_size, desc="Validation step"):
        x = x.squeeze().to(cfg.device)
        x_patches = image_to_patches_mnist(x, cfg)
        x_hat_patches = model.reconstruct(x_patches, denoising_steps=25)
        mse = ((x_patches-x_hat_patches)**2).mean()
        mse_acc.append(mse.item())
    mean_mse = sum(mse_acc)/len(mse_acc)
    wandb_run.log({"reconstruction_mse": mean_mse})
    print(f"Validation mse: {mean_mse:.4f}")
    model.train()

@torch.no_grad
def log_reconstructed_images(model, dataloader, cfg, wandb_run):
    model.eval()
    x, y = next(iter(dataloader))
    x = x.squeeze().to(cfg.device)
    x_patches = image_to_patches_mnist(x, cfg)
    x_hat_patches = model.reconstruct(x_patches, denoising_steps=25)
    x_hat = patches_to_image_mnist(x_hat_patches, cfg)
    img_comparison = rearrange([x, x_hat], "l (eight1 eight2) h w -> (eight1 h) (eight2 l w)", eight1=8)
    
    imgs = img_comparison.cpu().numpy()
    imgs = ((imgs - imgs.min())/(imgs.max() - imgs.min())) * 255
    
    imgs = Image.fromarray(imgs)
    imgs = imgs.resize((3*imgs.size[0], 3*imgs.size[1]), Image.BILINEAR)

    imgs = wandb.Image(np.array(imgs))
    wandb_run.log({"Sample reconstructed images": imgs})
    model.train()
