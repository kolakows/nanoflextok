import wandb

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from dataclasses import dataclass, fields
from model.flextok import FlexTokMnistConfig, FlexTok

input_h = 28
input_dim = 28**2
patch_size: int = 2
patch_dim: int = patch_size**2
assert input_dim % patch_dim == 0
d: int = 128                 
cond_d: int = 64             
nh: int = 4
n_layers: int = 4
n_registers: int = 32
register_subset_lengths = (32,)
fsq_n_levels: int = 5
fsq_l: int = 8         
time_dim: int = 64           
num_classes: int = 10
device: str = "cuda"
lr = 1e-4

epochs = 100
validation_sample_size = 5

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, tuple))]
# exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys}

valid_fields = {f.name for f in fields(FlexTokMnistConfig)}
filtered_cfg_dict = {k: v for k, v in user_config.items() if k in valid_fields}
cfg = FlexTokMnistConfig(**filtered_cfg_dict)

run = "test"
wandb_run = wandb.init(project="nanoflextok", config=user_config)


# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=2)


import torch
from einops import rearrange
from torch.optim import AdamW
from tqdm import tqdm

def warp_time(t, dt=None, s=.25):
    # https://drscotthawley.github.io/blog/posts/FlowModels.html#more-points-where-needed-via-time-warping
    tw = 4*(1-s)*t**3 + 6*(s-1)*t**2 + (3-2*s)*t 
    if dt:
        return tw,  dt * 12*(1-s)*t**2 + 12*(s-1)*t + (3-2*s) 
    return tw


@torch.no_grad
def log_test_mse(model, dataloader, sample_size):
    model.eval()
    mse_acc = []
    for _, (x, y) in tqdm(zip(range(sample_size), dataloader), total=sample_size, desc=f"Validation step"):
        x = x.squeeze().to(device)
        x_patches = rearrange(x, "b (h p1) (w p2) -> b (h w) (p1 p2)", p1=patch_size, p2=patch_size)
        x_hat_patches = model.reconstruct(x_patches, denoising_steps=25)
        mse = ((x_patches-x_hat_patches)**2).mean()
        mse_acc.append(mse.item())
    mean_mse = sum(mse_acc)/len(mse_acc)
    wandb_run.log({"reconstruction_mse": mean_mse})
    print(f"Validation mse: {mean_mse:.4f}")
    model.train()

@torch.no_grad
def log_reconstructed_images(model, dataloader):
    model.eval()
    x, y = next(iter(dataloader))
    x = x.squeeze().to(device)
    x_patches = rearrange(x, "b (h p1) (w p2) -> b (h w) (p1 p2)", p1=patch_size, p2=patch_size)
    x_hat_patches = model.reconstruct(x_patches, denoising_steps=25)
    x_hat = rearrange(x_hat_patches, "b (h w) (p1 p2) -> b (h p1) (w p2)", p1=patch_size, p2=patch_size, h=input_h//patch_size)    
    img_comparison = rearrange([x, x_hat], "l (eight1 eight2) h w -> (eight1 h) (eight2 l w)", eight1=8)
    
    imgs = img_comparison.cpu().numpy()
    imgs = ((imgs - imgs.min())/(imgs.max() - imgs.min())) * 255
    
    imgs = Image.fromarray(imgs)
    imgs = imgs.resize((3*imgs.size[0], 3*imgs.size[1]), Image.BILINEAR)

    imgs = wandb.Image(np.array(imgs))
    wandb_run.log({"Sample reconstructed images": imgs})
    model.train()

def train(
    model, 
    dataloader, 
    num_epochs=10,
    lr=1e-4,
    device='cuda'
):
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    log_test_mse(model, test_loader, sample_size=validation_sample_size)
    log_reconstructed_images(model, test_loader)
    
    for epoch in range(num_epochs):
        epoch_loss = []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            x = x.squeeze()
            x = rearrange(x, "b (h p1) (w p2) -> b (h w) (p1 p2)", p1=patch_size, p2=patch_size)

            t = torch.rand(x.shape[0], device=x.device)
            t = warp_time(t)

            flow, noise = model(x, t)
            target_flow = noise - x

            loss = ((flow - target_flow) ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                epoch_loss.append(loss.item())
                pbar.set_postfix({'loss': f'{epoch_loss[-1]:.4f}'})
                wandb_run.log({
                    "loss": loss
                })
        
        avg_loss = sum(epoch_loss)/len(epoch_loss)
        log_test_mse(model, test_loader, sample_size=validation_sample_size)
        print(f'Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}')
        log_reconstructed_images(model, test_loader)


device = "cuda"
model = FlexTok(cfg)
model.to(device)
train(model, train_loader, num_epochs=epochs, lr=lr)
torch.save(model.state_dict(), f"mnist_flow_{epochs}.pt")
 