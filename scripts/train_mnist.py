from dataclasses import fields

from model.flextok import FlexTok, FlexTokMnistConfig
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.mnist_logging import log_reconstructed_images, log_test_mse
from utils.utils import warp_time

import wandb

input_h = 28
input_dim = 28**2
patch_size: int = 4
patch_dim: int = patch_size**2
assert input_dim % patch_dim == 0
d: int = 128                 
cond_d: int = 64             
nh: int = 4
n_layers: int = 4
n_registers: int = 32
register_subset_lengths = (1,2,4,8,16,32)
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
    log_test_mse(model, test_loader, validation_sample_size, cfg, wandb_run)
    log_reconstructed_images(model, test_loader, cfg, wandb_run)
    
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
        log_test_mse(model, test_loader, validation_sample_size, cfg, wandb_run)
        print(f'Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}')
        log_reconstructed_images(model, test_loader, cfg, wandb_run)


device = "cuda"
model = FlexTok(cfg)
model.to(device)
train(model, train_loader, num_epochs=epochs, lr=lr)
torch.save(model.state_dict(), f"mnist_flow_{epochs}.pt")
 