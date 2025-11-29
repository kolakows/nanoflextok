import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask
from einops import rearrange

from dataclasses import dataclass

from model.encoder import ViTRegr
from model.fsq import FSQ
from model.decoder import ViTDecoder
from model.iteration_methods import rk4_step


# @dataclass
# class FlexTokConfig:
#     patch_size: int = 8
#     input_dim: int = 28**2
#     d: int = 256
#     cond_d: int = 128
#     nh: int = 8
#     n_layers: int = 6
#     n_registers: int = 64
#     register_subset_lengths: tuple = (1,2,4,8,16,32,64)
#     fsq_n_levels: int = 5
#     fsq_l: int = 8
#     time_dim: int = 128
#     num_classes: int = 1001
#     device: str = "cuda"
        
#     @property
#     def total_seq_len(self):
#         return self.n_patches + self.n_registers
    
#     @property
#     def patch_dim(self):
#         return self.patch_size**2
    
#     @property
#     def n_patches(self):
#         return self.input_dim // self.patch_dim
    
#     def __post_init__(self):
#         assert self.input_dim % self.patch_dim == 0
    
# @dataclass
# class FlexTokMnistConfig(FlexTokConfig):
#     patch_size: int = 2
#     input_dim: int = 28**2
#     d: int = 128                 
#     cond_d: int = 64             
#     nh: int = 4
#     n_layers: int = 4
#     n_registers: int = 32
#     register_subset_lengths: tuple =(1,2,4,8,16,32)
#     fsq_n_levels: int = 5
#     fsq_l: int = 8         
#     time_dim: int = 64           
#     num_classes: int = 10
#     device: str = "cuda"

@dataclass
class FlexTokConfig:
    patch_size: int = 8
    input_h: int = 28
    input_w: int = 28
    input_channels: int = 1  # NEW: support RGB
    d: int = 256
    cond_d: int = 128
    nh: int = 8
    n_layers: int = 6
    n_registers: int = 64
    register_subset_lengths: tuple = (1, 2, 4, 8, 16, 32, 64)
    fsq_n_levels: int = 5
    fsq_l: int = 8
    time_dim: int = 128
    num_classes: int = 1001
    device: str = "cuda"
    
    @property
    def input_dim(self):
        """Total pixels (H * W), not including channels"""
        return self.input_h * self.input_w
    
    @property
    def patch_dim(self):
        """Dimension of each flattened patch (patch_size² * channels)"""
        return self.patch_size ** 2 * self.input_channels
    
    @property
    def n_patches(self):
        """Number of patches in the image"""
        return (self.input_h // self.patch_size) * (self.input_w // self.patch_size)
    
    @property
    def patches_h(self):
        """Number of patches along height"""
        return self.input_h // self.patch_size
    
    @property
    def patches_w(self):
        """Number of patches along width"""
        return self.input_w // self.patch_size
    
    @property
    def total_seq_len(self):
        return self.n_patches + self.n_registers
    
    def __post_init__(self):
        assert self.input_h % self.patch_size == 0, f"input_h ({self.input_h}) must be divisible by patch_size ({self.patch_size})"
        assert self.input_w % self.patch_size == 0, f"input_w ({self.input_w}) must be divisible by patch_size ({self.patch_size})"


@dataclass
class FlexTokMnistConfig(FlexTokConfig):
    patch_size: int = 2
    input_h: int = 28
    input_w: int = 28
    input_channels: int = 1  # Grayscale
    d: int = 128
    cond_d: int = 64
    nh: int = 4
    n_layers: int = 4
    n_registers: int = 32
    register_subset_lengths: tuple = (1, 2, 4, 8, 16, 32)
    fsq_n_levels: int = 5
    fsq_l: int = 8
    time_dim: int = 64
    num_classes: int = 10
    device: str = "cuda"

@dataclass
class FlexTokCc3mConfig(FlexTokConfig):
    patch_size: int = 16        # Standard ViT patch size for 224x224
    input_h: int = 224
    input_w: int = 224
    input_channels: int = 3     # RGB
    d: int = 512                # Larger model dimension
    cond_d: int = 256
    nh: int = 8
    n_layers: int = 8
    n_registers: int = 64
    register_subset_lengths: tuple = (1, 2, 4, 8, 16, 32, 64)
    fsq_n_levels: int = 5
    fsq_l: int = 8
    time_dim: int = 128
    device: str = "cuda"

class FlexTok(nn.Module):

    def __init__(self, config: FlexTokConfig):
        super().__init__()
        self.config = config
        
        self.encoder = ViTRegr(
            config.patch_dim, 
            config.n_patches, 
            config.d, 
            config.nh, 
            config.n_layers, 
            config.n_registers
        )
        
        self.fsq = FSQ(config.fsq_l, config.fsq_n_levels, config.d)
        
        self.decoder = ViTDecoder(
            config.patch_dim, 
            config.n_patches, 
            config.fsq_n_levels, 
            config.time_dim, 
            config.d, 
            config.nh, 
            config.n_layers, 
            config.n_registers, 
            cond_d=config.cond_d, 
            num_classes=config.num_classes
        )
        
        self.register_buffer('register_subset_lengths', torch.tensor(config.register_subset_lengths))

        # Learnable mask tokens for dropped registers during nested dropout
        self.registers_mask_tokens = nn.Parameter(torch.randn((len(config.register_subset_lengths), config.d)))
        self.registers_used_to_mask_idx = {
            subset_length: i for i, subset_length in enumerate(config.register_subset_lengths)
        }

        # Create block mask once during init
        self.enc_block_mask = self._create_prefix_lm_mask()

    def _create_prefix_lm_mask(self):
        """Create prefix LM mask: full attention on patches, causal on registers"""
        def prefix_lm_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            prefix_mask = kv_idx <= self.config.n_patches
            return prefix_mask | causal_mask
        
        return create_block_mask(
            prefix_lm_causal, 
            B=None, 
            H=None, 
            Q_LEN=self.config.total_seq_len, 
            KV_LEN=self.config.total_seq_len, 
            device=self.config.device
        )

    def encode(self, patchified_latents):
        registers = self.encoder(patchified_latents, self.enc_block_mask)
        qregisters = self.fsq(registers)  # b x n_registers x fsq_n_levels
        deq_registers = self.fsq.dequantize(qregisters)
        return deq_registers

    def forward(self, patchified_latents, timestep):
        # Encode to register tokens
        registers = self.encode(patchified_latents)

        # Sample random k for nested dropout
        idx = torch.randint(len(self.register_subset_lengths), (1,), device=patchified_latents.device)
        k = self.register_subset_lengths[idx]
        
        registers_subset = registers[:, :k, :]
        # paper: "When performing nested dropout, we replace the dropped tokens with a learnable mask token"
        register_mask_token = self.get_registers_mask_token(registers_subset)
        
        t = rearrange(timestep, "b -> b 1 1")
        noise = torch.randn_like(patchified_latents)
        noised_patchified_latents = (1 - t) * noise + t * patchified_latents

        # Predict flow
        pred_flow = self.decoder(registers_subset, register_mask_token, noised_patchified_latents, timestep)
    
        return pred_flow, noise
    
    @torch.no_grad
    def reconstruct(self, x, denoising_steps, iteration_method=rk4_step):
        registers = self.encode(x)
        registers_mask = self.get_registers_mask_token(registers)

        ts = torch.linspace(0, 1, denoising_steps).to(registers.device)

        def f(y, t):
            return self.decoder(registers, registers_mask, y, t)

        reconstructed = torch.randn_like(x)
        for i in range(len(ts) - 1):
            dt = ts[i+1] - ts[i]
            timestep = rearrange(ts[i], " -> 1")
            pred_flow = iteration_method(f, reconstructed, timestep, dt)

            reconstructed -= dt * pred_flow

        return reconstructed

    def get_registers_mask_token(self, registers):
        bs, k, _ = registers.shape
        mask_idx = self.registers_used_to_mask_idx[k]
        registers_mask_token = self.registers_mask_tokens[mask_idx]
        return registers_mask_token.expand(bs, 1, -1)
    
    def parameter_count(self):
        return sum((p.numel() for p in self.parameters() if p.requires_grad))