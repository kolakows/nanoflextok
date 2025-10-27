import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, or_masks
from einops import rearrange

import random
from dataclasses import dataclass

from model.encoder import ViTRegr
from model.fsq import FSQ
from model.decoder import ViTDecoder


@dataclass
class FlexTokConfig:
    patch_dim: int = 64  # 16 latent channels * 2x2 patches
    n_patches: int = 196  # 14x14 patches (from 28x28 latent)
    d: int = 256
    cond_d: int = 128
    nh: int = 8
    n_layers: int = 6
    n_registers: int = 2**6
    possible_k: int = 7
    fsq_n_levels: int = 5
    fsq_l: int = 8
    time_dim: int = 128
    num_classes: int = 1001
    device: str = "cuda"
        
    @property
    def total_seq_len(self):
        return self.n_patches + self.n_registers
    
    @property
    def regr_k_keep(self):
        return [2**i for i in range(self.possible_k)]
    
@dataclass
class FlexTokMnistConfig(FlexTokConfig):
    patch_dim: int = 16          # 4x4 patches x 1 channel = 4 pixels per patch
    n_patches: int = 49          # 7x7 patches (28/4 = 7)
    d: int = 128                 
    cond_d: int = 64             
    nh: int = 4
    n_layers: int = 4
    n_registers: int = 2**5
    possible_k: int = 6
    fsq_n_levels: int = 5
    fsq_l: int = 8         
    time_dim: int = 64           
    num_classes: int = 10
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
        
        self.register_buffer('possible_k', torch.tensor(config.regr_k_keep))
        
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
        registers = self.encoder(patchified_latents, self.enc_block_mask)
        qregisters = self.fsq(registers)  # b x n_registers x fsq_n_levels
        deq_registers = self.fsq.dequantize(qregisters)

        # Sample random k for nested dropout
        k = self.possible_k[torch.randint(len(self.possible_k), (1,), device=patchified_latents.device)]
        # TODO: add learnable mask token for dropped tokens
        # paper: "When performing nested dropout, we replace the dropped tokens with a learnable mask token"
        deq_registers_subset = deq_registers[:, :k, :]

        t = rearrange(timestep, "b -> b 1 1")
        noise = torch.randn_like(patchified_latents)
        noised_patchified_latents = (1 - t) * noise + t * patchified_latents

        # Predict flow
        pred_flow = self.decoder(deq_registers_subset, noised_patchified_latents, timestep)
    
        return pred_flow, noise