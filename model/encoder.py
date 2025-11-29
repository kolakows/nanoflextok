import torch.nn as nn
from einops import repeat, pack, unpack


from model.transformer import TransformerBlock


class ViTRegr(nn.Module):

    def __init__(self, patch_dim, max_patches_len, d, nh, n_layers, n_registers):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, d, bias=False)
        # TODO: rotary embs? rotary for decoding FSQ, abs for encoding VAE patches?
        self.wpe = nn.Embedding(max_patches_len + n_registers, d)
        self.encoder = nn.ModuleList(
            [TransformerBlock(d, nh) for _ in range(n_layers)]
        )
        self.registers = nn.Embedding(n_registers, d)
        self.n_registers = n_registers
        

    def forward(self, x, block_mask=None):
        b, t, patch_dim = x.shape
        x = self.patch_proj(x) # shape (b, t, d)
        # registers.expand(b, -1, -1)
        registers = repeat(self.registers.weight, "n d -> b n d", b=b)

        # x = torch.cat([x, registers], dim=1)
        x, ps = pack([x, registers], "b * d")

        # TODO: doesn't handle variable sized latent patches
        pos_emb = self.wpe.weight # shape (t, d)
        x = x + pos_emb
        for block in self.encoder:
            x = block(x, block_mask)

        # regrs = x[:, -self.n_registers:, :]
        _, registers = unpack(x, ps, "b * d")

        return registers