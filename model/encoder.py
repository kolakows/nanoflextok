import torch.nn as nn
from einops import repeat, pack, unpack


from model.transformer import ReZeroTransformerBlock, TransformerBlock


class ViTRegr(nn.Module):

    def __init__(self, patch_dim, max_patches_len, d, nh, n_layers, n_registers, rezero):
        super().__init__()
        self.patch_ln = nn.LayerNorm(patch_dim)
        self.patch_proj = nn.Linear(patch_dim, d, bias=False)
        # TODO: rotary embs? rotary for decoding FSQ, abs for encoding VAE patches?
        self.wpe = nn.Embedding(max_patches_len + n_registers, d)
        block_cls = ReZeroTransformerBlock if rezero else TransformerBlock
        self.blocks = nn.ModuleList(
            [block_cls(d, nh) for _ in range(n_layers)]
        )
        self.registers = nn.Embedding(n_registers, d)
        self.n_registers = n_registers
        

    def forward(self, x, block_mask=None):
        b, t, patch_dim = x.shape
        # ln ensures that tokens after packing are of similar scale at init
        x = self.patch_ln(x)
        x = self.patch_proj(x) # shape (b, t, d)
        registers = repeat(self.registers.weight, "n d -> b n d", b=b)

        x, ps = pack([x, registers], "b * d")

        # TODO: doesn't handle variable sized latent patches
        pos_emb = self.wpe.weight # shape (t, d)
        x = x + pos_emb
        for block in self.blocks:
            x = block(x, block_mask)

        _, registers = unpack(x, ps, "b * d")

        return registers