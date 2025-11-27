import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention
from einops import rearrange


class MHAttention(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        assert d % nh == 0
        self.proj = nn.Linear(d, 3*d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.nh = nh

    def forward(self, x, block_mask=None):
        qkv = self.proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        # q, k, v = rearrange(qkv, '... (three d) -> three ... d', three=3)

        q = rearrange(q, "b s (h hd) -> b h s hd", h=self.nh)
        k = rearrange(k, "b s (h hd) -> b h s hd", h=self.nh)
        v = rearrange(v, "b s (h hd) -> b h s hd", h=self.nh)

        attn = flex_attention(q, k, v, block_mask=block_mask)

        attn = rearrange(attn, "b h s hd -> b s (h hd)")
        
        attn = self.o_proj(attn)

        return attn

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj_up = nn.Linear(d, 4*d, bias=False)
        self.proj_down = nn.Linear(4*d, d, bias=False)
        self.silu = nn.SiLU()
        # self.dropout = nn.Dropout(0.1)
     
    def forward(self, x):
        x = self.proj_up(x)
        x = self.silu(x)
        x = self.proj_down(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.ln_2 = nn.LayerNorm(d)
        self.attn = MHAttention(d, nh)
        self.mlp = MLP(d)

    def forward(self, x, block_mask=None):
        x = x + self.attn(self.ln_1(x), block_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class TransformerBlockAdaLNZero(nn.Module):
    def __init__(self, d, nh, cond_d):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d, elementwise_affine=False)
        self.ln_2 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = MHAttention(d, nh)
        self.mlp = MLP(d)
        self.cond_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_d, 6*d, bias=True)
            )
        
        # AdaLNZero zero part, initially forward(x) = x
        nn.init.zeros_(self.cond_proj[-1].weight)
        nn.init.zeros_(self.cond_proj[-1].bias)

        
    def forward(self, x, conditioning):
        projected_cond = self.cond_proj(conditioning)
        # chunking and adding 1 sequence dim for broadcasting
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = rearrange(projected_cond, "b (six d) -> six b 1 d", six=6)

        x_attn = self.ln_1(x)
        x_attn = (gamma1 + 1) * x + beta1
        x_attn = self.attn(x_attn)
        x_attn = alpha1 * x_attn
        x = x + x_attn

        x_mlp = self.ln_2(x)
        x_mlp = (gamma2 + 1) * x + beta2
        x_mlp = self.mlp(x_mlp)
        x_mlp = alpha2 * x_mlp
        x = x + x_mlp

        return x