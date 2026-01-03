import torch
import torch.nn as nn
import torch.nn.functional as F


class FSQ(nn.Module):
    # TODO: audio and more advanced here https://arxiv.org/pdf/2411.19842
    def __init__(self, L, n_levels, d_model, eps: float = 1e-3):
        super().__init__()
        # levels = torch.tensor([8, 8, 8, 8, 8], dtype=torch.int)
        # self.register_buffer("levels", levels)
        self.L = L
        # TODO: refactor to encoder
        self.regrs_ln = nn.LayerNorm(d_model)
        self.regrs_head = nn.Linear(d_model, n_levels, bias=False)
        self.eps = eps
        
    def _round_ste(self, x):
        return x + (torch.round(x) - x).detach()

    def forward(self, regrs):

        regrs = self.regrs_head(self.regrs_ln(regrs))

        # eps to avoid boundary codes
        half_l = self.L * (1-self.eps) / 2 
        bounded = half_l * F.tanh(regrs)
        code = self._round_ste(bounded)

        return code
    
    def scale(self, quantizied):
        half = self.L // 2
        return quantizied / half