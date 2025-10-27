import torch
import torch.nn as nn
import torch.nn.functional as F


class FSQ(nn.Module):
    # TODO: audio and more advanced here https://arxiv.org/pdf/2411.19842
    def __init__(self, L, n_levels, d_model):
        super().__init__()
        # levels = torch.tensor([8, 8, 8, 8, 8], dtype=torch.int)
        # self.register_buffer("levels", levels)
        self.L = L
        self.regrs_ln = nn.LayerNorm(d_model)
        self.regrs_head = nn.Linear(d_model, n_levels, bias=False)
        
    def _round_ste(self, x):
        return x + (torch.round(x) - x).detach()

    def forward(self, regrs):

        regrs = self.regrs_head(self.regrs_ln(regrs))

        q = self._round_ste(self.L * F.tanh(regrs)/2)
        return q
    
    def dequantize(self, quantizied):
        return 2 * quantizied / self.L