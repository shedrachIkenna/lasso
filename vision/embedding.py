import math 
from typing import Dict, Any, Callable, List 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from fairscale.nn.model_parallel.layers import ColumnParallelLinear, RowParallelLinear

from ..args import VisionArgs

class PixelShuffle(nn.Module): 
    def __init__(self, ps_ratio):
        super().__init__()
        self.ps_ratio = ps_ratio

    def forward(self, x): 
        # x: [B, N, C] where N = number of patches 
        # N: number of tokens (or seq)
        # C: How many numbers are inside each patch (token embeddings - channels/embedding_dim)
        assert self.ps_ratio is not None, "ps_ratio is required for pixel shuffle"
        assert x.dim() == 3, "pixel shuffle requires encoded patches [B, N, C]"
        # get the height and width of the image tensor: x 
        hh = ww = int(math.sqrt(x.shape[1])) # x.shape[1] = N. so we take the sqrt of N which gives us the stacked hh and ww 
        x = x.reshape(x.shape[0], hh, ww, -1) # x changes from [B, N, C] to [B, hh, ww, C]
        x = pixel_shuffle_op(x, ps_ratio=self.ps_ratio) # [B, hh, ww, C] -> [B, hh/2, ww/2, 4C]
        pixel_shuffle_patches = x.reshape(x.shape[0], -1, x.shape[-1]) # [B, hh, ww, C] to [B, ((hh/2) * (ww/2), 4C]
        return pixel_shuffle_patches
    
def pixel_shuffle_op(input_x, ps_ratio): 
    """
    Logic: Think of this like folding a piece of paper in half vertically, then folding it in half horizontally. 
            The result is a smaller square with four layers thick 
    """
    n, w, h, c = input_x.size() 
    input_x = input_x.view(n, w, int(h * ps_ratio), int(c / ps_ratio))
    input_x = input_x.permute(0, 2, 1, 3).contiguous()
    input_x = input_x.view(n, int(h * ps_ratio), int(w * ps_ratio), int(c / (ps_ratio * ps_ratio)))
    input_x = input_x.permute(0, 2, 1, 3).contiguous()
    return input_x

class SimpleMLP(torch.nn.Module): 
    def __init__(self, dim: int, hidden_dim: int, bias: bool = True, dropout: float = 0.0, act_layer: Callable = nn.GELU):
        super().__init__()

        # Projects tokens to higher (hidden_dim) dimensions. Also split the weight matrix vertically (by columns)
        self.c_fc = ColumnParallelLinear(dim, hidden_dim, bias=bias, gather_output=False)

        # out projection (note that its dimension is still the same allowing the model to capture more complex relationships)
        self.c_proj = RowParallelLinear(hidden_dim, hidden_dim, bias=bias, input_is_parallel=True)

        # model activation function : GELU 
        self.non_linearity = act_layer()

        # Regularization to prevent overfitting
        self.drop_out = dropout
    
    def forward(self, x): 
        hidden = self.c_fc(x)
        hidden = self.non_linearity(hidden)
        hidden = F.dropout(hidden, p=self.drop_out, training=self.training)
        return self.non_linearity(self.c_proj(hidden))