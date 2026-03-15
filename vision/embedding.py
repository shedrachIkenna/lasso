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