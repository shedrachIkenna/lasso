import math 
from typing import Any, Dict, Optional, Tuple
import torch 
import torch.nn.functional as F 
from torch import nn 

def rmsnorm(x, eps):
    def _norm(y):
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return _norm(x.float()).type_as(x)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rmsnorm(x, self.eps) * self.weight
    

def apply_scaling(freq: torch.Tensor, scale_factor: float, high_freq_factor: float):
    """
    Increase the context window of the model 
    """
    low_freq_factor = 1 
    old_context_len = 8192 # original context limit defined by llama devs 
    low_freq_wavelen = old_context_len / low_freq_factor # = 8192. 
    high_freq_wavelen = old_context_len / high_freq_factor 
