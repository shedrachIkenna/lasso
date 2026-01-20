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
    

def apply_scaling(freqs: torch.Tensor, scale_factor: float, high_freq_factor: float):
    """
    Increase the context window of the model 
    """
    low_freq_factor = 1
    old_context_len = 8192 # original context limit defined by llama devs 
    low_freq_wavelen = old_context_len / low_freq_factor # = 8192. 
    high_freq_wavelen = old_context_len / high_freq_factor 
    new_freqs = [] # list to store new freqs 

    for freq in freqs: 
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen: 
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen: 
            new_freqs.append(freq / scale_factor)
        else: 
            assert low_freq_wavelen != high_freq_wavelen # safety check to avoid ZeroDivisionError in smooth calculation 
            smooth = (old_context_len / wavelen - low_freq_factor) / high_freq_factor - low_freq_factor
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis(dim: int, end: int, theta: float, use_scaled: bool, scale_factor: float, high_freq_factor: float): 
    

