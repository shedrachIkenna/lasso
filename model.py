import math 
from typing import Any, Dict, Optional, Tuple
import torch 
import torch.nn.functional as F 
from torch import nn 

from .args import ModelArgs

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
    # Base frequency calculation: inv_freq = 1 / theta^(2i/d) logic 
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # Generate list of increased position indexes (Assuming we are increasing context to 32k, t = [0, 1, 2, 3, ..., 31999])
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)

    # Apply scaling if use_scaled is True 
    if use_scaled: 
        freqs = apply_scaling(freqs, scale_factor, high_freq_factor)

    # Positional angle calculation: t * freq. Multiplies every position t with its corresponding frequency freq 
    freqs = torch.outer(t, freqs)

    # Complex Rotation: e^(i*theta) = cos(theta) + i*sin(theta) (Euler's formula)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # Get how many dimensions x has 
    ndim = x.ndim

    # check that x has at least 2 dimensions 
    assert 0 <= 1 < ndim

    # checks whether seq and D/2 in freqs_cis and x matches 
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    # looks at x tensor, loops through each dimension
    # replace all the dimensions with 1s except dimensions at index 1 and the last index (3) (dim - index - [0,1,2,3])
    # Goal: reshape freqs_cis from [seq, D/2] to [1, seq, 1, D/2]
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, use_qk_norm: bool, use_rope: bool, add_bias: bool = False):
        super().__init__()
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.attn_temperature_tuning = args.attn_temperature_tuning
        self.floor_scale = args.floor_scale
        self.attn_scale = args.attn_scale

        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # world_size = fs_init.get_model_parallel_world_size() - returns the number of GPU's we have. Lets us splits model heads across GPUs - Model Parallelism 
        world_size = 8 # random value 
        self.n_local_heads = args.n_heads // world_size # if we have 8 GPUs and 32 heads, each GPU will process 32/8 = 4 heads 
        self.n_local_kv_heads = args.n_kv_heads // world_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads 
        self.head_dim = args.dim // args.n_heads

