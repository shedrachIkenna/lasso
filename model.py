import math 
from typing import Any, Dict, Optional, Tuple, List
import torch 
import torch.nn.functional as F 
from torch import nn 

import fairscale.nn.model_parallel.initialize as fs_init

from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

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
        world_size = fs_init.get_model_parallel_world_size() # returns the number of GPU's we have. Lets us splits model heads across GPUs - Model Parallelism 
        self.n_local_heads = args.n_heads // world_size # if we have 8 GPUs and 32 heads, each GPU will process 32/8 = 4 heads 
        self.n_local_kv_heads = args.n_kv_heads // world_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads 
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim, 
            args.n_heads * self.head_dim, 
            bias=add_bias, 
            gather_output=False, 
            init_method=lambda x: x, 
        )

        self.wk = ColumnParallelLinear(
            args.dim, 
            args.n_kv_heads * self.head_dim, 
            bias=add_bias, 
            gather_output=False, 
            init_method= lambda x:x, 
        )

        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=add_bias,
            gather_output=False,
            init_method=lambda x: x,
        )

        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=add_bias,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros((
            args.max_batch_size,
            args.max_seq_len,
            self.n_local_kv_heads,
            self.head_dim,
        )).cuda()

        self.cache_v = torch.zeros((
            args.max_batch_size,
            args.max_seq_len,
            self.n_local_kv_heads,
            self.head_dim,
        )).cuda()

        self.norm_eps = args.norm_eps
        self._register_load_state_dict_pre_hook(self.load_hook) # Responsible for loading weights correctly so that each GPU gets the appropriate size of weight matrix 

    def load_hook(self, state_dict: Dict[str, Any], prefix: str, local_metadate: Dict[str, Any], strict: bool, missing_keys: List[str], unexpected_key: List[str], error_msgs: List[str]) -> None: 
        """
        Loads pretrained weights from file (wqkv.weight)

        Args: 
            state_dict: dictionary containing all the weights in the file 
            prefix: Points to exactly where an attention layer weight is (models usually have multiple attention layers)
            strict (bool): if true, any tiny mismatch between code and weight file results to an error. 
                example - Missing keys: The code has a layer (e.g. self.bias) but the weight file doesn't have any numbers for it 
                          Unexpected keys: The weight file has numbers for a layer (e.g extra_layer.weight) but the code doesn't have that layer
            missing_keys/unexpected_keys: These are empty lists that the hook can fill if it notices something is wrong during inspection 
        """
        if prefix + "wqkv.weight" in state_dict:
            wqkv = state_dict.pop(prefix + "wqkv.weight")
            # get how many rows belong to a single head
            d, r = divmod(wqkv.shape[0], self.n_heads + 2 * self.n_kv_heads)
            if r != 0:
                raise ValueError(
                    f"shape={tuple(wqkv.shape)} is not divisible by "
                    f"n_heads ({self.n_heads}) + 2 * n_kv_heads ({self.n_kv_heads})"
                )
            # split wqkv matrix into Query, key and value matrices 
            wq, wk, wv = wqkv.split([d * self.n_heads, d * self.n_kv_heads, d * self.n_kv_heads], dim=0)

            # save the wq, wk, wv matrices in state_dict
            state_dict[prefix + "wq.weight"] = wq
            state_dict[prefix + "wk.weight"] = wk 
            state_dict[prefix + "wv.weight"] = wv  


    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # bsz = batch_size, seqlen = sequence length, _ = dimension of each token
        bsz, seqlen, _ = x.shape # tuple unpacking 

        # Project x into Query, Key and Value by doing a matrix multiplication of the wq, wk, wk pretrained weights with x 
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape x into a 4-dimensional tensor  
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if self.use_rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        if self.use_qk_norm:
            xq = rmsnorm(xq, self.norm_eps)
            xk = rmsnorm(xk, self.norm_eps)