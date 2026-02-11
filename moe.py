from typing import List, Dict, Any 
import torch 
from torch import Tensor, nn 
from torch.nn import functional as F 

import fairscale.nn.model_parallel.initialize as fs_init 
from fairscale.nn.model_parallel.mappings import reduce_from_model_parallel_region

from .args import ModelArgs
from .ffn import FeedForward

def divide_exact(numerator: int, denominator: int) -> int: 
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)
    return numerator // denominator

class Experts(nn.Module):
    def __init__(self, num_local_experts: int, dim: int, hidden_dim: int) -> None:
        super().__init__()
        dtype = torch.get_default_dtype()
        self.num_local_experts = num_local_experts
        self.dim = dim 
        divide_factor = fs_init.get_model_parallel_world_size()

        self.w1: nn.Parameter = nn.Parameter(
            torch.empty(
                num_local_experts, 
                dim, 
                divide_exact(hidden_dim, divide_factor),
                dtype=dtype
            )
        )

        self.w2: nn.Parameter = nn.Parameter(
            torch.empty(
                num_local_experts, 
                dim,
                divide_exact(hidden_dim, divide_factor),
                dtype=dtype
            )
        )

        self.w3: nn.Parameter = nn.Parameter(
            torch.empty(
                num_local_experts, 
                dim, 
                divide_exact(hidden_dim, divide_factor), 
                dtype=dtype
            )
        )

        self._register_load_state_dict_pre_hook(self.load_hook)
