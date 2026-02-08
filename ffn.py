from typing import Any, List, Dict 

from fairscale.nn.model_parallel.layers import ColumnParallelLinear, RowParallelLinear
from fairscale.nn.model_parallel.mappings import reduce_from_model_parallel_region
from torch import nn 
from torch.nn import functional as F 

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, do_reduce: bool = True):
        super().__init__()
        self.do_reduce = do_reduce, 

        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self._register_load_state_dict_pre_hook(self.load_hook)

        