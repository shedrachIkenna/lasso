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

    
    def load_hook(self, state_dict: Dict[str, Any], prefix: str, local_metadata: Dict[str, Any], strict: bool, missing_keys: List[str], unexpected_keys: List[str], error_msgs: List[str]) -> None:
        """
        Intercepts weights files and unpacks the combined weight matrix into its individual layer weight components 
        """
        if prefix + "mlp.fc1_weight" in state_dict:
            w1, w3 = state_dict.pop(prefix + "mlp.fc1_weight").chunk(2, dim=0)
            state_dict[prefix + "w1.weight"] = w1 
            state_dict[prefix + "w3.weight"] = w3
            state_dict[prefix + "w2.weight"] = state_dict.pop(prefix + "mlp.fc2_weight")