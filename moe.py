from typing import List, Dict, Any 
import torch 
from torch import Tensor, nn 
from torch.nn import functional as F 

import fairscale.nn.model_parallel.initialize as fs_init 
from fairscale.nn.model_parallel.mappings import reduce_from_model_parallel_region

from .args import MoEArgs
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

    def load_hook(self, state_dict: Dict[str, Any], prefix: str, local_metadata: Dict[str, Any], strict: bool, missing_keys: List[str], unexpected: List[str], error_msgs: List[str]) -> None: 
        self.prefix = prefix 
        if prefix + "moe_w_in_eD_F" in state_dict: 
            e = self.num_local_experts
            D = self.dim 
            state_dict[prefix + "w1"] = state_dict.pop(prefix + "moe_w_in_eD_F").view(e, D, -1)
            state_dict[prefix + "w2"] = state_dict.pop(prefix + "moe_w_out_eF_D").view(e, D, -1)
            state_dict[prefix + "w3"] = state_dict.pop(prefix + "moe_w_swiglu_eD_F").view(e, D, -1)

    def forward(self, routed_in_egD: torch.Tensor) -> torch.Tensor: 
        e = self.num_local_experts
        D = self.dim

        x_egD = routed_in_egD.view(e, -1, D)
        out_egD = self.batched_swiglu(x_egD, self.w1, self.w3, self.w2)
        out_egD = out_egD.view(-1, D)

        return out_egD
    
    def  batched_swiglu(self, x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> Tensor: 
        middle_out_egF = F.silu(torch.bmm(x, w1)) * torch.bmm(x, w3)
        return torch.bmm(middle_out_egF, w2)
    
class MoE(torch.nn.Module):
    """
    This class uses the Experts class as tool to create MoEs and is also responsible for 
        - routing word(s) to appropriate experts which also includes 
            - scoring 
            - selection 
            - gathering 
            - merging, etc
        - passing words to shared experts 

    Tensors used in this module are annotated with the suffixes that indicate the shape of the tensor 
    Several commonly used annotations include: 
    - a: bsz*slen
    - E: number of experts
    - e: number of local experts per ep (n_experts/ep)
    - D: hidden dimension
    - d: D/tp
    - F: model dimension
    - G: number of tokens per expert (a * capacity_factor / E)
    - g: number of tokens per expert per TP rank (i.e., G/TP)

    Examples:
    x_aD [a, D]
    routed_in_etG_D [et*G, D]
    x_eGD: [e, G, D]
    """

    def __init__(self, dim: int, hidden_dim: int, ffn_dim_multiplier: float, multiple_of: int, moe_args: MoEArgs) -> None: 
        super().__init__()
        self.moe_args = moe_args # gives access to configs saved in MoEArgs class in .args 


        hidden_dim_denom: float = 1 
        if self.moe_args.auto_scale_F: # if capacity factor is high (meaning we are allowing many tokens to crowd one expert)
            # then make the hidden dimension smaller to compensate for memory and compute power 
            hidden_dim_denom = moe_args.capacity_factor + 1 # this keeps the total computational costs of the layer roughly the same 

        # multiply the hidden dimension by 2/3 to reduce it because we are using SwiGLU which uses two linear weight projections w1 and w3 
        hidden_dim = int(2 * hidden_dim / 3) 

        # custom knob to increase hidden_dim if necessary 
        hidden_dim = int(ffn_dim_multiplier * hidden_dim) # acts as an override for hidden_dim 

        # This is where we actually make the hidden dimensions smaller if capacity factor is high 
        if moe_args.auto_scale_F: 
            hidden_dim = int(hidden_dim / hidden_dim_denom)

        # GPU are designed to work with matrices of sizes of multiples of 8 
        hidden_dim += -hidden_dim % multiple_of # Round up the hidden_dim to the nearest multiple of 8 or 16 

        num_local_experts: int = moe_args.num_experts # total number of experts defined fo the model 

        dtype: torch.dtype = torch.get_default_dtype()

        self.experts = Experts(num_local_experts, dim, hidden_dim) # create the experts 

        # create a weight matrix with will be multiplied by each word vector and the resulting matrix will be used as a sorting machine to assign words to appropriate experts 
        self.router_DE: nn.Parameter = nn.Parameter(torch.empty(dim, moe_args.num_experts, dtype=dtype)) 

        # The shared expert is used by all the words. 
        # The idea is to have the expert learn basic stuffs like how to use a commma, basic sentence structure, etc.
        # while allowing the expert to focus on truly difficult, niche topics (coding logic, math, creating writing, etc)
        self.shared_expert = FeedForward(dim, hidden_dim, do_reduce=False) 

        self._register_load_state_dict_pre_hook(self.load_hook) # run load_hook function before loading weights for compactibility 

    def load_hook(self, state_dict: Dict[str, Any], prefix: str, local_metadata: Dict[str, Any], strict: bool, missing_keys: List[str], unexpected_keys: List[str], error_msgs: List[str]) -> None:
        if prefix + "w_in_shared_FD.weight" in state_dict: 
            state_dict[prefix + "shared_expert.w1.weight"] = state_dict.pop(prefix + "w_in_shared_FD.weight")
            state_dict[prefix + "shared_expert.w3.weight"] = state_dict.pop(prefix + "w_swiglu_FD.weight")
            state_dict[prefix + "shared_expert.w2.weight"] = state_dict.pop(prefix + "w_out_shared_DF.weight")

    def forward(self, x_bsD: Tensor) -> Tensor: 
        _, slen, D = x_bsD.shape   # unpack the dimensions into variables leaving out the batch_size (hence the "_")
        # reshape the tensor from (B, S, D) to (BxS, D)
        x_aD = x_bsD.view(-1, D) # because we need only the token's position and the token's dimension 

        a = x_aD.shape[0] # Get the (BxS) component of (BxS, D)

        # multiply the flattened x input with the weight matrix of the router 
        # x_aD shape = [a, D] = a words with D features 
        # self.router_DE shape = [D, E] = D features and E experts 
        # Perform a matrix multiplication of [a, D] * [D, E] = [a, E]
        router_scores: Tensor = torch.matmul(x_aD, self.router_DE).transpose(0,1) # transpose the result from [a, E] to [E, a]

        # router_scores_aK gives us the [words_rows x top_k_expert_scores_columns]. (top_k is usually 2) 
        # router_indices_aK gives us the [words_rows x position_indices_of_top_k_expert_scores_column]
        router_scores_aK, router_indices_aK = torch.topk(router_scores.transpose(0, 1), self.moe_args.top_k, dim=1) 

        # create a weight matrix like router_scores and transpose it 
        # replace the columns and row values it that from the router_indices_aK and router_scores_aK values.
        # The other values are -inf 
        router_scores = torch.full_like(router_scores.transpose(0, 1), float("-inf")).scatter_(1, router_indices_aK, router_scores_aK).transpose(0, 1)

        