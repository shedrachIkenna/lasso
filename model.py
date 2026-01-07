import math 
from typing import Any, Dict, Optional, Tuple
import torch 
import torch.nn.functional as F 
from torch import nn 


def rmsnorm(x, eps):
    def _norm(y):
        # y.pow(2): Square every number in the vector 
        # .mean(-1, keepdim=True): Average them across the last dimension 
        # + eps: Add the safety constant to prevent division by zero error 
        # torch.rsqrt(...): calculate 1 / sqrt(value)
        # y * ...: Multiply the original vector by the reciprocal 
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)

    # .float(): force to float32 for high precision during the math calculation 
    # .type_as(x): Convert back to original type (eg. bfloat16) to save memory 
    return _norm(x.float()).type_as(x)