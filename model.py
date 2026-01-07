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