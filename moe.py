from typing import List, Dict, Any 
import torch 
from torch import Tensor, nn 
from torch.nn import functional as F 

import fairscale.nn.model_parallel.initialize as fs_init 
from fairscale.nn.model_parallel.mappings import reduce_from_model_parallel_region

from .args import ModelArgs
from .ffn import FeedForward


