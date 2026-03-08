import math 
from typing import Dict, Any, Callable, List 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from fairscale.nn.model_parallel.layers import ColumnParallelLinear, RowParallelLinear

from ..args import VisionArgs

