from pydantic import BaseModel, model_validator 
from typing import Optional


class ModelArgs(BaseModel): # inherite all the methods from the basemodel class for data handling 
    dim: int = -1 
    n_layers: int = -1 
    n_heads: int = -1 
    n_kv_heads: Optional[int] = None 
    head_dim: Optional[int] = None 

    