from pydantic import BaseModel, model_validator 
from typing import Optional


class ModelArgs(BaseModel): # inherite all the methods from the basemodel class for data handling 
    dim: int = -1 
    n_layers: int = -1 
    n_heads: int = -1 
    n_kv_heads: Optional[int] = None 
    head_dim: Optional[int] = None 

    vocab_size: int = -1 
    multiple_of: int = 256 
    ffn_dim_multiplier: Optional[float] = None 
    ffn_exp: Optional[float] = None 
    norm_eps: float = 1e-5

    attention_chunk_size: Optional[int] = None 
    rope_theta: float = 500000
    use_scaled_rope: bool = False 
    rope_scaling_factor: Optional[float] = None 
    rope_high_freq_factor: Optional[float] = None 

    max_batch_size: int = 32 
    max_seq_len: int = 2048

    

    