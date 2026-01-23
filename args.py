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

    nope_layer_interval: Optional[int] = None  # No position encoding in every n layers
    use_qk_norm: bool = False
    # Set to True to enable inference-time temperature tuning (useful for very long context)
    attn_temperature_tuning: bool = False
    floor_scale: float = 8192.0
    attn_scale: float = 0.1

    attention_chunk_size: Optional[int] = None 
    rope_theta: float = 500000
    use_scaled_rope: bool = False 
    rope_scaling_factor: Optional[float] = None 
    rope_high_freq_factor: Optional[float] = None 

    max_batch_size: int = 32 
    max_seq_len: int = 2048

    @model_validator(mode="after") # after means: verify the datatypes of the settings above are correct. AFTER that, run the function below 
    # The function ensures that the settings specified above actually work together mathematically before the model starts building its neural network layers 
    def validate(self) -> "ModelArgs":
        assert self.n_kv_heads <= self.n_heads, f"n_kv_heads ({self.n_kv_heads}) must be <= n_heads ({self.n_heads})"
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        assert self.dim % self.n_heads == 0, f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"

        if self.use_scaled_rope:
            if self.rope_scaling_factor is None:
                self.rope_scaling_factor = 16
            if self.rope_high_freq_factor is None:
                self.rope_high_freq_factor = 1
        return self 