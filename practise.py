from logging import getLogger 
from pathlib import Path
from typing import (
    AbstractSet, 
    Collection, 
    Literal, 
    Optional, 
    Sequence, 
    Iterator, 
    List, 
    Dict, 
    Union, 
    cast 
)

import titktoken # OpenAI's fast BPE tokenizer implementation in Rust 

from tokenizer_utils import load_bpe_file # Get merge rules from model file using load_bpe_file function 

logger = getLogger(__name__)

TIKTOKEN_MAX_ENCODE_CHARS = 400_000 # max number of chars tiktoken can encode 

MAX_CONSECUTIVE_CHARS = 25_000 # max consecutive whitespace or nonwhitespace characters before splitting 

_INSTANCE = None # Global variable for singleton pattern: stores/caches one instance of titktoken to aviod reloading 

def get_reserved_special_tokens(name, count, start_index=0):
    """
    Function to generate a list of reserved special tokens 

    Args: 
        name: category name of the reserved special token 
        count: How many of the reserved special tokens to generate 
        start_index: start index of the sequence of reserved special tokens to generate 
    """
    return [f"<|{name}_reserved_special_token_{i}|>" for i in range(start_index, start_index+count)]

LLAMA_TEXT_POST_TRAIN_SPECIAL_TOKENS = [
    "|<header_start\>",
    "<|header_end|>",
    "<|eom|>",
    "<|eot|>",
    "<|step|>",
    "<|text_post_train_reserved_special_token_0|>",
    "<|text_post_train_reserved_special_token_1|>",
    "<|text_post_train_reserved_special_token_2|>",
    "<|text_post_train_reserved_special_token_3|>",
    "<|text_post_train_reserved_special_token_4|>",
    "<|text_post_train_reserved_special_token_5|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|finetune_right_pad|>"
] + get_reserved_special_tokens("text_post_train", 61, 8) # <|text_post_train_reserved_special_token_68|>, ..., <|text_post_train_reserved_special_token_68|>