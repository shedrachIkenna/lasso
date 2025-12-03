
from logging import getLogger
from pathlib import Path 

# type hints for better code documentation and IDE support 
from typing import (
    AbstractSet,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
    cast,
)

import tiktoken # OpenAI's fast BPE tokenizer implementation in Rust 

from tokenizer_utils import load_bpe_file # get merge rules from the model file using the load_bpe_file function 

logger = getLogger(__name__)

# CONSTANTS 
TIKTOKEN_MAX_ENCODE_CHARS = 400_000 # maximum characters tiktoken can encode 

MAX_NO_WHITESPACES_CHARS = 25_000 # Maximum consecutive whitespace or non-whitespace characters before splitting

_INSTANCE = None # Global variable for singleton pattern - stores and caches one tokenizer instance to avoid reloading 




