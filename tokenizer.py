
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

