
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

def get_reserved_special_tokens(name, count, start_index=0):
    """
    Function to generate a list of reserved special tokens 

    name:
        category name of the reserved special token 

    count: 
        How many of these reserved special tokens to generate 
    
    start_index: 
        start index for the sequence of reserved special tokens 
    """
    return [f"<|{name}_reserved_special_token_{i}|>" for i in range(start_index, start_index + count)]

LLAMA4_TEXT_POST_TRAIN_SPECIAL_TOKENS = [
    "<|header_start|>", 
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
    "<|finetune_right_pad|>",
] + get_reserved_special_tokens("text_post_train", 61, 8)  # <|text_post_train_reserved_special_token_8|>, ..., <|text_post_train_reserved_special_token_68|>


LLAMA4_VISION_SPECIAL_TOKENS = [
    "<|image_start|>",
    "<|image_end|>",
    "<|vision_reserved_special_token_0|>",
    "<|vision_reserved_special_token_1|>",
    "<|tile_x_separator|>",
    "<|tile_y_separator|>",
    "<|vision_reserved_special_token_2|>",
    "<|vision_reserved_special_token_3|>",
    "<|vision_reserved_special_token_4|>",
    "<|vision_reserved_special_token_5|>",
    "<|image|>",
    "<|vision_reserved_special_token_6|>",
    "<|patch|>",
] + get_reserved_special_tokens("vision", 1041, 7)  # <|vision_reserved_special_token_7|>, ..., <|vision_reserved_special_token_1047|>


LLAMA4_REASONING_SPECIAL_TOKENS = [
    "<|reasoning_reserved_special_token_0|>",
    "<|reasoning_reserved_special_token_1|>",
    "<|reasoning_reserved_special_token_2|>",
    "<|reasoning_reserved_special_token_3|>",
    "<|reasoning_reserved_special_token_4|>",
    "<|reasoning_reserved_special_token_5|>",
    "<|reasoning_reserved_special_token_6|>",
    "<|reasoning_reserved_special_token_7|>",
    "<|reasoning_thinking_start|>",
    "<|reasoning_thinking_end|>",
]

# Concatenate all the special tokens into one master special token list 
LLAMA4_SPECIAL_TOKENS = (
    LLAMA4_TEXT_POST_TRAIN_SPECIAL_TOKENS + LLAMA4_VISION_SPECIAL_TOKENS + LLAMA4_REASONING_SPECIAL_TOKENS
)

BASIC_SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
]


class Tokenizer:
    """
    Tokenize, encode and decode text using the Tiktoken tokenizer 
    """
    special_tokens: Dict[str, int] # we will have a dictionary of special tokens mapped to their int IDs

    num_reserved_special_tokens = 2048 # slots reserved for special tokens 

    # O200k Regex pattern for splitting text before tokenization 
    O200K_PATTERN = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

    @classmethod
    def get_instance(cls):
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = Tokenizer(Path(__file__).parent / "tokenizer.model")

        return _INSTANCE
    
    def __init__(self, model_path: Path):
        """
        Constructor method that initialize the Tokenizer with the tiktoken model 

        Args:
            model_path (Path): path to the tiktoken model file 
        """
        if not model_path.exists(): # check if the model file exists 
            # if not, raise an error
            raise FileNotFoundError(f"Tokenizer model file not found: {model_path}") 
        
        # Load the BPE merge rules from the model file 
        mergeable_ranks = load_bpe_file(model_path) 

        # count how many base tokens 
        num_base_tokens = len(mergeable_ranks)

        # combine the basic special tokens and Llama 4 special tokens 
        special_tokens = BASIC_SPECIAL_TOKENS + LLAMA4_SPECIAL_TOKENS

        # remove duplicates if exists 
        assert len(set(special_tokens)) == len(special_tokens)

        # verify we haven't exceeded the 2,048 token budget, (1,144 <= 2,048) - we are good 
        assert len(special_tokens) <= self.num_reserved_special_tokens

        # create remaining special tokens. Range is 2,048 - 1,144 = 904 tokens, numbered 0-903
        reserved_tokens = [
            f"<|reserved_special_token_{i}|>" for i in range(self.num_reserved_special_tokens - len(special_tokens))
        ]

        # add reserved tokens to special token list to have exactly 2,048 tokens 
        special_tokens = special_tokens + reserved_tokens

        # create dictionary mapping each special token string to its ID, IDs start after base tokens 
        # if base tokens ends at 199,999, special tokens are 200,000 - 202,047.
        self.special_tokens = {token: num_base_tokens + i for i, token in enumerate(special_tokens)}

        # create the tiktoken object 
        self.model = tiktoken.Encoding(
            name=model_path.name, # name 
            pat_str=self.O200K_PATTERN, # regex pattern for splitting 
            mergeable_ranks=mergeable_ranks # bpe merge rules loaded from file 
            special_tokens=self.special_tokens # special tokens 
        )
        # get the total vocabulary size 
        self.n_words: int = num_base_tokens + len(special_tokens)

        # get the ID for beginning of sequence token 
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]

        # Get the ID for end of sequence token  
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]

        # Get the ID of padding-token (used for making sequence same length)
        self.pad_id: int = self.special_tokens["<|finetune_right_pad|>"]
        
        # Get the ID for end-of-turn token 
        self.eot_id: int = self.special_tokens["<|eot|>"]

        # get the ID for end of message token 
        self.eom_id: int = self.special_tokens["<|eom|>"]

        # Get the ids for reasoning/thinking boundaries 
        self.thinking_start_id: int = self.special_tokens["<|reasoning_thinking_start|>"]
        self.thinking_end_id: int = self.special_tokens["<|reasoning_thinking_end|>"]

        # Create list of tokens that should stop generation (when model should stop producing text)
        self.stop_tokens = [
            self.eos_id,
            self.special_tokens["<|eom|>"],
            self.special_tokens["<|eot|>"],
        ]

        