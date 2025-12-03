import base64
from pathlib import Path 
import logging

log = logging.getLogger(__name__)

def load_bpe_file(model_path: Path) -> dict[bytes, int]:
    """
    Args: 
        model_path (Path): Path to the BPE model file 

    Returns the merge rules stored in the model file 
    """

    mergeable_ranks = {} # dictionary where the bpe token mappings (merge rules) will be stored after its extracted from the model 

    with open(model_path, encoding="utf-8") as f: # open the model file at the model_path 
        content = f.read() # read the entire contents of the model file into the variable "content" as a string 

    for line in content.splitlines(): # splits content by line breaks and iterates through each line 
        if not line.strip(): # skip empty lines 
            continue

        try: 
            # We are expecting two values per line 
            token, rank = line.split() # split the line by white-space and unpacks the result into two variables 

            # Decode the base64-encoded token string back into bytes using base64.b64decode() 
            # Convert the rank string to an interger 
            # store the mapping in the mergeable_ranks dictionary (bytes as key, integer as value)
            mergeable_ranks[base64.b64decode(token)] = int(rank)

        except Exception as e:  # catch any exception that occurs in the try block 
            # log a warning message if parsing fails, including the problematic line and error message 
            log.warning(f"Failed to parse line '{line}': {e}") 
            continue

    return mergeable_ranks


