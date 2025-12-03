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
