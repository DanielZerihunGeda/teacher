from typing import List, AsyncGenerator, Any, Tuple, Optional
import torch
from unsloth import FastLanguageModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#loading the model preferrably with cuda
def load_model(
    model_name: str = "unsloth/Qwen2.5-7B",
    max_seq_length: int = 2048,
    dtype: Optional[torch.dtype] = None,
    load_in_4bit: bool = True,
    token: Optional[str] = None
) -> Tuple[torch.nn.Module, 'PreTrainedTokenizer']:
    """Load a pre-trained model and tokenizer for inference with validation and persistence.

    Args:
        model_name (str): Name or path of the pre-trained model (e.g., "unsloth/Qwen2.5-7B").
        max_seq_length (int): Maximum sequence length for the model. Defaults to 2048.
        dtype (torch.dtype, optional): Data type for model weights (e.g., torch.float16). Defaults to None (auto-detect).
        load_in_4bit (bool): Use 4-bit quantization to reduce memory usage. Defaults to True.
        token (str, optional): Hugging Face token for gated models. Defaults to None.

    Returns:
        Tuple[torch.nn.Module, PreTrainedTokenizer]: Loaded model and tokenizer ready for inference.

    Raises:
        ValueError: If model_name is invalid or unsupported.
        RuntimeError: If model or tokenizer loading fails.
    """
    # Validate inputs
    if not model_name:
        raise ValueError("model_name must be a non-empty string")
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    
    try:
        # Load model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=token
            
        )
        
        # Enable 2x faster inference
        FastLanguageModel.for_inference(model)
        
        logger.info(f"Model {model_name} and tokenizer loaded successfully")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")