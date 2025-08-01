from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
import os
from typing import Dict, Optional
import json
from datasets import load_dataset
def log(*s, **kw):
    print(*s, flush=True, **kw)

#### loading the dataset from huggingface#######
def load_prompts(file_path: str = 'template.json') -> Dict[str, str]:
  # Get the directory of the current script
  SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

  # Construct the path to templates.json relative to the script
  TEMPLATE_PATH = os.path.join(SCRIPT_DIR, file_path)
  with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
    return json.load(f)

PROMPT = load_prompts()

def load_data(
    tokenizer,
    d_set: str,
    prompt_name: str = "alpaca_prompt",
    inst_prompt: Dict[str, str] = PROMPT,
    columns: tuple = ("instruction", "input", "output")
):
    if prompt_name not in inst_prompt:
        raise ValueError(f"Prompt '{prompt_name}' not found in prompt templates.")
    
    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        try:
            instruction_col, input_col, output_col = columns
            instructions = examples[instruction_col]
            inputs = examples[input_col]
            outputs = examples[output_col]
            texts = [
                inst_prompt[prompt_name].format(instruction, input, output) + EOS_TOKEN
                for instruction, input, output in zip(instructions, inputs, outputs)
            ]
            return {"text": texts}
        except KeyError as e:
            raise KeyError(f"Missing required field {e} in dataset.")
        except ValueError as e:
            raise ValueError(f"Prompt formatting error: {e}")

    try:
        dataset = load_dataset(d_set, split="train")  # Use Hugging Face's load_dataset
        dataset = dataset.map(formatting_prompts_func, batched=True, batch_size=1000)
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load or process dataset '{d_set}': {e}")