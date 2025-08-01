from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
import os
from typing import Dict, Optional
import json
from datasets import load_dataset

#dynamic logging
def log(*s, **kw):
    print(*s, flush=True, **kw)

#### loading the dataset from huggingface#######
def load_prompts(file_path: str = 'template.json') -> Dict[str, str]:
  # Get the directory of the current file
  SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

  TEMPLATE_PATH = os.path.join(SCRIPT_DIR, file_path)
  with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
    return json.load(f)

PROMPT = load_prompts()

def load_data(
    tokenizer,
    d_set: str,
    prompt_name: str = "alpaca_prompt",
    inst_prompt: Dict[str, str] = PROMPT,
    columns: tuple = ("instruction", "input", "output"),
    batch_size: int = 1000
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
        dataset = load_dataset(d_set, split="train")  
        dataset = dataset.map(formatting_prompts_func, batched=True, batch_size=batch_size)

        log("Dataset loaded successfully")
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load or process dataset '{d_set}': {e}")

# Finetune class
class FineTune:
  """A class for fine-tuning language models using LoRA adapters and supervised fine-tuning (SFT).

    This class facilitates loading a pre-trained model and tokenizer, adding LoRA (Low-Rank Adaptation)
    adapters for parameter-efficient fine-tuning, and training the model on a custom dataset using
    the SFTTrainer from the TRL library. It supports configurations for quantization, gradient
    checkpointing, and mixed-precision training (FP16/BF16)
    
    """
  def __init__(self,
               model = None,
               tokenizer = None,
               max_seq_length: int = 2048,
               dtype=None,
               load_in_4bit: bool = True,
               model_name: str = "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit",
               r: int = 16,
               target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
               lora_alpha: int = 16,
               lora_dropout: float = 0,
               bias: str = "none",
               use_gradient_checkpointing: str = "unsloth",
               random_state: int = 3407,
               use_rslora: bool = False,
               loftq_config=None,
               dataset_text_field: str = "text",
               dataset_num_proc: int = 2,
               packing: bool = False,
               per_device_train_batch_size: int = 2,
               gradient_accumulation_steps: int = 4,
               warmup_steps: int = 5,
               max_steps: int = 60,
               learning_rate = 2e-4,
               fp16 = not is_bfloat16_supported(),
               bf16 = is_bfloat16_supported(),
               logging_steps: int = 1,
               optim: str = "adamw_8bit",
               weight_decay: float = 0.01,
               lr_scheduler_type: str = "linear",
               seed: int = 3407,
               output_dir: str = "outputs",
               report_to: str = "none",):
    self.max_seq_length = max_seq_length
    self.dtype = dtype
    self.load_in_4bit = load_in_4bit
    self.model_name = model_name
    self.r = r
    self.target_modules = target_modules
    self.lora_alpha = lora_alpha
    self.lora_dropout = lora_dropout
    self.bias = bias
    self.use_gradient_checkpointing = use_gradient_checkpointing
    self.random_state = random_state
    self.use_rslora = use_rslora
    self.loftq_config = loftq_config
    self.model = model
    self.tokenizer = tokenizer
    self.dataset_text_field = dataset_text_field
    self.dataset_num_proc = dataset_num_proc
    self.packing = packing
    self.per_device_train_batch_size = per_device_train_batch_size
    self.gradient_accumulation_steps = gradient_accumulation_steps
    self.warmup_steps = warmup_steps
    self.max_steps = max_steps
    self.learning_rate = learning_rate
    self.fp16 = fp16
    self.bf16 = bf16
    self.logging_steps = logging_steps
    self.optim = optim

  # def __call__(self):

  #   return self.load_model()

  def load_model(self):
    if self.model is not None and self.tokenizer is not None:
        print("Model and tokenizer already provided, skipping load")
        return
    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
        model_name=self.model_name,
        max_seq_length=self.max_seq_length,
        dtype=self.dtype,
        load_in_4bit=self.load_in_4bit
    )
    log("######### Model and Tokenizer Loaded successfully ###########")

  def add_lora_adaptor(self):

    #make sure the model is loaded successfully
    if self.model is None:
      raise RuntimeError("Model must be loaded before adding LoRA adapters")


    self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.r,
            target_modules=self.target_modules,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            random_state=self.random_state,
            use_rslora=self.use_rslora,
            loftq_config=self.loftq_config
        )
    log("####### LoRA adapters added successfully! ##############")

  def train(self,
            train_dataset):
    if self.model is None or self.tokenizer is None:
      raise RuntimeError("Model and tokenizer must be loaded before training")

    trainer = SFTTrainer(
        model=self.model,
        tokenizer=self.tokenizer,
        train_dataset=train_dataset,
        dataset_text_field=self.dataset_text_field,
        max_seq_length=self.max_seq_length,
        dataset_num_proc=self.dataset_num_proc,
        packing=self.packing,
        args=TrainingArguments(
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            fp16=self.fp16,
            bf16=self.bf16,
            logging_steps=self.logging_steps,
            optim=self.optim,
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler_type,
            seed=self.seed,
            output_dir=self.output_dir,
            report_to=self.report_to,
        ),
    )
    trainer.train()
    log(f"#########Training completed! Checkpoints saved in {self.output_dir}############")
