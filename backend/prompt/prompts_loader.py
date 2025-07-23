import json
from pathlib import Path
import asyncio
import os
import fire
template_path = os.path.join(os.path.dirname(__file__), "templates", "templates.json")

# Load templates safely
try:
    with open(template_path, 'r', encoding='utf-8') as f:
        PROMPT = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Could not find templates file at: {template_path}")
except json.JSONDecodeError:
    raise RuntimeError(f"Invalid JSON format in templates file: {template_path}")

async def load_prompt(
    prompt_name: str = "query",
    context: str = '',
    evaluation: str = '',
    *args,
    **kwargs
) -> str:
    """
    Args:
        prompt_name: Name of the prompt template to load
        context: Context string to include in the prompt
        evaluation: Evaluation criteria to include in the prompt
        *args: Additional positional arguments (currently unused)
        **kwargs: Additional keyword arguments for template formatting
    
    Returns:
        str: Formatted prompt template with inserted values
    
    Raises:
        ValueError: If prompt_name is not found or required template variables are missing
    """
    prompt_template = PROMPT.get(prompt_name)
    if prompt_template is None:
        raise ValueError(f"Prompt template '{prompt_name}' not found.")
    
    prompt_template = prompt_template["template"]
    
    if context:
        fmt_kwargs = {
            k: v for k, v in {'context': context, 'evaluation': evaluation, **kwargs}.items() 
            if v is not None
        }
        try:
            prompt_template = prompt_template.format(**fmt_kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required template variable: {e.args[0]}")
    
    return prompt_template



if __name__=="__main__":
  fire.Fire(load_prompt)