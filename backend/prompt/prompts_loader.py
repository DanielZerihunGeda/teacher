import json
from pathlib import Path
from langchain_core.prompts import PromptTemplate
import asyncio

_semaphore = asyncio.Semaphore(100)
async def load_prompt(prompt_path: str = "templates.json", 
                  prompt_name: str = "query",
                  time_out : float = 60) -> PromptTemplate:
  """
  Args:

    prompt_path : json file where prompts and corresponding variables live

    prompt_name: selected template
  
  return :

    instance of PromptTemplate to be formatted according to the 
    input_variables. 

    
  """
  try:
    async with asyncio.timeout(time_out):
      async with _semaphore:

        data = json.loads(Path(prompt_path).read_text())
        prompt_data = data[prompt_name]

        return PromptTemplate(
          input_variables=prompt_data.get("input_variables"),
          template=prompt_data.get("template"),
          template_format=prompt_data.get("template_format", "f-string")
        )

  except asyncio.TimeoutError:
    return "Timed out while loading prompt"