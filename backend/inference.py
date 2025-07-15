import asyncio
from typing import List, AsyncGenerator, Any, Tuple
from langchain_core.prompts import PromptTemplate

_semaphore = asyncio.Semaphore(32)


async def client(model: str):
  #TODO

  # pass the model with proper model and return the client itself from the very begining
  pass

  return client



async def run(prompt: str, time_out: float = 60.0) -> str:


  '''

  it accepts a prompt and respond, restricted to max_concurrency with semaphore


  raise timeout error iff timeout reached for each coroutine

  Args:

    prompt:


  return: 

    AI response
    
  '''

  try:

    async with asyncio.timeout(time_out):
      async with _semaphore:
        cli = await client()
        return await cli.create.chat.completions(prompt)


  except asyncio.TimeoutError:
    return "Timed out"