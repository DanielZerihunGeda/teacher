from typing import Callable, Any
import asyncio
from prometheus_client import Gauge



class RateLimiter:
    def __init__(self, 
    max_concurrency: int = 32,
    timeout: int = 60):
        """
        Initialize the RateLimiter with maximum concurrent operations.


        Args:
            max_concurrency: max_concurrency
        """
        self.semaphore = asyncio.Semaphore(value=max_concurrency)
        self.timeout = timeout
        self.active_requests = Gauge('active_requests', 'Number of active requests')

    def rate_limited(self, func: Callable) -> Callable:
        """
        Decorator to limit concurrent execution of a function.

        Args:
            func: The function to be rate limited

        Returns:
            A wrapped function with rate limiting
        """
        async def wrapper(*args, **kwargs):
          try:
              async with asyncio.timeout(self.timeout): 
                  async with self.semaphore:
                    with self.active_requests.track_inprogress():
                      return await func(*args, **kwargs)
          except asyncio.TimeoutError:
              raise TimeoutError("Request timed out waiting for semaphore")

    async def acquire(self) -> None:
        """Acquire the semaphore for manual rate limiting."""
        await self.semaphore.acquire()

    async def release(self) -> None:
        """Release the semaphore for manual rate limiting."""
        self.semaphore.release()