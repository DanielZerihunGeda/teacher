from typing import Callable, Any
import asyncio

class RateLimiter:
    def __init__(self, max_concurrency: int = 32):
        """
        Initialize the RateLimiter with maximum concurrent operations.


        Args:
            max_concurrency: max_concurrency
        """
        self.semaphore = asyncio.Semaphore(value=max_concurrency)

    def rate_limited(self, func: Callable) -> Callable:
        """
        Decorator to limit concurrent execution of a function.

        Args:
            func: The function to be rate limited

        Returns:
            A wrapped function with rate limiting
        """
        async def wrapper(*args, **kwargs):
            async with self.semaphore:
                return await func(*args, **kwargs)
        return wrapper

    async def acquire(self) -> None:
        """Acquire the semaphore for manual rate limiting."""
        await self.semaphore.acquire()

    async def release(self) -> None:
        """Release the semaphore for manual rate limiting."""
        self.semaphore.release()