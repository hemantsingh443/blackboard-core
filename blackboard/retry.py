"""
Retry Policies for Blackboard System

Provides retry mechanism with exponential backoff for worker execution.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Type, TypeVar, Awaitable

logger = logging.getLogger("blackboard.retry")

T = TypeVar('T')


@dataclass
class RetryPolicy:
    """
    Configuration for retry behavior.
    
    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retries)
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay after each retry
        max_delay: Maximum delay between retries
        retryable_exceptions: Tuple of exception types that should trigger retry
        
    Example:
        policy = RetryPolicy(
            max_retries=3,
            initial_delay=1.0,
            backoff_factor=2.0,
            retryable_exceptions=(TimeoutError, ConnectionError)
        )
    """
    max_retries: int = 3
    initial_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    retryable_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (TimeoutError, ConnectionError, OSError)
    )
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Check if we should retry based on exception type and attempt count."""
        if attempt >= self.max_retries:
            return False
        return isinstance(exception, self.retryable_exceptions)
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed)."""
        delay = self.initial_delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay)


# Default policy for workers
DEFAULT_RETRY_POLICY = RetryPolicy(
    max_retries=3,
    initial_delay=1.0,
    backoff_factor=2.0
)

# No retry policy
NO_RETRY = RetryPolicy(max_retries=0)


async def retry_with_backoff(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
) -> T:
    """
    Execute an async function with retry and exponential backoff.
    
    Args:
        fn: Async function to execute
        policy: Retry policy configuration
        on_retry: Optional callback called on each retry (attempt, exception, delay)
        
    Returns:
        Result of the function
        
    Raises:
        The last exception if all retries are exhausted
        
    Example:
        async def flaky_api_call():
            response = await fetch_data()
            return response
        
        result = await retry_with_backoff(
            flaky_api_call,
            policy=RetryPolicy(max_retries=3)
        )
    """
    last_exception: Optional[Exception] = None
    
    for attempt in range(policy.max_retries + 1):
        try:
            return await fn()
        except Exception as e:
            last_exception = e
            
            if not policy.should_retry(e, attempt):
                logger.debug(f"Not retrying: {type(e).__name__} is not retryable or max attempts reached")
                raise
            
            delay = policy.get_delay(attempt)
            logger.warning(
                f"Attempt {attempt + 1}/{policy.max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            
            if on_retry:
                on_retry(attempt, e, delay)
            
            await asyncio.sleep(delay)
    
    # Should not reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry loop exited without result or exception")


def is_transient_error(exception: Exception) -> bool:
    """
    Check if an exception is likely transient (worth retrying).
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the exception is likely transient
    """
    transient_types = (
        TimeoutError,
        ConnectionError,
        ConnectionResetError,
        ConnectionRefusedError,
        OSError,
    )
    
    # Check exception type
    if isinstance(exception, transient_types):
        return True
    
    # Check for common transient error messages
    error_msg = str(exception).lower()
    transient_keywords = [
        "timeout",
        "connection",
        "temporarily unavailable",
        "rate limit",
        "too many requests",
        "503",
        "502",
        "504",
    ]
    
    return any(keyword in error_msg for keyword in transient_keywords)
