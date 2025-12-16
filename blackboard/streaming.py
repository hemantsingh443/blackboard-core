"""
Streaming Support for LLM Responses

Provides protocols and utilities for token-by-token streaming from LLMs.
"""

import asyncio
import logging
from typing import Any, AsyncIterator, Callable, Coroutine, Optional, Protocol, runtime_checkable, Union

from .events import Event, EventType, EventBus

logger = logging.getLogger("blackboard.streaming")


@runtime_checkable
class StreamingLLMClient(Protocol):
    """
    Protocol for LLMs that support streaming responses.
    
    Implement this alongside LLMClient for streaming support.
    
    Example:
        class OpenAIStreamingLLM(StreamingLLMClient):
            async def generate(self, prompt: str) -> str:
                # Non-streaming fallback
                chunks = []
                async for chunk in self.generate_stream(prompt):
                    chunks.append(chunk)
                return "".join(chunks)
            
            async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
                response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                )
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
    """
    
    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """
        Generate a streaming response for the given prompt.
        
        Yields:
            Individual tokens/chunks as they become available
        """
        ...


class StreamCollector:
    """
    Collects streaming tokens and publishes events.
    
    Wraps a streaming LLM to emit events for each token.
    
    Args:
        event_bus: Event bus to publish streaming events to
        source: Identifier for the streaming source (e.g., "supervisor")
        
    Example:
        collector = StreamCollector(event_bus, source="supervisor")
        full_response = await collector.collect(llm.generate_stream(prompt))
    """
    
    def __init__(self, event_bus: EventBus, source: str = "llm"):
        self.event_bus = event_bus
        self.source = source
    
    async def collect(
        self, 
        stream: AsyncIterator[str],
        on_token: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Collect all tokens from a stream, emitting events.
        
        Args:
            stream: Async iterator of tokens
            on_token: Optional callback for each token
            
        Returns:
            Complete collected response
        """
        tokens = []
        
        # Emit stream start
        await self.event_bus.publish_async(Event(
            EventType.STREAM_START,
            {"source": self.source}
        ))
        
        try:
            async for token in stream:
                tokens.append(token)
                
                # Emit token event
                await self.event_bus.publish_async(Event(
                    EventType.STREAM_TOKEN,
                    {"source": self.source, "token": token, "position": len(tokens)}
                ))
                
                # Call optional callback
                if on_token:
                    on_token(token)
            
            full_response = "".join(tokens)
            
            # Emit stream end
            await self.event_bus.publish_async(Event(
                EventType.STREAM_END,
                {"source": self.source, "total_tokens": len(tokens), "length": len(full_response)}
            ))
            
            return full_response
            
        except Exception as e:
            await self.event_bus.publish_async(Event(
                EventType.STREAM_END,
                {"source": self.source, "error": str(e)}
            ))
            raise


class BufferedStream:
    """
    Buffers streaming tokens for consumption.
    
    Useful for building UIs that need to display tokens as they arrive.
    
    Example:
        buffer = BufferedStream()
        
        # In one coroutine (producer):
        async for token in llm.generate_stream(prompt):
            await buffer.add(token)
        await buffer.close()
        
        # In another coroutine (consumer):
        async for token in buffer:
            print(token, end="", flush=True)
    """
    
    def __init__(self, max_buffer: int = 1000):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_buffer)
        self._closed = False
        self._error: Optional[Exception] = None
    
    async def add(self, token: str) -> None:
        """Add a token to the buffer."""
        if self._closed:
            raise RuntimeError("Stream is closed")
        await self._queue.put(token)
    
    async def close(self, error: Optional[Exception] = None) -> None:
        """Close the stream."""
        self._closed = True
        self._error = error
        await self._queue.put(None)  # Sentinel
    
    def __aiter__(self):
        return self
    
    async def __anext__(self) -> str:
        if self._closed and self._queue.empty():
            if self._error:
                raise self._error
            raise StopAsyncIteration
        
        token = await self._queue.get()
        if token is None:
            if self._error:
                raise self._error
            raise StopAsyncIteration
        return token


def wrap_non_streaming(
    generate_fn: Callable[[str], Union[str, Coroutine[Any, Any, str]]]
) -> Callable[[str], AsyncIterator[str]]:
    """
    Wrap a non-streaming generate function to return a fake stream.
    
    Useful for testing or when streaming isn't available.
    Exceptions are properly propagated to the caller.
    
    Args:
        generate_fn: Regular generate function
        
    Returns:
        Function that yields the full response as a single token
    """
    async def streaming_wrapper(prompt: str) -> AsyncIterator[str]:
        try:
            result = generate_fn(prompt)
            if asyncio.iscoroutine(result):
                result = await result
            yield result
        except Exception as e:
            logger.error(f"Error in wrapped generate function: {e}")
            raise
    
    return streaming_wrapper
