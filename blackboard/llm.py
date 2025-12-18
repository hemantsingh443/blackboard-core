"""
LiteLLM Integration

Provides a unified LLM client using LiteLLM for 100+ model support.
This is the default LLM client for blackboard-core.

Example:
    from blackboard import Orchestrator
    from blackboard.llm import LiteLLMClient
    
    # Simple usage (auto-detects API key from environment)
    llm = LiteLLMClient(model="gpt-4o")
    
    # With fallback models
    llm = LiteLLMClient(
        model="gpt-4o",
        fallback_models=["gpt-4o-mini", "claude-3-5-sonnet-20241022"]
    )
    
    orchestrator = Orchestrator(llm=llm, workers=[...])
"""

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Awaitable, Dict, List, Optional, Union

from .core import LLMClient
from .usage import LLMResponse, LLMUsage
from .streaming import StreamingLLMClient
from .tools import ToolCallingLLMClient, ToolDefinition, ToolCall

logger = logging.getLogger("blackboard.llm")


class LiteLLMClient(LLMClient, StreamingLLMClient, ToolCallingLLMClient):
    """
    Unified LLM client using LiteLLM.
    
    Supports 100+ models from OpenAI, Anthropic, Google, Azure, Cohere,
    Mistral, Ollama, and more with a single API.
    
    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        api_key: Optional API key (auto-detects from environment by default)
        fallback_models: List of fallback models if primary fails
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        **kwargs: Additional LiteLLM parameters
        
    Environment Variables:
        OPENAI_API_KEY: For OpenAI models
        ANTHROPIC_API_KEY: For Claude models
        GOOGLE_API_KEY: For Gemini models
        See LiteLLM docs for full list
        
    Example:
        # Basic usage
        llm = LiteLLMClient(model="gpt-4o")
        response = llm.generate("Hello!")
        
        # With fallbacks for reliability
        llm = LiteLLMClient(
            model="gpt-4o",
            fallback_models=["gpt-4o-mini"],
            temperature=0.7
        )
        
        # Async streaming
        async for token in llm.generate_stream("Write a story"):
            print(token, end="", flush=True)
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
        **kwargs: Any
    ):
        self.model = model
        self.fallback_models = fallback_models or []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.kwargs = kwargs
        
        # Set API key if provided
        if api_key:
            import litellm
            litellm.api_key = api_key
    
    async def generate(self, prompt: str) -> LLMResponse:
        """
        Generate a response.
        
        NOTE: This is async to avoid blocking the event loop.
        The Orchestrator handles async returns automatically.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLMResponse with content and usage stats
        """
        import litellm
        
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                fallbacks=self.fallback_models if self.fallback_models else None,
                **self.kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content or "",
                usage=LLMUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    model=response.model
                ),
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
            
        except Exception as e:
            logger.error(f"LiteLLM generation error: {e}")
            raise
    
    async def agenerate(self, prompt: str) -> LLMResponse:
        """
        Generate a response asynchronously.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLMResponse with content and usage stats
        """
        import litellm
        
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                fallbacks=self.fallback_models if self.fallback_models else None,
                **self.kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content or "",
                usage=LLMUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    model=response.model
                ),
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
            
        except Exception as e:
            logger.error(f"LiteLLM async generation error: {e}")
            raise
    
    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """
        Generate a streaming response.
        
        Yields tokens as they become available from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Yields:
            Individual tokens/chunks as strings
        """
        import litellm
        
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                stream=True,
                **self.kwargs
            )
            
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"LiteLLM streaming error: {e}")
            raise
    
    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[ToolDefinition]
    ) -> Union[str, List[ToolCall]]:
        """
        Generate a response with tool calling support (async).
        
        Uses LiteLLM's native function calling to let the model
        call tools directly, avoiding JSON parsing.
        
        NOTE: This is now async to avoid blocking the event loop.
        
        Args:
            prompt: The prompt to send
            tools: Available tools
            
        Returns:
            Either a string response or a list of tool calls
        """
        import litellm
        
        # Convert tools to OpenAI format (LiteLLM uses this for all providers)
        tool_defs = [tool.to_openai_format() for tool in tools]
        
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=tool_defs,
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                **self.kwargs
            )
            
            message = response.choices[0].message
            
            # Check if LLM made tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": tc.function.arguments}
                    
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments
                    ))
                return tool_calls
            
            # No tool calls - return text response
            return message.content or ""
            
        except Exception as e:
            logger.error(f"LiteLLM tool calling error: {e}")
            raise
    
    async def agenerate_with_tools(
        self,
        prompt: str,
        tools: List[ToolDefinition]
    ) -> Union[str, List[ToolCall]]:
        """
        Generate a response with tool calling support (async).
        
        Args:
            prompt: The prompt to send
            tools: Available tools
            
        Returns:
            Either a string response or a list of tool calls
        """
        import litellm
        
        tool_defs = [tool.to_openai_format() for tool in tools]
        
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=tool_defs,
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                **self.kwargs
            )
            
            message = response.choices[0].message
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": tc.function.arguments}
                    
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments
                    ))
                return tool_calls
            
            return message.content or ""
            
        except Exception as e:
            logger.error(f"LiteLLM async tool calling error: {e}")
            raise
    
    def __repr__(self) -> str:
        fallbacks = f", fallbacks={self.fallback_models}" if self.fallback_models else ""
        return f"LiteLLMClient(model={self.model!r}{fallbacks})"


def create_llm(
    model: str = "gpt-4o",
    **kwargs: Any
) -> LiteLLMClient:
    """
    Convenience function to create a LiteLLM client.
    
    Args:
        model: Model identifier
        **kwargs: Additional parameters passed to LiteLLMClient
        
    Returns:
        Configured LiteLLMClient instance
        
    Example:
        from blackboard.llm import create_llm
        
        llm = create_llm("claude-3-5-sonnet-20241022", temperature=0.5)
    """
    return LiteLLMClient(model=model, **kwargs)
