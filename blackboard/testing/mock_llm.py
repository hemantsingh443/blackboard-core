"""
Mock LLM Client for Deterministic Testing

Provides a fake LLM that returns predefined responses for unit testing.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core import LLMClient, LLMResponse, LLMUsage


@dataclass
class MockResponse:
    """A mock response configuration."""
    pattern: Optional[str] = None  # Regex pattern to match prompt
    response: str = ""  # Response text to return
    usage: Optional[LLMUsage] = None  # Optional usage stats


class MockLLMClient(LLMClient):
    """
    Mock LLM client for deterministic testing.
    
    Supports two modes:
    1. Pattern matching: Match prompts against regex patterns
    2. Sequential: Return responses in order, ignoring prompts
    
    Example (Pattern matching):
        mock = MockLLMClient(responses=[
            (".*plan.*", "<response><action>call</action>...</response>"),
            (".*search.*", "Found results"),
        ])
        
    Example (Sequential):
        mock = MockLLMClient(sequence=[
            "<response><action>call</action><workers>Searcher</workers></response>",
            "<response><action>done</action></response>",
        ])
    """
    
    def __init__(
        self,
        responses: Optional[List[Tuple[str, str]]] = None,
        sequence: Optional[List[str]] = None,
        default_response: str = "<response><action>done</action><reasoning>Mock response</reasoning></response>",
        track_calls: bool = True
    ):
        """
        Initialize mock LLM.
        
        Args:
            responses: List of (pattern, response) tuples for pattern matching
            sequence: List of responses to return in order (ignores prompt)
            default_response: Fallback response when no pattern matches
            track_calls: If True, record all calls for inspection
        """
        self.responses = responses or []
        self.sequence = sequence or []
        self.default_response = default_response
        self.track_calls = track_calls
        
        self._call_history: List[Dict[str, Any]] = []
        self._sequence_index = 0
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a mock response."""
        # Track call
        if self.track_calls:
            self._call_history.append({
                "prompt": prompt,
                "system_prompt": system_prompt,
                "kwargs": kwargs
            })
        
        # Determine response
        response_text = self._get_response(prompt)
        
        return LLMResponse(
            content=response_text,
            usage=LLMUsage(
                input_tokens=len(prompt.split()),
                output_tokens=len(response_text.split()),
                total_tokens=len(prompt.split()) + len(response_text.split()),
                model="mock-model"
            ),
            metadata={"mock": True}
        )
    
    def _get_response(self, prompt: str) -> str:
        """Determine which response to return."""
        # Sequential mode takes priority
        if self.sequence and self._sequence_index < len(self.sequence):
            response = self.sequence[self._sequence_index]
            self._sequence_index += 1
            return response
        
        # Pattern matching mode
        for pattern, response in self.responses:
            if re.search(pattern, prompt, re.IGNORECASE | re.DOTALL):
                return response
        
        # Fallback
        return self.default_response
    
    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get all recorded calls."""
        return self._call_history.copy()
    
    def get_call_count(self) -> int:
        """Get number of calls made."""
        return len(self._call_history)
    
    def reset(self) -> None:
        """Reset call history and sequence index."""
        self._call_history.clear()
        self._sequence_index = 0
    
    def assert_called_with_pattern(self, pattern: str) -> bool:
        """Assert that at least one call matched the pattern."""
        for call in self._call_history:
            if re.search(pattern, call["prompt"], re.IGNORECASE | re.DOTALL):
                return True
        raise AssertionError(f"No call matched pattern: {pattern}")
