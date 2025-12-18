"""
Usage Tracking for LLM Calls

Provides token counting and cost tracking for LLM API usage.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


@dataclass
class LLMUsage:
    """
    Token usage statistics from an LLM call.
    
    Attributes:
        input_tokens: Number of tokens in the prompt
        output_tokens: Number of tokens in the response
        total_tokens: Total tokens used (input + output)
        model: The model used for this call
        latency_ms: Time taken for the call in milliseconds
    """
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0
    
    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    """
    Structured response from an LLM call.
    
    Use this instead of returning raw strings to enable usage tracking.
    
    Example:
        class MyLLMClient:
            def generate(self, prompt: str) -> LLMResponse:
                response = self.client.chat.completions.create(...)
                return LLMResponse(
                    content=response.choices[0].message.content,
                    usage=LLMUsage(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        model=response.model
                    )
                )
    """
    content: str
    usage: Optional[LLMUsage] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Convenience property to get content as string
    def __str__(self) -> str:
        return self.content


@dataclass
class UsageRecord:
    """A single usage record for tracking."""
    timestamp: datetime
    usage: LLMUsage
    cost: float = 0.0
    context: str = ""  # e.g., "supervisor", "worker:Writer"


class UsageTracker:
    """
    Tracks and aggregates LLM usage across a session.
    
    Args:
        cost_per_1k_input: Default cost per 1000 input tokens
        cost_per_1k_output: Default cost per 1000 output tokens
        model_costs: Dict of model-specific costs
        max_records: Maximum records to keep (oldest evicted when exceeded)
        on_flush: Optional callback when records are evicted: fn(evicted_records)
    
    Example:
        tracker = UsageTracker(
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            max_records=1000,
            on_flush=lambda records: save_to_db(records)
        )
        
        # Record usage
        tracker.record(usage, context="supervisor")
        
        # Get totals
        print(f"Total cost: ${tracker.total_cost:.4f}")
        print(f"Total tokens: {tracker.total_tokens}")
    """
    
    def __init__(
        self,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0,
        model_costs: Optional[Dict[str, Dict[str, float]]] = None,
        max_records: Optional[int] = 10000,
        on_flush: Optional[Callable[[List["UsageRecord"]], None]] = None
    ):
        """
        Initialize the usage tracker.
        
        Args:
            cost_per_1k_input: Default cost per 1000 input tokens
            cost_per_1k_output: Default cost per 1000 output tokens
            model_costs: Optional dict of model-specific costs:
                {"gpt-4": {"input": 0.03, "output": 0.06}}
            max_records: Max records before eviction (None = unlimited)
            on_flush: Callback when records are evicted
        """
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        self.model_costs = model_costs or {}
        self.max_records = max_records
        self.on_flush = on_flush
        
        self._records: List[UsageRecord] = []
        # Aggregate counters for evicted records
        self._evicted_tokens: int = 0
        self._evicted_cost: float = 0.0
        self._evicted_count: int = 0
    
    def record(self, usage: LLMUsage, context: str = "") -> float:
        """
        Record an LLM usage event.
        
        Args:
            usage: The usage statistics
            context: Context string (e.g., "supervisor", "worker:Name")
            
        Returns:
            The cost of this usage
        """
        cost = self._calculate_cost(usage)
        
        record = UsageRecord(
            timestamp=datetime.now(),
            usage=usage,
            cost=cost,
            context=context
        )
        self._records.append(record)
        
        # Evict oldest records if over limit
        if self.max_records is not None and len(self._records) > self.max_records:
            evict_count = len(self._records) - self.max_records
            evicted = self._records[:evict_count]
            self._records = self._records[evict_count:]
            
            # Aggregate evicted stats
            for r in evicted:
                self._evicted_tokens += r.usage.total_tokens
                self._evicted_cost += r.cost
                self._evicted_count += 1
            
            # Notify callback if provided
            if self.on_flush:
                self.on_flush(evicted)
        
        return cost
    
    def _calculate_cost(self, usage: LLMUsage) -> float:
        """Calculate cost for a usage record."""
        # Check for model-specific costs
        if usage.model and usage.model in self.model_costs:
            model_pricing = self.model_costs[usage.model]
            input_cost = (usage.input_tokens / 1000) * model_pricing.get("input", 0)
            output_cost = (usage.output_tokens / 1000) * model_pricing.get("output", 0)
        else:
            input_cost = (usage.input_tokens / 1000) * self.cost_per_1k_input
            output_cost = (usage.output_tokens / 1000) * self.cost_per_1k_output
        
        return input_cost + output_cost
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used across all records (including evicted)."""
        return sum(r.usage.total_tokens for r in self._records) + self._evicted_tokens
    
    @property
    def total_input_tokens(self) -> int:
        """Total input tokens used (current records only)."""
        return sum(r.usage.input_tokens for r in self._records)
    
    @property
    def total_output_tokens(self) -> int:
        """Total output tokens used (current records only)."""
        return sum(r.usage.output_tokens for r in self._records)
    
    @property
    def total_cost(self) -> float:
        """Total cost across all records (including evicted)."""
        return sum(r.cost for r in self._records) + self._evicted_cost
    
    @property
    def call_count(self) -> int:
        """Total number of LLM calls (including evicted)."""
        return len(self._records) + self._evicted_count
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of usage statistics."""
        return {
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_cost": round(self.total_cost, 6),
            "avg_tokens_per_call": (
                self.total_tokens / self.call_count if self.call_count > 0 else 0
            ),
            "avg_latency_ms": (
                sum(r.usage.latency_ms for r in self._records) / self.call_count
                if self.call_count > 0 else 0
            )
        }
    
    def get_by_context(self, context: str) -> List[UsageRecord]:
        """Get records filtered by context."""
        return [r for r in self._records if context in r.context]
    
    def get_context_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get usage summary grouped by context."""
        contexts: Dict[str, List[UsageRecord]] = {}
        
        for record in self._records:
            ctx = record.context or "unknown"
            if ctx not in contexts:
                contexts[ctx] = []
            contexts[ctx].append(record)
        
        summary = {}
        for ctx, records in contexts.items():
            summary[ctx] = {
                "call_count": len(records),
                "total_tokens": sum(r.usage.total_tokens for r in records),
                "total_cost": round(sum(r.cost for r in records), 6)
            }
        
        return summary
    
    def reset(self) -> None:
        """Clear all usage records."""
        self._records.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export usage data as dictionary."""
        return {
            "summary": self.get_summary(),
            "by_context": self.get_context_summary(),
            "records": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "context": r.context,
                    "input_tokens": r.usage.input_tokens,
                    "output_tokens": r.usage.output_tokens,
                    "cost": r.cost,
                    "model": r.usage.model
                }
                for r in self._records
            ]
        }

