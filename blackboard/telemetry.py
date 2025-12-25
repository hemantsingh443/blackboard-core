"""
OpenTelemetry Integration for Blackboard Orchestrator

Provides distributed tracing and metrics for observability.
Integrates with any OTEL-compatible backend (Jaeger, Zipkin, Honeycomb, etc).

Install: pip install 'blackboard-core[telemetry]'

Example:
    from blackboard.telemetry import OpenTelemetryMiddleware
    
    # With default tracer
    otel = OpenTelemetryMiddleware(service_name="my-agent")
    orchestrator = Orchestrator(llm=llm, workers=workers, middleware=[otel])
    
    # With custom tracer
    from opentelemetry import trace
    tracer = trace.get_tracer("my-service")
    otel = OpenTelemetryMiddleware(tracer=tracer)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, TYPE_CHECKING, runtime_checkable
from contextlib import contextmanager

from .middleware import Middleware, StepContext, WorkerContext
from .events import Event, EventBus, EventType

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer, Span

logger = logging.getLogger("blackboard.telemetry")


# =============================================================================
# TraceExporter Protocol (Pluggable Exporters)
# =============================================================================

@dataclass
class LLMSpan:
    """
    Represents a single LLM call span for export.
    
    Contains all information about an LLM invocation in a format
    suitable for export to any tracing backend.
    """
    model: str
    system: str  # e.g., "openai", "anthropic", "google"
    input_tokens: int
    output_tokens: int
    duration_ms: float
    cost: float = 0.0
    
    # Optional: full prompt/completion (may be redacted)
    prompt: Optional[str] = None
    completion: Optional[str] = None
    
    # Context IDs for correlation
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    step_index: Optional[int] = None
    
    # Metadata
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    error: Optional[str] = None


@runtime_checkable
class TraceExporter(Protocol):
    """
    Protocol for pluggable trace exporters.
    
    Implement this to export spans to custom observability backends
    like LangSmith, Arize Phoenix, or Helicone.
    
    Example:
        class LangSmithExporter:
            async def export_llm_span(self, span: LLMSpan) -> None:
                langsmith_client.log_run(
                    model=span.model,
                    inputs={"prompt": span.prompt},
                    outputs={"completion": span.completion},
                )
    """
    
    async def export_llm_span(self, span: LLMSpan) -> None:
        """Export an LLM call span."""
        ...


# =============================================================================
# Redaction Support
# =============================================================================

import threading

# Thread-safe redaction callback
_redaction_lock = threading.Lock()
_redaction_callback: Optional[Callable[[str], str]] = None


def configure_redaction(callback: Callable[[str], str]) -> None:
    """
    Configure a redaction callback for PII scrubbing.
    
    The callback receives raw text (prompts/completions) and should
    return the text with sensitive data redacted.
    
    Thread-safe: can be called from any thread.
    
    Example:
        import re
        
        def redact_pii(text: str) -> str:
            # Redact SSNs
            text = re.sub(r'\\d{3}-\\d{2}-\\d{4}', '[SSN REDACTED]', text)
            # Redact API keys (AWS, OpenAI, etc.)
            text = re.sub(r'sk-[a-zA-Z0-9]{48}', '[API_KEY REDACTED]', text)
            return text
        
        configure_redaction(redact_pii)
    """
    global _redaction_callback
    with _redaction_lock:
        _redaction_callback = callback


def redact_text(text: Optional[str]) -> Optional[str]:
    """Apply redaction callback if configured. Thread-safe."""
    if text is None:
        return None
    with _redaction_lock:
        callback = _redaction_callback
    if callback is not None:
        return callback(text)
    return text


# =============================================================================
# Telemetry Data Classes
# =============================================================================

@dataclass
class SpanAttributes:
    """
    Standard span attributes for blackboard operations.
    
    Follows OpenTelemetry Semantic Conventions for GenAI:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/
    """
    # Orchestrator attributes
    GOAL = "blackboard.goal"
    SESSION_ID = "blackboard.session_id"
    MAX_STEPS = "blackboard.max_steps"
    
    # Step attributes
    STEP_NUMBER = "blackboard.step.number"
    STEP_ACTION = "blackboard.step.action"
    STEP_REASONING = "blackboard.step.reasoning"
    
    # Worker attributes
    WORKER_NAME = "blackboard.worker.name"
    WORKER_INSTRUCTIONS = "blackboard.worker.instructions"
    WORKER_PARALLEL = "blackboard.worker.parallel"
    
    # GenAI Semantic Conventions (OTEL standard)
    GEN_AI_SYSTEM = "gen_ai.system"                     # e.g., "openai", "anthropic"
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"       # e.g., "gpt-4o"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"     # actual model used
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
    GEN_AI_REQUEST_PROMPT = "gen_ai.request.prompt"     # Optional: full prompt (redactable)
    GEN_AI_RESPONSE_COMPLETION = "gen_ai.response.completion"  # Optional: full response (redactable)
    
    # Legacy LLM attributes (kept for backward compat)
    LLM_MODEL = "llm.model"
    LLM_INPUT_TOKENS = "llm.tokens.input"
    LLM_OUTPUT_TOKENS = "llm.tokens.output"
    LLM_COST = "llm.cost"
    
    # Result attributes
    ARTIFACT_TYPE = "blackboard.artifact.type"
    ARTIFACT_CREATOR = "blackboard.artifact.creator"
    STATUS = "blackboard.status"
    ERROR = "blackboard.error"


@dataclass
class TelemetryMetrics:
    """Aggregated metrics for a session."""
    total_steps: int = 0
    total_workers_called: int = 0
    total_artifacts: int = 0
    total_errors: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "total_workers_called": self.total_workers_called,
            "total_artifacts": self.total_artifacts,
            "total_errors": self.total_errors,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "duration_ms": self.duration_ms
        }


# =============================================================================
# OpenTelemetry Middleware
# =============================================================================

class OpenTelemetryMiddleware(Middleware):
    """
    Distributed tracing middleware using OpenTelemetry.
    
    Creates hierarchical spans for:
    - Orchestrator session (root span)
    - Each step (child of session)
    - Each worker call (child of step)
    
    Attributes:
        service_name: Name for the tracing service
        tracer: Optional custom OTel tracer
        record_inputs: Whether to record full input content
        record_outputs: Whether to record full output content
        
    Example:
        # Basic usage
        otel = OpenTelemetryMiddleware(service_name="my-agent")
        orchestrator = Orchestrator(..., middleware=[otel])
        
        # With Jaeger exporter
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(JaegerExporter()))
        trace.set_tracer_provider(provider)
        
        otel = OpenTelemetryMiddleware(service_name="my-agent")
    """
    
    name = "OpenTelemetryMiddleware"
    
    def __init__(
        self,
        service_name: str = "blackboard-agent",
        tracer: Optional["Tracer"] = None,
        record_inputs: bool = True,
        record_outputs: bool = True,
        max_attribute_length: int = 1024
    ):
        self.service_name = service_name
        self.record_inputs = record_inputs
        self.record_outputs = record_outputs
        self.max_attribute_length = max_attribute_length
        
        # Initialize tracer
        self._tracer = tracer
        self._otel_available = False
        self._init_tracer()
        
        # Span tracking
        self._session_span: Optional["Span"] = None
        self._step_spans: Dict[int, "Span"] = {}
        self._worker_spans: Dict[str, "Span"] = {}
        
        # Metrics
        self.metrics = TelemetryMetrics()
        self._session_start_time: float = 0.0
    
    def _init_tracer(self) -> None:
        """Initialize OpenTelemetry tracer if available."""
        if self._tracer is not None:
            self._otel_available = True
            return
            
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource
            
            # Check if a global provider is already set
            provider = trace.get_tracer_provider()
            
            # If no provider is configured, set up a default one
            if isinstance(provider, trace.ProxyTracerProvider):
                resource = Resource.create({"service.name": self.service_name})
                provider = TracerProvider(resource=resource)
                trace.set_tracer_provider(provider)
            
            self._tracer = trace.get_tracer(self.service_name)
            self._otel_available = True
            logger.info(f"OpenTelemetry initialized for {self.service_name}")
            
        except ImportError:
            logger.warning(
                "OpenTelemetry not installed. Install with: "
                "pip install 'blackboard-core[telemetry]'"
            )
            self._otel_available = False
    
    def _truncate(self, value: Any) -> str:
        """Truncate value for span attributes."""
        s = str(value)
        if len(s) > self.max_attribute_length:
            return s[:self.max_attribute_length] + "..."
        return s
    
    def _safe_set_attribute(self, span: "Span", key: str, value: Any) -> None:
        """Safely set span attribute with truncation."""
        if span is None or value is None:
            return
        try:
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(key, value if not isinstance(value, str) 
                                  else self._truncate(value))
            else:
                span.set_attribute(key, self._truncate(value))
        except Exception as e:
            logger.debug(f"Failed to set attribute {key}: {e}")
    
    # =========================================================================
    # Session Lifecycle
    # =========================================================================
    
    def start_session(
        self,
        goal: str,
        session_id: Optional[str] = None,
        max_steps: int = 20
    ) -> None:
        """
        Start a new tracing session.
        
        Called automatically by the Orchestrator at the start of run().
        """
        if not self._otel_available:
            return
            
        self._session_start_time = time.time()
        self.metrics = TelemetryMetrics()
        
        self._session_span = self._tracer.start_span(
            name=f"orchestrator.run",
            attributes={
                SpanAttributes.GOAL: self._truncate(goal),
                SpanAttributes.SESSION_ID: session_id or "unknown",
                SpanAttributes.MAX_STEPS: max_steps,
                "service.name": self.service_name
            }
        )
        logger.debug(f"Started session span for goal: {goal[:50]}...")
    
    def end_session(self, status: str, error: Optional[str] = None) -> None:
        """End the current tracing session."""
        if not self._otel_available or self._session_span is None:
            return
            
        # Calculate duration
        self.metrics.duration_ms = (time.time() - self._session_start_time) * 1000
        
        # Record final metrics
        self._safe_set_attribute(
            self._session_span, SpanAttributes.STATUS, status
        )
        self._safe_set_attribute(
            self._session_span, "metrics.total_steps", self.metrics.total_steps
        )
        self._safe_set_attribute(
            self._session_span, "metrics.total_workers", self.metrics.total_workers_called
        )
        self._safe_set_attribute(
            self._session_span, "metrics.total_tokens", 
            self.metrics.total_input_tokens + self.metrics.total_output_tokens
        )
        self._safe_set_attribute(
            self._session_span, "metrics.duration_ms", self.metrics.duration_ms
        )
        
        if error:
            from opentelemetry.trace import StatusCode
            self._session_span.set_status(StatusCode.ERROR, error)
            self._safe_set_attribute(self._session_span, SpanAttributes.ERROR, error)
        
        self._session_span.end()
        self._session_span = None
        logger.debug("Ended session span")
    
    # =========================================================================
    # Middleware Hooks
    # =========================================================================
    
    async def before_step(self, ctx: StepContext) -> None:
        """Create a span for the orchestration step."""
        if not self._otel_available:
            return
            
        step_span = self._tracer.start_span(
            name=f"step.{ctx.step_number}",
            context=self._get_session_context()
        )
        
        self._safe_set_attribute(step_span, SpanAttributes.STEP_NUMBER, ctx.step_number)
        self._step_spans[ctx.step_number] = step_span
    
    async def after_step(self, ctx: StepContext) -> None:
        """End the step span and record decision."""
        if not self._otel_available:
            return
            
        step_span = self._step_spans.pop(ctx.step_number, None)
        if step_span is None:
            return
        
        self.metrics.total_steps += 1
        
        if ctx.decision:
            self._safe_set_attribute(
                step_span, SpanAttributes.STEP_ACTION, ctx.decision.action
            )
            self._safe_set_attribute(
                step_span, SpanAttributes.STEP_REASONING, ctx.decision.reasoning
            )
        
        step_span.end()
    
    async def before_worker(self, ctx: WorkerContext) -> None:
        """Create a span for worker execution."""
        if not self._otel_available:
            return
            
        worker_key = f"{ctx.worker.name}_{id(ctx.call)}"
        worker_span = self._tracer.start_span(
            name=f"worker.{ctx.worker.name}",
            context=self._get_step_context(ctx)
        )
        
        self._safe_set_attribute(worker_span, SpanAttributes.WORKER_NAME, ctx.worker.name)
        
        if self.record_inputs and ctx.call:
            self._safe_set_attribute(
                worker_span, SpanAttributes.WORKER_INSTRUCTIONS, 
                getattr(ctx.call, 'instructions', '')
            )
        
        self._worker_spans[worker_key] = worker_span
    
    async def after_worker(self, ctx: WorkerContext) -> None:
        """End worker span and record output."""
        if not self._otel_available:
            return
        
        worker_key = f"{ctx.worker.name}_{id(ctx.call)}"
        worker_span = self._worker_spans.pop(worker_key, None)
        if worker_span is None:
            return
        
        self.metrics.total_workers_called += 1
        
        # Record output metadata
        if ctx.modified_output:
            output = ctx.modified_output
            if output.artifact:
                self.metrics.total_artifacts += 1
                self._safe_set_attribute(
                    worker_span, SpanAttributes.ARTIFACT_TYPE, output.artifact.type
                )
                self._safe_set_attribute(
                    worker_span, SpanAttributes.ARTIFACT_CREATOR, output.artifact.creator
                )
        
        worker_span.end()
    
    async def on_error(self, ctx: WorkerContext) -> None:
        """Record error in current span."""
        if not self._otel_available:
            return
        
        self.metrics.total_errors += 1
        
        worker_key = f"{ctx.worker.name}_{id(ctx.call)}"
        worker_span = self._worker_spans.get(worker_key)
        
        if worker_span and ctx.error:
            from opentelemetry.trace import StatusCode
            worker_span.set_status(StatusCode.ERROR, str(ctx.error))
            worker_span.record_exception(ctx.error)
    
    # =========================================================================
    # Context Helpers
    # =========================================================================
    
    def _get_session_context(self):
        """Get trace context from session span."""
        if self._session_span is None:
            return None
        try:
            from opentelemetry import trace
            return trace.set_span_in_context(self._session_span)
        except Exception:
            return None
    
    def _get_step_context(self, ctx: WorkerContext):
        """Get trace context from current step span."""
        step_span = self._step_spans.get(ctx.state.step_count)
        if step_span is None:
            return self._get_session_context()
        try:
            from opentelemetry import trace
            return trace.set_span_in_context(step_span)
        except Exception:
            return self._get_session_context()
    
    # =========================================================================
    # LLM Instrumentation
    # =========================================================================
    
    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float = 0.0,
        duration_ms: float = 0.0
    ) -> None:
        """
        Record LLM call metrics.
        
        Call this after each LLM invocation to track usage.
        """
        self.metrics.total_input_tokens += input_tokens
        self.metrics.total_output_tokens += output_tokens
        self.metrics.total_cost += cost
        
        if not self._otel_available or self._session_span is None:
            return
        
        # Create a span for the LLM call if we have a step context
        current_step = max(self._step_spans.keys()) if self._step_spans else None
        parent_context = (
            self._get_session_context() if current_step is None
            else self._get_step_context(type('obj', (object,), {'state': type('obj', (object,), {'step_count': current_step})()})())
        )
        
        with self._tracer.start_span("llm.call", context=parent_context) as span:
            self._safe_set_attribute(span, SpanAttributes.LLM_MODEL, model)
            self._safe_set_attribute(span, SpanAttributes.LLM_INPUT_TOKENS, input_tokens)
            self._safe_set_attribute(span, SpanAttributes.LLM_OUTPUT_TOKENS, output_tokens)
            self._safe_set_attribute(span, SpanAttributes.LLM_COST, cost)
            self._safe_set_attribute(span, "llm.duration_ms", duration_ms)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_metrics(self) -> TelemetryMetrics:
        """Get current session metrics."""
        return self.metrics
    
    def reset(self) -> None:
        """Reset all tracking state."""
        self._session_span = None
        self._step_spans.clear()
        self._worker_spans.clear()
        self.metrics = TelemetryMetrics()
        self._session_start_time = 0.0


# =============================================================================
# Metrics Exporter (for non-OTEL use cases)
# =============================================================================

class MetricsCollector:
    """
    Simple metrics collector for environments without full OpenTelemetry.
    
    Collects timing and usage metrics that can be exported to any backend.
    
    Example:
        collector = MetricsCollector()
        orchestrator.event_bus.subscribe_all(collector.on_event)
        
        result = await orchestrator.run(goal="...")
        print(collector.get_summary())
    """
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
    def on_event(self, event: Event) -> None:
        """Record an event."""
        if event.type == EventType.ORCHESTRATOR_STARTED:
            self.start_time = time.time()
        elif event.type == EventType.ORCHESTRATOR_COMPLETED:
            self.end_time = time.time()
            
        self.events.append({
            "type": event.type.value,
            "timestamp": event.timestamp.isoformat(),
            "data": event.data
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated metrics summary."""
        duration = (self.end_time - self.start_time) if self.start_time and self.end_time else 0
        
        step_events = [e for e in self.events if "step" in e["type"].lower()]
        worker_events = [e for e in self.events if "worker" in e["type"].lower()]
        error_events = [e for e in self.events if "error" in e["type"].lower()]
        
        return {
            "duration_seconds": duration,
            "total_events": len(self.events),
            "total_steps": len([e for e in step_events if "completed" in e["type"]]),
            "total_worker_calls": len([e for e in worker_events if "completed" in e["type"]]),
            "total_errors": len(error_events),
            "events_by_type": self._count_by_type()
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count events by type."""
        counts: Dict[str, int] = {}
        for event in self.events:
            counts[event["type"]] = counts.get(event["type"], 0) + 1
        return counts
    
    def reset(self) -> None:
        """Reset collector."""
        self.events.clear()
        self.start_time = None
        self.end_time = None
