"""
Structured Logging for Blackboard SDK

Provides machine-readable JSON logs with automatic context injection
for session tracking, step correlation, and trace propagation.

Features:
- ContextVars for async-safe session/trace correlation
- Stdlib logging bridge (captures httpx, asyncpg, etc.)
- Dev mode (colored console) vs Prod mode (JSON)
- Automatic context binding for Orchestrator and Workers

Usage:
    from blackboard.logging import configure_logging, get_logger, bind_context

    # Configure at app startup
    configure_logging()

    # Get a logger
    log = get_logger("my_worker")
    log.info("worker_started", worker="search")

    # Bind context for a session
    with bind_context(session_id="123", step=1):
        log.info("processing")  # Includes session_id and step automatically
"""

import logging
import os
import sys
import uuid
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import structlog
from structlog.typing import EventDict, WrappedLogger

# =============================================================================
# Context Variables (async-safe correlation)
# =============================================================================

_session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_step_index: ContextVar[Optional[int]] = ContextVar("step_index", default=None)
_worker_name: ContextVar[Optional[str]] = ContextVar("worker_name", default=None)


def get_session_id() -> Optional[str]:
    """Get the current session ID from context."""
    return _session_id.get()


def get_trace_id() -> Optional[str]:
    """Get the current trace ID from context."""
    return _trace_id.get()


def get_step_index() -> Optional[int]:
    """Get the current step index from context."""
    return _step_index.get()


def get_worker_name() -> Optional[str]:
    """Get the current worker name from context."""
    return _worker_name.get()


def set_session_id(value: Optional[str]) -> None:
    """Set the session ID in context."""
    _session_id.set(value)


def set_trace_id(value: Optional[str]) -> None:
    """Set the trace ID in context."""
    _trace_id.set(value)


def set_step_index(value: Optional[int]) -> None:
    """Set the step index in context."""
    _step_index.set(value)


def set_worker_name(value: Optional[str]) -> None:
    """Set the worker name in context."""
    _worker_name.set(value)


def generate_trace_id() -> str:
    """Generate a new unique trace ID."""
    return str(uuid.uuid4())


def clear_context() -> None:
    """
    Clear all context variables.
    
    IMPORTANT: Call this at the end of each orchestrator run to prevent
    context leaking into subsequent operations (e.g., connection pools).
    """
    _session_id.set(None)
    _trace_id.set(None)
    _step_index.set(None)
    _worker_name.set(None)


@contextmanager
def bind_context(
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    step: Optional[int] = None,
    worker: Optional[str] = None
) -> Generator[None, None, None]:
    """
    Context manager for binding log context.
    
    Example:
        with bind_context(session_id="sess_123", step=2):
            log.info("processing")  # Logs include session_id and step
    """
    # Store old values
    old_session = _session_id.get()
    old_trace = _trace_id.get()
    old_step = _step_index.get()
    old_worker = _worker_name.get()
    
    try:
        # Set new values (only if provided)
        if session_id is not None:
            _session_id.set(session_id)
        if trace_id is not None:
            _trace_id.set(trace_id)
        if step is not None:
            _step_index.set(step)
        if worker is not None:
            _worker_name.set(worker)
        yield
    finally:
        # Restore old values
        _session_id.set(old_session)
        _trace_id.set(old_trace)
        _step_index.set(old_step)
        _worker_name.set(old_worker)


# =============================================================================
# Structlog Processors
# =============================================================================

def inject_context_vars(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """
    Processor that injects context variables into every log event.
    """
    # Inject context if present (don't clutter logs with None values)
    session_id = _session_id.get()
    if session_id:
        event_dict["session_id"] = session_id
    
    trace_id = _trace_id.get()
    if trace_id:
        event_dict["trace_id"] = trace_id
    
    step = _step_index.get()
    if step is not None:
        event_dict["step"] = step
    
    worker = _worker_name.get()
    if worker:
        event_dict["worker"] = worker
    
    return event_dict


# =============================================================================
# Configuration
# =============================================================================

_configured = False


def configure_logging(
    level: str = "INFO",
    force_json: bool = False,
    force_console: bool = False
) -> None:
    """
    Configure structlog for the Blackboard SDK.
    
    Auto-detects dev vs prod environment:
    - Dev (TTY): Colored, pretty-printed console output
    - Prod (non-TTY, Docker, etc.): JSON output
    
    Can be overridden via:
    - BLACKBOARD_LOG_FORMAT env var ("json" or "console")
    - force_json/force_console parameters
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        force_json: Force JSON output regardless of environment
        force_console: Force console output regardless of environment
    """
    global _configured
    
    # Determine output format
    log_format = os.environ.get("BLACKBOARD_LOG_FORMAT", "").lower()
    
    if force_json:
        use_json = True
    elif force_console:
        use_json = False
    elif log_format == "json":
        use_json = True
    elif log_format == "console":
        use_json = False
    else:
        # Auto-detect: use console for TTY, JSON otherwise
        use_json = not sys.stderr.isatty()
    
    # Shared processors for both formats
    # Note: add_log_level/add_logger_name are stdlib-specific and not needed
    # for native structlog loggers (level is added by ConsoleRenderer/JSONRenderer)
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        inject_context_vars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if use_json:
        # Production: JSON output
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    else:
        # Development: Colored console output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.rich_traceback
                if _rich_available() else structlog.dev.plain_traceback
            )
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Bridge stdlib logging to structlog
    _configure_stdlib_bridge(level)
    
    _configured = True


def _rich_available() -> bool:
    """Check if rich is available for enhanced tracebacks."""
    try:
        import rich  # noqa
        return True
    except ImportError:
        return False


def _configure_stdlib_bridge(level: str) -> None:
    """
    Configure stdlib logging to be captured and formatted by structlog.
    
    This ensures logs from httpx, asyncpg, and other libraries appear
    in the same format as Blackboard logs.
    """
    # Determine if JSON or console output
    use_json = not sys.stderr.isatty()
    log_format = os.environ.get("BLACKBOARD_LOG_FORMAT", "").lower()
    if log_format == "json":
        use_json = True
    elif log_format == "console":
        use_json = False
    
    # Create processor chain for stdlib logs
    pre_chain = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        inject_context_vars,
    ]
    
    if use_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    
    # Create formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=pre_chain + [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ]
    )
    
    # Configure root logger
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))


# =============================================================================
# Logger Factory
# =============================================================================

def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a structlog logger.
    
    Args:
        name: Logger name (e.g., "blackboard.core", "my_worker")
        
    Returns:
        A bound logger with context injection
        
    Example:
        log = get_logger("my_module")
        log.info("event_name", key="value")
    """
    global _configured
    if not _configured:
        configure_logging()
    
    return structlog.get_logger(name)


# =============================================================================
# Convenience Functions
# =============================================================================

def log_llm_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    duration_ms: float,
    success: bool = True,
    error: Optional[str] = None
) -> None:
    """
    Log an LLM call with standard attributes (useful for cost tracking).
    
    Uses OpenTelemetry GenAI semantic conventions for attribute names.
    """
    log = get_logger("blackboard.llm")
    
    event_data: Dict[str, Any] = {
        "gen_ai.request.model": model,
        "gen_ai.usage.input_tokens": input_tokens,
        "gen_ai.usage.output_tokens": output_tokens,
        "duration_ms": duration_ms,
        "success": success,
    }
    
    if error:
        event_data["error"] = error
        log.error("llm_call_failed", **event_data)
    else:
        log.info("llm_call_completed", **event_data)


def log_worker_execution(
    worker_name: str,
    duration_ms: float,
    success: bool = True,
    error: Optional[str] = None
) -> None:
    """Log a worker execution event."""
    log = get_logger("blackboard.worker")
    
    with bind_context(worker=worker_name):
        if success:
            log.info("worker_completed", duration_ms=duration_ms)
        else:
            log.error("worker_failed", duration_ms=duration_ms, error=error)
