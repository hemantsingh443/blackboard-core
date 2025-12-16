"""
Blackboard-Core SDK v1.0.3

A Python SDK implementing the Blackboard Pattern for LLM-powered multi-agent systems.

## Quick Start

```python
from blackboard import Orchestrator, Worker, WorkerOutput, Artifact, Blackboard

class Writer(Worker):
    name = "Writer"
    description = "Generates content"
    
    async def run(self, state, inputs=None):
        return WorkerOutput(
            artifact=Artifact(type="text", content="Hello!", creator=self.name)
        )

orchestrator = Orchestrator(llm=my_llm, workers=[Writer()])
result = await orchestrator.run(goal="Write a greeting")
```

## Namespace Organization

Core API (always stable):
- `from blackboard import Orchestrator, Worker, Blackboard, Artifact, Feedback`

Advanced features (opt-in submodules):
- `from blackboard.middleware import BudgetMiddleware, HumanApprovalMiddleware`
- `from blackboard.persistence import RedisPersistence, JSONFilePersistence`
- `from blackboard.hierarchy import SubOrchestratorWorker`
- `from blackboard.streaming import StreamingLLMClient`
- `from blackboard.vectordb import ChromaMemory`
- `from blackboard.evals import Evaluator, LLMJudge`
- `from blackboard.sandbox import SubprocessSandbox`
"""

# =============================================================================
# CORE API - The essential, stable public interface
# =============================================================================

# State models
from .state import (
    Blackboard,
    Artifact,
    Feedback,
    Status,
    StateConflictError,
)

# Worker protocol
from .protocols import (
    Worker,
    WorkerOutput,
    WorkerInput,
    WorkerRegistry,
)

# Orchestrator
from .core import (
    Orchestrator,
    LLMClient,
    run_blackboard,
    run_blackboard_sync,
)

# =============================================================================
# VERSION
# =============================================================================

__version__ = "1.0.3"

# =============================================================================
# CORE PUBLIC API (__all__)
# Only the most essential items - users import advanced features from submodules
# =============================================================================

__all__ = [
    # State (stable)
    "Blackboard",
    "Artifact",
    "Feedback",
    "Status",
    "StateConflictError",
    # Worker (stable)
    "Worker",
    "WorkerOutput",
    "WorkerInput",
    "WorkerRegistry",
    # Orchestrator (stable)
    "Orchestrator",
    "LLMClient",
    "run_blackboard",
    "run_blackboard_sync",
    # Version
    "__version__",
]

# =============================================================================
# CONVENIENCE RE-EXPORTS (for backward compatibility)
# Users should migrate to submodule imports for advanced features
# =============================================================================

# These are available at the top level but NOT in __all__
# This way they work but don't pollute IDE autocomplete

# Middleware (prefer: from blackboard.middleware import ...)
from .middleware import (
    Middleware,
    MiddlewareStack,
    StepContext,
    WorkerContext,
    BudgetMiddleware,
    LoggingMiddleware,
    HumanApprovalMiddleware,
    AutoSummarizationMiddleware,
    ApprovalRequired,
)

# Events (prefer: from blackboard.events import ...)
from .events import (
    EventBus,
    Event,
    EventType,
    get_event_bus,
    reset_event_bus,
)

# Retry (prefer: from blackboard.retry import ...)
from .retry import (
    RetryPolicy,
    retry_with_backoff,
    DEFAULT_RETRY_POLICY,
    NO_RETRY,
    is_transient_error,
)

# Usage tracking (prefer: from blackboard.usage import ...)
from .usage import (
    UsageTracker,
    UsageRecord,
    LLMResponse,
    LLMUsage,
    create_openai_tracker,
)

# Tool calling (prefer: from blackboard.tools import ...)
from .tools import (
    ToolDefinition,
    ToolParameter,
    ToolCall,
    ToolCallResponse,
    ToolCallingLLMClient,
    worker_to_tool_definition,
    workers_to_tool_definitions,
    DONE_TOOL,
    FAIL_TOOL,
)

# Memory (prefer: from blackboard.memory import ...)
from .memory import (
    Memory,
    MemoryEntry,
    SearchResult,
    SimpleVectorMemory,
    MemoryWorker,
    MemoryInput,
)

# Embeddings (prefer: from blackboard.embeddings import ...)
from .embeddings import (
    EmbeddingModel,
    NoOpEmbedder,
    TFIDFEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    cosine_similarity,
    get_default_embedder,
)

# Internal re-exports for advanced usage
from .core import SupervisorDecision, WorkerCall, LLMResult
