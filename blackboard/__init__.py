"""
Blackboard-Core SDK v1.1.0

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
- `from blackboard.sandbox import InsecureLocalExecutor, DockerSandbox`
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

__version__ = "1.1.0"

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
# ADVANCED FEATURES - Import from submodules
# =============================================================================
# Users should import advanced features directly from submodules:
#
# from blackboard.middleware import BudgetMiddleware, HumanApprovalMiddleware
# from blackboard.events import EventBus, Event, EventType
# from blackboard.retry import RetryPolicy, retry_with_backoff
# from blackboard.usage import UsageTracker, LLMResponse, LLMUsage
# from blackboard.tools import ToolDefinition, ToolCallingLLMClient
# from blackboard.memory import Memory, SimpleVectorMemory
# from blackboard.embeddings import EmbeddingModel, LocalEmbedder
# from blackboard.persistence import RedisPersistence, JSONFilePersistence
# from blackboard.hierarchy import SubOrchestratorWorker
# from blackboard.streaming import StreamingLLMClient
# from blackboard.vectordb import ChromaMemory
# from blackboard.evals import Evaluator, LLMJudge
# from blackboard.sandbox import DockerSandbox
