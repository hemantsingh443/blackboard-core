"""
Blackboard-Core SDK v1.1.0

A Python SDK implementing the Blackboard Pattern for LLM-powered multi-agent systems.

## Quick Start (30 seconds)

```python
from blackboard import Orchestrator, worker
from blackboard.llm import LiteLLMClient

@worker(name="Greeter", description="Says hello")
def greet(name: str = "World") -> str:
    return f"Hello, {name}!"

llm = LiteLLMClient(model="gpt-4o")  # Auto-detects OPENAI_API_KEY
orchestrator = Orchestrator(llm=llm, workers=[greet])
result = orchestrator.run_sync(goal="Greet the user")
```

## Namespace Organization

Core API (always stable):
- `from blackboard import Orchestrator, Worker, Blackboard, Artifact, Feedback`
- `from blackboard import worker, critic`  # Decorators

Advanced features (opt-in submodules):
- `from blackboard.llm import LiteLLMClient`
- `from blackboard.middleware import BudgetMiddleware, HumanApprovalMiddleware`
- `from blackboard.tui import BlackboardTUI, watch`
- `from blackboard.persistence import RedisPersistence, JSONFilePersistence`
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

# Functional worker decorators
from .decorators import (
    worker,
    critic,
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
    # Decorators (new!)
    "worker",
    "critic",
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
# from blackboard.llm import LiteLLMClient, create_llm
# from blackboard.decorators import worker, critic
# from blackboard.tui import BlackboardTUI, watch
# from blackboard.middleware import BudgetMiddleware, HumanApprovalMiddleware
# from blackboard.events import EventBus, Event, EventType
# from blackboard.usage import UsageTracker, LLMResponse, LLMUsage
# from blackboard.memory import Memory, SimpleVectorMemory
# from blackboard.persistence import RedisPersistence, JSONFilePersistence

