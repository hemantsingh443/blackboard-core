"""Blackboard-Core SDK - Multi-agent orchestration with the Blackboard Pattern."""

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
    LLMResponse,
    LLMUsage,
    run_blackboard,
    run_blackboard_sync,
    Agent,
    RecursionDepthExceededError,
)

# Configuration
from .config import BlackboardConfig

# Persistence
from .persistence import (
    SQLitePersistence,
    InMemoryPersistence,
    PersistenceError,
    SessionNotFoundError,
)

# Runtime
from .runtime import (
    LocalRuntime,
    DockerRuntime,
    Runtime,
    ExecutionResult,
    RuntimeSecurityError,
)

# Decorators
from .decorators import (
    worker,
    critic,
)

__version__ = "1.5.3"

__all__ = [
    # State
    "Blackboard",
    "Artifact",
    "Feedback",
    "Status",
    "StateConflictError",
    # Worker
    "Worker",
    "WorkerOutput",
    "WorkerInput",
    "WorkerRegistry",
    # Decorators
    "worker",
    "critic",
    # Orchestrator
    "Orchestrator",
    "LLMClient",
    "LLMResponse",
    "LLMUsage",
    "run_blackboard",
    "run_blackboard_sync",
    # Fractal Agents
    "Agent",
    "RecursionDepthExceededError",
    # Configuration
    "BlackboardConfig",
    # Persistence
    "SQLitePersistence",
    "InMemoryPersistence",
    "PersistenceError",
    "SessionNotFoundError",
    # Runtime
    "LocalRuntime",
    "DockerRuntime",
    "Runtime",
    "ExecutionResult",
    "RuntimeSecurityError",
    # Version
    "__version__",
]
