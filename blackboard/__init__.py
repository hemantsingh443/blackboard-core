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

# Decorators
from .decorators import (
    worker,
    critic,
)

__version__ = "1.5.1"

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
    # Configuration
    "BlackboardConfig",
    # Persistence
    "SQLitePersistence",
    "InMemoryPersistence",
    "PersistenceError",
    "SessionNotFoundError",
    # Version
    "__version__",
]
