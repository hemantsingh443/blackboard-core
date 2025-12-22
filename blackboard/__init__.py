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

# Decorators
from .decorators import (
    worker,
    critic,
)

__version__ = "1.5.0"

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
    # Version
    "__version__",
]
