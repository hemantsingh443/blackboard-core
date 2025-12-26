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
    PersistenceLayer,
    SQLitePersistence,
    RedisPersistence,
    InMemoryPersistence,
    PersistenceError,
    SessionNotFoundError,
    SessionConflictError,
)

# Postgres is optional - requires asyncpg
try:
    from .persistence import PostgresPersistence
except ImportError:
    PostgresPersistence = None  # type: ignore

# Runtime
from .runtime import (
    LocalRuntime,  # Deprecated alias
    InsecureLocalRuntime,
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

# Squad Patterns
from .patterns import (
    create_squad,
    research_squad,
    code_squad,
    memory_squad,
    Squad,
)

# Logging (v1.6.2)
from .logging import (
    get_logger,
    configure_logging,
    bind_context,
    set_session_id,
    clear_context,
)

# Pricing (v1.6.2)
from .pricing import (
    get_model_cost,
    estimate_cost,
    BudgetExceededError,
    configure_pricing,
)

# Prompt Registry (v1.6.3)
from .prompts import (
    PromptRegistry,
    create_default_prompts_dir,
    create_default_config,
)

# Optimizer (v1.6.3)
from .optimize import (
    Optimizer,
    PromptPatch,
    run_optimization,
)

__version__ = "1.6.3"

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
    "PersistenceLayer",
    "SQLitePersistence",
    "RedisPersistence",
    "PostgresPersistence",
    "InMemoryPersistence",
    "PersistenceError",
    "SessionNotFoundError",
    "SessionConflictError",
    # Runtime
    "LocalRuntime",
    "DockerRuntime",
    "Runtime",
    "ExecutionResult",
    "RuntimeSecurityError",
    # Squad Patterns
    "create_squad",
    "research_squad",
    "code_squad",
    "memory_squad",
    "Squad",
    # Logging (v1.6.2)
    "get_logger",
    "configure_logging",
    "bind_context",
    "set_session_id",
    "clear_context",
    # Pricing (v1.6.2)
    "get_model_cost",
    "estimate_cost",
    "BudgetExceededError",
    "configure_pricing",
    # Prompt Registry (v1.6.3)
    "PromptRegistry",
    "create_default_prompts_dir",
    "create_default_config",
    # Optimizer (v1.6.3)
    "Optimizer",
    "PromptPatch",
    "run_optimization",
    # Version
    "__version__",
]
