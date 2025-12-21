"""
Blackboard-Core SDK

A Python SDK implementing the Blackboard Pattern for LLM-powered multi-agent systems.

## Quick Start

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

Core API:
- `from blackboard import Orchestrator, Worker, Blackboard, Artifact, Feedback`
- `from blackboard import worker, critic`  # Decorators

Advanced features (opt-in submodules):
- `from blackboard.llm import LiteLLMClient`
- `from blackboard.mcp import MCPServerWorker, MCPRegistry`
- `from blackboard.telemetry import OpenTelemetryMiddleware`
- `from blackboard.replay import SessionRecorder, ReplayOrchestrator`
- `from blackboard.middleware import BudgetMiddleware, HumanApprovalMiddleware`
- `from blackboard.tui import BlackboardTUI, watch`
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
    LLMResponse,
    LLMUsage,
    run_blackboard,
    run_blackboard_sync,
)

# Configuration
from .config import BlackboardConfig

# Functional worker decorators
from .decorators import (
    worker,
    critic,
)

# =============================================================================
# VERSION
# =============================================================================

__version__ = "1.4.0"

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
    # Decorators
    "worker",
    "critic",
    # Orchestrator (stable)
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

# =============================================================================
# ADVANCED FEATURES - Import from submodules
# =============================================================================
# 
# LiteLLM Integration (100+ models):
#   from blackboard.llm import LiteLLMClient, create_llm
#
# Model Context Protocol (v1.2.0):
#   from blackboard.mcp import MCPServerWorker, MCPToolWorker, MCPRegistry
#
# Observability (v1.2.0):
#   from blackboard.telemetry import OpenTelemetryMiddleware, MetricsCollector
#   from blackboard.replay import SessionRecorder, ReplayOrchestrator
#
# TUI Visualization:
#   from blackboard.tui import BlackboardTUI, watch
#
# Middleware:
#   from blackboard.middleware import BudgetMiddleware, HumanApprovalMiddleware
#
# Events:
#   from blackboard.events import EventBus, Event, EventType
#
# Memory:
#   from blackboard.memory import Memory, SimpleVectorMemory, MemoryWorker
#   from blackboard.embeddings import TFIDFEmbedder, LocalEmbedder, OpenAIEmbedder
#
# Persistence:
#   from blackboard.persistence import RedisPersistence, JSONFilePersistence
#
# Blueprints (Structured Workflows) - NEW in Phase 2:
#   from blackboard.flow import Step, Blueprint, simple_blueprint
#
# Graph Memory (GraphRAG) - NEW in Phase 2:
#   from blackboard.graph import GraphMemory, NetworkXStore, Triple
#
# Reference UI - NEW in Phase 2:
#   CLI: blackboard ui --api-url http://localhost:8000

