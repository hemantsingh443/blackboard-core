# Blackboard-Core

A Python SDK for building **LLM-powered multi-agent systems** using the Blackboard Pattern.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/blackboard-core.svg)](https://pypi.org/project/blackboard-core/)

![Blackboard TUI Demo](public/demo.gif)`

## What is Blackboard-Core?

Blackboard-Core provides a **centralized state architecture** for multi-agent AI systems. Instead of agents messaging each other directly, all agents read from and write to a shared **Blackboard** (state), while a **Supervisor LLM** orchestrates which agent runs next.

```
┌─────────────────────────────────────────────────────────────┐
│                       ORCHESTRATOR                          │
│  ┌─────────────┐    ┌──────────────────────────────────┐    │
│  │  Supervisor │──▶│          BLACKBOARD              │    │
│  │    (LLM)    │    │  • Goal      • Artifacts         │    │
│  └─────────────┘    │  • Status    • Feedback          │    │
│         │           │  • History   • Metadata          │    │
│         ▼           └──────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                       WORKERS                       │    │
│  │  [Writer]  [Critic]  [Refiner]  [Researcher]  ...   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Features

**Core**

- **Centralized State** - All agents share a typed Pydantic state model
- **LLM Orchestration** - A supervisor LLM decides which worker runs next
- **Magic Decorators** - Define workers with simple typed functions
- **Async-First** - Built for high-performance async/await patterns

**Orchestration**

- **Chain-of-Thought** - Pluggable reasoning strategies
- **Fractal Agents** - Nest agents as workers with recursion limits
- **Squad Patterns** - Pre-configured agent factories
- **Blueprints** - Constrain execution to specific workflows

**Persistence & Memory**

- **SQLite/Postgres** - Production-grade persistence
- **Time-Travel Debugging** - Fork sessions at any checkpoint
- **Vector Memory** - Semantic search with pluggable embedders

**Swarm Intelligence (v1.8.0)**

- **Delta Protocol** - Incremental artifact patching with search-replace
- **Map-Reduce** - Parallel sub-agent execution with conflict resolution
- **Branch-Merge** - Fork states, work in isolation, merge results

**Developer Experience**

- **Interactive TUI** - Textual-based Mission Control dashboard
- **CLI Tools** - Project scaffolding and optimization
- **Session Replay** - Record and replay for debugging

**Production**

- **Runtime Security** - Explicit acknowledgment for code execution
- **Cost Control** - Budget middleware with LiteLLM pricing
- **OpenTelemetry** - Distributed tracing

**Ecosystem**

- **LiteLLM Integration** - 100+ LLM providers
- **LangChain Adapter** - Wrap LangChain tools as Workers
- **LlamaIndex Adapter** - Wrap QueryEngines as Workers
- **FastAPI Dependencies** - Easy API integration
- **Model Context Protocol** - Connect to MCP servers

## Installation

```bash
pip install blackboard-core

# Optional extras
pip install blackboard-core[mcp]        # Model Context Protocol
pip install blackboard-core[telemetry]  # OpenTelemetry
pip install blackboard-core[chroma]     # ChromaDB for memory
pip install blackboard-core[serve]      # FastAPI server
pip install blackboard-core[all]        # Everything
```

## Quick Start

```python
from blackboard import Orchestrator, worker
from blackboard.llm import LiteLLMClient

# Define workers with simple type hints - schemas are auto-generated!
@worker
def write(topic: str) -> str:
    """Writes content about a topic."""
    return f"Article about {topic}..."

@worker
def critique(content: str) -> str:
    """Reviews content for quality."""
    return "Approved!" if len(content) > 50 else "Needs more detail"

# Create orchestrator
llm = LiteLLMClient(model="gpt-4o")
orchestrator = Orchestrator(llm=llm, workers=[write, critique])

# Run
result = orchestrator.run_sync(goal="Write about AI safety")
print(result.artifacts[-1].content)
```

## Core Concepts

| Concept          | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| **Blackboard**   | Shared state containing goal, artifacts, feedback, and metadata |
| **Worker**       | An agent that reads state and produces artifacts or feedback    |
| **Orchestrator** | Manages the control loop and calls the supervisor LLM           |
| **Supervisor**   | The LLM that decides which worker to call next                  |
| **Artifact**     | Versioned output produced by a worker                           |
| **Feedback**     | Review/critique of an artifact                                  |

## The Magic Decorator

Define workers with just type hints - no boilerplate:

```python
from blackboard import worker
from blackboard.state import Blackboard

# Simple function - schema auto-generated
@worker
def calculate(a: int, b: int, operation: str = "add") -> str:
    """Performs math operations."""
    if operation == "add":
        return str(a + b)
    return str(a - b)

# With state access
@worker
def summarize(state: Blackboard) -> str:
    """Summarizes current progress."""
    return f"Goal: {state.goal}, Artifacts: {len(state.artifacts)}"

# Async support
@worker
async def research(topic: str) -> str:
    """Researches a topic online."""
    # ... async HTTP calls
    return f"Research on {topic}"
```

## Chain-of-Thought Reasoning

Enable smarter decision-making with CoT:

```python
from blackboard import Orchestrator, BlackboardConfig
from blackboard.reasoning import ChainOfThoughtStrategy

# Enable Chain-of-Thought via config
config = BlackboardConfig(reasoning_strategy="cot")
orchestrator = Orchestrator(llm=llm, workers=workers, config=config)

# Or use the strategy directly
from blackboard.reasoning import ChainOfThoughtStrategy

strategy = ChainOfThoughtStrategy()
# The LLM will now output <thinking>...</thinking> before deciding
```

## State Persistence

Save and resume sessions reliably:

```python
from blackboard.persistence import SQLitePersistence

# Use SQLite for production (supports concurrent access)
persistence = SQLitePersistence("./blackboard.db")
await persistence.initialize()
orchestrator.set_persistence(persistence)

# Save with ID
await persistence.save(state, "session-123")

# Resume later
state = await persistence.load("session-123")
result = await orchestrator.run(state=state)
```

## Advanced Features

### Middleware

```python
from blackboard.middleware import BudgetMiddleware, HumanApprovalMiddleware

orchestrator = Orchestrator(
    llm=my_llm,
    workers=[...],
    middleware=[
        BudgetMiddleware(max_tokens=100000),
        HumanApprovalMiddleware(require_approval_for=["Deployer"])
    ]
)
```

### Memory System

```python
from blackboard.memory import SimpleVectorMemory, MemoryWorker
from blackboard.embeddings import OpenAIEmbedder

memory = SimpleVectorMemory(embedder=OpenAIEmbedder())
worker = MemoryWorker(memory=memory)
```

### Model Context Protocol

```python
from blackboard.mcp import MCPServerWorker

# Local via stdio
fs_server = await MCPServerWorker.create(
    name="Filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-fs", "/tmp"]
)

# Remote via SSE
remote = await MCPServerWorker.create(
    name="RemoteAPI",
    url="http://mcp-server:8080/sse"
)

# Each MCP tool becomes a worker
workers = fs_server.expand_to_workers()
```

### Blueprints (Workflow Patterns)

```python
from blackboard.flow import SequentialPipeline, Router

# Force A → B → C execution
pipeline = SequentialPipeline([Searcher(), Writer(), Critic()])

# Let supervisor choose best worker
router = Router([MathAgent(), CodeAgent(), ResearchAgent()])

result = await orchestrator.run(goal="...", blueprint=pipeline)
```

## Configuration

Use environment variables or direct config:

```bash
export BLACKBOARD_MAX_STEPS=50
export BLACKBOARD_REASONING_STRATEGY=cot
export BLACKBOARD_VERBOSE=true
```

```python
from blackboard import BlackboardConfig

config = BlackboardConfig.from_env()
# Or direct:
config = BlackboardConfig(
    max_steps=50,
    reasoning_strategy="cot",
    enable_parallel=True
)
```

## Documentation

See [DOCS.md](DOCS.md) for the complete API reference and advanced usage guide.

## License

MIT License - see [LICENSE](LICENSE) for details.
